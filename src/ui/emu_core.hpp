#pragma once
// Threaded wrapper around the emulator core.
//
// The emulator runs on its own thread so the UI stays responsive (60+ fps
// menus, smooth animations) even when a game runs below full speed. The UI
// thread communicates through atomics / a mutex-protected frame buffer:
//   UI  -> core : run state, input snapshot, speed, settings
//   core -> UI  : latest video frame, live statistics

#include <atomic>
#include "../audio_stream.hpp"
#include "../rdp.hpp"
#include "../video_frame.hpp"
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class Emulator;

namespace ui {

// CPU state captured at a frame boundary (for the Registers panel).
struct CpuSnapshot {
    std::uint64_t gpr[32] = {};
    std::uint64_t pc = 0, hi = 0, lo = 0;
    std::uint64_t cp0[32] = {};
    std::uint64_t fpr[32] = {};
    std::uint32_t fcsr = 0;
    std::uint32_t instr = 0; // word at PC
    bool delay_slot = false;
};

// Consistent view of the machine between two frames, published for the
// debugger/memory tools. RDRAM is stored big-endian, exactly as on hardware.
struct DebugSnapshot {
    std::vector<std::uint8_t> rdram;
    CpuSnapshot cpu;
    std::uint32_t rsp_segments[16] = {};
    std::uint64_t frame = 0;
    // Geometry of the most recently rendered frame (Object Viewer / 3D Object
    // Inspector). Games rendering at 30 fps only draw every other VI frame, so
    // frames without a display list keep the previous geometry.
    std::shared_ptr<const std::vector<CapturedMesh>> meshes;
    std::shared_ptr<const std::vector<CapturedTexture>> textures; // referenced by CapturedMesh::tex
    Matrix4x4 projection{};

    bool valid(std::uint32_t addr, std::uint32_t size) const {
        std::uint32_t p = addr & 0x1FFFFFFFu;
        return p + size <= rdram.size();
    }
    std::uint8_t u8(std::uint32_t a) const { std::uint32_t p = a & 0x1FFFFFFFu; return p < rdram.size() ? rdram[p] : 0; }
    std::uint16_t u16(std::uint32_t a) const { return static_cast<std::uint16_t>((u8(a) << 8) | u8(a + 1)); }
    std::uint32_t u32(std::uint32_t a) const { return (static_cast<std::uint32_t>(u16(a)) << 16) | u16(a + 2); }
    std::int16_t s16(std::uint32_t a) const { return static_cast<std::int16_t>(u16(a)); }
    float f32(std::uint32_t a) const { std::uint32_t v = u32(a); float f; std::memcpy(&f, &v, 4); return f; }
};

// A value continuously forced into memory (Freeze List).
struct MemoryFreeze {
    std::uint32_t addr = 0;
    std::vector<std::uint8_t> bytes; // big-endian
};

enum class RunState { Stopped, Running, Paused };

// Outcome of a save state request, for the UI to report (EmuCore::poll_state_event).
struct StateEvent {
    enum class Op { Save, Load, UndoLoad } op = Op::Save;
    bool ok = false;
    std::filesystem::path path; // Save / Load
    std::string error;          // why it failed
};

// Averages each scale x scale block of a frame rendered at an internal
// resolution above 1, giving an image at the game's own resolution.
void downscale_frame(std::vector<std::uint32_t>& px, int& w, int& h, int scale);

struct ControllerSnapshot {
    std::uint16_t buttons = 0;
    std::int8_t stick_x = 0, stick_y = 0;
    bool plugged = false;
    int pak = 0; // PortConfig::pak
    std::string gb_rom; // Transfer Pak: the Game Boy ROM in it ("" = none)
};

struct CoreStats {
    float fps = 0.0f;           // emulated frames per second
    float speed_pct = 0.0f;     // relative to the VI rate
    float frame_ms = 0.0f;      // host time spent emulating one frame
    std::uint64_t frame = 0;
    int ucode = -1;             // MicrocodeType index, -1 = none seen yet
    int audio_abi = -1;         // AudioABI index, -1 = no audio task yet
    bool rsp_active = false;
    bool rdp_active = false;
    int vi_hz = 60;
    std::uint32_t ai_rate = 0;  // native AI DAC rate, Hz (0 = unknown)
    float audio_latency_ms = 0; // queued sound + the device buffer
    float audio_target_ms = 0;  // what the pacing steers the queue to
    std::uint64_t audio_underruns = 0;
    float pacing_pct = 0;       // emulation speed trim that keeps the queue on target
    double uptime_s = 0;        // emulated wall time while running
    std::string cart_title;     // internal header name reported by the core
    std::string cic;
    std::string save_type;
};

class EmuCore {
public:
    EmuCore();
    ~EmuCore();

    // Loads a ROM and starts emulation (paused if start_paused).
    bool start(const std::filesystem::path& rom, std::string& error);
    void stop();
    void pause(bool p);
    void reset();

    RunState state() const { return state_.load(); }
    bool loaded() const { return state_.load() != RunState::Stopped; }
    const std::filesystem::path& rom_path() const { return rom_path_; }

    // Speed control
    void set_fast_forward(bool on) { fast_forward_ = on; }
    bool fast_forward() const { return fast_forward_.load(); }
    void set_ff_multiplier(int m) { ff_multiplier_ = m; }
    void set_limit_speed(bool on) { limit_speed_ = on; }
    void set_ucode_override(int idx) { ucode_override_ = idx; ucode_dirty_ = true; }
    // CPU core: 0 = Interpreter, 1 = Dynamic Recompiler (JIT). Applied at the
    // start of the next emulated frame.
    void set_cpu_core(int core) { cpu_core_ = core; }
    // Frame rate limit: 0 = console default (VI rate), >0 = that many frames per second.
    void set_fps_limit(int fps) { fps_limit_ = fps; }
    // Internal resolution of the software RDP: 1 = the game's frame buffer
    // size, 2..8 = that many times larger. Applied at the next frame.
    void set_internal_scale(int scale) { internal_scale_ = scale; }
    // Who renders internal resolutions above 1 (the GPU renderer), or
    // nullptr for the CPU. Applied at the next frame; the high-resolution
    // buffers start over.
    void set_hires_factory(HiResFactory f);
    // Turbo: run continuously at the fast-forward multiplier (like holding Tab).
    void set_turbo(bool on) { turbo_ = on; }
    bool turbo() const { return turbo_.load(); }
    // Executes exactly one frame while paused.
    void frame_advance();

    // ---- Debugger / memory tools
    // While enabled, a DebugSnapshot is published after every emulated frame.
    // `ram` copies RDRAM (8 MB) and `geometry` records drawn meshes; each is
    // only enabled when a tool that needs it is open, to keep emulation fast.
    void set_debug_capture(bool on, bool ram, bool geometry);
    std::shared_ptr<const DebugSnapshot> debug_snapshot();
    // Queues a big-endian write to RDRAM; applied between frames (or at once while paused).
    void poke(std::uint32_t addr, const std::vector<std::uint8_t>& bytes);
    void set_freezes(std::vector<MemoryFreeze> freezes);
    // Decode textures for meshes using this vertex buffer (physical address, 0 = none).
    void set_texture_target(std::uint32_t vtx_phys) { texture_target_ = vtx_phys; }

    void set_input(int port, const ControllerSnapshot& s);
    // Expansion Pak: 8 MB of RDRAM instead of 4. Applied when the game
    // starts or is reset.
    void set_expansion_pak(bool on) { expansion_pak_ = on; }
    // Problems the UI should show (e.g. a Game Boy ROM the Transfer Pak
    // can't use), oldest first.
    bool poll_message(std::string& msg);
    // Whether the game has the port's Rumble Pak motor running.
    bool rumble(int port) const { return rumble_[port & 3].load(std::memory_order_relaxed); }

    // ---- Save states
    // Saves the machine after the current frame / loads one before the next
    // frame; both work while paused too. The emulation thread only copies the
    // state: compressing and writing the file happens on a worker thread, so
    // saving doesn't stall the game. Results arrive through poll_state_event().
    void save_state(const std::filesystem::path& path) { request_state(StateEvent::Op::Save, path); }
    void load_state(const std::filesystem::path& path) { request_state(StateEvent::Op::Load, path); }
    // Goes back to the machine the last load_state() replaced (kept in memory).
    void undo_load_state() { request_state(StateEvent::Op::UndoLoad, {}); }
    bool can_undo_load() const { return has_undo_.load(); }
    bool poll_state_event(StateEvent& ev);

    // Video: the newest frame if it changed since `seen_serial`. Pixels are
    // ARGB8888 (0xAARRGGBB) with alpha forced opaque, unless the frame is
    // in video memory (VideoFrame::gpu). `scale` is the internal resolution
    // it was rendered at: the game's frame buffer is (w / scale) x (h / scale).
    bool fetch_frame(VideoFrame& out, std::uint64_t& seen_serial);
    // The last completed frame as pixels regardless of serial (screenshots/thumbnails).
    bool snapshot(std::vector<std::uint32_t>& out, int& w, int& h, int* scale = nullptr);

    CoreStats stats();

    // Audio: called from the SDL audio thread. Always fills `count` floats.
    void pull_audio(float* out, std::size_t count, float volume);
    // The host device: its rate and callback size (frames), rate 0 = none
    // open. With a device the emulator paces itself to its clock.
    void set_audio_output(std::uint32_t rate, std::uint32_t device_frames);
    float audio_peak() const { return audio_peak_.load(); }

private:
    void thread_main();
    struct FrameClock;
    void run_one_frame(FrameClock& fc);
    void log_pacing(FrameClock& fc);
    void apply_memory_writes();
    void publish_debug(std::uint64_t frame, bool frame_ran);
    void publish_frame(FrameClock& fc);
    void request_state(StateEvent::Op op, const std::filesystem::path& path);
    // Save states, on the emulation thread.
    void process_state_requests(FrameClock& fc);
    void save_state_now(const std::filesystem::path& path);
    bool load_state_now(const std::vector<std::uint8_t>& state, FrameClock& fc, std::string& error);
    void push_state_event(StateEvent ev);
    void wait_save_jobs();

    std::unique_ptr<Emulator> emu_;
    std::filesystem::path rom_path_;
    std::thread thread_;
    std::atomic<RunState> state_{RunState::Stopped};
    std::atomic<bool> quit_{false};
    std::atomic<bool> reset_req_{false};
    std::atomic<bool> fast_forward_{false};
    std::atomic<int> ff_multiplier_{3};
    std::atomic<bool> limit_speed_{true};
    std::atomic<int> ucode_override_{0};
    std::atomic<bool> ucode_dirty_{true};
    std::atomic<int> cpu_core_{1};
    std::atomic<int> fps_limit_{0};
    std::atomic<int> internal_scale_{1};
    std::mutex factory_mutex_;
    HiResFactory hires_factory_;
    bool factory_dirty_ = false; // guarded by factory_mutex_
    std::atomic<bool> turbo_{false};
    std::atomic<int> advance_req_{0};
    std::mutex wake_mutex_;

    std::atomic<bool> debug_capture_{false};
    std::atomic<bool> debug_dirty_{false};
    std::atomic<bool> geometry_capture_changed_{true};
    std::atomic<bool> debug_ram_{false};
    std::atomic<bool> debug_geometry_{false};
    std::mutex debug_mutex_;
    std::shared_ptr<DebugSnapshot> debug_snap_;
    std::mutex poke_mutex_;
    std::vector<std::pair<std::uint32_t, std::vector<std::uint8_t>>> pokes_;
    std::vector<MemoryFreeze> freezes_;
    std::shared_ptr<const std::vector<CapturedMesh>> last_meshes_;
    std::shared_ptr<const std::vector<CapturedTexture>> last_textures_;
    std::atomic<std::uint32_t> texture_target_{0};
    std::condition_variable wake_;

    struct StateRequest {
        StateEvent::Op op;
        std::filesystem::path path;
    };
    std::mutex state_mutex_; // guards the three vectors below
    std::vector<StateRequest> state_requests_;
    std::vector<StateEvent> state_events_;
    // Compress + write, one per save. Each waits for the one before it, so
    // saves reach the disk in the order they were made.
    std::vector<std::shared_future<void>> save_jobs_;
    std::atomic<bool> state_request_pending_{false};
    std::vector<std::uint8_t> undo_state_; // emulation thread only
    std::atomic<bool> has_undo_{false};

    std::mutex input_mutex_;
    ControllerSnapshot input_[4];
    std::atomic<bool> rumble_[4] = {};
    std::atomic<bool> expansion_pak_{true};
    std::mutex message_mutex_;
    std::vector<std::string> messages_;

    std::mutex frame_mutex_;
    VideoFrame frame_;
    std::uint64_t frame_serial_ = 0;

    std::mutex stats_mutex_;
    CoreStats stats_;

    // Held while the machine is replaced wholesale (reset, state load).
    std::mutex audio_mutex_;
    // Sound on its way to the device: filled by the emulation thread (the
    // AI), drained by the audio callback. Outlives every Emulator.
    AudioStream audio_;
    std::atomic<bool> audio_open_{false};
    std::atomic<float> audio_peak_{0.0f};
    float audio_volume_ = 0.0f; // audio thread: last applied volume, ramped from
};

} // namespace ui
