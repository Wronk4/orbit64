#pragma once
// Threaded wrapper around the emulator core.
//
// The emulator runs on its own thread so the UI stays responsive (60+ fps
// menus, smooth animations) even when a game runs below full speed. The UI
// thread communicates through atomics / a mutex-protected frame buffer:
//   UI  -> core : run state, input snapshot, speed, settings
//   core -> UI  : latest video frame, live statistics

#include <atomic>
#include "../rdp.hpp"
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <filesystem>
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

struct ControllerSnapshot {
    std::uint16_t buttons = 0;
    std::int8_t stick_x = 0, stick_y = 0;
    bool plugged = false;
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

    // Video: copies the newest frame if it changed since `seen_serial`.
    // Pixels are ARGB8888 (0xAARRGGBB) with alpha forced opaque. `scale` is
    // the internal resolution it was rendered at: the game's frame buffer is
    // (w / scale) x (h / scale).
    bool fetch_frame(std::vector<std::uint32_t>& out, int& w, int& h, int& scale, std::uint64_t& seen_serial);
    // Copy of the last completed frame regardless of serial (screenshots/thumbnails).
    bool snapshot(std::vector<std::uint32_t>& out, int& w, int& h, int* scale = nullptr);

    CoreStats stats();

    // Audio: called from the SDL audio thread. Always fills `count` floats.
    void pull_audio(float* out, std::size_t count, float volume);
    void set_audio_rate(std::uint32_t rate);
    float audio_peak() const { return audio_peak_.load(); }

private:
    void thread_main();
    struct FrameClock;
    void run_one_frame(FrameClock& fc);
    void apply_memory_writes();
    void publish_debug(std::uint64_t frame, bool frame_ran);

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

    std::mutex input_mutex_;
    ControllerSnapshot input_[4];

    std::mutex frame_mutex_;
    std::vector<std::uint32_t> frame_;
    int frame_w_ = 0, frame_h_ = 0, frame_scale_ = 1;
    std::uint64_t frame_serial_ = 0;

    std::mutex stats_mutex_;
    CoreStats stats_;

    // Guards emu_ lifetime against the audio callback.
    std::mutex audio_mutex_;
    std::uint32_t audio_rate_ = 44100;
    std::atomic<float> audio_peak_{0.0f};
};

} // namespace ui
