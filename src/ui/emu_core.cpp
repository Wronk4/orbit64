#include "emu_core.hpp"
#include "../emulator.hpp"
#include "../jit/jit_invalidate.hpp"
#include "../savestate.hpp"
#include "platform.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>

namespace ui {

using Clock = std::chrono::steady_clock;

EmuCore::EmuCore() = default;

EmuCore::~EmuCore() { stop(); }

static const char* cic_name(CICType t) {
    switch (t) {
        case CICType::CIC_6101: return "CIC-NUS-6101";
        case CICType::CIC_6102: return "CIC-NUS-6102";
        case CICType::CIC_6103: return "CIC-NUS-6103";
        case CICType::CIC_6105: return "CIC-NUS-6105";
        case CICType::CIC_6106: return "CIC-NUS-6106";
        default: return "Unknown";
    }
}

static const char* save_name(SaveType t) {
    switch (t) {
        case SaveType::NONE: return "None";
        case SaveType::EEPROM_4K: return "EEPROM 4 Kbit";
        case SaveType::EEPROM_16K: return "EEPROM 16 Kbit";
        case SaveType::SRAM_32K: return "SRAM 256 Kbit";
        case SaveType::FLASHRAM_128K: return "FlashRAM 1 Mbit";
    }
    return "Unknown";
}

struct EmuCore::FrameClock {
    std::vector<std::uint32_t> pixels;
    Clock::time_point next = Clock::now();
    Clock::time_point window_start = Clock::now();
    int window_frames = 0;
    double window_work_ms = 0.0;
    std::uint64_t frame_counter = 0;
    std::uint64_t last_gfx = 0, last_audio = 0, last_dl = 0;
    double uptime = 0.0;
};

bool EmuCore::start(const std::filesystem::path& rom, std::string& error) {
    stop();
    auto emu = std::make_unique<Emulator>();
    if (!emu->load_rom(platform::path_to_utf8(rom))) {
        error = "The file could not be read or is not a valid Nintendo 64 ROM.";
        return false;
    }
    emu->set_audio_output_rate(audio_rate_);

    {
        std::lock_guard<std::mutex> lk(stats_mutex_);
        stats_ = CoreStats{};
        stats_.cart_title = emu->get_cartridge().get_title();
        stats_.cic = cic_name(emu->get_cartridge().get_cic_type());
        stats_.save_type = save_name(emu->get_cartridge().get_save_type());
    }
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        frame_.clear();
        frame_w_ = frame_h_ = 0;
        frame_scale_ = 1;
        frame_serial_++;
    }
    {
        std::lock_guard<std::mutex> lk(audio_mutex_);
        emu_ = std::move(emu);
    }
    rom_path_ = rom;
    undo_state_.clear();
    has_undo_ = false;
    ucode_dirty_ = true;
    quit_ = false;
    debug_dirty_ = true;
    state_ = RunState::Running;
    thread_ = std::thread(&EmuCore::thread_main, this);
    return true;
}

void EmuCore::stop() {
    quit_ = true;
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
    // States being written still go to disk; unhandled requests are dropped.
    wait_save_jobs();
    {
        std::lock_guard<std::mutex> lk(state_mutex_);
        state_requests_.clear();
    }
    state_request_pending_ = false;
    undo_state_.clear();
    undo_state_.shrink_to_fit();
    has_undo_ = false;
    {
        // Destroying the emulator flushes SRAM/EEPROM saves to disk.
        std::lock_guard<std::mutex> lk(audio_mutex_);
        emu_.reset();
    }
    state_ = RunState::Stopped;
    fast_forward_ = false;
    advance_req_ = 0;
    {
        std::lock_guard<std::mutex> lk(debug_mutex_);
        debug_snap_.reset();
    }
    last_meshes_.reset();
    last_textures_.reset();
    {
        std::lock_guard<std::mutex> lk(poke_mutex_);
        pokes_.clear();
    }
    audio_peak_ = 0.0f;
    std::lock_guard<std::mutex> lk(stats_mutex_);
    stats_.fps = 0;
    stats_.speed_pct = 0;
    stats_.rsp_active = stats_.rdp_active = false;
}

void EmuCore::pause(bool p) {
    if (state_ == RunState::Stopped) return;
    state_ = p ? RunState::Paused : RunState::Running;
    wake_.notify_all();
}

void EmuCore::reset() {
    if (state_ == RunState::Stopped) return;
    reset_req_ = true;
    wake_.notify_all();
}

void downscale_frame(std::vector<std::uint32_t>& px, int& w, int& h, int scale) {
    if (scale <= 1 || w < scale || h < scale) return;
    const int nw = w / scale, nh = h / scale;
    std::vector<std::uint32_t> small(static_cast<size_t>(nw) * nh);
    for (int y = 0; y < nh; ++y) {
        for (int x = 0; x < nw; ++x) {
            std::uint32_t r = 0, g = 0, b = 0;
            for (int j = 0; j < scale; ++j) {
                const std::uint32_t* row = px.data() + static_cast<size_t>(y * scale + j) * w + x * scale;
                for (int i = 0; i < scale; ++i) {
                    r += (row[i] >> 16) & 0xFF;
                    g += (row[i] >> 8) & 0xFF;
                    b += row[i] & 0xFF;
                }
            }
            const std::uint32_t n = static_cast<std::uint32_t>(scale * scale);
            small[static_cast<size_t>(y) * nw + x] = 0xFF000000u | (r / n) << 16 | (g / n) << 8 | (b / n);
        }
    }
    px.swap(small);
    w = nw;
    h = nh;
}

void EmuCore::request_state(StateEvent::Op op, const std::filesystem::path& path) {
    if (state_ == RunState::Stopped) return;
    {
        std::lock_guard<std::mutex> lk(state_mutex_);
        state_requests_.push_back({op, path});
    }
    state_request_pending_ = true;
    wake_.notify_all();
}

bool EmuCore::poll_state_event(StateEvent& ev) {
    std::lock_guard<std::mutex> lk(state_mutex_);
    if (state_events_.empty()) return false;
    ev = std::move(state_events_.front());
    state_events_.erase(state_events_.begin());
    return true;
}

void EmuCore::push_state_event(StateEvent ev) {
    std::lock_guard<std::mutex> lk(state_mutex_);
    state_events_.push_back(std::move(ev));
}

void EmuCore::wait_save_jobs() {
    std::vector<std::shared_future<void>> jobs;
    {
        std::lock_guard<std::mutex> lk(state_mutex_);
        jobs.swap(save_jobs_);
    }
    // Not under state_mutex_: the jobs report through it.
    for (auto& j : jobs) j.wait();
}

void EmuCore::save_state_now(const std::filesystem::path& path) {
    const Cartridge& cart = emu_->get_cartridge();
    savestate::FileInfo info;
    info.rom_crc1 = cart.get_crc1();
    info.rom_crc2 = cart.get_crc2();
    info.game_code = cart.get_game_code();
    info.rom_title = cart.get_title();
    info.created = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch()).count();
    // The last published frame is what the saved machine shows.
    int w = 0, h = 0, scale = 1;
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        info.thumb = frame_;
        w = frame_w_;
        h = frame_h_;
        scale = frame_scale_;
    }
    downscale_frame(info.thumb, w, h, scale);
    info.thumb_w = static_cast<std::uint32_t>(w);
    info.thumb_h = static_cast<std::uint32_t>(h);
    std::vector<std::uint8_t> state = emu_->save_state();

    std::lock_guard<std::mutex> lk(state_mutex_);
    std::shared_future<void> before = save_jobs_.empty() ? std::shared_future<void>() : save_jobs_.back();
    save_jobs_.erase(std::remove_if(save_jobs_.begin(), save_jobs_.end(),
                                    [](const std::shared_future<void>& j) {
                                        return j.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                                    }),
                     save_jobs_.end());
    auto job = [this, before, path, state = std::move(state), info = std::move(info)] {
        if (before.valid()) before.wait(); // an earlier save of the same slot must not land last
        StateEvent ev;
        ev.op = StateEvent::Op::Save;
        ev.path = path;
        ev.ok = savestate::write_file(path, state, info, ev.error);
        push_state_event(std::move(ev));
    };
    save_jobs_.push_back(std::async(std::launch::async, std::move(job)).share());
}

bool EmuCore::load_state_now(const std::vector<std::uint8_t>& state, FrameClock& fc, std::string& error) {
    {
        std::lock_guard<std::mutex> al(audio_mutex_);
        if (!emu_->load_state(state, error)) return false;
    }
    // Show the loaded machine right away, also while paused.
    int w = 0, h = 0, scale = 1;
    emu_->render_frame(fc.pixels, w, h, &scale);
    publish_frame(fc, w, h, scale);
    debug_dirty_ = true;
    fc.next = Clock::now();
    return true;
}

void EmuCore::process_state_requests(FrameClock& fc) {
    if (!state_request_pending_.exchange(false)) return;
    std::vector<StateRequest> requests;
    {
        std::lock_guard<std::mutex> lk(state_mutex_);
        requests.swap(state_requests_);
    }
    for (const StateRequest& r : requests) {
        if (r.op == StateEvent::Op::Save) {
            save_state_now(r.path); // reported by its worker
            continue;
        }
        StateEvent ev;
        ev.op = r.op;
        ev.path = r.path;
        if (r.op == StateEvent::Op::Load) {
            wait_save_jobs(); // this very file may still be being written
            std::vector<std::uint8_t> state;
            savestate::FileInfo info;
            const Cartridge& cart = emu_->get_cartridge();
            if (!savestate::read_file(r.path, &state, info, ev.error)) {
                // reported below
            } else if (info.rom_crc1 != cart.get_crc1() || info.rom_crc2 != cart.get_crc2()) {
                ev.error = "it was saved from " + (info.rom_title.empty() ? std::string("another game") : info.rom_title);
            } else {
                std::vector<std::uint8_t> before = emu_->save_state();
                ev.ok = load_state_now(state, fc, ev.error);
                if (ev.ok) {
                    undo_state_ = std::move(before);
                    has_undo_ = true;
                }
            }
        } else if (undo_state_.empty()) {
            ev.error = "there is no load to undo";
        } else {
            ev.ok = load_state_now(undo_state_, fc, ev.error);
            if (ev.ok) {
                undo_state_.clear();
                undo_state_.shrink_to_fit();
                has_undo_ = false;
            }
        }
        push_state_event(std::move(ev));
    }
}

void EmuCore::set_input(int port, const ControllerSnapshot& s) {
    std::lock_guard<std::mutex> lk(input_mutex_);
    input_[port & 3] = s;
}

bool EmuCore::fetch_frame(std::vector<std::uint32_t>& out, int& w, int& h, int& scale, std::uint64_t& seen) {
    std::lock_guard<std::mutex> lk(frame_mutex_);
    if (seen == frame_serial_) return false;
    seen = frame_serial_;
    out = frame_;
    w = frame_w_;
    h = frame_h_;
    scale = frame_scale_;
    return true;
}

bool EmuCore::snapshot(std::vector<std::uint32_t>& out, int& w, int& h, int* scale) {
    std::lock_guard<std::mutex> lk(frame_mutex_);
    if (frame_.empty()) return false;
    out = frame_;
    w = frame_w_;
    h = frame_h_;
    if (scale) *scale = frame_scale_;
    return true;
}

CoreStats EmuCore::stats() {
    std::lock_guard<std::mutex> lk(stats_mutex_);
    return stats_;
}

void EmuCore::set_audio_rate(std::uint32_t rate) {
    std::lock_guard<std::mutex> lk(audio_mutex_);
    audio_rate_ = rate;
    if (emu_) emu_->set_audio_output_rate(rate);
}

void EmuCore::pull_audio(float* out, std::size_t count, float volume) {
    std::unique_lock<std::mutex> lk(audio_mutex_, std::try_to_lock);
    if (!lk.owns_lock() || !emu_ || state_ != RunState::Running) {
        std::fill(out, out + count, 0.0f);
        audio_peak_ = audio_peak_.load() * 0.85f;
        return;
    }
    emu_->get_audio_samples(out, count);
    float peak = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        out[i] *= volume;
        peak = std::fmax(peak, std::fabs(out[i]));
    }
    // Fast attack, slow release for a pleasant meter.
    float prev = audio_peak_.load();
    audio_peak_ = peak > prev ? peak : prev * 0.9f + peak * 0.1f;
}

void EmuCore::frame_advance() {
    if (state_ != RunState::Paused) return;
    advance_req_++;
    wake_.notify_all();
}

void EmuCore::set_debug_capture(bool on, bool ram, bool geometry) {
    if (on && (!debug_capture_ || (ram && !debug_ram_))) debug_dirty_ = true; // publish immediately, even while paused
    debug_capture_ = on;
    debug_ram_ = ram;
    if (debug_geometry_ != geometry) geometry_capture_changed_ = true;
    debug_geometry_ = geometry;
    wake_.notify_all();
}

std::shared_ptr<const DebugSnapshot> EmuCore::debug_snapshot() {
    std::lock_guard<std::mutex> lk(debug_mutex_);
    return debug_snap_;
}

void EmuCore::poke(std::uint32_t addr, const std::vector<std::uint8_t>& bytes) {
    {
        std::lock_guard<std::mutex> lk(poke_mutex_);
        pokes_.emplace_back(addr, bytes);
    }
    debug_dirty_ = true;
    wake_.notify_all();
}

void EmuCore::set_freezes(std::vector<MemoryFreeze> freezes) {
    std::lock_guard<std::mutex> lk(poke_mutex_);
    freezes_ = std::move(freezes);
}

void EmuCore::apply_memory_writes() {
    std::lock_guard<std::mutex> lk(poke_mutex_);
    Bus& bus = emu_->get_bus();
    std::uint8_t* ram = bus.get_rdram();
    const size_t size = bus.get_rdram_size();
    // A poke into code must drop the JIT blocks compiled from it, like any
    // other RDRAM write. Freezes rewrite the same bytes every frame, so only
    // an actual change is written and reported.
    auto write = [&](std::uint32_t addr, const std::vector<std::uint8_t>& b) {
        std::uint32_t p = addr & 0x1FFFFFFFu;
        if (p + b.size() <= size && std::memcmp(ram + p, b.data(), b.size()) != 0) {
            std::memcpy(ram + p, b.data(), b.size());
            jit::notify_code_write(p, static_cast<std::uint32_t>(b.size()));
        }
    };
    for (const auto& [a, b] : pokes_) write(a, b);
    pokes_.clear();
    for (const auto& f : freezes_) write(f.addr, f.bytes);
}

void EmuCore::publish_debug(std::uint64_t frame, bool frame_ran) {
    // Reuse the previous buffer when the UI no longer holds it (avoids an 8 MB allocation per frame).
    std::shared_ptr<DebugSnapshot> snap;
    {
        std::lock_guard<std::mutex> lk(debug_mutex_);
        if (debug_snap_ && debug_snap_.use_count() == 1) snap = std::move(debug_snap_);
    }
    if (!snap) snap = std::make_shared<DebugSnapshot>();
    Bus& bus = emu_->get_bus();
    if (debug_ram_) snap->rdram.assign(bus.get_rdram(), bus.get_rdram() + bus.get_rdram_size());
    else snap->rdram.clear();
    const CPU& cpu = emu_->get_cpu();
    CpuSnapshot& c = snap->cpu;
    for (int i = 0; i < 32; ++i) {
        c.gpr[i] = cpu.get_gpr(i);
        c.cp0[i] = cpu.get_cp0(i);
        c.fpr[i] = cpu.get_fpr_raw(i);
    }
    c.pc = cpu.get_pc();
    c.hi = cpu.get_hi();
    c.lo = cpu.get_lo();
    c.fcsr = cpu.get_fcsr();
    c.delay_slot = cpu.is_in_delay_slot();
    std::uint32_t pp = static_cast<std::uint32_t>(c.pc) & 0x1FFFFFFFu;
    const std::uint8_t* ram = bus.get_rdram();
    c.instr = pp + 4 <= bus.get_rdram_size() ? (ram[pp] << 24) | (ram[pp + 1] << 16) | (ram[pp + 2] << 8) | ram[pp + 3] : 0;
    for (int i = 0; i < 16; ++i) snap->rsp_segments[i] = emu_->get_rdp().get_segment(i);
    if (frame_ran) {
        auto captured = emu_->get_rdp().take_capture();
        auto textures = emu_->get_rdp().take_textures();
        if (!captured.empty()) {
            last_meshes_ = std::make_shared<const std::vector<CapturedMesh>>(std::move(captured));
            last_textures_ = std::make_shared<const std::vector<CapturedTexture>>(std::move(textures));
        }
        snap->projection = emu_->get_rdp().get_projection();
    }
    snap->meshes = last_meshes_;
    snap->textures = last_textures_;
    snap->frame = frame;
    std::lock_guard<std::mutex> lk(debug_mutex_);
    debug_snap_ = std::move(snap);
}

void EmuCore::run_one_frame(FrameClock& fc) {
    if (ucode_dirty_) {
        emu_->get_rdp().set_ucode_type(static_cast<MicrocodeType>(ucode_override_.load()));
        ucode_dirty_ = false;
    }
    emu_->set_cpu_core(cpu_core_.load() == 0 ? CpuCore::Interpreter : CpuCore::Recompiler);
    emu_->get_rdp().set_hires_scale(static_cast<u32>(std::clamp(internal_scale_.load(), 1, 8)));
    {
        std::lock_guard<std::mutex> lk(input_mutex_);
        for (int p = 0; p < 4; ++p) {
            Controller& c = emu_->get_controller(p);
            c.set_plugged_in(input_[p].plugged);
            for (int bit = 0; bit < 16; ++bit) {
                std::uint16_t mask = static_cast<std::uint16_t>(1u << bit);
                c.set_button(mask, (input_[p].buttons & mask) != 0);
            }
            c.set_stick(input_[p].stick_x, input_[p].stick_y);
        }
    }

    if (geometry_capture_changed_.exchange(false)) emu_->get_rdp().set_capture(debug_geometry_.load());
    emu_->get_rdp().set_texture_capture_target(debug_geometry_ ? texture_target_.load() : 0);

    // Pokes and frozen values are applied before and after the frame, so the
    // game always reads the forced value and the tools display it too.
    apply_memory_writes();
    auto t0 = Clock::now();
    int w = 0, h = 0, scale = 1;
    emu_->step_frame();
    emu_->render_frame(fc.pixels, w, h, &scale);
    auto t1 = Clock::now();
    apply_memory_writes();
    fc.frame_counter++;
    fc.window_frames++;
    fc.window_work_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (debug_capture_) publish_debug(fc.frame_counter, true);
    publish_frame(fc, w, h, scale);

    const int vi_hz = static_cast<int>(emu_->get_vi().get_refresh_hz());
    fc.uptime += 1.0 / vi_hz;

    // Publish statistics a few times per second.
    double window_s = std::chrono::duration<double>(t1 - fc.window_start).count();
    if (window_s >= 0.25) {
        RSP& rsp = emu_->get_rsp();
        RDP& rdp = emu_->get_rdp();
        std::uint64_t gfx = rsp.get_gfx_task_count(), aud = rsp.get_audio_task_count();
        std::uint64_t dl = rdp.get_display_list_count();
        std::lock_guard<std::mutex> lk(stats_mutex_);
        stats_.fps = static_cast<float>(fc.window_frames / window_s);
        stats_.speed_pct = stats_.fps / vi_hz * 100.0f;
        stats_.frame_ms = static_cast<float>(fc.window_work_ms / fc.window_frames);
        stats_.rsp_active = gfx != fc.last_gfx || aud != fc.last_audio;
        stats_.rdp_active = dl != fc.last_dl;
        stats_.ucode = rdp.has_processed_display_list() ? static_cast<int>(rdp.get_active_ucode()) : -1;
        stats_.audio_abi = aud > 0 ? rsp.get_audio_hle().get_abi_index() : -1;
        stats_.vi_hz = vi_hz;
        stats_.ai_rate = emu_->get_ai().get_native_sample_rate();
        fc.last_gfx = gfx;
        fc.last_audio = aud;
        fc.last_dl = dl;
        fc.window_start = t1;
        fc.window_frames = 0;
        fc.window_work_ms = 0.0;
    }
    std::lock_guard<std::mutex> lk(stats_mutex_);
    stats_.frame = fc.frame_counter;
    stats_.uptime_s = fc.uptime;
}

void EmuCore::publish_frame(FrameClock& fc, int w, int h, int scale) {
    for (auto& px : fc.pixels) px |= 0xFF000000u;
    std::lock_guard<std::mutex> lk(frame_mutex_);
    frame_.swap(fc.pixels);
    frame_w_ = w;
    frame_h_ = h;
    frame_scale_ = scale;
    frame_serial_++;
}

void EmuCore::thread_main() {
    FrameClock fc;
    while (!quit_) {
        if (state_ == RunState::Paused) {
            {
                std::unique_lock<std::mutex> lk(wake_mutex_);
                wake_.wait_for(lk, std::chrono::milliseconds(100), [&] {
                    return quit_ || state_ != RunState::Paused || reset_req_ || advance_req_ > 0 || debug_dirty_ ||
                           state_request_pending_;
                });
            }
            if (reset_req_) {
                std::lock_guard<std::mutex> al(audio_mutex_);
                emu_->reset();
                reset_req_ = false;
                fc.frame_counter = 0;
                debug_dirty_ = true;
            }
            process_state_requests(fc);
            if (advance_req_ > 0) {
                // Frame advance: exactly one frame, then stay paused.
                advance_req_--;
                run_one_frame(fc);
            } else if (debug_dirty_) {
                // Memory edits while paused take effect (and are shown) immediately.
                apply_memory_writes();
                if (debug_capture_) publish_debug(fc.frame_counter, false);
            }
            debug_dirty_ = false;
            {
                // Nothing is executing while paused: report it that way.
                std::lock_guard<std::mutex> lk(stats_mutex_);
                stats_.fps = 0.0f;
                stats_.speed_pct = 0.0f;
                stats_.rsp_active = stats_.rdp_active = false;
                stats_.frame = fc.frame_counter;
            }
            fc.next = Clock::now();
            fc.window_start = fc.next;
            fc.window_frames = 0;
            fc.window_work_ms = 0.0;
            continue;
        }

        if (reset_req_) {
            std::lock_guard<std::mutex> al(audio_mutex_);
            emu_->reset();
            reset_req_ = false;
            fc.frame_counter = 0;
            fc.uptime = 0.0;
        }
        process_state_requests(fc);
        debug_dirty_ = false;
        run_one_frame(fc);

        // Frame pacing: console refresh rate, a user FPS limit, or unlimited.
        const int vi_hz = static_cast<int>(emu_->get_vi().get_refresh_hz());
        const int limit = fps_limit_.load();
        const double period = 1.0 / (limit > 0 ? limit : vi_hz);
        bool ff = fast_forward_.load() || turbo_.load();
        int mult = ff_multiplier_.load();
        if (!limit_speed_ || (ff && mult == 0)) {
            fc.next = Clock::now();
            continue;
        }
        double target = ff ? period / mult : period;
        fc.next += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(target));
        auto now = Clock::now();
        if (fc.next < now - std::chrono::milliseconds(100)) fc.next = now; // running slow: don't spiral
        std::unique_lock<std::mutex> lk(wake_mutex_);
        wake_.wait_until(lk, fc.next, [&] { return quit_.load() || state_ == RunState::Paused; });
    }
}

} // namespace ui
