// Headless save state checker: a loaded state has to continue exactly like
// the machine it was saved from.
//
// Emulator A runs --at frames and saves a state, which also goes through a
// file (write_file / read_file), then runs --after more frames. B, a fresh
// emulator, and C, one that has already been running for a while on other
// input, load the state and run the same frames on the same input. Every
// frame's RDRAM and displayed image, and the complete state at the end, must
// be identical to A's. A state loaded and saved again must give back the very
// same bytes, and a damaged one must be refused without touching the machine.
//
// The JIT's code-write hook is process-wide (jit/jit_invalidate.hpp), so only
// one emulator may exist at a time: A, B and C run one after another.
//
//   make savestate_check   -> bin/savestate_check
//
//   bin/savestate_check rom.z64 [--at N] [--after M] [--mash btn|none] [--int] [--scale S] [--gpu-exact]
//
// --int runs on the interpreter instead of the JIT. --scale runs B and C at
// that internal resolution: RDRAM must not change (images are compared at 1
// only). --gpu-exact draws low-level graphics with the bit-exact RDP on the
// GPU (with ORBIT64_RSP=lle-gfx). Save files next to the ROM are neither
// read nor written.

#include "emulator.hpp"
#include "gpu/device.hpp"
#include "gpu/rdp_exact_gpu.hpp"
#include "savestate.hpp"
#include <SDL3/SDL.h>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

uint64_t fnv(const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Options {
    std::string rom, mash = "start";
    int at = 600, after = 600, scale = 1;
    bool interp = false;
    ExactAccelFactory exact_gpu; // --gpu-exact
};

struct Run {
    std::unique_ptr<Emulator> emu = std::make_unique<Emulator>();
    std::vector<u32> px;
    int w = 0, h = 0;
};

bool boot(Run& r, const Options& o, int scale) {
    r.emu->get_cartridge().set_use_save_file(false);
    if (!r.emu->load_rom(o.rom)) return false;
    r.emu->set_cpu_core(o.interp ? CpuCore::Interpreter : CpuCore::Recompiler);
    r.emu->get_rdp().set_hires_scale(static_cast<u32>(scale));
    if (o.exact_gpu) {
        r.emu->get_rdp().set_exact_accel_factory(o.exact_gpu);
        r.emu->get_rdp().set_exact_gpu(true);
    }
    return true;
}

// One frame the way the frontend runs it (the state is saved between frames).
// The whole controller is set every frame, like the frontend does: input
// isn't part of a state, so a button left held from before a load would
// otherwise stay held.
void frame(Run& r, int f, const std::string& mash) {
    Controller& pad = r.emu->get_controller(0);
    pad.reset();
    if (!mash.empty()) pad.press_named_button(mash, ((f / 6) % 2) == 0);
    r.emu->step_frame();
    r.emu->render_frame(r.px, r.w, r.h);
}

struct Trace {
    std::vector<uint64_t> ram, image;
    std::vector<u8> end_state;
};

// Runs frames [at, at + after) and records what they leave behind.
Trace continue_run(Run& r, const Options& o) {
    Trace t;
    for (int f = o.at; f < o.at + o.after; ++f) {
        frame(r, f, o.mash);
        t.ram.push_back(fnv(r.emu->get_bus().get_rdram(), r.emu->get_bus().get_rdram_size()));
        t.image.push_back(fnv(r.px.data(), r.px.size() * 4) ^ (static_cast<uint64_t>(r.w) << 32 | r.h));
    }
    t.end_state = r.emu->save_state();
    return t;
}

// Empty when `t` continues exactly like `ref`.
std::string compare(const Trace& ref, const Trace& t, bool images, int at) {
    for (size_t i = 0; i < ref.ram.size(); ++i) {
        if (t.ram[i] != ref.ram[i] || (images && t.image[i] != ref.image[i])) {
            char buf[160];
            std::snprintf(buf, sizeof buf, "diverged %zu frames after the load (frame %zu, %s differs)", i + 1, at + i,
                          t.ram[i] != ref.ram[i] ? "RDRAM" : "image");
            return buf;
        }
    }
    if (t.end_state != ref.end_state) return "RDRAM and images match, but the full state differs at the end";
    return {};
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--at") o.at = std::stoi(next());
        else if (a == "--after") o.after = std::stoi(next());
        else if (a == "--mash") o.mash = next();
        else if (a == "--int") o.interp = true;
        else if (a == "--scale") o.scale = std::stoi(next());
        else if (a == "--gpu-exact") {
            static std::shared_ptr<gpu::Device> dev;
            if (!SDL_Init(SDL_INIT_VIDEO) || !(dev = std::make_shared<gpu::Device>(gpu::create_device()))->exact_ok()) {
                std::fprintf(stderr, "no GPU for the bit-exact RDP\n");
                return 1;
            }
            o.exact_gpu = gpu::make_exact_accel_factory(dev);
        }
        else o.rom = a;
    }
    if (o.mash == "none") o.mash.clear();
    const bool images = o.scale == 1;

    // A: the original run.
    std::vector<u8> state;
    double save_ms = 0, write_ms = 0, read_ms = 0;
    unsigned long long file_size = 0;
    Trace ref;
    {
        Run A;
        if (o.rom.empty() || !boot(A, o, 1)) {
            std::fprintf(stderr, "usage: savestate_check rom.z64 [options] (see tools/savestate_check.cpp)\n");
            return 1;
        }
        for (int f = 0; f < o.at; ++f) frame(A, f, o.mash);
        auto t = Clock::now();
        state = A.emu->save_state();
        save_ms = ms_since(t);

        const Cartridge& cart = A.emu->get_cartridge();
        savestate::FileInfo info;
        info.rom_crc1 = cart.get_crc1();
        info.rom_crc2 = cart.get_crc2();
        info.game_code = cart.get_game_code();
        info.rom_title = cart.get_title();
        info.created = 1234567890;
        info.thumb_w = static_cast<u32>(A.w);
        info.thumb_h = static_cast<u32>(A.h);
        info.thumb = A.px;
        const auto path = std::filesystem::temp_directory_path() / "orbit64_savestate_check.state";
        std::string err;
        t = Clock::now();
        if (!savestate::write_file(path, state, info, err)) {
            std::printf("FAIL write_file: %s\n", err.c_str());
            return 1;
        }
        write_ms = ms_since(t);
        file_size = std::filesystem::file_size(path);
        std::vector<u8> from_file;
        savestate::FileInfo back;
        t = Clock::now();
        const bool read_ok = savestate::read_file(path, &from_file, back, err);
        read_ms = ms_since(t);
        std::filesystem::remove(path);
        if (!read_ok) {
            std::printf("FAIL read_file: %s\n", err.c_str());
            return 1;
        }
        if (from_file != state || back.rom_crc1 != info.rom_crc1 || back.rom_crc2 != info.rom_crc2 ||
            back.game_code != info.game_code || back.rom_title != info.rom_title || back.created != info.created ||
            back.thumb != info.thumb || back.thumb_w != info.thumb_w || back.thumb_h != info.thumb_h) {
            std::printf("FAIL: the file doesn't give back what was written\n");
            return 1;
        }
        ref = continue_run(A, o);
    }

    // B: a fresh emulator.
    double load_ms = 0;
    {
        Run B;
        boot(B, o, o.scale);
        std::string err;
        const auto t = Clock::now();
        if (!B.emu->load_state(state, err)) {
            std::printf("FAIL load into a fresh emulator: %s\n", err.c_str());
            return 1;
        }
        load_ms = ms_since(t);
        if (B.emu->save_state() != state) {
            std::printf("FAIL: saving a loaded state doesn't give the same bytes\n");
            return 1;
        }
        const std::string diff = compare(ref, continue_run(B, o), images, o.at);
        if (!diff.empty()) {
            std::printf("FAIL (fresh emulator): %s\n", diff.c_str());
            return 1;
        }
    }

    // C: an emulator that was somewhere else first (other input, half the time).
    std::string refused;
    {
        Run C;
        boot(C, o, o.scale);
        for (int f = 0; f < o.at / 2; ++f) frame(C, f, "a");
        // A damaged state has to be refused, leaving the machine alone.
        const std::vector<u8> before = C.emu->save_state();
        std::vector<u8> bad = state;
        bad.resize(bad.size() / 2);
        if (C.emu->load_state(bad, refused) || C.emu->save_state() != before) {
            std::printf("FAIL: a truncated state was accepted or changed the machine\n");
            return 1;
        }
        std::string err;
        if (!C.emu->load_state(state, err)) {
            std::printf("FAIL load into a used emulator: %s\n", err.c_str());
            return 1;
        }
        if (C.emu->save_state() != state) {
            std::printf("FAIL: saving a loaded state doesn't give the same bytes (used emulator)\n");
            return 1;
        }
        const std::string diff = compare(ref, continue_run(C, o), images, o.at);
        if (!diff.empty()) {
            std::printf("FAIL (used emulator): %s\n", diff.c_str());
            return 1;
        }
    }

    std::printf("OK state=%zu bytes file=%llu bytes save=%.1fms write=%.1fms read=%.1fms load=%.1fms "
                "(damaged state refused: \"%s\")\n",
                state.size(), file_size, save_ms, write_ms, read_ms, load_ms, refused.c_str());
    return 0;
}
