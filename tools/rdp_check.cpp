// Headless RDP checker. Runs a ROM the way the frontend does (step_frame +
// render_frame every frame), then prints an FNV-1a hash of RDRAM and of every
// displayed frame, plus the speed after a warm-up.
//
//   make rdp_check         -> bin/rdp_check
//   make rdp_check_exact   -> bin/rdp_check_exact
//
// rdp_check_exact is built with HIRES_EXACT_TEST: its high-resolution pass
// keeps exactly what RDRAM holds, and --scale 1 runs that pass instead of
// switching it off. Its frame hash must then equal rdp_check's at --scale 1,
// which checks the whole high-resolution pipeline (recording, TMEM and state
// copies, worker threads, composing) against the native renderer.
//
//   bin/rdp_check rom.z64 [--frames N] [--scale S] [--warm N] [--int]
//                 [--mash btn] [--mash-from frame] [--press frame:btn] [--shot frame:out.png]
//                 [--shot-every n:dir]
//                 [--frame-log out.txt] [--timing] [--dump-ram out.bin] [--raw frame:out.argb] [--vi-native]
//
// --vi-native shows the frame buffer as it is instead of through the exact
// VI (which native-resolution frames otherwise go through).
// --timing skips the per-frame hashing (it costs several ms per frame at
// high scales). --frame-log writes one hash per frame, to find the first frame
// two builds disagree on.

#include "emulator.hpp"
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void save_png(const std::string& path, const std::vector<u32>& px, int w, int h) {
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        rgb[i * 3 + 0] = (px[i] >> 16) & 0xFF;
        rgb[i * 3 + 1] = (px[i] >> 8) & 0xFF;
        rgb[i * 3 + 2] = px[i] & 0xFF;
    }
    stbi_write_png(path.c_str(), w, h, 3, rgb.data(), w * 3);
}

int main(int argc, char** argv) {
    std::string rom, mash, frame_log, dump_ram, shot_dir;
    int mash_from = 0, shot_every = 0;
    int frames = 900, scale = 1, warm = 300;
    bool interp = false, timing = false, vi_native = false;
    std::vector<std::pair<int, std::string>> presses, shots, raws;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--frames") frames = std::stoi(next());
        else if (a == "--scale") scale = std::stoi(next());
        else if (a == "--warm") warm = std::stoi(next());
        else if (a == "--int") interp = true;
        else if (a == "--vi-native") vi_native = true;
        else if (a == "--timing") timing = true;
        else if (a == "--mash") mash = next();
        else if (a == "--mash-from") mash_from = std::stoi(next());
        else if (a == "--shot-every") {
            std::string v = next();
            size_t c = v.find(':');
            if (c == std::string::npos) continue;
            shot_every = std::stoi(v.substr(0, c));
            shot_dir = v.substr(c + 1);
        }
        else if (a == "--frame-log") frame_log = next();
        else if (a == "--dump-ram") dump_ram = next();
        else if (a == "--press" || a == "--shot" || a == "--raw") {
            std::string v = next();
            size_t c = v.find(':');
            if (c == std::string::npos) continue;
            (a == "--press" ? presses : a == "--shot" ? shots : raws).emplace_back(std::stoi(v.substr(0, c)), v.substr(c + 1));
        } else rom = a;
    }
    Emulator emu;
    if (rom.empty() || !emu.load_rom(rom)) {
        std::fprintf(stderr, "usage: rdp_check rom.z64 [options] (see tools/rdp_check.cpp)\n");
        return 1;
    }
    emu.set_cpu_core(interp ? CpuCore::Interpreter : CpuCore::Recompiler);
    if (vi_native) emu.set_vi_exact(false);
#ifdef HIRES_EXACT_TEST
    emu.get_rdp().force_hires(static_cast<u32>(scale));
#else
    emu.get_rdp().set_hires_scale(static_cast<u32>(scale));
#endif

    std::vector<u32> px;
    int w = 0, h = 0;
    uint64_t frames_hash = 1469598103934665603ull;
    FILE* flog = frame_log.empty() ? nullptr : std::fopen(frame_log.c_str(), "w");
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now(), t_warm = t0;
    for (int f = 0; f < frames; ++f) {
        if (f == warm) t_warm = Clock::now();
        for (const auto& p : presses) {
            if (f == p.first) emu.get_controller(0).press_named_button(p.second, true);
            else if (f == p.first + 10) emu.get_controller(0).press_named_button(p.second, false);
        }
        if (!mash.empty() && f >= mash_from) emu.get_controller(0).press_named_button(mash, ((f / 6) % 2) == 0);
        emu.step_frame();
        emu.render_frame(px, w, h);
        if (!timing) {
            frames_hash = fnv(frames_hash, &w, sizeof w);
            frames_hash = fnv(frames_hash, &h, sizeof h);
            frames_hash = fnv(frames_hash, px.data(), px.size() * 4);
        }
        if (flog) std::fprintf(flog, "%d %016llx\n", f, (unsigned long long)fnv(1469598103934665603ull, px.data(), px.size() * 4));
        for (const auto& s : shots)
            if (f == s.first) save_png(s.second, px, w, h);
        if (shot_every > 0 && f % shot_every == 0) {
            char name[64];
            std::snprintf(name, sizeof name, "/f%06d.png", f);
            save_png(shot_dir + name, px, w, h);
        }
        for (const auto& r : raws)
            if (f == r.first)
                if (FILE* rf = std::fopen(r.second.c_str(), "wb")) {
                    std::fwrite(px.data(), 4, px.size(), rf);
                    std::fclose(rf);
                }
    }
    auto t1 = Clock::now();
    if (flog) std::fclose(flog);
    if (!dump_ram.empty()) {
        if (FILE* f = std::fopen(dump_ram.c_str(), "wb")) {
            std::fwrite(emu.get_bus().get_rdram(), 1, emu.get_bus().get_rdram_size(), f);
            std::fclose(f);
        }
    }
    const uint64_t rdram_hash = fnv(1469598103934665603ull, emu.get_bus().get_rdram(), emu.get_bus().get_rdram_size());
    const double after = std::chrono::duration<double>(t1 - t_warm).count();
    std::fprintf(stderr, "rdram=%016llx frames=%016llx size=%dx%d total=%.2fs fps_after_warm=%.1f\n",
                 (unsigned long long)rdram_hash, (unsigned long long)frames_hash, w, h,
                 std::chrono::duration<double>(t1 - t0).count(), frames > warm ? (frames - warm) / after : 0.0);
    return 0;
}
