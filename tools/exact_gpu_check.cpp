// Checks the bit-exact RDP on the GPU (src/gpu/rdp_exact_gpu.*) against the
// one on the CPU (ExactRdp): replays an RDP trace (ORBIT64_RDP_TRACE, see
// rdp.cpp) through both, each with its own RDRAM, and compares all of RDRAM
// and every halfword's ninth bits after each Sync Full - they must be
// identical. --each-draw also flushes the GPU after every primitive, which
// finds the first one that differs.
//
//   make exact_gpu_check -> bin/exact_gpu_check
//   bin/exact_gpu_check trace [--frames N] [--each-draw [--from F]] [--max-report N] [--bench]
//   bin/exact_gpu_check --rom game.z64 [--frames N] [--mash btn] [--bench] [--twice cpu|gpu]
//
// --bench replays through the GPU only and reports the time per frame.
// --rom runs a game on the low-level RSP twice, the bit-exact RDP on the CPU
// (lanes) and on the GPU, and compares RDRAM after every frame (the games
// must run identically); with --bench only the GPU run, timed.

#include "emulator.hpp"
#include "gpu/device.hpp"
#include "gpu/rdp_exact_gpu.hpp"
#include "rdp_exact.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

void save_png(const std::string& path, const std::vector<u32>& px, int w, int h) {
    std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        rgb[i * 3] = (px[i] >> 16) & 0xff;
        rgb[i * 3 + 1] = (px[i] >> 8) & 0xff;
        rgb[i * 3 + 2] = px[i] & 0xff;
    }
    stbi_write_png(path.c_str(), w, h, 3, rgb.data(), w * 3);
}

bool rd32(std::FILE* f, u32& v) {
    u8 b[4];
    if (std::fread(b, 1, 4, f) != 4) return false;
    v = b[0] | b[1] << 8 | b[2] << 16 | static_cast<u32>(b[3]) << 24;
    return true;
}

u64 hash_bytes(const u8* p, size_t n) {
    u64 h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i += 8) {
        u64 v;
        std::memcpy(&v, p + i, 8);
        h = (h ^ v) * 1099511628211ull;
    }
    return h;
}

// Runs `rom` for `frames` frames; returns ms per frame (after the first 60)
// and each frame's RDRAM hash.
struct Shot {
    std::vector<u32> px;
    int w = 0, h = 0;
};
double run_rom(const std::string& rom, int frames, const std::string& mash, const ExactAccelFactory& accel,
               std::vector<u64>& hashes, int scale, int every, std::vector<Shot>* shots) {
    Emulator emu;
    emu.get_cartridge().set_use_save_file(false);
    if (!emu.load_rom(rom)) {
        std::fprintf(stderr, "can't load %s\n", rom.c_str());
        std::exit(1);
    }
    emu.get_rsp().set_mode(RspMode::LLEGraphics);
    if (std::getenv("INTERP")) emu.set_cpu_core(CpuCore::Interpreter); // (debugging)
    emu.get_rdp().set_exact(true);
    emu.get_rdp().set_exact_accel_factory(accel);
    emu.get_rdp().set_exact_gpu(accel != nullptr);
    emu.get_rdp().set_hires_scale(static_cast<u32>(scale));
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    VideoFrame f;
    hashes.clear();
    for (int i = 0; i < frames; ++i) {
        if (i == 60) t0 = Clock::now();
        if (!mash.empty()) emu.get_controller(0).press_named_button(mash, ((i / 6) % 2) == 0);
        emu.step_frame();
        emu.render_frame(f);
        if (shots && every > 0 && i % every == 0) {
            Shot sh;
            f.to_pixels(sh.px);
            sh.w = f.w;
            sh.h = f.h;
            shots->push_back(std::move(sh));
        }
        hashes.push_back(hash_bytes(emu.get_bus().get_rdram(), emu.get_bus().get_rdram_size()));
        static const int dump_up = [] { const char* e = std::getenv("DUMP_UP_AT"); return e ? std::atoi(e) : -1; }(); // (debugging)
        if (i == dump_up && emu.get_rdp().exact_rdp()->upscaled()) {
            static int run = 0;
            const UpStore* up = emu.get_rdp().exact_rdp()->upscaled();
            const std::string name = "/tmp/up_" + std::to_string(run++) + ".bin";
            if (std::FILE* d = std::fopen(name.c_str(), "wb")) {
                for (u32 sl = 0; sl < up->scale * up->scale; ++sl) std::fwrite(up->color(sl), 1, up->size, d);
                std::fclose(d);
            }
        }
        static const int dump_at = [] { const char* e = std::getenv("DUMP_AT"); return e ? std::atoi(e) : -1; }(); // (debugging)
        if (i == dump_at) {
            static int run = 0;
            const std::string name = "/tmp/rdram_" + std::to_string(run++) + ".bin";
            if (std::FILE* d = std::fopen(name.c_str(), "wb")) {
                std::fwrite(emu.get_bus().get_rdram(), 1, emu.get_bus().get_rdram_size(), d);
                std::fclose(d);
            }
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return frames > 60 ? ms / (frames - 60) : 0.0;
}

} // namespace

int main(int argc, char** argv) {
    std::string path, rom, mash, twice;
    int scale = 1, every = 0;
    std::string shots;
    long max_frames = 1L << 40, from_frame = 0;
    bool each_draw = false, bench = false, verbose = false;
    int max_report = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--frames" && i + 1 < argc) max_frames = std::atol(argv[++i]);
        else if (a == "--each-draw") each_draw = true;
        else if (a == "--from" && i + 1 < argc) from_frame = std::atol(argv[++i]);
        else if (a == "--bench") bench = true;
        else if (a == "-v") verbose = true;
        else if (a == "--rom" && i + 1 < argc) rom = argv[++i];
        else if (a == "--mash" && i + 1 < argc) mash = argv[++i];
        else if (a == "--twice" && i + 1 < argc) twice = argv[++i];
        else if (a == "--scale" && i + 1 < argc) scale = std::atoi(argv[++i]);
        else if (a == "--every" && i + 1 < argc) every = std::atoi(argv[++i]);
        else if (a == "--shots" && i + 1 < argc) shots = argv[++i];
        else if (a == "--max-report" && i + 1 < argc) max_report = std::atoi(argv[++i]);
        else path = a;
    }
    if (path.empty() && rom.empty()) {
        std::fprintf(stderr, "usage: exact_gpu_check trace [--frames N] [--each-draw] [--max-report N] [--bench]\n");
        return 1;
    }
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GPUDevice* device = gpu::create_device();
    auto dev = std::make_shared<gpu::Device>(device);
    if (!dev->exact_ok()) {
        std::fprintf(stderr, "GPU bit-exact RDP unavailable\n");
        return 1;
    }
    std::printf("driver=%s\n", dev->driver());
    std::fflush(stdout);

    if (!rom.empty()) {
        const int frames = max_frames < (1L << 30) ? static_cast<int>(max_frames) : 1200;
        std::vector<u64> a, b;
        double cpu_ms = 0;
        // (--twice cpu|gpu: the same back end twice - whether the game runs the same at all)
        ExactAccelFactory gpu_factory = gpu::make_exact_accel_factory(dev);
        std::vector<Shot> sa, sb;
        if (!bench) cpu_ms = run_rom(rom, frames, mash, twice == "gpu" ? gpu_factory : nullptr, a, scale, every, &sa);
        // (--twice none: the CPU only, e.g. to record an RDP trace of it)
        const double gpu_ms = twice == "none" ? 0.0 : run_rom(rom, frames, mash, twice == "cpu" ? nullptr : gpu_factory, b, scale, every, &sb);
        int first_diff = -1, diffs = 0;
        for (size_t i = 0; !bench && i < a.size() && i < b.size(); ++i)
            if (a[i] != b[i]) {
                if (first_diff < 0) first_diff = static_cast<int>(i);
                ++diffs;
            }
        // (--every K: the shown frames too, every K-th - at internal resolution
        // the high-resolution copies are compared through them)
        long long px_total = 0, px_diff = 0;
        int img_frames = 0;
        for (size_t i = 0; i < sa.size() && i < sb.size(); ++i) {
            if (sa[i].px.size() != sb[i].px.size()) {
                ++img_frames;
                continue;
            }
            long long d = 0;
            for (size_t k = 0; k < sa[i].px.size(); ++k) d += sa[i].px[k] != sb[i].px[k];
            px_total += static_cast<long long>(sa[i].px.size());
            px_diff += d;
            if (d) ++img_frames;
            if (d && !shots.empty()) {
                const std::string base = shots + "/f" + std::to_string(i * every);
                save_png(base + "_cpu.png", sa[i].px, sa[i].w, sa[i].h);
                save_png(base + "_gpu.png", sb[i].px, sb[i].w, sb[i].h);
            }
        }
        if (every > 0)
            std::printf("images: %zu compared, %d differ, %lld of %lld pixels (%.4f%%)\n", sa.size(), img_frames, px_diff,
                        px_total, px_total ? 100.0 * px_diff / px_total : 0.0);
        std::printf("%s: %d frames, %d differ (first %d); cpu %.2f ms/frame, gpu %.2f ms/frame\n",
                    bench ? "BENCH" : diffs ? "FAIL" : "OK", frames, diffs, first_diff, cpu_ms, gpu_ms);
        std::fflush(stdout);
        gpu_factory = nullptr; // (holds the device)
        dev.reset();
        SDL_DestroyGPUDevice(device);
        SDL_Quit();
        return diffs ? 2 : 0;
    }
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "can't open %s\n", path.c_str());
        return 1;
    }

    const size_t size = 8u << 20;
    std::vector<u8> A(size), B(size), pre, pre_hidden;
    auto ref = std::make_unique<ExactRdp>();
    auto test = std::make_unique<ExactRdpLanes>();
    test->set_accel(gpu::make_exact_accel_factory(dev)());

    u32 words[64];
    u32 om_h = 0, om_l = 0, ci = 0, ci_w = 0, zi = 0, cc0 = 0, cc1 = 0;
    long frame = 0, ncmd = 0, bad_frames = 0, reports = 0;
    auto compare = [&](const char* when) -> bool {
        long diff_bytes = 0, diff_hidden = 0;
        long first = -1;
        for (size_t a = 0; a < size; a += 4096) {
            if (std::memcmp(&A[a], &B[a], 4096) == 0) continue;
            for (size_t b = a; b < a + 4096; ++b)
                if (A[b] != B[b]) {
                    if (first < 0) first = static_cast<long>(b);
                    ++diff_bytes;
                }
        }
        long first_h = -1;
        for (size_t h = 0; h < size / 2; ++h) {
            const u16 wa = static_cast<u16>(A[h * 2] << 8 | A[h * 2 + 1]), wb = static_cast<u16>(B[h * 2] << 8 | B[h * 2 + 1]);
            if (wa != wb) continue; // counted above
            if (ref->hidden_at(h, wa) != test->primary().hidden_at(h, wb)) {
                if (first_h < 0) first_h = static_cast<long>(h * 2);
                ++diff_hidden;
            }
        }
        if (!diff_bytes && !diff_hidden) return true;
        if (reports++ < max_report)
            std::printf("frame %ld cmd %ld (%s): %ld bytes differ (first %06lx: cpu %02x gpu %02x), %ld ninth bits (first %06lx)\n",
                        frame, ncmd, when, diff_bytes, first, first >= 0 ? A[first] : 0, first >= 0 ? B[first] : 0, diff_hidden,
                        first_h);
        if (verbose) {
            int shown = 0;
            for (size_t h = 0; h < size / 2 && shown < 24; ++h) {
                const u16 wa = static_cast<u16>(A[h * 2] << 8 | A[h * 2 + 1]), wb = static_cast<u16>(B[h * 2] << 8 | B[h * 2 + 1]);
                const u8 ha = ref->hidden_at(h, wa), hb = test->primary().hidden_at(h, wb);
                if (wa == wb && ha == hb) continue;
                const u16 wp = pre.empty() ? 0 : static_cast<u16>(pre[h * 2] << 8 | pre[h * 2 + 1]);
                std::printf("    %06zx: cpu %04x/%u gpu %04x/%u (before %04x/%u)\n", h * 2, wa, ha, wb, hb, wp,
                            pre.empty() ? 9 : pre_hidden[h]);
                ++shown;
            }
        }
        // Carry on from the CPU's picture.
        B = A;
        for (size_t h = 0; h < size / 2; ++h) {
            const u16 w = static_cast<u16>(A[h * 2] << 8 | A[h * 2 + 1]);
            test->primary().force_hidden(h, ref->hidden_at(h, w), w);
        }
        return false;
    };

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    double ref_ms = 0, gpu_ms = 0;
    int c;
    bool frame_bad = false;
    while ((c = std::fgetc(f)) != EOF && frame < max_frames) {
        if (c == 'M') {
            u32 addr, len;
            rd32(f, addr);
            rd32(f, len);
            std::vector<u8> buf(len);
            if (std::fread(buf.data(), 1, len, f) != len) break;
            std::memcpy(&A[addr], buf.data(), len);
            std::memcpy(&B[addr], buf.data(), len);
            ref->cpu_wrote(addr, len);
            test->cpu_wrote(addr, len);
            continue;
        }
        if (c != 'C') {
            std::fprintf(stderr, "bad record\n");
            return 1;
        }
        u32 n;
        rd32(f, n);
        for (u32 i = 0; i < n; ++i) rd32(f, words[i]);
        ++ncmd;
        const u32 op = (words[0] >> 24) & 0x3f;
        if (op == 0x2f) { om_h = words[0] & 0xffffff; om_l = words[1]; }
        if (op == 0x3f) { ci = words[1] & 0xffffff; ci_w = (words[0] & 0x3ff) + 1; }
        if (op == 0x3e) zi = words[1] & 0xffffff;
        if (op == 0x3c) { cc0 = words[0] & 0xffffff; cc1 = words[1]; }
        const bool is_draw = (op >= 0x08 && op <= 0x0f) || op == 0x24 || op == 0x25 || op == 0x36;
        if (verbose && each_draw && is_draw && frame >= from_frame) {
            pre = A;
            pre_hidden.resize(size / 2);
            for (size_t h = 0; h < size / 2; ++h) pre_hidden[h] = ref->hidden_at(h, static_cast<u16>(A[h * 2] << 8 | A[h * 2 + 1]));
        }
        auto r0 = Clock::now();
        if (!bench) ref->command(words, A.data(), size);
        auto r1 = Clock::now();
        test->command(words, B.data(), size);
        auto r2 = Clock::now();
        ref_ms += std::chrono::duration<double, std::milli>(r1 - r0).count();
        gpu_ms += std::chrono::duration<double, std::milli>(r2 - r1).count();
        const bool draw = (op >= 0x08 && op <= 0x0f) || op == 0x24 || op == 0x25 || op == 0x36;
        static const long watch = [] { const char* e = std::getenv("WATCH"); return e ? std::strtol(e, nullptr, 16) : -1L; }();
        if (watch >= 0 && frame >= from_frame) {
            test->flush();
            const u16 wa = static_cast<u16>(A[watch] << 8 | A[watch + 1]), wb = static_cast<u16>(B[watch] << 8 | B[watch + 1]);
            std::printf("cmd %ld op %02x %08x %08x: cpu %04x/%u gpu %04x/%u\n", ncmd, op, words[0], words[1], wa,
                        ref->hidden_at(watch / 2, wa), wb, test->primary().hidden_at(watch / 2, wb));
        }
        if (!bench && each_draw && draw && frame >= from_frame) {
            test->flush();
            if (!compare("draw")) {
                frame_bad = true;
                std::printf("  command %08x %08x (op %02x) other modes %06x:%08x combine %06x:%08x ci %06x w %u zi %06x\n",
                            words[0], words[1], op, om_h, om_l, cc0, cc1, ci, ci_w, zi);
            }
        }
        if (op == 0x29) {
            if (!bench && !compare("sync full")) frame_bad = true;
            if (frame_bad) ++bad_frames;
            frame_bad = false;
            ++frame;
        }
    }
    const double total = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::printf("%s: %ld frames, %ld commands, %ld frames differ; cpu %.2f ms/frame, gpu %.2f ms/frame (total %.0f ms)\n",
                bad_frames ? "FAIL" : "OK", frame, ncmd, bad_frames, frame ? ref_ms / frame : 0.0,
                frame ? gpu_ms / frame : 0.0, total);
    test.reset();
    dev.reset();
    SDL_DestroyGPUDevice(device);
    SDL_Quit();
    return bad_frames ? 2 : 0;
}
