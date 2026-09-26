// Checks the GPU RDP renderer (src/gpu/) against the CPU one (src/hires.cpp).
//
// Runs a ROM twice with the same input, once per renderer, at the same
// internal resolution, and compares the displayed frames: how many pixels
// differ, and by how much. Both pipelines run the same arithmetic, so apart
// from floating-point rounding on the GPU (fused multiply-adds, reciprocals)
// the frames must match. --bench instead measures the GPU renderer alone.
//
//   make gpu_check -> bin/gpu_check
//   bin/gpu_check rom.z64 [--scale S] [--frames N] [--every K] [--mash btn]
//                 [--tolerance T] [--shots dir] [--all] [--bench] [--int]
//
// --every K compares every K-th frame (reading GPU frames back is slow).
// --shots writes cpu/gpu/diff PNGs of the worst frame. ORBIT64_GPU_DRIVER
// picks the driver (e.g. vulkan through MoltenVK on macOS).

#include "emulator.hpp"
#include "gpu/device.hpp"
#include "gpu/rdp_gpu.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

struct Options {
    std::string rom, mash, shots;
    bool all = false;
    int frames = 900, every = 10, scale = 4, tolerance = 2;
    bool bench = false, interp = false;
};

using Frames = std::map<int, VideoFrame>;

void save_png(const std::string& path, const std::vector<u32>& px, int w, int h) {
    std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        rgb[i * 3 + 0] = (px[i] >> 16) & 0xFF;
        rgb[i * 3 + 1] = (px[i] >> 8) & 0xFF;
        rgb[i * 3 + 2] = px[i] & 0xFF;
    }
    stbi_write_png(path.c_str(), w, h, 3, rgb.data(), w * 3);
}

// Runs the ROM, keeping every `every`-th frame (as pixels). Returns the
// average milliseconds per frame after the first 60.
double run(const Options& o, const HiResFactory& factory, Frames* keep) {
    Emulator emu;
    emu.get_cartridge().set_use_save_file(false);
    if (!emu.load_rom(o.rom)) {
        std::fprintf(stderr, "can't load %s\n", o.rom.c_str());
        std::exit(1);
    }
    emu.set_cpu_core(o.interp ? CpuCore::Interpreter : CpuCore::Recompiler);
    emu.get_rdp().set_hires_factory(factory);
    emu.get_rdp().set_hires_scale(static_cast<u32>(o.scale));
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0 = Clock::now();
    VideoFrame f;
    for (int i = 0; i < o.frames; ++i) {
        if (i == 60) t0 = Clock::now();
        if (!o.mash.empty()) emu.get_controller(0).press_named_button(o.mash, ((i / 6) % 2) == 0);
        emu.step_frame();
        emu.render_frame(f);
        if (keep && i % o.every == 0) {
            VideoFrame& k = (*keep)[i];
            f.to_pixels(k.pixels);
            k.w = f.w;
            k.h = f.h;
            k.scale = f.scale;
        }
    }
    if (f.gpu) f.gpu->read(f.pixels); // waits for the GPU to finish
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return o.frames > 60 ? ms / (o.frames - 60) : 0.0;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--scale") o.scale = std::stoi(next());
        else if (a == "--frames") o.frames = std::stoi(next());
        else if (a == "--every") o.every = std::max(1, std::stoi(next()));
        else if (a == "--mash") o.mash = next();
        else if (a == "--tolerance") o.tolerance = std::stoi(next());
        else if (a == "--shots") o.shots = next();
        else if (a == "--bench") o.bench = true;
        else if (a == "--int") o.interp = true;
        else if (a == "--all") o.all = true;
        else o.rom = a;
    }
    if (o.rom.empty()) {
        std::fprintf(stderr, "usage: gpu_check rom.z64 [options] (see tools/gpu_check.cpp)\n");
        return 1;
    }
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GPUDevice* device = gpu::create_device();
    auto dev = std::make_shared<gpu::Device>(device);
    if (!dev->ok()) {
        std::fprintf(stderr, "GPU renderer unavailable: %s\n", dev->error().c_str());
        return 1;
    }
    std::printf("driver=%s scale=%d\n", dev->driver(), o.scale);
    std::fflush(stdout);
    HiResFactory gpu_factory = gpu::make_hires_factory(dev);

    int status = 0;
    if (o.bench) {
        const double ms = run(o, gpu_factory, nullptr);
        std::printf("gpu: %.2f ms/frame (%.0f fps) at %dx\n", ms, ms > 0 ? 1000.0 / ms : 0.0, o.scale);
    } else {
        Frames cpu, gpu;
        const double cpu_ms = run(o, nullptr, &cpu);
        const double gpu_ms = run(o, gpu_factory, &gpu);
        long long total = 0, differ = 0;
        int worst_frame = -1, max_diff = 0;
        long long worst_count = 0;
        int size_mismatch = 0;
        for (const auto& [n, c] : cpu) {
            const VideoFrame& g = gpu[n];
            if (c.w != g.w || c.h != g.h || c.pixels.size() != g.pixels.size()) {
                ++size_mismatch;
                continue;
            }
            long long frame_differ = 0;
            for (size_t i = 0; i < c.pixels.size(); ++i) {
                int d = 0;
                for (int sh = 0; sh < 24; sh += 8)
                    d = std::max(d, std::abs(static_cast<int>((c.pixels[i] >> sh) & 0xFF) - static_cast<int>((g.pixels[i] >> sh) & 0xFF)));
                max_diff = std::max(max_diff, d);
                if (d > o.tolerance) ++frame_differ;
            }
            total += static_cast<long long>(c.pixels.size());
            differ += frame_differ;
            if (frame_differ > worst_count) {
                worst_count = frame_differ;
                worst_frame = n;
            }
        }
        std::printf("frames=%zu pixels=%lld differ(>%d)=%lld (%.4f%%) max_channel_diff=%d worst_frame=%d (%lld px) "
                    "size_mismatch=%d cpu=%.2fms gpu=%.2fms\n",
                    cpu.size(), total, o.tolerance, differ, total ? 100.0 * differ / total : 0.0, max_diff, worst_frame,
                    worst_count, size_mismatch, cpu_ms, gpu_ms);
        if (!o.shots.empty() && worst_frame >= 0) {
            const VideoFrame& c = cpu[worst_frame];
            const VideoFrame& g = gpu[worst_frame];
            std::vector<u32> diff(c.pixels.size());
            for (size_t i = 0; i < diff.size(); ++i) {
                int d = 0;
                for (int sh = 0; sh < 24; sh += 8)
                    d = std::max(d, std::abs(static_cast<int>((c.pixels[i] >> sh) & 0xFF) - static_cast<int>((g.pixels[i] >> sh) & 0xFF)));
                diff[i] = d > o.tolerance ? 0xFFFF0000u : ((c.pixels[i] >> 2) & 0x3F3F3F);
            }
            const std::string base = o.shots + "/frame" + std::to_string(worst_frame);
            save_png(base + "_cpu.png", c.pixels, c.w, c.h);
            save_png(base + "_gpu.png", g.pixels, g.w, g.h);
            save_png(base + "_diff.png", diff, c.w, c.h);
        }
        if (!o.shots.empty() && o.all)
            for (const auto& [n, g] : gpu) {
                save_png(o.shots + "/all" + std::to_string(n) + "_gpu.png", g.pixels, g.w, g.h);
                save_png(o.shots + "/all" + std::to_string(n) + "_cpu.png", cpu[n].pixels, cpu[n].w, cpu[n].h);
            }
        status = differ > total / 1000 || size_mismatch ? 2 : 0;
    }
    gpu_factory = nullptr;
    dev.reset();
    SDL_DestroyGPUDevice(device);
    SDL_Quit();
    return status;
}
