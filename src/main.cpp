#include "emulator.hpp"
#include "ui/app.hpp"
#include "profiler.hpp"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <limits>

namespace fs = std::filesystem;

static bool save_bmp(const std::string& filename, const u32* pixels, int width, int height) {
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) return false;

    u32 file_size = 54 + (width * height * 4);
    u32 data_offset = 54;
    u32 header_size = 40;
    u16 planes = 1;
    u16 bpp = 32;

    // Bitmap File Header
    f.put('B').put('M');
    f.write(reinterpret_cast<const char*>(&file_size), 4);
    u32 reserved = 0;
    f.write(reinterpret_cast<const char*>(&reserved), 4);
    f.write(reinterpret_cast<const char*>(&data_offset), 4);

    // DIB Header
    f.write(reinterpret_cast<const char*>(&header_size), 4);
    f.write(reinterpret_cast<const char*>(&width), 4);
    f.write(reinterpret_cast<const char*>(&height), 4);
    f.write(reinterpret_cast<const char*>(&planes), 2);
    f.write(reinterpret_cast<const char*>(&bpp), 2);
    u32 compression = 0;
    f.write(reinterpret_cast<const char*>(&compression), 4);
    u32 image_size = width * height * 4;
    f.write(reinterpret_cast<const char*>(&image_size), 4);
    u32 ppm = 2835;
    f.write(reinterpret_cast<const char*>(&ppm), 4);
    f.write(reinterpret_cast<const char*>(&ppm), 4);
    u32 colors = 0;
    f.write(reinterpret_cast<const char*>(&colors), 4);
    f.write(reinterpret_cast<const char*>(&colors), 4);

    // Pixels (BMP stores bottom-to-top, BGRA)
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            u32 pixel = pixels[y * width + x];
            u8 b = pixel & 0xFF;
            u8 g = (pixel >> 8) & 0xFF;
            u8 r = (pixel >> 16) & 0xFF;
            u8 a = (pixel >> 24) & 0xFF;
            f.put(b).put(g).put(r).put(a);
        }
    }

    return true;
}

static bool save_wav(const std::string& filename, const std::vector<s16>& samples, u32 rate) {
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) return false;
    auto u32le = [&](u32 v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16le = [&](u16 v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    const u32 data_bytes = static_cast<u32>(samples.size() * 2);
    f.write("RIFF", 4); u32le(36 + data_bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); u32le(16); u16le(1); u16le(2); u32le(rate); u32le(rate * 4); u16le(4); u16le(16);
    f.write("data", 4); u32le(data_bytes);
    f.write(reinterpret_cast<const char*>(samples.data()), data_bytes);
    return true;
}

int main(int argc, char* argv[]) {
    std::string rom_path;
    bool headless = false;
    std::string ui_test_dir;
    int headless_frames = 180;
    int internal_scale = 1;
    std::string screenshot_path;
    std::string dump_ram_path;
    std::string mash_btn;
    std::vector<std::pair<int, std::string>> scheduled_presses;
    std::vector<std::pair<int, std::string>> scheduled_screenshots;
    std::string cpu_core_arg;
    bool jit_stats = false;
    std::string profile_path;
    bool use_save_file = true;
    std::string wav_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless" && i + 1 < argc) {
            headless = true;
            headless_frames = std::stoi(argv[++i]);
        } else if (arg == "--scale" && i + 1 < argc) {
            internal_scale = std::stoi(argv[++i]);
        } else if (arg == "--screenshot" && i + 1 < argc) {
            screenshot_path = argv[++i];
        } else if (arg == "--screenshot-at" && i + 1 < argc) {
            std::string p = argv[++i];
            size_t colon = p.find(':');
            if (colon != std::string::npos) {
                int frame = std::stoi(p.substr(0, colon));
                std::string path = p.substr(colon + 1);
                scheduled_screenshots.emplace_back(frame, path);
            }
        } else if (arg == "--ui-test" && i + 1 < argc) {
            ui_test_dir = argv[++i];
        } else if (arg == "--dump-ram" && i + 1 < argc) {
            dump_ram_path = argv[++i];
        } else if (arg == "--cpu" && i + 1 < argc) {
            cpu_core_arg = argv[++i]; // "interp" or "jit"
        } else if (arg == "--jit-stats") {
            jit_stats = true;
        } else if (arg == "--no-save") {
            use_save_file = false; // don't read or write the .sav next to the ROM
        } else if (arg == "--wav" && i + 1 < argc) {
            wav_path = argv[++i]; // raw AI output (native rate, s16 stereo) of the run
        } else if (arg == "--profile" && i + 1 < argc) {
            profile_path = argv[++i]; // sampling profile of the run, see src/profiler.hpp
        } else if (arg == "--mash" && i + 1 < argc) {
            mash_btn = argv[++i];
        } else if (arg == "--press" && i + 1 < argc) {
            std::string p = argv[++i];
            size_t colon = p.find(':');
            if (colon != std::string::npos) {
                int frame = std::stoi(p.substr(0, colon));
                std::string btn = p.substr(colon + 1);
                scheduled_presses.emplace_back(frame, btn);
            }
        } else if (rom_path.empty() && arg[0] != '-') {
            rom_path = arg;
        }
    }

    if (!headless) {
        // Graphical frontend. A ROM passed on the command line starts immediately;
        // otherwise the game library is shown.
        ui::App app;
        if (!ui_test_dir.empty()) app.enable_ui_test(ui_test_dir);
        return app.run(rom_path);
    }

    if (rom_path.empty()) {
        std::vector<std::string> found_roms;
        for (const char* dir : {"roms", "."}) {
            std::error_code ec;
            if (!fs::is_directory(dir, ec)) continue;
            for (const auto& entry : fs::directory_iterator(dir, ec)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    if (ext == ".z64" || ext == ".v64" || ext == ".n64") {
                        found_roms.push_back(entry.path().string());
                    }
                }
            }
            if (!found_roms.empty()) break;
        }
        if (found_roms.empty()) {
            std::cerr << "Usage: n64 [rom_file] [--headless <frames>] [--scale <1-8>] [--screenshot <path.bmp>] [--mash <btn>] [--press <frame:btn>]\n";
            return 1;
        }
        std::sort(found_roms.begin(), found_roms.end());
        // Non-interactive mode: just pick the first ROM found.
        rom_path = found_roms[0];
        std::cout << "[Main] No ROM specified, using: " << rom_path << "\n";
    }

    Emulator emu;
    emu.get_cartridge().set_use_save_file(use_save_file);
    if (!emu.load_rom(rom_path)) {
        std::cerr << "[Main] Failed to load ROM: " << rom_path << "\n";
        return 1;
    }
    // Internal resolution of the RDP (screenshots come out that many times larger).
    emu.get_rdp().set_hires_scale(static_cast<u32>(std::clamp(internal_scale, 1, 8)));
    if (cpu_core_arg == "interp") emu.set_cpu_core(CpuCore::Interpreter);
    else if (cpu_core_arg == "jit") emu.set_cpu_core(CpuCore::Recompiler);
    emu.get_jit().set_stats_enabled(jit_stats);
    emu.set_profiling(jit_stats);
    std::vector<s16> wav_samples;
    if (!wav_path.empty()) emu.get_ai().set_capture(&wav_samples);
    const auto run_start = std::chrono::steady_clock::now();

    std::vector<u32> frame_pixels;
    int frame_w = 320, frame_h = 240;

    if (headless) {
        std::cout << "[Main] Running in headless mode for " << headless_frames << " frames...\n";
        if (!profile_path.empty() && !profiler::start(profile_path)) {
            std::cerr << "[Main] --profile is not supported on this platform\n";
        }
        for (int frame = 0; frame < headless_frames; ++frame) {
            // Apply scheduled button presses
            for (const auto& sp : scheduled_presses) {
                if (frame == sp.first) {
                    std::cout << "[Main] Frame " << frame << ": Pressing " << sp.second << "\n";
                    emu.get_controller(0).press_named_button(sp.second, true);
                } else if (frame == sp.first + 10) {
                    emu.get_controller(0).press_named_button(sp.second, false);
                }
            }

            // Apply mashing
            if (!mash_btn.empty()) {
                bool state = ((frame / 6) % 2) == 0;
                emu.get_controller(0).press_named_button(mash_btn, state);
            }

            emu.step_frame();

            for (const auto& ss : scheduled_screenshots) {
                if (frame == ss.first) {
                    emu.render_frame(frame_pixels, frame_w, frame_h);
                    if (save_bmp(ss.second, frame_pixels.data(), frame_w, frame_h)) {
                        std::cout << "[Main] Scheduled screenshot saved to: " << ss.second << " (frame " << frame << ")\n";
                    }
                }
            }

            if (frame % 60 == 0) {
                std::cout << "[Main] Frame " << frame << " / " << headless_frames << "\n";
            }
        }

        profiler::stop();

        {
            // Wall time plus a fingerprint of the final machine state, so two
            // runs (e.g. --cpu interp vs --cpu jit) can be compared at a glance.
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
            u64 hash = 0xCBF29CE484222325ULL; // FNV-1a
            const u8* ram = emu.get_bus().get_rdram();
            for (size_t i = 0; i < emu.get_bus().get_rdram_size(); ++i) hash = (hash ^ ram[i]) * 0x100000001B3ULL;
            std::cout << "[Main] core=" << (emu.cpu_core() == CpuCore::Recompiler ? "jit" : "interp")
                      << " frames=" << headless_frames << " time=" << std::fixed << std::setprecision(2) << secs
                      << "s (" << (secs > 0 ? headless_frames / secs : 0.0) << " fps)"
                      << " rdram_hash=" << std::hex << hash << std::dec << "\n";
            // What the game ran on, for compatibility reports (tools/compat_sweep.py).
            static const char* ucodes[] = {"none", "Fast3D", "F3DEX", "F3DEX2", "S2DEX", "S2DEX2", "F3DGOLDEN", "F3DPD"};
            RDP& rdp = emu.get_rdp();
            RSP& rsp = emu.get_rsp();
            const int uc = rdp.has_processed_display_list() ? static_cast<int>(rdp.get_active_ucode()) : 0;
            std::cout << "[Main] info: gfx_ucode=" << ucodes[uc < 8 ? uc : 0] << " gfx_tasks=" << rsp.get_gfx_task_count()
                      << " display_lists=" << rdp.get_display_list_count() << " audio_tasks=" << rsp.get_audio_task_count()
                      << " audio_abi=" << (rsp.get_audio_task_count() ? rsp.get_audio_hle().get_abi_index() : -1)
                      << " ai_rate=" << emu.get_ai().get_native_sample_rate()
                      << " save=" << static_cast<int>(emu.get_cartridge().get_save_type())
                      << " cic=" << static_cast<int>(emu.get_cartridge().get_cic_type())
                      << " title=\"" << emu.get_cartridge().get_title() << "\"\n";
            if (jit_stats) {
                std::cout << "[Main] time split: cpu=" << emu.prof_cpu_seconds() << "s, ai/vi/rsp/rdp="
                          << emu.prof_other_seconds() << "s\n";
                if (emu.cpu_core() == CpuCore::Recompiler) emu.get_jit().print_stats(std::cout);
            }
        }

        if (!wav_path.empty()) {
            const u32 rate = emu.get_ai().get_native_sample_rate();
            u64 hash = 0xCBF29CE484222325ULL;
            for (s16 v : wav_samples) hash = (hash ^ static_cast<u16>(v)) * 0x100000001B3ULL;
            if (save_wav(wav_path, wav_samples, rate ? rate : 32000)) {
                std::cout << "[Main] Audio: " << wav_samples.size() / 2 << " frames @ " << rate << " Hz, abi="
                          << emu.get_rsp().get_audio_hle().get_abi_index() << " hash=" << std::hex << hash << std::dec
                          << " -> " << wav_path << "\n";
            }
        }

        emu.render_frame(frame_pixels, frame_w, frame_h);

        if (!screenshot_path.empty()) {
            if (save_bmp(screenshot_path, frame_pixels.data(), frame_w, frame_h)) {
                std::cout << "[Main] Screenshot saved to: " << screenshot_path << "\n";
            } else {
                std::cerr << "[Main] Failed to save screenshot to: " << screenshot_path << "\n";
            }
        }

        if (!dump_ram_path.empty()) {
            std::ofstream f(dump_ram_path, std::ios::binary);
            if (f.is_open()) {
                f.write(reinterpret_cast<const char*>(emu.get_bus().get_rdram()), emu.get_bus().get_rdram_size());
                std::cout << "[Main] RAM dumped to: " << dump_ram_path << " (" << emu.get_bus().get_rdram_size() << " bytes)\n";
            } else {
                std::cerr << "[Main] Failed to dump RAM to: " << dump_ram_path << "\n";
            }
        }

        std::cout << "[Main] Headless run completed successfully.\n";
        return 0;
    }
}
