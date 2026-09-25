// Headless stress test of ui::EmuCore - the threaded emulator wrapper the
// frontend drives - without opening a window. The main thread plays the UI
// (the way --ui-test does, but faster and in a loop): it toggles turbo, the
// internal resolution, pause / frame advance / reset, debug capture, memory
// pokes and freezes, saves / loads / undoes save states, and restarts the
// game, while constantly pulling frames, audio, stats and debug snapshots. Build it with sanitizer-style checks
// (make core_stress enables _GLIBCXX_ASSERTIONS) to catch memory errors in the
// act instead of as a later heap corruption.
//
//   bin/core_stress rom.z64 [seconds] [--interp]

#include "ui/emu_core.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <thread>

using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: core_stress rom.z64 [seconds] [--interp]\n");
        return 2;
    }
    const std::string rom = argv[1];
    const int seconds = argc > 2 && argv[2][0] != '-' ? std::atoi(argv[2]) : 60;
    bool interp = false;
    for (int i = 2; i < argc; ++i) interp |= std::string(argv[i]) == "--interp";

    ui::EmuCore core;
    core.set_cpu_core(interp ? 0 : 1);
    core.set_audio_output(48000, 1024);
    std::string err;
    if (!core.start(rom, err)) {
        std::fprintf(stderr, "start failed: %s\n", err.c_str());
        return 1;
    }

    // Save states go to three slots in a scratch folder.
    const std::filesystem::path state_dir = std::filesystem::temp_directory_path() / "orbit64_core_stress_states";
    std::filesystem::remove_all(state_dir);
    auto slot = [&](unsigned i) { return state_dir / ("slot" + std::to_string(i % 3) + ".state"); };
    int saved = 0, loaded = 0, undone = 0, state_errors = 0;

    std::mt19937 rng(1234);
    std::vector<std::uint32_t> frame;
    std::vector<float> audio(1024);
    std::uint64_t seen = 0;
    const auto end = steady_clock::now() + std::chrono::seconds(seconds);
    int step = 0;
    while (steady_clock::now() < end) {
        ++step;
        int w = 0, h = 0, scale = 1;
        core.fetch_frame(frame, w, h, scale, seen);
        core.pull_audio(audio.data(), audio.size() / 2, 1.0f);
        (void)core.stats();
        for (ui::StateEvent ev; core.poll_state_event(ev);) {
            if (!ev.ok) {
                // Expected: loading a slot nothing was saved to yet, undoing
                // when there's nothing to undo (e.g. right after a restart).
                if (ev.error == "the file couldn't be opened" || ev.error == "there is no load to undo") continue;
                std::printf("STATE ERROR (op %d): %s\n", static_cast<int>(ev.op), ev.error.c_str());
                ++state_errors;
                continue;
            }
            (ev.op == ui::StateEvent::Op::Save ? saved : ev.op == ui::StateEvent::Op::Load ? loaded : undone)++;
        }
        if (auto snap = core.debug_snapshot()) {
            volatile std::uint32_t sink = snap->u32(0x80000400) + static_cast<std::uint32_t>(snap->rdram.size());
            (void)sink;
            // What the Object Viewer / Inspector rely on: consistent per-triangle
            // arrays and texture indices inside the frame's texture list.
            const size_t ntex = snap->textures ? snap->textures->size() : 0;
            if (snap->meshes) {
                for (const CapturedMesh& m : *snap->meshes) {
                    const size_t tris = m.pos.size() / 9;
                    bool bad = m.pos.size() % 9 != 0 || m.col.size() != tris ||
                               (!m.tex.empty() && (m.tex.size() != tris || m.uv.size() != tris * 6));
                    for (s32 t : m.tex) bad |= t >= static_cast<s32>(ntex) || t < -1;
                    if (bad) {
                        std::printf("BAD MESH: tris=%zu col=%zu tex=%zu uv=%zu ntex=%zu\n", tris, m.col.size(),
                                    m.tex.size(), m.uv.size(), ntex);
                        return 3;
                    }
                }
                if (!snap->meshes->empty() && rng() % 4 == 0) {
                    const CapturedMesh& m = (*snap->meshes)[rng() % snap->meshes->size()];
                    core.set_texture_target(m.vtx_addr & 0x1FFFFFFFu);
                }
            }
            if (snap->textures) {
                for (const CapturedTexture& t : *snap->textures) {
                    if (t.argb.size() != static_cast<size_t>(t.width) * t.height) {
                        std::printf("BAD TEXTURE: %ux%u argb=%zu\n", t.width, t.height, t.argb.size());
                        return 3;
                    }
                }
            }
        }

        switch (rng() % 14) {
        case 0: core.set_turbo(rng() % 2); break;
        case 1: core.set_internal_scale(1 + static_cast<int>(rng() % 4)); break;
        case 2: core.set_debug_capture(rng() % 2, rng() % 2, rng() % 2); break;
        case 3: core.pause(true); break;
        case 4: core.pause(false); break;
        case 5: core.frame_advance(); break;
        case 6: if (rng() % 8 == 0) core.reset(); break;
        case 7: core.poke(0x80000000u + (rng() % 0x400000u) * 4, {1, 2, 3, 4}); break;
        case 8: {
            std::vector<ui::MemoryFreeze> f(1);
            f[0].addr = 0x80000000u + (rng() % 0x400000u) * 4;
            f[0].bytes = {0, 0, 0, 1};
            core.set_freezes(rng() % 2 ? f : std::vector<ui::MemoryFreeze>{});
            break;
        }
        case 9: core.set_fps_limit(rng() % 3 == 0 ? 120 : 0); break;
        case 10: core.set_limit_speed(rng() % 2); break;
        case 11:
            if (rng() % 20 == 0) {
                core.stop();
                if (!core.start(rom, err)) {
                    std::fprintf(stderr, "restart failed: %s\n", err.c_str());
                    return 1;
                }
            }
            break;
        case 12: core.save_state(slot(rng())); break;
        case 13:
            if (rng() % 3 == 0) core.undo_load_state();
            else core.load_state(slot(rng()));
            break;
        }
        std::this_thread::sleep_for(milliseconds(rng() % 20));
    }
    const ui::CoreStats st = core.stats();
    core.stop();
    for (ui::StateEvent ev; core.poll_state_event(ev);) {
        if (ev.ok) (ev.op == ui::StateEvent::Op::Save ? saved : ev.op == ui::StateEvent::Op::Load ? loaded : undone)++;
    }
    std::filesystem::remove_all(state_dir);
    if (state_errors) {
        std::printf("FAIL: %d save state errors\n", state_errors);
        return 3;
    }
    std::printf("OK: %d UI steps, last frame %llu, states saved %d / loaded %d / undone %d\n", step,
                static_cast<unsigned long long>(st.frame), saved, loaded, undone);
    return 0;
}
