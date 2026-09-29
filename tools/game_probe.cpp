// Headless game prober: plays a ROM on its own as far as it can get and
// records how the game reacts to the controller, for tools/game_status.py.
//
// The emulator is deterministic, so a save state lets every decision be made
// by experiment: from the same moment the game is run once without input and
// once each with A, START and the stick held. Whatever differs between those
// runs is the game reacting to that input - no guessing from a single image.
// The prober then goes on with the input the game reacted to (A before START;
// nothing while it reacts to nothing: logos, cutscenes, loading) until the
// stick moves a large part of the picture (gameplay) or the time is up. At the
// end the same experiment is repeated with longer runs, and the last seconds
// are checked for a frozen machine.
//
//   make game_probe   -> bin/game_probe
//
//   bin/game_probe rom.z64 --out dir [--frames 3000] [--step 45] [--shot-every 4]
//
// Writes dir/probe.json and PNG screenshots (boot, a timeline, the final
// picture and the picture after the stick was held). Save files next to the
// ROM are neither read nor written.

#include "emulator.hpp"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// A picture reduced to kSigW x kSigH cells (box average), for comparing two
// pictures of any size.
constexpr int kSigW = 64, kSigH = 48;
struct Sig {
    std::array<u8, kSigW * kSigH * 3> rgb{};
    int w = 0, h = 0;
};

struct Picture {
    std::vector<u32> px;
    int w = 0, h = 0;
    Sig sig;
};

Sig make_sig(const std::vector<u32>& px, int w, int h) {
    Sig s;
    s.w = w;
    s.h = h;
    if (w <= 0 || h <= 0) return s;
    for (int cy = 0; cy < kSigH; ++cy) {
        const int y0 = cy * h / kSigH, y1 = std::max(y0 + 1, (cy + 1) * h / kSigH);
        for (int cx = 0; cx < kSigW; ++cx) {
            const int x0 = cx * w / kSigW, x1 = std::max(x0 + 1, (cx + 1) * w / kSigW);
            u32 r = 0, g = 0, b = 0, n = 0;
            for (int y = y0; y < y1; y += 2)
                for (int x = x0; x < x1; x += 2) {
                    const u32 p = px[static_cast<size_t>(y) * w + x];
                    r += (p >> 16) & 0xFF; g += (p >> 8) & 0xFF; b += p & 0xFF; ++n;
                }
            u8* o = &s.rgb[(cy * kSigW + cx) * 3];
            o[0] = static_cast<u8>(r / n); o[1] = static_cast<u8>(g / n); o[2] = static_cast<u8>(b / n);
        }
    }
    return s;
}

// Share of cells that changed between two pictures (1 when the size changed).
double sig_diff(const Sig& a, const Sig& b) {
    if (a.w != b.w || a.h != b.h) return 1.0;
    int changed = 0;
    for (int i = 0; i < kSigW * kSigH; ++i) {
        const u8* p = &a.rgb[i * 3];
        const u8* q = &b.rgb[i * 3];
        const int d = std::max({std::abs(p[0] - q[0]), std::abs(p[1] - q[1]), std::abs(p[2] - q[2])});
        if (d > 20) ++changed;
    }
    return static_cast<double>(changed) / (kSigW * kSigH);
}

// What a picture looks like, for telling a black or blank screen from content.
struct Look {
    double luma = 0;      // mean 0..255
    double lit = 0;       // share of pixels brighter than near-black
    int colors = 0;       // distinct colours (4 bits per channel) among the cells
    double edges = 0;     // share of neighbouring cells that differ clearly
};

Look look(const Picture& p) {
    Look l;
    if (p.px.empty()) return l;
    u64 sum = 0, lit = 0, n = 0;
    for (size_t i = 0; i < p.px.size(); i += 3) {
        const u32 c = p.px[i];
        const u32 y = (((c >> 16) & 0xFF) * 77 + ((c >> 8) & 0xFF) * 150 + (c & 0xFF) * 29) >> 8;
        sum += y;
        if (y > 24) ++lit;
        ++n;
    }
    l.luma = static_cast<double>(sum) / n;
    l.lit = static_cast<double>(lit) / n;
    std::vector<bool> seen(4096);
    int edges = 0, pairs = 0;
    for (int i = 0; i < kSigW * kSigH; ++i) {
        const u8* c = &p.sig.rgb[i * 3];
        const int q = ((c[0] >> 4) << 8) | ((c[1] >> 4) << 4) | (c[2] >> 4);
        if (!seen[q]) { seen[q] = true; ++l.colors; }
        if ((i % kSigW) + 1 < kSigW) {
            const u8* d = c + 3;
            ++pairs;
            if (std::abs(c[0] - d[0]) + std::abs(c[1] - d[1]) + std::abs(c[2] - d[2]) > 48) ++edges;
        }
    }
    l.edges = pairs ? static_cast<double>(edges) / pairs : 0;
    return l;
}

// Input for one frame of a run.
struct Input {
    u16 buttons = 0;
    s8 x = 0, y = 0;
};
using Script = std::function<Input(int)>;

Script idle() { return [](int) { return Input{}; }; }
Script tap(u16 button) { return [button](int f) { Input i; if (f < 6) i.buttons = button; return i; }; }
// Stick down for the first half, right for the second: moves a character or
// the camera in most games, and a menu cursor in many menus.
Script stick(int len, u16 held = 0) {
    return [len, held](int f) { Input i; i.buttons = held; if (f < len / 2) i.y = -80; else i.x = 80; return i; };
}
// Menus that need the cursor moved onto an entry before A does anything (a
// character select): the stick in one direction, then A. dir: 0 up, 1 down,
// 2 left, 3 right.
Script nav_a(int dir) {
    return [dir](int f) {
        Input i;
        if (f < 20) {
            if (dir == 0) i.y = 80;
            else if (dir == 1) i.y = -80;
            else if (dir == 2) i.x = -80;
            else i.x = 80;
        } else if (f >= 24 && f < 30) {
            i.buttons = Button::A;
        }
        return i;
    };
}
// A button held down the whole run (accelerates in racing games, which only
// steer while moving).
Script hold(u16 button) { return [button](int) { Input i; i.buttons = button; return i; }; }

// What happened during a run.
struct Window {
    int frames = 0;
    u64 gfx_tasks = 0, audio_tasks = 0, triangles = 0, tex_rects = 0, polls = 0;
    int vi_swaps = 0;         // frames whose VI origin differed from the previous one
    int moving_samples = 0;   // sampled frames that differed from the sample before
    int samples = 0;
    double max_diff_vs_ref = 0;  // largest difference from the reference run at the same frame
    double audio_rms_db = -240;
    std::vector<Sig> sigs;    // every kSampleEvery frames
    Picture last;
    std::vector<u8> end_state;
};

constexpr int kSampleEvery = 3;

struct Prober {
    std::unique_ptr<Emulator> emu = std::make_unique<Emulator>();
    std::vector<s16> audio;
    int frame = 0;  // frames of the committed timeline
    u64 emulated = 0;  // all frames run, experiments included

    Picture grab() {
        Picture p;
        emu->render_frame(p.px, p.w, p.h);
        p.sig = make_sig(p.px, p.w, p.h);
        return p;
    }

    // Runs `len` frames with `script` from the current state. With `ref` the
    // sampled pictures are compared with the reference run's.
    Window run(int len, const Script& script, const Window* ref = nullptr, bool keep_state = true) {
        Window w;
        w.frames = len;
        RSP& rsp = emu->get_rsp();
        RDP& rdp = emu->get_rdp();
        Controller& pad = emu->get_controller(0);
        const u64 g0 = rsp.get_gfx_task_count() + rsp.get_lle_task_count(), a0 = rsp.get_audio_task_count();
        const u64 t0 = rdp.get_triangle_count(), r0 = rdp.get_tex_rect_count(), p0 = pad.poll_count();
        audio.clear();
        u32 origin = emu->get_vi().get_origin();
        for (int f = 0; f < len; ++f) {
            const Input in = script(f);
            pad.reset();
            pad.set_button(Button::A, in.buttons & Button::A);
            pad.set_button(Button::START, in.buttons & Button::START);
            pad.set_stick(in.x, in.y);
            emu->step_frame();
            ++emulated;
            const u32 o = emu->get_vi().get_origin();
            if (o != origin) ++w.vi_swaps;
            origin = o;
            if (f % kSampleEvery == kSampleEvery - 1 || f == len - 1) {
                Picture p = grab();
                if (!w.sigs.empty() && sig_diff(w.sigs.back(), p.sig) > 0.002) ++w.moving_samples;
                const size_t k = w.sigs.size();
                if (ref && k < ref->sigs.size()) w.max_diff_vs_ref = std::max(w.max_diff_vs_ref, sig_diff(ref->sigs[k], p.sig));
                w.sigs.push_back(p.sig);
                ++w.samples;
                if (f == len - 1) w.last = std::move(p);
            }
        }
        pad.reset();
        w.gfx_tasks = rsp.get_gfx_task_count() + rsp.get_lle_task_count() - g0;
        w.audio_tasks = rsp.get_audio_task_count() - a0;
        w.triangles = rdp.get_triangle_count() - t0;
        w.tex_rects = rdp.get_tex_rect_count() - r0;
        w.polls = pad.poll_count() - p0;
        if (!audio.empty()) {
            double s = 0;
            for (s16 v : audio) s += static_cast<double>(v) * v;
            w.audio_rms_db = 20 * std::log10(std::sqrt(s / audio.size()) / 32768.0 + 1e-12);
        }
        if (keep_state) w.end_state = emu->save_state();
        return w;
    }

    bool load(const std::vector<u8>& s) {
        std::string err;
        if (emu->load_state(s, err)) return true;
        std::fprintf(stderr, "[probe] load_state failed: %s\n", err.c_str());
        return false;
    }
};

bool save_png(const std::string& path, const Picture& p) {
    if (p.px.empty()) return false;
    std::vector<u8> rgb(static_cast<size_t>(p.w) * p.h * 3);
    for (size_t i = 0; i < p.px.size(); ++i) {
        rgb[i * 3 + 0] = (p.px[i] >> 16) & 0xFF;
        rgb[i * 3 + 1] = (p.px[i] >> 8) & 0xFF;
        rgb[i * 3 + 2] = p.px[i] & 0xFF;
    }
    return stbi_write_png(path.c_str(), p.w, p.h, 3, rgb.data(), p.w * 3) != 0;
}

std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\', o += c;
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o + "\"";
}

std::string jnum(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.4g", v);
    return b;
}

std::string window_json(const Window& w) {
    std::ostringstream o;
    const Look l = look(w.last);
    o << "{\"frames\":" << w.frames << ",\"gfx_tasks\":" << w.gfx_tasks << ",\"audio_tasks\":" << w.audio_tasks
      << ",\"triangles\":" << w.triangles << ",\"tex_rects\":" << w.tex_rects << ",\"polls\":" << w.polls
      << ",\"vi_swaps\":" << w.vi_swaps << ",\"moving\":" << jnum(w.samples ? double(w.moving_samples) / w.samples : 0)
      << ",\"luma\":" << jnum(l.luma) << ",\"lit\":" << jnum(l.lit) << ",\"colors\":" << l.colors
      << ",\"edges\":" << jnum(l.edges) << ",\"w\":" << w.last.w << ",\"h\":" << w.last.h;
    if (w.audio_rms_db > -239) o << ",\"audio_rms_db\":" << jnum(w.audio_rms_db);
    o << "}";
    return o.str();
}

struct Options {
    std::string rom, out = ".";
    int frames = 4200;     // committed timeline after boot
    int boot = 240;        // frames without input first
    int step = 45;         // length of one experiment
    int shot_every = 4;    // timeline screenshot every that many steps
    int final_len = 120;
    std::string rsp = "hle";  // hle | lle-gfx | lle
    double max_seconds = 240;  // wall time; a slow game gets fewer steps
};

// A reaction smaller than this share of the picture is noise (a blinking
// cursor that happens to blink differently is still a reaction, though).
constexpr double kReact = 0.004;
// The stick moving this much of the picture looks like gameplay (camera or
// a character moving), rather than a menu cursor.
constexpr double kBigStick = 0.10;
constexpr int kIngameSteps = 6;
// The picture stays (nearly) the same for this many steps: something is
// waiting for a button the experiments don't find (a slow fade after START,
// "press start" after a character was chosen). START is then held over a
// longer stretch, as a transition may take a while to show.
constexpr int kStuckSteps = 3;
constexpr double kSame = 0.02;
// That many steps without anything happening end the run early.
constexpr int kDeadSteps = 12;

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--out") o.out = next();
        else if (a == "--frames") o.frames = std::stoi(next());
        else if (a == "--boot") o.boot = std::stoi(next());
        else if (a == "--step") o.step = std::stoi(next());
        else if (a == "--shot-every") o.shot_every = std::stoi(next());
        else if (a == "--rsp") o.rsp = next();
        else if (a == "--max-seconds") o.max_seconds = std::stod(next());
        else if (a[0] != '-') o.rom = a;
    }
    if (o.rom.empty()) {
        std::fprintf(stderr, "usage: game_probe rom.z64 --out dir [--frames N] [--boot N] [--step N] [--shot-every N]\n");
        return 2;
    }
    std::filesystem::create_directories(o.out);
    const auto t_start = Clock::now();

    Prober pr;
    pr.emu->get_cartridge().set_use_save_file(false);
    if (!pr.emu->load_rom(o.rom)) {
        std::fprintf(stderr, "[probe] cannot load %s\n", o.rom.c_str());
        return 3;
    }
    pr.emu->set_cpu_core(CpuCore::Recompiler);
    if (o.rsp == "lle-gfx") pr.emu->get_rsp().set_mode(RspMode::LLEGraphics);
    else if (o.rsp == "lle") pr.emu->get_rsp().set_mode(RspMode::LLE);
    pr.emu->get_ai().set_capture(&pr.audio);

    std::ostringstream timeline;
    std::vector<std::string> shots;
    auto shot = [&](const std::string& name, const Picture& p) {
        const std::string file = name + ".png";
        if (save_png((std::filesystem::path(o.out) / file).string(), p)) shots.push_back(jstr(file));
    };

    // Boot: logos, no input.
    Window boot = pr.run(o.boot, idle(), nullptr, false);
    pr.frame = o.boot;
    shot("boot", boot.last);
    const Look boot_look = look(boot.last);

    bool states_ok = true;
    int big_stick_steps = 0, steps = 0, reacted_steps = 0, dead_steps = 0;
    int first_react_frame = -1, ingame_frame = -1, first_big_frame = -1;
    int presses_a = 0, presses_start = 0, presses_nav = 0;
    bool first_entry = true;
    int stuck_steps = 0;
    Sig prev_sig, prev2_sig;  // the last two committed pictures (A may toggle between two)
    bool have_prev_sig = false, have_prev2_sig = false;
    auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t_start).count(); };
    bool out_of_time = false;
    while (pr.frame < o.boot + o.frames) {
        if (elapsed() > o.max_seconds * 0.7) { out_of_time = true; break; }
        if (stuck_steps >= kStuckSteps) {
            Window wu = pr.run(2 * o.step, tap(Button::START), nullptr);
            timeline << (first_entry ? "" : ",\n  ") << "{\"frame\":" << pr.frame << ",\"action\":\"UNSTICK\"}";
            first_entry = false;
            pr.frame += 2 * o.step;
            ++presses_start;
            shot("t" + std::to_string(pr.frame), wu.last);
            prev_sig = wu.last.sig;
            have_prev2_sig = false;
            stuck_steps = 0;
            continue;
        }
        const std::vector<u8> s0 = pr.emu->save_state();
        Window w0 = pr.run(o.step, idle(), nullptr);
        if (!pr.load(s0)) { states_ok = false; pr.load(w0.end_state); pr.frame += o.step; break; }
        Window wa = pr.run(o.step, tap(Button::A), &w0);
        const bool react_a = wa.max_diff_vs_ref > kReact;
        Window ws;
        if (!react_a) {
            pr.load(s0);
            ws = pr.run(o.step, tap(Button::START), &w0);
        }
        const bool react_s = ws.max_diff_vs_ref > kReact;
        pr.load(s0);
        Window wk = pr.run(o.step, stick(o.step), &w0);
        const bool react_k = wk.max_diff_vs_ref > kReact;
        const bool big_k = wk.max_diff_vs_ref >= kBigStick;
        // Neither A nor START nor a big stick effect: maybe a menu that wants
        // the cursor moved first. Try stick + A in each direction; it counts
        // when it changes the picture clearly more than the stick alone.
        Window wn;
        int nav_dir = -1;
        double nav_diff = 0;
        if (!react_a && !react_s && react_k && !big_k) {  // (only where the stick moves something: a cursor)
            for (int d = 0; d < 4; ++d) {
                pr.load(s0);
                Window w = pr.run(o.step, nav_a(d), &w0);
                if (w.max_diff_vs_ref > nav_diff) { nav_diff = w.max_diff_vs_ref; nav_dir = d; wn = std::move(w); }
            }
        }
        const bool react_n = nav_dir >= 0 && nav_diff > 0.02 && nav_diff > wk.max_diff_vs_ref * 1.2;

        // A comes first: it confirms the highlighted menu entry (usually
        // the one that starts a game) and skips text; in a game it is mostly
        // harmless (jump, accelerate, shoot). START only when neither A nor
        // the stick does much, so a game in progress isn't paused.
        big_stick_steps = big_k ? big_stick_steps + 1 : 0;
        const char* action = "idle";
        const Window* chosen = &w0;
        if (react_a) {
            action = "A";
            chosen = &wa;
            ++presses_a;
        } else if (big_k) {
            action = "stick";
            chosen = &wk;
        } else if (react_s) {
            action = "START";
            chosen = &ws;
            ++presses_start;
        } else if (react_n) {
            action = "NAV";
            chosen = &wn;
            ++presses_nav;
        }
        // Dead: no graphics task (and no frame buffer flips: some games, like
        // Namco Museum 64, draw with the CPU), no change on screen and no reaction.
        const bool dead = w0.gfx_tasks == 0 && w0.vi_swaps == 0 && w0.moving_samples == 0 && !react_a && !react_s && !react_k;
        dead_steps = dead ? dead_steps + 1 : 0;
        if ((react_a || react_s || react_k) && first_react_frame < 0) first_react_frame = pr.frame;
        if (react_a || react_s || react_k) ++reacted_steps;
        const Look l = look(w0.last);
        timeline << (first_entry ? "" : ",\n  ") << "{\"frame\":" << pr.frame << ",\"a\":" << jnum(wa.max_diff_vs_ref)
                 << ",\"start\":" << (react_a ? "null" : jnum(ws.max_diff_vs_ref)) << ",\"stick\":" << jnum(wk.max_diff_vs_ref) << ",\"nav\":" << jnum(nav_diff)
                 << ",\"action\":\"" << action << "\",\"luma\":" << jnum(l.luma) << ",\"colors\":" << l.colors
                 << ",\"moving\":" << jnum(w0.samples ? double(w0.moving_samples) / w0.samples : 0)
                 << ",\"gfx\":" << w0.gfx_tasks << ",\"tris\":" << w0.triangles << ",\"rects\":" << w0.tex_rects
                 << ",\"polls\":" << w0.polls << ",\"vi\":" << w0.vi_swaps << "}";
        first_entry = false;
        // The first moment the stick moves much of the picture: likely
        // gameplay, kept even if the game later falls back to a menu.
        if (big_k && first_big_frame < 0) {
            first_big_frame = pr.frame;
            shot("first_play", wk.last);
        }
        if (!pr.load(chosen->end_state)) { states_ok = false; break; }
        if ((have_prev_sig && sig_diff(prev_sig, chosen->last.sig) < kSame) ||
            (have_prev2_sig && sig_diff(prev2_sig, chosen->last.sig) < kSame)) ++stuck_steps;
        else stuck_steps = 0;
        prev2_sig = prev_sig;
        have_prev2_sig = have_prev_sig;
        prev_sig = chosen->last.sig;
        have_prev_sig = true;
        pr.frame += o.step;
        if (steps % o.shot_every == 0) shot("t" + std::to_string(pr.frame), chosen->last);
        ++steps;
        // Gameplay: the stick has moved much of the picture for a while (a
        // menu is usually left by the A presses before that).
        if (big_stick_steps >= kIngameSteps) { ingame_frame = pr.frame; break; }
        if (dead_steps >= kDeadSteps) break;
    }

    // Final experiment from where the game is now, with longer runs (shorter
    // when the game runs so slowly that they wouldn't fit in the time left).
    const double fps = pr.emulated / std::max(elapsed(), 1e-3);
    const double left = std::max(o.max_seconds - elapsed(), 10.0);
    o.final_len = std::clamp(static_cast<int>(fps * left / 5 / kSampleEvery) * kSampleEvery, 30, o.final_len);
    const std::vector<u8> sf = pr.emu->save_state();
    Window f0 = pr.run(o.final_len, idle(), nullptr, false);
    pr.load(sf);
    Window fa = pr.run(o.final_len, tap(Button::A), &f0, false);
    pr.load(sf);
    Window fs = pr.run(o.final_len, tap(Button::START), &f0, false);
    pr.load(sf);
    Window fk = pr.run(o.final_len, stick(o.final_len), &f0, false);
    // Racing games only steer while moving: the stick again, with A held,
    // against A held alone.
    pr.load(sf);
    Window fh = pr.run(o.final_len, hold(Button::A), nullptr, false);
    pr.load(sf);
    Window fhk = pr.run(o.final_len, stick(o.final_len, Button::A), &fh, false);
    shot("final", f0.last);
    shot("final_stick", fk.last);
    shot("final_start", fs.last);

    const double secs = std::chrono::duration<double>(Clock::now() - t_start).count();
    std::ostringstream j;
    j << "{\n\"rom\":" << jstr(std::filesystem::path(o.rom).filename().string())
      << ",\n\"title\":" << jstr(pr.emu->get_cartridge().get_title())
      << ",\n\"frames\":" << pr.frame << ",\"steps\":" << steps << ",\"step\":" << o.step
      << ",\"seconds\":" << jnum(secs) << ",\"emulated_frames\":" << pr.emulated << ",\"states_ok\":" << (states_ok ? "true" : "false")
      << ",\"out_of_time\":" << (out_of_time ? "true" : "false") << ",\"dead_steps\":" << dead_steps << ",\"final_len\":" << o.final_len
      << ",\n\"first_react_frame\":" << first_react_frame << ",\"ingame_frame\":" << ingame_frame << ",\"first_big_frame\":" << first_big_frame
      << ",\"reacted_steps\":" << reacted_steps << ",\"presses_a\":" << presses_a << ",\"presses_start\":" << presses_start << ",\"presses_nav\":" << presses_nav
      << ",\n\"gfx_ucode_tasks\":" << pr.emu->get_rsp().get_gfx_task_count()
      << ",\"display_lists\":" << pr.emu->get_rdp().get_display_list_count() + pr.emu->get_rsp().get_lle_task_count()
      << ",\"rsp\":" << jstr(o.rsp) << ",\"lle_tasks\":" << pr.emu->get_rsp().get_lle_task_count()
      << ",\n\"boot\":" << window_json(boot) << ",\"boot_look\":{\"luma\":" << jnum(boot_look.luma) << "}"
      << ",\n\"final\":" << window_json(f0)
      << ",\n\"final_react\":{\"a\":" << jnum(fa.max_diff_vs_ref) << ",\"start\":" << jnum(fs.max_diff_vs_ref)
      << ",\"stick\":" << jnum(fk.max_diff_vs_ref) << ",\"stick_with_a\":" << jnum(fhk.max_diff_vs_ref) << "}"
      << ",\n\"final_stick\":" << window_json(fk)
      << ",\n\"shots\":[" ;
    for (size_t i = 0; i < shots.size(); ++i) j << (i ? "," : "") << shots[i];
    j << "],\n\"timeline\":[\n  " << timeline.str() << "]\n}\n";
    std::ofstream(std::filesystem::path(o.out) / "probe.json") << j.str();
    std::printf("[probe] %s: %d frames, %d steps, ingame_frame=%d, %.1f s\n", o.rom.c_str(), pr.frame, steps, ingame_frame, secs);
    return 0;
}
