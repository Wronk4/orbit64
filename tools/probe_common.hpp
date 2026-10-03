// Shared by tools/game_probe.cpp and tools/game_qa.cpp: pictures and how they are compared,
// scripted controller input, and the Prober that runs a game frame by frame headless.
#pragma once

#include "emulator.hpp"
#include "savestate.hpp"
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

constexpr u16 kAllButtons[] = {Button::A, Button::B, Button::Z, Button::START, Button::D_UP, Button::D_DOWN, Button::D_LEFT,
                               Button::D_RIGHT, Button::L, Button::R, Button::C_UP, Button::C_DOWN, Button::C_LEFT, Button::C_RIGHT};

void apply_input(Controller& pad, const Input& in) {
    pad.reset();
    for (u16 b : kAllButtons) pad.set_button(b, (in.buttons & b) != 0);
    pad.set_stick(in.x, in.y);
}

// A script that plays back recorded input (a route); past its end: nothing pressed.
Script from_inputs(const std::vector<Input>& v, size_t from = 0) {
    return [&v, from](int f) { return from + f < v.size() ? v[from + f] : Input{}; };
}

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
    std::vector<Input> inputs;  // what was fed to the controller, frame by frame
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
            apply_input(pad, in);
            w.inputs.push_back(in);
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

}  // namespace
