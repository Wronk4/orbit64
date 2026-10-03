// Game QA: plays a recorded route (tools/game_probe.cpp --route) to where a game
// is playable, plays it for a while, and judges what is drawn - the same
// machine in HLE and (with --lle) on the real microcode, which shows where the
// high-level emulation draws something else. Driven by tools/qa.py.
//
//   make game_qa  ->  bin/game_qa
//   bin/game_qa rom.z64 --route r.route --out dir [--checkpoint c.state]
//               [--frames 300] [--sample 30] [--lle] [--lle-frames 120]
//
// What it does, in order:
//   1. Replays the route's inputs from power-on (or, with --checkpoint, loads
//      the state saved when the route was recorded: quicker, but it skips
//      everything the boot does, so only for quick looks).
//   2. Checks that the game is still playable there: the stick has to move a
//      large part of the picture. If not, the route is stale (a fix changed
//      the timing) and the tool says so: "route_ok": false.
//   3. Plays on with the route's steering for --frames frames, recording
//      pictures every --sample frames and, over the whole stretch: graphics
//      tasks, triangles and how many were culled or failed a test, picture
//      changes, audio level, where the CPU is, the speed.
//   4. With --lle: the machine of step 2 again, but with the RSP running the
//      game's own microcode, and the same steering; each picture is compared
//      with the HLE one (a difference picture is written next to them).
//
// Writes dir/qa.json, dir/hle_NN.png, dir/lle_NN.png, dir/diff_NN.png.
// Save files next to the ROM are neither read nor written.

#include "probe_route.hpp"
#include <iostream>
#include <map>

namespace {

struct Options {
    std::string rom, route, out = ".", checkpoint;
    int frames = 300;       // gameplay frames to judge
    int sample = 30;        // a picture every that many frames
    bool lle = false;
    bool mash = false;      // tap the other buttons now and then while playing
    bool update_sigs = false;  // rewrite the route's picture signatures from this replay
    int lle_frames = 120;   // the low-level run is much slower: fewer frames
    double max_seconds = 120;
};

// Share of 64x48 cells that differ clearly; unlike probe's sig_diff it also
// compares pictures of different size (the cells are relative).
double cells_diff(const Sig& a, const Sig& b) {
    int changed = 0;
    for (int i = 0; i < kSigW * kSigH; ++i) {
        const u8* p = &a.rgb[i * 3];
        const u8* q = &b.rgb[i * 3];
        if (std::max({std::abs(p[0] - q[0]), std::abs(p[1] - q[1]), std::abs(p[2] - q[2])}) > 24) ++changed;
    }
    return static_cast<double>(changed) / (kSigW * kSigH);
}

// The picture reduced to tw x th by averaging the source pixels each target
// pixel covers (the machine draws at 1x or 2x depending on the RDP in use).
Picture resize_box(const Picture& p, int tw, int th) {
    if (p.w == tw && p.h == th) return p;
    Picture r;
    r.w = tw;
    r.h = th;
    r.px.resize(static_cast<size_t>(tw) * th);
    for (int y = 0; y < th; ++y) {
        const int y0 = y * p.h / th, y1 = std::max(y0 + 1, (y + 1) * p.h / th);
        for (int x = 0; x < tw; ++x) {
            const int x0 = x * p.w / tw, x1 = std::max(x0 + 1, (x + 1) * p.w / tw);
            u32 rr = 0, gg = 0, bb = 0, n = 0;
            for (int yy = y0; yy < y1 && yy < p.h; ++yy)
                for (int xx = x0; xx < x1 && xx < p.w; ++xx) {
                    const u32 c = p.px[static_cast<size_t>(yy) * p.w + xx];
                    rr += (c >> 16) & 0xFF; gg += (c >> 8) & 0xFF; bb += c & 0xFF; ++n;
                }
            if (!n) n = 1;
            r.px[static_cast<size_t>(y) * tw + x] = 0xFF000000u | ((rr / n) << 16) | ((gg / n) << 8) | (bb / n);
        }
    }
    r.sig = p.sig;
    return r;
}

// High-frequency energy: mean absolute difference between horizontal neighbours,
// 0..1. Snow where there should be a texture (a palette that wasn't applied, a
// texture decoded wrongly) makes it jump.
double noise_level(const Picture& full) {
    if (full.w < 2 || full.h < 1) return 0;
    const Picture p = full.w > 320 ? resize_box(full, 320, 240) : full;  // comparable between 1x and 2x
    u64 sum = 0, n = 0;
    for (int y = 0; y < p.h; y += 2) {
        const u32* row = &p.px[static_cast<size_t>(y) * p.w];
        for (int x = 0; x + 1 < p.w; ++x) {
            const u32 a = row[x], b = row[x + 1];
            sum += std::abs(static_cast<int>((a >> 16) & 0xFF) - static_cast<int>((b >> 16) & 0xFF)) +
                   std::abs(static_cast<int>((a >> 8) & 0xFF) - static_cast<int>((b >> 8) & 0xFF)) +
                   std::abs(static_cast<int>(a & 0xFF) - static_cast<int>(b & 0xFF));
            n += 3;
        }
    }
    return n ? static_cast<double>(sum) / n / 255.0 : 0;
}

// The share of 10x10-pixel cells (a 32x24 grid) whose colour differs clearly. Both machines
// draw the same scene but not the same pixels - HLE uses its own, faster renderer, LLE the
// bit-exact one - so edges, anti-aliasing and one-pixel offsets always differ; averaged over
// a cell they do not, while a missing object or a wrong texture still does.
double coarse_diff(const Picture& a0, const Picture& b0) {
    if (a0.px.empty() || b0.px.empty()) return 0;
    const Picture a = resize_box(a0, 32, 24), b = resize_box(b0, 32, 24);
    int changed = 0;
    for (size_t i = 0; i < a.px.size(); ++i) {
        const u32 p = a.px[i], q = b.px[i];
        const int d = std::max({std::abs(static_cast<int>((p >> 16) & 0xFF) - static_cast<int>((q >> 16) & 0xFF)),
                                std::abs(static_cast<int>((p >> 8) & 0xFF) - static_cast<int>((q >> 8) & 0xFF)),
                                std::abs(static_cast<int>(p & 0xFF) - static_cast<int>(q & 0xFF))});
        if (d > 32) ++changed;
    }
    return static_cast<double>(changed) / a.px.size();
}

struct PixDiff {
    double mean = 0;      // mean absolute difference per channel, 0..1
    double blocks = 0;    // share of 8x8 blocks that differ clearly
    Picture heat;         // the difference, amplified (grey)
};

// Pixel comparison on the smaller of the two sizes.
PixDiff pix_diff(const Picture& a0, const Picture& b0) {
    PixDiff d;
    if (a0.px.empty() || b0.px.empty()) return d;
    const int tw = std::min(a0.w, b0.w), th = std::min(a0.h, b0.h);
    const Picture a = resize_box(a0, tw, th), b = resize_box(b0, tw, th);
    d.heat.w = a.w;
    d.heat.h = a.h;
    d.heat.px.resize(a.px.size());
    u64 sum = 0;
    const int bw = (a.w + 7) / 8, bh = (a.h + 7) / 8;
    std::vector<u32> block(static_cast<size_t>(bw) * bh, 0);
    for (int y = 0; y < a.h; ++y)
        for (int x = 0; x < a.w; ++x) {
            const size_t i = static_cast<size_t>(y) * a.w + x;
            const u32 p = a.px[i], q = b.px[i];
            const int dr = std::abs(static_cast<int>((p >> 16) & 0xFF) - static_cast<int>((q >> 16) & 0xFF));
            const int dg = std::abs(static_cast<int>((p >> 8) & 0xFF) - static_cast<int>((q >> 8) & 0xFF));
            const int db = std::abs(static_cast<int>(p & 0xFF) - static_cast<int>(q & 0xFF));
            sum += dr + dg + db;
            const int m = std::max({dr, dg, db});
            block[static_cast<size_t>(y / 8) * bw + x / 8] += static_cast<u32>(m);
            const u32 v = static_cast<u32>(std::min(255, m * 4));
            d.heat.px[i] = 0xFF000000u | (v << 16) | (v << 8) | v;
        }
    d.mean = static_cast<double>(sum) / (static_cast<double>(a.px.size()) * 3 * 255.0);
    int changed = 0;
    for (u32 v : block)
        if (v / 64 > 24) ++changed;
    d.blocks = static_cast<double>(changed) / block.size();
    return d;
}

u64 fnv(const u8* p, size_t n) {
    u64 h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

std::string hex64(u64 v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}

// What one stretch of play looked like.
struct Play {
    int frames = 0;
    double seconds = 0;
    u64 gfx_tasks = 0, audio_tasks = 0, lle_tasks = 0;
    u32 triangles = 0, rasterized = 0, cull_back = 0, cull_front = 0, drawn = 0, z_fail = 0, a_fail = 0;
    int vi_swaps = 0, moving = 0, samples = 0;
    double audio_rms_db = -240;
    std::vector<Picture> pics;
    std::vector<u64> pcs;       // the CPU's pc at each sample
    u64 pc = 0, epc = 0;
    u32 cause = 0, status = 0;
};

// The steering plus a tap of one of the other buttons now and then (jump, attack,
// camera, a menu entry): more of the game's drawing gets used. START is never
// pressed - it would pause the game. Deterministic like everything else here.
Script mash_script(const std::string& play, int len) {
    const Script base = play_script(play, len);
    std::vector<u16> pool = {Button::B, Button::Z, Button::C_DOWN, Button::C_RIGHT, Button::R, Button::C_UP, Button::L, Button::C_LEFT};
    if (play.rfind("stick_", 0) != 0) pool.insert(pool.begin(), Button::A);  // not where a held button is the steering
    return [base, pool](int f) {
        Input in = base(f);
        const int slot = f / 24;
        if (slot % 2 == 1 && f % 24 < 6) in.buttons |= pool[(slot / 2) % pool.size()];
        return in;
    };
}

Play play(Prober& pr, const Route& route, int frames, int sample, double max_seconds, int stick_len, bool mash) {
    Play p;
    RSP& rsp = pr.emu->get_rsp();
    RDP& rdp = pr.emu->get_rdp();
    Controller& pad = pr.emu->get_controller(0);
    CPU& cpu = pr.emu->get_cpu();
    const u64 g0 = rsp.get_gfx_task_count() + rsp.get_lle_task_count(), a0 = rsp.get_audio_task_count(), l0 = rsp.get_lle_task_count();
    const RDP::RenderCounters c0 = RDP::render_counters();
    const Script script = mash ? mash_script(route.play, stick_len) : play_script(route.play, stick_len);
    pr.audio.clear();
    u32 origin = pr.emu->get_vi().get_origin();
    const auto t0 = Clock::now();
    Sig prev;
    bool have_prev = false;
    for (int f = 0; f < frames; ++f) {
        apply_input(pad, script(f));
        pr.emu->step_frame();
        ++p.frames;
        const u32 o = pr.emu->get_vi().get_origin();
        if (o != origin) ++p.vi_swaps;
        origin = o;
        if ((f + 1) % sample == 0 || f == frames - 1) {
            Picture pic = pr.grab();
            if (have_prev && cells_diff(prev, pic.sig) > 0.004) ++p.moving;
            prev = pic.sig;
            have_prev = true;
            ++p.samples;
            p.pcs.push_back(cpu.get_pc());
            p.pics.push_back(std::move(pic));
        }
        if (std::chrono::duration<double>(Clock::now() - t0).count() > max_seconds) break;
    }
    pad.reset();
    p.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    p.gfx_tasks = rsp.get_gfx_task_count() + rsp.get_lle_task_count() - g0;
    p.audio_tasks = rsp.get_audio_task_count() - a0;
    p.lle_tasks = rsp.get_lle_task_count() - l0;
    const RDP::RenderCounters c1 = RDP::render_counters();
    p.triangles = c1.triangles - c0.triangles;
    p.rasterized = c1.rasterized - c0.rasterized;
    p.cull_back = c1.cull_back - c0.cull_back;
    p.cull_front = c1.cull_front - c0.cull_front;
    p.drawn = c1.pixels_drawn - c0.pixels_drawn;
    p.z_fail = c1.z_fail - c0.z_fail;
    p.a_fail = c1.a_fail - c0.a_fail;
    if (!pr.audio.empty()) {
        double s = 0;
        for (s16 v : pr.audio) s += static_cast<double>(v) * v;
        p.audio_rms_db = 20 * std::log10(std::sqrt(s / pr.audio.size()) / 32768.0 + 1e-12);
    }
    p.pc = cpu.get_pc();
    p.epc = cpu.get_cp0(14);
    p.cause = static_cast<u32>(cpu.get_cp0(13));
    p.status = static_cast<u32>(cpu.get_cp0(12));
    return p;
}

// The CPU sits at the exception vectors or at address 0 at several samples in a
// row: the game jumped into nothing (a crash, however the game's own handler
// reports it).
bool cpu_lost(const Play& p) {
    int run = 0, best = 0;
    for (u64 pc : p.pcs) {
        const u32 lo = static_cast<u32>(pc);
        const bool lost = lo < 0x400 || (lo >= 0x80000000u && lo < 0x80000400u) || (lo >= 0xA0000000u && lo < 0xA0000400u);
        run = lost ? run + 1 : 0;
        best = std::max(best, run);
    }
    return best >= 3;
}

std::string play_json(const Play& p) {
    std::ostringstream o;
    o << "{\"frames\":" << p.frames << ",\"seconds\":" << jnum(p.seconds) << ",\"fps\":" << jnum(p.seconds > 0 ? p.frames / p.seconds : 0)
      << ",\"lle_tasks\":" << p.lle_tasks << ",\"gfx_tasks\":" << p.gfx_tasks << ",\"audio_tasks\":" << p.audio_tasks << ",\"triangles\":" << p.triangles
      << ",\"rasterized\":" << p.rasterized << ",\"cull_back\":" << p.cull_back << ",\"cull_front\":" << p.cull_front
      << ",\"pixels_drawn\":" << p.drawn << ",\"z_fail\":" << p.z_fail << ",\"a_fail\":" << p.a_fail
      << ",\"vi_swaps\":" << p.vi_swaps << ",\"moving\":" << jnum(p.samples > 1 ? double(p.moving) / (p.samples - 1) : 0)
      << ",\"cpu_lost\":" << (cpu_lost(p) ? "true" : "false");
    char b[160];
    std::snprintf(b, sizeof b, ",\"pc\":\"%llx\",\"epc\":\"%llx\",\"cause\":\"%x\",\"status\":\"%x\"",
                  static_cast<unsigned long long>(p.pc), static_cast<unsigned long long>(p.epc), p.cause, p.status);
    o << b;
    if (p.audio_rms_db > -239) o << ",\"audio_rms_db\":" << jnum(p.audio_rms_db);
    o << ",\"pictures\":[";
    for (size_t i = 0; i < p.pics.size(); ++i) {
        const Look l = look(p.pics[i]);
        o << (i ? "," : "") << "{\"luma\":" << jnum(l.luma) << ",\"lit\":" << jnum(l.lit) << ",\"colors\":" << l.colors
          << ",\"edges\":" << jnum(l.edges) << ",\"noise\":" << jnum(noise_level(p.pics[i])) << ",\"w\":" << p.pics[i].w
          << ",\"h\":" << p.pics[i].h << "}";
    }
    o << "]}";
    return o.str();
}

// What the core writes to stdout is mostly one line per display list; nothing
// here needs it, except the few lines that say something is wrong. They are
// counted (distinct lines, the first few kept) and go into qa.json.
class LogCatcher : public std::streambuf {
public:
    std::map<std::string, int> lines;
    int total = 0;

protected:
    int overflow(int c) override {
        if (c == EOF) return c;
        if (c == '\n') {
            ++total;
            if (cur_.rfind("[UCODE-DETECT-FAIL]", 0) == 0 || cur_.find("nknown") != std::string::npos || cur_.find("nimplemented") != std::string::npos ||
                cur_.find("rror") != std::string::npos || cur_.find("WARN") != std::string::npos) {
                if (cur_.size() > 160) cur_.resize(160);
                if (lines.size() < 20 || lines.count(cur_)) ++lines[cur_];
            }
            cur_.clear();
        } else if (cur_.size() < 400) {
            cur_ += static_cast<char>(c);
        }
        return c;
    }

private:
    std::string cur_;
};

std::string log_json(const LogCatcher& l) {
    std::ostringstream o;
    o << "[";
    bool first = true;
    for (const auto& kv : l.lines) {
        o << (first ? "" : ",") << "{\"line\":" << jstr(kv.first) << ",\"count\":" << kv.second << "}";
        first = false;
    }
    o << "]";
    return o.str();
}

const char* ucode_name(MicrocodeType t) {
    switch (t) {
        case MicrocodeType::Auto: return "none";
        case MicrocodeType::Fast3D: return "Fast3D";
        case MicrocodeType::F3DEX: return "F3DEX";
        case MicrocodeType::F3DEX2: return "F3DEX2";
        case MicrocodeType::S2DEX: return "S2DEX";
        case MicrocodeType::S2DEX2: return "S2DEX2";
        case MicrocodeType::F3DGOLDEN: return "F3DGOLDEN";
        case MicrocodeType::F3DPD: return "F3DPD";
        case MicrocodeType::F3DDKR: return "F3DDKR";
        case MicrocodeType::F3DJFG: return "F3DJFG";
        case MicrocodeType::F3DWRUS: return "F3DWRUS";
    }
    return "?";
}

void setup(Prober& pr, const std::string& rom, const std::string& rsp) {
    pr.emu->get_cartridge().set_use_save_file(false);
    if (!pr.emu->load_rom(rom)) {
        std::fprintf(stderr, "[qa] cannot load %s\n", rom.c_str());
        std::exit(3);
    }
    pr.emu->set_cpu_core(CpuCore::Recompiler);
    if (rsp == "lle-gfx") pr.emu->get_rsp().set_mode(RspMode::LLEGraphics);
    else if (rsp == "lle") pr.emu->get_rsp().set_mode(RspMode::LLE);
    pr.emu->get_ai().set_capture(&pr.audio);
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--route") o.route = next();
        else if (a == "--out") o.out = next();
        else if (a == "--checkpoint") o.checkpoint = next();
        else if (a == "--frames") o.frames = std::stoi(next());
        else if (a == "--sample") o.sample = std::max(1, std::stoi(next()));
        else if (a == "--lle") o.lle = true;
        else if (a == "--mash") o.mash = true;
        else if (a == "--update-sigs") o.update_sigs = true;
        else if (a == "--lle-frames") o.lle_frames = std::stoi(next());
        else if (a == "--max-seconds") o.max_seconds = std::stod(next());
        else if (a[0] != '-') o.rom = a;
    }
    if (o.rom.empty() || o.route.empty()) {
        std::fprintf(stderr, "usage: game_qa rom.z64 --route r.route --out dir [--checkpoint c.state] [--frames N] [--sample N] [--lle]\n");
        return 2;
    }
    // The core logs a lot to stdout (one line per display list); nothing here needs it.
    LogCatcher log;
    std::cout.rdbuf(&log);
    std::filesystem::create_directories(o.out);
    const auto t_start = Clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t_start).count(); };

    Route route;
    if (!read_route(o.route, route)) {
        std::fprintf(stderr, "[qa] cannot read route %s\n", o.route.c_str());
        return 4;
    }
    Prober pr;
    setup(pr, o.rom, "hle");
    const std::string crc = crc_text(pr.emu->get_cartridge());
    if (!route.crc.empty() && route.crc != crc) {
        std::fprintf(stderr, "[qa] the route is for another ROM (%s, this is %s)\n", route.crc.c_str(), crc.c_str());
        return 5;
    }

    // 1. Where the route ends: a checkpoint, or the whole route played from power-on.
    bool from_checkpoint = false;
    if (!o.checkpoint.empty()) {
        std::vector<u8> st;
        savestate::FileInfo info;
        std::string err;
        if (savestate::read_file(o.checkpoint, &st, info, err) && info.state_version == savestate::kStateVersion && pr.load(st)) {
            from_checkpoint = true;
        } else {
            std::fprintf(stderr, "[qa] checkpoint not usable (%s): playing the route instead\n", err.c_str());
        }
    }
    const auto t_replay = Clock::now();
    // The replay looks at the screen the way the probe did while it found the route (every third
    // frame, which also clears the HLE depth buffer) - so it stays the same run - and compares it with
    // the picture recorded for every committed step. Where they part is where the game has drifted.
    struct Drift {
        int checked = 0, first = -1, last_ok = 0;
        double max = 0;
    } drift;
    std::vector<RouteSig> now_sigs;
    // A route without pictures (rebuilt from an old run) gets them when --update-sigs asks: every 45 frames.
    std::vector<RouteSig> want = route.sigs;
    if (want.empty() && o.update_sigs) {
        for (int n = 240; n <= static_cast<int>(route.inputs.size()); n += 45) want.push_back({n, {}});
        if (want.empty() || want.back().frame != static_cast<int>(route.inputs.size()))
            want.push_back({static_cast<int>(route.inputs.size()), {}});
    }
    const bool can_compare = !route.sigs.empty();
    if (!from_checkpoint) {
        Controller& pad = pr.emu->get_controller(0);
        size_t si = 0;
        for (size_t f = 0; f < route.inputs.size(); ++f) {
            apply_input(pad, route.inputs[f]);
            pr.emu->step_frame();
            const int n = static_cast<int>(f) + 1;
            while (si < want.size() && want[si].frame < n) ++si;
            const bool sig_now = si < want.size() && want[si].frame == n;
            if (n % 3 == 0 || sig_now) {
                const Picture pic = pr.grab();
                if (sig_now) {
                    const RouteSigCells cur = route_sig(pic);
                    now_sigs.push_back({n, cur});
                    if (can_compare) {
                        const double d = route_sig_diff(cur, want[si].cells);
                        ++drift.checked;
                        drift.max = std::max(drift.max, d);
                        if (d > 0.10 && drift.first < 0) drift.first = n;
                        if (drift.first < 0) drift.last_ok = n;
                    }
                    ++si;
                }
            }
        }
        pad.reset();
    }
    const double replay_seconds = std::chrono::duration<double>(Clock::now() - t_replay).count();
    std::vector<u8> route_end = pr.emu->save_state();
    u64 end_hash = fnv(pr.emu->get_bus().get_rdram(), pr.emu->get_bus().get_rdram_size());
    Picture end_pic = pr.grab();

    // 2. Still playable? (The same experiment as the probe's: the stick against nothing.)
    const int check_len = 45;
    auto playable = [&](Prober& p, const std::vector<u8>& st) {
        Window idle_run = p.run(check_len, idle(), nullptr, false);
        p.load(st);
        Window stick_run = p.run(check_len, play_script(route.play, check_len), &idle_run, false);
        p.load(st);
        return stick_run.max_diff_vs_ref;
    };
    double check_diff = playable(pr, route_end);
    const bool wants_play = route.reached == "ingame" || route.reached == "big";
    bool route_ok = !wants_play || check_diff >= 0.05;
    // A route that was cut after an action that spoiled it (an A press that opened a menu): the
    // earlier steps' ends are tried, up to three, and the first one where the stick works is the end.
    int trimmed_to = 0;
    if (!route_ok && !from_checkpoint && route.sigs.size() >= 3) {
        std::vector<int> cands;
        for (auto it = route.sigs.rbegin(); it != route.sigs.rend() && cands.size() < 3; ++it)
            if (it->frame < static_cast<int>(route.inputs.size()) && it->frame >= 300 && (cands.empty() || cands.back() != it->frame))
                cands.push_back(it->frame);
        for (int cand : cands) {
            Prober p2;
            setup(p2, o.rom, "hle");
            Controller& pad2 = p2.emu->get_controller(0);
            for (int f = 0; f < cand; ++f) {
                apply_input(pad2, route.inputs[static_cast<size_t>(f)]);
                p2.emu->step_frame();
                if ((f + 1) % 3 == 0) (void)p2.grab();
            }
            pad2.reset();
            const std::vector<u8> st = p2.emu->save_state();
            const double d2 = playable(p2, st);
            if (d2 >= 0.05) {
                pr = std::move(p2);
                pr.emu->get_ai().set_capture(&pr.audio);  // it pointed at p2's buffer
                route_end = st;
                check_diff = d2;
                route_ok = true;
                trimmed_to = cand;
                end_hash = fnv(pr.emu->get_bus().get_rdram(), pr.emu->get_bus().get_rdram_size());
                end_pic = pr.grab();
                break;
            }
        }
    }

    // 3. Play on in HLE.
    Play hle = play(pr, route, o.frames, o.sample, o.max_seconds, o.frames, o.mash);
    for (size_t i = 0; i < hle.pics.size(); ++i) {
        char name[32];
        std::snprintf(name, sizeof name, "hle_%02zu.png", i);
        save_png((std::filesystem::path(o.out) / name).string(), hle.pics[i]);
    }
    save_png((std::filesystem::path(o.out) / "route_end.png").string(), end_pic);

    // 4. The same on the real microcode, from the same machine.
    std::string lle_json = "null", cmp_json = "[]";
    if (o.lle) {
        Prober lp;
        setup(lp, o.rom, "lle-gfx");
        std::string err;
        if (lp.emu->load_state(route_end, err)) {
            const int lle_frames = std::min(o.lle_frames, o.frames);
            Play ll = play(lp, route, lle_frames, o.sample, std::max(20.0, o.max_seconds - elapsed()), o.frames, o.mash);
            std::ostringstream cmp;
            cmp << "[";
            for (size_t i = 0; i < ll.pics.size() && i < hle.pics.size(); ++i) {
                char name[32];
                std::snprintf(name, sizeof name, "lle_%02zu.png", i);
                save_png((std::filesystem::path(o.out) / name).string(), ll.pics[i]);
                const PixDiff d = pix_diff(hle.pics[i], ll.pics[i]);
                if (!d.heat.px.empty()) {
                    std::snprintf(name, sizeof name, "diff_%02zu.png", i);
                    save_png((std::filesystem::path(o.out) / name).string(), d.heat);
                }
                cmp << (i ? "," : "") << "{\"coarse\":" << jnum(coarse_diff(hle.pics[i], ll.pics[i])) << ",\"cells\":" << jnum(cells_diff(hle.pics[i].sig, ll.pics[i].sig)) << ",\"pixels\":" << jnum(d.mean)
                    << ",\"blocks\":" << jnum(d.blocks) << "}";
            }
            cmp << "]";
            cmp_json = cmp.str();
            lle_json = play_json(ll);
        } else {
            std::fprintf(stderr, "[qa] the low-level run cannot start from that state: %s\n", err.c_str());
            lle_json = "{\"error\":" + jstr(err) + "}";
        }
    }

    std::ostringstream j;
    j << "{\n\"rom\":" << jstr(std::filesystem::path(o.rom).filename().string()) << ",\"crc\":" << jstr(crc)
      << ",\"title\":" << jstr(pr.emu->get_cartridge().get_title()) << ",\n\"route\":{\"reached\":" << jstr(route.reached)
      << ",\"play\":" << jstr(route.play) << ",\"frames\":" << route.inputs.size() << ",\"from_checkpoint\":"
      << (from_checkpoint ? "true" : "false") << ",\"replay_seconds\":" << jnum(replay_seconds) << ",\"end_hash\":" << jstr(hex64(end_hash))
      << ",\"playable_diff\":" << jnum(check_diff) << ",\"route_ok\":" << (route_ok ? "true" : "false")
      << ",\"trimmed_to\":" << trimmed_to << ",\"drift\":{\"checked\":" << drift.checked << ",\"max\":" << jnum(drift.max) << ",\"first\":" << drift.first
      << ",\"last_ok\":" << drift.last_ok << "}}"
      << ",\n\"ucode\":" << jstr(ucode_name(pr.emu->get_rdp().get_ucode_type()))
      << ",\n\"hle\":" << play_json(hle) << ",\n\"lle\":" << lle_json << ",\n\"compare\":" << cmp_json
      << ",\n\"log\":" << log_json(log)
      << ",\n\"seconds\":" << jnum(elapsed()) << "\n}\n";
    std::ofstream(std::filesystem::path(o.out) / "qa.json") << j.str();
    // --update-sigs: the pictures of this replay become the recorded ones ("this is what it looks like now").
    if (o.update_sigs && route_ok && !from_checkpoint && !now_sigs.empty()) {
        Route r2 = route;
        r2.sigs = now_sigs;
        if (!write_route(o.route, r2)) std::fprintf(stderr, "[qa] cannot write %s\n", o.route.c_str());
    }
    std::fprintf(stderr, "[qa] %s: route %s (%s), %d samples, %.1f s\n", o.rom.c_str(), route_ok ? "ok" : "STALE",
                 route.reached.c_str(), hle.samples, elapsed());
    return 0;
}
