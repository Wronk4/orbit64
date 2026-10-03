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

#include "probe_route.hpp"
#include <map>
#include <thread>

namespace {

// Advice: what is on the screen and which buttons to try, from tools/qa.py (OCR + rules, tools/screen_reader.py).
// The prober writes the picture as <dir>/req_<n>.png and waits for <dir>/resp_<n>.txt:
//     kind <screen kind>
//     conf <0..1>
//     cand <score><TAB><name><TAB><segments>      one line per candidate, best first; segments are
//                                                 buttons-hex,stick-x,stick-y,frames separated by ';'
// The candidates are only hypotheses: each is run from the same state and kept only if it changes the picture.
struct AdviceCand {
    std::string name;
    double score = 0;
    std::vector<Input> inputs;
};
struct AdviceReply {
    std::string kind;
    double conf = 0;
    std::vector<AdviceCand> cands;
};

AdviceReply ask_advice(const std::string& dir, int n, const Picture& pic, double timeout_s, double* waited) {
    namespace fs = std::filesystem;
    AdviceReply rep;
    const auto t0 = Clock::now();
    const fs::path tmp = fs::path(dir) / ("req_" + std::to_string(n) + ".tmp.png");
    const fs::path png = fs::path(dir) / ("req_" + std::to_string(n) + ".png");
    const fs::path resp = fs::path(dir) / ("resp_" + std::to_string(n) + ".txt");
    if (!save_png(tmp.string(), pic)) return rep;
    std::error_code ec;
    fs::rename(tmp, png, ec);  // whole file or nothing: the other side polls for it
    while (!fs::exists(resp, ec)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (std::chrono::duration<double>(Clock::now() - t0).count() > timeout_s) break;
    }
    if (waited) *waited += std::chrono::duration<double>(Clock::now() - t0).count();
    std::ifstream f(resp);
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("kind ", 0) == 0) rep.kind = line.substr(5);
        else if (line.rfind("conf ", 0) == 0) rep.conf = std::atof(line.c_str() + 5);
        else if (line.rfind("cand ", 0) == 0) {
            const size_t t1 = line.find('\t', 5), t2 = t1 == std::string::npos ? t1 : line.find('\t', t1 + 1);
            if (t2 == std::string::npos) continue;
            AdviceCand c;
            c.score = std::atof(line.substr(5, t1 - 5).c_str());
            c.name = line.substr(t1 + 1, t2 - t1 - 1);
            std::istringstream segs(line.substr(t2 + 1));
            std::string seg;
            while (std::getline(segs, seg, ';')) {
                unsigned b = 0;
                int x = 0, y = 0, fr = 0;
                if (std::sscanf(seg.c_str(), "%x,%d,%d,%d", &b, &x, &y, &fr) == 4)
                    for (int i = 0; i < fr && i < 400; ++i) c.inputs.push_back(Input{static_cast<u16>(b), static_cast<s8>(x), static_cast<s8>(y)});
            }
            if (!c.inputs.empty()) rep.cands.push_back(std::move(c));
        }
    }
    return rep;
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
    std::string route;         // write the inputs that reached the game here (see probe_route.hpp)
    std::string checkpoint;    // and a save state of the machine where the route ends
    std::string frontier;      // the whole route to where the search got, whatever class it reached (to go on from)
    std::string continue_route;  // start from this route's end instead of power-on (more time for a slow game)
    int mash_len = 480;        // frames of one mash attempt (0: none)
    int strategy = 0;          // 0..3: which action the search prefers when several change the picture
    std::string advice;        // directory for the advice handshake (empty: none)
    int advice_every = 8;      // ask every that many steps (and whenever the picture is stuck)
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
// A picture is "new" when it differs from every screen seen so far in this game by at least this share of the
// cells (the same scale as kReact: a cursor that moved to another entry already counts).
constexpr double kNovel = 0.006;
// That many steps in a row that lead to screens already seen: the search is going round in a circle.
constexpr int kStagnantSteps = 30;
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
        else if (a == "--route") o.route = next();
        else if (a == "--checkpoint") o.checkpoint = next();
        else if (a == "--frontier") o.frontier = next();
        else if (a == "--strategy") o.strategy = std::stoi(next()) & 3;
        else if (a == "--continue-route") o.continue_route = next();
        else if (a == "--mash") o.mash_len = std::stoi(next());
        else if (a == "--advice") o.advice = next();
        else if (a == "--advice-every") o.advice_every = std::max(1, std::stoi(next()));
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

    // Boot: logos, no input. (Or, with --continue-route, the route so far: the machine
    // ends up where that run stopped, and the search goes on from there.)
    Route cont;
    if (!o.continue_route.empty()) {
        if (!read_route(o.continue_route, cont)) {
            std::fprintf(stderr, "[probe] cannot read route %s\n", o.continue_route.c_str());
            return 4;
        }
        o.boot = static_cast<int>(cont.inputs.size());
    }
    Window boot = o.continue_route.empty() ? pr.run(o.boot, idle(), nullptr, false)
                                           : pr.run(o.boot, from_inputs(cont.inputs), nullptr, false);
    pr.frame = o.boot;
    // The route: every input of the committed timeline, from power-on. The
    // experiments that were tried and dropped aren't part of it.
    std::vector<Input> route_inputs = boot.inputs;
    // What the picture looked like at the end of every committed step (see probe_route.hpp).
    std::vector<RouteSig> route_sigs = cont.sigs;
    route_sigs.push_back({static_cast<int>(route_inputs.size()), route_sig(boot.last)});
    auto sig_here = [&](const Picture& p) { route_sigs.push_back({static_cast<int>(route_inputs.size()), route_sig(p)}); };
    // Which screens the game has shown so far (the route replayed above, and every step after it). A step that
    // leads to a screen already seen goes nowhere; the search prefers steps that lead somewhere new, and the
    // frontier it hands to the next search ends at the last one that did - never in a circle.
    std::vector<Sig> seen;
    for (size_t i = 0; i < boot.sigs.size(); i += 15) seen.push_back(boot.sigs[i]);
    seen.push_back(boot.last.sig);
    auto novelty = [&](const Sig& s) {
        double m = 1.0;
        for (const Sig& q : seen) {
            m = std::min(m, sig_diff(s, q));
            if (m < kNovel) break;
        }
        return m;
    };
    size_t novel_len = route_inputs.size();  // route length at the last step that led to a new screen
    int stagnant_steps = 0;
    bool stagnant_out = false;
    auto note_commit = [&](const Window& w) {
        if (novelty(w.last.sig) >= kNovel) {
            novel_len = route_inputs.size();
            stagnant_steps = 0;
        } else {
            ++stagnant_steps;
        }
        for (size_t i = 4; i < w.sigs.size(); i += 5) seen.push_back(w.sigs[i]);
        seen.push_back(w.last.sig);
    };
    // Where the game first looked playable, kept in case it falls back to a menu later.
    size_t big_route_len = 0;
    std::vector<u8> big_state;
    // The start of the latest step: where the stick was tested. A route that ends there is verified;
    // one that ends after the step's own action (an A press may open a menu) is not.
    size_t step_route_len = 0;
    std::vector<u8> step_state;
    size_t ingame_route_len = 0;
    std::vector<u8> ingame_state;
    // A 3D scene that keeps moving (a race, a flight, a cutscene, a level): good to judge the drawing
    // by even where the stick does nothing. `scene_*` is the latest state of a streak of six such
    // steps, `last_scene_*` the latest of any such step (where a route can end if nothing better came).
    size_t scene_route_len = 0, last_scene_route_len = 0;
    std::vector<u8> scene_state, last_scene_state;
    int scene_steps = 0, scene_frame = -1;
    char play_hold = 0;  // 'a' / 'z' / 'b': the button that has to be held for the stick to matter
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
    double advice_wait = 0;  // seconds spent waiting for advice: not part of the time budget
    int advice_n = 0, advice_asked = 0, advice_used = 0;
    // Advice that was followed and left the screen the same kind (a menu still, the same prompt) did not work;
    // after two such failures it is not tried again for that kind of screen.
    std::map<std::string, int> advice_fails;
    std::string last_ask_kind, last_used_advice;
    auto note_reply = [&](const AdviceReply& r) {
        if (!last_used_advice.empty() && r.kind == last_ask_kind) ++advice_fails[r.kind + "|" + last_used_advice];
        last_ask_kind = r.kind;
        last_used_advice.clear();
    };
    auto advice_usable = [&](const AdviceReply& r, const AdviceCand& c) {
        const auto it = advice_fails.find(r.kind + "|" + c.name);
        return it == advice_fails.end() || it->second < 2;
    };
    auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t_start).count() - advice_wait; };
    bool out_of_time = false;
    // A quick shot before (and now and then during) the careful search: START and A tapped in
    // turn for a few hundred frames - what a person does to get into most games - and then the
    // usual test whether the stick moves the picture. If it does, those frames are part of the
    // route, and the search carries on from there; if not, the machine goes back.
    int mash_attempts = 0;
    auto mash_try = [&]() -> bool {
        ++mash_attempts;
        const std::vector<u8> before = pr.emu->save_state();
        Window wm = pr.run(o.mash_len, [](int f) {
            Input i;
            if (f % 40 < 4) i.buttons = Button::START;
            else if (f % 40 >= 20 && f % 40 < 24) i.buttons = Button::A;
            return i;
        }, nullptr);
        const std::vector<u8> after = wm.end_state;
        Window wi = pr.run(o.step, idle(), nullptr, false);
        pr.load(after);
        Window wk2 = pr.run(o.step, stick(o.step), &wi, false);
        pr.load(after);
        if (wk2.max_diff_vs_ref < kBigStick) {
            pr.load(before);
            return false;
        }
        route_inputs.insert(route_inputs.end(), wm.inputs.begin(), wm.inputs.end());
        sig_here(wm.last);
        note_commit(wm);
        pr.frame += o.mash_len;
        timeline << (first_entry ? "" : ",\n  ") << "{\"frame\":" << pr.frame << ",\"action\":\"MASH\"}";
        first_entry = false;
        shot("t" + std::to_string(pr.frame), wm.last);
        return true;
    };
    if (o.mash_len > 0 && mash_try()) { prev_sig = Sig{}; have_prev_sig = false; }
    while (pr.frame < o.boot + o.frames) {
        if (elapsed() > o.max_seconds * 0.7) { out_of_time = true; break; }
        if (o.mash_len > 0 && mash_attempts < 5 && steps > 0 && steps % 20 == 0 && big_stick_steps == 0 && mash_try()) {
            have_prev_sig = have_prev2_sig = false;
            stuck_steps = 0;
        }
        if (stuck_steps >= kStuckSteps) {
            // The picture has not changed for a while: a prompt that wants the cursor moved ("No / Yes"),
            // a screen to go back from, a button the experiments do not use. Everything that could help
            // is tried once from here, and what changes the picture most is kept (START by default).
            const std::vector<u8> su = pr.emu->save_state();
            if (!o.advice.empty()) {
                const Picture pic = pr.grab();
                AdviceReply rep = ask_advice(o.advice, ++advice_n, pic, 120, &advice_wait);
                ++advice_asked;
                note_reply(rep);
                Window aref = pr.run(o.step, idle(), nullptr, false);
                double bd = 0;
                Window bw;
                std::string bname;
                int tried = 0;
                for (size_t i = 0; i < rep.cands.size() && tried < 4; ++i) {
                    if (rep.cands[i].score < 0.4) break;
                    if (!advice_usable(rep, rep.cands[i])) continue;
                    ++tried;
                    std::vector<Input> in = rep.cands[i].inputs;
                    if (in.size() < static_cast<size_t>(o.step)) in.resize(o.step);
                    pr.load(su);
                    Window w = pr.run(static_cast<int>(in.size()), from_inputs(in), &aref);
                    if (w.max_diff_vs_ref > bd) { bd = w.max_diff_vs_ref; bw = std::move(w); bname = rep.cands[i].name; }
                }
                if (bd >= 0.02) {
                    ++advice_used;
                    last_used_advice = bname;
                    pr.load(bw.end_state);
                    timeline << (first_entry ? "" : ",\n  ") << "{\"frame\":" << pr.frame << ",\"action\":\"UNSTICK:ADVICE\",\"advice\":" << jstr(bname)
                             << ",\"kind\":" << jstr(rep.kind) << "}";
                    first_entry = false;
                    pr.frame += bw.frames;
                    route_inputs.insert(route_inputs.end(), bw.inputs.begin(), bw.inputs.end());
                    sig_here(bw.last);
                    note_commit(bw);
                    shot("t" + std::to_string(pr.frame), bw.last);
                    prev_sig = bw.last.sig;
                    have_prev2_sig = false;
                    stuck_steps = 0;
                    continue;
                }
                pr.load(su);
            }
            Window ref = pr.run(o.step, idle(), nullptr, false);
            struct Cand { const char* name; Script sc; int len; };
            // every button a menu may want: the cursor + A in four directions, B, Z, the C buttons, L, R, the
            // D-pad, and START (which is also what is done when nothing else changes anything)
            const Cand cands[] = {{"NAV0", nav_a(0), o.step}, {"NAV1", nav_a(1), o.step}, {"NAV2", nav_a(2), o.step},
                                  {"NAV3", nav_a(3), o.step}, {"B", tap(Button::B), o.step}, {"Z", tap(Button::Z), o.step},
                                  {"C_DOWN", tap(Button::C_DOWN), o.step}, {"C_UP", tap(Button::C_UP), o.step},
                                  {"C_LEFT", tap(Button::C_LEFT), o.step}, {"C_RIGHT", tap(Button::C_RIGHT), o.step},
                                  {"L", tap(Button::L), o.step}, {"R", tap(Button::R), o.step},
                                  {"D_DOWN", tap(Button::D_DOWN), o.step}, {"D_UP", tap(Button::D_UP), o.step},
                                  {"D_LEFT", tap(Button::D_LEFT), o.step}, {"D_RIGHT", tap(Button::D_RIGHT), o.step},
                                  {"START", tap(Button::START), 2 * o.step}};
            const size_t n_cands = sizeof cands / sizeof cands[0];
            std::vector<Window> ws_;
            ws_.reserve(n_cands);
            size_t best = n_cands - 1;
            double best_d = 0.02, best_nov = -1;
            for (size_t c = 0; c < n_cands; ++c) {
                pr.load(su);
                ws_.push_back(pr.run(cands[c].len, cands[c].sc, &ref));
                if (c + 1 == n_cands) break;
                const double d = ws_.back().max_diff_vs_ref;
                if (d <= 0.02) continue;
                // what leads to a screen not seen before beats what only changes the one in front of us
                const double nv = novelty(ws_.back().last.sig);
                const bool new_ok = nv >= kNovel;
                const bool better = new_ok ? (best_nov < kNovel || nv > best_nov) : (best_nov < kNovel && d > best_d);
                if (better) { best = c; best_d = d; best_nov = nv; }
            }
            Window& wu = ws_[best];
            pr.load(wu.end_state);
            timeline << (first_entry ? "" : ",\n  ") << "{\"frame\":" << pr.frame << ",\"action\":\"UNSTICK:" << cands[best].name << "\"}";
            first_entry = false;
            pr.frame += cands[best].len;
            route_inputs.insert(route_inputs.end(), wu.inputs.begin(), wu.inputs.end());
            sig_here(wu.last);
            note_commit(wu);
            if (best + 1 == n_cands) ++presses_start;
            shot("t" + std::to_string(pr.frame), wu.last);
            prev_sig = wu.last.sig;
            have_prev2_sig = false;
            stuck_steps = 0;
            continue;
        }
        const std::vector<u8> s0 = pr.emu->save_state();
        step_route_len = route_inputs.size();
        step_state = s0;
        AdviceReply rep;
        bool asked = false;
        // every few steps, and while nothing new has appeared for a while (the picture may be moving - a dialog, a
        // repeated explanation - and still lead nowhere)
        if (!o.advice.empty() && (steps % o.advice_every == 0 || (stagnant_steps >= 4 && stagnant_steps % 4 == 0)) && big_stick_steps == 0) {
            const Picture pic = pr.grab();
            rep = ask_advice(o.advice, ++advice_n, pic, 120, &advice_wait);
            asked = true;
            ++advice_asked;
            note_reply(rep);
        }
        Window w0 = pr.run(o.step, idle(), nullptr);
        if (!pr.load(s0)) {
            states_ok = false;
            pr.load(w0.end_state);
            pr.frame += o.step;
            route_inputs.insert(route_inputs.end(), w0.inputs.begin(), w0.inputs.end());
            sig_here(w0.last);
            break;
        }
        Window wa = pr.run(o.step, tap(Button::A), &w0);
        const bool react_a = wa.max_diff_vs_ref > kReact;
        const double nov_a = react_a ? novelty(wa.last.sig) : 0;
        Window ws;
        if (!react_a || nov_a < kNovel) {  // (A that goes back to a known screen is no answer)
            pr.load(s0);
            ws = pr.run(o.step, tap(Button::START), &w0);
        }
        const bool react_s = ws.max_diff_vs_ref > kReact;
        const double nov_s = react_s ? novelty(ws.last.sig) : 0;
        pr.load(s0);
        Window wk = pr.run(o.step, stick(o.step), &w0);
        const bool react_k = wk.max_diff_vs_ref > kReact;
        const bool big_k = wk.max_diff_vs_ref >= kBigStick;
        // Racing, flying and shooting games only answer the stick once a button is held (gas, fire). In a
        // 3D scene that keeps moving, the stick is tried again with A, Z and B held, one at a time.
        const double moving0 = w0.samples ? static_cast<double>(w0.moving_samples) / w0.samples : 0;
        const bool scene3d = w0.triangles >= 150 && moving0 >= 0.3;
        Window wh;
        double hold_diff = 0;
        char hold_btn = 0;
        if (scene3d && !big_k) {
            const struct { u16 b; char c; } holds[] = {{Button::A, 'a'}, {Button::Z, 'z'}, {Button::B, 'b'}};
            for (const auto& h : holds) {
                pr.load(s0);
                Window w = pr.run(o.step, stick(o.step, h.b), &w0);
                if (w.max_diff_vs_ref > hold_diff) { hold_diff = w.max_diff_vs_ref; hold_btn = h.c; wh = std::move(w); }
            }
        }
        const bool big_h = hold_diff >= kBigStick;
        const bool big_any = big_k || big_h;
        // Neither A nor START nor a big stick effect: maybe a menu that wants
        // the cursor moved first. Try stick + A in each direction; it counts
        // when it changes the picture clearly more than the stick alone.
        Window wn;
        int nav_dir = -1;
        double nav_diff = 0;
        if (!(react_a && nov_a >= kNovel) && !(react_s && nov_s >= kNovel) && react_k && !big_k) {  // (only where the stick moves something: a cursor)
            for (int d = 0; d < 4; ++d) {
                pr.load(s0);
                Window w = pr.run(o.step, nav_a(d), &w0);
                if (w.max_diff_vs_ref > nav_diff) { nav_diff = w.max_diff_vs_ref; nav_dir = d; wn = std::move(w); }
            }
        }
        const bool react_n = nav_dir >= 0 && nav_diff > 0.02 && nav_diff > wk.max_diff_vs_ref * 1.2;
        // The advice: each candidate is run from here; the one that changes the picture most counts.
        Window wadv;
        double adv_diff = 0;
        std::string adv_name;
        if (asked) {
            int tried = 0;
            for (size_t i = 0; i < rep.cands.size() && tried < 3; ++i) {
                if (rep.cands[i].score < 0.45) break;
                if (!advice_usable(rep, rep.cands[i])) continue;
                ++tried;
                std::vector<Input> in = rep.cands[i].inputs;
                if (in.size() < static_cast<size_t>(o.step)) in.resize(o.step);
                pr.load(s0);
                Window w = pr.run(static_cast<int>(in.size()), from_inputs(in), &w0);
                if (w.max_diff_vs_ref > adv_diff) { adv_diff = w.max_diff_vs_ref; wadv = std::move(w); adv_name = rep.cands[i].name; }
            }
        }
        const bool adv_ok = adv_diff >= 0.02 && rep.conf >= 0.55;

        // A comes first: it confirms the highlighted menu entry (usually
        // the one that starts a game) and skips text; in a game it is mostly
        // harmless (jump, accelerate, shoot). START only when neither A nor
        // the stick does much, so a game in progress isn't paused.
        big_stick_steps = big_any ? big_stick_steps + 1 : 0;
        const char* action = "idle";
        const Window* chosen = &w0;
        if (big_any) {
            // gameplay: A (jump, gas) when it does something, else the steering that moved the picture
            if (react_a) {
                action = "A";
                chosen = &wa;
                ++presses_a;
            } else {
                action = big_k ? "stick" : "hold";
                chosen = big_k ? &wk : &wh;
                if (!big_k) play_hold = hold_btn;
            }
        } else {
            // Of the actions that change the picture, one that leads to a screen not seen before wins; the
            // order among them is the strategy's (what the screen says first, or START before A, or the
            // cursor first, or blind taps first). If none leads anywhere new the first one is taken anyway.
            struct Opt { const char* name; const Window* w; double nov; };
            Opt opts[4];
            int n_opts = 0;
            static const int kOrder[4][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {3, 0, 2, 1}, {1, 2, 3, 0}};  // per strategy: indices of ADVICE, A, START, NAV in preference order
            const Opt all[4] = {{"ADVICE", &wadv, adv_ok ? novelty(wadv.last.sig) : 0.0}, {"A", &wa, nov_a},
                                {"START", &ws, nov_s}, {"NAV", &wn, react_n ? novelty(wn.last.sig) : 0.0}};
            const bool has[4] = {adv_ok, react_a, react_s, react_n};
            for (int k = 0; k < 4; ++k) {
                const int i = kOrder[o.strategy][k];
                if (has[i]) opts[n_opts++] = all[i];
            }
            int pick = -1;
            for (int k = 0; k < n_opts && pick < 0; ++k)
                if (opts[k].nov >= kNovel) pick = k;
            if (pick < 0 && n_opts > 0) pick = 0;
            if (pick >= 0) {
                action = opts[pick].name;
                chosen = opts[pick].w;
                if (action[0] == 'A' && action[1] == 'D') { ++advice_used; last_used_advice = adv_name; }
                else if (action[0] == 'A') ++presses_a;
                else if (action[0] == 'S') ++presses_start;
                else ++presses_nav;
            }
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
                 << ",\"polls\":" << w0.polls << ",\"vi\":" << w0.vi_swaps << ",\"hold\":" << jnum(hold_diff)
                 << ",\"scene\":" << (scene3d ? 1 : 0);
        if (asked) timeline << ",\"kind\":" << jstr(rep.kind) << ",\"advice\":" << jstr(adv_name) << ",\"advice_diff\":" << jnum(adv_diff);
        timeline << "}";
        first_entry = false;
        scene_steps = scene3d ? scene_steps + 1 : 0;
        if (scene3d) {
            last_scene_route_len = step_route_len;
            last_scene_state = step_state;
        }
        if (scene_steps >= 6) {
            scene_frame = pr.frame;
            scene_route_len = step_route_len;
            scene_state = step_state;
        }
        // The first moment the stick moves much of the picture: likely
        // gameplay, kept even if the game later falls back to a menu.
        if (big_any && first_big_frame < 0) {
            first_big_frame = pr.frame;
            big_route_len = route_inputs.size();
            big_state = s0;
            shot("first_play", big_k ? wk.last : wh.last);
        }
        if (!pr.load(chosen->end_state)) { states_ok = false; break; }
        route_inputs.insert(route_inputs.end(), chosen->inputs.begin(), chosen->inputs.end());
        sig_here(chosen->last);
        note_commit(*chosen);
        if ((have_prev_sig && sig_diff(prev_sig, chosen->last.sig) < kSame) ||
            (have_prev2_sig && sig_diff(prev2_sig, chosen->last.sig) < kSame)) ++stuck_steps;
        else stuck_steps = 0;
        prev2_sig = prev_sig;
        have_prev2_sig = have_prev_sig;
        prev_sig = chosen->last.sig;
        have_prev_sig = true;
        pr.frame += chosen->frames;
        if (steps % o.shot_every == 0) shot("t" + std::to_string(pr.frame), chosen->last);
        ++steps;
        // Gameplay: the stick has moved much of the picture for a while (a
        // menu is usually left by the A presses before that).
        if (big_stick_steps >= kIngameSteps) {
            ingame_frame = pr.frame;
            ingame_route_len = step_route_len;
            ingame_state = step_state;
            break;
        }
        if (dead_steps >= kDeadSteps) break;
        if (stagnant_steps >= kStagnantSteps && big_stick_steps == 0 && scene_steps == 0) {
            stagnant_out = true;  // round and round the same screens: more time will not help
            break;
        }
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

    // The route and the machine where it ends. "ingame": the stick moved much of the
    // picture for several steps in a row. "big": it did once, then the game fell back
    // to a menu - the route stops there. "furthest": never, the route goes as far as
    // the run got (a menu, a title screen, a frozen game).
    // The frontier: every committed input up to where the search ended, in whatever state the game is. The route
    // above is cut at the best moment; this is where the next search starts, so that nothing explored is explored
    // again.
    if (!o.frontier.empty()) {
        Route fr;
        fr.rom = std::filesystem::path(o.rom).filename().string();
        fr.crc = crc_text(pr.emu->get_cartridge());
        fr.title = pr.emu->get_cartridge().get_title();
        fr.state_version = savestate::kStateVersion;
        fr.rsp = o.rsp;
        fr.boot = o.boot;
        fr.reached = "furthest";
        fr.play = "stick";
        fr.inputs.assign(route_inputs.begin(), route_inputs.begin() + static_cast<std::ptrdiff_t>(std::min(novel_len, route_inputs.size())));
        for (const RouteSig& sg : route_sigs)
            if (sg.frame <= static_cast<int>(fr.inputs.size())) fr.sigs.push_back(sg);
        if (!write_route(o.frontier, fr)) std::fprintf(stderr, "[probe] cannot write %s\n", o.frontier.c_str());
    }
    std::string route_reached;
    if (!o.route.empty()) {
        Route r;
        r.rom = std::filesystem::path(o.rom).filename().string();
        r.crc = crc_text(pr.emu->get_cartridge());
        r.title = pr.emu->get_cartridge().get_title();
        r.state_version = savestate::kStateVersion;
        r.rsp = o.rsp;
        r.boot = o.boot;
        r.ingame_frame = ingame_frame;
        r.play = play_hold ? std::string("stick_") + play_hold
                           : ((fk.max_diff_vs_ref < kBigStick && fhk.max_diff_vs_ref >= kBigStick) ? "stick_a" : "stick");
        const std::vector<u8>* cp = &sf;
        if (ingame_frame >= 0) {
            // up to the start of the step in which the sixth proof came: the stick moved the picture there
            r.reached = "ingame";
            r.inputs.assign(route_inputs.begin(), route_inputs.begin() + static_cast<std::ptrdiff_t>(ingame_route_len));
            cp = &ingame_state;
        } else if (scene_route_len > 0 && (big_state.empty() || scene_route_len >= big_route_len)) {
            // six steps in a row of a moving 3D scene, later than (or without) the stick's first big effect
            r.reached = "scene";
            r.inputs.assign(route_inputs.begin(), route_inputs.begin() + static_cast<std::ptrdiff_t>(scene_route_len));
            cp = &scene_state;
        } else if (first_big_frame >= 0 && !big_state.empty()) {
            r.reached = "big";
            r.inputs.assign(route_inputs.begin(), route_inputs.begin() + static_cast<std::ptrdiff_t>(big_route_len));
            cp = &big_state;
        } else if (last_scene_route_len > 0) {
            // never a streak: end at the latest moment a 3D scene was moving (not in a menu or a pause)
            r.reached = "furthest";
            r.ends = "scene";
            r.inputs.assign(route_inputs.begin(), route_inputs.begin() + static_cast<std::ptrdiff_t>(last_scene_route_len));
            cp = &last_scene_state;
        } else {
            r.reached = "furthest";
            r.inputs = route_inputs;
        }
        route_reached = r.reached;
        for (const RouteSig& sg : route_sigs)
            if (sg.frame <= static_cast<int>(r.inputs.size())) r.sigs.push_back(sg);
        if (!write_route(o.route, r)) std::fprintf(stderr, "[probe] cannot write %s\n", o.route.c_str());
        if (!o.checkpoint.empty()) {
            std::string err;
            if (!savestate::write_file(o.checkpoint, *cp, checkpoint_info(*pr.emu), err))
                std::fprintf(stderr, "[probe] cannot write %s: %s\n", o.checkpoint.c_str(), err.c_str());
        }
    }

    const double secs = std::chrono::duration<double>(Clock::now() - t_start).count();
    std::ostringstream j;
    j << "{\n\"rom\":" << jstr(std::filesystem::path(o.rom).filename().string())
      << ",\n\"title\":" << jstr(pr.emu->get_cartridge().get_title())
      << ",\n\"frames\":" << pr.frame << ",\"steps\":" << steps << ",\"step\":" << o.step
      << ",\"seconds\":" << jnum(secs) << ",\"emulated_frames\":" << pr.emulated << ",\"states_ok\":" << (states_ok ? "true" : "false")
      << ",\"out_of_time\":" << (out_of_time ? "true" : "false") << ",\"dead_steps\":" << dead_steps << ",\"final_len\":" << o.final_len
      << ",\n\"stagnant\":" << (stagnant_out ? "true" : "false") << ",\"novel_len\":" << novel_len << ",\"strategy\":" << o.strategy
      << ",\n\"advice\":{\"asked\":" << advice_asked << ",\"used\":" << advice_used << ",\"wait_s\":" << jnum(advice_wait) << "}"
      << ",\n\"route_reached\":" << jstr(route_reached) << ",\"scene_frame\":" << scene_frame
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
