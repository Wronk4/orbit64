// Routes: the inputs that took a game from power-on to where it can be played,
// frame by frame. The emulator is deterministic, so playing them back repeats
// the run that found them - in seconds instead of the minutes the experiments
// took. Used by tools/game_probe.cpp (writes them) and tools/game_qa.cpp
// (plays them); tools/qa.py reads the text format too.
//
//   orbit-route 1
//   rom <file name>
//   crc <CRC1>-<CRC2>
//   title <internal name>
//   state_version <n>        (savestate::kStateVersion the checkpoint was made with)
//   rsp hle
//   boot <idle frames>
//   reached ingame | scene | big | furthest   (how far the run got, see game_probe.cpp)
//   play stick | stick_a | stick_z | stick_b   (how the game is steered once it is playing; the
//                                    letter is the button held while the stick moves)
//   ingame_frame <n>
//   in <first frame> <frames> <buttons hex> <stick x> <stick y>   (run-length encoded)
//   sig <frame> <96 hex digits>      what the picture looked like after that many frames: 8x6 grey
//                                    cells. Replaying checks them, which tells where a game has drifted
//                                    away from the recording (and a lost route can be continued from the
//                                    last frame that still matched).
#pragma once

#include "probe_common.hpp"

namespace {

using RouteSigCells = std::array<u8, 48>;
struct RouteSig {
    int frame = 0;
    RouteSigCells cells{};
};

struct Route {
    std::string rom, crc, title, rsp = "hle", reached = "furthest", play = "stick";
    std::string ends;  // "scene": the route ends where a 3D scene was moving (not in a menu or a pause)
    u32 state_version = 0;
    int boot = 0, ingame_frame = -1;
    std::vector<Input> inputs;
    std::vector<RouteSig> sigs;
};

// 8x6 grey cells of a picture (from its 64x48 colour cells: 8x8 of them per grey cell).
RouteSigCells route_sig(const Picture& p) {
    RouteSigCells g{};
    for (int cy = 0; cy < 6; ++cy)
        for (int cx = 0; cx < 8; ++cx) {
            u32 sum = 0;
            for (int y = 0; y < 8; ++y)
                for (int x = 0; x < 8; ++x) {
                    const u8* c = &p.sig.rgb[((cy * 8 + y) * kSigW + cx * 8 + x) * 3];
                    sum += (c[0] * 77 + c[1] * 150 + c[2] * 29) >> 8;
                }
            g[cy * 8 + cx] = static_cast<u8>(sum / 64);
        }
    return g;
}

// How much two pictures differ by their grey cells: 0 (the same) .. 1.
double route_sig_diff(const RouteSigCells& a, const RouteSigCells& b) {
    int sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    return sum / (255.0 * a.size());
}

std::string crc_text(const Cartridge& c) {
    char b[24];
    std::snprintf(b, sizeof b, "%08X-%08X", c.get_crc1(), c.get_crc2());
    return b;
}

bool write_route(const std::string& path, const Route& r) {
    std::ofstream f(path);
    if (!f) return false;
    f << "orbit-route 1\n";
    f << "rom " << r.rom << "\ncrc " << r.crc << "\ntitle " << r.title << "\n";
    f << "state_version " << r.state_version << "\nrsp " << r.rsp << "\nboot " << r.boot << "\n";
    f << "reached " << r.reached << "\nplay " << r.play << "\ningame_frame " << r.ingame_frame << "\n";
    if (!r.ends.empty()) f << "ends " << r.ends << "\n";
    for (const RouteSig& sg : r.sigs) {
        char hex[100];
        for (size_t k = 0; k < sg.cells.size(); ++k) std::snprintf(hex + 2 * k, 3, "%02x", sg.cells[k]);
        f << "sig " << sg.frame << " " << hex << "\n";
    }
    size_t i = 0;
    while (i < r.inputs.size()) {
        size_t j = i + 1;
        while (j < r.inputs.size() && r.inputs[j].buttons == r.inputs[i].buttons && r.inputs[j].x == r.inputs[i].x &&
               r.inputs[j].y == r.inputs[i].y)
            ++j;
        f << "in " << i << " " << (j - i) << " " << std::hex << r.inputs[i].buttons << std::dec << " "
          << static_cast<int>(r.inputs[i].x) << " " << static_cast<int>(r.inputs[i].y) << "\n";
        i = j;
    }
    return static_cast<bool>(f);
}

bool read_route(const std::string& path, Route& r) {
    std::ifstream f(path);
    std::string line;
    if (!f || !std::getline(f, line) || line.rfind("orbit-route 1", 0) != 0) return false;
    r = Route{};
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string k;
        is >> k;
        auto rest = [&]() {
            std::string v;
            std::getline(is, v);
            return v.empty() ? v : v.substr(1);
        };
        if (k == "rom") r.rom = rest();
        else if (k == "crc") r.crc = rest();
        else if (k == "title") r.title = rest();
        else if (k == "rsp") r.rsp = rest();
        else if (k == "reached") r.reached = rest();
        else if (k == "play") r.play = rest();
        else if (k == "ends") r.ends = rest();
        else if (k == "state_version") is >> r.state_version;
        else if (k == "boot") is >> r.boot;
        else if (k == "ingame_frame") is >> r.ingame_frame;
        else if (k == "sig") {
            RouteSig sg;
            std::string hex;
            is >> sg.frame >> hex;
            if (hex.size() != 2 * sg.cells.size()) return false;
            for (size_t k2 = 0; k2 < sg.cells.size(); ++k2) sg.cells[k2] = static_cast<u8>(std::stoi(hex.substr(2 * k2, 2), nullptr, 16));
            r.sigs.push_back(sg);
        } else if (k == "in") {
            size_t first = 0, n = 0;
            unsigned buttons = 0;
            int x = 0, y = 0;
            is >> first >> std::dec >> n >> std::hex >> buttons >> std::dec >> x >> y;
            if (n > 100000000 || first > 100000000) return false;
            if (r.inputs.size() < first + n) r.inputs.resize(first + n);
            for (size_t k2 = 0; k2 < n; ++k2)
                r.inputs[first + k2] = Input{static_cast<u16>(buttons), static_cast<s8>(x), static_cast<s8>(y)};
        }
    }
    return !r.inputs.empty();
}

// How a game is steered once it is playing: the stick walks down, then right
// (moves a character or the camera in most games); some games, racing ones,
// only steer while A is held.
Script play_script(const std::string& play, int len) {
    u16 held = 0;
    if (play == "stick_a") held = Button::A;
    else if (play == "stick_z") held = Button::Z;
    else if (play == "stick_b") held = Button::B;
    return stick(len, held);
}

// The picture-neutral part of a checkpoint's file info.
savestate::FileInfo checkpoint_info(Emulator& emu) {
    savestate::FileInfo info;
    info.rom_crc1 = emu.get_cartridge().get_crc1();
    info.rom_crc2 = emu.get_cartridge().get_crc2();
    info.rom_title = emu.get_cartridge().get_title();
    return info;
}

}  // namespace
