#include "settings.hpp"
#include "platform.hpp"

#include <SDL3/SDL.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

namespace ui {

static const char* kInputNames[] = {
    "A", "B", "Z Trigger", "Start", "L", "R",
    "C Up", "C Down", "C Left", "C Right",
    "D-Pad Up", "D-Pad Down", "D-Pad Left", "D-Pad Right",
    "Stick Up", "Stick Down", "Stick Left", "Stick Right",
};
static const char* kInputIds[] = {
    "a", "b", "z", "start", "l", "r",
    "cup", "cdown", "cleft", "cright",
    "dup", "ddown", "dleft", "dright",
    "sup", "sdown", "sleft", "sright",
};

const char* n64_input_name(N64Input in) { return kInputNames[static_cast<int>(in)]; }
const char* n64_input_id(N64Input in) { return kInputIds[static_cast<int>(in)]; }

void Settings::reset_general() {
    Settings d;
    accent = d.accent; ui_scale_pct = d.ui_scale_pct; confirm_stop = d.confirm_stop;
    pause_on_focus_loss = d.pause_on_focus_loss; start_in_library = d.start_in_library;
    show_status_bar = d.show_status_bar; status_bar_in_fullscreen = d.status_bar_in_fullscreen;
    hide_cursor_fullscreen = d.hide_cursor_fullscreen; use_native_dialogs = d.use_native_dialogs;
    remember_window = d.remember_window;
}
void Settings::reset_graphics() {
    Settings d;
    internal_scale = d.internal_scale;
    aspect = d.aspect; integer_scale = d.integer_scale; filter = d.filter; vsync = d.vsync;
    video_backend = d.video_backend;
    scanlines = d.scanlines; show_fps_overlay = d.show_fps_overlay; fullscreen_mode = d.fullscreen_mode;
}
void Settings::reset_audio() {
    Settings d;
    audio_enabled = d.audio_enabled; volume = d.volume; mute_on_fast_forward = d.mute_on_fast_forward;
    mute_on_focus_loss = d.mute_on_focus_loss; audio_device = d.audio_device;
    sample_rate = d.sample_rate; buffer_frames = d.buffer_frames;
}
void Settings::reset_emulation() {
    Settings d;
    cpu_core = d.cpu_core; ucode_override = d.ucode_override; rsp_mode = d.rsp_mode; rdp_exact = d.rdp_exact; ff_speed = d.ff_speed; limit_speed = d.limit_speed;
    autosave_backup = d.autosave_backup; fps_limit = d.fps_limit; expansion_pak = d.expansion_pak;
}

void Settings::reset_port(int port) {
    PortConfig& p = ports[port];
    p = PortConfig{};
    p.plugged = (port == 0);
    p.device = (port == 0) ? 0 : port; // port 1 -> keyboard, others -> gamepads
    p.pak = (port == 0) ? 1 : 0;       // Controller Pak in port 1
    p.keys.fill(SDL_SCANCODE_UNKNOWN);
    p.pad.fill(-1);
    if (port == 0) {
        auto& k = p.keys;
        k[(int)N64Input::A] = SDL_SCANCODE_X;
        k[(int)N64Input::B] = SDL_SCANCODE_Z;
        k[(int)N64Input::Z] = SDL_SCANCODE_SPACE;
        k[(int)N64Input::Start] = SDL_SCANCODE_RETURN;
        k[(int)N64Input::L] = SDL_SCANCODE_Q;
        k[(int)N64Input::R] = SDL_SCANCODE_E;
        k[(int)N64Input::CUp] = SDL_SCANCODE_I;
        k[(int)N64Input::CDown] = SDL_SCANCODE_K;
        k[(int)N64Input::CLeft] = SDL_SCANCODE_J;
        k[(int)N64Input::CRight] = SDL_SCANCODE_L;
        k[(int)N64Input::DUp] = SDL_SCANCODE_T;
        k[(int)N64Input::DDown] = SDL_SCANCODE_G;
        k[(int)N64Input::DLeft] = SDL_SCANCODE_F;
        k[(int)N64Input::DRight] = SDL_SCANCODE_H;
        k[(int)N64Input::StickUp] = SDL_SCANCODE_UP;
        k[(int)N64Input::StickDown] = SDL_SCANCODE_DOWN;
        k[(int)N64Input::StickLeft] = SDL_SCANCODE_LEFT;
        k[(int)N64Input::StickRight] = SDL_SCANCODE_RIGHT;
    }
    auto& g = p.pad;
    g[(int)N64Input::A] = SDL_GAMEPAD_BUTTON_SOUTH;
    g[(int)N64Input::B] = SDL_GAMEPAD_BUTTON_WEST;
    g[(int)N64Input::Z] = kPadAxisBase + SDL_GAMEPAD_AXIS_LEFT_TRIGGER * 2 + 1;
    g[(int)N64Input::Start] = SDL_GAMEPAD_BUTTON_START;
    g[(int)N64Input::L] = SDL_GAMEPAD_BUTTON_LEFT_SHOULDER;
    g[(int)N64Input::R] = SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER;
    g[(int)N64Input::CUp] = kPadAxisBase + SDL_GAMEPAD_AXIS_RIGHTY * 2 + 0;
    g[(int)N64Input::CDown] = kPadAxisBase + SDL_GAMEPAD_AXIS_RIGHTY * 2 + 1;
    g[(int)N64Input::CLeft] = kPadAxisBase + SDL_GAMEPAD_AXIS_RIGHTX * 2 + 0;
    g[(int)N64Input::CRight] = kPadAxisBase + SDL_GAMEPAD_AXIS_RIGHTX * 2 + 1;
    g[(int)N64Input::DUp] = SDL_GAMEPAD_BUTTON_DPAD_UP;
    g[(int)N64Input::DDown] = SDL_GAMEPAD_BUTTON_DPAD_DOWN;
    g[(int)N64Input::DLeft] = SDL_GAMEPAD_BUTTON_DPAD_LEFT;
    g[(int)N64Input::DRight] = SDL_GAMEPAD_BUTTON_DPAD_RIGHT;
    g[(int)N64Input::StickUp] = kPadAxisBase + SDL_GAMEPAD_AXIS_LEFTY * 2 + 0;
    g[(int)N64Input::StickDown] = kPadAxisBase + SDL_GAMEPAD_AXIS_LEFTY * 2 + 1;
    g[(int)N64Input::StickLeft] = kPadAxisBase + SDL_GAMEPAD_AXIS_LEFTX * 2 + 0;
    g[(int)N64Input::StickRight] = kPadAxisBase + SDL_GAMEPAD_AXIS_LEFTX * 2 + 1;
}

void Settings::reset_defaults() {
    *this = Settings{};
    for (int i = 0; i < 4; ++i) reset_port(i);
}

// ---------------------------------------------------------------------------
// INI serialisation
// ---------------------------------------------------------------------------

namespace {

struct Writer {
    std::ostringstream out;
    void section(const char* s) { out << "\n[" << s << "]\n"; }
    void kv(const std::string& k, int v) { out << k << "=" << v << "\n"; }
    void kv(const std::string& k, bool v) { out << k << "=" << (v ? 1 : 0) << "\n"; }
    void kv(const std::string& k, float v) { out << k << "=" << v << "\n"; }
    void kv(const std::string& k, const std::string& v) { out << k << "=" << v << "\n"; }
};

using Ini = std::map<std::string, std::string>; // "section.key" -> value

struct Reader {
    Ini& ini;
    std::string sec;
    const std::string* find(const char* k) const {
        auto it = ini.find(sec + "." + k);
        return it == ini.end() ? nullptr : &it->second;
    }
    void get(const char* k, int& v) const { if (auto* s = find(k)) v = std::atoi(s->c_str()); }
    void get(const char* k, bool& v) const { if (auto* s = find(k)) v = std::atoi(s->c_str()) != 0; }
    void get(const char* k, float& v) const { if (auto* s = find(k)) v = static_cast<float>(std::atof(s->c_str())); }
    void get(const char* k, std::string& v) const { if (auto* s = find(k)) v = *s; }
};

} // namespace

bool Settings::save(const std::string& path) const {
    Writer w;
    w.out << "; Orbit64 settings\n";
    w.section("general");
    w.kv("accent", accent); w.kv("ui_scale", ui_scale_pct); w.kv("confirm_stop", confirm_stop);
    w.kv("pause_on_focus_loss", pause_on_focus_loss); w.kv("start_in_library", start_in_library);
    w.kv("show_status_bar", show_status_bar); w.kv("status_bar_fullscreen", status_bar_in_fullscreen);
    w.kv("hide_cursor_fullscreen", hide_cursor_fullscreen); w.kv("show_info_panel", show_info_panel);
    w.kv("native_dialogs", use_native_dialogs); w.kv("remember_window", remember_window);
    w.kv("window_x", window_x); w.kv("window_y", window_y); w.kv("window_w", window_w); w.kv("window_h", window_h);
    w.kv("window_maximized", window_maximized);

    w.section("graphics");
    w.kv("internal_scale", internal_scale);
    w.kv("aspect", aspect); w.kv("integer_scale", integer_scale); w.kv("filter", filter); w.kv("vsync", vsync);
    w.kv("video_backend", video_backend);
    w.kv("scanlines", scanlines); w.kv("fps_overlay", show_fps_overlay); w.kv("fullscreen_mode", fullscreen_mode);

    w.section("audio");
    w.kv("enabled", audio_enabled); w.kv("volume", volume); w.kv("mute_ff", mute_on_fast_forward);
    w.kv("mute_unfocused", mute_on_focus_loss); w.kv("device", audio_device);
    w.kv("sample_rate", sample_rate); w.kv("buffer", buffer_frames);

    w.section("emulation");
    w.kv("cpu_core", cpu_core);
    w.kv("ucode", ucode_override); w.kv("rsp_mode", rsp_mode); w.kv("rdp_exact", rdp_exact); w.kv("ff_speed", ff_speed); w.kv("limit_speed", limit_speed);
    w.kv("autosave", autosave_backup); w.kv("expansion_pak", expansion_pak);
    w.kv("fps_limit", fps_limit); w.kv("render_scale", render_scale);
    w.kv("render_w", render_w); w.kv("render_h", render_h);

    for (int i = 0; i < 4; ++i) {
        std::string s = "port" + std::to_string(i + 1);
        w.section(s.c_str());
        const PortConfig& p = ports[i];
        w.kv("plugged", p.plugged); w.kv("device", p.device); w.kv("deadzone", p.deadzone);
        w.kv("sensitivity", p.sensitivity); w.kv("accessory", p.pak); w.kv("rumble_strength", p.rumble_strength); w.kv("gb_rom", p.gb_rom);
        for (int k = 0; k < kN64InputCount; ++k) {
            w.kv(std::string("key_") + kInputIds[k], p.keys[k]);
            w.kv(std::string("pad_") + kInputIds[k], p.pad[k]);
        }
    }

    w.section("library");
    w.kv("recursive", scan_recursive); w.kv("view", library_view); w.kv("sort", library_sort);
    w.kv("card_size", grid_card_size);
    w.kv("boxart_dir", boxart_dir);
    w.kv("dir_count", static_cast<int>(rom_dirs.size()));
    for (size_t i = 0; i < rom_dirs.size(); ++i) w.kv("dir" + std::to_string(i), rom_dirs[i]);

    std::ofstream f(platform::utf8_to_path(path), std::ios::binary);
    if (!f) return false;
    f << w.out.str();
    return true;
}

bool Settings::load(const std::string& path) {
    reset_defaults();
    std::ifstream f(platform::utf8_to_path(path), std::ios::binary);
    if (!f) return false;
    Ini ini;
    std::string line, sec;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        if (line[0] == '[') { sec = line.substr(1, line.find(']') - 1); continue; }
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        ini[sec + "." + line.substr(0, eq)] = line.substr(eq + 1);
    }

    Reader r{ini, "general"};
    r.get("accent", accent); r.get("ui_scale", ui_scale_pct); r.get("confirm_stop", confirm_stop);
    r.get("pause_on_focus_loss", pause_on_focus_loss); r.get("start_in_library", start_in_library);
    r.get("show_status_bar", show_status_bar); r.get("status_bar_fullscreen", status_bar_in_fullscreen);
    r.get("hide_cursor_fullscreen", hide_cursor_fullscreen); r.get("show_info_panel", show_info_panel);
    r.get("native_dialogs", use_native_dialogs); r.get("remember_window", remember_window);
    r.get("window_x", window_x); r.get("window_y", window_y); r.get("window_w", window_w); r.get("window_h", window_h);
    r.get("window_maximized", window_maximized);

    r.sec = "graphics";
    r.get("internal_scale", internal_scale);
    internal_scale = internal_scale < 1 ? 1 : (internal_scale > 8 ? 8 : internal_scale);
    r.get("aspect", aspect); r.get("integer_scale", integer_scale); r.get("filter", filter); r.get("vsync", vsync);
    r.get("video_backend", video_backend);
    video_backend = video_backend == 1 ? 1 : 0;
    r.get("scanlines", scanlines); r.get("fps_overlay", show_fps_overlay); r.get("fullscreen_mode", fullscreen_mode);

    r.sec = "audio";
    r.get("enabled", audio_enabled); r.get("volume", volume); r.get("mute_ff", mute_on_fast_forward);
    r.get("mute_unfocused", mute_on_focus_loss); r.get("device", audio_device);
    r.get("sample_rate", sample_rate); r.get("buffer", buffer_frames);

    r.sec = "emulation";
    r.get("cpu_core", cpu_core);
    r.get("ucode", ucode_override); r.get("rsp_mode", rsp_mode); r.get("rdp_exact", rdp_exact); r.get("ff_speed", ff_speed); r.get("limit_speed", limit_speed);
    r.get("autosave", autosave_backup); r.get("expansion_pak", expansion_pak);
    r.get("fps_limit", fps_limit); r.get("render_scale", render_scale);
    r.get("render_w", render_w); r.get("render_h", render_h);

    for (int i = 0; i < 4; ++i) {
        r.sec = "port" + std::to_string(i + 1);
        PortConfig& p = ports[i];
        r.get("plugged", p.plugged); r.get("device", p.device); r.get("deadzone", p.deadzone);
        r.get("sensitivity", p.sensitivity); r.get("accessory", p.pak); r.get("rumble_strength", p.rumble_strength); r.get("gb_rom", p.gb_rom);
        if (p.pak < 0 || p.pak > 3) p.pak = 0;
        p.rumble_strength = std::clamp(p.rumble_strength, 0, 100);
        for (int k = 0; k < kN64InputCount; ++k) {
            r.get((std::string("key_") + kInputIds[k]).c_str(), p.keys[k]);
            r.get((std::string("pad_") + kInputIds[k]).c_str(), p.pad[k]);
        }
    }

    r.sec = "library";
    r.get("recursive", scan_recursive); r.get("view", library_view); r.get("sort", library_sort);
    r.get("card_size", grid_card_size);
    r.get("boxart_dir", boxart_dir);
    int n = 0;
    r.get("dir_count", n);
    rom_dirs.clear();
    for (int i = 0; i < n; ++i) {
        std::string d;
        r.get(("dir" + std::to_string(i)).c_str(), d);
        if (!d.empty()) rom_dirs.push_back(d);
    }
    return true;
}

} // namespace ui
