#include "settings.hpp"
#include "platform.hpp"

#include <SDL3/SDL.h>
#include <algorithm>
#include <cmath>
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

static const char* kHotkeyNames[] = {
    "Open ROM", "Add ROM folder", "Toggle Library / Game", "Pause / Resume", "Stop emulation", "Reset",
    "Save state to the current slot", "Load state from the current slot", "Fast forward (hold)", "Fullscreen",
    "Screenshot", "Game info panel", "Settings", "Quit",
};
static const char* kHotkeyIds[] = {
    "open_rom", "add_folder", "toggle_view", "pause", "stop", "reset", "save_state", "load_state",
    "fast_forward", "fullscreen", "screenshot", "info_panel", "settings", "quit",
};
static_assert(sizeof(kHotkeyNames) / sizeof(kHotkeyNames[0]) == kHotkeyCount);
static_assert(sizeof(kHotkeyIds) / sizeof(kHotkeyIds[0]) == kHotkeyCount);

const char* hotkey_name(Hotkey h) { return kHotkeyNames[static_cast<int>(h)]; }

int key_mods_from_sdl(unsigned m) {
    const bool mac = platform::current_os() == platform::OS::MacOS;
    int out = 0;
    if (platform::primary_mod_down(m)) out |= kModPrimary;
    if (mac && (m & SDL_KMOD_CTRL)) out |= kModCtrl;
    if (m & SDL_KMOD_SHIFT) out |= kModShift;
    if (m & SDL_KMOD_ALT) out |= kModAlt;
    return out;
}

bool is_modifier_key(int key) {
    switch (key) {
        case SDLK_LCTRL: case SDLK_RCTRL: case SDLK_LSHIFT: case SDLK_RSHIFT:
        case SDLK_LALT: case SDLK_RALT: case SDLK_LGUI: case SDLK_RGUI:
        case SDLK_MODE: case SDLK_CAPSLOCK:
            return true;
        default: return false;
    }
}

std::string key_combo_label(const KeyCombo& c) {
    if (c.key == 0) return "";
    std::string s;
    if (c.mods & kModPrimary) s += std::string(platform::primary_mod_name()) + "+";
    if (c.mods & kModCtrl) s += "Ctrl+";
    if (c.mods & kModShift) s += "Shift+";
    if (c.mods & kModAlt) s += std::string(platform::alt_mod_name()) + "+";
    const char* name = SDL_GetKeyName(static_cast<SDL_Keycode>(c.key));
    s += name && *name ? name : "?";
    return s;
}

void Settings::reset_shortcuts() {
    const bool mac = platform::current_os() == platform::OS::MacOS;
    for (auto& h : hotkeys) h = {};
    auto set = [&](Hotkey h, KeyCombo a, KeyCombo b = {}) { hotkeys[static_cast<int>(h)] = {a, b}; };
    const int P = kModPrimary, S = kModShift;
    set(Hotkey::OpenRom, {SDLK_O, P});
    set(Hotkey::AddFolder, {SDLK_O, P | S});
    set(Hotkey::ToggleView, {SDLK_L, P});
    set(Hotkey::Pause, {SDLK_P, P}, {SDLK_F5, 0});
    set(Hotkey::Stop, {SDLK_PERIOD, P}, {SDLK_F5, S});
    set(Hotkey::Reset, {SDLK_R, P});
    set(Hotkey::SaveState, {SDLK_S, P}, {SDLK_F2, 0});
    set(Hotkey::LoadState, {SDLK_L, P | S}, {SDLK_F4, 0});
    set(Hotkey::FastForward, {SDLK_TAB, 0});
    if (mac) set(Hotkey::Fullscreen, {SDLK_F, P | kModCtrl}, {SDLK_F11, 0});
    else set(Hotkey::Fullscreen, {SDLK_F11, 0}, {SDLK_RETURN, kModAlt});
    set(Hotkey::Screenshot, {SDLK_F12, 0});
    set(Hotkey::InfoPanel, {SDLK_I, P});
    set(Hotkey::Settings, {SDLK_COMMA, P});
    set(Hotkey::Quit, {SDLK_Q, P});
}

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
    hle_raytracing = d.hle_raytracing; rt_shadow = d.rt_shadow; rt_ao = d.rt_ao;
    rt_pixel_lighting = d.rt_pixel_lighting; rt_specular = d.rt_specular;
    postfx = d.postfx; pfx_bloom = d.pfx_bloom; pfx_sharpen = d.pfx_sharpen; pfx_vibrance = d.pfx_vibrance;
    pfx_contrast = d.pfx_contrast; pfx_vignette = d.pfx_vignette; pfx_fxaa = d.pfx_fxaa; pfx_tonemap = d.pfx_tonemap;
    dlss_mode = d.dlss_mode; dlss_temporal = d.dlss_temporal;
}
void Settings::reset_audio() {
    Settings d;
    audio_enabled = d.audio_enabled; volume = d.volume; mute_on_fast_forward = d.mute_on_fast_forward;
    mute_on_focus_loss = d.mute_on_focus_loss; audio_device = d.audio_device;
    sample_rate = d.sample_rate; buffer_frames = d.buffer_frames;
}
void Settings::reset_emulation() {
    Settings d;
    cpu_core = d.cpu_core; ucode_override = d.ucode_override; rsp_mode = d.rsp_mode; rdp_exact = d.rdp_exact; rdp_exact_gpu = d.rdp_exact_gpu; ff_speed = d.ff_speed; limit_speed = d.limit_speed;
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
    reset_shortcuts();
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

bool read_ini(const std::string& path, Ini& ini) {
    std::ifstream f(platform::utf8_to_path(path), std::ios::binary);
    if (!f) return false;
    std::string line, sec;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        if (line[0] == '[') { sec = line.substr(1, line.find(']') - 1); continue; }
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        ini[sec + "." + line.substr(0, eq)] = line.substr(eq + 1);
    }
    return true;
}

// The profile part of a port: shared by settings.ini and controller profiles.
void write_profile(Writer& w, const PortConfig& p) {
    w.kv("deadzone", p.deadzone); w.kv("sensitivity", p.sensitivity);
    w.kv("octagon", p.octagon); w.kv("deadzone_rescale", p.deadzone_rescale);
    w.kv("rumble_strength", p.rumble_strength);
    for (int k = 0; k < kN64InputCount; ++k) {
        w.kv(std::string("key_") + kInputIds[k], p.keys[k]);
        w.kv(std::string("pad_") + kInputIds[k], p.pad[k]);
    }
}

void read_profile(const Reader& r, PortConfig& p) {
    r.get("deadzone", p.deadzone); r.get("sensitivity", p.sensitivity);
    r.get("octagon", p.octagon); r.get("deadzone_rescale", p.deadzone_rescale);
    r.get("rumble_strength", p.rumble_strength);
    p.deadzone = std::clamp(p.deadzone, 0.0f, 0.5f);
    p.sensitivity = std::clamp(p.sensitivity, 0.5f, 1.5f);
    p.rumble_strength = std::clamp(p.rumble_strength, 0, 100);
    for (int k = 0; k < kN64InputCount; ++k) {
        r.get((std::string("key_") + kInputIds[k]).c_str(), p.keys[k]);
        r.get((std::string("pad_") + kInputIds[k]).c_str(), p.pad[k]);
    }
}

} // namespace

bool save_controller_profile(const PortConfig& p, const std::string& path) {
    Writer w;
    w.out << "; Orbit64 controller profile\n";
    w.section("profile");
    write_profile(w, p);
    std::ofstream f(platform::utf8_to_path(path), std::ios::binary);
    if (!f) return false;
    f << w.out.str();
    return static_cast<bool>(f);
}

bool load_controller_profile(PortConfig& p, const std::string& path) {
    Ini ini;
    if (!read_ini(path, ini)) return false;
    Reader r{ini, "profile"};
    if (!r.find("deadzone")) return false; // not a profile
    read_profile(r, p);
    return true;
}

void apply_controller_profile(PortConfig& dst, const PortConfig& src) {
    dst.keys = src.keys;
    dst.pad = src.pad;
    dst.deadzone = src.deadzone;
    dst.sensitivity = src.sensitivity;
    dst.octagon = src.octagon;
    dst.deadzone_rescale = src.deadzone_rescale;
    dst.rumble_strength = src.rumble_strength;
}

bool same_controller_profile(const PortConfig& a, const PortConfig& b) {
    return a.keys == b.keys && a.pad == b.pad && std::abs(a.deadzone - b.deadzone) < 1e-4f &&
           std::abs(a.sensitivity - b.sensitivity) < 1e-4f && a.octagon == b.octagon &&
           a.deadzone_rescale == b.deadzone_rescale && a.rumble_strength == b.rumble_strength;
}

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
    w.kv("hle_raytracing", hle_raytracing); w.kv("rt_shadow", rt_shadow); w.kv("rt_ao", rt_ao);
    w.kv("rt_pixel_lighting", rt_pixel_lighting); w.kv("rt_specular", rt_specular);
    w.kv("postfx", postfx); w.kv("pfx_bloom", pfx_bloom); w.kv("pfx_sharpen", pfx_sharpen);
    w.kv("pfx_vibrance", pfx_vibrance); w.kv("pfx_contrast", pfx_contrast); w.kv("pfx_vignette", pfx_vignette);
    w.kv("pfx_fxaa", pfx_fxaa); w.kv("pfx_tonemap", pfx_tonemap);
    w.kv("dlss_mode", dlss_mode); w.kv("dlss_temporal", dlss_temporal);

    w.section("audio");
    w.kv("enabled", audio_enabled); w.kv("volume", volume); w.kv("mute_ff", mute_on_fast_forward);
    w.kv("mute_unfocused", mute_on_focus_loss); w.kv("device", audio_device);
    w.kv("sample_rate", sample_rate); w.kv("buffer", buffer_frames);

    w.section("emulation");
    w.kv("cpu_core", cpu_core);
    w.kv("ucode", ucode_override); w.kv("rsp_mode", rsp_mode); w.kv("rdp_exact", rdp_exact); w.kv("rdp_exact_gpu", rdp_exact_gpu); w.kv("ff_speed", ff_speed); w.kv("limit_speed", limit_speed);
    w.kv("autosave", autosave_backup); w.kv("expansion_pak", expansion_pak);
    w.kv("fps_limit", fps_limit); w.kv("render_scale", render_scale);
    w.kv("render_w", render_w); w.kv("render_h", render_h);

    for (int i = 0; i < 4; ++i) {
        std::string s = "port" + std::to_string(i + 1);
        w.section(s.c_str());
        const PortConfig& p = ports[i];
        w.kv("plugged", p.plugged); w.kv("device", p.device); w.kv("accessory", p.pak); w.kv("gb_rom", p.gb_rom);
        write_profile(w, p);
    }

    w.section("shortcuts");
    for (int h = 0; h < kHotkeyCount; ++h)
        for (int k = 0; k < kHotkeySlots; ++k) {
            const KeyCombo& c = hotkeys[h][k];
            w.kv(std::string(kHotkeyIds[h]) + (k ? "_2" : ""), std::to_string(c.key) + ":" + std::to_string(c.mods));
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
    Ini ini;
    if (!read_ini(path, ini)) return false;

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
    r.get("hle_raytracing", hle_raytracing); r.get("rt_shadow", rt_shadow); r.get("rt_ao", rt_ao);
    r.get("rt_pixel_lighting", rt_pixel_lighting); r.get("rt_specular", rt_specular);
    r.get("postfx", postfx); r.get("pfx_bloom", pfx_bloom); r.get("pfx_sharpen", pfx_sharpen);
    r.get("pfx_vibrance", pfx_vibrance); r.get("pfx_contrast", pfx_contrast); r.get("pfx_vignette", pfx_vignette);
    r.get("pfx_fxaa", pfx_fxaa); r.get("pfx_tonemap", pfx_tonemap);
    r.get("dlss_mode", dlss_mode); r.get("dlss_temporal", dlss_temporal);
    dlss_mode = std::clamp(dlss_mode, 0, 5);

    r.sec = "audio";
    r.get("enabled", audio_enabled); r.get("volume", volume); r.get("mute_ff", mute_on_fast_forward);
    r.get("mute_unfocused", mute_on_focus_loss); r.get("device", audio_device);
    r.get("sample_rate", sample_rate); r.get("buffer", buffer_frames);

    r.sec = "emulation";
    r.get("cpu_core", cpu_core);
    r.get("ucode", ucode_override); r.get("rsp_mode", rsp_mode); r.get("rdp_exact", rdp_exact); r.get("rdp_exact_gpu", rdp_exact_gpu); r.get("ff_speed", ff_speed); r.get("limit_speed", limit_speed);
    r.get("autosave", autosave_backup); r.get("expansion_pak", expansion_pak);
    r.get("fps_limit", fps_limit); r.get("render_scale", render_scale);
    r.get("render_w", render_w); r.get("render_h", render_h);

    for (int i = 0; i < 4; ++i) {
        r.sec = "port" + std::to_string(i + 1);
        PortConfig& p = ports[i];
        r.get("plugged", p.plugged); r.get("device", p.device); r.get("accessory", p.pak); r.get("gb_rom", p.gb_rom);
        if (p.pak < 0 || p.pak > 3) p.pak = 0;
        read_profile(r, p);
    }

    r.sec = "shortcuts";
    for (int h = 0; h < kHotkeyCount; ++h)
        for (int k = 0; k < kHotkeySlots; ++k) {
            std::string v;
            r.get((std::string(kHotkeyIds[h]) + (k ? "_2" : "")).c_str(), v);
            const size_t colon = v.find(':');
            if (colon == std::string::npos) continue; // keep the default
            hotkeys[h][k] = {std::atoi(v.c_str()), std::atoi(v.c_str() + colon + 1) & 15};
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
