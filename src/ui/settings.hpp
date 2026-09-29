#pragma once
// Persistent user settings, stored as a small INI file in the per-user
// config directory (see platform::config_dir()).

#include <array>
#include <string>
#include <vector>

namespace ui {

// N64 controller inputs that can be bound.
enum class N64Input {
    A, B, Z, Start, L, R,
    CUp, CDown, CLeft, CRight,
    DUp, DDown, DLeft, DRight,
    StickUp, StickDown, StickLeft, StickRight,
    Count
};
constexpr int kN64InputCount = static_cast<int>(N64Input::Count);
const char* n64_input_name(N64Input in);
const char* n64_input_id(N64Input in);

// Gamepad binding encoding: -1 = unbound, 0..(SDL_GAMEPAD_BUTTON_COUNT-1) = button,
// kPadAxisBase + axis*2 + (0 = negative, 1 = positive) = analog axis direction.
constexpr int kPadAxisBase = 100;

struct PortConfig {
    bool plugged = true;
    int device = 0; // 0 = keyboard, 1 = first gamepad, 2 = second gamepad...
    std::array<int, kN64InputCount> keys{};  // SDL_Scancode
    std::array<int, kN64InputCount> pad{};   // see encoding above
    float deadzone = 0.15f;
    float sensitivity = 1.0f;  // analog scale, 1.0 = full N64 range (±80)
    // Accessory slot: 0 None, 1 Controller Pak (saved as <rom>.mpk, or
    // <rom>.pN.mpk for port N > 1), 2 Rumble Pak (vibrates the gamepad),
    // 3 Transfer Pak (holds the Game Boy ROM `gb_rom`).
    int pak = 0;
    int rumble_strength = 100; // 0..100 %
    std::string gb_rom;        // UTF-8 path, "" = no cartridge
};

struct Settings {
    // General
    int accent = 0;
    int ui_scale_pct = 0; // 0 = automatic (follow display DPI)
    bool confirm_stop = true;
    bool pause_on_focus_loss = true;
    bool start_in_library = true;
    bool show_status_bar = true;
    bool status_bar_in_fullscreen = false;
    bool hide_cursor_fullscreen = true;
    bool show_info_panel = true;
    bool use_native_dialogs = true;
    bool remember_window = true;
    int window_x = -1, window_y = -1, window_w = 1280, window_h = 800;
    bool window_maximized = false;

    // Graphics
    int internal_scale = 1; // RDP internal resolution: 1 = native, 2..8 = multiple of the frame buffer
    int aspect = 0;      // 0 = 4:3, 1 = 16:9, 2 = Stretch, 3 = Native frame size
    bool integer_scale = false;
    int filter = 1;      // 0 = Nearest, 1 = Bilinear
    bool vsync = true;
    int scanlines = 0;   // 0..100 intensity
    bool show_fps_overlay = false;
    int fullscreen_mode = 0; // 0 = borderless desktop, 1 = exclusive
    // 0 = GPU (SDL_GPU: Metal / Vulkan / Direct3D 12; high resolutions render
    // on the graphics card), 1 = compatibility (any SDL renderer, CPU only).
    // Takes effect at the next start.
    int video_backend = 0;

    // Audio
    bool audio_enabled = true;
    int volume = 80;             // 0..100
    bool mute_on_fast_forward = true;
    bool mute_on_focus_loss = false;
    std::string audio_device;    // empty = system default
    int sample_rate = 48000;
    int buffer_frames = 512;

    // Emulation
    int cpu_core = 1;        // 0 = Interpreter, 1 = Dynamic Recompiler (JIT)
    int ucode_override = 0;  // 0 = Auto, then MicrocodeType order
    int rsp_mode = 0;        // RspMode: 0 = HLE, 1 = LLE graphics, 2 = LLE graphics and audio
    bool rdp_exact = true;   // low-level graphics: bit-exact RDP instead of the fast one
    int ff_speed = 3;        // fast-forward multiplier (0 = unlimited)
    bool limit_speed = true;
    bool expansion_pak = true; // 8 MB RDRAM; off = 4 MB (applies at the next start / reset)
    bool autosave_backup = true;
    int fps_limit = 0;       // 0 = console default (VI rate), >0 = locked FPS
    int render_scale = 1;    // 1 = native, 2..8 = integer scale, 0 = custom size
    int render_w = 1920, render_h = 1080;

    // Input
    std::array<PortConfig, 4> ports{};

    // Library
    std::vector<std::string> rom_dirs;
    std::string boxart_dir; // empty = auto-detect inside ROM folders
    bool scan_recursive = true;
    int library_view = 0;    // 0 = grid, 1 = list
    int library_sort = 0;    // 0 = title, 1 = recently played, 2 = size, 3 = region
    float grid_card_size = 1.0f;

    void reset_defaults();
    void reset_general();
    void reset_graphics();
    void reset_audio();
    void reset_emulation();
    void reset_port(int port);

    bool load(const std::string& path);
    bool save(const std::string& path) const;
};

} // namespace ui
