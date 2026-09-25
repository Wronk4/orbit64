#include "app.hpp"
#include "platform.hpp"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>

namespace ui {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Audio callback (SDL audio thread)
// ---------------------------------------------------------------------------

namespace {
struct AudioCtx {
    EmuCore* core;
    std::atomic<float>* volume;
};
AudioCtx g_audio_ctx;

void audio_callback(void* userdata, Uint8* stream, int len) {
    auto* ctx = static_cast<AudioCtx*>(userdata);
    ctx->core->pull_audio(reinterpret_cast<float*>(stream), static_cast<size_t>(len) / sizeof(float), ctx->volume->load());
}
} // namespace

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

int App::run(const std::string& initial_rom) {
    if (!init()) return 1;
    if (!initial_rom.empty()) launch(platform::utf8_to_path(initial_rom));
    main_loop();
    shutdown();
    return 0;
}

bool App::init() {
    // Per-monitor DPI awareness on Windows; harmless elsewhere.
    // Hints are set by name so older SDL2 headers without these macros still compile.
    SDL_SetHint("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2");
    SDL_SetHint("SDL_IME_SHOW_UI", "1");
    SDL_SetHint(SDL_HINT_VIDEO_ALLOW_SCREENSAVER, "0");
    // Keep gamepad input flowing while the window is in the background.
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    // Don't minimise when a fullscreen window loses focus (alt-tab friendly).
    SDL_SetHint(SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS, "0");

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMECONTROLLER | SDL_INIT_TIMER) != 0) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, kAppName, SDL_GetError(), nullptr);
        return false;
    }

    settings_path_ = platform::path_to_utf8(platform::config_dir() / "settings.ini");
    settings_.load(settings_path_);
    if (settings_.rom_dirs.empty()) {
        // First run: seed the library with the working directory if it has ROMs.
        std::error_code ec;
        for (const auto& e : fs::directory_iterator(fs::current_path(ec), ec)) {
            if (is_rom_extension(e.path())) {
                settings_.rom_dirs.push_back(platform::path_to_utf8(fs::current_path(ec)));
                break;
            }
        }
    }

    // Window
    Uint32 flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_HIDDEN;
    int x = SDL_WINDOWPOS_CENTERED, y = SDL_WINDOWPOS_CENTERED;
    int w = 1280, h = 800;
    if (settings_.remember_window) {
        w = std::max(900, settings_.window_w);
        h = std::max(560, settings_.window_h);
        if (settings_.window_x != -1) {
            // Only restore the position if it's still on a connected display.
            SDL_Rect r{settings_.window_x, settings_.window_y, w, h};
            for (int d = 0; d < SDL_GetNumVideoDisplays(); ++d) {
                SDL_Rect b;
                if (SDL_GetDisplayUsableBounds(d, &b) == 0 && SDL_HasIntersection(&r, &b)) {
                    x = settings_.window_x;
                    y = settings_.window_y;
                    break;
                }
            }
        }
    }
    std::string title = std::string(kAppName) + " \xE2\x80\x94 Nintendo 64 Emulator";
    window_ = SDL_CreateWindow(title.c_str(), x, y, w, h, flags);
    if (!window_) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, kAppName, SDL_GetError(), nullptr);
        return false;
    }
    SDL_SetWindowMinimumSize(window_, 800, 520);
    if (settings_.remember_window && settings_.window_maximized) SDL_MaximizeWindow(window_);

    // SDL picks the best native backend: Direct3D on Windows, Metal on macOS,
    // OpenGL on Linux. No platform-specific rendering code is needed.
    Uint32 rflags = SDL_RENDERER_ACCELERATED | (settings_.vsync ? SDL_RENDERER_PRESENTVSYNC : 0);
    renderer_ = SDL_CreateRenderer(window_, -1, rflags);
    if (!renderer_) renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_SOFTWARE);
    if (!renderer_) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, kAppName, SDL_GetError(), window_);
        return false;
    }
    SDL_RendererInfo info;
    if (SDL_GetRendererInfo(renderer_, &info) == 0) renderer_name_ = info.name;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    static std::string ini_path = platform::path_to_utf8(platform::config_dir() / "layout.ini");
    io.IniFilename = ini_path.c_str();
    // Gamepads drive the game, not the UI, so gamepad navigation stays off.
    io.ConfigWindowsMoveFromTitleBarOnly = true;
    // macOS convention: Cmd acts as the shortcut modifier inside text fields.
    io.ConfigMacOSXBehaviors = platform::current_os() == platform::OS::MacOS;

    ImGui_ImplSDL2_InitForSDLRenderer(window_, renderer_);
    ImGui_ImplSDLRenderer2_Init(renderer_);

    update_ui_scale(true);
    input_.open_all();
    open_audio();

    browser_.native_available = platform::native_dialogs_available();
    browser_.on_native_requested = [this](FileBrowser::Mode m) {
        native_mode_ = m;
        native_job_ = std::async(std::launch::async, [m]() {
            return m == FileBrowser::Mode::OpenRom ? platform::native_open_file("Open Nintendo 64 ROM", {"z64", "n64", "v64"})
                                                   : platform::native_pick_folder("Choose ROM Folder");
        });
    };

    browser_.cover_for = [this](const RomInfo& rom) -> ImTextureID {
        SDL_Texture* t = images_.get(library_.boxart_for(rom.path, rom.internal_name));
        return (ImTextureID)(intptr_t)t;
    };

    library_.load(platform::config_dir() / "library.tsv");
    rescan_library();
    library_sidebar_ = true;

    SDL_ShowWindow(window_);
    return true;
}

void App::shutdown() {
    if (core_.loaded()) stop_now();
    clear_state_slots();
    close_audio();
    save_settings();
    library_.save();
    for (auto& [k, t] : thumbs_) if (t) SDL_DestroyTexture(t);
    thumbs_.clear();
    images_.clear();
    if (game_tex_) SDL_DestroyTexture(game_tex_);
    if (scaled_tex_) SDL_DestroyTexture(scaled_tex_);
    for (auto& [k, t] : dbg_textures_) SDL_DestroyTexture(t);
    dbg_textures_.clear();
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(renderer_);
    SDL_DestroyWindow(window_);
    SDL_Quit();
}

void App::save_settings() {
    if (window_ && !fullscreen_) {
        Uint32 f = SDL_GetWindowFlags(window_);
        settings_.window_maximized = (f & SDL_WINDOW_MAXIMIZED) != 0;
        if (!settings_.window_maximized && !(f & SDL_WINDOW_MINIMIZED)) {
            SDL_GetWindowPosition(window_, &settings_.window_x, &settings_.window_y);
            SDL_GetWindowSize(window_, &settings_.window_w, &settings_.window_h);
        }
    }
    settings_.save(settings_path_);
}

void App::update_ui_scale(bool force) {
    int ww = 0, wh = 0, dw = 0, dh = 0;
    SDL_GetWindowSize(window_, &ww, &wh);
    SDL_GetRendererOutputSize(renderer_, &dw, &dh);
    float fb = ww > 0 ? static_cast<float>(dw) / ww : 1.0f;
    float sys = platform::system_ui_scale(std::max(0, SDL_GetWindowDisplayIndex(window_)));
    float scale = settings_.ui_scale_pct > 0 ? settings_.ui_scale_pct / 100.0f : sys;
    scale = std::clamp(scale, 0.75f, 3.0f);
    if (!force && std::fabs(scale - applied_scale_) < 0.01f && std::fabs(fb - applied_fb_scale_) < 0.01f &&
        applied_accent_ == settings_.accent)
        return;
    bool fonts_changed = force || std::fabs(scale - applied_scale_) >= 0.01f || std::fabs(fb - applied_fb_scale_) >= 0.01f;
    g_scale = scale;
    fb_scale_ = fb;
    applied_scale_ = scale;
    applied_fb_scale_ = fb;
    applied_accent_ = settings_.accent;
    apply_palette(settings_.accent);
    if (fonts_changed) {
        build_fonts(scale, fb);
        ImGui_ImplSDLRenderer2_DestroyFontsTexture();
        ImGui_ImplSDLRenderer2_CreateFontsTexture();
    }
}

void App::open_audio() {
    close_audio();
    audio_devices_.clear();
    for (int i = 0; i < SDL_GetNumAudioDevices(0); ++i)
        if (const char* n = SDL_GetAudioDeviceName(i, 0)) audio_devices_.push_back(n);
    if (!settings_.audio_enabled) return;

    SDL_AudioSpec want{}, have{};
    want.freq = settings_.sample_rate;
    want.format = AUDIO_F32SYS;
    want.channels = 2;
    want.samples = static_cast<Uint16>(settings_.buffer_frames);
    want.callback = audio_callback;
    g_audio_ctx = {&core_, &audio_volume_};
    want.userdata = &g_audio_ctx;
    const char* dev = settings_.audio_device.empty() ? nullptr : settings_.audio_device.c_str();
    audio_dev_ = SDL_OpenAudioDevice(dev, 0, &want, &have, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    if (!audio_dev_ && dev) audio_dev_ = SDL_OpenAudioDevice(nullptr, 0, &want, &have, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    if (audio_dev_) {
        audio_freq_ = have.freq;
        core_.set_audio_output(static_cast<std::uint32_t>(have.freq), have.samples);
        SDL_PauseAudioDevice(audio_dev_, 0);
    }
}

void App::close_audio() {
    if (audio_dev_) SDL_CloseAudioDevice(audio_dev_);
    audio_dev_ = 0;
    audio_freq_ = 0;
    core_.set_audio_output(0, 0);
}

void App::update_audio_volume() {
    float v = settings_.volume / 100.0f;
    v = v * v; // perceptual curve
    if (settings_.mute_on_fast_forward && core_.fast_forward()) v = 0.0f;
    if (settings_.mute_on_focus_loss && !window_focused_) v = 0.0f;
    audio_volume_ = v;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

void App::main_loop() {
    while (running_) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) process_event(e);

        Uint32 wflags = SDL_GetWindowFlags(window_);
        if (wflags & SDL_WINDOW_MINIMIZED) {
            SDL_Delay(30);
            continue;
        }

        update_ui_scale(false);
        library_.poll();
        images_.pump(renderer_);
        update_input();
        update_game_texture();
        update_scaled_texture();
        update_audio_volume();
        debug_update();
        poll_state_events();

        // Native file dialog finished?
        if (native_job_ && native_job_->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            auto res = native_job_->get();
            native_job_.reset();
            if (res && !res->empty()) {
                if (native_mode_ == FileBrowser::Mode::OpenRom) launch(*res);
                else on_folder_picked(*res);
            }
            SDL_RaiseWindow(window_);
        }

        // Session bookkeeping
        if (core_.state() == RunState::Running) {
            double now = ImGui::GetTime();
            if (now - fps_sample_time_ >= 0.25) {
                fps_sample_time_ = now;
                fps_history_.push_back(core_.stats().fps);
                while (fps_history_.size() > 120) fps_history_.pop_front();
            }
            if (now - last_thumb_time_ > 60.0 && has_frame_) {
                last_thumb_time_ = now;
                save_thumbnail();
            }
        }

        // Cursor auto-hide in fullscreen gameplay.
        bool hide_cursor = fullscreen_ && settings_.hide_cursor_fullscreen && view_ == View::Game &&
                           core_.state() == RunState::Running && ImGui::GetTime() - last_mouse_move_ > 2.0 &&
                           !settings_open_;
        SDL_ShowCursor(hide_cursor ? SDL_DISABLE : SDL_ENABLE);

        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        draw_root();
        ImGui::Render();
        if (!pending_launch_.empty()) {
            fs::path p = std::move(pending_launch_);
            pending_launch_.clear();
            launch(p);
        }

        // The SDL_Renderer backend emits logical (window) coordinates; SDL scales
        // them to physical pixels on Retina / high-DPI displays.
        const ImVec2 fbs = ImGui::GetIO().DisplayFramebufferScale;
        SDL_RenderSetScale(renderer_, fbs.x, fbs.y);
        SDL_SetRenderDrawColor(renderer_, 10, 11, 14, 255);
        SDL_RenderClear(renderer_);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer_);
        if (!ui_test_dir_.empty()) ui_test_tick();
        SDL_RenderPresent(renderer_);
        if (!settings_.vsync) SDL_Delay(2); // keep the UI thread from spinning at thousands of fps
    }
}

void App::process_event(const SDL_Event& e) {
    // Rebinding capture takes priority over everything else.
    if (input_.capturing() && (e.type == SDL_KEYDOWN || e.type == SDL_CONTROLLERBUTTONDOWN || e.type == SDL_CONTROLLERAXISMOTION)) {
        input_.handle_event(e);
        int port, in, key, pad;
        if (input_.consume_capture(port, in, key, pad)) {
            PortConfig& pc = settings_.ports[port];
            if (key >= 0) pc.keys[in] = key;
            if (pad >= 0) pc.pad[in] = pad;
            save_settings();
        }
        return;
    }

    ImGui_ImplSDL2_ProcessEvent(&e);
    input_.handle_event(e);

    switch (e.type) {
        case SDL_QUIT: request_quit(); break;
        case SDL_WINDOWEVENT:
            if (e.window.windowID != SDL_GetWindowID(window_)) break;
            if (e.window.event == SDL_WINDOWEVENT_CLOSE) request_quit();
            if (e.window.event == SDL_WINDOWEVENT_FOCUS_LOST) {
                window_focused_ = false;
                if (settings_.pause_on_focus_loss && core_.state() == RunState::Running && !native_job_) {
                    core_.pause(true);
                    auto_paused_ = true;
                }
            }
            if (e.window.event == SDL_WINDOWEVENT_FOCUS_GAINED) {
                window_focused_ = true;
                if (auto_paused_ && core_.state() == RunState::Paused) core_.pause(false);
                auto_paused_ = false;
            }
            if (e.window.event == SDL_WINDOWEVENT_DISPLAY_CHANGED || e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                update_ui_scale(false);
            break;
        case SDL_MOUSEMOTION: last_mouse_move_ = ImGui::GetTime(); break;
        case SDL_KEYDOWN:
            if (handle_shortcut(e.key)) break;
            break;
        case SDL_DROPFILE: {
            fs::path p = platform::utf8_to_path(e.drop.file);
            SDL_free(e.drop.file);
            std::error_code ec;
            if (fs::is_directory(p, ec)) {
                std::string s = platform::path_to_utf8(p);
                if (std::find(settings_.rom_dirs.begin(), settings_.rom_dirs.end(), s) == settings_.rom_dirs.end()) {
                    settings_.rom_dirs.push_back(s);
                    save_settings();
                }
                rescan_library();
                toast("Folder added to library", ToastKind::Success);
            } else if (is_rom_extension(p)) {
                launch(p);
            } else {
                toast("Unsupported file type (expected .z64, .n64 or .v64)", ToastKind::Warning);
            }
            break;
        }
        default: break;
    }
}

std::vector<Shortcut> App::shortcuts() const {
    using platform::shortcut_label;
    bool mac = platform::current_os() == platform::OS::MacOS;
    return {
        {"Open ROM", shortcut_label(true, false, false, "O")},
        {"Add ROM folder", shortcut_label(true, true, false, "O")},
        {"Toggle Library / Game", shortcut_label(true, false, false, "L")},
        {"Pause / Resume", shortcut_label(true, false, false, "P") + "  or  F5"},
        {"Stop emulation", shortcut_label(true, false, false, ".") + "  or  Shift+F5"},
        {"Reset", shortcut_label(true, false, false, "R")},
        {"Save state to the current slot", shortcut_label(true, false, false, "S") + "  or  F2"},
        {"Load state from the current slot", shortcut_label(true, true, false, "L") + "  or  F4"},
        {"Choose state slot", shortcut_label(true, false, false, "1\xE2\x80\x93" "9")},
        {"Fast forward (hold)", "Tab"},
        {"Fullscreen", mac ? "Cmd+Ctrl+F  or  F11" : "F11  or  Alt+Enter"},
        {"Exit fullscreen", "Esc"},
        {"Screenshot", "F12"},
        {"Game info panel", shortcut_label(true, false, false, "I")},
        {"Settings", shortcut_label(true, false, false, ",")},
        {"Quit", mac ? "Cmd+Q" : "Ctrl+Q  or  Alt+F4"},
    };
}

bool App::handle_shortcut(const SDL_KeyboardEvent& k) {
    if (k.repeat) return false;
    const unsigned mod = k.keysym.mod;
    const bool primary = platform::primary_mod_down(mod);
    const bool shift = (mod & KMOD_SHIFT) != 0;
    const bool alt = (mod & KMOD_ALT) != 0;
    const bool mac = platform::current_os() == platform::OS::MacOS;
    const SDL_Keycode key = k.keysym.sym;
    const bool modal_open = settings_open_ || about_open_ || confirm_open_ || browser_.is_open() || props_open_ || error_open_;

    if (primary) {
        switch (key) {
            case SDLK_o:
                if (modal_open) return false;
                if (shift) action_add_folder(); else action_open_rom();
                return true;
            case SDLK_l:
                if (modal_open) return false;
                if (shift) load_state(state_slot_);
                else view_ = view_ == View::Library ? View::Game : View::Library;
                return true;
            case SDLK_s:
                if (modal_open || shift) return false;
                save_state(state_slot_);
                return true;
            case SDLK_1: case SDLK_2: case SDLK_3: case SDLK_4: case SDLK_5:
            case SDLK_6: case SDLK_7: case SDLK_8: case SDLK_9:
                if (modal_open || shift) return false;
                select_state_slot(static_cast<int>(key - SDLK_0));
                return true;
            case SDLK_p: if (!modal_open) toggle_pause(); return true;
            case SDLK_r: if (!modal_open) reset_game(); return true;
            case SDLK_PERIOD: if (!modal_open) request_stop(); return true;
            case SDLK_i: settings_.show_info_panel = !settings_.show_info_panel; return true;
            case SDLK_COMMA: if (!modal_open) open_settings(SettingsPage::General); return true;
            case SDLK_q: request_quit(); return true;
            case SDLK_f:
                // macOS: Cmd+Ctrl+F is the system-standard fullscreen shortcut.
                if (mac && (mod & KMOD_CTRL)) { toggle_fullscreen(); return true; }
                return false;
            default: break;
        }
    }
    switch (key) {
        case SDLK_F11: toggle_fullscreen(); return true;
        case SDLK_RETURN:
            if (alt && !mac) { toggle_fullscreen(); return true; }
            return false;
        case SDLK_F5:
            if (modal_open) return false;
            if (shift) request_stop(); else toggle_pause();
            return true;
        case SDLK_F12: take_screenshot(); return true;
        case SDLK_F2:
            if (modal_open) return false;
            save_state(state_slot_);
            return true;
        case SDLK_F4:
            if (modal_open || alt) return false; // Alt+F4 closes the window
            load_state(state_slot_);
            return true;
        case SDLK_ESCAPE:
            if (fullscreen_ && !modal_open && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId)) {
                set_fullscreen(false);
                return true;
            }
            return false;
        default: return false;
    }
}

void App::update_input() {
    ImGuiIO& io = ImGui::GetIO();
    bool modal_open = settings_open_ || about_open_ || confirm_open_ || browser_.is_open() || props_open_ || error_open_;
    bool keyboard_to_game = view_ == View::Game && window_focused_ && !io.WantTextInput && !modal_open && !input_.capturing();
    for (int p = 0; p < 4; ++p) {
        ControllerSnapshot s = input_.poll(settings_.ports[p], keyboard_to_game);
        if (p == 0 && test_input_) s.buttons |= test_input_(core_.stats().frame);
        core_.set_input(p, s);
    }

    // Fast-forward while Tab is held (only when the game has keyboard focus).
    const Uint8* keys = SDL_GetKeyboardState(nullptr);
    core_.set_fast_forward(keyboard_to_game && keys[SDL_SCANCODE_TAB]);
    core_.set_ff_multiplier(settings_.ff_speed);
    core_.set_limit_speed(settings_.limit_speed);
    core_.set_fps_limit(settings_.fps_limit);
    core_.set_internal_scale(settings_.internal_scale);
}

void App::update_game_texture() {
    int w = 0, h = 0, scale = 1;
    if (!core_.fetch_frame(frame_pixels_, w, h, scale, frame_serial_)) return;
    if (frame_pixels_.empty() || w <= 0 || h <= 0) {
        has_frame_ = false;
        return;
    }
    if (!game_tex_ || w != game_tex_w_ || h != game_tex_h_) {
        if (game_tex_) SDL_DestroyTexture(game_tex_);
        game_tex_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, w, h);
        game_tex_w_ = w;
        game_tex_h_ = h;
        game_tex_filter_ = -1;
    }
    game_scale_ = std::max(1, scale);
    if (game_tex_filter_ != settings_.filter) {
        SDL_SetTextureScaleMode(game_tex_, settings_.filter == 0 ? SDL_ScaleModeNearest : SDL_ScaleModeLinear);
        game_tex_filter_ = settings_.filter;
    }
    SDL_UpdateTexture(game_tex_, nullptr, frame_pixels_.data(), w * static_cast<int>(sizeof(std::uint32_t)));
    has_frame_ = true;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

void App::action_open_rom() {
    if (native_job_) return;
    if (settings_.use_native_dialogs && browser_.native_available) {
        browser_.on_native_requested(FileBrowser::Mode::OpenRom);
        return;
    }
    fs::path start;
    if (!current_rom_.path.empty()) start = current_rom_.path.parent_path();
    else if (!settings_.rom_dirs.empty()) start = platform::utf8_to_path(settings_.rom_dirs.front());
    browser_.library_dirs.clear();
    for (const auto& d : settings_.rom_dirs) browser_.library_dirs.push_back(platform::utf8_to_path(d));
    browser_.open(FileBrowser::Mode::OpenRom, start);
}

void App::rescan_library() { library_.rescan(settings_.rom_dirs, settings_.scan_recursive, settings_.boxart_dir); }

SDL_Texture* App::card_texture(const GameEntry& g) {
    if (g.stats.last_played > 0)
        if (SDL_Texture* t = thumbnail_texture(g.key)) return t;
    return images_.get(g.rom.boxart);
}

void App::on_folder_picked(const fs::path& dir) {
    std::string d = platform::path_to_utf8(dir);
    if (folder_purpose_ == FolderPurpose::BoxArt) {
        settings_.boxart_dir = d;
        save_settings();
        images_.clear();
        rescan_library();
        toast("Box art folder set to " + platform::path_to_utf8(dir.filename()), ToastKind::Success);
        return;
    }
    if (std::find(settings_.rom_dirs.begin(), settings_.rom_dirs.end(), d) != settings_.rom_dirs.end()) {
        toast("That folder is already in your library", ToastKind::Info);
        return;
    }
    settings_.rom_dirs.push_back(d);
    save_settings();
    rescan_library();
    toast("Added " + platform::path_to_utf8(dir.filename()) + " to your library", ToastKind::Success);
}

void App::action_choose_boxart_folder() {
    if (native_job_) return;
    folder_purpose_ = FolderPurpose::BoxArt;
    if (settings_.use_native_dialogs && browser_.native_available) {
        browser_.on_native_requested(FileBrowser::Mode::PickFolder);
        return;
    }
    browser_.library_dirs.clear();
    for (const auto& d : settings_.rom_dirs) browser_.library_dirs.push_back(platform::utf8_to_path(d));
    browser_.open(FileBrowser::Mode::PickFolder,
                  settings_.rom_dirs.empty() ? platform::home_dir() : platform::utf8_to_path(settings_.rom_dirs.front()));
}

void App::action_add_folder() {
    if (native_job_) return;
    folder_purpose_ = FolderPurpose::RomDir;
    if (settings_.use_native_dialogs && browser_.native_available) {
        browser_.on_native_requested(FileBrowser::Mode::PickFolder);
        return;
    }
    browser_.library_dirs.clear();
    browser_.open(FileBrowser::Mode::PickFolder, platform::home_dir());
}

void App::launch(const fs::path& rom) {
    auto do_launch = [this, rom]() {
        if (core_.loaded()) stop_now();
        RomInfo info = inspect_rom(rom);
        if (!info.valid) {
            error_title_ = "Can't open this file";
            error_msg_ = platform::path_to_utf8(rom.filename()) +
                         " is not a valid Nintendo 64 ROM. Supported formats are .z64 (big-endian), .v64 (byte-swapped) and .n64 (little-endian).";
            error_open_ = true;
            return;
        }
        core_.set_ucode_override(settings_.ucode_override);
        core_.set_cpu_core(settings_.cpu_core);
        std::string err;
        if (!core_.start(rom, err)) {
            error_title_ = "Failed to start emulation";
            error_msg_ = err;
            error_open_ = true;
            return;
        }
        GameEntry& g = library_.ensure(rom);
        current_key_ = g.key;
        current_rom_ = info;
        current_rom_.boxart = g.rom.boxart;
        library_.mark_launched(current_key_);
        session_start_ = ImGui::GetTime();
        session_accum_ = 0;
        last_thumb_time_ = ImGui::GetTime() - 50.0; // first thumbnail ~10 s in
        fps_history_.clear();
        has_frame_ = false;
        view_ = View::Game;
        std::string title = current_rom_.display_title + " \xE2\x80\x94 " + kAppName;
        SDL_SetWindowTitle(window_, title.c_str());
        toast("Now playing: " + current_rom_.display_title, ToastKind::Success);
    };
    if (core_.loaded() && settings_.confirm_stop) {
        confirm("Switch game?", "The current game will be stopped. Unsaved in-game progress will be lost.", "Switch Game",
                true, do_launch);
    } else {
        do_launch();
    }
}

void App::stop_now() {
    if (!core_.loaded()) return;
    save_thumbnail();
    double played = core_.stats().uptime_s;
    core_.stop();
    clear_state_slots();
    library_.add_play_time(current_key_, static_cast<std::int64_t>(played));
    has_frame_ = false;
    // Debugger state refers to the stopped game's memory and scene.
    dbg_.snap.reset();
    dbg_.objects.clear();
    dbg_.selected = {};
    dbg_.player = {};
    dbg_.player_found = false;
    dbg_.player_trail.clear();
    dbg_.mesh_pos.clear();
    dbg_.mesh_col.clear();
    dbg_.mesh_uv.clear();
    dbg_.mesh_tex.clear();
    dbg_.mesh_texture_count = 0;
    dbg_.search_started = false;
    dbg_.search_mask.clear();
    dbg_.search_prev.clear();
    dbg_.search_results.clear();
    dbg_.search_count = 0;
    dbg_.search_scans = 0;
    fps_history_.clear();
    std::string title = std::string(kAppName) + " \xE2\x80\x94 Nintendo 64 Emulator";
    SDL_SetWindowTitle(window_, title.c_str());
}

void App::request_stop(std::function<void()> then) {
    if (!core_.loaded()) return;
    auto act = [this, then]() {
        stop_now();
        toast("Emulation stopped");
        if (then) then();
    };
    if (settings_.confirm_stop) confirm("Stop emulation?", "Unsaved in-game progress will be lost. Cartridge saves (EEPROM/SRAM) are written to disk automatically.", "Stop", true, act);
    else act();
}

void App::toggle_pause() {
    if (!core_.loaded()) return;
    core_.pause(core_.state() == RunState::Running);
    auto_paused_ = false;
}

void App::reset_game() {
    if (!core_.loaded()) return;
    core_.reset();
    fps_history_.clear();
    toast("Console reset");
}

void App::set_fullscreen(bool on) {
    if (on == fullscreen_) return;
    Uint32 mode = on ? (settings_.fullscreen_mode == 1 ? SDL_WINDOW_FULLSCREEN : SDL_WINDOW_FULLSCREEN_DESKTOP) : 0;
    if (on) save_settings(); // remember windowed geometry first
    if (SDL_SetWindowFullscreen(window_, mode) == 0) {
        fullscreen_ = on;
        chrome_reveal_ = 1.0f;
        last_mouse_move_ = ImGui::GetTime();
    }
}

void App::toggle_fullscreen() { set_fullscreen(!fullscreen_); }

static bool write_bmp(const fs::path& path, const std::vector<std::uint32_t>& px, int w, int h) {
    SDL_Surface* s = SDL_CreateRGBSurfaceWithFormatFrom(const_cast<std::uint32_t*>(px.data()), w, h, 32, w * 4,
                                                        SDL_PIXELFORMAT_ARGB8888);
    if (!s) return false;
    SDL_RWops* rw = SDL_RWFromFile(platform::path_to_utf8(path).c_str(), "wb"); // SDL expects UTF-8 on all platforms
    bool ok = rw && SDL_SaveBMP_RW(s, rw, 1) == 0;
    SDL_FreeSurface(s);
    return ok;
}

void App::take_screenshot() {
    std::vector<std::uint32_t> px;
    int w, h;
    if (!core_.loaded() || !core_.snapshot(px, w, h)) {
        toast("Nothing to capture \xE2\x80\x94 start a game first", ToastKind::Warning);
        return;
    }
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tm);
    std::string base = current_rom_.display_title.empty() ? "screenshot" : current_rom_.display_title;
    for (char& c : base)
        if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') c = '_';
    fs::path path = platform::screenshots_dir() / platform::utf8_to_path(base + " " + stamp + ".bmp");
    if (write_bmp(path, px, w, h)) toast("Screenshot saved to " + platform::path_to_utf8(path.parent_path().filename()) + "/" + platform::path_to_utf8(path.filename()), ToastKind::Success, 4.0f);
    else toast("Couldn't save screenshot", ToastKind::Error);
}

void App::save_thumbnail() {
    if (current_key_.empty()) return;
    std::vector<std::uint32_t> px;
    int w, h, scale = 1;
    if (!core_.snapshot(px, w, h, &scale) || w <= 0 || h <= 0) return;
    // Library cards only need the game's own resolution.
    downscale_frame(px, w, h, scale);
    // Skip all-black frames (boot, fades) so thumbnails stay meaningful.
    std::uint64_t sum = 0;
    for (size_t i = 0; i < px.size(); i += 37) sum += (px[i] & 0xFF) + ((px[i] >> 8) & 0xFF) + ((px[i] >> 16) & 0xFF);
    if (sum < px.size() / 37 * 12) return;
    if (write_bmp(library_.thumbnail_path(current_key_), px, w, h)) invalidate_thumbnail(current_key_);
}

SDL_Texture* App::thumbnail_texture(const std::string& key) {
    auto it = thumbs_.find(key);
    if (it != thumbs_.end()) return it->second;
    if (thumb_missing_.count(key)) return nullptr;
    SDL_Texture* tex = nullptr;
    fs::path p = library_.thumbnail_path(key);
    std::error_code ec;
    if (fs::exists(p, ec)) {
        if (SDL_Surface* s = SDL_LoadBMP(platform::path_to_utf8(p).c_str())) {
            tex = SDL_CreateTextureFromSurface(renderer_, s);
            if (tex) SDL_SetTextureScaleMode(tex, SDL_ScaleModeLinear);
            SDL_FreeSurface(s);
        }
    }
    if (tex) thumbs_[key] = tex;
    else thumb_missing_[key] = true;
    return tex;
}

void App::invalidate_thumbnail(const std::string& key) {
    auto it = thumbs_.find(key);
    if (it != thumbs_.end()) {
        if (it->second) SDL_DestroyTexture(it->second);
        thumbs_.erase(it);
    }
    thumb_missing_.erase(key);
}

void App::request_quit() {
    if (core_.loaded() && settings_.confirm_stop) {
        confirm("Quit Orbit64?", "The running game will be stopped. Unsaved in-game progress will be lost.", "Quit", true,
                [this]() { running_ = false; });
    } else {
        running_ = false;
    }
}

void App::confirm(const std::string& title, const std::string& message, const std::string& ok_label, bool danger,
                  std::function<void()> on_ok) {
    confirm_title_ = title;
    confirm_msg_ = message;
    confirm_ok_ = ok_label;
    confirm_danger_ = danger;
    confirm_cb_ = std::move(on_ok);
    confirm_open_ = true;
}

void App::open_settings(SettingsPage page) {
    settings_page_ = page;
    settings_open_ = true;
}

} // namespace ui
