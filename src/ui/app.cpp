#include "app.hpp"
#include "platform.hpp"
#include "../gpu/rdp_gpu.hpp"

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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
    std::vector<float> buf; // audio thread only
};
AudioCtx g_audio_ctx;

// SDL asks for `additional` more bytes of interleaved stereo float samples.
void SDLCALL audio_callback(void* userdata, SDL_AudioStream* stream, int additional, int /*total*/) {
    auto* ctx = static_cast<AudioCtx*>(userdata);
    const size_t count = static_cast<size_t>(std::max(additional, 0)) / sizeof(float) & ~size_t(1);
    if (count == 0) return;
    ctx->buf.resize(count);
    ctx->core->pull_audio(ctx->buf.data(), count, ctx->volume->load());
    SDL_PutAudioStreamData(stream, ctx->buf.data(), static_cast<int>(count * sizeof(float)));
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
    SDL_SetAppMetadata(kAppName, kAppVersion, "com.orbit64.emulator");
    SDL_SetHint(SDL_HINT_IME_IMPLEMENTED_UI, "0"); // the OS draws the IME candidate window
    SDL_SetHint(SDL_HINT_VIDEO_ALLOW_SCREENSAVER, "0");
    // Keep gamepad input flowing while the window is in the background.
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    // Don't minimise when a fullscreen window loses focus (alt-tab friendly).
    SDL_SetHint(SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS, "0");

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMEPAD)) {
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
    SDL_WindowFlags flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_HIDDEN;
    int x = SDL_WINDOWPOS_CENTERED, y = SDL_WINDOWPOS_CENTERED;
    int w = 1280, h = 800;
    if (settings_.remember_window) {
        w = std::max(900, settings_.window_w);
        h = std::max(560, settings_.window_h);
        if (settings_.window_x != -1) {
            // Only restore the position if it's still on a connected display.
            SDL_Rect r{settings_.window_x, settings_.window_y, w, h};
            int count = 0;
            SDL_DisplayID* displays = SDL_GetDisplays(&count);
            for (int d = 0; displays && d < count; ++d) {
                SDL_Rect b;
                if (SDL_GetDisplayUsableBounds(displays[d], &b) && SDL_HasRectIntersection(&r, &b)) {
                    x = settings_.window_x;
                    y = settings_.window_y;
                    break;
                }
            }
            SDL_free(displays);
        }
    }
    std::string title = std::string(kAppName) + " \xE2\x80\x94 Nintendo 64 Emulator";
    window_ = SDL_CreateWindow(title.c_str(), w, h, flags);
    if (!window_) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, kAppName, SDL_GetError(), nullptr);
        return false;
    }
    SDL_SetWindowPosition(window_, x, y);
    SDL_SetWindowMinimumSize(window_, 800, 520);
    if (settings_.remember_window && settings_.window_maximized) SDL_MaximizeWindow(window_);

    if (!create_renderer()) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, kAppName, SDL_GetError(), window_);
        return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    static std::string ini_path = platform::path_to_utf8(platform::config_dir() / "layout.ini");
    io.IniFilename = ini_path.c_str();
    // Gamepads drive the game, not the UI, so gamepad navigation stays off.
    io.ConfigWindowsMoveFromTitleBarOnly = true;
    // macOS convention: Cmd acts as the shortcut modifier inside text fields.
    io.ConfigMacOSXBehaviors = platform::current_os() == platform::OS::MacOS;

    ImGui_ImplSDL3_InitForSDLRenderer(window_, renderer_);
    ImGui_ImplSDLRenderer3_Init(renderer_);

    update_ui_scale(true);
    input_.open_all();
    open_audio();

    browser_.native_available = platform::native_dialogs_available();
    browser_.on_native_requested = [this](FileBrowser::Mode m) {
        native_mode_ = m;
        native_job_ = std::async(std::launch::async, [m]() {
            if (m == FileBrowser::Mode::OpenGbRom) return platform::native_open_file("Choose Game Boy ROM", {"gb", "gbc"});
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
    if (scaled_tex_) SDL_DestroyTexture(scaled_tex_);
    for (auto& [k, t] : dbg_textures_) SDL_DestroyTexture(t);
    dbg_textures_.clear();
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    destroy_renderer();
    SDL_DestroyWindow(window_);
    SDL_Quit();
}

// SDL's GPU renderer on a device of our own (Metal on macOS, Vulkan or
// Direct3D 12 on Windows, Vulkan on Linux), so the high-resolution RDP
// renderer can run its compute shaders on the device that shows the UI.
// Without a usable GPU driver any other SDL renderer will do; the RDP then
// renders high resolutions on the CPU.
bool App::create_renderer() {
    if (settings_.video_backend != 1) { // 1 = compatibility (no SDL_GPU)
        gpu_ = gpu::create_device();
        if (gpu_) {
            renderer_ = SDL_CreateGPURenderer(gpu_, window_);
            if (!renderer_) {
                SDL_Log("GPU renderer unavailable: %s", SDL_GetError());
                SDL_DestroyGPUDevice(gpu_);
                gpu_ = nullptr;
            }
        } else {
            SDL_Log("No SDL_GPU device: %s", SDL_GetError());
        }
    }
    if (!renderer_) renderer_ = SDL_CreateRenderer(window_, nullptr);
    if (!renderer_) return false;
    SDL_SetRenderVSync(renderer_, settings_.vsync ? 1 : SDL_RENDERER_VSYNC_DISABLED);
    const char* rn = SDL_GetRendererName(renderer_);
    renderer_name_ = rn ? rn : "SDL";
    if (gpu_) {
        renderer_name_ += std::string(" (") + SDL_GetGPUDeviceDriver(gpu_) + ")";
        gpu_dev_ = std::make_shared<gpu::Device>(gpu_);
    }
    apply_hires_renderer();
    return true;
}

// Internal resolutions render on the GPU when its pipelines work (and
// ORBIT64_HIRES=cpu doesn't ask for the CPU renderer).
void App::apply_hires_renderer() {
    const char* force = std::getenv("ORBIT64_HIRES");
    const bool use_gpu = gpu_dev_ && gpu_dev_->ok() && !(force && std::string(force) == "cpu");
    core_.set_hires_factory(use_gpu ? gpu::make_hires_factory(gpu_dev_) : HiResFactory());
}

void App::destroy_renderer() {
    core_.set_hires_factory(nullptr);
    release_gpu_frames();
    if (stream_tex_) SDL_DestroyTexture(stream_tex_);
    stream_tex_ = nullptr;
    game_tex_ = nullptr;
    gpu_dev_.reset();
    if (renderer_) SDL_DestroyRenderer(renderer_);
    renderer_ = nullptr;
    if (gpu_) SDL_DestroyGPUDevice(gpu_);
    gpu_ = nullptr;
}

void App::release_gpu_frames() {
    frame_ = VideoFrame{};
    for (auto& w : wrapped_) SDL_DestroyTexture(w.tex);
    wrapped_.clear();
}

void App::pump_video() {}

void App::save_settings() {
    if (window_ && !fullscreen_) {
        SDL_WindowFlags f = SDL_GetWindowFlags(window_);
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
    SDL_GetWindowSizeInPixels(window_, &dw, &dh);
    float fb = ww > 0 ? static_cast<float>(dw) / ww : 1.0f;
    float sys = platform::system_ui_scale(SDL_GetDisplayForWindow(window_));
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
        ImGui_ImplSDLRenderer3_DestroyFontsTexture();
        ImGui_ImplSDLRenderer3_CreateFontsTexture();
    }
}

void App::open_audio() {
    close_audio();
    audio_devices_.clear();
    SDL_AudioDeviceID chosen = SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
    int count = 0;
    if (SDL_AudioDeviceID* ids = SDL_GetAudioPlaybackDevices(&count)) {
        for (int i = 0; i < count; ++i) {
            const char* n = SDL_GetAudioDeviceName(ids[i]);
            if (!n) continue;
            audio_devices_.push_back(n);
            if (!settings_.audio_device.empty() && settings_.audio_device == n) chosen = ids[i];
        }
        SDL_free(ids);
    }
    if (!settings_.audio_enabled) return;

    // The device buffer size; SDL treats it as a hint.
    SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES, std::to_string(settings_.buffer_frames).c_str());
    SDL_AudioSpec want{};
    want.freq = settings_.sample_rate;
    want.format = SDL_AUDIO_F32;
    want.channels = 2;
    g_audio_ctx.core = &core_;
    g_audio_ctx.volume = &audio_volume_;
    audio_dev_ = SDL_OpenAudioDeviceStream(chosen, &want, audio_callback, &g_audio_ctx);
    if (!audio_dev_ && chosen != SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK)
        audio_dev_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &want, audio_callback, &g_audio_ctx);
    if (audio_dev_) {
        // Produce samples at the rate the device really runs at, so SDL
        // doesn't resample them a second time (the core's own resampler is
        // better), and pace the emulation to the device's clock.
        SDL_AudioSpec have{};
        int frames = settings_.buffer_frames;
        if (SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(audio_dev_), &have, &frames) && have.freq > 0 &&
            have.freq != want.freq) {
            want.freq = have.freq;
            SDL_SetAudioStreamFormat(audio_dev_, &want, nullptr);
        }
        audio_freq_ = want.freq;
        core_.set_audio_output(static_cast<std::uint32_t>(want.freq), static_cast<std::uint32_t>(std::max(frames, 64)));
        SDL_ResumeAudioStreamDevice(audio_dev_);
    }
}

void App::close_audio() {
    if (audio_dev_) SDL_DestroyAudioStream(audio_dev_);
    audio_dev_ = nullptr;
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

        pump_video(); // keeps the GPU renderer's queue moving even while minimised
        SDL_WindowFlags wflags = SDL_GetWindowFlags(window_);
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
        for (std::string msg; core_.poll_message(msg);) toast(msg, ToastKind::Error);

        // Native file dialog finished?
        if (native_job_ && native_job_->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            auto res = native_job_->get();
            native_job_.reset();
            if (res && !res->empty()) {
                if (native_mode_ == FileBrowser::Mode::OpenRom) launch(*res);
                else if (native_mode_ == FileBrowser::Mode::OpenGbRom) on_gb_rom_picked(*res);
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
        if (hide_cursor) SDL_HideCursor();
        else SDL_ShowCursor();

        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
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
        SDL_SetRenderScale(renderer_, fbs.x, fbs.y);
        SDL_SetRenderDrawColor(renderer_, 10, 11, 14, 255);
        SDL_RenderClear(renderer_);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer_);
        if (!ui_test_dir_.empty()) ui_test_tick();
        SDL_RenderPresent(renderer_);
        if (!settings_.vsync) SDL_Delay(2); // keep the UI thread from spinning at thousands of fps
    }
}

void App::process_event(const SDL_Event& e) {
    // Rebinding capture takes priority over everything else.
    if (hotkey_capture_ >= 0 && e.type == SDL_EVENT_KEY_DOWN) {
        capture_hotkey(e.key);
        return;
    }
    if (input_.capturing() && (e.type == SDL_EVENT_KEY_DOWN || e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN || e.type == SDL_EVENT_GAMEPAD_AXIS_MOTION)) {
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

    ImGui_ImplSDL3_ProcessEvent(&e);
    input_.handle_event(e);

    switch (e.type) {
        case SDL_EVENT_QUIT: request_quit(); break;
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            if (e.window.windowID == SDL_GetWindowID(window_)) request_quit();
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            if (e.window.windowID != SDL_GetWindowID(window_)) break;
            window_focused_ = false;
            // Clicking into another app stops recording a binding, like a click elsewhere in ours.
            hotkey_capture_ = -1;
            input_.cancel_capture();
            if (settings_.pause_on_focus_loss && core_.state() == RunState::Running && !native_job_) {
                core_.pause(true);
                auto_paused_ = true;
            }
            break;
        case SDL_EVENT_WINDOW_FOCUS_GAINED:
            if (e.window.windowID != SDL_GetWindowID(window_)) break;
            window_focused_ = true;
            if (auto_paused_ && core_.state() == RunState::Paused) core_.pause(false);
            auto_paused_ = false;
            break;
        case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_RESIZED:
            if (e.window.windowID == SDL_GetWindowID(window_)) update_ui_scale(false);
            break;
        case SDL_EVENT_MOUSE_MOTION: last_mouse_move_ = ImGui::GetTime(); break;
        case SDL_EVENT_KEY_DOWN:
            if (handle_shortcut(e.key)) break;
            break;
        case SDL_EVENT_DROP_FILE: {
            if (!e.drop.data) break;
            fs::path p = platform::utf8_to_path(e.drop.data);
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

std::string App::hotkey_label(Hotkey h) const {
    for (const KeyCombo& c : settings_.hotkeys[static_cast<int>(h)])
        if (c.key) return key_combo_label(c);
    return "";
}

std::string App::hotkey_hint(Hotkey h) const {
    std::string s;
    for (const KeyCombo& c : settings_.hotkeys[static_cast<int>(h)]) {
        if (!c.key) continue;
        if (!s.empty()) s += " or ";
        s += key_combo_label(c);
    }
    return s;
}

bool App::handle_shortcut(const SDL_KeyboardEvent& k) {
    if (k.repeat || is_modifier_key(static_cast<int>(k.key))) return false;
    const KeyCombo pressed{static_cast<int>(k.key), key_mods_from_sdl(k.mod)};
    const bool modal_open = settings_open_ || about_open_ || confirm_open_ || browser_.is_open() || props_open_ || error_open_;

    for (int h = 0; h < kHotkeyCount; ++h)
        for (const KeyCombo& c : settings_.hotkeys[h])
            if (c.key && c == pressed) return run_hotkey(static_cast<Hotkey>(h), modal_open);

    // Fixed shortcuts: state slots and leaving fullscreen.
    if (pressed.mods == kModPrimary && k.key >= SDLK_1 && k.key <= SDLK_9) {
        if (modal_open) return false;
        select_state_slot(static_cast<int>(k.key - SDLK_0));
        return true;
    }
    if (k.key == SDLK_ESCAPE && fullscreen_ && !modal_open && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId)) {
        set_fullscreen(false);
        return true;
    }
    return false;
}

bool App::run_hotkey(Hotkey h, bool modal_open) {
    switch (h) {
        case Hotkey::OpenRom: if (modal_open) return false; action_open_rom(); return true;
        case Hotkey::AddFolder: if (modal_open) return false; action_add_folder(); return true;
        case Hotkey::ToggleView:
            if (modal_open) return false;
            view_ = view_ == View::Library ? View::Game : View::Library;
            return true;
        case Hotkey::Pause: if (modal_open) return false; toggle_pause(); return true;
        case Hotkey::Stop: if (modal_open) return false; request_stop(); return true;
        case Hotkey::Reset: if (modal_open) return false; reset_game(); return true;
        case Hotkey::SaveState: if (modal_open) return false; save_state(state_slot_); return true;
        case Hotkey::LoadState: if (modal_open) return false; load_state(state_slot_); return true;
        case Hotkey::FastForward: return false; // held, see update_input()
        case Hotkey::Fullscreen: toggle_fullscreen(); return true;
        case Hotkey::Screenshot: take_screenshot(); return true;
        case Hotkey::InfoPanel: settings_.show_info_panel = !settings_.show_info_panel; return true;
        case Hotkey::Settings: if (modal_open) return false; open_settings(SettingsPage::General); return true;
        case Hotkey::Quit: request_quit(); return true;
        default: return false;
    }
}

void App::capture_hotkey(const SDL_KeyboardEvent& k) {
    if (k.repeat || is_modifier_key(static_cast<int>(k.key))) return; // wait for the actual key
    const int h = hotkey_capture_ / kHotkeySlots, slot = hotkey_capture_ % kHotkeySlots;
    hotkey_capture_ = -1;
    const KeyCombo combo{static_cast<int>(k.key), key_mods_from_sdl(k.mod)};
    if (combo.key == SDLK_ESCAPE && combo.mods == 0) return; // cancel
    // One combination does one thing: take it away from any other shortcut.
    for (int o = 0; o < kHotkeyCount; ++o)
        for (int j = 0; j < kHotkeySlots; ++j)
            if ((o != h || j != slot) && settings_.hotkeys[o][j] == combo) {
                settings_.hotkeys[o][j] = {};
                if (o != h) toast(key_combo_label(combo) + " removed from \xE2\x80\x9C" + hotkey_name(static_cast<Hotkey>(o)) + "\xE2\x80\x9D");
            }
    settings_.hotkeys[h][slot] = combo;
    save_settings();
}

void App::update_input() {
    ImGuiIO& io = ImGui::GetIO();
    bool modal_open = settings_open_ || about_open_ || confirm_open_ || browser_.is_open() || props_open_ || error_open_;
    bool keyboard_to_game = view_ == View::Game && window_focused_ && !io.WantTextInput && !modal_open && !input_.capturing();
    for (int p = 0; p < 4; ++p) {
        ControllerSnapshot s = input_.poll(settings_.ports[p], keyboard_to_game);
        if (p == 0 && test_input_) s.buttons |= test_input_(core_.stats().frame);
        core_.set_input(p, s);
        // Rumble Pak: the game's motor drives the gamepad while it runs.
        const PortConfig& pc = settings_.ports[p];
        const bool motor = pc.pak == 2 && core_.state() == RunState::Running && core_.rumble(p);
        input_.rumble(p, pc, motor ? pc.rumble_strength / 100.0f : 0.0f);
    }

    // Fast-forward while its shortcut is held (only when the game has keyboard
    // focus). Extra modifiers are allowed, since games may use them as buttons.
    const bool* keys = SDL_GetKeyboardState(nullptr);
    const int mods = key_mods_from_sdl(SDL_GetModState());
    bool ff = false;
    for (const KeyCombo& c : settings_.hotkeys[static_cast<int>(Hotkey::FastForward)]) {
        if (!c.key) continue;
        const SDL_Scancode sc = SDL_GetScancodeFromKey(static_cast<SDL_Keycode>(c.key), nullptr);
        if (sc != SDL_SCANCODE_UNKNOWN && keys[sc] && (mods & c.mods) == c.mods) ff = true;
    }
    core_.set_fast_forward(keyboard_to_game && hotkey_capture_ < 0 && ff);
    core_.set_ff_multiplier(settings_.ff_speed);
    core_.set_limit_speed(settings_.limit_speed);
    core_.set_fps_limit(settings_.fps_limit);
    core_.set_internal_scale(settings_.internal_scale);
    core_.set_expansion_pak(settings_.expansion_pak);
}

void App::update_game_texture() {
    if (!core_.fetch_frame(frame_, frame_serial_)) return;
    if (frame_.empty()) {
        has_frame_ = false;
        return;
    }
    const int w = frame_.w, h = frame_.h;
    SDL_Texture* tex = frame_.gpu ? gpu_frame_texture(frame_.gpu) : nullptr;
    if (!tex) {
        std::vector<std::uint32_t> read;
        const std::vector<std::uint32_t>* px = &frame_.pixels;
        if (frame_.gpu) {
            frame_.gpu->read(read);
            px = &read;
        }
        if (px->size() != static_cast<size_t>(w) * h) {
            has_frame_ = false;
            return;
        }
        if (!stream_tex_ || w != stream_w_ || h != stream_h_) {
            if (stream_tex_) SDL_DestroyTexture(stream_tex_);
            stream_tex_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, w, h);
            stream_w_ = w;
            stream_h_ = h;
            game_tex_filter_ = -1;
        }
        if (!stream_tex_) {
            has_frame_ = false;
            return;
        }
        SDL_UpdateTexture(stream_tex_, nullptr, px->data(), w * static_cast<int>(sizeof(std::uint32_t)));
        tex = stream_tex_;
    }
    if (tex != game_tex_) game_tex_filter_ = -1;
    game_tex_ = tex;
    game_tex_w_ = w;
    game_tex_h_ = h;
    game_scale_ = std::max(1, frame_.scale);
    if (game_tex_filter_ != settings_.filter) {
        SDL_SetTextureScaleMode(game_tex_, settings_.filter == 0 ? SDL_SCALEMODE_NEAREST : SDL_SCALEMODE_LINEAR);
        game_tex_filter_ = settings_.filter;
    }
    has_frame_ = true;
}

SDL_Texture* App::gpu_frame_texture(const std::shared_ptr<GpuImage>& image) {
    // Wrappers of images the renderer has let go of are stale. (Nothing
    // queued for drawing uses them: the frame shown last time is frame_.)
    wrapped_.erase(std::remove_if(wrapped_.begin(), wrapped_.end(),
                                  [](const WrappedImage& w) {
                                      if (!w.image.expired()) return false;
                                      SDL_DestroyTexture(w.tex);
                                      return true;
                                  }),
                   wrapped_.end());
    if (!gpu_) return nullptr;
    for (const auto& w : wrapped_)
        if (w.image.lock() == image) return w.tex;
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetPointerProperty(props, SDL_PROP_TEXTURE_CREATE_GPU_TEXTURE_POINTER, image->texture());
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_FORMAT_NUMBER, SDL_PIXELFORMAT_ABGR8888); // R8G8B8A8 bytes
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_ACCESS_NUMBER, SDL_TEXTUREACCESS_STATIC);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_WIDTH_NUMBER, image->w);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_HEIGHT_NUMBER, image->h);
    SDL_Texture* tex = SDL_CreateTextureWithProperties(renderer_, props);
    SDL_DestroyProperties(props);
    if (!tex) {
        SDL_Log("Can't show GPU frames directly: %s", SDL_GetError());
        return nullptr;
    }
    SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_NONE);
    wrapped_.push_back({image, tex});
    return tex;
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

void App::action_choose_gb_rom(int port) {
    if (native_job_) return;
    gb_pick_port_ = port;
    if (settings_.use_native_dialogs && browser_.native_available) {
        browser_.on_native_requested(FileBrowser::Mode::OpenGbRom);
        return;
    }
    const std::string& cur = settings_.ports[port].gb_rom;
    browser_.library_dirs.clear();
    for (const auto& d : settings_.rom_dirs) browser_.library_dirs.push_back(platform::utf8_to_path(d));
    browser_.open(FileBrowser::Mode::OpenGbRom, !cur.empty() ? platform::utf8_to_path(cur).parent_path()
                                              : settings_.rom_dirs.empty() ? platform::home_dir()
                                                                           : platform::utf8_to_path(settings_.rom_dirs.front()));
}

void App::on_gb_rom_picked(const fs::path& rom) {
    settings_.ports[gb_pick_port_ & 3].gb_rom = platform::path_to_utf8(rom);
    save_settings();
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
        core_.set_rsp_mode(settings_.rsp_mode);
        core_.set_rdp_exact(settings_.rdp_exact);
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
        game_tex_ = nullptr;
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
    game_tex_ = nullptr;
    release_gpu_frames(); // the stopped game's frames in video memory
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
    if (on) {
        save_settings(); // remember windowed geometry first
        // Exclusive fullscreen uses the desktop's own mode; "borderless" none.
        const SDL_DisplayMode* dm = nullptr;
        if (settings_.fullscreen_mode == 1) dm = SDL_GetDesktopDisplayMode(SDL_GetDisplayForWindow(window_));
        SDL_SetWindowFullscreenMode(window_, dm);
    }
    if (SDL_SetWindowFullscreen(window_, on)) {
        fullscreen_ = on;
        chrome_reveal_ = 1.0f;
        last_mouse_move_ = ImGui::GetTime();
    }
}

void App::toggle_fullscreen() { set_fullscreen(!fullscreen_); }

static bool write_bmp(const fs::path& path, const std::vector<std::uint32_t>& px, int w, int h) {
    SDL_Surface* s = SDL_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_ARGB8888, const_cast<std::uint32_t*>(px.data()), w * 4);
    if (!s) return false;
    bool ok = SDL_SaveBMP(s, platform::path_to_utf8(path).c_str()); // SDL expects UTF-8 on all platforms
    SDL_DestroySurface(s);
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
            if (tex) SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_LINEAR);
            SDL_DestroySurface(s);
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
    profiles_scanned_ = false;
}

} // namespace ui
