// Window chrome: root layout, menu bar, toolbar and status bar.

#include "app.hpp"
#include "platform.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>

namespace ui {

namespace fs = std::filesystem;

static const char* kUcodeNames[] = {"Auto", "Fast3D", "F3DEX", "F3DEX2", "S2DEX", "S2DEX2", "F3DEX (GoldenEye)", "F3DPD", "F3DDKR", "F3DJFG", "F3DWRUS"};
static const char* kAbiShort[] = {"ABI", "n_audio", "Nintendo Audio", "Nintendo Audio", "Nintendo Audio", "MusyX"};
static const char* kAbiLong[] = {
    "Standard libultra audio microcode (ABI 1)",
    "n_audio microcode \xE2\x80\x94 Rare and many third-party titles",
    "Nintendo EAD audio (Mario Kart 64 family)",
    "Nintendo EAD audio (Star Fox 64 / F-Zero X family)",
    "Nintendo EAD audio (Zelda family)",
    "Factor 5 MusyX synthesizer",
};

// ---------------------------------------------------------------------------
// Root layout
// ---------------------------------------------------------------------------

void App::draw_root() {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGuiIO& io = ImGui::GetIO();

    // In fullscreen gameplay the chrome hides and slides back in when the
    // mouse approaches the top edge (or the game is paused).
    const bool immersive = fullscreen_ && view_ == View::Game && core_.loaded();
    float reveal_target = 1.0f;
    if (immersive) {
        bool near_top = io.MousePos.y >= 0 && io.MousePos.y < dp(70) && ImGui::GetTime() - last_mouse_move_ < 2.5;
        bool keep = core_.state() != RunState::Running || ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId);
        reveal_target = (near_top || keep) ? 1.0f : 0.0f;
    }
    chrome_reveal_ = anim(ImHashStr("##chrome_reveal"), reveal_target, 12.0f);

    // Esc dismisses menus, dropdowns and context menus (modals handle Esc themselves).
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) &&
        ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel))
        ImGui::ClosePopupsExceptModals();

    ImGui::SetNextWindowPos(vp->Pos);
    ImGui::SetNextWindowSize(vp->Size);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, g_pal.bg0);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                             ImGuiWindowFlags_NoScrollWithMouse;
    if (!immersive) flags |= ImGuiWindowFlags_MenuBar;
    ImGui::Begin("##root", nullptr, flags);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);

    if (!immersive) draw_menubar();

    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImVec2 avail(vp->Pos.x + vp->Size.x - origin.x, vp->Pos.y + vp->Size.y - origin.y);
    const float toolbar_h = dp(54);
    const bool status = immersive ? settings_.status_bar_in_fullscreen : settings_.show_status_bar;
    const float status_h = status ? dp(28) : 0.0f;

    if (immersive) {
        draw_game_view(origin, ImVec2(avail.x, avail.y - status_h));
        if (chrome_reveal_ > 0.01f) {
            // Separate child so the toolbar layers above the game viewport child.
            float y = origin.y - toolbar_h * (1.0f - chrome_reveal_);
            ImGui::SetCursorScreenPos(ImVec2(origin.x, y));
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, chrome_reveal_);
            ImGui::BeginChild("##overlay_toolbar", ImVec2(avail.x, toolbar_h), ImGuiChildFlags_None,
                              ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            draw_toolbar(ImVec2(origin.x, y), ImVec2(avail.x, toolbar_h));
            ImGui::EndChild();
            ImGui::PopStyleVar();
        }
    } else {
        draw_toolbar(origin, ImVec2(avail.x, toolbar_h));
        ImVec2 cpos(origin.x, origin.y + toolbar_h);
        ImVec2 csize(avail.x, avail.y - toolbar_h - status_h);
        if (view_ == View::Library) draw_library(cpos, csize);
        else draw_game_view(cpos, csize);
    }
    if (status) draw_statusbar(ImVec2(origin.x, origin.y + avail.y - status_h), ImVec2(avail.x, status_h));

    ImGui::End();

    draw_debug_windows();
    draw_dialogs();
    draw_toasts(ImVec2(vp->Pos.x + vp->Size.x - dp(20), vp->Pos.y + vp->Size.y - status_h - dp(16)));
}

// ---------------------------------------------------------------------------
// Menu bar
// ---------------------------------------------------------------------------

// Items of the "Internal Resolution" submenus (menu bar and game screen).
void App::internal_resolution_menu() {
    const int fw = frame_native_w(), fh = frame_native_h();
    for (int k = 1; k <= 8; ++k) {
        char label[48];
        if (k == 1) std::snprintf(label, sizeof label, "Native (%d\xC3\x97%d)", fw, fh);
        else std::snprintf(label, sizeof label, "%d\xC3\x97 (%d\xC3\x97%d)", k, fw * k, fh * k);
        if (ImGui::MenuItem(label, nullptr, settings_.internal_scale == k)) settings_.internal_scale = k;
    }
}

void App::draw_menubar() {
    using platform::shortcut_label;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, dp(10, 7));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, dp(10, 8));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, dp(8));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(8, 8));
    ImGui::PushStyleColor(ImGuiCol_MenuBarBg, g_pal.bg1);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, g_pal.bg2);
    ImGui::PushStyleColor(ImGuiCol_Border, g_pal.border_strong);
    ImGui::PushStyleColor(ImGuiCol_Header, g_pal.bg4);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, g_pal.bg4);
    ImGui::PushStyleColor(ImGuiCol_Text, g_pal.text);
    if (!ImGui::BeginMenuBar()) {
        ImGui::PopStyleColor(6);
        ImGui::PopStyleVar(4);
        return;
    }
    const bool loaded = core_.loaded();
    const bool running = core_.state() == RunState::Running;

    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open ROM\xE2\x80\xA6", shortcut_label(true, false, false, "O").c_str())) action_open_rom();
        if (ImGui::MenuItem("Add ROM Folder\xE2\x80\xA6", shortcut_label(true, true, false, "O").c_str())) action_add_folder();
        auto recent = library_.recent(10);
        if (ImGui::BeginMenu("Open Recent", !recent.empty())) {
            for (const GameEntry* g : recent) {
                std::string key = g->key;
                fs::path path = g->rom.path;
                if (ImGui::MenuItem(g->rom.display_title.c_str(), g->rom.region_short.c_str())) pending_launch_ = path;
                if (ImGui::IsItemHovered()) tooltip(key.c_str());
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Clear Recent Games")) library_.clear_recent();
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Show Screenshots Folder")) platform::reveal_in_file_manager(platform::screenshots_dir());
        if (ImGui::MenuItem("Show Save States Folder")) reveal_states_folder();
        if (ImGui::MenuItem("Show Configuration Folder")) platform::reveal_in_file_manager(platform::config_dir());
        ImGui::Separator();
        if (ImGui::MenuItem("Quit", platform::current_os() == platform::OS::MacOS ? "Cmd+Q" : "Ctrl+Q")) request_quit();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Emulation")) {
        if (ImGui::MenuItem(running ? "Pause" : "Resume", shortcut_label(true, false, false, "P").c_str(), false, loaded)) toggle_pause();
        if (ImGui::MenuItem("Stop", shortcut_label(true, false, false, ".").c_str(), false, loaded)) request_stop();
        if (ImGui::MenuItem("Reset", shortcut_label(true, false, false, "R").c_str(), false, loaded)) reset_game();
        ImGui::Separator();
        draw_state_menu();
        ImGui::Separator();
        if (ImGui::BeginMenu("Fast-Forward Speed")) {
            const int speeds[] = {2, 3, 4, 8, 0};
            const char* names[] = {"2\xC3\x97", "3\xC3\x97", "4\xC3\x97", "8\xC3\x97", "Unlimited"};
            for (int i = 0; i < 5; ++i)
                if (ImGui::MenuItem(names[i], nullptr, settings_.ff_speed == speeds[i])) settings_.ff_speed = speeds[i];
            ImGui::EndMenu();
        }
        ImGui::MenuItem("Limit Speed", nullptr, &settings_.limit_speed);
        ImGui::Separator();
        if (ImGui::MenuItem("Take Screenshot", "F12", false, loaded)) take_screenshot();
        ImGui::Separator();
        if (ImGui::MenuItem("Emulation Settings\xE2\x80\xA6")) open_settings(SettingsPage::Emulation);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Game Library", shortcut_label(true, false, false, "L").c_str(), view_ == View::Library)) view_ = View::Library;
        if (ImGui::MenuItem("Game Screen", nullptr, view_ == View::Game)) view_ = View::Game;
        ImGui::Separator();
        if (ImGui::MenuItem("Fullscreen", platform::current_os() == platform::OS::MacOS ? "Cmd+Ctrl+F" : "F11", fullscreen_)) toggle_fullscreen();
        ImGui::MenuItem("Game Info Panel", shortcut_label(true, false, false, "I").c_str(), &settings_.show_info_panel);
        ImGui::MenuItem("Status Bar", nullptr, &settings_.show_status_bar);
        ImGui::MenuItem("Library Sidebar", nullptr, &library_sidebar_);
        ImGui::Separator();
        if (ImGui::BeginMenu("Internal Resolution")) {
            internal_resolution_menu();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Aspect Ratio")) {
            const char* names[] = {"4:3 (Original)", "16:9 (Widescreen)", "Stretch to Window", "Native Frame Size"};
            for (int i = 0; i < 4; ++i)
                if (ImGui::MenuItem(names[i], nullptr, settings_.aspect == i)) settings_.aspect = i;
            ImGui::Separator();
            ImGui::MenuItem("Integer Scaling", nullptr, &settings_.integer_scale);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Texture Filter")) {
            if (ImGui::MenuItem("Nearest (Sharp Pixels)", nullptr, settings_.filter == 0)) settings_.filter = 0;
            if (ImGui::MenuItem("Bilinear (Smooth)", nullptr, settings_.filter == 1)) settings_.filter = 1;
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Interface Scale")) {
            const int opts[] = {0, 90, 100, 110, 125, 150, 175, 200};
            for (int o : opts) {
                char label[32];
                if (o == 0) std::snprintf(label, sizeof label, "Automatic");
                else std::snprintf(label, sizeof label, "%d%%", o);
                if (ImGui::MenuItem(label, nullptr, settings_.ui_scale_pct == o)) settings_.ui_scale_pct = o;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Debug")) {
        draw_debug_menu(false);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Settings")) {
        struct { const char* name; SettingsPage page; } pages[] = {
            {"General", SettingsPage::General},     {"Graphics", SettingsPage::Graphics},
            {"Audio", SettingsPage::Audio},         {"Controllers", SettingsPage::Controller},
            {"Emulation", SettingsPage::Emulation}, {"Library", SettingsPage::Library},
        };
        for (auto& p : pages) {
            std::string label = std::string(p.name) + "\xE2\x80\xA6";
            if (ImGui::MenuItem(label.c_str(), p.page == SettingsPage::General ? shortcut_label(true, false, false, ",").c_str() : nullptr))
                open_settings(p.page);
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("Keyboard Shortcuts")) open_settings(SettingsPage::Shortcuts);
        if (ImGui::MenuItem("Controller Mapping")) open_settings(SettingsPage::Controller);
        ImGui::Separator();
        if (ImGui::MenuItem("About Orbit64")) about_open_ = true;
        ImGui::EndMenu();
    }

    // Right side: live status pill (visible even when the status bar is hidden).
    {
        RunState st = core_.state();
        const char* label = st == RunState::Running ? "RUNNING" : st == RunState::Paused ? "PAUSED" : "IDLE";
        ImVec4 c = st == RunState::Running ? g_pal.success : st == RunState::Paused ? g_pal.warning : g_pal.text_faint;
        ImFont* f = g_fonts.small_bold;
        float tw = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, label).x;
        float x = ImGui::GetWindowPos().x + ImGui::GetWindowWidth() - tw - dp(38);
        float y = ImGui::GetWindowPos().y + (ImGui::GetFrameHeight() - font_px(f)) * 0.5f;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        float pulse = st == RunState::Running ? 0.6f + 0.4f * std::sin(static_cast<float>(ImGui::GetTime()) * 3.0f) : 1.0f;
        dl->AddCircleFilled(ImVec2(x - dp(2), y + font_px(f) * 0.5f), dp(3.5f), col(c, pulse), 12);
        dl->AddText(f, font_px(f), ImVec2(x + dp(8), y), col(c), label);
    }

    ImGui::EndMenuBar();
    ImGui::PopStyleColor(6);
    ImGui::PopStyleVar(4);
}

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------

void App::draw_toolbar(ImVec2 pos, ImVec2 size) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), col(g_pal.bg1));
    dl->AddLine(ImVec2(pos.x, pos.y + size.y - 1), ImVec2(pos.x + size.x, pos.y + size.y - 1), col(g_pal.border));

    const bool loaded = core_.loaded();
    const RunState st = core_.state();
    const bool compact = size.x < dp(1060);
    const bool tiny = size.x < dp(900);
    const float cy = pos.y + size.y * 0.5f;
    float x = pos.x + dp(16);

    // Brand
    draw_logo(dl, ImVec2(x + dp(14), cy), dp(26), col(g_pal.accent), col(g_pal.accent2));
    x += dp(34);
    if (!tiny) {
        ImFont* tf = g_fonts.title;
        dl->AddText(tf, font_px(tf) * 0.9f, ImVec2(x, cy - font_px(tf) * 0.47f), col(g_pal.text), kAppName);
        x += tf->CalcTextSizeA(font_px(tf) * 0.9f, FLT_MAX, 0, kAppName).x + dp(20);
    } else {
        x += dp(10);
    }

    // Navigation
    const char* nav_items[] = {"Library", "Game"};
    const Icon nav_icons[] = {Icon::Grid, Icon::Monitor};
    int nav = view_ == View::Library ? 0 : 1;
    ImGui::SetCursorScreenPos(ImVec2(x, cy - dp(15)));
    if (segmented("nav", nav_items, 2, &nav, compact ? dp(180) : dp(210), nav_icons)) view_ = nav == 0 ? View::Library : View::Game;
    float nav_end = x + (compact ? dp(180) : dp(210));

    // Transport controls, centred in the window when there's room.
    const float bs = dp(36);
    const float open_w = compact ? bs : dp(124);
    const float group_w = open_w + dp(14) + bs * 4 + dp(6) * 3;
    float gx = pos.x + (size.x - group_w) * 0.5f;
    gx = std::max(gx, nav_end + dp(20));
    ImGui::SetCursorScreenPos(ImVec2(gx, cy - bs * 0.5f));
    if (compact) {
        if (icon_button("open", Icon::FolderOpen, bs, "Open ROM", false, true, ButtonKind::Subtle)) action_open_rom();
    } else if (button("Open ROM", Icon::FolderOpen, ButtonKind::Subtle, open_w, true, bs)) {
        action_open_rom();
    }
    float sep_x = gx + open_w + dp(7);
    dl->AddLine(ImVec2(sep_x, cy - dp(12)), ImVec2(sep_x, cy + dp(12)), col(g_pal.border_strong));
    float tx = gx + open_w + dp(14);

    ImGui::SetCursorScreenPos(ImVec2(tx, cy - bs * 0.5f));
    bool can_play = loaded || !library_.recent(1).empty();
    const char* play_tip = st == RunState::Running ? "Pause" : (loaded ? "Resume" : "Start (last played game)");
    if (icon_button("play", st == RunState::Running ? Icon::Pause : Icon::Play, bs, play_tip, false, can_play,
                    ButtonKind::Primary)) {
        if (loaded) toggle_pause();
        else if (auto r = library_.recent(1); !r.empty()) pending_launch_ = r.front()->rom.path;
    }
    ImGui::SetCursorScreenPos(ImVec2(tx + (bs + dp(6)), cy - bs * 0.5f));
    if (icon_button("stop", Icon::Stop, bs, "Stop", false, loaded, ButtonKind::Subtle)) request_stop();
    ImGui::SetCursorScreenPos(ImVec2(tx + (bs + dp(6)) * 2, cy - bs * 0.5f));
    if (icon_button("reset", Icon::Reset, bs, "Reset", false, loaded, ButtonKind::Subtle)) reset_game();
    ImGui::SetCursorScreenPos(ImVec2(tx + (bs + dp(6)) * 3, cy - bs * 0.5f));
    {
        bool ff = core_.fast_forward() || core_.turbo();
        std::string tip = "Fast forward \xE2\x80\x94 hold Tab (" +
                          (settings_.ff_speed == 0 ? std::string("unlimited") : std::to_string(settings_.ff_speed) + "\xC3\x97") + ")";
        if (icon_button("ff", Icon::FastForward, bs, tip.c_str(), ff, loaded, ButtonKind::Subtle)) {
            // Clicking cycles the fast-forward multiplier.
            const int cycle[] = {2, 3, 4, 8, 0};
            int idx = 0;
            for (int i = 0; i < 5; ++i) if (cycle[i] == settings_.ff_speed) idx = i;
            settings_.ff_speed = cycle[(idx + 1) % 5];
            toast(settings_.ff_speed == 0 ? "Fast-forward: unlimited" : "Fast-forward: " + std::to_string(settings_.ff_speed) + "\xC3\x97");
        }
    }

    // Right-hand actions
    float rx = pos.x + size.x - dp(16);
    const float is = dp(34);
    auto right_button = [&](const char* id, Icon icon, const char* tip, bool active, bool enabled) {
        rx -= is;
        ImGui::SetCursorScreenPos(ImVec2(rx, cy - is * 0.5f));
        bool r = icon_button(id, icon, is, tip, active, enabled);
        rx -= dp(4);
        return r;
    };
    if (right_button("settings", Icon::Settings, "Settings", settings_open_, true)) open_settings(SettingsPage::General);
    {
        bool any = std::any_of(std::begin(dbg_.open), std::end(dbg_.open), [](bool b) { return b; });
        if (right_button("debug", Icon::Bug, "Debug & memory tools", any, true)) ImGui::OpenPopup("##debug_pop");
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y + dp(6)), ImGuiCond_Always, ImVec2(1, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(8, 8));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, dp(8, 4));
        ImGui::PushStyleColor(ImGuiCol_PopupBg, g_pal.bg2);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, g_pal.bg4);
        if (ImGui::BeginPopup("##debug_pop")) {
            draw_debug_menu(true);
            ImGui::EndPopup();
        }
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(2);
    }
    if (right_button("fullscreen", fullscreen_ ? Icon::ExitFullscreen : Icon::Fullscreen, fullscreen_ ? "Exit fullscreen" : "Fullscreen", false, true))
        toggle_fullscreen();
    if (right_button("shot", Icon::Camera, "Screenshot (F12)", false, loaded)) take_screenshot();
    if (!tiny && view_ == View::Game &&
        right_button("info", Icon::Sidebar, "Game info panel", settings_.show_info_panel, true))
        settings_.show_info_panel = !settings_.show_info_panel;

    // Volume with popover slider.
    rx -= dp(6);
    Icon vol_icon = (settings_.volume == 0 || !settings_.audio_enabled) ? Icon::VolumeMute : Icon::Volume;
    if (right_button("volume", vol_icon, nullptr, false, true)) ImGui::OpenPopup("##volume_pop");
    if (ImGui::IsItemHovered() && !ImGui::IsPopupOpen("##volume_pop")) {
        char tip[48];
        std::snprintf(tip, sizeof tip, "Volume %d%%", settings_.volume);
        tooltip(tip);
    }
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y + dp(6)), ImGuiCond_Always, ImVec2(1, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(14, 12));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, g_pal.bg2);
    if (ImGui::BeginPopup("##volume_pop")) {
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::TextColored(g_pal.text_dim, "VOLUME");
        ImGui::PopFont();
        slider_int("vol", &settings_.volume, 0, 100, "%d%%", dp(210));
        // Live output meter
        ImVec2 p = ImGui::GetCursorScreenPos();
        float w = dp(210), h = dp(4);
        float peak = std::min(1.0f, core_.audio_peak());
        ImDrawList* pdl = ImGui::GetWindowDrawList();
        pdl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(g_pal.bg4), h);
        pdl->AddRectFilled(p, ImVec2(p.x + w * peak, p.y + h), col(peak > 0.9f ? g_pal.warning : g_pal.success), h);
        ImGui::Dummy(ImVec2(w, h + dp(6)));
        bool mute = !settings_.audio_enabled;
        if (toggle("mute", &mute)) {
            settings_.audio_enabled = !mute;
            open_audio();
        }
        ImGui::SameLine();
        ImGui::TextColored(g_pal.text_dim, "Mute all audio");
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();

    // Fast-forward indicator next to the transport group.
    if (core_.fast_forward() || core_.turbo()) {
        char ffl[16];
        std::snprintf(ffl, sizeof ffl, "%s", settings_.ff_speed == 0 ? "MAX" : (std::to_string(settings_.ff_speed) + "\xC3\x97").c_str());
        badge_at(dl, ImVec2(tx + (bs + dp(6)) * 4 + dp(4), cy - dp(9)), ffl, g_pal.warning, false);
    }
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

namespace {
struct Segment {
    std::string label;  // faint prefix, e.g. "GPU"
    std::string value;  // bright mono value
    ImVec4 value_col;
    ImVec4 dot;         // w == 0: no dot
    std::string tip;
    int priority;       // lower = more important (kept longer when narrow)
    float width = 0;
    bool visible = true;
};
} // namespace

void App::draw_statusbar(ImVec2 pos, ImVec2 size) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), col(g_pal.bg1));
    dl->AddLine(pos, ImVec2(pos.x + size.x, pos.y), col(g_pal.border));

    const RunState st = core_.state();
    const bool loaded = st != RunState::Stopped;
    CoreStats s = core_.stats();
    ImFont* lf = g_fonts.small;
    ImFont* vf = g_fonts.mono;
    const float pad = dp(12);
    const float cy = pos.y + size.y * 0.5f;
    const ImVec4 dim = g_pal.text_dim;

    // --- state pill (always visible)
    const char* st_label = st == RunState::Running ? "Running" : st == RunState::Paused ? "Paused" : "Stopped";
    ImVec4 st_col = st == RunState::Running ? g_pal.success : st == RunState::Paused ? g_pal.warning : g_pal.text_faint;
    float x = pos.x + pad;
    {
        ImVec2 ts = g_fonts.small_bold->CalcTextSizeA(font_px(g_fonts.small_bold), FLT_MAX, 0, st_label);
        ImVec2 mn(x, cy - dp(9)), mx(x + ts.x + dp(26), cy + dp(9));
        dl->AddRectFilled(mn, mx, col(st_col, 0.14f), dp(9));
        dl->AddCircleFilled(ImVec2(x + dp(10), cy), dp(3.5f), col(st_col), 12);
        dl->AddText(g_fonts.small_bold, font_px(g_fonts.small_bold), ImVec2(x + dp(18), cy - ts.y * 0.5f), col(st_col), st_label);
        ImGui::SetCursorScreenPos(mn);
        ImGui::InvisibleButton("##st_pill", ImVec2(mx.x - mn.x, mx.y - mn.y));
        if (ImGui::IsItemHovered()) tooltip(st == RunState::Stopped ? "Emulation: Stopped" : st == RunState::Running ? "Emulation: Running" : "Emulation: Paused");
        x = mx.x + dp(12);
    }

    // --- technical segments
    char buf[64];
    std::vector<Segment> segs;
    {
        float fps = loaded ? s.fps : 0.0f;
        ImVec4 fc = !loaded ? dim : (s.speed_pct >= 95 ? g_pal.success : s.speed_pct >= 70 ? g_pal.warning : g_pal.danger);
        std::snprintf(buf, sizeof buf, "%.0f", fps);
        char tip[160];
        std::snprintf(tip, sizeof tip, "Emulated frames per second\nSpeed: %.0f%% of %d Hz\nHost time per frame: %.2f ms",
                      loaded ? s.speed_pct : 0.0f, s.vi_hz, loaded ? s.frame_ms : 0.0f);
        segs.push_back({"FPS", buf, fc, {}, tip, 0});
    }
    std::snprintf(buf, sizeof buf, "%llu", static_cast<unsigned long long>(loaded ? s.frame : 0));
    segs.push_back({"Frame", buf, g_pal.text, {}, "Frames emulated since power-on / reset", 5});
    {
        std::string u = (loaded && s.ucode >= 0 && s.ucode < 11) ? kUcodeNames[s.ucode] : "Unknown";
        std::string tip = "Graphics microcode detected from the game's display lists";
        if (settings_.ucode_override > 0) tip += "\nOverride active in Settings \xE2\x80\xBA Emulation";
        segs.push_back({"GPU", u, loaded && s.ucode >= 0 ? g_pal.text : dim, {}, tip, 2});
    }
    {
        bool known = loaded && s.audio_abi >= 0 && s.audio_abi < 6;
        segs.push_back({"Audio", known ? kAbiShort[s.audio_abi] : "Unknown", known ? g_pal.text : dim, {},
                        known ? kAbiLong[s.audio_abi] : "No audio task has been submitted yet", 3});
    }
    segs.push_back({"RSP", loaded && s.rsp_active ? "Active" : "Idle", loaded && s.rsp_active ? g_pal.text : dim,
                    loaded && s.rsp_active ? g_pal.success : g_pal.text_faint,
                    "Reality Signal Processor \xE2\x80\x94 active when graphics/audio tasks ran recently", 4});
    segs.push_back({"RDP", loaded && s.rdp_active ? "Active" : "Idle", loaded && s.rdp_active ? g_pal.text : dim,
                    loaded && s.rdp_active ? g_pal.success : g_pal.text_faint,
                    "Reality Display Processor \xE2\x80\x94 active when display lists were rasterised recently", 4});
    std::snprintf(buf, sizeof buf, "%d Hz", loaded ? s.vi_hz : 60);
    segs.push_back({"VI", buf, loaded ? g_pal.text : dim, {}, s.vi_hz == 50 ? "Video Interface refresh (PAL)" : "Video Interface refresh (NTSC / MPAL)", 6});
    if (loaded && s.ai_rate > 0) std::snprintf(buf, sizeof buf, "%.1f kHz", s.ai_rate / 1000.0);
    else std::snprintf(buf, sizeof buf, "\xE2\x80\x94");
    {
        char tip[128];
        std::snprintf(tip, sizeof tip, "Audio Interface DAC rate programmed by the game\nResampled to the %s output device",
                      audio_dev_ ? "system" : "(disabled)");
        segs.push_back({"AI", buf, loaded && s.ai_rate ? g_pal.text : dim, {}, tip, 6});
    }
    {
        int t = static_cast<int>(loaded ? s.uptime_s : 0);
        std::snprintf(buf, sizeof buf, "%02d:%02d:%02d", t / 3600, (t / 60) % 60, t % 60);
        segs.push_back({"Uptime", buf, loaded ? g_pal.text : dim, {}, "Emulated time in this session (excludes pauses)", 7});
    }

    const float sep = dp(22);
    auto measure = [&](Segment& g) {
        float w = lf->CalcTextSizeA(font_px(lf), FLT_MAX, 0, g.label.c_str()).x + dp(6) +
                  vf->CalcTextSizeA(font_px(vf), FLT_MAX, 0, g.value.c_str()).x;
        if (g.dot.w > 0) w += dp(12);
        g.width = w;
    };
    for (auto& g : segs) measure(g);

    // Drop least important segments until everything fits next to a
    // reasonably sized game title.
    const float min_title = dp(180);
    auto total = [&]() {
        float t = 0;
        for (auto& g : segs) if (g.visible) t += g.width + sep;
        return t;
    };
    float right_limit = pos.x + size.x - pad;
    for (int prio = 7; prio >= 1 && (x + min_title + total() > right_limit); --prio)
        for (auto& g : segs) if (g.priority == prio) g.visible = false;

    // Draw right-aligned segments
    float rx = right_limit - total() + sep;
    float seg_start = rx - sep;
    for (auto& g : segs) {
        if (!g.visible) continue;
        ImVec2 mn(rx - dp(6), pos.y + dp(3)), mx(rx + g.width + dp(6), pos.y + size.y - dp(3));
        ImGui::SetCursorScreenPos(mn);
        ImGui::PushID(g.label.c_str());
        ImGui::InvisibleButton("##seg", ImVec2(mx.x - mn.x, mx.y - mn.y));
        bool hov = ImGui::IsItemHovered();
        ImGui::PopID();
        if (hov) {
            dl->AddRectFilled(mn, mx, col(g_pal.bg3), dp(5));
            tooltip(g.tip.c_str());
        }
        float gx = rx;
        dl->AddText(lf, font_px(lf), ImVec2(gx, cy - font_px(lf) * 0.5f), col(g_pal.text_faint), g.label.c_str());
        gx += lf->CalcTextSizeA(font_px(lf), FLT_MAX, 0, g.label.c_str()).x + dp(6);
        if (g.dot.w > 0) {
            dl->AddCircleFilled(ImVec2(gx + dp(3), cy), dp(3), col(g.dot), 12);
            gx += dp(12);
        }
        dl->AddText(vf, font_px(vf), ImVec2(gx, cy - font_px(vf) * 0.5f), col(g.value_col), g.value.c_str());
        rx += g.width + sep;
        float lx = rx - sep * 0.5f;
        if (rx < right_limit) dl->AddLine(ImVec2(lx, cy - dp(7)), ImVec2(lx, cy + dp(7)), col(g_pal.border_strong));
    }

    // Game / ROM segment fills the remaining space.
    float title_w = seg_start - x - dp(12);
    if (title_w > dp(40)) {
        std::string text;
        if (loaded) {
            text = current_rom_.display_title + "   ROM: " + current_rom_.file_name;
        } else {
            text = "No ROM loaded";
        }
        if (loaded) {
            ImFont* bf = g_fonts.small_bold;
            float tw = bf->CalcTextSizeA(font_px(bf), FLT_MAX, 0, current_rom_.display_title.c_str()).x;
            text_ellipsis(dl, bf, ImVec2(x, cy - font_px(bf) * 0.5f), title_w, col(g_pal.text), current_rom_.display_title.c_str());
            float rem = title_w - tw - dp(14);
            if (rem > dp(60)) {
                float rx2 = x + tw + dp(14);
                dl->AddText(lf, font_px(lf), ImVec2(rx2, cy - font_px(lf) * 0.5f), col(g_pal.text_faint), "ROM");
                float lw = lf->CalcTextSizeA(font_px(lf), FLT_MAX, 0, "ROM").x + dp(6);
                text_ellipsis(dl, vf, ImVec2(rx2 + lw, cy - font_px(vf) * 0.5f), rem - lw, col(g_pal.text_dim), current_rom_.file_name.c_str());
            }
            ImGui::SetCursorScreenPos(ImVec2(x, pos.y));
            ImGui::InvisibleButton("##romseg", ImVec2(title_w, size.y));
            if (ImGui::IsItemHovered()) {
                std::string tip = current_rom_.display_title + "\n" + platform::path_to_utf8(current_rom_.path) +
                                  "\nInternal name: " + s.cart_title + "\nCIC: " + s.cic + "  \xC2\xB7  Save: " + s.save_type;
                tooltip(tip.c_str());
            }
        } else {
            dl->AddText(lf, font_px(lf), ImVec2(x, cy - font_px(lf) * 0.5f), col(g_pal.text_faint), text.c_str());
        }
    }
}

} // namespace ui
