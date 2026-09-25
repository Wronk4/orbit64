// Settings dialog: General, Graphics, Audio, Controllers, Emulation, Library, Shortcuts.

#include "app.hpp"
#include "platform.hpp"
#include "../controller.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>

namespace ui {

namespace fs = std::filesystem;

namespace {
struct PageInfo {
    const char* name;
    const char* subtitle;
    Icon icon;
};
const PageInfo kPages[] = {
    {"General", "Appearance, window behaviour and confirmations.", Icon::Settings},
    {"Graphics", "How the game is rendered, scaled and presented.", Icon::Monitor},
    {"Audio", "Output device, volume and latency.", Icon::Speaker},
    {"Controllers", "Map keyboards and gamepads to the four N64 controller ports.", Icon::Gamepad},
    {"Emulation", "Speed, fast-forward and core options.", Icon::Cpu},
    {"Library", "Where Orbit64 looks for your ROMs.", Icon::Library},
    {"Shortcuts", "Keyboard shortcuts for this platform.", Icon::Keyboard},
};
constexpr float kControlW = 260.0f;
} // namespace

void App::draw_settings() {
    bool was_open = settings_open_;
    if (!begin_modal("##settings", "Settings", dp(1000, 680), &settings_open_, Icon::Settings)) {
        if (was_open != settings_open_ || (!settings_open_ && was_open)) save_settings();
        return;
    }
    ImVec2 ws = ImGui::GetWindowSize();
    const float header = dp(56), footer = dp(64), nav_w = dp(220);
    const float body_h = ws.y - header - footer;

    // Navigation
    ImGui::SetCursorPos(ImVec2(0, header));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, g_pal.bg0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(12, 14));
    ImGui::BeginChild("##settings_nav", ImVec2(nav_w, body_h), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (int i = 0; i < static_cast<int>(SettingsPage::Count); ++i) {
        ImGui::PushID(i);
        ImVec2 p = ImGui::GetCursorScreenPos();
        float w = ImGui::GetContentRegionAvail().x, h = dp(38);
        if (ImGui::InvisibleButton("##page", ImVec2(w, h))) {
            settings_page_ = static_cast<SettingsPage>(i);
            input_.cancel_capture();
        }
        bool active = static_cast<int>(settings_page_) == i;
        bool hov = ImGui::IsItemHovered();
        float a = anim(ImGui::GetID("a"), active ? 1.0f : 0.0f, 16.0f);
        if (hov && !active) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(g_pal.bg3), dp(8));
        if (a > 0.01f) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(g_pal.accent_soft, a), dp(8));
        ImVec4 fc = active ? g_pal.text : (hov ? g_pal.text : g_pal.text_dim);
        draw_icon(dl, kPages[i].icon, ImVec2(p.x + dp(20), p.y + h * 0.5f), dp(16), col(active ? g_pal.accent_hover : fc));
        ImFont* f = active ? g_fonts.body_bold : g_fonts.body;
        dl->AddText(f, font_px(f), ImVec2(p.x + dp(40), p.y + (h - font_px(f)) * 0.5f), col(fc), kPages[i].name);
        ImGui::PopID();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    {
        ImVec2 mn = ImGui::GetItemRectMin(), mx = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddLine(ImVec2(mx.x, mn.y), mx, col(g_pal.border));
    }

    // Page content
    ImGui::SetCursorPos(ImVec2(nav_w, header));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(30, 24));
    ImGui::BeginChild("##settings_page", ImVec2(ws.x - nav_w, body_h), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleVar();
    const PageInfo& pi = kPages[static_cast<int>(settings_page_)];
    // Page fade-in on switch
    float fade = anim(ImHashStr("##page_fade") + static_cast<ImGuiID>(settings_page_), 1.0f, 12.0f);
    (void)fade;
    section_title(pi.name, pi.subtitle);
    ImGui::Dummy(dp(0, 4));
    switch (settings_page_) {
        case SettingsPage::General: settings_general(); break;
        case SettingsPage::Graphics: settings_graphics(); break;
        case SettingsPage::Audio: settings_audio(); break;
        case SettingsPage::Controller: settings_controller(); break;
        case SettingsPage::Emulation: settings_emulation(); break;
        case SettingsPage::Library: settings_library(); break;
        case SettingsPage::Shortcuts: settings_shortcuts(); break;
        default: break;
    }
    ImGui::Dummy(dp(0, 12));
    ImGui::EndChild();

    // Footer
    modal_footer_begin(footer);
    bool resettable = settings_page_ != SettingsPage::Shortcuts && settings_page_ != SettingsPage::Library;
    if (resettable && button("Restore Defaults", Icon::Refresh, ButtonKind::Ghost)) {
        switch (settings_page_) {
            case SettingsPage::General: settings_.reset_general(); break;
            case SettingsPage::Graphics:
                settings_.reset_graphics();
#if SDL_VERSION_ATLEAST(2, 0, 18)
                SDL_RenderSetVSync(renderer_, settings_.vsync ? 1 : 0);
#endif
                break;
            case SettingsPage::Audio: settings_.reset_audio(); open_audio(); break;
            case SettingsPage::Controller: settings_.reset_port(settings_port_); break;
            case SettingsPage::Emulation: settings_.reset_emulation(); core_.set_ucode_override(0); core_.set_cpu_core(settings_.cpu_core); break;
            default: break;
        }
        toast(std::string(pi.name) + " settings restored to defaults");
    }
    ImGui::SameLine(0, dp(14));
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(9));
    ImGui::PushFont(g_fonts.small);
    ImGui::TextColored(g_pal.text_faint, "Changes apply immediately and are saved automatically.");
    ImGui::PopFont();
    modal_footer_end();
    ImGui::SetCursorPos(ImVec2(ws.x - dp(20) - dp(110), ws.y - footer + (footer - dp(34)) * 0.5f));
    if (button("Done", Icon::None, ButtonKind::Primary, dp(110))) {
        settings_open_ = false;
        ImGui::CloseCurrentPopup();
    }
    draw_picker_dialogs(); // "Add Folder…" / "Choose…" pickers stack on top of Settings
    end_modal();
    if (!settings_open_) {
        input_.cancel_capture();
        save_settings();
    }
}

// ---------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------

void App::settings_general() {
    const float cw = dp(kControlW);

    row_begin("Accent color", "Highlight color used across the interface.", cw);
    {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        for (int i = 0; i < kAccentPresetCount; ++i) {
            ImGui::SetCursorScreenPos(ImVec2(p.x + i * dp(36), p.y));
            ImGui::PushID(i);
            if (ImGui::InvisibleButton("##sw", dp(28, 28))) settings_.accent = i;
            bool hov = ImGui::IsItemHovered();
            if (hov) tooltip(kAccentPresets[i].name);
            ImGui::PopID();
            ImVec2 c(p.x + i * dp(36) + dp(14), p.y + dp(14));
            if (settings_.accent == i) dl->AddCircle(c, dp(13), col(kAccentPresets[i].color), 32, dp(2));
            dl->AddCircleFilled(c, settings_.accent == i ? dp(9) : (hov ? dp(11) : dp(10)), col(kAccentPresets[i].color), 32);
        }
    }
    row_end();

    row_begin("Interface scale", "Size of text and controls. Automatic follows the display's scaling.", cw);
    {
        const char* items[] = {"Automatic", "90%", "100%", "110%", "125%", "150%", "175%", "200%"};
        const int values[] = {0, 90, 100, 110, 125, 150, 175, 200};
        int cur = 0;
        for (int i = 0; i < 8; ++i) if (values[i] == settings_.ui_scale_pct) cur = i;
        if (combo("uiscale", &cur, items, 8, cw)) settings_.ui_scale_pct = values[cur];
    }
    row_end();

    row_begin("Confirm before stopping", "Ask before stopping, switching games or quitting while a game runs.", cw);
    toggle("confirm", &settings_.confirm_stop);
    row_end();

    row_begin("Pause when in background", "Pause emulation when the window loses focus and resume when it returns.", cw);
    toggle("pausefocus", &settings_.pause_on_focus_loss);
    row_end();

    {
        std::string desc = std::string("Use the ") +
                           (platform::current_os() == platform::OS::Windows ? "Windows" :
                            platform::current_os() == platform::OS::MacOS ? "macOS" : "desktop (zenity / kdialog)") +
                           " file picker. When off or unavailable, Orbit64's built-in browser is used.";
        row_begin("Use system file dialogs", desc.c_str(), cw);
        ImGui::BeginDisabled(!browser_.native_available);
        toggle("native", &settings_.use_native_dialogs);
        ImGui::EndDisabled();
        if (!browser_.native_available) {
            ImGui::SameLine();
            ImGui::TextColored(g_pal.text_faint, "Not available");
        }
        row_end();
    }

    row_begin("Show status bar", "Emulator state and technical details at the bottom of the window.", cw);
    toggle("statusbar", &settings_.show_status_bar);
    row_end();

    row_begin("Status bar in fullscreen", "Keep the status bar visible during fullscreen gameplay.", cw);
    toggle("statusfs", &settings_.status_bar_in_fullscreen);
    row_end();

    row_begin("Hide cursor in fullscreen", "Hide the mouse pointer after two seconds of inactivity.", cw);
    toggle("hidecursor", &settings_.hide_cursor_fullscreen);
    row_end();

    row_begin("Remember window layout", "Restore the window size, position and maximised state on launch.", cw);
    toggle("remember", &settings_.remember_window);
    row_end();

    row_begin("Configuration folder", platform::path_to_utf8(platform::config_dir()).c_str(), cw);
    if (button("Open Folder", Icon::FolderOpen, ButtonKind::Subtle, cw)) platform::reveal_in_file_manager(platform::config_dir());
    row_end();
}

void App::settings_graphics() {
    const float cw = dp(kControlW);

    // Live preview of scaling settings using a test pattern (or the game).
    {
        float w = ImGui::GetContentRegionAvail().x, h = dp(170);
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), IM_COL32(5, 5, 7, 255), dp(10));
        dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(g_pal.border), dp(10));
        float ratio = settings_.aspect == 1 ? 16.0f / 9.0f : settings_.aspect == 2 ? (w - dp(20)) / (h - dp(20)) : 4.0f / 3.0f;
        float ih = h - dp(28), iw = ih * ratio;
        if (iw > w - dp(28)) { iw = w - dp(28); ih = iw / ratio; }
        ImVec2 mn(p.x + (w - iw) * 0.5f, p.y + (h - ih) * 0.5f), mx(mn.x + iw, mn.y + ih);
        if (has_frame_ && game_tex_) {
            dl->AddImage((ImTextureID)(intptr_t)display_texture(), mn, mx);
        } else {
            // SMPTE-like colour bars
            const ImU32 bars[] = {IM_COL32(192, 192, 192, 255), IM_COL32(192, 192, 0, 255), IM_COL32(0, 192, 192, 255), IM_COL32(0, 192, 0, 255),
                                  IM_COL32(192, 0, 192, 255),   IM_COL32(192, 0, 0, 255),   IM_COL32(0, 0, 192, 255)};
            for (int i = 0; i < 7; ++i)
                dl->AddRectFilled(ImVec2(mn.x + iw * i / 7, mn.y), ImVec2(mn.x + iw * (i + 1) / 7, mn.y + ih * 0.7f), bars[i]);
            for (int i = 0; i < 16; ++i) {
                int v = i * 17;
                dl->AddRectFilled(ImVec2(mn.x + iw * i / 16, mn.y + ih * 0.7f), ImVec2(mn.x + iw * (i + 1) / 16, mx.y), IM_COL32(v, v, v, 255));
            }
        }
        if (settings_.scanlines > 0) {
            float lh = ih / 120.0f;
            for (int y = 0; y < 120; ++y) {
                float yy = mn.y + y * lh + lh * 0.5f;
                dl->AddRectFilled(ImVec2(mn.x, yy), ImVec2(mx.x, yy + lh * 0.5f), IM_COL32(0, 0, 0, (int)(settings_.scanlines * 2.2f)));
            }
        }
        const char* lbl = has_frame_ ? "LIVE PREVIEW" : "PREVIEW";
        badge_at(dl, ImVec2(p.x + dp(10), p.y + dp(10)), lbl, g_pal.text_dim);
        ImGui::Dummy(ImVec2(w, h + dp(18)));
    }

    row_begin("Internal resolution",
              "Draws the game at a multiple of its own resolution for sharper 3D, textures and edges. "
              "It runs on all CPU cores; pick a lower value if games slow down.", cw);
    {
        const int fw = frame_native_w(), fh = frame_native_h();
        static char labels[8][48];
        const char* items[8];
        for (int i = 0; i < 8; ++i) {
            if (i == 0) std::snprintf(labels[i], sizeof labels[i], "Native (%d\xC3\x97%d)", fw, fh);
            else std::snprintf(labels[i], sizeof labels[i], "%d\xC3\x97  (%d\xC3\x97%d)", i + 1, fw * (i + 1), fh * (i + 1));
            items[i] = labels[i];
        }
        int cur = std::clamp(settings_.internal_scale, 1, 8) - 1;
        if (combo("iscale", &cur, items, 8, cw)) settings_.internal_scale = cur + 1;
    }
    row_end();

    row_begin("Aspect ratio", "4:3 matches the original TV image. Stretch fills the window.", cw);
    {
        const char* items[] = {"4:3", "16:9", "Stretch", "Native"};
        segmented("aspect", items, 4, &settings_.aspect, cw);
    }
    row_end();

    row_begin("Integer scaling", "Scale by whole multiples only for perfectly even pixels (may leave borders).", cw);
    toggle("integer", &settings_.integer_scale);
    row_end();

    row_begin("Texture filtering", "Nearest keeps pixels sharp; Bilinear smooths the upscaled image.", cw);
    {
        const char* items[] = {"Nearest", "Bilinear"};
        segmented("filter", items, 2, &settings_.filter, cw);
    }
    row_end();

    row_begin("Scanlines", "Darken alternate lines to mimic a CRT television.", cw);
    slider_int("scan", &settings_.scanlines, 0, 100, "%d%%", cw);
    row_end();

    row_begin("Vertical sync", "Synchronise the interface with the display refresh to avoid tearing.", cw);
    if (toggle("vsync", &settings_.vsync)) {
#if SDL_VERSION_ATLEAST(2, 0, 18)
        SDL_RenderSetVSync(renderer_, settings_.vsync ? 1 : 0);
#endif
    }
    row_end();

    row_begin("Fullscreen mode", "Borderless is instant and alt-tab friendly. Exclusive switches the display mode.", cw);
    {
        const char* items[] = {"Borderless", "Exclusive"};
        if (segmented("fsmode", items, 2, &settings_.fullscreen_mode, cw) && fullscreen_) {
            set_fullscreen(false);
            set_fullscreen(true);
        }
    }
    row_end();

    row_begin("FPS overlay", "Show frame rate and speed in the corner of the game screen.", cw);
    toggle("fpsov", &settings_.show_fps_overlay);
    row_end();

    row_begin("Renderer", "Chosen automatically for this platform (Direct3D, Metal or OpenGL).", cw);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(5));
    ImGui::PushFont(g_fonts.mono);
    ImGui::TextColored(g_pal.text_dim, "%s", renderer_name_.c_str());
    ImGui::PopFont();
    row_end();
}

void App::settings_audio() {
    const float cw = dp(kControlW);

    row_begin("Enable audio", "Turn all emulator sound on or off.", cw);
    if (toggle("audio", &settings_.audio_enabled)) open_audio();
    row_end();

    ImGui::BeginDisabled(!settings_.audio_enabled);
    row_begin("Volume", nullptr, cw);
    slider_int("vol", &settings_.volume, 0, 100, "%d%%", cw);
    row_end();

    row_begin("Output level", "Live signal meter of the emulated audio.", cw);
    {
        ImVec2 p = ImGui::GetCursorScreenPos();
        p.y += dp(9);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const int segs = 24;
        float peak = std::min(1.0f, core_.audio_peak() * 1.4f);
        float sw = (cw - (segs - 1) * dp(2)) / segs;
        for (int i = 0; i < segs; ++i) {
            float t = static_cast<float>(i) / segs;
            ImVec4 c = t < 0.7f ? g_pal.success : t < 0.9f ? g_pal.warning : g_pal.danger;
            bool on = t < peak;
            dl->AddRectFilled(ImVec2(p.x + i * (sw + dp(2)), p.y), ImVec2(p.x + i * (sw + dp(2)) + sw, p.y + dp(10)),
                              col(c, on ? 1.0f : 0.15f), dp(2));
        }
        ImGui::Dummy(ImVec2(cw, dp(28)));
    }
    row_end();

    row_begin("Output device", "Where sound is played. System default follows your OS settings.", cw);
    {
        std::vector<const char*> items = {"System default"};
        int cur = 0;
        for (size_t i = 0; i < audio_devices_.size(); ++i) {
            items.push_back(audio_devices_[i].c_str());
            if (audio_devices_[i] == settings_.audio_device) cur = static_cast<int>(i) + 1;
        }
        if (combo("device", &cur, items.data(), static_cast<int>(items.size()), cw)) {
            settings_.audio_device = cur == 0 ? "" : audio_devices_[cur - 1];
            open_audio();
        }
    }
    row_end();

    row_begin("Sample rate", "Rate of the host output stream; the N64 audio is resampled to it.", cw);
    {
        const char* items[] = {"44.1 kHz", "48 kHz"};
        int cur = settings_.sample_rate == 48000 ? 1 : 0;
        if (segmented("rate", items, 2, &cur, cw)) {
            settings_.sample_rate = cur == 1 ? 48000 : 44100;
            open_audio();
        }
    }
    row_end();

    row_begin("Buffer size", "Smaller buffers lower latency; larger buffers prevent crackling on slow systems.", cw);
    {
        const int sizes[] = {256, 512, 1024, 2048, 4096};
        char labels[5][40];
        const char* items[5];
        int cur = 2;
        for (int i = 0; i < 5; ++i) {
            std::snprintf(labels[i], sizeof labels[i], "%d samples (%.0f ms)", sizes[i], sizes[i] * 1000.0 / settings_.sample_rate);
            items[i] = labels[i];
            if (sizes[i] == settings_.buffer_frames) cur = i;
        }
        if (combo("buffer", &cur, items, 5, cw)) {
            settings_.buffer_frames = sizes[cur];
            open_audio();
        }
    }
    row_end();

    row_begin("Mute while fast-forwarding", nullptr, cw);
    toggle("muteff", &settings_.mute_on_fast_forward);
    row_end();

    row_begin("Mute in background", "Silence audio when the window isn't focused.", cw);
    toggle("mutebg", &settings_.mute_on_focus_loss);
    row_end();
    ImGui::EndDisabled();
}

void App::draw_controller_diagram(ImVec2 o, float W, const ControllerSnapshot& s) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float H = W * 0.68f;
    auto P = [&](float x, float y) { return ImVec2(o.x + x * W, o.y + y * H); };
    auto pressed = [&](std::uint16_t m) { return (s.buttons & m) != 0; };
    ImU32 body = col(g_pal.bg4), body_edge = col(g_pal.border_strong);

    // Shoulder buttons
    auto shoulder = [&](float x0, float x1, bool on, const char* t) {
        dl->AddRectFilled(P(x0, 0.02f), P(x1, 0.13f), on ? col(g_pal.accent) : col(g_pal.bg3), dp(6));
        ImVec2 c((P(x0, 0).x + P(x1, 0).x) * 0.5f, P(0, 0.075f).y);
        ImVec2 ts = g_fonts.small_bold->CalcTextSizeA(font_px(g_fonts.small_bold), FLT_MAX, 0, t);
        dl->AddText(g_fonts.small_bold, font_px(g_fonts.small_bold), ImVec2(c.x - ts.x * 0.5f, c.y - ts.y * 0.5f), col(on ? ImVec4(1, 1, 1, 1) : g_pal.text_dim), t);
    };
    shoulder(0.10f, 0.30f, pressed(Button::L), "L");
    shoulder(0.70f, 0.90f, pressed(Button::R), "R");

    // Body: three prongs joined by a top band
    dl->AddRectFilled(P(0.06f, 0.10f), P(0.94f, 0.52f), body, W * 0.12f);
    dl->AddRectFilled(P(0.07f, 0.30f), P(0.30f, 0.92f), body, W * 0.1f);
    dl->AddRectFilled(P(0.39f, 0.36f), P(0.61f, 1.00f), body, W * 0.1f);
    dl->AddRectFilled(P(0.70f, 0.30f), P(0.93f, 0.92f), body, W * 0.1f);
    dl->AddRect(P(0.06f, 0.10f), P(0.94f, 0.52f), body_edge, W * 0.12f);

    // D-pad
    ImVec2 dc = P(0.19f, 0.32f);
    float a = W * 0.028f, b = W * 0.075f;
    ImU32 dbase = col(g_pal.bg1);
    dl->AddRectFilled(ImVec2(dc.x - a, dc.y - b), ImVec2(dc.x + a, dc.y + b), dbase, dp(3));
    dl->AddRectFilled(ImVec2(dc.x - b, dc.y - a), ImVec2(dc.x + b, dc.y + a), dbase, dp(3));
    ImU32 hi = col(g_pal.accent);
    if (pressed(Button::D_UP)) dl->AddRectFilled(ImVec2(dc.x - a, dc.y - b), ImVec2(dc.x + a, dc.y - a), hi, dp(3));
    if (pressed(Button::D_DOWN)) dl->AddRectFilled(ImVec2(dc.x - a, dc.y + a), ImVec2(dc.x + a, dc.y + b), hi, dp(3));
    if (pressed(Button::D_LEFT)) dl->AddRectFilled(ImVec2(dc.x - b, dc.y - a), ImVec2(dc.x - a, dc.y + a), hi, dp(3));
    if (pressed(Button::D_RIGHT)) dl->AddRectFilled(ImVec2(dc.x + a, dc.y - a), ImVec2(dc.x + b, dc.y + a), hi, dp(3));

    // Start
    ImVec2 sc = P(0.5f, 0.28f);
    dl->AddCircleFilled(sc, W * 0.03f, pressed(Button::START) ? IM_COL32(255, 90, 90, 255) : IM_COL32(150, 40, 45, 255), 24);
    dl->AddText(g_fonts.small, font_px(g_fonts.small) * 0.85f, ImVec2(sc.x - dp(13), sc.y + W * 0.04f), col(g_pal.text_faint), "START");

    // Analog stick
    ImVec2 ac = P(0.5f, 0.56f);
    float ar = W * 0.07f;
    dl->AddCircleFilled(ac, ar, col(g_pal.bg1), 32);
    dl->AddCircle(ac, ar, body_edge, 32, dp(1));
    ImVec2 knob(ac.x + (s.stick_x / 80.0f) * ar * 0.7f, ac.y - (s.stick_y / 80.0f) * ar * 0.7f);
    bool moved = s.stick_x != 0 || s.stick_y != 0;
    dl->AddCircleFilled(knob, ar * 0.5f, moved ? col(g_pal.accent) : col(g_pal.text_faint), 32);

    // Z trigger (underside) shown as a pill on the centre prong
    ImVec2 zc = P(0.5f, 0.86f);
    dl->AddRectFilled(ImVec2(zc.x - W * 0.04f, zc.y - W * 0.022f), ImVec2(zc.x + W * 0.04f, zc.y + W * 0.022f),
                      pressed(Button::Z) ? col(g_pal.accent) : col(g_pal.bg3), W * 0.022f);
    dl->AddText(g_fonts.small_bold, font_px(g_fonts.small_bold), ImVec2(zc.x - dp(4), zc.y - font_px(g_fonts.small_bold) * 0.5f),
                col(pressed(Button::Z) ? ImVec4(1, 1, 1, 1) : g_pal.text_dim), "Z");

    // A / B
    auto round_button = [&](ImVec2 c, float r, bool on, ImU32 on_col, ImU32 off_col, const char* t) {
        dl->AddCircleFilled(c, r, on ? on_col : off_col, 32);
        ImVec2 ts = g_fonts.small_bold->CalcTextSizeA(font_px(g_fonts.small_bold), FLT_MAX, 0, t);
        dl->AddText(g_fonts.small_bold, font_px(g_fonts.small_bold), ImVec2(c.x - ts.x * 0.5f, c.y - ts.y * 0.5f), IM_COL32(255, 255, 255, on ? 255 : 170), t);
    };
    round_button(P(0.73f, 0.42f), W * 0.038f, pressed(Button::A), IM_COL32(90, 140, 255, 255), IM_COL32(40, 70, 150, 255), "A");
    round_button(P(0.655f, 0.33f), W * 0.034f, pressed(Button::B), IM_COL32(80, 210, 110, 255), IM_COL32(30, 110, 55, 255), "B");

    // C buttons
    ImVec2 cc = P(0.83f, 0.26f);
    float co = W * 0.042f, cr = W * 0.022f;
    ImU32 yon = IM_COL32(255, 210, 60, 255), yoff = IM_COL32(140, 110, 20, 255);
    dl->AddCircleFilled(ImVec2(cc.x, cc.y - co), cr, pressed(Button::C_UP) ? yon : yoff, 20);
    dl->AddCircleFilled(ImVec2(cc.x, cc.y + co), cr, pressed(Button::C_DOWN) ? yon : yoff, 20);
    dl->AddCircleFilled(ImVec2(cc.x - co, cc.y), cr, pressed(Button::C_LEFT) ? yon : yoff, 20);
    dl->AddCircleFilled(ImVec2(cc.x + co, cc.y), cr, pressed(Button::C_RIGHT) ? yon : yoff, 20);

    ImGui::Dummy(ImVec2(W, H + dp(4)));
}

void App::settings_controller() {
    const char* ports[] = {"Port 1", "Port 2", "Port 3", "Port 4"};
    float full = ImGui::GetContentRegionAvail().x;
    if (segmented("ports", ports, 4, &settings_port_, std::min(full, dp(420)))) input_.cancel_capture();
    ImGui::Dummy(dp(0, 10));

    PortConfig& pc = settings_.ports[settings_port_];
    const float cw = dp(kControlW);

    row_begin("Connected", "Whether a controller is plugged into this port.", cw);
    toggle("plugged", &pc.plugged);
    row_end();

    row_begin("Input device", nullptr, cw);
    {
        std::vector<std::string> names = {"Keyboard"};
        for (const auto& g : input_.gamepads()) names.push_back(g.name);
        if (pc.device >= static_cast<int>(names.size())) names.push_back(input_.device_name(pc.device));
        std::vector<const char*> items;
        for (auto& n : names) items.push_back(n.c_str());
        combo("device", &pc.device, items.data(), static_cast<int>(items.size()), cw);
    }
    row_end();

    ImGui::BeginDisabled(!pc.plugged);
    const bool keyboard = pc.device == 0;
    float avail = ImGui::GetContentRegionAvail().x;
    bool two_col = avail > dp(620);
    float diag_w = two_col ? dp(280) : std::min(avail, dp(320));

    // Live controller preview
    ImGui::BeginGroup();
    ControllerSnapshot live = input_.poll(pc, true);
    ImGui::PushFont(g_fonts.small_bold);
    ImGui::TextColored(g_pal.text_faint, "LIVE PREVIEW");
    ImGui::PopFont();
    ImGui::Dummy(dp(0, 4));
    draw_controller_diagram(ImGui::GetCursorScreenPos(), diag_w, live);
    ImGui::PushFont(g_fonts.small);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + diag_w);
    ImGui::TextColored(g_pal.text_faint, "Press buttons on the selected device to test the mapping.");
    ImGui::PopTextWrapPos();
    ImGui::PopFont();
    ImGui::Dummy(dp(0, 8));
    ImGui::PushFont(g_fonts.small_bold);
    ImGui::TextColored(g_pal.text_faint, "ANALOG STICK");
    ImGui::PopFont();
    ImGui::PushFont(g_fonts.small);
    ImGui::TextColored(g_pal.text_dim, "Dead zone");
    ImGui::PopFont();
    slider_float("dz", &pc.deadzone, 0.0f, 0.5f, "%.2f", diag_w);
    ImGui::PushFont(g_fonts.small);
    ImGui::TextColored(g_pal.text_dim, "Range");
    ImGui::PopFont();
    slider_float("sens", &pc.sensitivity, 0.5f, 1.5f, "%.2f\xC3\x97", diag_w);
    ImGui::EndGroup();

    if (two_col) ImGui::SameLine(0, dp(28));
    else ImGui::Dummy(dp(0, 12));

    // Binding list
    ImGui::BeginGroup();
    float list_w = two_col ? avail - diag_w - dp(28) : avail;
    ImGui::PushFont(g_fonts.small_bold);
    ImGui::TextColored(g_pal.text_faint, keyboard ? "KEYBOARD BINDINGS" : "GAMEPAD BINDINGS");
    ImGui::PopFont();
    ImGui::Dummy(dp(0, 4));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    int cols = list_w > dp(460) ? 2 : 1;
    float cell_w = (list_w - (cols - 1) * dp(12)) / cols;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    const float rh = dp(36);
    for (int i = 0; i < kN64InputCount; ++i) {
        int c = i % cols, r = i / cols;
        ImVec2 p(origin.x + c * (cell_w + dp(12)), origin.y + r * (rh + dp(4)));
        ImGui::SetCursorScreenPos(p);
        ImGui::PushID(i);
        bool capturing = input_.capturing() && input_.capture_port() == settings_port_ && input_.capture_input() == i;
        bool clicked = ImGui::InvisibleButton("##bind", ImVec2(cell_w, rh));
        bool hov = ImGui::IsItemHovered();
        bool right = ImGui::IsItemClicked(ImGuiMouseButton_Right);
        ImGui::PopID();
        float pulse = capturing ? 0.5f + 0.5f * std::sin(static_cast<float>(ImGui::GetTime()) * 6.0f) : 0.0f;
        dl->AddRectFilled(p, ImVec2(p.x + cell_w, p.y + rh), col(capturing ? mix(g_pal.bg3, g_pal.accent_soft, 1.0f) : (hov ? g_pal.bg3 : g_pal.bg2)), dp(8));
        dl->AddRect(p, ImVec2(p.x + cell_w, p.y + rh), col(capturing ? with_alpha(g_pal.accent, 0.5f + 0.5f * pulse) : g_pal.border), dp(8));
        dl->AddText(g_fonts.body, font_px(g_fonts.body), ImVec2(p.x + dp(12), p.y + (rh - font_px(g_fonts.body)) * 0.5f), col(g_pal.text_dim),
                    n64_input_name(static_cast<N64Input>(i)));
        std::string b = capturing ? (keyboard ? "Press a key\xE2\x80\xA6" : "Press a button\xE2\x80\xA6")
                                  : (keyboard ? InputManager::key_label(pc.keys[i]) : InputManager::pad_label(pc.pad[i]));
        bool unbound = !capturing && (keyboard ? pc.keys[i] <= 0 : pc.pad[i] < 0);
        ImFont* f = g_fonts.mono_small;
        ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, b.c_str());
        float kw = std::min(ts.x + dp(14), cell_w * 0.55f);
        ImVec2 kmn(p.x + cell_w - kw - dp(8), p.y + dp(7)), kmx(p.x + cell_w - dp(8), p.y + rh - dp(7));
        if (!capturing) {
            dl->AddRectFilled(kmn, ImVec2(kmx.x, kmx.y + dp(1)), col(g_pal.border_strong), dp(5));
            dl->AddRectFilled(kmn, kmx, col(unbound ? g_pal.bg2 : g_pal.bg4), dp(5));
        }
        text_ellipsis(dl, f, ImVec2(kmn.x + dp(7), (kmn.y + kmx.y) * 0.5f - ts.y * 0.5f), kw - dp(10),
                      col(capturing ? g_pal.accent_hover : (unbound ? g_pal.text_faint : g_pal.text)), b.c_str());
        if (clicked) input_.begin_capture(settings_port_, i);
        if (right) {
            if (keyboard) pc.keys[i] = 0;
            else pc.pad[i] = -1;
        }
        if (hov && !capturing) tooltip("Click to rebind \xC2\xB7 right-click to clear \xC2\xB7 Esc cancels");
    }
    int rows = (kN64InputCount + cols - 1) / cols;
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + rows * (rh + dp(4))));
    ImGui::Dummy(ImVec2(list_w, dp(4)));
    if (!keyboard && input_.gamepads().empty()) {
        ImGui::TextColored(g_pal.warning, "No gamepads detected. Connect one \xE2\x80\x94 it will appear automatically.");
    } else {
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_faint, "%d gamepad%s detected. XInput, DualShock/DualSense, Switch Pro and most USB pads are supported.",
                           (int)input_.gamepads().size(), input_.gamepads().size() == 1 ? "" : "s");
        ImGui::PopFont();
    }
    ImGui::EndGroup();
    ImGui::EndDisabled();
}

void App::settings_emulation() {
    const float cw = dp(kControlW);

    row_begin("Limit speed", "Run at the console's real speed. Turn off to run as fast as your computer allows.", cw);
    toggle("limit", &settings_.limit_speed);
    row_end();

    row_begin("Fast-forward speed", "Speed while holding Tab.", cw);
    {
        const char* items[] = {"2\xC3\x97", "3\xC3\x97", "4\xC3\x97", "8\xC3\x97", "Max"};
        const int vals[] = {2, 3, 4, 8, 0};
        int cur = 1;
        for (int i = 0; i < 5; ++i) if (vals[i] == settings_.ff_speed) cur = i;
        if (segmented("ff", items, 5, &cur, cw)) settings_.ff_speed = vals[cur];
    }
    row_end();

    row_begin("CPU core", "Dynamic Recompiler translates game code to native machine code for much higher speed. Switch to Interpreter only to work around a compatibility issue.", cw);
    {
        const char* items[] = {"Interpreter", "Dynamic Recompiler (JIT)"};
        if (combo("cpucore", &settings_.cpu_core, items, 2, cw)) {
            core_.set_cpu_core(settings_.cpu_core);
        }
    }
    row_end();

    row_begin("Graphics microcode", "Auto-detect works for nearly every game. Override only if a game renders incorrectly.", cw);
    {
        const char* items[] = {"Auto-detect", "Fast3D", "F3DEX", "F3DEX2", "S2DEX", "S2DEX2", "F3DEX (GoldenEye)"};
        if (combo("ucode", &settings_.ucode_override, items, 7, cw)) {
            core_.set_ucode_override(settings_.ucode_override);
            if (settings_.ucode_override != 0) toast("Microcode override active", ToastKind::Warning);
        }
    }
    row_end();

    row_begin("Save data", "Cartridge saves (EEPROM, SRAM, FlashRAM) are written as .sav files next to each ROM when emulation stops.", cw);
    if (button("Open ROM Folder", Icon::FolderOpen, ButtonKind::Subtle, cw, !current_rom_.path.empty()))
        platform::reveal_in_file_manager(current_rom_.path);
    row_end();

    // Core information card
    ImGui::Dummy(dp(0, 6));
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float h = dp(118);
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(g_pal.bg2), dp(10));
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(g_pal.border), dp(10));
    draw_icon(dl, Icon::Cpu, ImVec2(p.x + dp(28), p.y + dp(28)), dp(20), col(g_pal.accent_hover));
    dl->AddText(g_fonts.body_bold, font_px(g_fonts.body_bold), ImVec2(p.x + dp(52), p.y + dp(19)), col(g_pal.text), "Emulation core");
    const char* lines[] = {
        "CPU      MIPS R4300i interpreter @ 93.75 MHz",
        "RCP      High-level RSP (graphics + audio microcode), software RDP",
        "Timing   525-line VI scheduling, AI DMA with adaptive resampling",
    };
    for (int i = 0; i < 3; ++i)
        dl->AddText(g_fonts.mono_small, font_px(g_fonts.mono_small), ImVec2(p.x + dp(52), p.y + dp(46) + i * dp(20)), col(g_pal.text_dim), lines[i]);
    ImGui::Dummy(ImVec2(w, h));
}

void App::settings_library() {
    const float cw = dp(kControlW);
    ImGui::PushFont(g_fonts.small_bold);
    ImGui::TextColored(g_pal.text_faint, "ROM FOLDERS");
    ImGui::PopFont();
    ImGui::Dummy(dp(0, 2));

    float w = ImGui::GetContentRegionAvail().x;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    int remove = -1;
    if (settings_.rom_dirs.empty()) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        dl->AddRectFilled(p, ImVec2(p.x + w, p.y + dp(56)), col(g_pal.bg2), dp(8));
        dl->AddText(ImVec2(p.x + dp(16), p.y + dp(18)), col(g_pal.text_dim), "No folders yet. Add one to start building your library.");
        ImGui::Dummy(ImVec2(w, dp(56)));
    }
    for (size_t i = 0; i < settings_.rom_dirs.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        ImVec2 p = ImGui::GetCursorScreenPos();
        float h = dp(48);
        dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(g_pal.bg2), dp(8));
        dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(g_pal.border), dp(8));
        draw_icon(dl, Icon::Folder, ImVec2(p.x + dp(22), p.y + h * 0.5f), dp(16), col(g_pal.accent));
        text_ellipsis(dl, g_fonts.body, ImVec2(p.x + dp(44), p.y + (h - font_px(g_fonts.body)) * 0.5f), w - dp(140), col(g_pal.text),
                      settings_.rom_dirs[i].c_str());
        ImGui::SetCursorScreenPos(ImVec2(p.x + w - dp(84), p.y + dp(9)));
        if (icon_button("reveal", Icon::FolderOpen, dp(30), platform::reveal_action_label()))
            platform::reveal_in_file_manager(platform::utf8_to_path(settings_.rom_dirs[i]));
        ImGui::SameLine(0, dp(6));
        if (icon_button("remove", Icon::Trash, dp(30), "Remove from library", false, true, ButtonKind::Danger)) remove = static_cast<int>(i);
        ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + h + dp(6)));
        ImGui::Dummy(ImVec2(w, 0));
        ImGui::PopID();
    }
    if (remove >= 0) {
        settings_.rom_dirs.erase(settings_.rom_dirs.begin() + remove);
        rescan_library();
        lib_filter_ = LibraryFilter::All;
    }
    ImGui::Dummy(dp(0, 4));
    if (button("Add Folder\xE2\x80\xA6", Icon::Plus, ButtonKind::Subtle)) action_add_folder();
    ImGui::SameLine(0, dp(10));
    ImGui::BeginDisabled(library_.scanning());
    if (button(library_.scanning() ? "Scanning\xE2\x80\xA6" : "Rescan Now", Icon::Refresh, ButtonKind::Ghost))
        rescan_library();
    ImGui::EndDisabled();
    ImGui::Dummy(dp(0, 18));

    {
        std::string desc;
        if (!settings_.boxart_dir.empty()) desc = settings_.boxart_dir;
        else if (!library_.boxart_dirs().empty()) desc = "Auto-detected: " + platform::path_to_utf8(library_.boxart_dirs().front());
        else desc = "Not found. Put a No-Intro / RetroArch box art folder (e.g. \"Named_Boxarts\") inside a ROM folder, or choose one.";
        char count[64];
        std::snprintf(count, sizeof count, "%zu covers indexed. ", library_.boxart_count());
        desc = std::string(library_.boxart_count() ? count : "") + desc;
        row_begin("Box art folder", desc.c_str(), cw);
        float half = (cw - dp(8)) * 0.5f;
        if (button("Choose\xE2\x80\xA6", Icon::FolderOpen, ButtonKind::Subtle, half)) action_choose_boxart_folder();
        ImGui::SameLine(0, dp(8));
        if (button("Auto-detect", Icon::Refresh, ButtonKind::Ghost, half, !settings_.boxart_dir.empty())) {
            settings_.boxart_dir.clear();
            images_.clear();
            rescan_library();
        }
        row_end();
    }

    row_begin("Scan subfolders", "Look for ROMs inside nested folders too.", cw);
    if (toggle("recursive", &settings_.scan_recursive)) rescan_library();
    row_end();

    row_begin("Card size", "Size of game cards in grid view.", cw);
    slider_float("cardsize", &settings_.grid_card_size, 0.7f, 1.5f, "%.1f\xC3\x97", cw);
    row_end();

    row_begin("Recently played", "Clear the list of recently played games (play time is kept).", cw);
    if (button("Clear History", Icon::Clock, ButtonKind::Subtle, cw)) {
        library_.clear_recent();
        toast("Recently played list cleared");
    }
    row_end();

    row_begin("Thumbnails", "Screenshots captured automatically while playing, shown on game cards.", cw);
    if (button("Clear Thumbnails", Icon::Trash, ButtonKind::Subtle, cw)) {
        std::error_code ec;
        fs::remove_all(platform::config_dir() / "thumbnails", ec);
        for (auto& [k, t] : thumbs_) if (t) SDL_DestroyTexture(t);
        thumbs_.clear();
        thumb_missing_.clear();
        toast("Thumbnail cache cleared");
    }
    row_end();
}

void App::settings_shortcuts() {
    ImGui::PushFont(g_fonts.small);
    ImGui::PushStyleColor(ImGuiCol_Text, g_pal.text_dim);
    std::string note = std::string("Showing shortcuts for ") + platform::os_display_name() + ". " +
                       (platform::current_os() == platform::OS::MacOS ? "On Windows and Linux, Ctrl replaces Cmd."
                                                                       : "On macOS, Cmd replaces Ctrl.");
    ImGui::TextUnformatted(note.c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
    ImGui::Dummy(dp(0, 8));

    float w = ImGui::GetContentRegionAvail().x;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    auto list = shortcuts();
    for (size_t i = 0; i < list.size(); ++i) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        float h = dp(40);
        if (i % 2 == 0) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(g_pal.bg2), dp(6));
        dl->AddText(g_fonts.body, font_px(g_fonts.body), ImVec2(p.x + dp(14), p.y + (h - font_px(g_fonts.body)) * 0.5f), col(g_pal.text),
                    list[i].action);
        // Render each alternative ("A  or  B") as key caps, right-aligned.
        std::vector<std::string> alts;
        std::string k = list[i].keys;
        size_t pos;
        while ((pos = k.find("  or  ")) != std::string::npos) {
            alts.push_back(k.substr(0, pos));
            k = k.substr(pos + 6);
        }
        alts.push_back(k);
        float x = p.x + w - dp(12);
        for (int a = static_cast<int>(alts.size()) - 1; a >= 0; --a) {
            // split on '+'
            std::vector<std::string> keys;
            std::string s = alts[a];
            size_t start = 0;
            while (true) {
                size_t plus = s.find('+', start == 0 ? 0 : start);
                if (plus == std::string::npos || plus == s.size() - 1) { keys.push_back(s.substr(start)); break; }
                keys.push_back(s.substr(start, plus - start));
                start = plus + 1;
            }
            for (int j = static_cast<int>(keys.size()) - 1; j >= 0; --j) {
                ImFont* f = g_fonts.mono_small;
                ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, keys[j].c_str());
                float kw = ts.x + dp(14);
                ImVec2 kmn(x - kw, p.y + dp(9)), kmx(x, p.y + h - dp(9));
                dl->AddRectFilled(kmn, ImVec2(kmx.x, kmx.y + dp(2)), col(g_pal.border_strong), dp(5));
                dl->AddRectFilled(kmn, kmx, col(g_pal.bg4), dp(5));
                dl->AddText(f, font_px(f), ImVec2(kmn.x + dp(7), (kmn.y + kmx.y) * 0.5f - ts.y * 0.5f), col(g_pal.text), keys[j].c_str());
                x -= kw + dp(4);
            }
            if (a > 0) {
                const char* orr = "or";
                ImVec2 ts = g_fonts.small->CalcTextSizeA(font_px(g_fonts.small), FLT_MAX, 0, orr);
                x -= dp(6);
                dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(x - ts.x, p.y + (h - ts.y) * 0.5f), col(g_pal.text_faint), orr);
                x -= ts.x + dp(10);
            }
        }
        ImGui::Dummy(ImVec2(w, h));
    }
    ImGui::Dummy(dp(0, 10));
    ImGui::PushFont(g_fonts.small_bold);
    ImGui::TextColored(g_pal.text_faint, "GAME CONTROLS (PORT 1, KEYBOARD)");
    ImGui::PopFont();
    ImGui::Dummy(dp(0, 2));
    ImGui::PushFont(g_fonts.small);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
    std::string summary;
    const PortConfig& p1 = settings_.ports[0];
    for (int i = 0; i < kN64InputCount; ++i) {
        if (i) summary += "   \xC2\xB7   ";
        summary += std::string(n64_input_name(static_cast<N64Input>(i))) + ": " + InputManager::key_label(p1.keys[i]);
    }
    ImGui::TextColored(g_pal.text_dim, "%s", summary.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopFont();
}

} // namespace ui
