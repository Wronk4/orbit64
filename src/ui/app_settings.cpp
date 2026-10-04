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
#include <cctype>
#include <cstring>

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
    {"Shortcuts", "Keyboard shortcuts for the app. Click one to change it.", Icon::Keyboard},
};
constexpr float kControlW = 260.0f;

// The controller preview is drawn in "diagram units": 100 across the canvas,
// kPadHeight down. The pad itself is laid out in "body units" (0..100 across
// the shell, 0 at the top of the centre hump) traced from a photo of the real
// controller, and mapped into the canvas with a margin for the shoulder
// buttons and the drop shadow.
constexpr float kPadHeight = 100.0f;
constexpr float kPadMarginX = 3.0f, kPadMarginY = 5.0f, kPadScale = 0.94f;

// Left half of the shell outline in body units, from the top of the centre
// hump round the left handle to the tip of the centre prong. The right half
// is its mirror image; the closed outline is a Catmull-Rom curve through
// these points.
const ImVec2 kPadLeft[] = {
    {50.0f, 0.0f},   {44.0f, 0.5f},   {40.0f, 1.25f},  {36.0f, 2.0f},   {33.5f, 3.0f},   {31.75f, 4.25f}, {30.9f, 5.6f},
    {30.5f, 6.5f},   {28.25f, 7.0f},  {26.0f, 7.5f},   {22.5f, 8.0f},   {20.0f, 8.5f},   {17.25f, 9.0f},  {15.5f, 9.5f},
    {13.0f, 10.5f},  {11.5f, 11.2f},  {9.5f, 13.5f},   {7.25f, 16.0f},  {5.5f, 18.5f},   {4.25f, 21.0f},  {3.5f, 23.5f},
    {2.75f, 28.5f},  {2.25f, 33.5f},  {1.25f, 38.5f},  {0.5f, 43.5f},   {0.2f, 48.0f},   {0.0f, 53.5f},   {0.5f, 58.5f},
    {1.0f, 63.5f},   {2.0f, 68.5f},   {3.5f, 73.0f},   {5.0f, 75.2f},   {7.0f, 76.6f},   {8.6f, 77.0f},   {10.2f, 76.6f},
    {11.7f, 75.2f},  {12.75f, 73.5f}, {14.75f, 68.5f}, {16.5f, 63.5f},  {18.0f, 58.5f},  {19.25f, 53.5f}, {20.2f, 50.5f},
    {21.6f, 48.6f},  {24.0f, 47.6f},  {26.6f, 48.3f},  {30.0f, 49.0f},  {32.2f, 49.6f},  {33.7f, 50.6f},  {34.6f, 51.8f},
    {35.3f, 53.5f},  {36.5f, 58.5f},  {37.25f, 63.5f}, {38.0f, 68.5f},  {39.25f, 73.5f}, {40.0f, 78.0f},  {40.9f, 81.5f},
    {41.6f, 84.5f},  {42.5f, 88.3f},  {43.6f, 91.3f},  {44.7f, 93.3f},  {46.0f, 95.3f},  {47.8f, 96.9f},  {50.0f, 97.5f},
};
// kPadLeft indices: the hump's top edge ends here, and the L button runs
// along the shoulder between these two points.
constexpr int kPadHumpEnd = 7;
constexpr int kPadShoulderBegin = 8, kPadShoulderEnd = 17;
constexpr int kPadCurveSteps = 4; // curve samples per outline point
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
    bool resettable = settings_page_ != SettingsPage::Library;
    if (resettable && button("Restore Defaults", Icon::Refresh, ButtonKind::Ghost)) {
        switch (settings_page_) {
            case SettingsPage::General: settings_.reset_general(); break;
            case SettingsPage::Graphics:
                settings_.reset_graphics();
                SDL_SetRenderVSync(renderer_, settings_.vsync ? 1 : SDL_RENDERER_VSYNC_DISABLED);
                break;
            case SettingsPage::Audio: settings_.reset_audio(); open_audio(); break;
            case SettingsPage::Controller: settings_.reset_port(settings_port_); break;
            case SettingsPage::Shortcuts: settings_.reset_shortcuts(); hotkey_capture_ = -1; break;
            case SettingsPage::Emulation: settings_.reset_emulation(); core_.set_ucode_override(0); core_.set_cpu_core(settings_.cpu_core); core_.set_rsp_mode(settings_.rsp_mode); core_.set_rdp_exact(settings_.rdp_exact); core_.set_rdp_exact_gpu(settings_.rdp_exact_gpu); break;
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
    if (settings_page_ != SettingsPage::Shortcuts) hotkey_capture_ = -1;
    if (!settings_open_) {
        input_.cancel_capture();
        hotkey_capture_ = -1;
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
    if (toggle("vsync", &settings_.vsync)) SDL_SetRenderVSync(renderer_, settings_.vsync ? 1 : SDL_RENDERER_VSYNC_DISABLED);
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

    row_begin("Ray tracing",
              "Traces every frame's 3D scene: soft cast shadows and ambient occlusion. Per pixel on the GPU at internal "
              "resolutions of 2x and up (Vulkan); per vertex otherwise. HLE graphics only.",
              cw);
    if (toggle("raytracing", &settings_.hle_raytracing)) {
        core_.set_raytracing(settings_.hle_raytracing);
    }
    row_end();
    if (settings_.hle_raytracing) {
        row_begin("Shadow strength", "How dark a surface in full shadow gets.", cw);
        if (slider_int("rtshadow", &settings_.rt_shadow, 0, 100, "%d%%", cw)) core_.set_rt_strength(settings_.rt_shadow, settings_.rt_ao);
        row_end();
        row_begin("Ambient occlusion", "How dark creases, corners and contact points get.", cw);
        if (slider_int("rtao", &settings_.rt_ao, 0, 100, "%d%%", cw)) core_.set_rt_strength(settings_.rt_shadow, settings_.rt_ao);
        row_end();
    }

    {
        const bool fx_ok = postfx_ && postfx_->ok();
        row_begin("Modern post-processing",
                  fx_ok ? "Bloom, anti-aliasing, sharpening, a filmic tone curve, vibrance and a vignette on the shown frame."
                        : "Bloom, anti-aliasing, sharpening and colour grading. Needs the GPU video backend on Vulkan.",
                  cw);
        toggle("postfx", &settings_.postfx);
        row_end();
        if (settings_.postfx && fx_ok) {
            row_begin("Bloom", "Glow around bright highlights.", cw);
            slider_int("pfxbloom", &settings_.pfx_bloom, 0, 100, "%d%%", cw);
            row_end();
            row_begin("Sharpening", "Contrast-adaptive sharpening of texture detail.", cw);
            slider_int("pfxsharp", &settings_.pfx_sharpen, 0, 100, "%d%%", cw);
            row_end();
            row_begin("Vibrance", "Richer colour, more for the dull ones.", cw);
            slider_int("pfxvib", &settings_.pfx_vibrance, 0, 100, "%d%%", cw);
            row_end();
            row_begin("Contrast", "A filmic S-curve: deeper shadows, brighter mid-tones.", cw);
            slider_int("pfxcon", &settings_.pfx_contrast, 0, 100, "%d%%", cw);
            row_end();
            row_begin("Vignette", "Darkens the corners of the picture.", cw);
            slider_int("pfxvig", &settings_.pfx_vignette, 0, 100, "%d%%", cw);
            row_end();
            row_begin("Anti-aliasing", "Smooths jagged polygon edges (FXAA).", cw);
            toggle("pfxaa", &settings_.pfx_fxaa);
            row_end();
            row_begin("Filmic highlights", "Bright areas roll off softly instead of clipping.", cw);
            toggle("pfxtm", &settings_.pfx_tonemap);
            row_end();
        }
    }

    row_begin("Video backend",
              "GPU uses Metal, Vulkan or Direct3D 12 and renders high internal resolutions on the graphics card. "
              "Compatibility works with any driver but renders on the CPU. Applies after a restart.",
              cw);
    {
        const char* items[] = {"GPU", "Compatibility"};
        if (segmented("vbackend", items, 2, &settings_.video_backend, cw)) {
            save_settings();
            toast("The video backend changes the next time Orbit64 starts", ToastKind::Info);
        }
    }
    row_end();

    row_begin("Renderer", "The graphics driver in use.", cw);
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

    row_begin("Latency", "Sound waiting to be played, including the device buffer. A dropout is a moment the "
                         "emulator fell behind and the sound paused briefly.", cw);
    {
        const CoreStats st = core_.stats();
        ImGui::PushFont(g_fonts.mono);
        if (core_.state() == RunState::Running && audio_dev_ && st.audio_latency_ms > 0.0f) {
            ImGui::Text("%.0f ms", st.audio_latency_ms);
            ImGui::SameLine();
            ImGui::TextColored(g_pal.text_dim, "%.1f kHz", audio_freq_ / 1000.0);
            ImGui::SameLine();
            ImGui::TextColored(st.audio_underruns ? g_pal.warning : g_pal.text_dim, "%llu dropout%s",
                               static_cast<unsigned long long>(st.audio_underruns), st.audio_underruns == 1 ? "" : "s");
        } else {
            ImGui::TextColored(g_pal.text_dim, "%s", "\xE2\x80\x94");
        }
        ImGui::PopFont();
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

    row_begin("Sample rate", "Preferred rate of the output stream; the N64 audio is resampled to it. The device's own "
                              "rate is used when it can't be changed.", cw);
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
        int cur = 1;
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
    const float u = W / 100.0f;
    const float bu = u * kPadScale; // one body unit in pixels
    const float H = kPadHeight * u;
    const float alpha = ImGui::GetStyle().Alpha; // dimmed while the port is unplugged
    auto B = [&](float x, float y) { return ImVec2(o.x + (kPadMarginX + x * kPadScale) * u, o.y + (kPadMarginY + y * kPadScale) * u); };
    auto rgba = [&](int r, int g, int b, float a = 1.0f) { return IM_COL32(r, g, b, static_cast<int>(255.0f * a * alpha)); };
    auto theme = [&](const ImVec4& c, float a = 1.0f) { return col(c, a * alpha); };
    auto pressed = [&](std::uint16_t m) { return (s.buttons & m) != 0; };

    // Vertical gradient over the vertices added since v0.
    auto shade = [&](int v0, float y0, float y1, ImU32 top, ImU32 bot) {
        ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, ImVec2(0, y0), ImVec2(0, y1), top, bot);
    };
    auto white = [&](ImU32 c) { return (c & IM_COL32_A_MASK) | 0x00FFFFFFu; };
    auto poly_grad = [&](const ImVec2* pts, int n, ImU32 top, ImU32 bot) {
        float y0 = FLT_MAX, y1 = -FLT_MAX;
        for (int i = 0; i < n; ++i) y0 = std::min(y0, pts[i].y), y1 = std::max(y1, pts[i].y);
        int v0 = dl->VtxBuffer.Size;
        dl->AddConcavePolyFilled(pts, n, white(top));
        shade(v0, y0, y1, top, bot);
    };
    auto circle_grad = [&](ImVec2 c, float r, ImU32 top, ImU32 bot) {
        int v0 = dl->VtxBuffer.Size;
        dl->AddCircleFilled(c, r, white(top), 0);
        shade(v0, c.y - r, c.y + r, top, bot);
    };
    auto rect_grad = [&](ImVec2 a, ImVec2 b, float rounding, ImU32 top, ImU32 bot, float y0, float y1) {
        int v0 = dl->VtxBuffer.Size;
        dl->AddRectFilled(a, b, white(top), rounding);
        shade(v0, y0, y1, top, bot);
    };
    auto label = [&](ImFont* f, float size, ImVec2 c, ImU32 color, const char* t) {
        ImVec2 ts = f->CalcTextSizeA(size, FLT_MAX, 0, t);
        dl->AddText(f, size, ImVec2(std::floor(c.x - ts.x * 0.5f), std::floor(c.y - ts.y * 0.5f)), color, t);
    };
    // Round face button: drop shadow, domed face, gloss, a halo when held.
    auto face_button = [&](ImVec2 c, float r, bool on, ImU32 top, ImU32 bot, ImU32 halo) {
        if (on) {
            dl->AddCircleFilled(c, r + 2.0f * bu, (halo & ~IM_COL32_A_MASK) | IM_COL32(0, 0, 0, static_cast<int>(80 * alpha)), 0);
            c.y += 0.4f * bu;
        } else {
            dl->AddCircleFilled(ImVec2(c.x, c.y + 0.8f * bu), r + 0.2f * bu, rgba(40, 42, 50, 0.45f), 0);
        }
        circle_grad(c, r, top, bot);
        dl->AddCircle(c, r, rgba(0, 0, 0, 0.3f), 0, dp(1));
        dl->AddCircleFilled(ImVec2(c.x - r * 0.28f, c.y - r * 0.42f), r * 0.36f, rgba(255, 255, 255, on ? 0.32f : 0.24f), 0);
        return c;
    };

    const ImU32 edge = rgba(16, 17, 22, 0.9f);
    ImFont* fb = g_fonts.small_bold;
    const float fbs = font_px(fb);

    // ---- Shell outline: mirror the traced half into a closed clockwise loop,
    // then smooth it with a Catmull-Rom curve.
    constexpr int kHalf = IM_ARRAYSIZE(kPadLeft);
    constexpr int kLoop = 2 * kHalf - 2;
    auto loop_point = [&](int i) {
        i = (i % kLoop + kLoop) % kLoop;
        if (i < kHalf) return ImVec2(100.0f - kPadLeft[i].x, kPadLeft[i].y); // right half, top to bottom
        return kPadLeft[kLoop - i];                                          // left half, bottom to top
    };
    // Loop index of a kPadLeft point on the right (mirrored) or left side.
    auto right_index = [](int i) { return i; };
    auto left_index = [](int i) { return i == 0 ? 0 : kLoop - i; };
    ImVector<ImVec2> body;
    body.reserve(kLoop * kPadCurveSteps);
    for (int i = 0; i < kLoop; ++i) {
        ImVec2 p0 = loop_point(i - 1), p1 = loop_point(i), p2 = loop_point(i + 1), p3 = loop_point(i + 2);
        for (int k = 0; k < kPadCurveSteps; ++k) {
            float t = static_cast<float>(k) / kPadCurveSteps, t2 = t * t, t3 = t2 * t;
            auto cr = [&](float a, float b, float c, float d) {
                return 0.5f * (2.0f * b + (c - a) * t + (2.0f * a - 5.0f * b + 4.0f * c - d) * t2 + (3.0f * b - a - 3.0f * c + d) * t3);
            };
            body.push_back(B(cr(p0.x, p1.x, p2.x, p3.x), cr(p0.y, p1.y, p2.y, p3.y)));
        }
    }
    // Outward normal of the smoothed outline at sample i (the loop runs clockwise).
    auto normal = [&](int i) {
        ImVec2 a = body[(i + body.Size - 1) % body.Size], b = body[(i + 1) % body.Size];
        ImVec2 d(b.x - a.x, b.y - a.y);
        float len = std::sqrt(d.x * d.x + d.y * d.y);
        return len > 0 ? ImVec2(d.y / len, -d.x / len) : ImVec2(0, -1);
    };

    // ---- Shoulder buttons: bands hugging the shell between the hump and the
    // outer corner, peeking out from behind it.
    auto shoulder = [&](int from, int to, bool on, const char* t) {
        if (from > to) std::swap(from, to);
        ImVector<ImVec2> band;
        const float depth = (on ? 3.6f : 4.4f) * bu;
        for (int i = from * kPadCurveSteps; i <= to * kPadCurveSteps; ++i) {
            ImVec2 n = normal(i);
            band.push_back(ImVec2(body[i].x + n.x * depth, body[i].y + n.y * depth));
        }
        for (int i = to * kPadCurveSteps; i >= from * kPadCurveSteps; --i) {
            ImVec2 n = normal(i);
            band.push_back(ImVec2(body[i].x - n.x * bu, body[i].y - n.y * bu));
        }
        // The band must wind clockwise for anti-aliasing; flip it if not.
        float area = 0.0f;
        for (int i = 0; i < band.Size; ++i) {
            const ImVec2 &a = band[i], &b = band[(i + 1) % band.Size];
            area += a.x * b.y - b.x * a.y;
        }
        if (area < 0) std::reverse(band.begin(), band.end());
        if (on) poly_grad(band.Data, band.Size, theme(g_pal.accent_hover), theme(g_pal.accent_active));
        else poly_grad(band.Data, band.Size, rgba(150, 153, 162), rgba(104, 107, 116));
        dl->AddPolyline(band.Data, band.Size, edge, ImDrawFlags_Closed, dp(1));
        const int mid = (from + to) * kPadCurveSteps / 2;
        ImVec2 n = normal(mid);
        ImVec2 c(body[mid].x + n.x * depth * 0.5f, body[mid].y + n.y * depth * 0.5f);
        label(fb, fbs, c, on ? rgba(255, 255, 255) : rgba(40, 42, 50), t);
    };
    shoulder(left_index(kPadShoulderBegin), left_index(kPadShoulderEnd), pressed(Button::L), "L");
    shoulder(right_index(kPadShoulderBegin), right_index(kPadShoulderEnd), pressed(Button::R), "R");

    // ---- Shell: soft drop shadow, gradient, silhouette, rim light
    for (int i = 3; i >= 1; --i) {
        ImVector<ImVec2> sh = body;
        for (ImVec2& v : sh) v.y += i * 0.7f * bu;
        dl->AddConcavePolyFilled(sh.Data, sh.Size, rgba(0, 0, 0, 0.16f));
    }
    poly_grad(body.Data, body.Size, rgba(202, 204, 211), rgba(146, 149, 158));
    dl->AddPolyline(body.Data, body.Size, edge, ImDrawFlags_Closed, dp(1.3f));
    {
        const int a = left_index(kPadHumpEnd) * kPadCurveSteps, b = right_index(kPadHumpEnd) * kPadCurveSteps;
        dl->PathClear();
        for (int i = a; i != b; i = (i + 1) % body.Size) dl->PathLineTo(ImVec2(body[i].x, body[i].y + 0.8f * bu));
        dl->PathStroke(rgba(255, 255, 255, 0.5f), 0, dp(1));
    }
    // Recessed badge on the hump.
    {
        ImVec2 a = B(40.5f, 7.8f), b = B(59.5f, 13.4f);
        const float r = (b.y - a.y) * 0.5f;
        rect_grad(a, b, r, rgba(160, 163, 172), rgba(196, 198, 206), a.y, b.y);
        dl->AddRect(a, b, rgba(90, 93, 102, 0.6f), r, 0, dp(1));
    }

    // ---- D-pad
    {
        ImVec2 dc = B(20.0f, 30.5f);
        const float a = 2.9f * bu, l = 8.4f * bu, rr = 1.0f * bu;
        const float y0 = dc.y - l, y1 = dc.y + l;
        auto cross = [&](float dy, ImU32 top, ImU32 bot) {
            rect_grad(ImVec2(dc.x - a, dc.y - l + dy), ImVec2(dc.x + a, dc.y + l + dy), rr, top, bot, y0 + dy, y1 + dy);
            rect_grad(ImVec2(dc.x - l, dc.y - a + dy), ImVec2(dc.x + l, dc.y + a + dy), rr, top, bot, y0 + dy, y1 + dy);
        };
        circle_grad(dc, 10.4f * bu, rgba(176, 179, 188), rgba(204, 206, 213)); // shallow dish
        cross(0.9f * bu, rgba(30, 32, 38, 0.5f), rgba(30, 32, 38, 0.5f));
        cross(0.0f, rgba(110, 113, 122), rgba(62, 65, 73));
        struct Arm {
            std::uint16_t mask;
            float dx, dy;
        };
        const Arm arms[] = {{Button::D_UP, 0, -1}, {Button::D_DOWN, 0, 1}, {Button::D_LEFT, -1, 0}, {Button::D_RIGHT, 1, 0}};
        for (const Arm& arm : arms) {
            const bool on = pressed(arm.mask);
            if (on) {
                ImVec2 mn(dc.x + (arm.dx != 0 ? std::min(arm.dx * a, arm.dx * l) : -a), dc.y + (arm.dy != 0 ? std::min(arm.dy * a, arm.dy * l) : -a));
                ImVec2 mx(dc.x + (arm.dx != 0 ? std::max(arm.dx * a, arm.dx * l) : a), dc.y + (arm.dy != 0 ? std::max(arm.dy * a, arm.dy * l) : a));
                rect_grad(mn, mx, rr, theme(g_pal.accent_hover), theme(g_pal.accent_active), mn.y, mx.y);
            }
            // Embossed arrow near the end of each arm.
            ImVec2 tip(dc.x + arm.dx * 6.9f * bu, dc.y + arm.dy * 6.9f * bu);
            ImVec2 base(dc.x + arm.dx * 4.9f * bu, dc.y + arm.dy * 4.9f * bu);
            ImVec2 side(-arm.dy * 1.4f * bu, arm.dx * 1.4f * bu);
            dl->AddTriangleFilled(tip, ImVec2(base.x + side.x, base.y + side.y), ImVec2(base.x - side.x, base.y - side.y),
                                  on ? rgba(255, 255, 255, 0.95f) : rgba(20, 22, 28, 0.45f));
        }
        circle_grad(dc, 2.0f * bu, rgba(52, 55, 62), rgba(96, 99, 108)); // centre dimple
    }

    // ---- Start
    {
        const bool on = pressed(Button::START);
        face_button(B(50.3f, 32.0f), 3.4f * bu, on, on ? rgba(255, 110, 105) : rgba(222, 52, 56), on ? rgba(232, 56, 58) : rgba(150, 22, 30),
                    rgba(255, 90, 90));
        label(g_fonts.small, font_px(g_fonts.small) * 0.72f, B(50.3f, 38.0f), rgba(70, 72, 82), "START");
    }

    // ---- Analog stick: dark collar, octagonal gate, light concave cap
    {
        ImVec2 ac = B(50.0f, 53.0f);
        circle_grad(ac, 9.6f * bu, rgba(98, 101, 114), rgba(132, 135, 148));
        dl->AddCircle(ac, 9.6f * bu, rgba(40, 42, 52, 0.7f), 0, dp(1));
        const float gr = 7.4f * bu;
        ImVec2 gate[8];
        for (int i = 0; i < 8; ++i) {
            float ang = (i * 45.0f - 90.0f + 22.5f) * 3.14159265f / 180.0f;
            gate[i] = ImVec2(ac.x + std::cos(ang) * gr, ac.y + std::sin(ang) * gr);
        }
        poly_grad(gate, 8, rgba(46, 48, 58), rgba(82, 85, 97));
        const float sx = std::clamp(s.stick_x / 80.0f, -1.0f, 1.0f), sy = std::clamp(s.stick_y / 80.0f, -1.0f, 1.0f);
        const bool moved = s.stick_x != 0 || s.stick_y != 0;
        ImVec2 knob(ac.x + sx * 3.0f * bu, ac.y - sy * 3.0f * bu);
        if (moved) {
            dl->AddLine(ac, knob, theme(g_pal.accent, 0.9f), dp(2));
            dl->AddCircleFilled(ac, 0.9f * bu, theme(g_pal.accent), 12);
        }
        const float kr = 4.6f * bu;
        dl->AddCircleFilled(ImVec2(knob.x, knob.y + 0.9f * bu), kr, rgba(0, 0, 0, 0.45f), 0);
        circle_grad(knob, kr, rgba(236, 237, 240), rgba(176, 178, 186));
        dl->AddCircle(knob, kr, moved ? theme(g_pal.accent) : rgba(60, 62, 72, 0.8f), 0, dp(moved ? 1.6f : 1.0f));
        circle_grad(knob, kr * 0.62f, rgba(182, 184, 192), rgba(232, 233, 237)); // concave top
        dl->AddCircle(knob, kr * 0.62f, rgba(0, 0, 0, 0.18f), 0, dp(1));
        circle_grad(knob, kr * 0.26f, rgba(160, 162, 170), rgba(214, 215, 220));
    }

    // ---- Z trigger (under the centre prong; shown on top of it)
    {
        const bool on = pressed(Button::Z);
        ImVec2 a = B(45.0f, on ? 74.4f : 74.0f), b = B(55.0f, on ? 80.4f : 80.0f);
        const float r = (b.y - a.y) * 0.5f;
        if (!on) dl->AddRectFilled(ImVec2(a.x, a.y + 0.7f * bu), ImVec2(b.x, b.y + 0.7f * bu), rgba(40, 42, 50, 0.4f), r);
        if (on) rect_grad(a, b, r, theme(g_pal.accent_hover), theme(g_pal.accent_active), a.y, b.y);
        else rect_grad(a, b, r, rgba(150, 153, 162), rgba(112, 115, 124), a.y, b.y);
        dl->AddRect(a, b, rgba(30, 32, 40, 0.6f), r, 0, dp(1));
        label(fb, fbs, ImVec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f), on ? rgba(255, 255, 255) : rgba(40, 42, 50), "Z");
    }

    // ---- B and A
    {
        const bool bon = pressed(Button::B);
        ImVec2 bc = face_button(B(68.3f, 32.0f), 3.8f * bu, bon, bon ? rgba(120, 236, 150) : rgba(46, 184, 88), bon ? rgba(46, 186, 90) : rgba(18, 112, 50),
                                rgba(80, 220, 120));
        label(fb, fbs, bc, rgba(255, 255, 255, bon ? 1.0f : 0.8f), "B");
        const bool aon = pressed(Button::A);
        ImVec2 acn = face_button(B(76.1f, 38.6f), 3.9f * bu, aon, aon ? rgba(140, 180, 255) : rgba(62, 116, 232), aon ? rgba(70, 122, 246) : rgba(26, 60, 160),
                                 rgba(100, 150, 255));
        label(fb, fbs, acn, rgba(255, 255, 255, aon ? 1.0f : 0.8f), "A");
    }

    // ---- C buttons
    {
        ImVec2 cc = B(83.5f, 25.8f);
        const float off = 5.8f * bu, r = 2.75f * bu;
        struct CBtn {
            std::uint16_t mask;
            float dx, dy;
        };
        label(g_fonts.small, font_px(g_fonts.small) * 0.7f, cc, rgba(80, 82, 92), "C");
        const CBtn cs[] = {{Button::C_UP, 0, -1}, {Button::C_DOWN, 0, 1}, {Button::C_LEFT, -1, 0}, {Button::C_RIGHT, 1, 0}};
        for (const CBtn& c : cs) {
            const bool on = pressed(c.mask);
            ImVec2 p = face_button(ImVec2(cc.x + c.dx * off, cc.y + c.dy * off), r, on, on ? rgba(255, 236, 130) : rgba(252, 204, 40),
                                   on ? rgba(250, 196, 40) : rgba(200, 136, 8), rgba(255, 215, 70));
            ImVec2 tip(p.x + c.dx * 1.2f * bu, p.y + c.dy * 1.2f * bu);
            ImVec2 base(p.x - c.dx * 0.8f * bu, p.y - c.dy * 0.8f * bu);
            ImVec2 side(-c.dy * 1.1f * bu, c.dx * 1.1f * bu);
            dl->AddTriangleFilled(tip, ImVec2(base.x + side.x, base.y + side.y), ImVec2(base.x - side.x, base.y - side.y), rgba(140, 92, 0, 0.75f));
        }
    }

    ImGui::Dummy(ImVec2(W, H + dp(4)));
}

namespace {
fs::path controller_profiles_dir() { return platform::config_dir() / "controller_profiles"; }

// A profile name usable as a file name on every platform.
std::string sanitize_profile_name(const char* in) {
    std::string out;
    for (const char* c = in; *c; ++c) {
        const unsigned char ch = static_cast<unsigned char>(*c);
        if (ch < 32 || std::strchr("<>:\"/\\|?*", ch)) continue;
        out += *c;
    }
    const size_t a = out.find_first_not_of(" ."), b = out.find_last_not_of(" .");
    out = a == std::string::npos ? std::string() : out.substr(a, b - a + 1);
    if (out.size() > 48) out.resize(48); // may cut a UTF-8 sequence only in pathological names
    return out;
}

bool same_name(const std::string& a, const std::string& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
               return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
           });
}
} // namespace

void App::scan_controller_profiles() {
    profiles_.clear();
    profiles_scanned_ = true;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(controller_profiles_dir(), ec)) {
        if (!e.is_regular_file(ec) || e.path().extension() != ".ini") continue;
        ControllerProfile p;
        p.name = platform::path_to_utf8(e.path().stem());
        p.cfg.keys.fill(0);
        p.cfg.pad.fill(-1);
        if (load_controller_profile(p.cfg, platform::path_to_utf8(e.path()))) profiles_.push_back(std::move(p));
    }
    std::sort(profiles_.begin(), profiles_.end(), [](const ControllerProfile& a, const ControllerProfile& b) {
        return std::lexicographical_compare(a.name.begin(), a.name.end(), b.name.begin(), b.name.end(), [](char x, char y) {
            return std::tolower(static_cast<unsigned char>(x)) < std::tolower(static_cast<unsigned char>(y));
        });
    });
}

void App::controller_profile_row(PortConfig& pc, float cw) {
    if (!profiles_scanned_) scan_controller_profiles();
    row_begin("Profile", "Save this port's bindings, dead zone and stick options under a name, and load them into any port.", cw);
    int match = -1;
    for (int i = 0; i < static_cast<int>(profiles_.size()); ++i)
        if (same_controller_profile(profiles_[i].cfg, pc)) {
            match = i;
            break;
        }
    const float gap = dp(6), save_w = dp(72);
    const float del_w = ImGui::GetFontSize() + dp(14); // square, as tall as the combo
    std::vector<const char*> items;
    if (match < 0) items.push_back(profiles_.empty() ? "No saved profiles" : "Not saved");
    for (const auto& p : profiles_) items.push_back(p.name.c_str());
    int cur = match < 0 ? 0 : match;
    if (combo("profile", &cur, items.data(), static_cast<int>(items.size()), cw - save_w - del_w - gap * 2)) {
        const int idx = match < 0 ? cur - 1 : cur;
        if (idx >= 0 && idx < static_cast<int>(profiles_.size())) {
            apply_controller_profile(pc, profiles_[idx].cfg);
            toast("Loaded \xE2\x80\x9C" + profiles_[idx].name + "\xE2\x80\x9D into Port " + std::to_string(settings_port_ + 1), ToastKind::Success);
        }
    }
    ImGui::SameLine(0, gap);
    if (button("Save\xE2\x80\xA6", Icon::None, ButtonKind::Subtle, save_w, true, del_w)) {
        std::snprintf(profile_name_, sizeof profile_name_, "%s", match >= 0 ? profiles_[match].name.c_str() : "");
        ImGui::OpenPopup("##save_profile");
    }
    ImGui::SameLine(0, gap);
    if (icon_button("delete_profile", Icon::Trash, del_w, match >= 0 ? "Delete this profile" : nullptr, false, match >= 0,
                    ButtonKind::Subtle) && match >= 0) {
        profile_to_delete_ = profiles_[match].name;
        ImGui::OpenPopup("##delete_profile");
    }
    // Both popups open under the row's controls, right-aligned with them.
    const ImVec2 anchor(ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y + dp(6));

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(14, 12));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, dp(10));
    ImGui::SetNextWindowPos(anchor, ImGuiCond_Appearing, ImVec2(1, 0));
    if (ImGui::BeginPopup("##save_profile")) {
        const float w = dp(260);
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::TextColored(g_pal.text_faint, "SAVE PROFILE");
        ImGui::PopFont();
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(w);
        const bool enter = ImGui::InputTextWithHint("##name", "Profile name", profile_name_, sizeof profile_name_,
                                                    ImGuiInputTextFlags_EnterReturnsTrue);
        const std::string name = sanitize_profile_name(profile_name_);
        const bool exists = std::any_of(profiles_.begin(), profiles_.end(), [&](const ControllerProfile& p) { return same_name(p.name, name); });
        ImGui::PushFont(g_fonts.small);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
        ImGui::TextColored(exists ? g_pal.warning : g_pal.text_faint, "%s",
                           exists ? "Replaces the saved profile with this name."
                                  : "Bindings, dead zone, range, stick options and rumble strength.");
        ImGui::PopTextWrapPos();
        ImGui::PopFont();
        if ((button("Save profile", Icon::Check, ButtonKind::Primary, w, !name.empty()) || enter) && !name.empty()) {
            std::error_code ec;
            fs::create_directories(controller_profiles_dir(), ec);
            const fs::path path = controller_profiles_dir() / platform::utf8_to_path(name + ".ini");
            if (save_controller_profile(pc, platform::path_to_utf8(path))) {
                toast("Saved profile \xE2\x80\x9C" + name + "\xE2\x80\x9D", ToastKind::Success);
                scan_controller_profiles();
            } else {
                toast("Couldn't save the profile", ToastKind::Error);
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    ImGui::SetNextWindowPos(anchor, ImGuiCond_Appearing, ImVec2(1, 0));
    if (ImGui::BeginPopup("##delete_profile")) {
        const float w = dp(240);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
        ImGui::Text("Delete \xE2\x80\x9C%s\xE2\x80\x9D?", profile_to_delete_.c_str());
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_faint, "The profile file is removed. Ports using it keep their current settings.");
        ImGui::PopFont();
        ImGui::PopTextWrapPos();
        if (button("Delete", Icon::Trash, ButtonKind::Danger, w)) {
            std::error_code ec;
            fs::remove(controller_profiles_dir() / platform::utf8_to_path(profile_to_delete_ + ".ini"), ec);
            if (ec) toast("Couldn't delete the profile", ToastKind::Error);
            else toast("Deleted profile \xE2\x80\x9C" + profile_to_delete_ + "\xE2\x80\x9D");
            scan_controller_profiles();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar(2);
    row_end();
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

    controller_profile_row(pc, cw);

    row_begin("Accessory", "Controller Pak: game saves, kept in a .mpk file next to the ROM. Rumble Pak: vibrates your gamepad. "
              "Transfer Pak: a Game Boy cartridge (Pok\xC3\xA9mon Stadium, Mario Golf...). Games check the slot when they start.", cw);
    {
        const char* items[] = {"None", "Controller Pak", "Rumble Pak", "Transfer Pak"};
        combo("pak", &pc.pak, items, 4, cw);
    }
    row_end();

    if (pc.pak == 3) {
        row_begin("Game Boy cartridge", pc.gb_rom.empty() ? "Choose a .gb or .gbc ROM. Its save is the .sav file next to it, shared with Game Boy emulators."
                                                          : pc.gb_rom.c_str(), cw);
        std::string name = pc.gb_rom.empty() ? "Choose ROM\xE2\x80\xA6" : platform::path_to_utf8(platform::utf8_to_path(pc.gb_rom).filename());
        const float eject_w = pc.gb_rom.empty() ? 0.0f : dp(34) + dp(8);
        if (button(name.c_str(), Icon::FolderOpen, ButtonKind::Subtle, cw - eject_w)) action_choose_gb_rom(settings_port_);
        if (!pc.gb_rom.empty()) {
            ImGui::SameLine(0, dp(8));
            if (icon_button("eject", Icon::Close, dp(34), "Remove the cartridge")) pc.gb_rom.clear();
        }
        row_end();
    }

    if (pc.pak == 2) {
        const bool can = input_.can_rumble(pc.device);
        const bool connected = pc.device > 0 && pc.device <= static_cast<int>(input_.gamepads().size());
        row_begin("Rumble strength", can ? nullptr
                  : pc.device == 0 ? "Select a gamepad as the input device to feel the Rumble Pak."
                  : !connected     ? "Connect the gamepad to feel the Rumble Pak."
                                   : "This gamepad doesn't report vibration support.", cw);
        slider_int("rumble", &pc.rumble_strength, 0, 100, "%d%%", cw * 0.62f);
        ImGui::SameLine(0, dp(8));
        if (button("Test", Icon::Play, ButtonKind::Subtle, cw * 0.38f - dp(8), can && pc.rumble_strength > 0))
            input_.test_rumble(pc);
        row_end();
    }

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
    {
        char pos[32];
        std::snprintf(pos, sizeof pos, "X %+d  Y %+d", live.stick_x, live.stick_y);
        ImFont* f = g_fonts.mono_small;
        float tw = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, pos).x;
        ImGui::SameLine(0, 0);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(dp(8), diag_w - tw - ImGui::GetItemRectSize().x));
        ImGui::PushFont(f);
        ImGui::TextColored(live.stick_x || live.stick_y ? g_pal.accent_hover : g_pal.text_faint, "%s", pos);
        ImGui::PopFont();
    }
    ImGui::PushFont(g_fonts.small);
    ImGui::TextColored(g_pal.text_dim, "Dead zone");
    ImGui::PopFont();
    slider_float("dz", &pc.deadzone, 0.0f, 0.5f, "%.2f", diag_w);
    ImGui::PushFont(g_fonts.small);
    ImGui::TextColored(g_pal.text_dim, "Range");
    ImGui::PopFont();
    slider_float("sens", &pc.sensitivity, 0.5f, 1.5f, "%.2f\xC3\x97", diag_w);
    ImGui::Dummy(dp(0, 4));
    auto stick_option = [&](const char* id, bool* v, const char* text, const char* tip) {
        const float y = ImGui::GetCursorPosY();
        toggle(id, v);
        const bool hov = ImGui::IsItemHovered();
        ImGui::SameLine(0, dp(10));
        ImGui::SetCursorPosY(y + (dp(22) - ImGui::GetTextLineHeight()) * 0.5f);
        ImGui::TextColored(g_pal.text_dim, "%s", text);
        if (hov || ImGui::IsItemHovered()) tooltip(tip);
    };
    stick_option("octagon", &pc.octagon, "Octagonal gate",
                 "Limit the stick to the N64's octagonal range: \xC2\xB1""80 along the axes and 70 on each axis in the corners, "
                 "like the original controller. Off: each axis reaches \xC2\xB1""80 on its own (a square range).");
    stick_option("dzscale", &pc.deadzone_rescale, "Rescale past dead zone",
                 "Stretch the travel past the dead zone back to the full range, so movement starts smoothly from zero. "
                 "Off: the stick jumps straight to the dead-zone value as it leaves the dead zone.");
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
    bool any_bind_hovered = false;
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
        any_bind_hovered |= hov;
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
        if (clicked) {
            hotkey_capture_ = -1;
            input_.begin_capture(settings_port_, i);
        }
        if (right) {
            if (keyboard) pc.keys[i] = 0;
            else pc.pad[i] = -1;
        }
        if (hov && !capturing) tooltip("Click to rebind \xC2\xB7 right-click to clear \xC2\xB7 Esc cancels");
    }
    // A click anywhere else stops recording and keeps the previous binding.
    if (input_.capturing() && !any_bind_hovered && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Right)))
        input_.cancel_capture();
    int rows = (kN64InputCount + cols - 1) / cols;
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + rows * (rh + dp(4))));
    ImGui::Dummy(ImVec2(list_w, dp(4)));
    // Wrap to the column so the page never gets wider than the window.
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + list_w);
    if (!keyboard && input_.gamepads().empty()) {
        ImGui::TextColored(g_pal.warning, "No gamepads detected. Connect one \xE2\x80\x94 it will appear automatically.");
    } else {
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_faint, "%d gamepad%s detected. XInput, DualShock/DualSense, Switch Pro and most USB pads are supported.",
                           (int)input_.gamepads().size(), input_.gamepads().size() == 1 ? "" : "s");
        ImGui::PopFont();
    }
    ImGui::PopTextWrapPos();
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

    row_begin("Expansion Pak", "The 4 MB memory upgrade (8 MB in total). Some games need it (Majora's Mask, Donkey Kong 64, Perfect Dark's campaign) "
              "and others use it for a high-resolution mode. Applies when a game starts or is reset.", cw);
    if (toggle("expak", &settings_.expansion_pak) && core_.loaded()) toast("Reset the game to apply", ToastKind::Info);
    row_end();

    row_begin("RSP / RDP emulation",
              "High-level (HLE) recreates what the graphics and audio microcodes do: fastest. Low-level (LLE) runs the game's own "
              "microcode on an emulated RSP and draws the RDP commands it produces, so any microcode works, at a higher CPU cost. "
              "Graphics microcodes HLE doesn't recognize always run low-level.", cw);
    {
        const char* items[] = {"HLE (fast)", "LLE graphics, HLE audio", "LLE graphics and audio (accurate)"};
        if (combo("rspmode", &settings_.rsp_mode, items, 3, cw)) core_.set_rsp_mode(settings_.rsp_mode);
    }
    row_end();

    row_begin("Low-level RDP",
              "How low-level graphics are drawn. Bit-exact reproduces the console's RDP pixel for pixel (dithering, "
              "coverage, depth precision); at a higher internal resolution the game still sees the exact picture "
              "and the screen shows the same pipeline drawn at that resolution. Bit-exact (GPU) draws the same "
              "pixels, internal resolution included, with compute shaders on the graphics card, leaving the CPU to "
              "the rest of the console (needs the GPU video backend). Fast draws them through the high-level "
              "renderer, which is quicker but not exact.", cw);
    {
        const char* items[] = {"Bit-exact (GPU)", "Bit-exact (CPU)", "Fast"};
        int cur = !settings_.rdp_exact ? 2 : settings_.rdp_exact_gpu ? 0 : 1;
        if (combo("rdpexact", &cur, items, 3, cw)) {
            settings_.rdp_exact = cur != 2;
            if (cur != 2) settings_.rdp_exact_gpu = cur == 0;
            core_.set_rdp_exact(settings_.rdp_exact);
            core_.set_rdp_exact_gpu(settings_.rdp_exact_gpu);
        }
    }
    row_end();

    row_begin("Graphics microcode", "Auto-detect works for nearly every game. Override only if a game renders incorrectly.", cw);
    {
        const char* items[] = {"Auto-detect", "Fast3D", "F3DEX", "F3DEX2", "S2DEX", "S2DEX2", "F3DEX (GoldenEye)", "F3D (Perfect Dark)",
                               "F3D (Diddy Kong Racing)", "F3D (Jet Force Gemini)", "F3D (Wave Race 64)"};
        if (combo("ucode", &settings_.ucode_override, items, 11, cw)) {
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
    const bool mac = platform::current_os() == platform::OS::MacOS;
    float w = ImGui::GetContentRegionAvail().x;
    ImGui::PushFont(g_fonts.small);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
    ImGui::TextColored(g_pal.text_dim, "Click a shortcut and press the new key combination. Right-click clears it, Esc cancels. %s",
                       mac ? "Shortcuts with Cmd use Ctrl on Windows and Linux." : "Shortcuts with Ctrl use Cmd on macOS.");
    ImGui::PopTextWrapPos();
    ImGui::PopFont();
    ImGui::Dummy(dp(0, 8));

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float cap_w = std::clamp((w - dp(240)) * 0.5f, dp(110), dp(170));
    const float cap_gap = dp(8);
    // Draws a key cap button; returns 1 = clicked, 2 = right-clicked.
    bool any_cap_hovered = false;
    auto key_cap = [&](const char* id, ImVec2 p, float cw, float ch, const std::string& text, bool capturing) {
        ImGui::SetCursorScreenPos(p);
        const bool clicked = ImGui::InvisibleButton(id, ImVec2(cw, ch));
        const bool hov = ImGui::IsItemHovered();
        any_cap_hovered |= hov;
        const bool right = ImGui::IsItemClicked(ImGuiMouseButton_Right);
        const bool unbound = !capturing && text.empty();
        float pulse = capturing ? 0.5f + 0.5f * std::sin(static_cast<float>(ImGui::GetTime()) * 6.0f) : 0.0f;
        ImVec2 mx(p.x + cw, p.y + ch);
        if (!capturing && !unbound) dl->AddRectFilled(p, ImVec2(mx.x, mx.y + dp(2)), col(g_pal.border_strong), dp(6));
        dl->AddRectFilled(p, mx, col(capturing ? mix(g_pal.bg3, g_pal.accent_soft, 1.0f) : unbound ? (hov ? g_pal.bg3 : g_pal.bg1)
                                                                                           : (hov ? mix(g_pal.bg4, g_pal.text, 0.06f) : g_pal.bg4)),
                          dp(6));
        if (capturing) dl->AddRect(p, mx, col(with_alpha(g_pal.accent, 0.5f + 0.5f * pulse)), dp(6));
        else if (unbound) dl->AddRect(p, mx, col(g_pal.border), dp(6));
        ImFont* f = capturing || unbound ? g_fonts.small : g_fonts.mono_small;
        const char* t = capturing ? "Press keys\xE2\x80\xA6" : unbound ? "Add" : text.c_str();
        ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, t);
        if (ts.x > cw - dp(12)) text_ellipsis(dl, f, ImVec2(p.x + dp(6), p.y + (ch - ts.y) * 0.5f), cw - dp(12), col(g_pal.text), t);
        else dl->AddText(f, font_px(f), ImVec2(p.x + (cw - ts.x) * 0.5f, p.y + (ch - ts.y) * 0.5f),
                         col(capturing ? g_pal.accent_hover : unbound ? g_pal.text_faint : g_pal.text), t);
        if (hov && !capturing) tooltip(unbound ? "Click to add a shortcut" : "Click to change \xC2\xB7 right-click to clear");
        return clicked ? 1 : right ? 2 : 0;
    };

    for (int h = 0; h < kHotkeyCount; ++h) {
        // A plain key that is also a game control on a keyboard port does both.
        std::string warn;
        for (const KeyCombo& c : settings_.hotkeys[h]) {
            if (!c.key || c.mods || !warn.empty()) continue;
            const SDL_Scancode sc = SDL_GetScancodeFromKey(static_cast<SDL_Keycode>(c.key), nullptr);
            for (int port = 0; port < 4 && warn.empty(); ++port) {
                const PortConfig& pc = settings_.ports[port];
                if (!pc.plugged || pc.device != 0) continue;
                for (int i = 0; i < kN64InputCount; ++i)
                    if (pc.keys[i] == static_cast<int>(sc)) {
                        warn = key_combo_label(c) + " is also " + n64_input_name(static_cast<N64Input>(i)) + " on Port " + std::to_string(port + 1);
                        break;
                    }
            }
        }
        ImVec2 p = ImGui::GetCursorScreenPos();
        const float h_row = warn.empty() ? dp(44) : dp(60);
        if (h % 2 == 0) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h_row), col(g_pal.bg2), dp(6));
        const float name_y = p.y + (dp(44) - font_px(g_fonts.body)) * 0.5f;
        dl->AddText(g_fonts.body, font_px(g_fonts.body), ImVec2(p.x + dp(14), name_y), col(g_pal.text), hotkey_name(static_cast<Hotkey>(h)));
        if (!warn.empty())
            dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(p.x + dp(14), name_y + font_px(g_fonts.body) + dp(4)), col(g_pal.warning),
                        warn.c_str());
        for (int slot = 0; slot < kHotkeySlots; ++slot) {
            const float x = p.x + w - dp(8) - (kHotkeySlots - slot) * cap_w - (kHotkeySlots - 1 - slot) * cap_gap;
            ImGui::PushID(h * kHotkeySlots + slot);
            const bool capturing = hotkey_capture_ == h * kHotkeySlots + slot;
            const int r = key_cap("##hk", ImVec2(x, p.y + dp(7)), cap_w, dp(30), key_combo_label(settings_.hotkeys[h][slot]), capturing);
            ImGui::PopID();
            if (r == 1) {
                input_.cancel_capture();
                hotkey_capture_ = capturing ? -1 : h * kHotkeySlots + slot;
            } else if (r == 2) {
                settings_.hotkeys[h][slot] = {};
                if (capturing) hotkey_capture_ = -1;
                save_settings();
            }
        }
        ImGui::SetCursorScreenPos(p);
        ImGui::Dummy(ImVec2(w, h_row));
    }

    // A click anywhere else stops recording and keeps the previous shortcut.
    if (hotkey_capture_ >= 0 && !any_cap_hovered && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Right)))
        hotkey_capture_ = -1;

    // Fixed shortcuts.
    ImGui::Dummy(dp(0, 10));
    ImGui::PushFont(g_fonts.small_bold);
    ImGui::TextColored(g_pal.text_faint, "FIXED");
    ImGui::PopFont();
    ImGui::Dummy(dp(0, 2));
    const std::string slot_keys = std::string(platform::primary_mod_name()) + "+1\xE2\x80\x93" "9";
    const struct {
        const char* action;
        std::string keys;
    } fixed[] = {{"Choose state slot", slot_keys}, {"Exit fullscreen", "Esc"}};
    for (size_t i = 0; i < std::size(fixed); ++i) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        const float h_row = dp(40);
        if (i % 2 == 0) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h_row), col(g_pal.bg2), dp(6));
        dl->AddText(g_fonts.body, font_px(g_fonts.body), ImVec2(p.x + dp(14), p.y + (h_row - font_px(g_fonts.body)) * 0.5f), col(g_pal.text_dim),
                    fixed[i].action);
        ImFont* f = g_fonts.mono_small;
        ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, fixed[i].keys.c_str());
        ImVec2 kmn(p.x + w - dp(8) - ts.x - dp(14), p.y + dp(9)), kmx(p.x + w - dp(8), p.y + h_row - dp(9));
        dl->AddRectFilled(kmn, ImVec2(kmx.x, kmx.y + dp(2)), col(g_pal.border_strong), dp(5));
        dl->AddRectFilled(kmn, kmx, col(g_pal.bg4), dp(5));
        dl->AddText(f, font_px(f), ImVec2(kmn.x + dp(7), (kmn.y + kmx.y) * 0.5f - ts.y * 0.5f), col(g_pal.text_dim), fixed[i].keys.c_str());
        ImGui::Dummy(ImVec2(w, h_row));
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
