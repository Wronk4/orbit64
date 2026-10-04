// Game screen: emulator viewport, pause overlay and the game info panel.

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

static const char* kUcodeLong[] = {"Auto", "Fast3D", "F3DEX", "F3DEX2", "S2DEX", "S2DEX2", "F3DEX (GoldenEye)", "F3D (Perfect Dark)", "F3D (Diddy Kong Racing)", "F3D (Jet Force Gemini)", "F3D (Wave Race 64)"};

void App::draw_game_view(ImVec2 pos, ImVec2 size) {
    ImGui::SetCursorScreenPos(pos);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::BeginChild("##game", size, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    const bool loaded = core_.loaded();
    const bool panel = settings_.show_info_panel && !fullscreen_ && loaded && size.x > dp(960);
    const float panel_w = panel ? dp(300) : 0.0f;
    // Slide the panel in/out smoothly.
    const float panel_anim = anim(ImHashStr("##info_panel_anim"), panel ? 1.0f : 0.0f, 14.0f);
    const float pw = dp(300) * panel_anim;
    (void)panel_w;

    ImVec2 vmn = pos, vmx(pos.x + size.x - pw, pos.y + size.y);
    dl->AddRectFilled(vmn, vmx, IM_COL32(5, 5, 7, 255));

    if (!loaded) {
        draw_no_game(vmn, ImVec2(vmx.x - vmn.x, vmx.y - vmn.y));
    } else {
        const float pad = fullscreen_ ? 0.0f : dp(10);
        ImVec2 amn(vmn.x + pad, vmn.y + pad), amx(vmx.x - pad, vmx.y - pad);
        float aw = amx.x - amn.x, ah = amx.y - amn.y;
        // Layout and scanlines follow the game's frame buffer, whatever the internal resolution.
        int tw = frame_native_w(), th = frame_native_h();

        float w = aw, h = ah;
        if (settings_.aspect == 3) {
            // Native: largest whole multiple of the frame buffer size.
            float k = std::max(1.0f, std::floor(std::min(aw / tw, ah / th)));
            w = tw * k;
            h = th * k;
        } else if (settings_.aspect != 2) {
            float ratio = settings_.aspect == 1 ? 16.0f / 9.0f : 4.0f / 3.0f;
            if (settings_.integer_scale) {
                float k = std::max(1.0f, std::floor(std::min(ah / th, aw / (th * ratio))));
                h = th * k;
                w = h * ratio;
            } else if (aw / ah > ratio) {
                h = ah;
                w = h * ratio;
            } else {
                w = aw;
                h = w / ratio;
            }
        }
        ImVec2 gmn(std::floor(amn.x + (aw - w) * 0.5f), std::floor(amn.y + (ah - h) * 0.5f));
        ImVec2 gmx(gmn.x + w, gmn.y + h);

        if (has_frame_ && game_tex_) {
            dl->AddImage((ImTextureID)(intptr_t)display_texture(), gmn, gmx);
            if (settings_.scanlines > 0) {
                // One dark line per emulated scanline, drawn at output resolution.
                float line_h = h / th;
                if (line_h >= 2.0f) {
                    ImU32 lc = IM_COL32(0, 0, 0, static_cast<int>(settings_.scanlines * 2.2f));
                    for (int y = 0; y < th; ++y) {
                        float yy = gmn.y + y * line_h + line_h * 0.5f;
                        dl->AddRectFilled(ImVec2(gmn.x, yy), ImVec2(gmx.x, yy + line_h * 0.5f), lc);
                    }
                }
            }
        } else {
            // Booting: frame buffer not yet valid.
            ImVec2 c((gmn.x + gmx.x) * 0.5f, (gmn.y + gmx.y) * 0.5f);
            ImGui::SetCursorScreenPos(ImVec2(c.x - dp(14), c.y - dp(24)));
            spinner(dp(14), dp(3), col(g_pal.accent));
            const char* msg = "Booting\xE2\x80\xA6";
            float mw = ImGui::CalcTextSize(msg).x;
            dl->AddText(ImVec2(c.x - mw * 0.5f, c.y + dp(12)), col(g_pal.text_dim), msg);
        }

        // Viewport interactions: double-click toggles fullscreen, right-click menu.
        ImGui::SetCursorScreenPos(vmn);
        // Allow overlap: the pause overlay's buttons are submitted after this
        // full-viewport item and must still receive hover and clicks.
        ImGui::SetNextItemAllowOverlap();
        ImGui::InvisibleButton("##viewport", ImVec2(vmx.x - vmn.x, vmx.y - vmn.y));
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) toggle_fullscreen();
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(8, 8));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, dp(10, 9));
        ImGui::PushStyleColor(ImGuiCol_PopupBg, g_pal.bg2);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, g_pal.bg4);
        ImGui::SetNextWindowSizeConstraints(ImVec2(dp(240), 0), ImVec2(FLT_MAX, FLT_MAX));
        if (ImGui::BeginPopupContextItem("##viewport_ctx")) {
            bool running = core_.state() == RunState::Running;
            if (ImGui::MenuItem(running ? "Pause" : "Resume", hotkey_label(Hotkey::Pause).c_str())) toggle_pause();
            if (ImGui::MenuItem("Reset", hotkey_label(Hotkey::Reset).c_str())) reset_game();
            if (ImGui::MenuItem("Stop", hotkey_label(Hotkey::Stop).c_str())) request_stop();
            ImGui::Separator();
            draw_state_menu();
            ImGui::Separator();
            if (ImGui::MenuItem("Take Screenshot", hotkey_label(Hotkey::Screenshot).c_str())) take_screenshot();
            if (ImGui::MenuItem(fullscreen_ ? "Exit Fullscreen" : "Fullscreen", hotkey_label(Hotkey::Fullscreen).c_str())) toggle_fullscreen();
            ImGui::Separator();
            if (ImGui::BeginMenu("Internal Resolution")) {
                internal_resolution_menu();
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Aspect Ratio")) {
                const char* names[] = {"4:3 (Original)", "16:9 (Widescreen)", "Stretch to Window", "Native Frame Size"};
                for (int i = 0; i < 4; ++i)
                    if (ImGui::MenuItem(names[i], nullptr, settings_.aspect == i)) settings_.aspect = i;
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Texture Filter")) {
                if (ImGui::MenuItem("Nearest", nullptr, settings_.filter == 0)) settings_.filter = 0;
                if (ImGui::MenuItem("Bilinear", nullptr, settings_.filter == 1)) settings_.filter = 1;
                ImGui::EndMenu();
            }
            ImGui::MenuItem("Show FPS Overlay", nullptr, &settings_.show_fps_overlay);
            ImGui::MenuItem("Ray Tracing", nullptr, &settings_.hle_raytracing);
            ImGui::MenuItem("Modern Post-processing", nullptr, &settings_.postfx);
            ImGui::Separator();
            if (ImGui::BeginMenu("Debug")) {
                draw_debug_menu(false);
                ImGui::EndMenu();
            }
            ImGui::MenuItem("Game Info Panel", nullptr, &settings_.show_info_panel);
            ImGui::EndPopup();
        }
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(2);

        // On-screen overlays
        CoreStats s = core_.stats();
        float oy = gmn.y + dp(12);
        if (settings_.show_fps_overlay) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%.0f FPS  \xC2\xB7  %.0f%%", s.fps, s.speed_pct);
            ImFont* f = g_fonts.mono;
            ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, buf);
            ImVec2 mn(gmn.x + dp(12), oy), mx(mn.x + ts.x + dp(16), oy + ts.y + dp(8));
            dl->AddRectFilled(mn, mx, IM_COL32(0, 0, 0, 150), dp(6));
            ImVec4 c = s.speed_pct >= 95 ? g_pal.success : s.speed_pct >= 70 ? g_pal.warning : g_pal.danger;
            dl->AddText(f, font_px(f), ImVec2(mn.x + dp(8), mn.y + dp(4)), col(c), buf);
        }
        if (core_.fast_forward() || core_.turbo()) {
            std::string label = settings_.ff_speed == 0 ? "FAST FORWARD  MAX" : "FAST FORWARD  " + std::to_string(settings_.ff_speed) + "\xC3\x97";
            ImFont* f = g_fonts.small_bold;
            ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, label.c_str());
            ImVec2 mx(gmx.x - dp(12), oy + ts.y + dp(10)), mn(mx.x - ts.x - dp(40), oy);
            dl->AddRectFilled(mn, mx, IM_COL32(0, 0, 0, 160), dp(6));
            draw_icon(dl, Icon::FastForward, ImVec2(mn.x + dp(14), (mn.y + mx.y) * 0.5f), dp(13), col(g_pal.warning));
            dl->AddText(f, font_px(f), ImVec2(mn.x + dp(28), mn.y + dp(5)), col(g_pal.warning), label.c_str());
        }

        float pause_t = anim(ImHashStr("##pause_overlay"), core_.state() == RunState::Paused ? 1.0f : 0.0f, 12.0f);
        if (pause_t > 0.01f) {
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, pause_t);
            draw_pause_overlay(vmn, vmx);
            ImGui::PopStyleVar();
        }
    }

    if (pw > 1.0f) draw_info_panel(ImVec2(vmx.x, pos.y), ImVec2(dp(300), size.y));
    ImGui::EndChild();
}

void App::draw_no_game(ImVec2 pos, ImVec2 size) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    // Subtle radial glow + grid backdrop
    ImVec2 c(pos.x + size.x * 0.5f, pos.y + size.y * 0.45f);
    for (int i = 6; i >= 1; --i) dl->AddCircleFilled(c, dp(60) * i, col(g_pal.accent, 0.012f * (7 - i)), 64);
    const float grid = dp(40);
    for (float x = pos.x + std::fmod(size.x * 0.5f, grid); x < pos.x + size.x; x += grid)
        dl->AddLine(ImVec2(x, pos.y), ImVec2(x, pos.y + size.y), IM_COL32(255, 255, 255, 6));
    for (float y = pos.y + std::fmod(size.y * 0.5f, grid); y < pos.y + size.y; y += grid)
        dl->AddLine(ImVec2(pos.x, y), ImVec2(pos.x + size.x, y), IM_COL32(255, 255, 255, 6));

    float top = c.y - dp(150);
    draw_logo(dl, ImVec2(c.x, top + dp(40)), dp(80), col(g_pal.accent), col(g_pal.accent2));

    auto centered = [&](ImFont* f, float y, const ImVec4& color, const char* text) {
        ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, text);
        dl->AddText(f, font_px(f), ImVec2(c.x - ts.x * 0.5f, y), col(color), text);
        return ts.y;
    };
    float y = top + dp(100);
    y += centered(g_fonts.display, y, g_pal.text, "No game running") + dp(6);
    y += centered(g_fonts.body, y, g_pal.text_dim, "Pick a game from your library or open a ROM file to start playing.") + dp(22);

    float bw1 = dp(170), bw2 = dp(150), gap = dp(12);
    ImGui::SetCursorScreenPos(ImVec2(c.x - (bw1 + bw2 + gap) * 0.5f, y));
    if (button("Browse Library", Icon::Grid, ButtonKind::Primary, bw1, true, dp(40))) view_ = View::Library;
    ImGui::SameLine(0, gap);
    if (button("Open ROM", Icon::FolderOpen, ButtonKind::Subtle, bw2, true, dp(40))) action_open_rom();
    y += dp(64);

    auto recent = library_.recent(5);
    if (!recent.empty()) {
        y += centered(g_fonts.small_bold, y, g_pal.text_faint, "RECENTLY PLAYED") + dp(10);
        ImFont* f = g_fonts.small_bold;
        float total = 0;
        std::vector<float> widths;
        for (const auto* g : recent) {
            float w = std::min(dp(200), f->CalcTextSizeA(font_px(f), FLT_MAX, 0, g->rom.display_title.c_str()).x) + dp(40);
            widths.push_back(w);
            total += w + dp(8);
        }
        // Drop chips that don't fit the width.
        while (total > size.x - dp(40) && !widths.empty()) {
            total -= widths.back() + dp(8);
            widths.pop_back();
        }
        float x = c.x - (total - dp(8)) * 0.5f;
        for (size_t i = 0; i < widths.size(); ++i) {
            const GameEntry* g = recent[i];
            ImGui::SetCursorScreenPos(ImVec2(x, y));
            ImGui::PushID(static_cast<int>(i));
            bool clicked = ImGui::InvisibleButton("##chip", ImVec2(widths[i], dp(32)));
            bool hov = ImGui::IsItemHovered();
            ImGui::PopID();
            ImVec2 mn(x, y), mx(x + widths[i], y + dp(32));
            dl->AddRectFilled(mn, mx, col(hov ? g_pal.bg4 : g_pal.bg2), dp(16));
            dl->AddRect(mn, mx, col(g_pal.border_strong), dp(16));
            draw_icon(dl, Icon::Play, ImVec2(x + dp(17), y + dp(16)), dp(11), col(hov ? g_pal.accent_hover : g_pal.text_dim));
            text_ellipsis(dl, f, ImVec2(x + dp(30), y + dp(16) - font_px(f) * 0.5f), widths[i] - dp(40), col(g_pal.text), g->rom.display_title.c_str());
            if (clicked) pending_launch_ = g->rom.path;
            if (hov) tooltip(g->rom.file_name.c_str());
            x += widths[i] + dp(8);
        }
    }

    // Keyboard hint at the bottom.
    const std::string open_key = hotkey_label(Hotkey::OpenRom);
    std::string hint = open_key.empty() ? "Tip: drop a ROM file onto this window to play it"
                                        : "Tip: press " + open_key + " to open a ROM, or drop a file onto this window";
    ImVec2 ts = g_fonts.small->CalcTextSizeA(font_px(g_fonts.small), FLT_MAX, 0, hint.c_str());
    if (size.y > dp(520))
        dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(c.x - ts.x * 0.5f, pos.y + size.y - dp(36)), col(g_pal.text_faint), hint.c_str());
}

void App::draw_pause_overlay(ImVec2 mn, ImVec2 mx) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float a = ImGui::GetStyle().Alpha;
    dl->AddRectFilled(mn, mx, col(ImVec4(0.02f, 0.02f, 0.04f, 0.62f * a)));

    const float cw = dp(320), ch = dp(318);
    ImVec2 c((mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f + (1.0f - a) * dp(16));
    ImVec2 cmn(c.x - cw * 0.5f, c.y - ch * 0.5f), cmx(c.x + cw * 0.5f, c.y + ch * 0.5f);
    if (mx.y - mn.y < ch + dp(20)) {
        // Very short window: compact pill only.
        const char* t = "Paused";
        ImVec2 ts = g_fonts.title->CalcTextSizeA(font_px(g_fonts.title), FLT_MAX, 0, t);
        dl->AddText(g_fonts.title, font_px(g_fonts.title), ImVec2(c.x - ts.x * 0.5f, c.y - ts.y * 0.5f), col(g_pal.text, a), t);
        return;
    }
    dl->AddRectFilled(ImVec2(cmn.x, cmn.y + dp(10)), ImVec2(cmx.x, cmx.y + dp(14)), IM_COL32(0, 0, 0, (int)(110 * a)), dp(16));
    dl->AddRectFilled(cmn, cmx, col(g_pal.bg1, 0.97f * a), dp(16));
    dl->AddRect(cmn, cmx, col(g_pal.border_strong, a), dp(16));

    ImVec2 ic(c.x, cmn.y + dp(46));
    dl->AddCircleFilled(ic, dp(24), col(g_pal.accent_soft, a), 40);
    draw_icon(dl, Icon::Pause, ic, dp(20), col(g_pal.accent_hover, a));
    auto centered = [&](ImFont* f, float y, const ImVec4& color, const char* text, float max_w) {
        ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, text);
        if (ts.x > max_w) text_ellipsis(dl, f, ImVec2(c.x - max_w * 0.5f, y), max_w, col(color, a), text);
        else dl->AddText(f, font_px(f), ImVec2(c.x - ts.x * 0.5f, y), col(color, a), text);
    };
    centered(g_fonts.title, cmn.y + dp(80), g_pal.text, "Paused", cw);
    centered(g_fonts.small, cmn.y + dp(108), g_pal.text_dim, current_rom_.display_title.c_str(), cw - dp(40));

    float bx = cmn.x + dp(24), bw = cw - dp(48), by = cmn.y + dp(138);
    ImGui::SetCursorScreenPos(ImVec2(bx, by));
    if (button("Resume", Icon::Play, ButtonKind::Primary, bw, true, dp(38))) toggle_pause();
    float half = (bw - dp(10)) * 0.5f;
    ImGui::SetCursorScreenPos(ImVec2(bx, by + dp(48)));
    if (button("Reset", Icon::Reset, ButtonKind::Subtle, half, true, dp(36))) {
        reset_game();
        core_.pause(false);
    }
    ImGui::SameLine(0, dp(10));
    if (button("Settings", Icon::Settings, ButtonKind::Subtle, half, true, dp(36))) open_settings(SettingsPage::General);
    ImGui::SetCursorScreenPos(ImVec2(bx, by + dp(94)));
    if (button("Stop Emulation", Icon::Stop, ButtonKind::DangerSubtle, bw, true, dp(36))) request_stop();

    const std::string resume_keys = hotkey_hint(Hotkey::Pause);
    std::string hint = resume_keys.empty() ? "Click Resume to continue" : "Press " + resume_keys + " to resume";
    centered(g_fonts.small, cmx.y - dp(30), g_pal.text_faint, hint.c_str(), cw);
}

void App::draw_info_panel(ImVec2 pos, ImVec2 size) {
    ImGui::SetCursorScreenPos(pos);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, g_pal.bg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(18, 18));
    ImGui::BeginChild("##info", size, ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddLine(wp, ImVec2(wp.x, wp.y + size.y), col(g_pal.border));
    const float w = ImGui::GetContentRegionAvail().x;

    CoreStats s = core_.stats();
    const GameEntry* entry = library_.find(current_key_);

    // Header: cover + title
    ImVec2 p = ImGui::GetCursorScreenPos();
    // Box art identifies the game best here; fall back to a screenshot.
    SDL_Texture* cover = boxart_texture(current_rom_);
    if (!cover) cover = thumbnail_texture(current_key_);
    float cover_h = w * 0.73f;
    if (cover) {
        draw_texture_cover(dl, cover, p, ImVec2(p.x + w, p.y + cover_h), dp(10));
    } else {
        draw_cover_art(dl, p, ImVec2(p.x + w, p.y + cover_h), current_rom_, dp(10));
    }
    ImGui::Dummy(ImVec2(w, cover_h + dp(6)));
    ImGui::PushFont(g_fonts.title);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
    ImGui::TextUnformatted(current_rom_.display_title.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopFont();
    badge(current_rom_.region_short.c_str(), current_rom_.homebrew ? g_pal.warning : g_pal.accent);
    ImGui::SameLine(0, dp(6));
    badge(("." + current_rom_.extension).c_str(), g_pal.text_dim);
    if (entry && entry->stats.favorite) {
        ImGui::SameLine(0, dp(6));
        badge("FAVORITE", g_pal.warning);
    }

    auto section = [&](const char* name) {
        ImGui::Dummy(dp(0, 10));
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::TextColored(g_pal.text_faint, "%s", name);
        ImGui::PopFont();
        ImVec2 q = ImGui::GetCursorScreenPos();
        dl->AddLine(ImVec2(q.x, q.y + dp(1)), ImVec2(q.x + w, q.y + dp(1)), col(g_pal.border));
        ImGui::Dummy(dp(0, 6));
    };
    auto kv = [&](const char* k, const std::string& v, const ImVec4* vc = nullptr) {
        ImVec2 q = ImGui::GetCursorScreenPos();
        ImFont* lf = g_fonts.small;
        ImFont* vf = g_fonts.mono_small;
        dl->AddText(lf, font_px(lf), ImVec2(q.x, q.y + dp(1)), col(g_pal.text_dim), k);
        float kw = lf->CalcTextSizeA(font_px(lf), FLT_MAX, 0, k).x + dp(12);
        ImVec2 ts = vf->CalcTextSizeA(font_px(vf), FLT_MAX, 0, v.c_str());
        float maxw = w - kw;
        if (ts.x <= maxw) dl->AddText(vf, font_px(vf), ImVec2(q.x + w - ts.x, q.y + dp(2)), col(vc ? *vc : g_pal.text), v.c_str());
        else text_ellipsis(dl, vf, ImVec2(q.x + kw, q.y + dp(2)), maxw, col(vc ? *vc : g_pal.text), v.c_str());
        ImGui::Dummy(ImVec2(w, font_px(lf) + dp(7)));
        if (ImGui::IsItemHovered() && ts.x > maxw) tooltip(v.c_str());
    };

    // Performance
    section("PERFORMANCE");
    {
        char big[16];
        std::snprintf(big, sizeof big, "%.0f", s.fps);
        ImVec2 q = ImGui::GetCursorScreenPos();
        ImVec4 fc = s.speed_pct >= 95 ? g_pal.success : s.speed_pct >= 70 ? g_pal.warning : g_pal.danger;
        if (core_.state() != RunState::Running) fc = g_pal.text_dim;
        dl->AddText(g_fonts.display, font_px(g_fonts.display), q, col(fc), big);
        float bw = g_fonts.display->CalcTextSizeA(font_px(g_fonts.display), FLT_MAX, 0, big).x;
        dl->AddText(g_fonts.small_bold, font_px(g_fonts.small_bold), ImVec2(q.x + bw + dp(6), q.y + dp(6)), col(g_pal.text_dim), "FPS");
        char sp[48];
        std::snprintf(sp, sizeof sp, "%.0f%% speed", s.speed_pct);
        dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(q.x + bw + dp(6), q.y + dp(21)), col(g_pal.text_faint), sp);

        // Sparkline of recent FPS samples.
        float gx = q.x + w * 0.45f, gw = w * 0.55f, gh = dp(40);
        ImVec2 gmn(gx, q.y), gmx(gx + gw, q.y + gh);
        dl->AddRectFilled(gmn, gmx, col(g_pal.bg0), dp(6));
        float target = static_cast<float>(s.vi_hz);
        float ty = gmx.y - (target / (target * 1.25f)) * gh;
        dl->AddLine(ImVec2(gmn.x + dp(4), ty), ImVec2(gmx.x - dp(4), ty), col(g_pal.border_strong));
        if (fps_history_.size() >= 2) {
            std::vector<ImVec2> pts;
            size_t n = fps_history_.size();
            for (size_t i = 0; i < n; ++i) {
                float x = gmn.x + dp(4) + (gw - dp(8)) * i / 119.0f + (gw - dp(8)) * (120 - n) / 119.0f;
                float v = std::clamp(fps_history_[i] / (target * 1.25f), 0.0f, 1.0f);
                pts.push_back(ImVec2(x, gmx.y - dp(3) - v * (gh - dp(6))));
            }
            // Filled area
            for (size_t i = 1; i < pts.size(); ++i) {
                dl->AddQuadFilled(pts[i - 1], pts[i], ImVec2(pts[i].x, gmx.y - dp(2)), ImVec2(pts[i - 1].x, gmx.y - dp(2)),
                                  col(g_pal.accent, 0.14f));
            }
            dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), col(g_pal.accent), ImDrawFlags_None, dp(1.5f));
        }
        ImGui::Dummy(ImVec2(w, gh + dp(8)));
        char ms[32];
        std::snprintf(ms, sizeof ms, "%.2f ms", s.frame_ms);
        kv("Frame time (host)", ms);
        std::snprintf(ms, sizeof ms, "%llu", static_cast<unsigned long long>(s.frame));
        kv("Frames emulated", ms);
        kv("Renderer", renderer_name_.empty() ? "SDL" : renderer_name_);
    }

    section("SESSION");
    {
        int t = static_cast<int>(s.uptime_s);
        char buf[32];
        std::snprintf(buf, sizeof buf, "%02d:%02d:%02d", t / 3600, (t / 60) % 60, t % 60);
        kv("This session", buf);
        std::int64_t total = (entry ? entry->stats.play_seconds : 0) + static_cast<std::int64_t>(s.uptime_s);
        std::snprintf(buf, sizeof buf, "%lldh %02lldm", (long long)(total / 3600), (long long)((total / 60) % 60));
        kv("Total play time", buf);
        kv("Launches", std::to_string(entry ? entry->stats.launches : 1));
    }

    section("HARDWARE");
    {
        kv("GPU microcode", s.ucode >= 0 && s.ucode < 11 ? kUcodeLong[s.ucode] : "Unknown");
        static const char* abi[] = {"ABI 1", "n_audio", "Nintendo Audio (MK)", "Nintendo Audio (SF)", "Nintendo Audio (Zelda)", "MusyX"};
        kv("Audio microcode", s.audio_abi >= 0 && s.audio_abi < 6 ? abi[s.audio_abi] : "Unknown");
        ImVec4 on = g_pal.success, off = g_pal.text_dim;
        kv("RSP", s.rsp_active ? "Active" : "Idle", s.rsp_active ? &on : &off);
        kv("RDP", s.rdp_active ? "Active" : "Idle", s.rdp_active ? &on : &off);
        char buf[32];
        std::snprintf(buf, sizeof buf, "%d Hz (%s)", s.vi_hz, s.vi_hz == 50 ? "PAL" : "NTSC");
        kv("VI refresh", buf);
        if (s.ai_rate) std::snprintf(buf, sizeof buf, "%.1f kHz", s.ai_rate / 1000.0);
        else std::snprintf(buf, sizeof buf, "\xE2\x80\x94");
        kv("AI sample rate", buf);
    }

    section("CARTRIDGE");
    {
        kv("Internal name", s.cart_title.empty() ? current_rom_.internal_name : s.cart_title);
        kv("Game code", current_rom_.game_code);
        kv("Region", current_rom_.region);
        kv("Revision", "1." + std::to_string(current_rom_.version));
        kv("CIC chip", s.cic);
        kv("Save type", s.save_type);
        char crc[40];
        std::snprintf(crc, sizeof crc, "%08X %08X", current_rom_.crc1, current_rom_.crc2);
        kv("CRC", crc);
        kv("ROM size", format_size(current_rom_.size));
        kv("Byte order", format_name(current_rom_.format));
    }

    section("CONTROLLERS");
    for (int i = 0; i < 4; ++i) {
        const PortConfig& pc = settings_.ports[i];
        ImVec2 q = ImGui::GetCursorScreenPos();
        char port[8];
        std::snprintf(port, sizeof port, "P%d", i + 1);
        ImVec4 dc = pc.plugged ? g_pal.success : g_pal.text_faint;
        dl->AddCircleFilled(ImVec2(q.x + dp(4), q.y + dp(8)), dp(3.5f), col(dc), 12);
        dl->AddText(g_fonts.small_bold, font_px(g_fonts.small_bold), ImVec2(q.x + dp(14), q.y + dp(1)), col(g_pal.text), port);
        std::string dev = pc.plugged ? input_.device_name(pc.device) : "Unplugged";
        text_ellipsis(dl, g_fonts.small, ImVec2(q.x + dp(42), q.y + dp(1)), w - dp(42), col(pc.plugged ? g_pal.text_dim : g_pal.text_faint), dev.c_str());
        ImGui::Dummy(ImVec2(w, dp(22)));
    }
    ImGui::Dummy(dp(0, 4));
    if (button("Configure Controllers", Icon::Gamepad, ButtonKind::Subtle, w)) open_settings(SettingsPage::Controller);
    ImGui::Dummy(dp(0, 8));
    ImGui::EndChild();
}

} // namespace ui
