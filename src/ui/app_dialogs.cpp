// Modal dialogs: About, confirmation, ROM properties, errors, native-dialog wait.

#include "app.hpp"
#include "platform.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include <chrono>
#include <algorithm>
#include <cfloat>
#include <ctime>
#include <cstdio>

namespace ui {

namespace fs = std::filesystem;

void App::draw_dialogs() {
    picker_drawn_ = false;
    draw_settings(); // may host the picker dialogs as nested modals
    draw_picker_dialogs();
    draw_properties();
    draw_about();
    draw_error();
    draw_confirm();
}

void App::draw_picker_dialogs() {
    // Opening a modal at the top level would close an open modal (Settings),
    // which then reopens and closes the picker again. Drawing the picker
    // inside the open modal stacks them instead.
    if (picker_drawn_) return;
    picker_drawn_ = true;
    if (browser_.draw()) {
        if (browser_.mode() == FileBrowser::Mode::OpenRom) pending_launch_ = browser_.result();
        else on_folder_picked(browser_.result());
    }
    draw_native_wait();
}

void App::draw_about() {
    if (!begin_modal("##about", "About", dp(560, 500), &about_open_, Icon::Info)) return;
    ImVec2 ws = ImGui::GetWindowSize();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    float cx = wp.x + ws.x * 0.5f;
    float y = wp.y + dp(56) + dp(26);

    for (int i = 3; i >= 1; --i) dl->AddCircleFilled(ImVec2(cx, y + dp(38)), dp(38) + dp(10) * i, col(g_pal.accent, 0.04f * (4 - i)), 48);
    draw_logo(dl, ImVec2(cx, y + dp(38)), dp(76), col(g_pal.accent), col(g_pal.accent2));
    y += dp(96);
    auto centered = [&](ImFont* f, const ImVec4& c, const std::string& t) {
        ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, t.c_str());
        dl->AddText(f, font_px(f), ImVec2(cx - ts.x * 0.5f, y), col(c), t.c_str());
        y += ts.y + dp(4);
    };
    centered(g_fonts.display, g_pal.text, kAppName);
    centered(g_fonts.body, g_pal.text_dim, std::string("Version ") + kAppVersion + "  \xC2\xB7  Nintendo 64 emulator");
    y += dp(16);

    const int linked = SDL_GetVersion();
    char sdl[48];
    std::snprintf(sdl, sizeof sdl, "%d.%d.%d", SDL_VERSIONNUM_MAJOR(linked), SDL_VERSIONNUM_MINOR(linked),
                  SDL_VERSIONNUM_MICRO(linked));
#if defined(_MSC_VER)
    std::string compiler = "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
    std::string compiler = std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    std::string compiler = "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    std::string compiler = "Unknown";
#endif
    if (compiler.size() > 40) compiler = compiler.substr(0, compiler.find(' ', 8));
    struct Row { const char* k; std::string v; };
    std::vector<Row> rows = {
        {"Platform", std::string(platform::os_display_name()) + " (" + SDL_GetPlatform() + ")"},
        {"Renderer", renderer_name_},
        {"SDL", sdl},
        {"Dear ImGui", IMGUI_VERSION},
        {"Compiler", compiler},
        {"Config folder", platform::path_to_utf8(platform::config_dir())},
    };
    float tx = wp.x + dp(48), tw = ws.x - dp(96);
    for (const auto& r : rows) {
        dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(tx, y + dp(1)), col(g_pal.text_faint), r.k);
        text_ellipsis(dl, g_fonts.mono_small, ImVec2(tx + dp(120), y + dp(2)), tw - dp(120), col(g_pal.text), r.v.c_str());
        y += dp(24);
        dl->AddLine(ImVec2(tx, y - dp(4)), ImVec2(tx + tw, y - dp(4)), col(g_pal.border, 0.6f));
    }

    const float footer = dp(64);
    modal_footer_begin(footer);
    if (button("Copy System Info", Icon::File, ButtonKind::Ghost)) {
        std::string info = std::string(kAppName) + " " + kAppVersion + "\n";
        for (const auto& r : rows) info += std::string(r.k) + ": " + r.v + "\n";
        ImGui::SetClipboardText(info.c_str());
        toast("System info copied to clipboard", ToastKind::Success);
    }
    modal_footer_end();
    ImGui::SetCursorPos(ImVec2(ws.x - dp(20) - dp(100), ws.y - footer + (footer - dp(34)) * 0.5f));
    if (button("Close", Icon::None, ButtonKind::Primary, dp(100))) {
        about_open_ = false;
        ImGui::CloseCurrentPopup();
    }
    end_modal();
}

void App::draw_confirm() {
    if (!begin_modal("##confirm", confirm_title_.c_str(), dp(480, 210), &confirm_open_, confirm_danger_ ? Icon::Warning : Icon::Info))
        return;
    ImVec2 ws = ImGui::GetWindowSize();
    ImGui::SetCursorPos(ImVec2(dp(24), dp(56) + dp(20)));
    ImGui::PushTextWrapPos(ws.x - dp(24));
    ImGui::TextColored(g_pal.text_dim, "%s", confirm_msg_.c_str());
    ImGui::PopTextWrapPos();

    const float footer = dp(64);
    float bw = std::max(dp(110), ImGui::CalcTextSize(confirm_ok_.c_str()).x + dp(40));
    float by = ws.y - footer + (footer - dp(34)) * 0.5f;
    modal_footer_begin(footer);
    modal_footer_end();
    ImGui::SetCursorPos(ImVec2(ws.x - dp(20) - bw, by));
    bool ok = button(confirm_ok_.c_str(), Icon::None, confirm_danger_ ? ButtonKind::Danger : ButtonKind::Primary, bw);
    ImGui::SetCursorPos(ImVec2(ws.x - dp(20) - bw - dp(10) - dp(100), by));
    bool cancel = button("Cancel", Icon::None, ButtonKind::Subtle, dp(100));
    if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)) ok = true;
    if (ok || cancel) {
        confirm_open_ = false;
        ImGui::CloseCurrentPopup();
    }
    end_modal();
    if (ok && confirm_cb_) {
        auto cb = std::move(confirm_cb_);
        confirm_cb_ = nullptr;
        cb();
    }
}

void App::draw_error() {
    if (!begin_modal("##error", error_title_.c_str(), dp(500, 240), &error_open_, Icon::Warning)) return;
    ImVec2 ws = ImGui::GetWindowSize();
    ImGui::SetCursorPos(ImVec2(dp(24), dp(56) + dp(20)));
    ImGui::PushTextWrapPos(ws.x - dp(24));
    ImGui::TextColored(g_pal.text_dim, "%s", error_msg_.c_str());
    ImGui::PopTextWrapPos();
    const float footer = dp(64);
    modal_footer_begin(footer);
    modal_footer_end();
    ImGui::SetCursorPos(ImVec2(ws.x - dp(20) - dp(100), ws.y - footer + (footer - dp(34)) * 0.5f));
    if (button("OK", Icon::None, ButtonKind::Primary, dp(100)) || ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
        error_open_ = false;
        ImGui::CloseCurrentPopup();
    }
    end_modal();
}

void App::draw_properties() {
    const GameEntry* g = props_open_ ? library_.find(props_key_) : nullptr;
    if (props_open_ && !g) props_open_ = false;
    std::string title = g ? g->rom.display_title : std::string("Properties");
    if (!begin_modal("##props", "Game Properties", dp(720, 570), &props_open_, Icon::Chip)) return;
    if (!g) {
        end_modal();
        return;
    }
    ImVec2 ws = ImGui::GetWindowSize();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    const float left = dp(24), top = dp(56) + dp(22);
    float cover_w = dp(220), cover_h = cover_w * 0.73f;
    ImVec2 cmn(wp.x + left, wp.y + top), cmx(cmn.x + cover_w, cmn.y + cover_h);
    SDL_Texture* cover = boxart_texture(g->rom);
    if (!cover) cover = thumbnail_texture(g->key);
    if (cover) {
        draw_texture_cover(dl, cover, cmn, cmx, dp(10));
    } else {
        draw_cover_art(dl, cmn, cmx, g->rom, dp(10));
    }
    ImGui::SetCursorScreenPos(ImVec2(cmn.x, cmx.y + dp(14)));
    ImGui::BeginGroup();
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + cover_w);
    ImGui::PushFont(g_fonts.title);
    ImGui::TextUnformatted(g->rom.display_title.c_str());
    ImGui::PopFont();
    if (!g->rom.tags.empty()) {
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_faint, "%s", g->rom.tags.c_str());
        ImGui::PopFont();
    }
    ImGui::PopTextWrapPos();
    ImGui::Dummy(dp(0, 4));
    badge(g->rom.region_short.c_str(), g->rom.homebrew ? g_pal.warning : g_pal.accent);
    if (g->stats.favorite) {
        ImGui::SameLine(0, dp(6));
        badge("FAVORITE", g_pal.warning);
    }
    ImGui::EndGroup();

    // Details table
    float tx = cmx.x + dp(28), tw = wp.x + ws.x - dp(24) - tx;
    float y = cmn.y;
    char crc[40];
    std::snprintf(crc, sizeof crc, "%08X  %08X", g->rom.crc1, g->rom.crc2);
    auto fmt_time = [](std::int64_t s) {
        char b[32];
        std::snprintf(b, sizeof b, "%lld h %02lld min", (long long)(s / 3600), (long long)((s / 60) % 60));
        return std::string(b);
    };
    std::string last = "Never";
    if (g->stats.last_played) {
        std::time_t t = static_cast<std::time_t>(g->stats.last_played);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char b[40];
        std::strftime(b, sizeof b, "%Y-%m-%d %H:%M", &tm);
        last = b;
    }
    struct Row { const char* k; std::string v; };
    std::vector<Row> rows = {
        {"File name", g->rom.file_name},
        {"Location", platform::path_to_utf8(g->rom.path.parent_path())},
        {"Size", format_size(g->rom.size)},
        {"Byte order", format_name(g->rom.format)},
        {"Internal name", g->rom.internal_name},
        {"Game code", g->rom.game_code},
        {"Region", g->rom.region},
        {"Revision", "1." + std::to_string(g->rom.version)},
        {"CRC", crc},
        {"Last played", last},
        {"Play time", fmt_time(g->stats.play_seconds)},
        {"Launches", std::to_string(g->stats.launches)},
        {"Box art", g->rom.boxart.empty() ? std::string("Not found") : platform::path_to_utf8(g->rom.boxart.filename())},
    };
    for (const auto& r : rows) {
        dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(tx, y + dp(2)), col(g_pal.text_faint), r.k);
        text_ellipsis(dl, g_fonts.mono_small, ImVec2(tx + dp(110), y + dp(3)), tw - dp(110), col(g_pal.text), r.v.c_str());
        ImGui::SetCursorScreenPos(ImVec2(tx, y));
        ImGui::PushID(r.k);
        ImGui::InvisibleButton("##row", ImVec2(tw, dp(26)));
        if (ImGui::IsItemHovered()) tooltip((r.v + "\n(click to copy)").c_str());
        if (ImGui::IsItemClicked()) {
            ImGui::SetClipboardText(r.v.c_str());
            toast(std::string(r.k) + " copied", ToastKind::Success, 1.5f);
        }
        ImGui::PopID();
        y += dp(28);
        dl->AddLine(ImVec2(tx, y - dp(4)), ImVec2(tx + tw, y - dp(4)), col(g_pal.border, 0.6f));
    }

    const float footer = dp(64);
    fs::path rom_path = g->rom.path;
    std::string key = g->key;
    modal_footer_begin(footer);
    if (button(platform::reveal_action_label(), Icon::FolderOpen, ButtonKind::Ghost)) platform::reveal_in_file_manager(rom_path);
    modal_footer_end();
    float by = ws.y - footer + (footer - dp(34)) * 0.5f;
    ImGui::SetCursorPos(ImVec2(ws.x - dp(20) - dp(110), by));
    if (button("Play", Icon::Play, ButtonKind::Primary, dp(110))) {
        props_open_ = false;
        ImGui::CloseCurrentPopup();
        if (core_.loaded() && key == current_key_) view_ = View::Game;
        else pending_launch_ = rom_path;
    }
    ImGui::SetCursorPos(ImVec2(ws.x - dp(20) - dp(110) - dp(10) - dp(100), by));
    if (button("Close", Icon::None, ButtonKind::Subtle, dp(100))) {
        props_open_ = false;
        ImGui::CloseCurrentPopup();
    }
    end_modal();
}

void App::draw_native_wait() {
    const char* id = "##native_wait";
    if (native_job_ && !ImGui::IsPopupOpen(id)) ImGui::OpenPopup(id);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(dp(360, 120));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, dp(12));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(24, 24));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, g_pal.bg1);
    if (ImGui::BeginPopupModal(id, nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings)) {
        if (!native_job_) ImGui::CloseCurrentPopup();
        spinner(dp(14), dp(3), col(g_pal.accent));
        ImGui::SameLine(0, dp(16));
        ImGui::BeginGroup();
        ImGui::PushFont(g_fonts.body_bold);
        ImGui::TextUnformatted("Waiting for file dialog\xE2\x80\xA6");
        ImGui::PopFont();
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_dim, "Choose a %s in the system window.",
                           native_mode_ == FileBrowser::Mode::OpenRom ? "ROM" : "folder");
        ImGui::PopFont();
        ImGui::EndGroup();
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

} // namespace ui
