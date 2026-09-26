// Game library view: sidebar filters, "Continue playing", grid & list modes.

#include "app.hpp"
#include "platform.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <ctime>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace ui {

namespace fs = std::filesystem;

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static std::string relative_time(std::int64_t unix_time) {
    if (unix_time <= 0) return "Never played";
    auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::int64_t d = now - unix_time;
    char buf[48];
    if (d < 60) return "Just now";
    if (d < 3600) { std::snprintf(buf, sizeof buf, "%lld min ago", (long long)(d / 60)); return buf; }
    if (d < 86400) { std::snprintf(buf, sizeof buf, "%lld h ago", (long long)(d / 3600)); return buf; }
    if (d < 2 * 86400) return "Yesterday";
    if (d < 30 * 86400) { std::snprintf(buf, sizeof buf, "%lld days ago", (long long)(d / 86400)); return buf; }
    std::time_t t = static_cast<std::time_t>(unix_time);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::strftime(buf, sizeof buf, "%b %d, %Y", &tm);
    return buf;
}

static std::string play_time(std::int64_t s) {
    if (s <= 0) return "\xE2\x80\x94";
    char buf[32];
    if (s < 60) return "< 1 min";
    if (s < 3600) { std::snprintf(buf, sizeof buf, "%lld min", (long long)(s / 60)); return buf; }
    std::snprintf(buf, sizeof buf, "%lld h %02lld min", (long long)(s / 3600), (long long)((s / 60) % 60));
    return buf;
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

std::vector<const GameEntry*> App::filtered_games() {
    std::vector<const GameEntry*> out;
    std::string q = lower(search_);
    for (const auto& g : library_.games()) {
        const RomInfo& r = g.rom;
        switch (lib_filter_) {
            case LibraryFilter::Favorites: if (!g.stats.favorite) continue; break;
            case LibraryFilter::Recent: if (g.stats.last_played == 0) continue; break;
            case LibraryFilter::Homebrew: if (!r.homebrew) continue; break;
            case LibraryFilter::RegionUSA: if (r.region_short != "USA") continue; break;
            case LibraryFilter::RegionEUR: if (r.region_short != "EUR" && r.region != "Europe") continue; break;
            case LibraryFilter::RegionJPN: if (r.region_short != "JPN") continue; break;
            case LibraryFilter::Folder:
                // Prefix must end at a separator so "Games" doesn't also match "Games2".
                if (g.key.rfind(lib_folder_, 0) != 0 ||
                    (g.key.size() > lib_folder_.size() && g.key[lib_folder_.size()] != '/' && g.key[lib_folder_.size()] != '\\'))
                    continue;
                break;
            default: break;
        }
        if (!q.empty()) {
            std::string hay = lower(r.display_title + " " + r.internal_name + " " + r.game_code + " " + r.file_name);
            if (hay.find(q) == std::string::npos) continue;
        }
        out.push_back(&g);
    }
    auto by_title = [](const GameEntry* a, const GameEntry* b) { return lower(a->rom.display_title) < lower(b->rom.display_title); };
    switch (lib_filter_ == LibraryFilter::Recent ? 1 : settings_.library_sort) {
        case 1:
            std::stable_sort(out.begin(), out.end(), [&](const GameEntry* a, const GameEntry* b) {
                if (a->stats.last_played != b->stats.last_played) return a->stats.last_played > b->stats.last_played;
                return by_title(a, b);
            });
            break;
        case 2:
            std::stable_sort(out.begin(), out.end(), [](const GameEntry* a, const GameEntry* b) { return a->rom.size > b->rom.size; });
            break;
        case 3:
            std::stable_sort(out.begin(), out.end(), [&](const GameEntry* a, const GameEntry* b) {
                if (a->rom.region_short != b->rom.region_short) return a->rom.region_short < b->rom.region_short;
                return by_title(a, b);
            });
            break;
        case 4:
            std::stable_sort(out.begin(), out.end(), [&](const GameEntry* a, const GameEntry* b) {
                if (a->stats.play_seconds != b->stats.play_seconds) return a->stats.play_seconds > b->stats.play_seconds;
                return by_title(a, b);
            });
            break;
        default: std::stable_sort(out.begin(), out.end(), by_title); break;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

void App::draw_library(ImVec2 pos, ImVec2 size) {
    ImGui::SetCursorScreenPos(pos);
    ImGui::BeginChild("##library", size, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    const bool sidebar = library_sidebar_ && size.x > dp(880);
    const float sidebar_w = sidebar ? dp(232) : 0.0f;
    if (sidebar) {
        ImGui::SetCursorScreenPos(pos);
        draw_library_sidebar(size.y);
    }

    ImGui::SetCursorScreenPos(ImVec2(pos.x + sidebar_w, pos.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(28, 22));
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, dp(10));
    ImGui::BeginChild("##lib_main", ImVec2(size.x - sidebar_w, size.y), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleVar(2);
    const float width = ImGui::GetContentRegionAvail().x;

    const bool no_dirs = settings_.rom_dirs.empty() && library_.games().empty();
    if (no_dirs && !library_.scanning()) {
        draw_library_empty(ImGui::GetContentRegionAvail());
    } else {
        draw_library_header(width);
        auto games = filtered_games();

        if (library_.scanning() && library_.games().empty()) {
            ImGui::Dummy(dp(0, 80));
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + width * 0.5f - dp(16));
            spinner(dp(16), dp(3), col(g_pal.accent));
            char msg[64];
            std::snprintf(msg, sizeof msg, "Scanning your ROM folders\xE2\x80\xA6  %d found", library_.scan_found());
            float tw = ImGui::CalcTextSize(msg).x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (width - tw) * 0.5f);
            ImGui::TextColored(g_pal.text_dim, "%s", msg);
        } else {
            bool show_continue = lib_filter_ == LibraryFilter::All && search_[0] == 0 && settings_.library_view == 0;
            auto recent = library_.recent(4);
            if (show_continue && !recent.empty()) draw_continue_row(recent, width);

            if (games.empty()) {
                ImGui::Dummy(dp(0, 60));
                ImVec2 p = ImGui::GetCursorScreenPos();
                draw_icon(ImGui::GetWindowDrawList(), search_[0] ? Icon::Search : Icon::Library,
                          ImVec2(p.x + width * 0.5f, p.y + dp(20)), dp(40), col(g_pal.text_faint));
                ImGui::Dummy(dp(0, 56));
                const char* t = search_[0] ? "No games match your search" : "Nothing here yet";
                const char* s = search_[0] ? "Try a different title, game code or file name."
                              : lib_filter_ == LibraryFilter::Favorites ? "Right-click a game and choose \xE2\x80\x9C" "Add to Favorites\xE2\x80\x9D."
                              : lib_filter_ == LibraryFilter::Recent ? "Games you play will show up here."
                                                                       : "No games in this category.";
                ImGui::PushFont(g_fonts.body_bold);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (width - ImGui::CalcTextSize(t).x) * 0.5f);
                ImGui::TextUnformatted(t);
                ImGui::PopFont();
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (width - ImGui::CalcTextSize(s).x) * 0.5f);
                ImGui::TextColored(g_pal.text_dim, "%s", s);
            } else if (settings_.library_view == 0) {
                draw_game_grid(games, width);
            } else {
                draw_game_list(games);
            }
        }
    }
    ImGui::EndChild();
    ImGui::EndChild();
}

void App::draw_library_sidebar(float height) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, g_pal.bg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(12, 16));
    ImGui::BeginChild("##lib_sidebar", ImVec2(dp(232), height), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImDrawList* dl = ImGui::GetWindowDrawList();

    int counts[8] = {};
    for (const auto& g : library_.games()) {
        counts[0]++;
        if (g.stats.favorite) counts[1]++;
        if (g.stats.last_played) counts[2]++;
        if (g.rom.homebrew) counts[3]++;
        if (g.rom.region_short == "USA") counts[4]++;
        if (g.rom.region_short == "EUR" || g.rom.region == "Europe") counts[5]++;
        if (g.rom.region_short == "JPN") counts[6]++;
    }

    auto item = [&](const char* label, Icon icon, bool active, int count, const char* id) {
        ImGui::PushID(id);
        ImVec2 p = ImGui::GetCursorScreenPos();
        float w = ImGui::GetContentRegionAvail().x, h = dp(34);
        bool clicked = ImGui::InvisibleButton("##it", ImVec2(w, h));
        bool hov = ImGui::IsItemHovered();
        float a = anim(ImGui::GetID("a"), active ? 1.0f : 0.0f, 16.0f);
        float hv = anim(ImGui::GetID("h"), hov ? 1.0f : 0.0f, 18.0f);
        if (hv > 0.01f) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(g_pal.bg3, hv), dp(7));
        if (a > 0.01f) {
            dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(g_pal.accent_soft, a), dp(7));
            dl->AddRectFilled(ImVec2(p.x, p.y + dp(8)), ImVec2(p.x + dp(3), p.y + h - dp(8)), col(g_pal.accent, a), dp(2));
        }
        ImVec4 fc = mix(mix(g_pal.text_dim, g_pal.text, hv), g_pal.text, a);
        draw_icon(dl, icon, ImVec2(p.x + dp(18), p.y + h * 0.5f), dp(15), col(mix(fc, g_pal.accent_hover, a)));
        text_ellipsis(dl, active ? g_fonts.body_bold : g_fonts.body, ImVec2(p.x + dp(36), p.y + (h - font_px(g_fonts.body)) * 0.5f),
                      w - dp(80), col(fc), label);
        if (count >= 0) {
            char c[16];
            std::snprintf(c, sizeof c, "%d", count);
            ImFont* f = g_fonts.mono_small;
            ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, c);
            dl->AddText(f, font_px(f), ImVec2(p.x + w - ts.x - dp(10), p.y + (h - ts.y) * 0.5f), col(g_pal.text_faint), c);
        }
        ImGui::PopID();
        return clicked;
    };
    auto group = [&](const char* name) {
        ImGui::Dummy(dp(0, 6));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + dp(8));
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::TextColored(g_pal.text_faint, "%s", name);
        ImGui::PopFont();
        ImGui::Dummy(dp(0, 2));
    };

    group("LIBRARY");
    if (item("All Games", Icon::Grid, lib_filter_ == LibraryFilter::All, counts[0], "all")) lib_filter_ = LibraryFilter::All;
    if (item("Favorites", Icon::Star, lib_filter_ == LibraryFilter::Favorites, counts[1], "fav")) lib_filter_ = LibraryFilter::Favorites;
    if (item("Recently Played", Icon::Clock, lib_filter_ == LibraryFilter::Recent, counts[2], "rec")) lib_filter_ = LibraryFilter::Recent;
    if (item("Homebrew & Demos", Icon::Sparkle, lib_filter_ == LibraryFilter::Homebrew, counts[3], "hb")) lib_filter_ = LibraryFilter::Homebrew;

    group("REGIONS");
    if (item("North America", Icon::Globe, lib_filter_ == LibraryFilter::RegionUSA, counts[4], "usa")) lib_filter_ = LibraryFilter::RegionUSA;
    if (item("Europe", Icon::Globe, lib_filter_ == LibraryFilter::RegionEUR, counts[5], "eur")) lib_filter_ = LibraryFilter::RegionEUR;
    if (item("Japan", Icon::Globe, lib_filter_ == LibraryFilter::RegionJPN, counts[6], "jpn")) lib_filter_ = LibraryFilter::RegionJPN;

    group("FOLDERS");
    for (size_t i = 0; i < settings_.rom_dirs.size(); ++i) {
        const std::string& d = settings_.rom_dirs[i];
        fs::path p = platform::utf8_to_path(d);
        std::string name = platform::path_to_utf8(p.filename());
        if (name.empty()) name = d;
        std::string key_prefix = platform::path_to_utf8(fs::weakly_canonical(p));
        int n = 0;
        for (const auto& g : library_.games()) {
            bool inside = g.key.rfind(key_prefix, 0) == 0 &&
                          (g.key.size() == key_prefix.size() || g.key[key_prefix.size()] == '/' || g.key[key_prefix.size()] == '\\');
            n += inside ? 1 : 0;
        }
        std::string id = "dir" + std::to_string(i);
        bool active = lib_filter_ == LibraryFilter::Folder && lib_folder_ == key_prefix;
        if (item(name.c_str(), Icon::Folder, active, n, id.c_str())) {
            lib_filter_ = LibraryFilter::Folder;
            lib_folder_ = key_prefix;
        }
        if (ImGui::IsItemHovered()) tooltip(d.c_str());
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem(platform::reveal_action_label())) platform::reveal_in_file_manager(p);
            if (ImGui::MenuItem("Remove from Library")) {
                settings_.rom_dirs.erase(settings_.rom_dirs.begin() + static_cast<long>(i));
                rescan_library();
                if (lib_filter_ == LibraryFilter::Folder) lib_filter_ = LibraryFilter::All;
                save_settings();
                ImGui::EndPopup();
                break;
            }
            ImGui::EndPopup();
        }
    }
    if (item("Add Folder\xE2\x80\xA6", Icon::Plus, false, -1, "addfolder")) action_add_folder();

    // Bottom: scan status
    float bottom = ImGui::GetWindowPos().y + height - dp(52);
    if (ImGui::GetCursorScreenPos().y < bottom) {
        ImGui::SetCursorScreenPos(ImVec2(ImGui::GetCursorScreenPos().x, bottom));
        ImVec2 p = ImGui::GetCursorScreenPos();
        dl->AddLine(ImVec2(ImGui::GetWindowPos().x, p.y - dp(8)), ImVec2(ImGui::GetWindowPos().x + dp(232), p.y - dp(8)), col(g_pal.border));
        if (library_.scanning()) {
            spinner(dp(8), dp(2), col(g_pal.accent));
            ImGui::SameLine();
            ImGui::PushFont(g_fonts.small);
            ImGui::TextColored(g_pal.text_dim, "Scanning\xE2\x80\xA6 %d found", library_.scan_found());
            ImGui::PopFont();
        } else {
            if (icon_button("rescan", Icon::Refresh, dp(28), "Rescan ROM folders"))
                rescan_library();
            ImGui::SameLine();
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(6));
            ImGui::PushFont(g_fonts.small);
            ImGui::TextColored(g_pal.text_faint, "%d games in %d folder%s", counts[0], (int)settings_.rom_dirs.size(),
                               settings_.rom_dirs.size() == 1 ? "" : "s");
            ImGui::PopFont();
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImVec2 mn = ImGui::GetItemRectMin(), mx = ImGui::GetItemRectMax();
    ImGui::GetWindowDrawList()->AddLine(ImVec2(mx.x - 1, mn.y), ImVec2(mx.x - 1, mx.y), col(g_pal.border));
}

void App::draw_library_header(float width) {
    const char* title = "All Games";
    switch (lib_filter_) {
        case LibraryFilter::Favorites: title = "Favorites"; break;
        case LibraryFilter::Recent: title = "Recently Played"; break;
        case LibraryFilter::Homebrew: title = "Homebrew & Demos"; break;
        case LibraryFilter::RegionUSA: title = "North America"; break;
        case LibraryFilter::RegionEUR: title = "Europe"; break;
        case LibraryFilter::RegionJPN: title = "Japan"; break;
        case LibraryFilter::Folder: title = "Folder"; break;
        default: break;
    }
    std::string folder_name;
    if (lib_filter_ == LibraryFilter::Folder) {
        folder_name = platform::path_to_utf8(platform::utf8_to_path(lib_folder_).filename());
        title = folder_name.c_str();
    }
    size_t n = filtered_games().size();

    ImVec2 start = ImGui::GetCursorScreenPos();
    ImGui::PushFont(g_fonts.display);
    ImGui::TextUnformatted(title);
    ImGui::PopFont();
    ImGui::SameLine(0, dp(12));
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(12));
    ImGui::PushFont(g_fonts.body);
    ImGui::TextColored(g_pal.text_faint, "%zu game%s", n, n == 1 ? "" : "s");
    ImGui::PopFont();

    // Right-aligned controls: search, sort, view mode. Wrap to a new line when narrow.
    const float search_w = std::clamp(width * 0.28f, dp(180), dp(300));
    const float sort_w = dp(170), view_w = dp(84);
    const float controls_w = search_w + sort_w + view_w + dp(20);
    const bool wrap = width < controls_w + dp(320);
    float cy = wrap ? ImGui::GetCursorScreenPos().y + dp(6) : start.y + dp(2);
    float cx = start.x + width - controls_w;
    if (wrap) cx = start.x;

    ImGui::SetCursorScreenPos(ImVec2(cx, cy));
    search_field("libsearch", search_, sizeof search_, "Search games\xE2\x80\xA6", search_w);
    ImGui::SetCursorScreenPos(ImVec2(cx + search_w + dp(10), cy));
    const char* sorts[] = {"Sort: Title", "Sort: Last Played", "Sort: Size", "Sort: Region", "Sort: Play Time"};
    combo("sort", &settings_.library_sort, sorts, 5, sort_w);
    ImGui::SetCursorScreenPos(ImVec2(cx + search_w + sort_w + dp(20), cy + dp(1)));
    const char* views[] = {"", ""};
    const Icon vicons[] = {Icon::Grid, Icon::List};
    segmented("view", views, 2, &settings_.library_view, view_w, vicons);
    if (ImGui::IsItemHovered()) tooltip("Grid / list view");

    ImGui::SetCursorScreenPos(ImVec2(start.x, std::max(ImGui::GetCursorScreenPos().y, start.y + dp(44)) + dp(10)));
    ImGui::Dummy(ImVec2(width, 1));
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

void draw_texture_cover(ImDrawList* dl, SDL_Texture* tex, ImVec2 mn, ImVec2 mx, float rounding) {
    const float tw = static_cast<float>(tex->w), th = static_cast<float>(tex->h);
    // "Cover" fit: fill the box, crop the overflow.
    float bw = mx.x - mn.x, bh = mx.y - mn.y;
    float s = std::max(bw / tw, bh / th);
    float uw = bw / (tw * s), uh = bh / (th * s);
    ImVec2 uv0((1 - uw) * 0.5f, (1 - uh) * 0.5f), uv1(uv0.x + uw, uv0.y + uh);
    dl->AddImageRounded((ImTextureID)(intptr_t)tex, mn, mx, uv0, uv1, IM_COL32_WHITE, rounding);
}

void App::draw_continue_row(const std::vector<const GameEntry*>& recent, float width) {
    ImGui::PushFont(g_fonts.body_bold);
    ImGui::TextUnformatted("Continue Playing");
    ImGui::PopFont();
    ImGui::Dummy(dp(0, 2));

    const float gap = dp(16);
    int cols = width > dp(1100) ? 4 : width > dp(760) ? 3 : 2;
    int n = std::min<int>(cols, static_cast<int>(recent.size()));
    float cw = (width - gap * (cols - 1)) / cols;
    float ch = cw * 9.0f / 16.0f;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    for (int i = 0; i < n; ++i) {
        const GameEntry& g = *recent[i];
        ImVec2 mn(origin.x + i * (cw + gap), origin.y);
        ImGui::SetCursorScreenPos(mn);
        ImGui::PushID(g.key.c_str());
        ImGui::PushID("cont");
        bool clicked = ImGui::InvisibleButton("##card", ImVec2(cw, ch));
        bool hov = ImGui::IsItemHovered();
        game_context_menu(g);
        float h = anim(ImGui::GetID("h"), hov ? 1.0f : 0.0f, 14.0f);
        ImGui::PopID();
        ImGui::PopID();
        float lift = dp(3) * h;
        ImVec2 a(mn.x, mn.y - lift), b(mn.x + cw, mn.y + ch - lift);
        dl->AddRectFilled(ImVec2(a.x + dp(2), a.y + dp(8)), ImVec2(b.x - dp(2), b.y + dp(10)), IM_COL32(0, 0, 0, (int)(60 + 60 * h)), dp(12));
        SDL_Texture* t = thumbnail_texture(g.key);
        if (!t) t = boxart_texture(g.rom);
        if (t) draw_texture_cover(dl, t, a, b, dp(12));
        else draw_cover_art(dl, a, b, g.rom, dp(12), false); // the card draws its own caption

        // Bottom gradient + text
        gradient_rect_v(dl, ImVec2(a.x, a.y + ch * 0.35f), b, IM_COL32(0, 0, 0, 255), 0, 215, dp(12), ImDrawFlags_RoundCornersBottom);
        float pad = dp(14);
        ImFont* tf = g_fonts.body_bold;
        text_ellipsis(dl, tf, ImVec2(a.x + pad, b.y - pad - font_px(tf) - font_px(g_fonts.small) - dp(4)), cw - pad * 2 - dp(44),
                      IM_COL32_WHITE, g.rom.display_title.c_str());
        std::string meta = relative_time(g.stats.last_played) + "  \xC2\xB7  " + play_time(g.stats.play_seconds);
        dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(a.x + pad, b.y - pad - font_px(g_fonts.small)),
                    IM_COL32(255, 255, 255, 170), meta.c_str());

        bool playing = core_.loaded() && g.key == current_key_;
        if (playing) {
            float bw = g_fonts.small_bold->CalcTextSizeA(font_px(g_fonts.small_bold), FLT_MAX, 0, "PLAYING").x + dp(12);
            badge_at(dl, ImVec2(b.x - pad - bw, a.y + pad), "PLAYING", g_pal.success, true);
        }

        // Play button
        float r = dp(18) + dp(2) * h;
        ImVec2 pc(b.x - pad - dp(18), b.y - pad - dp(16));
        dl->AddCircleFilled(pc, r, col(mix(with_alpha(ImVec4(1, 1, 1, 1), 0.22f), g_pal.accent, h)), 32);
        draw_icon(dl, Icon::Play, ImVec2(pc.x + dp(1.5f), pc.y), dp(16), IM_COL32_WHITE);
        if (h > 0.01f) dl->AddRect(a, b, col(g_pal.accent, h * 0.9f), dp(12), 0, dp(2));

        if (clicked) {
            if (playing) view_ = View::Game;
            else pending_launch_ = g.rom.path;
        }
        if (hov) {
            std::string tip = g.rom.display_title + "\n" + (playing ? "Currently running \xE2\x80\x94 click to return" : "Click to play");
            tooltip(tip.c_str());
        }
    }
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + ch + dp(26)));
    ImGui::PushFont(g_fonts.body_bold);
    ImGui::TextUnformatted("All Games");
    ImGui::PopFont();
    ImGui::Dummy(dp(0, 2));
}

void App::draw_game_grid(const std::vector<const GameEntry*>& games, float width) {
    const float gap = dp(18);
    const float base = dp(196) * settings_.grid_card_size;
    int cols = std::max(1, static_cast<int>((width + gap) / (base + gap)));
    float cw = (width - gap * (cols - 1)) / cols;
    float img_h = cw * 0.75f;
    float card_h = img_h + dp(60);
    ImDrawList* dl = ImGui::GetWindowDrawList();

    int rows = (static_cast<int>(games.size()) + cols - 1) / cols;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    // Only lay out visible rows (large libraries stay fast).
    ImGuiListClipper clipper;
    clipper.Begin(rows, card_h + gap);
    while (clipper.Step()) {
        for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
            for (int c = 0; c < cols; ++c) {
                int idx = row * cols + c;
                if (idx >= static_cast<int>(games.size())) break;
                const GameEntry& g = *games[idx];
                ImVec2 mn(origin.x + c * (cw + gap), origin.y + row * (card_h + gap));
                ImGui::SetCursorScreenPos(mn);
                ImGui::PushID(g.key.c_str());
                ImGui::SetNextItemAllowOverlap(); // lets the favourite star on top receive clicks
                ImGui::InvisibleButton("##card", ImVec2(cw, card_h));
                bool hov = ImGui::IsItemHovered();
                bool dbl = hov && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
                if (ImGui::IsItemClicked()) selected_key_ = g.key;
                if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) selected_key_ = g.key;
                game_context_menu(g);
                float h = anim(ImGui::GetID("h"), hov ? 1.0f : 0.0f, 14.0f);
                bool selected = selected_key_ == g.key;
                bool playing = core_.loaded() && g.key == current_key_;

                float lift = dp(4) * h;
                ImVec2 a(mn.x, mn.y - lift), b(mn.x + cw, mn.y + img_h - lift);
                // shadow
                dl->AddRectFilled(ImVec2(a.x + dp(3), a.y + dp(8)), ImVec2(b.x - dp(3), b.y + dp(10)), IM_COL32(0, 0, 0, (int)(50 + 70 * h)), dp(10));
                // Box art until the game has been played, then its latest screenshot.
                if (SDL_Texture* t = card_texture(g)) draw_texture_cover(dl, t, a, b, dp(10));
                else draw_cover_art(dl, a, b, g.rom, dp(10));
                if (selected || h > 0.01f)
                    dl->AddRect(ImVec2(a.x - dp(1), a.y - dp(1)), ImVec2(b.x + dp(1), b.y + dp(1)),
                                col(g_pal.accent, selected ? 1.0f : h * 0.8f), dp(11), 0, dp(2));

                // Hover play button
                if (h > 0.01f) {
                    dl->AddRectFilled(a, b, IM_COL32(0, 0, 0, (int)(70 * h)), dp(10));
                    ImVec2 pc((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
                    float r = dp(24) * (0.85f + 0.15f * h);
                    dl->AddCircleFilled(pc, r, col(g_pal.accent, h), 40);
                    draw_icon(dl, Icon::Play, ImVec2(pc.x + dp(2), pc.y), dp(20) * h, col(ImVec4(1, 1, 1, h)));
                }
                // Favourite star (clickable)
                if (g.stats.favorite || h > 0.01f) {
                    ImVec2 sc(b.x - dp(18), a.y + dp(18));
                    ImGui::SetCursorScreenPos(ImVec2(sc.x - dp(13), sc.y - dp(13)));
                    bool star_clicked = ImGui::InvisibleButton("##fav", dp(26, 26));
                    bool star_hov = ImGui::IsItemHovered();
                    if (star_hov) tooltip(g.stats.favorite ? "Remove from Favorites" : "Add to Favorites");
                    dl->AddCircleFilled(sc, dp(13), IM_COL32(0, 0, 0, star_hov ? 150 : 100), 24);
                    draw_icon(dl, g.stats.favorite ? Icon::StarFilled : Icon::Star, sc, dp(15),
                              g.stats.favorite ? col(g_pal.warning) : IM_COL32(255, 255, 255, (int)(230 * std::max(h, 0.3f))));
                    if (star_clicked) library_.toggle_favorite(g.key);
                }
                if (playing) badge_at(dl, ImVec2(a.x + dp(10), b.y - dp(28)), "PLAYING", g_pal.success, true);

                // Caption
                float ty = mn.y + img_h + dp(10);
                text_ellipsis(dl, g_fonts.body_bold, ImVec2(mn.x + dp(2), ty), cw - dp(4), col(g_pal.text), g.rom.display_title.c_str());
                float by = ty + font_px(g_fonts.body_bold) + dp(6);
                badge_at(dl, ImVec2(mn.x + dp(2), by), g.rom.region_short.c_str(),
                         g.rom.homebrew ? g_pal.warning : g_pal.accent);
                std::string meta = format_size(g.rom.size);
                if (g.stats.play_seconds > 0) meta += "  \xC2\xB7  " + play_time(g.stats.play_seconds);
                float bw = g_fonts.small_bold->CalcTextSizeA(font_px(g_fonts.small_bold), FLT_MAX, 0, g.rom.region_short.c_str()).x + dp(20);
                text_ellipsis(dl, g_fonts.small, ImVec2(mn.x + dp(2) + bw, by + dp(2)), cw - bw - dp(4), col(g_pal.text_faint), meta.c_str());

                if (hov && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId)) {
                    std::string tip = g.rom.display_title + "\n" + g.rom.file_name + "\nDouble-click to play \xC2\xB7 right-click for more";
                    tooltip(tip.c_str());
                }
                if (dbl) {
                    if (playing) view_ = View::Game;
                    else pending_launch_ = g.rom.path;
                }
                ImGui::PopID();
            }
        }
    }
    clipper.End();
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + rows * (card_h + gap)));
    ImGui::Dummy(ImVec2(width, dp(8)));

    // Enter launches the selected game.
    if (!selected_key_.empty() && ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && !ImGui::GetIO().WantTextInput &&
        (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter))) {
        if (GameEntry* g = library_.find(selected_key_)) pending_launch_ = g->rom.path;
    }
}

void App::draw_game_list(const std::vector<const GameEntry*>& games) {
    ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_Hideable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_SizingStretchProp |
                            ImGuiTableFlags_PadOuterX;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, dp(10, 9));
    if (!ImGui::BeginTable("##gamelist", 8, flags)) {
        ImGui::PopStyleVar();
        return;
    }
    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoHide, dp(26));
    ImGui::TableSetupColumn("Title", ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_NoHide, 3.0f);
    ImGui::TableSetupColumn("Region", ImGuiTableColumnFlags_WidthFixed, dp(70));
    ImGui::TableSetupColumn("Code", ImGuiTableColumnFlags_WidthFixed, dp(60));
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, dp(76));
    ImGui::TableSetupColumn("Format", ImGuiTableColumnFlags_WidthFixed, dp(64));
    ImGui::TableSetupColumn("Last Played", ImGuiTableColumnFlags_WidthFixed, dp(110));
    ImGui::TableSetupColumn("Play Time", ImGuiTableColumnFlags_WidthFixed, dp(96));
    ImGui::PushFont(g_fonts.small_bold);
    ImGui::PushStyleColor(ImGuiCol_Text, g_pal.text_faint);
    ImGui::TableHeadersRow();
    ImGui::PopStyleColor();
    ImGui::PopFont();

    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(games.size()));
    while (clipper.Step()) {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const GameEntry& g = *games[i];
            bool playing = core_.loaded() && g.key == current_key_;
            ImGui::TableNextRow(0, dp(38));
            ImGui::PushID(g.key.c_str());
            ImGui::TableSetColumnIndex(0);
            ImVec2 rp = ImGui::GetCursorScreenPos();
            bool sel = selected_key_ == g.key;
            if (ImGui::Selectable("##row", sel, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick |
                                                    ImGuiSelectableFlags_AllowOverlap, ImVec2(0, dp(22)))) {
                selected_key_ = g.key;
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    if (playing) view_ = View::Game;
                    else pending_launch_ = g.rom.path;
                }
            }
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) selected_key_ = g.key;
            game_context_menu(g);
            ImGui::SetCursorScreenPos(rp);
            ImGui::SetNextItemAllowOverlap();
            if (ImGui::InvisibleButton("##fav", dp(22, 22))) library_.toggle_favorite(g.key);
            bool sh = ImGui::IsItemHovered();
            draw_icon(ImGui::GetWindowDrawList(), g.stats.favorite ? Icon::StarFilled : Icon::Star,
                      ImVec2(rp.x + dp(11), rp.y + dp(11)), dp(14),
                      col(g.stats.favorite ? g_pal.warning : (sh ? g_pal.text_dim : g_pal.text_faint)));

            ImGui::TableSetColumnIndex(1);
            {
                ImVec2 cp = ImGui::GetCursorScreenPos();
                ImVec2 cmn(cp.x, cp.y - dp(3)), cmx(cp.x + dp(38), cp.y + dp(25));
                if (SDL_Texture* t = card_texture(g)) draw_texture_cover(ImGui::GetWindowDrawList(), t, cmn, cmx, dp(4));
                else draw_cover_art(ImGui::GetWindowDrawList(), cmn, cmx, g.rom, dp(4), false);
                ImGui::SetCursorScreenPos(ImVec2(cp.x + dp(48), cp.y));
            }
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(2));
            ImGui::PushFont(g_fonts.body_bold);
            ImGui::TextUnformatted(g.rom.display_title.c_str());
            ImGui::PopFont();
            if (playing) {
                ImGui::SameLine();
                badge("PLAYING", g_pal.success, true);
            }
            ImGui::TableSetColumnIndex(2);
            badge(g.rom.region_short.c_str(), g.rom.homebrew ? g_pal.warning : g_pal.accent);
            auto mono = [&](const std::string& s, const ImVec4& c) {
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(3));
                ImGui::PushFont(g_fonts.mono_small);
                ImGui::TextColored(c, "%s", s.c_str());
                ImGui::PopFont();
            };
            ImGui::TableSetColumnIndex(3); mono(g.rom.game_code, g_pal.text_dim);
            ImGui::TableSetColumnIndex(4); mono(format_size(g.rom.size), g_pal.text_dim);
            ImGui::TableSetColumnIndex(5); mono("." + g.rom.extension, g_pal.text_faint);
            ImGui::TableSetColumnIndex(6);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(2));
            ImGui::TextColored(g.stats.last_played ? g_pal.text_dim : g_pal.text_faint, "%s", relative_time(g.stats.last_played).c_str());
            ImGui::TableSetColumnIndex(7); mono(play_time(g.stats.play_seconds), g_pal.text_dim);
            ImGui::PopID();
        }
    }
    ImGui::EndTable();
    ImGui::PopStyleVar();
}

void App::game_context_menu(const GameEntry& g) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(8, 8));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, dp(10, 9));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, g_pal.bg2);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, g_pal.bg4);
    ImGui::SetNextWindowSizeConstraints(ImVec2(dp(230), 0), ImVec2(FLT_MAX, FLT_MAX));
    if (ImGui::BeginPopupContextItem("##ctx")) {
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::TextColored(g_pal.text_faint, "%s", g.rom.display_title.c_str());
        ImGui::PopFont();
        ImGui::Separator();
        bool playing = core_.loaded() && g.key == current_key_;
        ImGui::PushFont(g_fonts.body_bold);
        if (ImGui::MenuItem(playing ? "Return to Game" : "Play")) {
            if (playing) view_ = View::Game;
            else pending_launch_ = g.rom.path;
        }
        ImGui::PopFont();
        if (ImGui::MenuItem("Properties\xE2\x80\xA6")) {
            props_key_ = g.key;
            props_open_ = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem(g.stats.favorite ? "Remove from Favorites" : "Add to Favorites")) library_.toggle_favorite(g.key);
        if (g.stats.last_played && ImGui::MenuItem("Remove from Recently Played")) library_.forget_recent(g.key);
        ImGui::Separator();
        if (ImGui::MenuItem(platform::reveal_action_label())) platform::reveal_in_file_manager(g.rom.path);
        if (ImGui::MenuItem("Copy File Path")) ImGui::SetClipboardText(platform::path_to_utf8(g.rom.path).c_str());
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
}

void App::draw_library_empty(ImVec2 size) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float cx = p.x + size.x * 0.5f;
    float top = p.y + std::max(dp(40), size.y * 0.5f - dp(190));

    // Illustration: stacked cartridges in a glowing circle.
    ImVec2 c(cx, top + dp(70));
    for (int i = 4; i >= 1; --i) dl->AddCircleFilled(c, dp(40) + dp(14) * i, col(g_pal.accent, 0.035f * (5 - i)), 64);
    dl->AddCircleFilled(c, dp(48), col(g_pal.bg3), 64);
    draw_icon(dl, Icon::Chip, c, dp(46), col(g_pal.accent_hover));

    ImGui::SetCursorScreenPos(ImVec2(p.x, top + dp(160)));
    auto centered = [&](ImFont* f, const ImVec4& color, const char* text) {
        ImGui::PushFont(f);
        float tw = ImGui::CalcTextSize(text).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (size.x - tw) * 0.5f);
        ImGui::TextColored(color, "%s", text);
        ImGui::PopFont();
    };
    centered(g_fonts.display, g_pal.text, "Build your game library");
    ImGui::Dummy(dp(0, 4));
    centered(g_fonts.body, g_pal.text_dim, "Add a folder containing Nintendo 64 ROMs (.z64, .n64, .v64).");
    centered(g_fonts.body, g_pal.text_dim, "Orbit64 scans it automatically and keeps your library in sync.");
    ImGui::Dummy(dp(0, 18));

    float bw1 = dp(180), bw2 = dp(160), gap = dp(12);
    float bx = p.x + (size.x - bw1 - bw2 - gap) * 0.5f;
    ImGui::SetCursorScreenPos(ImVec2(bx, ImGui::GetCursorScreenPos().y));
    if (button("Add ROM Folder", Icon::Plus, ButtonKind::Primary, bw1, true, dp(40))) action_add_folder();
    ImGui::SameLine(0, gap);
    if (button("Open ROM File", Icon::FolderOpen, ButtonKind::Subtle, bw2, true, dp(40))) action_open_rom();
    ImGui::Dummy(dp(0, 18));

    // Drop hint
    ImVec2 hp = ImGui::GetCursorScreenPos();
    float hw = std::min(size.x - dp(40), dp(460)), hh = dp(64);
    ImVec2 hmn(p.x + (size.x - hw) * 0.5f, hp.y), hmx(hmn.x + hw, hp.y + hh);
    // Dashed border
    const float dash = dp(8);
    for (float x = hmn.x + dp(10); x < hmx.x - dp(10); x += dash * 2) {
        dl->AddLine(ImVec2(x, hmn.y), ImVec2(std::min(x + dash, hmx.x - dp(10)), hmn.y), col(g_pal.border_strong), dp(1.5f));
        dl->AddLine(ImVec2(x, hmx.y), ImVec2(std::min(x + dash, hmx.x - dp(10)), hmx.y), col(g_pal.border_strong), dp(1.5f));
    }
    for (float y = hmn.y + dp(10); y < hmx.y - dp(10); y += dash * 2) {
        dl->AddLine(ImVec2(hmn.x, y), ImVec2(hmn.x, std::min(y + dash, hmx.y - dp(10))), col(g_pal.border_strong), dp(1.5f));
        dl->AddLine(ImVec2(hmx.x, y), ImVec2(hmx.x, std::min(y + dash, hmx.y - dp(10))), col(g_pal.border_strong), dp(1.5f));
    }
    draw_icon(dl, Icon::Download, ImVec2(hmn.x + dp(34), (hmn.y + hmx.y) * 0.5f), dp(20), col(g_pal.text_faint));
    dl->AddText(g_fonts.body_bold, font_px(g_fonts.body_bold), ImVec2(hmn.x + dp(60), hmn.y + dp(13)), col(g_pal.text_dim),
                "Drag & drop");
    dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(hmn.x + dp(60), hmn.y + dp(35)), col(g_pal.text_faint),
                "Drop a ROM to play it, or a folder to add it to your library.");
    ImGui::Dummy(ImVec2(size.x, hh));
}

} // namespace ui
