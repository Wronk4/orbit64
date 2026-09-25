#include "file_browser.hpp"
#include "platform.hpp"
#include "widgets.hpp"

#include "imgui_internal.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>

namespace ui {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Procedural cover art
// ---------------------------------------------------------------------------

static std::uint32_t fnv1a(const std::string& s) {
    std::uint32_t h = 2166136261u;
    for (unsigned char c : s) { h ^= c; h *= 16777619u; }
    return h;
}

void draw_cover_art(ImDrawList* dl, ImVec2 mn, ImVec2 mx, const RomInfo& rom, float rounding, bool with_title) {
    std::uint32_t h = fnv1a(rom.display_title + rom.game_code);
    float hue = (h % 360) / 360.0f;
    float w = mx.x - mn.x, hgt = mx.y - mn.y;
    ImVec4 c0, c1;
    ImGui::ColorConvertHSVtoRGB(hue, 0.55f, 0.52f, c0.x, c0.y, c0.z);
    ImGui::ColorConvertHSVtoRGB(std::fmod(hue + 0.08f, 1.0f), 0.65f, 0.20f, c1.x, c1.y, c1.z);
    c0.w = c1.w = 1.0f;

    int v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(mn, mx, IM_COL32_WHITE, rounding);
    ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, mn, mx, col(c0), col(c1));

    dl->PushClipRect(mn, mx, true);
    // Decorative concentric rings, offset by hash.
    float cx = mn.x + w * (0.55f + ((h >> 8) % 40) / 100.0f);
    float cy = mn.y + hgt * (0.15f + ((h >> 16) % 30) / 100.0f);
    for (int i = 1; i <= 5; ++i)
        dl->AddCircle(ImVec2(cx, cy), w * 0.16f * i, IM_COL32(255, 255, 255, 16 + 3 * (5 - i)), 64, std::max(1.0f, w * 0.012f));
    // Large faint initials.
    std::string initials;
    bool take = true;
    for (char c : rom.display_title) {
        if (take && std::isalnum(static_cast<unsigned char>(c))) {
            initials.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
            if (initials.size() == 2) break;
        }
        take = (c == ' ' || c == '-');
    }
    ImFont* df = g_fonts.display;
    float isz = hgt * 0.42f;
    ImVec2 its = df->CalcTextSizeA(isz, FLT_MAX, 0, initials.c_str());
    dl->AddText(df, isz, ImVec2(mx.x - its.x - w * 0.06f, mn.y + hgt * 0.10f), IM_COL32(255, 255, 255, 34), initials.c_str());

    // Top label strip, like a cartridge label.
    ImFont* sf = g_fonts.small_bold;
    float pad = std::max(dp(8), w * 0.07f);
    dl->AddRectFilled(ImVec2(mn.x + pad, mn.y + pad), ImVec2(mn.x + pad + dp(34), mn.y + pad + dp(18)), IM_COL32(0, 0, 0, 90), dp(4));
    dl->AddText(sf, font_px(sf) * 0.9f, ImVec2(mn.x + pad + dp(6), mn.y + pad + dp(2.5f)), IM_COL32(255, 255, 255, 220), "N64");

    if (!with_title) {
        dl->PopClipRect();
        return;
    }
    // Title at the bottom with a darkening gradient behind it.
    gradient_rect_v(dl, ImVec2(mn.x, mn.y + hgt * 0.45f), mx, IM_COL32(0, 0, 0, 255), 0, 150, rounding, ImDrawFlags_RoundCornersBottom);
    ImFont* tf = g_fonts.title;
    float tsz = std::clamp(w * 0.105f, dp(12), font_px(tf));
    float wrap = w - pad * 2;
    ImVec2 ts = tf->CalcTextSizeA(tsz, FLT_MAX, wrap, rom.display_title.c_str());
    ts.y = std::min(ts.y, tsz * 3.3f);
    dl->AddText(tf, tsz, ImVec2(mn.x + pad, mx.y - pad - ts.y), IM_COL32(255, 255, 255, 240), rom.display_title.c_str(), nullptr, wrap);
    dl->PopClipRect();
}

// ---------------------------------------------------------------------------
// Browser
// ---------------------------------------------------------------------------

static std::string format_time(fs::file_time_type ft) {
    using namespace std::chrono;
    auto sys = time_point_cast<system_clock::duration>(ft - fs::file_time_type::clock::now() + system_clock::now());
    std::time_t t = system_clock::to_time_t(sys);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M", &tm);
    return buf;
}

void FileBrowser::open(Mode mode, const fs::path& start_dir) {
    mode_ = mode;
    open_ = true;
    confirmed_ = false;
    result_.clear();
    back_.clear();
    forward_.clear();
    filter_[0] = 0;
    editing_path_ = false;
    std::error_code ec;
    fs::path dir = start_dir;
    if (dir.empty() || !fs::is_directory(dir, ec)) dir = fs::current_path(ec);
    navigate(dir, false);
}

void FileBrowser::navigate(const fs::path& dir, bool push_history) {
    std::error_code ec;
    fs::path target = fs::weakly_canonical(dir, ec);
    if (ec) target = dir;
    if (push_history && !cwd_.empty() && target != cwd_) {
        back_.push_back(cwd_);
        forward_.clear();
    }
    cwd_ = target;
    selected_ = -1;
    scroll_to_top_ = true;
    preview_ = RomInfo{};
    std::snprintf(path_buf_, sizeof path_buf_, "%s", platform::path_to_utf8(cwd_).c_str());
    refresh();
}

void FileBrowser::refresh() {
    entries_.clear();
    error_.clear();
    std::error_code ec;
    fs::directory_iterator it(cwd_, fs::directory_options::skip_permission_denied, ec);
    if (ec) {
        error_ = "This folder can't be opened: " + ec.message();
        return;
    }
    for (const auto& de : it) {
        Entry e;
        e.path = de.path();
        e.name = platform::path_to_utf8(de.path().filename());
        if (e.name.empty() || e.name[0] == '.') continue; // hidden files (Unix convention)
        std::error_code ec2;
        e.dir = de.is_directory(ec2);
        if (!e.dir) {
            e.rom = is_rom_extension(e.path);
            e.size = de.file_size(ec2);
        }
        auto ft = de.last_write_time(ec2);
        if (!ec2) e.modified = format_time(ft);
        entries_.push_back(std::move(e));
    }
    std::sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) {
        if (a.dir != b.dir) return a.dir;
        std::string la = a.name, lb = b.name;
        std::transform(la.begin(), la.end(), la.begin(), [](unsigned char c) { return std::tolower(c); });
        std::transform(lb.begin(), lb.end(), lb.begin(), [](unsigned char c) { return std::tolower(c); });
        return la < lb;
    });
}

void FileBrowser::select(const std::string& file_name) {
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].name == file_name) {
            selected_ = static_cast<int>(i);
            scroll_to_selected_ = true;
            if (entries_[i].rom) preview_ = inspect_rom(entries_[i].path);
        }
    }
}

bool FileBrowser::confirm(const Entry* e) {
    if (mode_ == Mode::PickFolder) {
        result_ = (e && e->dir) ? e->path : cwd_;
        confirmed_ = true;
        open_ = false;
        return true;
    }
    if (!e) return false;
    if (e->dir) {
        navigate(e->path);
        return false;
    }
    result_ = e->path;
    confirmed_ = true;
    open_ = false;
    return true;
}

bool FileBrowser::draw() {
    confirmed_ = false;
    const char* title = mode_ == Mode::OpenRom ? "Open ROM" : "Choose Folder";
    if (!begin_modal("##file_browser", title, dp(1040, 660), &open_, mode_ == Mode::OpenRom ? Icon::Chip : Icon::Folder))
        return false;

    ImVec2 ws = ImGui::GetWindowSize();
    const float header = dp(56), toolbar = dp(52), footer = dp(64);
    const float body_h = ws.y - header - toolbar - footer;
    const float sidebar_w = dp(196);
    const bool show_preview = mode_ == Mode::OpenRom && ws.x > dp(860);
    const float preview_w = show_preview ? dp(270) : 0.0f;

    ImGui::SetCursorPos(ImVec2(0, header));
    draw_toolbar(ws.x);

    ImGui::SetCursorPos(ImVec2(0, header + toolbar));
    draw_sidebar(body_h);
    ImGui::SetCursorPos(ImVec2(sidebar_w, header + toolbar));
    draw_list(ws.x - sidebar_w - preview_w, body_h);
    if (show_preview) {
        ImGui::SetCursorPos(ImVec2(ws.x - preview_w, header + toolbar));
        draw_preview(preview_w, body_h);
    }

    // Footer
    modal_footer_begin(footer);
    if (native_available && on_native_requested) {
        if (button("Use system dialog", Icon::Monitor, ButtonKind::Ghost)) {
            open_ = false;
            ImGui::CloseCurrentPopup();
            on_native_requested(mode_);
        }
        ImGui::SameLine();
    }
    int roms = 0;
    for (const auto& e : entries_) roms += e.rom ? 1 : 0;
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(8));
    ImGui::PushFont(g_fonts.small);
    if (mode_ == Mode::OpenRom) ImGui::TextColored(g_pal.text_faint, "%d ROM%s in this folder", roms, roms == 1 ? "" : "s");
    else ImGui::TextColored(g_pal.text_faint, "Folders containing ROMs are scanned recursively");
    ImGui::PopFont();
    modal_footer_end();

    const Entry* sel = (selected_ >= 0 && selected_ < (int)entries_.size()) ? &entries_[selected_] : nullptr;
    float bw = dp(120);
    ImGui::SetCursorPos(ImVec2(ws.x - dp(20) - bw, ws.y - footer + (footer - dp(34)) * 0.5f));
    bool can_open = mode_ == Mode::PickFolder || (sel && (sel->rom || sel->dir || show_all_));
    const char* primary = mode_ == Mode::PickFolder ? (sel && sel->dir ? "Select Folder" : "Select This Folder")
                                                    : (sel && sel->dir ? "Open Folder" : "Open ROM");
    if (mode_ == Mode::PickFolder) bw = dp(160);
    ImGui::SetCursorPos(ImVec2(ws.x - dp(20) - bw, ws.y - footer + (footer - dp(34)) * 0.5f));
    bool result = false;
    if (button(primary, Icon::None, ButtonKind::Primary, bw, can_open)) result = confirm(sel);
    ImGui::SetCursorPos(ImVec2(ws.x - dp(20) - bw - dp(10) - dp(96), ws.y - footer + (footer - dp(34)) * 0.5f));
    if (button("Cancel", Icon::None, ButtonKind::Subtle, dp(96))) open_ = false;

    if (!open_) ImGui::CloseCurrentPopup();
    end_modal();
    return result || confirmed_;
}

void FileBrowser::draw_toolbar(float width) {
    ImVec2 wp = ImGui::GetWindowPos();
    ImVec2 p0(wp.x, ImGui::GetCursorScreenPos().y);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float h = dp(52);
    dl->AddLine(ImVec2(p0.x, p0.y + h), ImVec2(p0.x + width, p0.y + h), col(g_pal.border));

    float bs = dp(30);
    float y = p0.y + (h - bs) * 0.5f;
    ImGui::SetCursorScreenPos(ImVec2(p0.x + dp(16), y));
    if (icon_button("back", Icon::ChevronLeft, bs, "Back", false, !back_.empty()) && !back_.empty()) {
        forward_.push_back(cwd_);
        fs::path p = back_.back();
        back_.pop_back();
        navigate(p, false);
    }
    ImGui::SameLine(0, dp(2));
    if (icon_button("fwd", Icon::ChevronRight, bs, "Forward", false, !forward_.empty()) && !forward_.empty()) {
        back_.push_back(cwd_);
        fs::path p = forward_.back();
        forward_.pop_back();
        navigate(p, false);
    }
    ImGui::SameLine(0, dp(2));
    bool has_parent = cwd_.has_parent_path() && cwd_.parent_path() != cwd_;
    if (icon_button("up", Icon::ArrowUp, bs, "Parent folder", false, has_parent)) navigate(cwd_.parent_path());
    ImGui::SameLine(0, dp(10));

    float right_w = mode_ == Mode::OpenRom ? dp(200) + dp(130) : dp(200);
    float crumb_x = ImGui::GetCursorScreenPos().x;
    float crumb_w = width - (crumb_x - p0.x) - right_w - dp(28);

    // Breadcrumb (click the bar background to type a path instead).
    ImVec2 bmn(crumb_x, y), bmx(crumb_x + crumb_w, y + bs);
    if (editing_path_) {
        ImGui::SetCursorScreenPos(bmn);
        ImGui::SetNextItemWidth(crumb_w);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(dp(10), (bs - ImGui::GetFontSize()) * 0.5f));
        if (!ImGui::IsAnyItemActive()) ImGui::SetKeyboardFocusHere();
        bool enter = ImGui::InputText("##path", path_buf_, sizeof path_buf_, ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::PopStyleVar();
        if (enter) {
            fs::path p = platform::utf8_to_path(path_buf_);
            std::error_code ec;
            if (fs::is_directory(p, ec)) navigate(p);
            else if (fs::is_regular_file(p, ec)) {
                navigate(p.parent_path());
                for (size_t i = 0; i < entries_.size(); ++i)
                    if (entries_[i].path.filename() == p.filename()) selected_ = static_cast<int>(i);
            } else toast("Path not found", ToastKind::Warning);
            editing_path_ = false;
        } else if (ImGui::IsItemDeactivated()) {
            editing_path_ = false;
        }
    } else {
        dl->AddRectFilled(bmn, bmx, col(g_pal.bg2), dp(7));
        dl->AddRect(bmn, bmx, col(g_pal.border), dp(7));
        ImGui::SetCursorScreenPos(bmn);
        ImGui::SetNextItemAllowOverlap(); // the path segments drawn on top must stay clickable
        ImGui::InvisibleButton("##crumbbg", ImVec2(crumb_w, bs));
        if (ImGui::IsItemClicked()) editing_path_ = true;
        if (ImGui::IsItemHovered()) tooltip("Click to type a path");

        std::vector<std::pair<std::string, fs::path>> parts;
        fs::path acc;
        for (const auto& part : cwd_) {
            acc /= part;
            std::string s = platform::path_to_utf8(part);
            if (s == "/" || s == "\\") continue;
            if (s.empty()) continue;
            parts.push_back({s, acc});
        }
        if (cwd_.has_root_directory() && platform::current_os() != platform::OS::Windows)
            parts.insert(parts.begin(), {"/", cwd_.root_path()});
        // Fit from the right.
        ImFont* f = g_fonts.body;
        float avail = crumb_w - dp(20);
        float used = 0;
        int first = static_cast<int>(parts.size());
        for (int i = static_cast<int>(parts.size()) - 1; i >= 0; --i) {
            float wi = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, parts[i].first.c_str()).x + dp(26);
            if (used + wi > avail - dp(30) && i != static_cast<int>(parts.size()) - 1) break;
            used += wi;
            first = i;
        }
        float x = bmn.x + dp(8);
        if (first > 0) {
            dl->AddText(f, font_px(f), ImVec2(x, y + (bs - font_px(f)) * 0.5f), col(g_pal.text_faint), "\xE2\x80\xA6");
            x += dp(18);
        }
        for (int i = first; i < static_cast<int>(parts.size()); ++i) {
            const auto& [label, path] = parts[i];
            float tw = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, label.c_str()).x;
            ImGui::SetCursorScreenPos(ImVec2(x - dp(4), y + dp(3)));
            ImGui::PushID(i);
            bool last = i == static_cast<int>(parts.size()) - 1;
            if (ImGui::InvisibleButton("##crumb", ImVec2(tw + dp(8), bs - dp(6))) && !last) navigate(path);
            bool hov = ImGui::IsItemHovered();
            ImGui::PopID();
            if (hov) dl->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), col(g_pal.bg4), dp(5));
            float clip_right = bmx.x - dp(6);
            if (x > clip_right) break;
            dl->PushClipRect(bmn, ImVec2(clip_right, bmx.y), true);
            dl->AddText(f, font_px(f), ImVec2(x, y + (bs - font_px(f)) * 0.5f), col(last ? g_pal.text : g_pal.text_dim), label.c_str());
            dl->PopClipRect();
            x += tw + dp(6);
            if (!last) {
                draw_icon(dl, Icon::ChevronRight, ImVec2(x + dp(5), y + bs * 0.5f), dp(10), col(g_pal.text_faint));
                x += dp(14);
            }
        }
    }

    // Filter + "all files"
    float rx = p0.x + width - dp(16) - right_w;
    ImGui::SetCursorScreenPos(ImVec2(rx, y - dp(1)));
    search_field("filter", filter_, sizeof filter_, "Filter", dp(200));
    if (mode_ == Mode::OpenRom) {
        ImGui::SetCursorScreenPos(ImVec2(rx + dp(214), y + dp(4)));
        toggle("allfiles", &show_all_);
        ImGui::SameLine(0, dp(8));
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(1));
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_dim, "All files");
        ImGui::PopFont();
    }
}

void FileBrowser::draw_sidebar(float height) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, g_pal.bg0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(10, 12));
    ImGui::BeginChild("##places", ImVec2(dp(196), height), ImGuiChildFlags_AlwaysUseWindowPadding);
    auto place_item = [&](const std::string& label, const fs::path& p, Icon icon, int idx) {
        ImGui::PushID(idx);
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float w = ImGui::GetContentRegionAvail().x, h = dp(30);
        bool clicked = ImGui::InvisibleButton("##place", ImVec2(w, h));
        bool hov = ImGui::IsItemHovered();
        std::error_code ec;
        bool active = fs::equivalent(p, cwd_, ec);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        if (active) dl->AddRectFilled(pos, ImVec2(pos.x + w, pos.y + h), col(g_pal.accent_soft), dp(6));
        else if (hov) dl->AddRectFilled(pos, ImVec2(pos.x + w, pos.y + h), col(g_pal.bg3), dp(6));
        ImVec4 fc = active ? g_pal.accent_hover : (hov ? g_pal.text : g_pal.text_dim);
        draw_icon(dl, icon, ImVec2(pos.x + dp(16), pos.y + h * 0.5f), dp(15), col(fc));
        text_ellipsis(dl, g_fonts.body, ImVec2(pos.x + dp(32), pos.y + (h - font_px(g_fonts.body)) * 0.5f), w - dp(38),
                      col(active ? g_pal.text : fc), label.c_str());
        if (hov) tooltip(platform::path_to_utf8(p).c_str());
        if (clicked) navigate(p);
        ImGui::PopID();
    };
    auto group = [&](const char* name) {
        ImGui::Dummy(dp(0, 4));
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + dp(6));
        ImGui::TextColored(g_pal.text_faint, "%s", name);
        ImGui::PopFont();
    };
    int idx = 0;
    group("PLACES");
    for (const auto& pl : platform::quick_places())
        if (pl.icon != static_cast<int>(Icon::Drive)) place_item(pl.label, pl.path, static_cast<Icon>(pl.icon), idx++);
    if (!library_dirs.empty()) {
        group("LIBRARY FOLDERS");
        for (const auto& d : library_dirs) {
            std::string name = platform::path_to_utf8(d.filename());
            if (name.empty()) name = platform::path_to_utf8(d);
            place_item(name, d, Icon::Library, idx++);
        }
    }
    group(platform::current_os() == platform::OS::Windows ? "DRIVES" : "VOLUMES");
    for (const auto& pl : platform::quick_places())
        if (pl.icon == static_cast<int>(Icon::Drive)) place_item(pl.label, pl.path, Icon::Drive, idx++);
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImVec2 mn = ImGui::GetItemRectMin(), mx = ImGui::GetItemRectMax();
    ImGui::GetWindowDrawList()->AddLine(ImVec2(mx.x, mn.y), mx, col(g_pal.border));
}

void FileBrowser::draw_list(float width, float height) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(8, 8));
    ImGui::BeginChild("##list", ImVec2(width, height), ImGuiChildFlags_AlwaysUseWindowPadding);

    std::string filt = filter_;
    std::transform(filt.begin(), filt.end(), filt.begin(), [](unsigned char c) { return std::tolower(c); });
    std::vector<int> visible;
    for (int i = 0; i < static_cast<int>(entries_.size()); ++i) {
        const Entry& e = entries_[i];
        if (mode_ == Mode::PickFolder && !e.dir) continue;
        if (mode_ == Mode::OpenRom && !e.dir && !e.rom && !show_all_) continue;
        if (!filt.empty()) {
            std::string n = e.name;
            std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return std::tolower(c); });
            if (n.find(filt) == std::string::npos) continue;
        }
        visible.push_back(i);
    }

    // Keyboard navigation
    if (ImGui::IsWindowFocused() && !visible.empty() && !ImGui::IsAnyItemActive()) {
        int pos = static_cast<int>(std::find(visible.begin(), visible.end(), selected_) - visible.begin());
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) { pos = std::min<int>(pos + 1, (int)visible.size() - 1); if (selected_ < 0) pos = 0; selected_ = visible[pos]; scroll_to_selected_ = true; }
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) { pos = std::max(0, std::min<int>(pos, (int)visible.size()) - 1); selected_ = visible[pos]; scroll_to_selected_ = true; }
        if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) {
            if (selected_ >= 0) {
                Entry e = entries_[selected_]; // copy: confirm() may rebuild entries_
                confirm(&e);
            }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Backspace) && cwd_.has_parent_path()) navigate(cwd_.parent_path());
    }

    if (!error_.empty()) {
        ImGui::Dummy(dp(0, 40));
        ImVec2 p = ImGui::GetCursorScreenPos();
        draw_icon(ImGui::GetWindowDrawList(), Icon::Warning, ImVec2(p.x + width * 0.5f, p.y + dp(16)), dp(28), col(g_pal.warning));
        ImGui::Dummy(dp(0, 40));
        float tw = ImGui::CalcTextSize(error_.c_str()).x;
        ImGui::SetCursorPosX(std::max(dp(8), (width - tw) * 0.5f));
        ImGui::TextColored(g_pal.text_dim, "%s", error_.c_str());
    } else if (visible.empty()) {
        ImGui::Dummy(dp(0, 60));
        ImVec2 p = ImGui::GetCursorScreenPos();
        draw_icon(ImGui::GetWindowDrawList(), Icon::Folder, ImVec2(p.x + width * 0.5f - dp(8), p.y + dp(16)), dp(34), col(g_pal.text_faint));
        ImGui::Dummy(dp(0, 44));
        const char* msg = mode_ == Mode::OpenRom ? "No ROMs or folders here" : "No sub-folders here";
        float tw = ImGui::CalcTextSize(msg).x;
        ImGui::SetCursorPosX((width - tw) * 0.5f);
        ImGui::TextColored(g_pal.text_dim, "%s", msg);
    } else if (ImGui::BeginTable(("##files" + platform::path_to_utf8(cwd_)).c_str(), 3, ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_PadOuterX,
                                 ImVec2(0, 0))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, dp(80));
        ImGui::TableSetupColumn("Modified", ImGuiTableColumnFlags_WidthFixed, dp(130));
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::PushStyleColor(ImGuiCol_Text, g_pal.text_faint);
        ImGui::TableHeadersRow();
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(visible.size()), dp(32));
        if (scroll_to_selected_) {
            int pos = static_cast<int>(std::find(visible.begin(), visible.end(), selected_) - visible.begin());
            if (pos < (int)visible.size()) clipper.IncludeItemByIndex(pos);
        }
        while (clipper.Step()) {
            for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
                int i = visible[r];
                const Entry& e = entries_[i];
                ImGui::TableNextRow(0, dp(32));
                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(i);
                bool sel = selected_ == i;
                ImVec2 rp = ImGui::GetCursorScreenPos();
                if (ImGui::Selectable("##row", sel, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick,
                                      ImVec2(0, dp(32)))) {
                    selected_ = i;
                    if (e.rom) preview_ = inspect_rom(e.path);
                    // Deferred: confirming a folder rebuilds entries_ while this loop still uses them.
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) pending_confirm_ = i;
                }
                if (sel && scroll_to_selected_) {
                    ImGui::SetScrollHereY();
                    scroll_to_selected_ = false;
                    if (e.rom && preview_.path != e.path) preview_ = inspect_rom(e.path);
                }
                ImDrawList* dl = ImGui::GetWindowDrawList();
                Icon ic = e.dir ? Icon::Folder : (e.rom ? Icon::Chip : Icon::File);
                ImVec4 icol = e.dir ? g_pal.accent : (e.rom ? g_pal.success : g_pal.text_faint);
                draw_icon(dl, ic, ImVec2(rp.x + dp(12), rp.y + dp(16)), dp(16), col(icol));
                float name_w = ImGui::GetColumnWidth() - dp(36);
                text_ellipsis(dl, g_fonts.body, ImVec2(rp.x + dp(30), rp.y + dp(16) - font_px(g_fonts.body) * 0.5f), name_w,
                              col(e.dir || e.rom ? g_pal.text : g_pal.text_dim), e.name.c_str());
                ImGui::TableSetColumnIndex(1);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(8));
                ImGui::PushFont(g_fonts.mono_small);
                ImGui::TextColored(g_pal.text_dim, "%s", e.dir ? "\xE2\x80\x94" : format_size(e.size).c_str());
                ImGui::PopFont();
                ImGui::TableSetColumnIndex(2);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(8));
                ImGui::PushFont(g_fonts.mono_small);
                ImGui::TextColored(g_pal.text_faint, "%s", e.modified.c_str());
                ImGui::PopFont();
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    if (pending_confirm_ >= 0 && pending_confirm_ < static_cast<int>(entries_.size())) {
        Entry e = entries_[pending_confirm_]; // copy: confirm() may rebuild entries_
        pending_confirm_ = -1;
        confirm(&e);
    }
    pending_confirm_ = -1;
}

void FileBrowser::draw_preview(float width, float height) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, g_pal.bg0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(18, 18));
    ImGui::BeginChild("##preview", ImVec2(width, height), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddLine(wp, ImVec2(wp.x, wp.y + height), col(g_pal.border));
    float w = ImGui::GetContentRegionAvail().x;

    const Entry* sel = (selected_ >= 0 && selected_ < (int)entries_.size()) ? &entries_[selected_] : nullptr;
    if (!sel || !sel->rom) {
        ImGui::Dummy(ImVec2(0, height * 0.28f));
        ImVec2 p = ImGui::GetCursorScreenPos();
        draw_icon(dl, Icon::Chip, ImVec2(p.x + w * 0.5f, p.y + dp(20)), dp(40), col(g_pal.text_faint));
        ImGui::Dummy(dp(0, 52));
        const char* msg = "Select a ROM to see its details";
        ImGui::PushFont(g_fonts.small);
        float tw = ImGui::CalcTextSize(msg).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (w - tw) * 0.5f));
        ImGui::TextColored(g_pal.text_dim, "%s", msg);
        ImGui::PopFont();
    } else if (!preview_.valid) {
        ImGui::Dummy(ImVec2(0, height * 0.28f));
        ImVec2 p = ImGui::GetCursorScreenPos();
        draw_icon(dl, Icon::Warning, ImVec2(p.x + w * 0.5f, p.y + dp(20)), dp(36), col(g_pal.warning));
        ImGui::Dummy(dp(0, 52));
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
        ImGui::TextColored(g_pal.text_dim, "This file doesn't have a valid Nintendo 64 ROM header.");
        ImGui::PopTextWrapPos();
    } else {
        ImVec2 p = ImGui::GetCursorScreenPos();
        float cover_h = w * 0.72f;
        ImTextureID cover = cover_for ? cover_for(preview_) : ImTextureID{};
        if (cover) dl->AddImageRounded(cover, p, ImVec2(p.x + w, p.y + cover_h), ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, dp(10));
        else draw_cover_art(dl, p, ImVec2(p.x + w, p.y + cover_h), preview_, dp(10));
        ImGui::Dummy(ImVec2(w, cover_h + dp(8)));
        ImGui::PushFont(g_fonts.body_bold);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
        ImGui::TextUnformatted(preview_.display_title.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopFont();
        badge(preview_.region_short.c_str(), g_pal.accent);
        if (!preview_.tags.empty()) {
            ImGui::SameLine(0, dp(6));
            ImGui::PushFont(g_fonts.small);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(1));
            ImGui::TextColored(g_pal.text_faint, "%s", preview_.tags.c_str());
            ImGui::PopFont();
        }
        ImGui::Dummy(dp(0, 6));
        auto kv = [&](const char* k, const std::string& v) {
            ImGui::PushFont(g_fonts.small);
            ImGui::TextColored(g_pal.text_faint, "%s", k);
            ImGui::SameLine(dp(18) + dp(96));
            ImGui::PopFont();
            ImGui::PushFont(g_fonts.mono_small);
            ImGui::TextColored(g_pal.text, "%s", v.c_str());
            ImGui::PopFont();
        };
        char crc[40];
        std::snprintf(crc, sizeof crc, "%08X-%08X", preview_.crc1, preview_.crc2);
        kv("Internal name", preview_.internal_name);
        kv("Game code", preview_.game_code);
        kv("Region", preview_.region);
        kv("Revision", "1." + std::to_string(preview_.version));
        kv("Size", format_size(preview_.size));
        kv("Format", preview_.extension.empty() ? "?" : "." + preview_.extension);
        kv("CRC", crc);
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

} // namespace ui
