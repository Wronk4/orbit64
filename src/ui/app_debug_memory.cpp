// MEMORY tools: Memory Search, RAM Watch, Memory Editor, Freeze List.
// All of them operate on RDRAM (big-endian, as on hardware) and work in any game.

#include "app.hpp"
#include "platform.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace ui {

namespace {

std::string hex32(std::uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof b, "%08X", v);
    return b;
}

std::string hex_raw(std::uint32_t raw, MemType t) {
    char b[16];
    std::snprintf(b, sizeof b, "%0*X", mem_type_size(t) * 2, raw);
    return b;
}

bool type_combo(const char* id, int* type, float width) {
    return combo(id, type, kMemTypeNames, kMemTypeCount, width);
}

std::uint32_t read_be(const std::vector<std::uint8_t>& ram, std::size_t off, int size) {
    std::uint32_t v = 0;
    for (int i = 0; i < size; ++i) v = (v << 8) | ram[off + i];
    return v;
}

void small_label(const char* s) {
    ImGui::PushFont(g_fonts.small);
    ImGui::TextColored(g_pal.text_faint, "%s", s);
    ImGui::PopFont();
}

const char* const kFirstModes[] = {"Exact value", "Unknown initial value", "Greater than", "Less than"};
const char* const kNextModes[] = {"Equal to", "Not equal to", "Greater than", "Less than",
                                  "Increased", "Decreased", "Changed", "Unchanged"};
constexpr std::size_t kListLimit = 100000; // candidates listed individually below this count

} // namespace

// ---------------------------------------------------------------------------
// Cross-tool actions
// ---------------------------------------------------------------------------

void App::add_watch(const std::string& name, std::uint32_t addr, MemType type) {
    Watch w;
    w.name = name.empty() ? hex32(addr) : name;
    w.addr = addr;
    w.type = type;
    dbg_.watches.push_back(w);
    save_debug_lists();
    open_tool(DebugTool::RamWatch);
    toast("Added " + hex32(addr) + " to RAM Watch", ToastKind::Success, 2.0f);
}

void App::add_freeze(const std::string& name, std::uint32_t addr, MemType type, const std::string& value) {
    Freeze f;
    f.name = name.empty() ? hex32(addr) : name;
    f.addr = addr;
    f.type = type;
    f.value = value;
    if (f.value.empty() && dbg_.snap) f.value = mem_format(mem_read_raw(*dbg_.snap, addr, type), type);
    dbg_.freezes.push_back(f);
    dbg_.freezes_dirty = true;
    save_debug_lists();
    open_tool(DebugTool::FreezeList);
    toast("Freezing " + hex32(addr) + " at " + f.value, ToastKind::Success, 2.0f);
}

void App::open_in_editor(std::uint32_t addr, MemType type) {
    dbg_.edit_addr = addr;
    std::snprintf(dbg_.edit_addr_text, sizeof dbg_.edit_addr_text, "%08X", addr);
    dbg_.edit_type = static_cast<int>(type);
    dbg_.edit_scroll = true;
    dbg_.edit_value[0] = 0;
    open_tool(DebugTool::MemoryEditor);
}

// ---------------------------------------------------------------------------
// Memory Search
// ---------------------------------------------------------------------------

void App::tool_memory_search() {
    const float w = ImGui::GetContentRegionAvail().x;
    if (!dbg_.snap || dbg_.snap->rdram.empty()) {
        spinner(dp(10), dp(2), col(g_pal.accent));
        return;
    }
    const auto& ram = dbg_.snap->rdram;
    const MemType type = static_cast<MemType>(dbg_.search_type);
    const int size = mem_type_size(type);

    // ---- Controls
    const float gap = dp(8);
    float col_w = (w - gap * 2) / 3;
    const float gy = ImGui::GetCursorPosY(); // groups on one row share a top edge
    ImGui::BeginGroup();
    small_label("Data type");
    ImGui::BeginDisabled(dbg_.search_started);
    type_combo("stype", &dbg_.search_type, col_w);
    ImGui::EndDisabled();
    ImGui::EndGroup();
    ImGui::SameLine(0, gap);
    ImGui::SetCursorPosY(gy);
    ImGui::BeginGroup();
    small_label("Scan type");
    const char* const* modes = dbg_.search_started ? kNextModes : kFirstModes;
    int nmodes = dbg_.search_started ? 8 : 4;
    dbg_.search_mode = std::clamp(dbg_.search_mode, 0, nmodes - 1);
    combo("smode", &dbg_.search_mode, modes, nmodes, col_w);
    ImGui::EndGroup();
    ImGui::SameLine(0, gap);
    ImGui::SetCursorPosY(gy);
    ImGui::BeginGroup();
    small_label("Value");
    bool needs_value = dbg_.search_started ? dbg_.search_mode <= 3 : dbg_.search_mode != 1;
    ImGui::BeginDisabled(!needs_value);
    ImGui::SetNextItemWidth(col_w);
    bool enter = ImGui::InputTextWithHint("##sval", type == MemType::F32 ? "e.g. 1.5" : "e.g. 100 or 0x64", dbg_.search_value,
                                          sizeof dbg_.search_value, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::EndDisabled();
    ImGui::EndGroup();

    ImGui::Dummy(dp(0, 2));
    float bw = (w - gap * 2) / 3;
    bool do_first = false, do_next = false;
    if (button("First Scan", Icon::Search, dbg_.search_started ? ButtonKind::Subtle : ButtonKind::Primary, bw, true, dp(34)) ||
        (enter && !dbg_.search_started))
        do_first = true;
    ImGui::SameLine(0, gap);
    if (button("Next Scan", Icon::Refresh, dbg_.search_started ? ButtonKind::Primary : ButtonKind::Subtle, bw, dbg_.search_started, dp(34)) ||
        (enter && dbg_.search_started))
        do_next = dbg_.search_started;
    ImGui::SameLine(0, gap);
    if (button("Reset", Icon::Close, ButtonKind::Ghost, bw, dbg_.search_started, dp(34))) {
        dbg_.search_started = false;
        dbg_.search_mask.clear();
        dbg_.search_prev.clear();
        dbg_.search_results.clear();
        dbg_.search_count = 0;
        dbg_.search_scans = 0;
        dbg_.search_mode = 0;
    }

    // ---- Scan
    auto run_scan = [&](bool first) {
        double target = 0;
        if (needs_value) {
            std::vector<std::uint8_t> bytes;
            if (!mem_parse(dbg_.search_value, type, bytes)) {
                toast("Enter a valid " + std::string(kMemTypeNames[dbg_.search_type]) + " value", ToastKind::Warning);
                return;
            }
            std::uint32_t raw = 0;
            for (auto b : bytes) raw = (raw << 8) | b;
            target = mem_as_double(raw, type);
        }
        const std::size_t slots = ram.size() / size;
        if (first) dbg_.search_mask.assign(slots, 1);
        const bool is_float = type == MemType::F32;
        auto equal = [&](double a, double b) {
            return is_float ? std::fabs(a - b) <= 0.0005 + 1e-5 * std::fabs(b) : a == b;
        };
        const int mode = dbg_.search_mode;
        std::size_t count = 0;
        dbg_.search_results.clear();
        for (std::size_t s = 0; s < slots; ++s) {
            if (!dbg_.search_mask[s]) continue;
            std::size_t off = s * size;
            double v = mem_as_double(read_be(ram, off, size), type);
            if (is_float && !std::isfinite(v)) { dbg_.search_mask[s] = 0; continue; }
            bool keep;
            if (first) {
                switch (mode) {
                    case 0: keep = equal(v, target); break;
                    case 1: keep = true; break;
                    case 2: keep = v > target; break;
                    default: keep = v < target; break;
                }
            } else {
                double pv = mem_as_double(read_be(dbg_.search_prev, off, size), type);
                switch (mode) {
                    case 0: keep = equal(v, target); break;
                    case 1: keep = !equal(v, target); break;
                    case 2: keep = v > target; break;
                    case 3: keep = v < target; break;
                    case 4: keep = v > pv; break;
                    case 5: keep = v < pv; break;
                    case 6: keep = !equal(v, pv); break;
                    default: keep = equal(v, pv); break;
                }
            }
            if (!keep) { dbg_.search_mask[s] = 0; continue; }
            ++count;
            if (count <= kListLimit) dbg_.search_results.push_back(static_cast<std::uint32_t>(off));
        }
        if (count > kListLimit) dbg_.search_results.clear();
        dbg_.search_prev = ram;
        dbg_.search_count = count;
        dbg_.search_started = true;
        dbg_.search_scans = first ? 1 : dbg_.search_scans + 1;
        if (first) dbg_.search_mode = 0;
    };
    if (do_first) run_scan(true);
    if (do_next) run_scan(false);

    // ---- Results
    ImGui::Dummy(dp(0, 4));
    if (!dbg_.search_started) {
        ImGui::PushFont(g_fonts.small);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
        ImGui::TextColored(g_pal.text_dim,
                           "Scans all %zu MB of RDRAM. Start with the value you see in the game (or Unknown initial value), "
                           "change it in the game, then narrow the results with Next Scan.", ram.size() >> 20);
        ImGui::PopTextWrapPos();
        ImGui::PopFont();
        return;
    }
    ImGui::PushFont(g_fonts.body_bold);
    ImGui::Text("%zu result%s", dbg_.search_count, dbg_.search_count == 1 ? "" : "s");
    ImGui::PopFont();
    ImGui::SameLine(0, dp(8));
    ImGui::PushFont(g_fonts.small);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(2));
    ImGui::TextColored(g_pal.text_faint, "after %d scan%s \xC2\xB7 %s", dbg_.search_scans, dbg_.search_scans == 1 ? "" : "s",
                       kMemTypeNames[dbg_.search_type]);
    ImGui::PopFont();
    if (dbg_.search_count > kListLimit) {
        ImGui::TextColored(g_pal.warning, "Too many results to list \xE2\x80\x94 narrow them down with Next Scan.");
        return;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, dp(8, 4));
    if (ImGui::BeginTable("##sresults", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV,
                          ImVec2(0, 0))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Address", ImGuiTableColumnFlags_WidthFixed, dp(100));
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Previous scan", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, dp(90));
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::PushStyleColor(ImGuiCol_Text, g_pal.text_faint);
        ImGui::TableHeadersRow();
        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGuiListClipper clip;
        clip.Begin(static_cast<int>(dbg_.search_results.size()));
        while (clip.Step()) {
            for (int r = clip.DisplayStart; r < clip.DisplayEnd; ++r) {
                std::uint32_t off = dbg_.search_results[r];
                std::uint32_t addr = 0x80000000u | off;
                std::uint32_t cur = read_be(ram, off, size), prev = read_be(dbg_.search_prev, off, size);
                ImGui::TableNextRow(0, dp(26));
                ImGui::PushID(r);
                ImGui::PushFont(g_fonts.mono_small);
                ImGui::TableNextColumn();
                ImGui::TextColored(g_pal.text_dim, "%08X", addr);
                ImGui::TableNextColumn();
                ImGui::TextColored(cur != prev ? g_pal.warning : g_pal.text, "%s", mem_format(cur, type).c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(g_pal.text_faint, "%s", mem_format(prev, type).c_str());
                ImGui::PopFont();
                ImGui::TableNextColumn();
                if (icon_button("w", Icon::Eye, dp(24), "Add to RAM Watch")) add_watch("", addr, type);
                ImGui::SameLine(0, dp(4));
                if (icon_button("f", Icon::Snowflake, dp(24), "Freeze at current value")) add_freeze("", addr, type, "");
                ImGui::SameLine(0, dp(4));
                if (icon_button("e", Icon::Pencil, dp(24), "Open in Memory Editor")) open_in_editor(addr, type);
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
}

// ---------------------------------------------------------------------------
// RAM Watch
// ---------------------------------------------------------------------------

void App::tool_ram_watch() {
    const float w = ImGui::GetContentRegionAvail().x;
    const float gap = dp(8);

    // Add row
    float type_w = dp(80), add_w = dp(90);
    float name_w = (w - type_w - add_w - gap * 3) * 0.55f, addr_w = (w - type_w - add_w - gap * 3) * 0.45f;
    ImGui::SetNextItemWidth(name_w);
    ImGui::InputTextWithHint("##wname", "Name", dbg_.watch_name, sizeof dbg_.watch_name);
    ImGui::SameLine(0, gap);
    ImGui::SetNextItemWidth(addr_w);
    bool enter = ImGui::InputTextWithHint("##waddr", "Address (hex)", dbg_.watch_addr, sizeof dbg_.watch_addr,
                                          ImGuiInputTextFlags_CharsHexadecimal | ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine(0, gap);
    type_combo("wtype", &dbg_.watch_type, type_w);
    ImGui::SameLine(0, gap);
    if (button("Add", Icon::Plus, ButtonKind::Primary, add_w, true, dp(32)) || enter) {
        std::uint32_t a;
        if (parse_address(dbg_.watch_addr, a)) {
            add_watch(dbg_.watch_name, a, static_cast<MemType>(dbg_.watch_type));
            dbg_.watch_name[0] = dbg_.watch_addr[0] = 0;
        } else {
            toast("Enter a hexadecimal address, e.g. 8033B1AC", ToastKind::Warning);
        }
    }
    ImGui::Dummy(dp(0, 4));

    if (dbg_.watches.empty()) {
        ImGui::PushFont(g_fonts.small);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
        ImGui::TextColored(g_pal.text_dim, "No watched addresses yet. Add one above, or use the eye button in Memory Search results.");
        ImGui::PopTextWrapPos();
        ImGui::PopFont();
        return;
    }

    const double now = ImGui::GetTime();
    int remove = -1;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, dp(8, 4));
    if (ImGui::BeginTable("##watches", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV |
                                              ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("Address", ImGuiTableColumnFlags_WidthFixed, dp(90));
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, dp(76));
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Hex", ImGuiTableColumnFlags_WidthFixed, dp(80));
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, dp(90));
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::PushStyleColor(ImGuiCol_Text, g_pal.text_faint);
        ImGui::TableHeadersRow();
        ImGui::PopStyleColor();
        ImGui::PopFont();
        for (size_t i = 0; i < dbg_.watches.size(); ++i) {
            Watch& wt = dbg_.watches[i];
            ImGui::TableNextRow(0, dp(30));
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextColumn();
            char name[64];
            std::snprintf(name, sizeof name, "%s", wt.name.c_str());
            ImGui::SetNextItemWidth(-1);
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
            if (ImGui::InputText("##name", name, sizeof name)) wt.name = name;
            if (ImGui::IsItemDeactivatedAfterEdit()) save_debug_lists();
            ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            ImGui::PushFont(g_fonts.mono_small);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(5));
            ImGui::TextColored(g_pal.text_dim, "%08X", wt.addr);
            ImGui::PopFont();
            ImGui::TableNextColumn();
            int t = static_cast<int>(wt.type);
            if (type_combo("type", &t, dp(70))) {
                wt.type = static_cast<MemType>(t);
                save_debug_lists();
            }
            const bool have_ram = dbg_.snap && !dbg_.snap->rdram.empty();
            std::uint32_t raw = have_ram ? mem_read_raw(*dbg_.snap, wt.addr, wt.type) : wt.last;
            if (raw != wt.last) {
                wt.last = raw;
                wt.changed_at = now;
            }
            float flash = std::clamp(1.0f - static_cast<float>(now - wt.changed_at) / 0.6f, 0.0f, 1.0f);
            ImGui::TableNextColumn();
            ImGui::PushFont(g_fonts.mono);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(4));
            ImGui::TextColored(mix(g_pal.text, g_pal.warning, flash), "%s", have_ram ? mem_format(raw, wt.type).c_str() : "\xE2\x80\x94");
            ImGui::PopFont();
            ImGui::TableNextColumn();
            ImGui::PushFont(g_fonts.mono_small);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(5));
            ImGui::TextColored(g_pal.text_faint, "%s", hex_raw(raw, wt.type).c_str());
            ImGui::PopFont();
            ImGui::TableNextColumn();
            if (icon_button("e", Icon::Pencil, dp(26), "Edit in Memory Editor")) open_in_editor(wt.addr, wt.type);
            ImGui::SameLine(0, dp(4));
            if (icon_button("f", Icon::Snowflake, dp(26), "Freeze at current value")) add_freeze(wt.name, wt.addr, wt.type, "");
            ImGui::SameLine(0, dp(4));
            if (icon_button("x", Icon::Trash, dp(26), "Remove", false, true, ButtonKind::Danger)) remove = static_cast<int>(i);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
    if (remove >= 0) {
        dbg_.watches.erase(dbg_.watches.begin() + remove);
        save_debug_lists();
    }
}

// ---------------------------------------------------------------------------
// Memory Editor
// ---------------------------------------------------------------------------

void App::tool_memory_editor() {
    const float w = ImGui::GetContentRegionAvail().x;
    const float gap = dp(8);
    if (!dbg_.snap || dbg_.snap->rdram.empty()) {
        spinner(dp(10), dp(2), col(g_pal.accent));
        return;
    }
    const DebugSnapshot& s = *dbg_.snap;
    const MemType type = static_cast<MemType>(dbg_.edit_type);
    const int size = mem_type_size(type);

    // ---- Address / type / value row
    float addr_w = dp(130), type_w = dp(80), go_w = dp(56);
    const float gy = ImGui::GetCursorPosY();
    ImGui::BeginGroup();
    small_label("Address");
    ImGui::SetNextItemWidth(addr_w);
    bool go = ImGui::InputText("##eaddr", dbg_.edit_addr_text, sizeof dbg_.edit_addr_text,
                               ImGuiInputTextFlags_CharsHexadecimal | ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsUppercase);
    ImGui::SameLine(0, dp(4));
    go |= button("Go", Icon::None, ButtonKind::Subtle, go_w, true, dp(32));
    ImGui::EndGroup();
    if (go) {
        std::uint32_t a;
        if (parse_address(dbg_.edit_addr_text, a)) {
            dbg_.edit_addr = a;
            dbg_.edit_scroll = true;
        } else {
            toast("Enter a hexadecimal address", ToastKind::Warning);
        }
    }
    ImGui::SameLine(0, gap * 2);
    ImGui::SetCursorPosY(gy);
    ImGui::BeginGroup();
    small_label("Type");
    type_combo("etype", &dbg_.edit_type, type_w);
    ImGui::EndGroup();
    ImGui::SameLine(0, gap * 2);
    ImGui::SetCursorPosY(gy);
    ImGui::BeginGroup();
    small_label("Current value");
    std::uint32_t raw = mem_read_raw(s, dbg_.edit_addr, type);
    ImGui::PushFont(g_fonts.mono);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(7));
    ImGui::TextColored(g_pal.accent_hover, "%s", mem_format(raw, type).c_str());
    ImGui::SameLine(0, dp(8));
    ImGui::TextColored(g_pal.text_faint, "0x%s", hex_raw(raw, type).c_str());
    ImGui::PopFont();
    ImGui::EndGroup();

    ImGui::Dummy(dp(0, 2));
    float val_w = w - dp(110) - dp(40) - gap * 2;
    ImGui::SetNextItemWidth(val_w);
    bool enter = ImGui::InputTextWithHint("##eval", "New value", dbg_.edit_value, sizeof dbg_.edit_value, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine(0, gap);
    if (button("Write", Icon::Pencil, ButtonKind::Primary, dp(110), true, dp(32)) || enter) {
        std::vector<std::uint8_t> bytes;
        if (!s.valid(dbg_.edit_addr, size)) {
            toast("Address is outside RDRAM", ToastKind::Warning);
        } else if (mem_parse(dbg_.edit_value, type, bytes)) {
            core_.poke(dbg_.edit_addr, bytes);
            toast("Wrote " + std::string(dbg_.edit_value) + " to " + hex32(dbg_.edit_addr), ToastKind::Success, 2.0f);
        } else {
            toast("Enter a valid " + std::string(kMemTypeNames[dbg_.edit_type]) + " value", ToastKind::Warning);
        }
    }
    ImGui::SameLine(0, gap);
    if (icon_button("ewatch", Icon::Eye, dp(32), "Add to RAM Watch", false, true, ButtonKind::Subtle)) add_watch("", dbg_.edit_addr, type);

    // ---- Hex view
    ImGui::Dummy(dp(0, 4));
    ImFont* mf = g_fonts.mono;
    const float cw = mf->CalcTextSizeA(font_px(mf), FLT_MAX, 0, "0").x;
    const float row_h = dp(22);
    const std::uint32_t rows = static_cast<std::uint32_t>(s.rdram.size() / 16);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, g_pal.bg0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(10, 6));
    ImGui::BeginChild("##hex", ImVec2(0, 0), ImGuiChildFlags_AlwaysUseWindowPadding);
    const std::uint32_t sel = dbg_.edit_addr & 0x1FFFFFFFu;
    if (dbg_.edit_scroll) {
        ImGui::SetScrollY(std::max(0.0f, (sel / 16) * row_h - ImGui::GetWindowHeight() * 0.35f));
        dbg_.edit_scroll = false;
    }
    static std::vector<std::uint8_t> last_view;
    static std::uint32_t last_first = ~0u;
    static std::uint64_t last_frame = 0;
    std::vector<std::uint8_t> view;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGuiListClipper clip;
    clip.Begin(static_cast<int>(rows), row_h);
    std::uint32_t first_row = ~0u;
    while (clip.Step()) {
        for (int r = clip.DisplayStart; r < clip.DisplayEnd; ++r) {
            if (first_row == ~0u) first_row = static_cast<std::uint32_t>(r);
            std::uint32_t base = static_cast<std::uint32_t>(r) * 16;
            ImVec2 p = ImGui::GetCursorScreenPos();
            char a[12];
            std::snprintf(a, sizeof a, "%08X", 0x80000000u | base);
            dl->AddText(mf, font_px(mf), ImVec2(p.x, p.y + dp(3)), col(g_pal.text_faint), a);
            float x0 = p.x + cw * 10;
            for (int i = 0; i < 16; ++i) {
                std::uint32_t off = base + i;
                std::uint8_t b = s.rdram[off];
                view.push_back(b);
                float x = x0 + i * cw * 3 + (i / 4) * cw;
                bool in_sel = off >= sel && off < sel + static_cast<std::uint32_t>(size);
                std::size_t vi = (static_cast<std::size_t>(r) - (last_first == ~0u ? 0 : last_first)) * 16 + i;
                bool changed = last_first != ~0u && r >= static_cast<int>(last_first) && vi < last_view.size() && last_view[vi] != b &&
                               last_frame != s.frame;
                ImVec2 bmn(x - cw * 0.3f, p.y + dp(1)), bmx(x + cw * 2.3f, p.y + row_h - dp(1));
                if (in_sel) dl->AddRectFilled(bmn, bmx, col(g_pal.accent, 0.35f), dp(3));
                else if (changed) dl->AddRectFilled(bmn, bmx, col(g_pal.warning, 0.25f), dp(3));
                char hx[4];
                std::snprintf(hx, sizeof hx, "%02X", b);
                dl->AddText(mf, font_px(mf), ImVec2(x, p.y + dp(3)), col(b ? g_pal.text : g_pal.text_faint), hx);
                // Click a byte to select it.
                ImGui::SetCursorScreenPos(bmn);
                ImGui::PushID(static_cast<int>(off));
                if (ImGui::InvisibleButton("##b", ImVec2(bmx.x - bmn.x, bmx.y - bmn.y))) {
                    dbg_.edit_addr = 0x80000000u | off;
                    std::snprintf(dbg_.edit_addr_text, sizeof dbg_.edit_addr_text, "%08X", dbg_.edit_addr);
                }
                ImGui::PopID();
            }
            // ASCII
            float ax = x0 + 16 * cw * 3 + 4 * cw + cw;
            char ascii[17];
            for (int i = 0; i < 16; ++i) {
                unsigned char c = s.rdram[base + i];
                ascii[i] = (c >= 32 && c < 127) ? static_cast<char>(c) : '.';
            }
            ascii[16] = 0;
            dl->AddText(mf, font_px(mf), ImVec2(ax, p.y + dp(3)), col(g_pal.text_faint), ascii);
            ImGui::SetCursorScreenPos(p);
            ImGui::Dummy(ImVec2(ax + cw * 16 - p.x, row_h));
        }
    }
    if (first_row != ~0u) {
        if (last_frame != s.frame || last_first != first_row) {
            last_view = std::move(view);
            last_first = first_row;
            last_frame = s.frame;
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// ---------------------------------------------------------------------------
// Freeze List
// ---------------------------------------------------------------------------

void App::tool_freeze_list() {
    const float w = ImGui::GetContentRegionAvail().x;
    const float gap = dp(8);

    float type_w = dp(80), add_w = dp(90);
    float rest = w - type_w - add_w - gap * 4;
    ImGui::SetNextItemWidth(rest * 0.34f);
    ImGui::InputTextWithHint("##fname", "Name", dbg_.freeze_name, sizeof dbg_.freeze_name);
    ImGui::SameLine(0, gap);
    ImGui::SetNextItemWidth(rest * 0.36f);
    ImGui::InputTextWithHint("##faddr", "Address (hex)", dbg_.freeze_addr, sizeof dbg_.freeze_addr, ImGuiInputTextFlags_CharsHexadecimal);
    ImGui::SameLine(0, gap);
    type_combo("ftype", &dbg_.freeze_type, type_w);
    ImGui::SameLine(0, gap);
    ImGui::SetNextItemWidth(rest * 0.30f);
    bool enter = ImGui::InputTextWithHint("##fval", "Value", dbg_.freeze_value, sizeof dbg_.freeze_value, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine(0, gap);
    if (button("Add", Icon::Plus, ButtonKind::Primary, add_w, true, dp(32)) || enter) {
        std::uint32_t a;
        std::vector<std::uint8_t> bytes;
        MemType t = static_cast<MemType>(dbg_.freeze_type);
        if (!parse_address(dbg_.freeze_addr, a)) {
            toast("Enter a hexadecimal address", ToastKind::Warning);
        } else if (dbg_.freeze_value[0] && !mem_parse(dbg_.freeze_value, t, bytes)) {
            toast("Enter a valid " + std::string(kMemTypeNames[dbg_.freeze_type]) + " value", ToastKind::Warning);
        } else {
            add_freeze(dbg_.freeze_name, a, t, dbg_.freeze_value);
            dbg_.freeze_name[0] = dbg_.freeze_addr[0] = dbg_.freeze_value[0] = 0;
        }
    }
    ImGui::Dummy(dp(0, 4));

    if (dbg_.freezes.empty()) {
        ImGui::PushFont(g_fonts.small);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
        ImGui::TextColored(g_pal.text_dim,
                           "Frozen values are written into memory before and after every frame, so the game can't change them. "
                           "Leave Value empty to freeze the current value.");
        ImGui::PopTextWrapPos();
        ImGui::PopFont();
        return;
    }

    int active = 0;
    for (const auto& f : dbg_.freezes) active += f.enabled ? 1 : 0;
    ImGui::PushFont(g_fonts.small);
    ImGui::TextColored(active ? g_pal.accent_hover : g_pal.text_faint, "%d of %zu freezes active", active, dbg_.freezes.size());
    ImGui::PopFont();

    int remove = -1;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, dp(8, 4));
    if (ImGui::BeginTable("##freezes", 7, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV |
                                              ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("On", ImGuiTableColumnFlags_WidthFixed, dp(48));
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableSetupColumn("Address", ImGuiTableColumnFlags_WidthFixed, dp(90));
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, dp(76));
        ImGui::TableSetupColumn("Frozen value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("In memory", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, dp(34));
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::PushStyleColor(ImGuiCol_Text, g_pal.text_faint);
        ImGui::TableHeadersRow();
        ImGui::PopStyleColor();
        ImGui::PopFont();
        for (size_t i = 0; i < dbg_.freezes.size(); ++i) {
            Freeze& f = dbg_.freezes[i];
            ImGui::TableNextRow(0, dp(30));
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextColumn();
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(2));
            if (toggle("on", &f.enabled)) {
                dbg_.freezes_dirty = true;
                save_debug_lists();
            }
            ImGui::TableNextColumn();
            char name[64];
            std::snprintf(name, sizeof name, "%s", f.name.c_str());
            ImGui::SetNextItemWidth(-1);
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
            if (ImGui::InputText("##name", name, sizeof name)) f.name = name;
            if (ImGui::IsItemDeactivatedAfterEdit()) save_debug_lists();
            ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            ImGui::PushFont(g_fonts.mono_small);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(5));
            ImGui::TextColored(g_pal.text_dim, "%08X", f.addr);
            ImGui::PopFont();
            ImGui::TableNextColumn();
            int t = static_cast<int>(f.type);
            if (type_combo("type", &t, dp(70))) {
                f.type = static_cast<MemType>(t);
                dbg_.freezes_dirty = true;
                save_debug_lists();
            }
            ImGui::TableNextColumn();
            char val[32];
            std::snprintf(val, sizeof val, "%s", f.value.c_str());
            std::vector<std::uint8_t> tmp;
            bool ok = mem_parse(f.value, f.type, tmp);
            ImGui::SetNextItemWidth(-1);
            ImGui::PushStyleColor(ImGuiCol_Text, ok ? g_pal.text : g_pal.danger);
            ImGui::PushFont(g_fonts.mono);
            if (ImGui::InputText("##val", val, sizeof val)) {
                f.value = val;
                dbg_.freezes_dirty = true;
            }
            if (ImGui::IsItemDeactivatedAfterEdit()) save_debug_lists();
            ImGui::PopFont();
            ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            ImGui::PushFont(g_fonts.mono_small);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(5));
            if (dbg_.snap && !dbg_.snap->rdram.empty()) ImGui::TextColored(f.enabled ? g_pal.accent_hover : g_pal.text_faint, "%s",
                                              mem_format(mem_read_raw(*dbg_.snap, f.addr, f.type), f.type).c_str());
            else ImGui::TextColored(g_pal.text_faint, "\xE2\x80\x94");
            ImGui::PopFont();
            ImGui::TableNextColumn();
            if (icon_button("x", Icon::Trash, dp(26), "Remove", false, true, ButtonKind::Danger)) remove = static_cast<int>(i);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
    if (remove >= 0) {
        dbg_.freezes.erase(dbg_.freezes.begin() + remove);
        dbg_.freezes_dirty = true;
        save_debug_lists();
    }
}

} // namespace ui
