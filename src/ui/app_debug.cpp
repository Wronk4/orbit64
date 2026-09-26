// DEBUG / MEMORY tools: shared infrastructure, menu, Registers, Frame Control
// and Screen Resolution. Every tool here is game-independent.

#include "app.hpp"
#include "platform.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace ui {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Tool registry
// ---------------------------------------------------------------------------

const DebugToolInfo& debug_tool_info(DebugTool t) {
    static const DebugToolInfo kInfo[kDebugToolCount] = {
        {"Object Viewer", "Objects drawn this frame, with position, rotation and scale", Icon::Layers, false},
        {"3D Object Inspector", "Orbitable 3D view of the selected object", Icon::Cube, false},
        {"Player Viewer", "Live position, rotation and velocity of the player", Icon::User, false},
        {"Registers", "MIPS R4300i, COP0 and FPU state", Icon::Cpu, false},
        {"Frame Control", "Pause, frame advance, FPS limit and turbo", Icon::Step, false},
        {"Screen Resolution", "Internal rendering and output resolution", Icon::Monitor, false},
        {"Memory Search", "Find values in RDRAM with successive scans", Icon::Search, true},
        {"RAM Watch", "Named addresses with live values", Icon::Eye, true},
        {"Memory Editor", "Read and write memory at any address", Icon::Pencil, true},
        {"Freeze List", "Values continuously forced into memory", Icon::Snowflake, true},
    };
    return kInfo[static_cast<int>(t)];
}

// ---------------------------------------------------------------------------
// Memory value helpers
// ---------------------------------------------------------------------------

const char* const kMemTypeNames[kMemTypeCount] = {"U8", "U16", "U32", "S8", "S16", "S32", "F32"};

int mem_type_size(MemType t) {
    switch (t) {
        case MemType::U8: case MemType::S8: return 1;
        case MemType::U16: case MemType::S16: return 2;
        default: return 4;
    }
}

std::uint32_t mem_read_raw(const DebugSnapshot& s, std::uint32_t addr, MemType t) {
    switch (mem_type_size(t)) {
        case 1: return s.u8(addr);
        case 2: return s.u16(addr);
        default: return s.u32(addr);
    }
}

double mem_as_double(std::uint32_t raw, MemType t) {
    switch (t) {
        case MemType::U8: return static_cast<std::uint8_t>(raw);
        case MemType::U16: return static_cast<std::uint16_t>(raw);
        case MemType::U32: return raw;
        case MemType::S8: return static_cast<std::int8_t>(raw);
        case MemType::S16: return static_cast<std::int16_t>(raw);
        case MemType::S32: return static_cast<std::int32_t>(raw);
        case MemType::F32: { float f; std::memcpy(&f, &raw, 4); return f; }
        default: return 0;
    }
}

std::string mem_format(std::uint32_t raw, MemType t) {
    char buf[48];
    if (t == MemType::F32) {
        float f;
        std::memcpy(&f, &raw, 4);
        std::snprintf(buf, sizeof buf, "%.4f", f);
    } else if (t == MemType::S8 || t == MemType::S16 || t == MemType::S32) {
        std::snprintf(buf, sizeof buf, "%lld", static_cast<long long>(mem_as_double(raw, t)));
    } else {
        std::snprintf(buf, sizeof buf, "%llu", static_cast<unsigned long long>(mem_as_double(raw, t)));
    }
    return buf;
}

bool mem_parse(const std::string& text, MemType t, std::vector<std::uint8_t>& out) {
    std::string s = text;
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }), s.end());
    if (s.empty()) return false;
    std::uint32_t raw = 0;
    char* end = nullptr;
    if (t == MemType::F32) {
        float f = std::strtof(s.c_str(), &end);
        if (end == s.c_str() || *end) return false;
        std::memcpy(&raw, &f, 4);
    } else {
        bool hex = s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X');
        long long v = std::strtoll(s.c_str(), &end, hex ? 16 : 10);
        if (end == s.c_str() || *end) return false;
        raw = static_cast<std::uint32_t>(v);
    }
    int n = mem_type_size(t);
    out.resize(n);
    for (int i = 0; i < n; ++i) out[i] = static_cast<std::uint8_t>(raw >> (8 * (n - 1 - i)));
    return true;
}

bool parse_address(const std::string& text, std::uint32_t& out) {
    std::string s = text;
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }), s.end());
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);
    if (s.empty() || s.size() > 8) return false;
    char* end = nullptr;
    unsigned long v = std::strtoul(s.c_str(), &end, 16);
    if (*end) return false;
    // KSEG0/KSEG1 and physical offsets all map to the same RDRAM byte.
    out = 0x80000000u | (static_cast<std::uint32_t>(v) & 0x1FFFFFFFu);
    return true;
}

// ---------------------------------------------------------------------------
// Menu
// ---------------------------------------------------------------------------

static bool g_focus_request[kDebugToolCount] = {};

void App::open_tool(DebugTool t) {
    dbg_.open[static_cast<int>(t)] = true;
    g_focus_request[static_cast<int>(t)] = true; // bring an already open tool to the front
}

void App::draw_debug_menu(bool descriptions) {
    auto header = [](const char* text) {
        ImGui::Dummy(dp(0, 2));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + dp(6));
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::TextColored(g_pal.text_faint, "%s", text);
        ImGui::PopFont();
    };
    auto item = [&](DebugTool t) {
        const DebugToolInfo& info = debug_tool_info(t);
        const int i = static_cast<int>(t);
        const bool open = dbg_.open[i];
        const float h = descriptions ? dp(46) : dp(30);
        const float w = std::max(ImGui::GetContentRegionAvail().x, descriptions ? dp(330) : dp(230));
        ImGui::PushID(i);
        ImVec2 p = ImGui::GetCursorScreenPos();
        bool clicked = ImGui::Selectable("##tool", false, ImGuiSelectableFlags_None, ImVec2(w, h));
        ImGui::PopID();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 ic(p.x + dp(14), p.y + h * 0.5f);
        if (descriptions) {
            dl->AddRectFilled(ImVec2(ic.x - dp(11), ic.y - dp(11)), ImVec2(ic.x + dp(11), ic.y + dp(11)),
                              col(open ? g_pal.accent_soft : g_pal.bg4), dp(6));
            ic.x += dp(2);
        }
        draw_icon(dl, info.icon, ic, dp(14), col(open ? g_pal.accent_hover : g_pal.text_dim));
        float tx = p.x + (descriptions ? dp(40) : dp(32));
        if (descriptions) {
            dl->AddText(g_fonts.body_bold, font_px(g_fonts.body_bold), ImVec2(tx, p.y + dp(6)), col(g_pal.text), info.title);
            dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(tx, p.y + dp(25)), col(g_pal.text_faint), info.description);
        } else {
            dl->AddText(g_fonts.body, font_px(g_fonts.body), ImVec2(tx, p.y + (h - font_px(g_fonts.body)) * 0.5f), col(g_pal.text), info.title);
        }
        if (open) draw_icon(dl, Icon::Check, ImVec2(p.x + w - dp(14), p.y + h * 0.5f), dp(13), col(g_pal.accent));
        if (clicked) {
            if (open) dbg_.open[i] = false;
            else open_tool(t);
        }
    };
    header("DEBUG");
    for (int i = 0; i < kDebugToolCount; ++i)
        if (!debug_tool_info(static_cast<DebugTool>(i)).memory) item(static_cast<DebugTool>(i));
    ImGui::Separator();
    header("MEMORY");
    for (int i = 0; i < kDebugToolCount; ++i)
        if (debug_tool_info(static_cast<DebugTool>(i)).memory) item(static_cast<DebugTool>(i));
    ImGui::Separator();
    bool any = std::any_of(std::begin(dbg_.open), std::end(dbg_.open), [](bool b) { return b; });
    if (ImGui::MenuItem("Close All Tools", nullptr, false, any))
        for (bool& b : dbg_.open) b = false;
}

// ---------------------------------------------------------------------------
// Tool window frame
// ---------------------------------------------------------------------------

static bool g_tool_begun = false;

bool App::begin_tool(DebugTool t, ImVec2 default_size) {
    const int i = static_cast<int>(t);
    g_tool_begun = false;
    if (!dbg_.open[i]) return false;
    const DebugToolInfo& info = debug_tool_info(t);
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    // Cascade first-time positions so newly opened tools don't stack exactly.
    ImVec2 pos(vp->WorkPos.x + dp(60) + i * dp(30), vp->WorkPos.y + dp(100) + i * dp(26));
    ImGui::SetNextWindowPos(pos, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(default_size, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(dp(300, 180), ImVec2(FLT_MAX, FLT_MAX));
    if (g_focus_request[i]) {
        ImGui::SetNextWindowFocus();
        g_focus_request[i] = false;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(14, 12));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, dp(10));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, dp(12, 9)); // title bar height
    ImGui::PushStyleColor(ImGuiCol_WindowBg, g_pal.bg1);
    ImGui::PushStyleColor(ImGuiCol_TitleBg, g_pal.bg2);
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, g_pal.bg3);
    ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, g_pal.bg2);
    ImGui::PushStyleColor(ImGuiCol_Border, g_pal.border_strong);
    char label[96];
    std::snprintf(label, sizeof label, "          %s###debugtool%d", info.title, i);
    bool visible = ImGui::Begin(label, &dbg_.open[i], ImGuiWindowFlags_NoCollapse);
    g_tool_begun = true;
    ImGui::PopStyleColor(5);
    ImGui::PopStyleVar(4);
    if (visible) {
        ImGuiWindow* w = ImGui::GetCurrentWindow();
        ImRect tb = w->TitleBarRect();
        w->DrawList->PushClipRect(tb.Min, tb.Max, false);
        draw_icon(w->DrawList, info.icon, ImVec2(tb.Min.x + dp(24), (tb.Min.y + tb.Max.y) * 0.5f), dp(15),
                  col(ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) ? g_pal.accent_hover : g_pal.text_dim));
        w->DrawList->PopClipRect();
        if (!core_.loaded() && t != DebugTool::Resolution && t != DebugTool::FrameControl) {
            // Empty state shared by all tools that need a running game.
            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImVec2 c(p.x + avail.x * 0.5f, p.y + avail.y * 0.42f);
            dl->AddCircleFilled(c, dp(26), col(g_pal.bg3), 32);
            draw_icon(dl, info.icon, c, dp(22), col(g_pal.text_faint));
            const char* msg = "Start a game to use this tool";
            ImVec2 ts = g_fonts.body->CalcTextSizeA(font_px(g_fonts.body), FLT_MAX, 0, msg);
            dl->AddText(g_fonts.body, font_px(g_fonts.body), ImVec2(c.x - ts.x * 0.5f, c.y + dp(38)), col(g_pal.text_dim), msg);
            ImGui::Dummy(avail);
            return false;
        }
    }
    return visible;
}

void App::end_tool() {
    if (g_tool_begun) ImGui::End();
    g_tool_begun = false;
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------

void App::debug_update() {
    bool any = std::any_of(std::begin(dbg_.open), std::end(dbg_.open), [](bool b) { return b; });
    auto is_open = [&](DebugTool t) { return dbg_.open[static_cast<int>(t)]; };
    const bool player_objects = is_open(DebugTool::PlayerViewer) && dbg_.player_source == 0;
    const bool geometry = is_open(DebugTool::ObjectViewer) || is_open(DebugTool::ObjectInspector) || player_objects;
    const bool ram = is_open(DebugTool::MemorySearch) || is_open(DebugTool::RamWatch) || is_open(DebugTool::MemoryEditor) ||
                     is_open(DebugTool::FreezeList) || (is_open(DebugTool::PlayerViewer) && dbg_.player_source == 1);
    if (any != dbg_.capture_active || ram != dbg_.capture_ram || geometry != dbg_.capture_geometry) {
        core_.set_debug_capture(any, ram, geometry);
        dbg_.capture_active = any;
        dbg_.capture_ram = ram;
        dbg_.capture_geometry = geometry;
    }
    if (!core_.loaded()) {
        dbg_.snap.reset();
        return;
    }

    // Per-game watch / freeze lists.
    std::string game = current_rom_.game_code + "-" + std::to_string(current_rom_.crc1);
    if (game != dbg_.lists_game) {
        dbg_.lists_game = game;
        load_debug_lists();
    }
    if (dbg_.freezes_dirty) {
        std::vector<MemoryFreeze> list;
        for (const auto& f : dbg_.freezes) {
            MemoryFreeze m;
            m.addr = f.addr;
            if (f.enabled && mem_parse(f.value, f.type, m.bytes)) list.push_back(std::move(m));
        }
        core_.set_freezes(std::move(list));
        dbg_.freezes_dirty = false;
    }

    const bool inspecting = is_open(DebugTool::ObjectInspector) && dbg_.selected.valid && dbg_.show_textures;
    core_.set_texture_target(inspecting ? (dbg_.selected.vtx_addr & 0x1FFFFFFFu) : 0);

    if (!any) return;
    auto s = core_.debug_snapshot();
    if (s && s != dbg_.snap) {
        if (dbg_.snap && dbg_.snap->frame != s->frame) {
            dbg_.prev_cpu = dbg_.snap->cpu;
            dbg_.cpu_frame = dbg_.snap->frame;
        }
        dbg_.snap = std::move(s);
        compute_scene_objects();
    }
}

void App::draw_debug_windows() {
    if (begin_tool(DebugTool::ObjectViewer, dp(720, 520))) tool_object_viewer();
    end_tool();
    if (begin_tool(DebugTool::ObjectInspector, dp(640, 560))) tool_object_inspector();
    end_tool();
    if (begin_tool(DebugTool::PlayerViewer, dp(420, 560))) tool_player_viewer();
    end_tool();
    if (begin_tool(DebugTool::Registers, dp(560, 600))) tool_registers();
    end_tool();
    if (begin_tool(DebugTool::FrameControl, dp(460, 520))) tool_frame_control();
    end_tool();
    if (begin_tool(DebugTool::Resolution, dp(500, 700))) tool_resolution();
    end_tool();
    if (begin_tool(DebugTool::MemorySearch, dp(560, 600))) tool_memory_search();
    end_tool();
    if (begin_tool(DebugTool::RamWatch, dp(600, 420))) tool_ram_watch();
    end_tool();
    if (begin_tool(DebugTool::MemoryEditor, dp(660, 560))) tool_memory_editor();
    end_tool();
    if (begin_tool(DebugTool::FreezeList, dp(620, 420))) tool_freeze_list();
    end_tool();
}

// ---------------------------------------------------------------------------
// Shared little layout helpers for tools
// ---------------------------------------------------------------------------

namespace {

void tool_section(const char* title) {
    ImGui::Dummy(dp(0, 4));
    ImGui::PushFont(g_fonts.small_bold);
    ImGui::TextColored(g_pal.text_faint, "%s", title);
    ImGui::PopFont();
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddLine(ImVec2(p.x, p.y), ImVec2(p.x + ImGui::GetContentRegionAvail().x, p.y), col(g_pal.border));
    ImGui::Dummy(dp(0, 4));
}

// Compact stat card: small caption over a large monospace value.
void stat_card(const char* caption, const std::string& value, float width, const ImVec4& value_col, const char* sub = nullptr) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    float h = sub ? dp(66) : dp(54);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + width, p.y + h), col(g_pal.bg2), dp(8));
    dl->AddRect(p, ImVec2(p.x + width, p.y + h), col(g_pal.border), dp(8));
    dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(p.x + dp(12), p.y + dp(8)), col(g_pal.text_faint), caption);
    text_ellipsis(dl, g_fonts.mono, ImVec2(p.x + dp(12), p.y + dp(27)), width - dp(20), col(value_col), value.c_str());
    if (sub) text_ellipsis(dl, g_fonts.small, ImVec2(p.x + dp(12), p.y + dp(46)), width - dp(20), col(g_pal.text_faint), sub);
    ImGui::Dummy(ImVec2(width, h));
}

std::string hex32(std::uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof b, "%08X", v);
    return b;
}

} // namespace

// ---------------------------------------------------------------------------
// Registers
// ---------------------------------------------------------------------------

void App::tool_registers() {
    if (!dbg_.snap) {
        spinner(dp(10), dp(2), col(g_pal.accent));
        return;
    }
    const CpuSnapshot& c = dbg_.snap->cpu;
    const CpuSnapshot& p = dbg_.prev_cpu;
    const float w = ImGui::GetContentRegionAvail().x;
    const float gap = dp(8);
    const float cw = (w - gap * 3) / 4;

    char buf[64];
    std::snprintf(buf, sizeof buf, "%08X", static_cast<std::uint32_t>(c.pc));
    std::snprintf(buf + 20, 40, "op %08X%s", c.instr, c.delay_slot ? " \xC2\xB7 delay" : "");
    stat_card("PC", buf, cw, g_pal.accent_hover, buf + 20);
    ImGui::SameLine(0, gap);
    auto hilo = [&](const char* name, std::uint64_t v, std::uint64_t pv) {
        char b[24];
        bool sext = static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(v))) == v;
        if (sext) std::snprintf(b, sizeof b, "%08X", static_cast<std::uint32_t>(v));
        else std::snprintf(b, sizeof b, "%016llX", static_cast<unsigned long long>(v));
        stat_card(name, b, cw, dbg_.cpu_frame && v != pv ? g_pal.warning : g_pal.text, sext ? "sign-extended 32-bit" : "64-bit");
    };
    hilo("HI", c.hi, p.hi);
    ImGui::SameLine(0, gap);
    hilo("LO", c.lo, p.lo);
    ImGui::SameLine(0, gap);
    std::snprintf(buf, sizeof buf, "%llu", static_cast<unsigned long long>(dbg_.snap->frame));
    stat_card("FRAME", buf, cw, g_pal.text, core_.state() == RunState::Paused ? "paused" : "captured at frame end");

    ImGui::Dummy(dp(0, 6));
    const char* tabs[] = {"General Purpose", "COP0", "FPU"};
    segmented("regtabs", tabs, 3, &dbg_.reg_tab, std::min(w, dp(420)));
    ImGui::Dummy(dp(0, 4));

    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, dp(8, 4));
    ImGui::BeginChild("##regs", ImVec2(0, 0));
    auto value64 = [&](std::uint64_t v, bool changed) {
        // Upper half dimmed when it is only the sign extension of the lower half.
        bool sext = static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(v))) == v;
        char hi[12], lo[12];
        std::snprintf(hi, sizeof hi, "%08X", static_cast<std::uint32_t>(v >> 32));
        std::snprintf(lo, sizeof lo, "%08X", static_cast<std::uint32_t>(v));
        ImGui::PushFont(g_fonts.mono);
        ImGui::TextColored(changed ? g_pal.warning : (sext ? g_pal.text_faint : g_pal.text), "%s", hi);
        ImGui::SameLine(0, 0);
        ImGui::TextColored(changed ? g_pal.warning : g_pal.text, "%s", lo);
        ImGui::PopFont();
    };
    const bool fresh = dbg_.cpu_frame != 0; // have a previous frame to compare against

    if (dbg_.reg_tab == 0) {
        static const char* kNames[32] = {"zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2",
                                         "t3", "t4", "t5", "t6", "t7", "s0", "s1", "s2", "s3", "s4", "s5",
                                         "s6", "s7", "t8", "t9", "k0", "k1", "gp", "sp", "s8", "ra"};
        int cols = w > dp(520) ? 2 : 1;
        if (ImGui::BeginTable("##gpr", cols * 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
            int rows = 32 / cols;
            for (int r = 0; r < rows; ++r) {
                ImGui::TableNextRow();
                for (int k = 0; k < cols; ++k) {
                    int i = r + k * rows;
                    ImGui::TableNextColumn();
                    ImGui::PushFont(g_fonts.mono_small);
                    ImGui::TextColored(g_pal.text_faint, "r%-2d", i);
                    ImGui::PopFont();
                    ImGui::TableNextColumn();
                    ImGui::PushFont(g_fonts.body_bold);
                    ImGui::TextColored(g_pal.text_dim, "%s", kNames[i]);
                    ImGui::PopFont();
                    ImGui::TableNextColumn();
                    value64(c.gpr[i], fresh && c.gpr[i] != p.gpr[i]);
                }
            }
            ImGui::EndTable();
        }
    } else if (dbg_.reg_tab == 1) {
        struct R { int idx; const char* name; const char* desc; };
        static const R kCop0[] = {
            {12, "Status", "Interrupt mask, mode and coprocessor enables"}, {13, "Cause", "Pending interrupts and exception code"},
            {14, "EPC", "Exception return address"}, {8, "BadVAddr", "Last faulting virtual address"},
            {9, "Count", "Timer (increments at half the CPU clock)"}, {11, "Compare", "Timer interrupt threshold"},
            {10, "EntryHi", "TLB entry high word"}, {2, "EntryLo0", "TLB entry low word (even page)"},
            {3, "EntryLo1", "TLB entry low word (odd page)"}, {0, "Index", "TLB index"}, {1, "Random", "TLB random index"},
            {5, "PageMask", "TLB page size"}, {6, "Wired", "TLB wired entries"}, {4, "Context", "TLB miss context"},
            {15, "PRId", "Processor revision"}, {16, "Config", "Cache and endianness configuration"},
            {17, "LLAddr", "Load-linked address"}, {20, "XContext", "64-bit TLB miss context"}, {30, "ErrorEPC", "Error return address"},
        };
        std::uint32_t st = static_cast<std::uint32_t>(c.cp0[12]), cause = static_cast<std::uint32_t>(c.cp0[13]);
        std::snprintf(buf, sizeof buf, "IE=%u EXL=%u ERL=%u KSU=%u FR=%u IM=%02X", st & 1, (st >> 1) & 1, (st >> 2) & 1,
                      (st >> 3) & 3, (st >> 26) & 1, (st >> 8) & 0xFF);
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_dim, "Status: %s   \xC2\xB7   Cause: ExcCode=%u IP=%02X", buf, (cause >> 2) & 0x1F, (cause >> 8) & 0xFF);
        ImGui::PopFont();
        if (ImGui::BeginTable("##cop0", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("reg", ImGuiTableColumnFlags_WidthFixed, dp(90));
            ImGui::TableSetupColumn("val", ImGuiTableColumnFlags_WidthFixed, dp(150));
            ImGui::TableSetupColumn("desc", ImGuiTableColumnFlags_WidthStretch);
            for (const auto& r : kCop0) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushFont(g_fonts.body_bold);
                ImGui::TextColored(g_pal.text_dim, "%s", r.name);
                ImGui::PopFont();
                ImGui::TableNextColumn();
                value64(c.cp0[r.idx], fresh && c.cp0[r.idx] != p.cp0[r.idx] && r.idx != 9 && r.idx != 1);
                ImGui::TableNextColumn();
                ImGui::PushFont(g_fonts.small);
                ImGui::TextColored(g_pal.text_faint, "%s", r.desc);
                ImGui::PopFont();
            }
            ImGui::EndTable();
        }
    } else {
        static const char* kRound[] = {"nearest", "toward zero", "toward +inf", "toward -inf"};
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_dim, "FCSR %08X   \xC2\xB7   rounding %s   \xC2\xB7   condition %u   \xC2\xB7   FR=%u", c.fcsr,
                           kRound[c.fcsr & 3], (c.fcsr >> 23) & 1, static_cast<unsigned>((c.cp0[12] >> 26) & 1));
        ImGui::PopFont();
        if (ImGui::BeginTable("##fpr", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("reg", ImGuiTableColumnFlags_WidthFixed, dp(44));
            ImGui::TableSetupColumn("raw", ImGuiTableColumnFlags_WidthFixed, dp(150));
            ImGui::TableSetupColumn("f32", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("f64", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            for (int i = 0; i < 32; ++i) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushFont(g_fonts.body_bold);
                ImGui::TextColored(g_pal.text_dim, "f%d", i);
                ImGui::PopFont();
                ImGui::TableNextColumn();
                value64(c.fpr[i], fresh && c.fpr[i] != p.fpr[i]);
                std::uint32_t lo32 = static_cast<std::uint32_t>(c.fpr[i]);
                float f;
                double d;
                std::memcpy(&f, &lo32, 4);
                std::memcpy(&d, &c.fpr[i], 8);
                ImGui::PushFont(g_fonts.mono_small);
                ImGui::TableNextColumn();
                ImGui::TextColored(g_pal.text, "%.6g", f);
                ImGui::TableNextColumn();
                ImGui::TextColored(g_pal.text_dim, "%.6g", d);
                ImGui::PopFont();
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
}

// ---------------------------------------------------------------------------
// Frame Control
// ---------------------------------------------------------------------------

void App::tool_frame_control() {
    const float w = ImGui::GetContentRegionAvail().x;
    const bool loaded = core_.loaded();
    const RunState st = core_.state();
    CoreStats s = core_.stats();

    // Status cards
    const float gap = dp(8), cw = (w - gap * 2) / 3;
    const char* st_label = st == RunState::Running ? "Running" : st == RunState::Paused ? "Paused" : "Stopped";
    ImVec4 st_col = st == RunState::Running ? g_pal.success : st == RunState::Paused ? g_pal.warning : g_pal.text_faint;
    stat_card("STATE", st_label, cw, st_col);
    ImGui::SameLine(0, gap);
    stat_card("FRAME", std::to_string(loaded ? s.frame : 0), cw, g_pal.text);
    ImGui::SameLine(0, gap);
    char fps[32];
    std::snprintf(fps, sizeof fps, "%.1f", loaded ? s.fps : 0.0f);
    char speed[32];
    std::snprintf(speed, sizeof speed, "%.0f%% of %d Hz", loaded ? s.speed_pct : 0.0f, s.vi_hz);
    stat_card("FPS", fps, cw, st == RunState::Running ? g_pal.success : g_pal.text_dim, speed);

    // Execution
    tool_section("EXECUTION");
    float bw = (w - gap) / 2;
    if (button(st == RunState::Running ? "Pause" : "Resume", st == RunState::Running ? Icon::Pause : Icon::Play,
               ButtonKind::Primary, bw, loaded, dp(38)))
        toggle_pause();
    ImGui::SameLine(0, gap);
    if (button("Frame Advance", Icon::Step, ButtonKind::Subtle, bw, loaded && st == RunState::Paused, dp(38))) core_.frame_advance();
    ImGui::PushFont(g_fonts.small);
    ImGui::TextColored(g_pal.text_faint, st == RunState::Paused ? "Frame Advance runs exactly one frame, then pauses again."
                                                                : "Pause first to step through single frames.");
    ImGui::PopFont();

    // Frame rate limit
    tool_section("FRAME RATE");
    int mode = !settings_.limit_speed ? 2 : settings_.fps_limit > 0 ? 1 : 0;
    char def_label[40];
    std::snprintf(def_label, sizeof def_label, "N64 (%d Hz)", s.vi_hz);
    const char* modes[] = {def_label, "Lock", "Unlock"};
    if (segmented("fpsmode", modes, 3, &mode, w)) {
        settings_.limit_speed = mode != 2;
        if (mode == 0) settings_.fps_limit = 0;
        if (mode == 1 && settings_.fps_limit <= 0) settings_.fps_limit = 60;
    }
    ImGui::Dummy(dp(0, 2));
    if (mode == 1) {
        static const int kPresets[] = {30, 60, 120, 144, 165, 240};
        float chip = (w - gap * 5) / 6;
        for (int i = 0; i < 6; ++i) {
            if (i) ImGui::SameLine(0, gap);
            char l[16];
            std::snprintf(l, sizeof l, "%d", kPresets[i]);
            bool on = settings_.fps_limit == kPresets[i];
            if (button(l, Icon::None, on ? ButtonKind::Primary : ButtonKind::Subtle, chip, true, dp(30))) settings_.fps_limit = kPresets[i];
        }
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_dim, "Custom limit");
        ImGui::PopFont();
        ImGui::SameLine(dp(110));
        ImGui::SetNextItemWidth(dp(120));
        int v = settings_.fps_limit;
        if (ImGui::InputInt("##fpscustom", &v, 0, 0)) settings_.fps_limit = std::clamp(v, 1, 1000);
        ImGui::SameLine();
        ImGui::TextColored(g_pal.text_faint, "FPS");
    } else {
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_faint, mode == 0 ? "Runs at the console's video refresh rate (NTSC 60 Hz / PAL 50 Hz)."
                                                       : "No limit: runs as fast as this computer allows.");
        ImGui::PopFont();
    }
    if (button("Reset to N64 Default", Icon::Refresh, ButtonKind::Ghost, 0, mode != 0)) {
        settings_.limit_speed = true;
        settings_.fps_limit = 0;
    }

    // Turbo
    tool_section("TURBO");
    bool turbo = core_.turbo();
    if (toggle("turbo", &turbo)) core_.set_turbo(turbo);
    ImGui::SameLine(0, dp(10));
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(1));
    ImGui::TextColored(turbo ? g_pal.warning : g_pal.text_dim, turbo ? "Turbo on" : "Turbo off");
    const char* mult[] = {"2\xC3\x97", "3\xC3\x97", "4\xC3\x97", "8\xC3\x97", "Max"};
    const int vals[] = {2, 3, 4, 8, 0};
    int cur = 1;
    for (int i = 0; i < 5; ++i) if (vals[i] == settings_.ff_speed) cur = i;
    if (segmented("turbomult", mult, 5, &cur, w)) settings_.ff_speed = vals[cur];
    ImGui::PushFont(g_fonts.small);
    ImGui::TextColored(g_pal.text_faint, "Multiplies the current frame rate. Holding Tab gives the same boost temporarily.");
    ImGui::PopFont();
}

// ---------------------------------------------------------------------------
// Screen Resolution
// ---------------------------------------------------------------------------

void App::output_resolution(int& w, int& h) const {
    const int fw = frame_native_w(), fh = frame_native_h();
    if (settings_.render_scale == 0) {
        w = std::clamp(settings_.render_w, 64, 7680);
        h = std::clamp(settings_.render_h, 64, 4320);
    } else {
        int k = std::clamp(settings_.render_scale, 1, 8);
        w = fw * k;
        h = fh * k;
    }
    // An output smaller than what the RDP rendered isn't resampled first;
    // the frame goes to the window as it is.
    if (game_tex_w_ > 0 && w <= game_tex_w_ && h <= game_tex_h_) {
        w = game_tex_w_;
        h = game_tex_h_;
    }
}

void App::update_scaled_texture() {
    int w = 0, h = 0;
    output_resolution(w, h);
    bool native = !game_tex_ || !has_frame_ || (w == game_tex_w_ && h == game_tex_h_);
    if (native) {
        if (scaled_tex_) {
            SDL_DestroyTexture(scaled_tex_);
            scaled_tex_ = nullptr;
            game_tex_filter_ = -1; // restore the user's filter on the frame texture
        }
        return;
    }
    if (!scaled_tex_ || w != scaled_w_ || h != scaled_h_) {
        if (scaled_tex_) SDL_DestroyTexture(scaled_tex_);
        scaled_tex_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_TARGET, w, h);
        scaled_w_ = w;
        scaled_h_ = h;
        if (!scaled_tex_) return;
    }
    // Integer scales use a nearest-neighbour prescale (crisp pixels); custom
    // sizes are resampled bilinearly. The output texture then uses the
    // user's filter when drawn to the window.
    bool integer = settings_.render_scale != 0;
    SDL_SetTextureScaleMode(game_tex_, integer ? SDL_SCALEMODE_NEAREST : SDL_SCALEMODE_LINEAR);
    game_tex_filter_ = -1;
    SDL_SetTextureScaleMode(scaled_tex_, settings_.filter == 0 ? SDL_SCALEMODE_NEAREST : SDL_SCALEMODE_LINEAR);
    SDL_SetRenderScale(renderer_, 1.0f, 1.0f);
    SDL_SetRenderTarget(renderer_, scaled_tex_);
    SDL_RenderTexture(renderer_, game_tex_, nullptr, nullptr);
    SDL_SetRenderTarget(renderer_, nullptr);
}

void App::tool_resolution() {
    const float w = ImGui::GetContentRegionAvail().x;
    const float gap = dp(8);
    int ow = 0, oh = 0;
    output_resolution(ow, oh);
    const int fw = frame_native_w(), fh = frame_native_h();
    const int iscale = std::clamp(settings_.internal_scale, 1, 8);

    char out[48], fb[48], in[48];
    std::snprintf(fb, sizeof fb, "%d \xC3\x97 %d", fw, fh);
    std::snprintf(in, sizeof in, "%d \xC3\x97 %d", fw * iscale, fh * iscale);
    std::snprintf(out, sizeof out, "%d \xC3\x97 %d", ow, oh);
    const float cw = (w - gap * 2) / 3;
    const char* mode = settings_.render_scale == 0 ? "custom" : settings_.render_scale == 1 ? "as rendered" : "integer scale";
    stat_card("FRAME BUFFER", fb, cw, g_pal.text, core_.loaded() ? "set by the game" : "no game running");
    ImGui::SameLine(0, gap);
    stat_card("INTERNAL", in, cw, g_pal.accent_hover, iscale == 1 ? "native" : "drawn by the RDP");
    ImGui::SameLine(0, gap);
    stat_card("OUTPUT", out, cw, g_pal.text, mode);

    // Tiles: an internal or output scale factor with the resolution it gives.
    ImDrawList* dl = ImGui::GetWindowDrawList();
    auto scale_tiles = [&](const char* id, const int* scales, int n, int current, int& picked) {
        const int cols = w > dp(420) ? 4 : 3;
        const float tile_w = (w - gap * (cols - 1)) / cols, tile_h = dp(52);
        picked = 0;
        for (int i = 0; i < n; ++i) {
            if (i % cols) ImGui::SameLine(0, gap);
            const int k = scales[i];
            const bool on = current == k;
            ImGui::PushID(id);
            ImGui::PushID(k);
            ImVec2 p = ImGui::GetCursorScreenPos();
            if (ImGui::InvisibleButton("##scale", ImVec2(tile_w, tile_h))) picked = k;
            const bool hov = ImGui::IsItemHovered();
            ImGui::PopID();
            ImGui::PopID();
            dl->AddRectFilled(p, ImVec2(p.x + tile_w, p.y + tile_h), col(on ? g_pal.accent_soft : (hov ? g_pal.bg3 : g_pal.bg2)), dp(8));
            dl->AddRect(p, ImVec2(p.x + tile_w, p.y + tile_h), col(on ? g_pal.accent : g_pal.border), dp(8), 0, on ? dp(1.5f) : 1.0f);
            char t[24], r[32];
            std::snprintf(t, sizeof t, k == 1 ? "Native 1\xC3\x97" : "%d\xC3\x97", k);
            std::snprintf(r, sizeof r, "%d \xC3\x97 %d", fw * k, fh * k);
            dl->AddText(g_fonts.body_bold, font_px(g_fonts.body_bold), ImVec2(p.x + dp(12), p.y + dp(8)), col(on ? g_pal.text : g_pal.text_dim), t);
            dl->AddText(g_fonts.mono_small, font_px(g_fonts.mono_small), ImVec2(p.x + dp(12), p.y + dp(30)), col(g_pal.text_faint), r);
        }
    };

    tool_section("INTERNAL RESOLUTION");
    static const int kInternal[] = {1, 2, 3, 4, 5, 6, 8};
    int picked = 0;
    scale_tiles("internal", kInternal, 7, iscale, picked);
    if (picked) settings_.internal_scale = picked;
    ImGui::PushFont(g_fonts.small);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
    ImGui::TextColored(g_pal.text_faint,
                       "The RDP draws every triangle, texture and 2D element at this multiple of the game's frame buffer, "
                       "so edges and textures get sharper. It runs in software on all CPU cores; if the game slows down, "
                       "pick a lower value. The game itself still sees its normal frame buffer.");
    ImGui::PopTextWrapPos();
    ImGui::PopFont();

    tool_section("OUTPUT SCALE");
    static const int kScales[] = {1, 2, 3, 4, 5, 6, 8};
    scale_tiles("output", kScales, 7, settings_.render_scale, picked);
    if (picked) settings_.render_scale = picked;

    tool_section("CUSTOM OUTPUT RESOLUTION");
    static int cw_in = 0, ch_in = 0;
    if (cw_in == 0) { cw_in = settings_.render_w; ch_in = settings_.render_h; }
    float fwid = (w - dp(150) - gap * 2) / 2;
    ImGui::SetNextItemWidth(fwid);
    ImGui::InputInt("##cw", &cw_in, 0, 0);
    ImGui::SameLine(0, gap);
    ImGui::TextColored(g_pal.text_faint, "\xC3\x97");
    ImGui::SameLine(0, gap);
    ImGui::SetNextItemWidth(fwid);
    ImGui::InputInt("##ch", &ch_in, 0, 0);
    ImGui::SameLine(0, gap);
    if (button("Apply", Icon::Check, settings_.render_scale == 0 ? ButtonKind::Primary : ButtonKind::Subtle, dp(120), true, dp(32))) {
        settings_.render_w = std::clamp(cw_in, 64, 7680);
        settings_.render_h = std::clamp(ch_in, 64, 4320);
        cw_in = settings_.render_w;
        ch_in = settings_.render_h;
        settings_.render_scale = 0;
    }
    static const int kCustom[][2] = {{1280, 720}, {1920, 1080}, {2560, 1440}, {3840, 2160}};
    float chip = (w - gap * 3) / 4;
    for (int i = 0; i < 4; ++i) {
        if (i) ImGui::SameLine(0, gap);
        char l[24];
        std::snprintf(l, sizeof l, "%d\xC3\x97%d", kCustom[i][0], kCustom[i][1]);
        bool on = settings_.render_scale == 0 && settings_.render_w == kCustom[i][0] && settings_.render_h == kCustom[i][1];
        if (button(l, Icon::None, on ? ButtonKind::Primary : ButtonKind::Ghost, chip, true, dp(30))) {
            settings_.render_scale = 0;
            settings_.render_w = cw_in = kCustom[i][0];
            settings_.render_h = ch_in = kCustom[i][1];
        }
    }

    ImGui::Dummy(dp(0, 6));
    const bool at_native = settings_.render_scale == 1 && settings_.internal_scale == 1;
    if (button("Back to Native", Icon::Reset, ButtonKind::Subtle, w, !at_native, dp(36))) {
        settings_.render_scale = 1;
        settings_.internal_scale = 1;
    }
    ImGui::PushFont(g_fonts.small);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
    ImGui::TextColored(g_pal.text_faint,
                       "Output scaling only resamples the finished image (sharp nearest-neighbour for integer scales) before "
                       "it is shown with the texture filter and aspect ratio from Settings \xE2\x80\xBA Graphics. An output no "
                       "larger than the internal resolution is shown as rendered.");
    ImGui::PopTextWrapPos();
    ImGui::PopFont();
}

// ---------------------------------------------------------------------------
// Watch / freeze persistence (per game)
// ---------------------------------------------------------------------------

static fs::path debug_lists_path(const std::string& game) {
    std::string safe = game;
    for (char& c : safe) if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-') c = '_';
    fs::path dir = platform::config_dir() / "debug";
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir / (safe + ".txt");
}

void App::load_debug_lists() {
    dbg_.watches.clear();
    dbg_.freezes.clear();
    dbg_.freezes_dirty = true;
    std::ifstream f(debug_lists_path(dbg_.lists_game));
    std::string line;
    while (std::getline(f, line)) {
        std::vector<std::string> parts;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, '\t')) parts.push_back(item);
        if (parts.size() < 4) continue;
        std::uint32_t addr = 0;
        if (!parse_address(parts[1], addr)) continue;
        MemType t = static_cast<MemType>(std::clamp(std::atoi(parts[2].c_str()), 0, kMemTypeCount - 1));
        if (parts[0] == "W") {
            Watch w;
            w.addr = addr;
            w.type = t;
            w.name = parts[3];
            dbg_.watches.push_back(w);
        } else if (parts[0] == "F" && parts.size() >= 6) {
            Freeze fz;
            fz.addr = addr;
            fz.type = t;
            fz.name = parts[3];
            fz.value = parts[4];
            fz.enabled = parts[5] == "1";
            dbg_.freezes.push_back(fz);
        }
    }
}

void App::save_debug_lists() {
    if (dbg_.lists_game.empty()) return;
    std::ofstream f(debug_lists_path(dbg_.lists_game));
    for (const auto& w : dbg_.watches)
        f << "W\t" << hex32(w.addr) << "\t" << static_cast<int>(w.type) << "\t" << w.name << "\n";
    for (const auto& fz : dbg_.freezes)
        f << "F\t" << hex32(fz.addr) << "\t" << static_cast<int>(fz.type) << "\t" << fz.name << "\t" << fz.value << "\t"
          << (fz.enabled ? 1 : 0) << "\n";
}

} // namespace ui
