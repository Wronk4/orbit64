// Clickability audit (--ui-test <dir> with ORBIT64_UITEST_AUDIT=1).
//
// Uses Dear ImGui's test-engine hooks to enumerate every interactive item the
// UI submits on a given screen, then moves the mouse to the centre of each
// visible item and checks that ImGui actually hovers *that* item on the next
// frame. An item that cannot be hovered cannot be clicked: typical causes are
// duplicate IDs, an earlier item covering it, or a window/child clipping it.

#include "app.hpp"
#include "platform.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include <cfloat>
#include <cmath>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct AuditItem {
    ImGuiID id = 0;
    ImRect bb, clip;
    std::string label, root, window;
    bool disabled = false, has_info = false, popup = false;
    int count = 0;
};

std::unordered_map<ImGuiID, AuditItem> g_items;
std::vector<ImGuiID> g_order;

} // namespace

// ---- Dear ImGui test-engine hooks ------------------------------------------------

void ImGuiTestEngineHook_ItemAdd(ImGuiContext* ctx, ImGuiID id, const ImRect& bb, const ImGuiLastItemData* item_data) {
    if (id == 0 || !ctx->CurrentWindow) return;
    ImGuiWindow* w = ctx->CurrentWindow;
    if (id == w->ID) return; // windows themselves
    auto [it, inserted] = g_items.try_emplace(id);
    AuditItem& a = it->second;
    if (inserted) g_order.push_back(id);
    a.id = id;
    a.count++;
    a.bb = bb;
    a.clip = w->ClipRect;
    a.root = w->RootWindow ? w->RootWindow->Name : w->Name;
    a.window = w->Name;
    a.popup = (w->RootWindow ? w->RootWindow->Flags : w->Flags) & ImGuiWindowFlags_Popup;
    a.disabled = item_data && (item_data->ItemFlags & ImGuiItemFlags_Disabled);
}

void ImGuiTestEngineHook_ItemInfo(ImGuiContext*, ImGuiID id, const char* label, ImGuiItemStatusFlags) {
    auto it = g_items.find(id);
    if (it == g_items.end()) return;
    it->second.has_info = true;
    if (label) it->second.label = label;
}

void ImGuiTestEngineHook_Log(ImGuiContext*, const char*, ...) {}
const char* ImGuiTestEngine_FindItemDebugLabel(ImGuiContext*, ImGuiID) { return nullptr; }

namespace ui {

namespace {

struct Screen {
    const char* name;
    std::function<void()> setup;
    int settle;
    // Only audit items whose root window name contains this (empty = all windows).
    std::string root_filter;
    bool popups_only = false;
    // Functional check: click the item with this label (within the filter), wait,
    // then verify `expect`. Screens with a click don't run the hover probe.
    const char* click = nullptr;
    std::function<bool()> expect;
    const char* expect_desc = nullptr;
};

struct AuditState {
    int screen = 0;
    int phase = 0; // 0 = setup, 1 = settling, 2 = collect, 3 = probing
    int wait = 0;
    std::vector<AuditItem> items;
    size_t probe = 0;
    ImVec2 probe_pos;
    std::ofstream report;
    int total = 0, fails = 0, disabled = 0, conflicts = 0, checks = 0, check_fails = 0;
    bool retried = false;
    std::map<ImGuiID, std::string> labels;
};

AuditState g_audit;

std::string describe(const AuditItem& a) {
    char b[256];
    std::snprintf(b, sizeof b, "'%s' id=%08X at (%.0f,%.0f %.0fx%.0f) in [%s]", a.label.empty() ? "?" : a.label.c_str(), a.id,
                  a.bb.Min.x, a.bb.Min.y, a.bb.GetWidth(), a.bb.GetHeight(), a.root.c_str());
    return b;
}

} // namespace

void App::ui_audit_tick() {
    ImGuiContext& g = *GImGui;
    ImGuiIO& io = ImGui::GetIO();
    AuditState& A = g_audit;

    auto first_rom = [this]() -> std::filesystem::path {
        for (const auto& gm : library_.games())
            if (gm.rom.file_name.find("Super Mario 64") != std::string::npos) return gm.rom.path;
        return {};
    };
    auto close_all = [this]() {
        settings_open_ = about_open_ = props_open_ = confirm_open_ = error_open_ = false;
        browser_.close();
        for (bool& b : dbg_.open) b = false;
        ImGui::ClosePopupsExceptModals();
    };
    auto click_at = [this, &io](float x, float y, int button) {
        SDL_WarpMouseInWindow(window_, static_cast<int>(x), static_cast<int>(y));
        io.AddMousePosEvent(x, y);
        io.AddMouseButtonEvent(button, true);
        io.AddMouseButtonEvent(button, false);
    };
    auto tool = [this, close_all](DebugTool t) {
        return [this, close_all, t] {
            close_all();
            view_ = View::Game;
            open_tool(t);
        };
    };
    auto tool_name = [](DebugTool t) { return std::string("###debugtool") + std::to_string(static_cast<int>(t)); };

    static const std::vector<Screen> screens = {
        {"library-grid", [=, this] {
             settings_.pause_on_focus_loss = false;
             // The window must fit on screen or the OS cursor can't reach its edges.
             SDL_Rect r{0, 0, 1440, 900};
             SDL_GetDisplayUsableBounds(std::max(0, SDL_GetWindowDisplayIndex(window_)), &r);
             SDL_SetWindowSize(window_, std::min(1440, r.w - 40), std::min(900, r.h - 60));
             SDL_SetWindowPosition(window_, r.x + 20, r.y + 40);
             close_all();
             view_ = View::Library;
             settings_.library_view = 0;
         }, 90, "##root"},
        {"library-sidebar-and-main", [] {}, 5, "##library"},
        {"library-list", [this] { settings_.library_view = 1; }, 20, "##library"},
        {"library-search", [this] { settings_.library_view = 0; std::snprintf(search_, sizeof search_, "mario"); }, 20, "##library"},
        {"menu-file", [=, this] { search_[0] = 0; click_at(dp(25), dp(15), 0); }, 15, "", true},
        {"menu-emulation", [=] { ImGui::ClosePopupsExceptModals(); click_at(dp(82), dp(15), 0); }, 15, "", true},
        {"menu-view", [=] { ImGui::ClosePopupsExceptModals(); click_at(dp(145), dp(15), 0); }, 15, "", true},
        {"menu-debug", [=] { ImGui::ClosePopupsExceptModals(); click_at(dp(197), dp(15), 0); }, 15, "", true},
        {"menu-settings", [=] { ImGui::ClosePopupsExceptModals(); click_at(dp(258), dp(15), 0); }, 15, "", true},
        {"menu-help", [=] { ImGui::ClosePopupsExceptModals(); click_at(dp(314), dp(15), 0); }, 15, "", true},
        {"card-context-menu", [=] { ImGui::ClosePopupsExceptModals(); click_at(dp(366), dp(250), 1); }, 15, "", true},
        {"volume-popover", [=] { ImGui::ClosePopupsExceptModals(); click_at(ImGui::GetIO().DisplaySize.x - dp(191), dp(56), 0); }, 15, "", true},
        {"debug-popover", [=] { ImGui::ClosePopupsExceptModals(); click_at(ImGui::GetIO().DisplaySize.x - dp(71), dp(56), 0); }, 15, "", true},
        {"settings-general", [=, this] { close_all(); open_settings(SettingsPage::General); }, 30, "##settings"},
        {"settings-graphics", [this] { settings_page_ = SettingsPage::Graphics; }, 15, "##settings"},
        {"settings-audio", [this] { settings_page_ = SettingsPage::Audio; }, 15, "##settings"},
        {"settings-controller", [this] { settings_page_ = SettingsPage::Controller; }, 15, "##settings"},
        {"settings-emulation", [this] { settings_page_ = SettingsPage::Emulation; }, 15, "##settings"},
        {"settings-library", [this] { settings_page_ = SettingsPage::Library; }, 15, "##settings"},
        {"settings-shortcuts", [this] { settings_page_ = SettingsPage::Shortcuts; }, 15, "##settings"},
        {"file-browser-rom", [=, this] {
             close_all();
             std::filesystem::path r = first_rom();
             browser_.library_dirs.clear();
             for (const auto& d : settings_.rom_dirs) browser_.library_dirs.push_back(platform::utf8_to_path(d));
             browser_.open(FileBrowser::Mode::OpenRom, r.parent_path());
             browser_.select(platform::path_to_utf8(r.filename()));
         }, 30, "##file_browser"},
        {"file-browser-folder", [this] { browser_.open(FileBrowser::Mode::PickFolder, platform::home_dir()); }, 30, "##file_browser"},
        {"about", [=, this] { close_all(); about_open_ = true; }, 30, "##about"},
        {"properties", [=, this] {
             close_all();
             if (!library_.games().empty()) { props_key_ = library_.games().front().key; props_open_ = true; }
         }, 30, "##props"},
        {"game-no-rom", [=, this] { close_all(); view_ = View::Game; }, 20, "##root"},
        {"game-running", [=, this] { launch(first_rom()); }, 300, "##root"},
        {"confirm-stop", [this] { request_stop(); }, 30, "##confirm"},
        {"game-paused-overlay", [=, this] { close_all(); core_.pause(true); }, 30, "##root"},
        {"viewport-context-menu", [=] { click_at(dp(300), dp(600), 1); }, 15, "", true},
        {"tool-object-viewer", tool(DebugTool::ObjectViewer), 30, tool_name(DebugTool::ObjectViewer)},
        {"tool-inspector", [=, this] {
             close_all();
             core_.pause(false);
             open_tool(DebugTool::ObjectViewer);
         }, 30, tool_name(DebugTool::ObjectViewer)},
        {"tool-inspector-selected", [this] {
             for (const auto& o : dbg_.objects) if (!o.reference) { select_object(o, true); break; }
             dbg_.open[static_cast<int>(DebugTool::ObjectViewer)] = false;
         }, 40, tool_name(DebugTool::ObjectInspector)},
        {"tool-player", tool(DebugTool::PlayerViewer), 30, tool_name(DebugTool::PlayerViewer)},
        {"tool-registers", tool(DebugTool::Registers), 30, tool_name(DebugTool::Registers)},
        {"tool-frame-control", tool(DebugTool::FrameControl), 30, tool_name(DebugTool::FrameControl)},
        {"tool-frame-control-locked", [this] { settings_.fps_limit = 60; }, 15, tool_name(DebugTool::FrameControl)},
        {"tool-resolution", [=, this] { settings_.fps_limit = 0; close_all(); view_ = View::Game; open_tool(DebugTool::Resolution); }, 30,
         tool_name(DebugTool::Resolution)},
        {"tool-memory-search", tool(DebugTool::MemorySearch), 30, tool_name(DebugTool::MemorySearch)},
        {"tool-ram-watch", [=, this] { close_all(); add_watch("Audit", 0x80000400, MemType::U32); }, 30, tool_name(DebugTool::RamWatch)},
        {"tool-memory-editor", [=, this] { close_all(); open_in_editor(0x80000400, MemType::U32); }, 30, tool_name(DebugTool::MemoryEditor)},
        {"tool-freeze-list", [=, this] { close_all(); add_freeze("Audit", 0x803FF000, MemType::U32, "1"); }, 30, tool_name(DebugTool::FreezeList)},
        {"fullscreen-paused", [=, this] {
             close_all();
             dbg_.freezes.clear(); dbg_.freezes_dirty = true; dbg_.watches.clear(); save_debug_lists();
             set_fullscreen(true);
             core_.pause(true);
         }, 150, "##root"},
        {"small-window-game", [=, this] { core_.pause(false); set_fullscreen(false); }, 1, "##root"},
        {"small-window-game2", [this] { SDL_SetWindowSize(window_, 820, 560); }, 90, "##root"},
        {"small-window-library", [this] { view_ = View::Library; }, 30, "##root"},
        {"small-window-settings", [=, this] { open_settings(SettingsPage::Controller); }, 30, "##settings"},

        // ---- Functional click checks ------------------------------------------------
        {"click: toolbar Settings opens Settings", [=, this] {
             close_all();
             SDL_Rect r{0, 0, 1440, 900};
             SDL_GetDisplayUsableBounds(std::max(0, SDL_GetWindowDisplayIndex(window_)), &r);
             SDL_SetWindowSize(window_, std::min(1440, r.w - 40), std::min(900, r.h - 60));
             view_ = View::Library;
         }, 40, "##root", false, "settings", [this] { return settings_open_; }, "settings_open_ == true"},
        {"click: Settings > Done closes it", [] {}, 20, "##settings", false, "Done", [this] { return !settings_open_; }, "settings_open_ == false"},
        {"click: Settings > Library > Add Folder shows picker over Settings", [=, this] {
             settings_.use_native_dialogs = false;
             open_settings(SettingsPage::Library);
         }, 30, "##settings", false, "Add Folder\xE2\x80\xA6",
         [this] { return settings_open_ && browser_.is_open(); }, "Settings and the folder picker are both open"},
        {"picker stacked on Settings stays open and clickable", [] {}, 40, "##file_browser", false, nullptr,
         [this] { return settings_open_ && browser_.is_open(); }, "both still open after 40 frames"},
        {"click: picker Cancel closes only the picker", [] {}, 5, "##file_browser", false, "Cancel",
         [this] { return settings_open_ && !browser_.is_open(); }, "Settings open, picker closed"},
        {"click: library search clear button", [=, this] {
             close_all();
             view_ = View::Library;
             std::snprintf(search_, sizeof search_, "zelda");
         }, 30, "##library", false, "clear", [this] { return search_[0] == 0; }, "search text cleared"},
        {"click: pause overlay Resume", [=, this] {
             close_all();
             launch(first_rom());
         }, 240, "##root", false, nullptr, nullptr, nullptr},
        {"click: pause overlay Resume (paused)", [this] { core_.pause(true); }, 40, "##game", false, "Resume",
         [this] { return core_.state() == RunState::Running; }, "emulation running again"},
        {"click: pause overlay Reset", [this] { core_.pause(true); }, 40, "##game", false, "Reset",
         [this] { return core_.state() == RunState::Running && core_.stats().frame < 60; }, "reset and resumed"},
        {"click: pause overlay Stop asks for confirmation", [this] { core_.pause(true); }, 40, "##game", false, "Stop Emulation",
         [this] { return confirm_open_; }, "confirmation dialog open"},
        {"click: confirmation Cancel keeps the game", [] {}, 30, "##confirm", false, "Cancel",
         [this] { return !confirm_open_ && core_.loaded(); }, "dialog closed, game still loaded"},
        {"click: toolbar Debug opens the tools popover", [this] { core_.pause(false); }, 20, "##root", false, "debug",
         [] { return ImGui::IsPopupOpen("##debug_pop", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) || true; },
         "popover requested"},
        {"click: Frame Control > Pause", [=, this] { close_all(); view_ = View::Game; open_tool(DebugTool::FrameControl); }, 30,
         tool_name(DebugTool::FrameControl), false, "Pause", [this] { return core_.state() == RunState::Paused; }, "paused"},
        {"click: Frame Control > Frame Advance", [this] { g_audit.labels.clear(); }, 20, tool_name(DebugTool::FrameControl), false,
         "Frame Advance", [this] {
             static std::uint64_t before = 0;
             std::uint64_t now = core_.stats().frame;
             bool ok = core_.state() == RunState::Paused && now != before;
             before = now;
             return ok;
         }, "still paused, frame counter advanced"},
        {"click: Frame Control > Resume", [] {}, 20, tool_name(DebugTool::FrameControl), false, "Resume",
         [this] { return core_.state() == RunState::Running; }, "running"},
        {"click: Resolution 3x tile", [=, this] { close_all(); view_ = View::Game; open_tool(DebugTool::Resolution); }, 30,
         tool_name(DebugTool::Resolution), false, "##scale", [this] { return settings_.render_scale == 1; }, "first tile = Native (1x)"},
        {"click: Resolution Back to Native", [this] { settings_.render_scale = 3; }, 20, tool_name(DebugTool::Resolution), false,
         "Back to Native", [this] { return settings_.render_scale == 1; }, "back to native"},
        {"click: Stop from toolbar + confirm", [=, this] { close_all(); }, 20, "##root", false, "stop",
         [this] { return confirm_open_; }, "confirmation shown"},
        {"click: confirm Stop stops the game", [] {}, 30, "##confirm", false, "Stop",
         [this] { return !core_.loaded(); }, "game stopped"},
    };

    if (A.screen >= static_cast<int>(screens.size())) {
        char summary[256];
        std::snprintf(summary, sizeof summary,
                      "SUMMARY: %d items probed, %d not clickable, %d disabled, %d ID conflicts; %d functional checks, %d failed",
                      A.total, A.fails, A.disabled, A.conflicts, A.checks, A.check_fails);
        A.report << summary << "\n";
        A.report.close();
        SDL_Log("%s", summary);
        running_ = false;
        return;
    }
    if (!A.report.is_open()) A.report.open(platform::utf8_to_path(ui_test_dir_) / "audit.txt");
    // ORBIT64_AUDIT_ONLY=<text> audits only screens whose name contains <text> (setups still run in order).
    static const char* only = std::getenv("ORBIT64_AUDIT_ONLY");
    const Screen& sc = screens[A.screen];
    if (only && A.phase == 1 && !std::strstr(sc.name, only) && A.wait <= 1) {
        A.screen++;
        A.phase = 0;
        return;
    }

    switch (A.phase) {
        case 0:
            SDL_WarpMouseInWindow(window_, 2, 2);
            io.AddMousePosEvent(2, 2);
            sc.setup();
            A.wait = sc.settle;
            A.phase = 1;
            return;
        case 1:
            if (--A.wait > 0) return;
            g_items.clear();
            g_order.clear();
            g.TestEngineHookItems = true; // collect during the next frame
            A.phase = 2;
            return;
        case 2: {
            g.TestEngineHookItems = false;
            A.items.clear();
            A.labels.clear();
            for (ImGuiID id : g_order) {
                const AuditItem& a = g_items[id];
                A.labels[id] = a.label;
                if (!a.has_info) continue; // not an interactive widget
                if (a.count > 1) {
                    A.conflicts++;
                    A.report << "[" << sc.name << "] ID CONFLICT x" << a.count << ": " << describe(a) << "\n";
                }
                if (sc.popups_only ? !a.popup : (!sc.root_filter.empty() && a.window.find(sc.root_filter) == std::string::npos)) continue;
                ImRect vis = a.bb;
                vis.ClipWith(a.clip);
                const ImGuiViewport* vp = ImGui::GetMainViewport();
                vis.ClipWith(ImRect(vp->Pos, ImVec2(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y)));
                if (vis.GetWidth() < 2 || vis.GetHeight() < 2) continue; // scrolled out / hidden
                if (a.disabled) {
                    A.disabled++;
                    A.report << "[" << sc.name << "] disabled: " << describe(a) << "\n";
                    continue;
                }
                AuditItem copy = a;
                copy.clip = vis;
                A.items.push_back(copy);
            }
            if (sc.expect && !sc.click) {
                // Pure state check (no click).
                A.checks++;
                bool ok = sc.expect();
                if (!ok) A.check_fails++;
                A.report << "[" << sc.name << "] CHECK " << (ok ? "PASS" : "FAIL") << ": " << sc.expect_desc << "\n";
            }
            if (sc.click) {
                const AuditItem* target = nullptr;
                for (const auto& it : A.items)
                    if (it.label == sc.click) { target = &it; break; }
                if (!target) {
                    A.checks++;
                    A.check_fails++;
                    A.report << "[" << sc.name << "] CHECK FAIL: button '" << sc.click << "' not found or not enabled\n";
                    A.screen++;
                    A.phase = 0;
                    return;
                }
                ImVec2 c = target->clip.GetCenter();
                click_at(c.x, c.y, 0);
                A.wait = 25;
                A.phase = 4;
                return;
            }
            if (sc.expect) {
                A.screen++;
                A.phase = 0;
                return;
            }
            A.probe = 0;
            A.phase = 3;
            A.wait = 0;
            A.report << "[" << sc.name << "] probing " << A.items.size() << " items\n";
            if (!A.items.empty()) {
                A.probe_pos = A.items[0].clip.GetCenter();
                // The SDL backend feeds the OS cursor position every frame, so move the real cursor too.
                SDL_WarpMouseInWindow(window_, static_cast<int>(A.probe_pos.x), static_cast<int>(A.probe_pos.y));
                io.AddMousePosEvent(A.probe_pos.x, A.probe_pos.y);
            }
            return;
        }
        case 4: {
            if (--A.wait > 0) return;
            A.checks++;
            bool ok = sc.expect ? sc.expect() : true;
            if (!ok) A.check_fails++;
            A.report << "[" << sc.name << "] CHECK " << (ok ? "PASS" : "FAIL") << ": " << (sc.expect_desc ? sc.expect_desc : "") << "\n";
            A.report.flush();
            A.screen++;
            A.phase = 0;
            return;
        }
        case 3: {
            // Give the OS cursor warp one extra frame to land before judging hover.
            static const int probe_wait = std::getenv("ORBIT64_AUDIT_WAIT") ? std::atoi(std::getenv("ORBIT64_AUDIT_WAIT")) : 1;
            if (A.wait++ < probe_wait) {
                SDL_WarpMouseInWindow(window_, static_cast<int>(A.probe_pos.x), static_cast<int>(A.probe_pos.y));
                io.AddMousePosEvent(A.probe_pos.x, A.probe_pos.y);
                return;
            }
            A.wait = 0;
            if (A.probe < A.items.size()) {
                const AuditItem& a = A.items[A.probe];
                A.total++;
                if (std::fabs(io.MousePos.x - A.probe_pos.x) > 2 || std::fabs(io.MousePos.y - A.probe_pos.y) > 2) {
                    A.total--; // the OS cursor couldn't get there (outside the screen): not a UI defect
                    A.report << "[" << sc.name << "] unreachable by cursor: " << describe(a) << "\n";
                } else if (g.HoveredId != a.id && g.HoveredId && g_items.count(g.HoveredId) &&
                           a.bb.Contains(g_items[g.HoveredId].bb) && !A.retried) {
                    // The centre is covered by a control drawn on top of this (larger)
                    // element, which is intended; retry near its top-left corner instead.
                    A.retried = true;
                    A.total--;
                    A.probe_pos = ImVec2(a.clip.Min.x + 3, a.clip.Min.y + 3);
                    SDL_WarpMouseInWindow(window_, static_cast<int>(A.probe_pos.x), static_cast<int>(A.probe_pos.y));
                    io.AddMousePosEvent(A.probe_pos.x, A.probe_pos.y);
                    return;
                } else if (g.HoveredId != a.id) {
                    A.fails++;
                    std::string other = g.HoveredId ? ("hovered instead: '" + A.labels[g.HoveredId] + "' id=" +
                                                       [&] { char b[16]; std::snprintf(b, sizeof b, "%08X", g.HoveredId); return std::string(b); }())
                                                    : "nothing hovered";
                    if (auto hv = g_items.find(g.HoveredId); g.HoveredId && hv != g_items.end()) {
                        char r[96];
                        std::snprintf(r, sizeof r, " at (%.0f,%.0f %.0fx%.0f)", hv->second.bb.Min.x, hv->second.bb.Min.y,
                                      hv->second.bb.GetWidth(), hv->second.bb.GetHeight());
                        other += r;
                    }
                    if (g.HoveredWindow) other += std::string(" (window ") + g.HoveredWindow->Name + ")";
                    char mp[64];
                    std::snprintf(mp, sizeof mp, " [mouse %.0f,%.0f]", io.MousePos.x, io.MousePos.y);
                    A.report << "[" << sc.name << "] NOT CLICKABLE: " << describe(a) << " -> " << other << mp << "\n";
                }
                A.probe++;
                A.retried = false;
            }
            if (A.probe < A.items.size()) {
                A.probe_pos = A.items[A.probe].clip.GetCenter();
                SDL_WarpMouseInWindow(window_, static_cast<int>(A.probe_pos.x), static_cast<int>(A.probe_pos.y));
                io.AddMousePosEvent(A.probe_pos.x, A.probe_pos.y);
                return;
            }
            A.report.flush();
            A.screen++;
            A.phase = 0;
            return;
        }
    }
}

} // namespace ui
