// Scripted UI walkthrough (--ui-test <dir>).
//
// Drives the real application through every screen and dialog and saves a
// capture of the rendered frame after each step. Because it reads back the
// renderer output it works the same with Direct3D, Metal and OpenGL.

#include "app.hpp"
#include "platform.hpp"

#include "imgui.h"

#include <cfloat>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <vector>

namespace ui {

namespace fs = std::filesystem;

void App::ui_test_capture(const std::string& name) {
    int w = 0, h = 0;
    SDL_GetRendererOutputSize(renderer_, &w, &h);
    std::vector<std::uint32_t> px(static_cast<size_t>(w) * h);
    if (SDL_RenderReadPixels(renderer_, nullptr, SDL_PIXELFORMAT_ARGB8888, px.data(), w * 4) != 0) {
        SDL_Log("ui-test: read pixels failed: %s", SDL_GetError());
        return;
    }
    // Classic 24-bit BMP: readable by every image viewer / converter.
    fs::path out = platform::utf8_to_path(ui_test_dir_) / (name + ".bmp");
    std::ofstream f(out, std::ios::binary);
    const int row = (w * 3 + 3) & ~3;
    const std::uint32_t data_size = static_cast<std::uint32_t>(row) * h;
    auto u32 = [&](std::uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](std::uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    f.put('B').put('M');
    u32(54 + data_size); u32(0); u32(54);
    u32(40); u32(static_cast<std::uint32_t>(w)); u32(static_cast<std::uint32_t>(h)); u16(1); u16(24);
    u32(0); u32(data_size); u32(2835); u32(2835); u32(0); u32(0);
    std::vector<char> line(row, 0);
    for (int y = h - 1; y >= 0; --y) {
        for (int x = 0; x < w; ++x) {
            std::uint32_t p = px[static_cast<size_t>(y) * w + x];
            line[x * 3 + 0] = static_cast<char>(p & 0xFF);
            line[x * 3 + 1] = static_cast<char>((p >> 8) & 0xFF);
            line[x * 3 + 2] = static_cast<char>((p >> 16) & 0xFF);
        }
        f.write(line.data(), row);
    }
    SDL_Log("ui-test: captured %s (%dx%d)", name.c_str(), w, h);
}

void App::ui_test_tick() {
    static const bool audit_mode = std::getenv("ORBIT64_UITEST_AUDIT") != nullptr;
    if (audit_mode) {
        ui_audit_tick();
        return;
    }
    struct Step {
        std::function<void()> action;
        int wait;
        const char* shot;
    };
    auto first_rom = [this]() -> fs::path {
        for (const auto& g : library_.games())
            if (g.rom.file_name.find("Super Mario 64") != std::string::npos) return g.rom.path;
        return library_.games().empty() ? fs::path() : library_.games().front().rom.path;
    };
    auto resize = [this](int w, int h) {
        SDL_RestoreWindow(window_);
        SDL_SetWindowSize(window_, w, h);
        SDL_SetWindowPosition(window_, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    };
    // Synthetic input goes through ImGui's event queue exactly like real
    // mouse/keyboard events, so hit-testing and popups are exercised too.
    auto click = [this](float x, float y, int button) {
        ImGuiIO& io = ImGui::GetIO();
        SDL_WarpMouseInWindow(window_, static_cast<int>(x), static_cast<int>(y)); // keep the OS cursor in sync
        io.AddMousePosEvent(x, y);
        io.AddMouseButtonEvent(button, true);
        io.AddMouseButtonEvent(button, false);
    };
    auto hover = [this](float x, float y) {
        SDL_WarpMouseInWindow(window_, static_cast<int>(x), static_cast<int>(y));
        ImGui::GetIO().AddMousePosEvent(x, y);
    };
    auto escape = []() {
        ImGuiIO& io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiKey_Escape, true);
        io.AddKeyEvent(ImGuiKey_Escape, false);
        io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
    };
    static const std::vector<Step> steps = {
        {[this, resize] { settings_.pause_on_focus_loss = false; settings_.confirm_stop = true; resize(1440, 900); }, 90, "01_library_grid"},
        {[this] { settings_.library_view = 1; }, 20, "02_library_list"},
        {[this] { settings_.library_view = 0; std::snprintf(search_, sizeof search_, "zelda"); }, 20, "03_library_search"},
        {[this] { search_[0] = 0; lib_filter_ = LibraryFilter::Homebrew; }, 20, "04_library_filter_homebrew"},
        {[this, click] { lib_filter_ = LibraryFilter::All; click(dp(25), dp(15), 0); }, 15, "04a_menu_file"},
        {[escape, click] { escape(); click(dp(145), dp(15), 0); }, 15, "04b_menu_view"},
        {[escape] { escape(); }, 10, nullptr},
        {[click] { click(dp(366), dp(490), 1); }, 15, "04c_context_menu"},
        {[escape] { escape(); }, 10, nullptr},
        {[click] { click(dp(1287), dp(56), 0); }, 15, "04d_volume_popover"},
        {[escape] { escape(); }, 10, nullptr},
        {[click] { click(dp(1222), dp(122), 0); }, 15, "04e_sort_dropdown"},
        {[escape] { escape(); }, 10, nullptr},
        {[this] { lib_filter_ = LibraryFilter::All; open_settings(SettingsPage::General); }, 30, "05_settings_general"},
        {[this] { settings_page_ = SettingsPage::Graphics; }, 20, "06_settings_graphics"},
        {[this] { settings_page_ = SettingsPage::Audio; }, 20, "07_settings_audio"},
        {[this] { settings_page_ = SettingsPage::Controller; }, 20, "08_settings_controller"},
        {[this] { input_.begin_capture(0, 0); }, 10, "09_settings_controller_rebind"},
        {[this] { input_.cancel_capture(); settings_page_ = SettingsPage::Emulation; }, 20, "10_settings_emulation"},
        {[this] { settings_page_ = SettingsPage::Library; }, 20, "11_settings_library"},
        {[this] { settings_page_ = SettingsPage::Shortcuts; }, 20, "12_settings_shortcuts"},
        {[this] { settings_open_ = false; }, 20, nullptr},
        {[this, first_rom] {
             browser_.library_dirs.clear();
             for (const auto& d : settings_.rom_dirs) browser_.library_dirs.push_back(platform::utf8_to_path(d));
             fs::path r = first_rom();
             browser_.open(FileBrowser::Mode::OpenRom, r.parent_path());
             browser_.select(platform::path_to_utf8(r.filename()));
         }, 30, "13_rom_browser"},
        {[this] { browser_.open(FileBrowser::Mode::PickFolder, platform::home_dir()); }, 30, "14_folder_picker"},
        {[this] { browser_.close(); about_open_ = true; }, 30, "15_about"},
        {[this] { about_open_ = false; view_ = View::Game; }, 30, "16_game_no_rom"},
        {[this, first_rom] { launch(first_rom()); }, 420, "17_game_running"},
        // ---- Save states (kept in <capture dir>/states)
        {[this] { save_state(1); }, 40, "17a_state_saved"},
        {[click] { click(dp(88), dp(15), 0); }, 20, "17b_menu_emulation_states"},
        {[hover] { hover(dp(108), dp(191)); }, 20, nullptr},                       // Load State from Slot >
        {[hover] { hover(dp(340), dp(191)); }, 30, "17b2_menu_load_slot_preview"}, // Slot 1: picture + time
        {[escape] { escape(); }, 10, nullptr},
        {[this] { save_state(3); }, 90, nullptr},
        {[this] { load_state(1); }, 40, "17c_state_loaded"},
        {[this] { undo_load_state(); }, 40, "17d_state_load_undone"},
        {[this] { core_.pause(true); load_state(3); }, 40, "17e_state_loaded_paused"},
        {[this] { core_.pause(false); }, 10, nullptr},
        // ---- DEBUG / MEMORY tools
        {[click] { click(dp(1369), dp(56), 0); }, 20, "40_debug_toolbar_menu"},
        {[escape, click] { escape(); click(dp(318), dp(15), 0); }, 20, "41_debug_menubar"},
        {[this, escape] { escape(); open_tool(DebugTool::ObjectViewer); }, 30, "42_object_viewer"},
        {[this] {
             const SceneObject* best = nullptr;
             for (const auto& o : dbg_.objects)
                 if (!o.reference && (!best || o.triangles > best->triangles)) best = &o;
             if (!best && !dbg_.objects.empty()) best = &dbg_.objects.front();
             if (best) select_object(*best, true);
         }, 40, "43_object_inspector"},
        {[this] { dbg_.open[static_cast<int>(DebugTool::ObjectViewer)] = false; dbg_.cam.yaw += 1.4f; dbg_.cam.pitch = 0.1f; }, 30, "44_inspector_rotated"},
        {[this] { dbg_.wireframe = true; }, 20, "45_inspector_wireframe"},
        {[this] {
             dbg_.wireframe = false;
             dbg_.open[static_cast<int>(DebugTool::ObjectInspector)] = false;
             dbg_.player = dbg_.selected;
             open_tool(DebugTool::PlayerViewer);
         }, 90, "46_player_viewer"},
        {[this] { dbg_.open[static_cast<int>(DebugTool::PlayerViewer)] = false; open_tool(DebugTool::Registers); }, 30, "47_registers_gpr"},
        {[this] { dbg_.reg_tab = 2; }, 20, "48_registers_fpu"},
        {[this] { dbg_.reg_tab = 1; }, 20, "49_registers_cop0"},
        {[this] { dbg_.open[static_cast<int>(DebugTool::Registers)] = false; open_tool(DebugTool::FrameControl); core_.pause(true); }, 20, nullptr},
        {[this] { core_.frame_advance(); }, 10, nullptr},
        {[this] { core_.frame_advance(); settings_.fps_limit = 120; }, 20, "50_frame_control_paused"},
        {[this] { settings_.fps_limit = 0; core_.pause(false); core_.set_turbo(true); }, 40, "51_frame_control_turbo"},
        {[this] {
             core_.set_turbo(false);
             dbg_.open[static_cast<int>(DebugTool::FrameControl)] = false;
             open_tool(DebugTool::Resolution);
             settings_.render_scale = 4;
         }, 30, "52_resolution_4x"},
        {[this] { settings_.render_scale = 1; settings_.internal_scale = 3; }, 60, "52a_internal_resolution_3x"},
        {[this] { settings_.internal_scale = 1; dbg_.open[static_cast<int>(DebugTool::Resolution)] = false; open_tool(DebugTool::MemorySearch);
                  dbg_.search_type = static_cast<int>(MemType::U32); dbg_.search_mode = 1; }, 10, nullptr},
        {[this] {
             // Unknown initial value, then "Changed" twice to narrow to live counters.
             const auto& ram = dbg_.snap->rdram;
             dbg_.search_mask.assign(ram.size() / 4, 1);
             dbg_.search_prev = ram;
             dbg_.search_started = true;
             dbg_.search_scans = 1;
             dbg_.search_count = ram.size() / 4;
             dbg_.search_mode = 6;
         }, 30, nullptr},
        {[this] { open_tool(DebugTool::MemorySearch); }, 30, "53_memory_search_started"},
        {[click] { click(dp(520), dp(392), 0); }, 20, nullptr},
        {[click] { click(dp(520), dp(392), 0); }, 30, "53b_memory_search_narrowed"},
        {[this] {
             dbg_.open[static_cast<int>(DebugTool::MemorySearch)] = false;
             add_watch("Frame counter?", 0x8032D5D4, MemType::U32);
             add_watch("Stack pointer area", 0x80200600, MemType::U32);
             add_watch("Some float", 0x80339E00, MemType::F32);
         }, 40, "54_ram_watch"},
        {[this] { dbg_.open[static_cast<int>(DebugTool::RamWatch)] = false; open_in_editor(0x80000400, MemType::U32); }, 30, "55_memory_editor"},
        {[this] {
             dbg_.open[static_cast<int>(DebugTool::MemoryEditor)] = false;
             add_freeze("Test freeze", 0x803FF000, MemType::U32, "1234");
         }, 30, "56_freeze_list"},
        {[this] { dbg_.freezes.clear(); dbg_.freezes_dirty = true; dbg_.watches.clear(); save_debug_lists();
                  for (bool& b : dbg_.open) b = false; }, 20, nullptr},
        {[this] { settings_.show_fps_overlay = true; }, 30, "18_game_fps_overlay"},
        {[this] { settings_.show_fps_overlay = false; toggle_pause(); }, 30, "19_game_paused"},
        {[this] { toggle_pause(); settings_.show_info_panel = false; settings_.scanlines = 45; }, 30, "20_game_no_panel_scanlines"},
        {[this] { settings_.show_info_panel = true; settings_.scanlines = 0; view_ = View::Library; }, 40, "21_library_while_playing"},
        {[this] { props_key_ = current_key_; props_open_ = true; }, 30, "22_properties"},
        {[this] { props_open_ = false; request_stop(); }, 30, "23_confirm_stop"},
        {[this] { confirm_open_ = false; view_ = View::Game; set_fullscreen(true); }, 150, "24_fullscreen_game"},
        {[this] { core_.pause(true); }, 30, "25_fullscreen_paused"},
        {[this] { core_.pause(false); set_fullscreen(false); }, 120, nullptr},
        {[resize] { resize(820, 560); }, 40, "26_small_window_game"},
        {[this] { view_ = View::Library; }, 30, "27_small_window_library"},
        {[this] { open_settings(SettingsPage::Controller); }, 30, "28_small_window_settings"},
        {[this, resize] { settings_open_ = false; resize(1280, 800); toast("Screenshot saved to Orbit64/example.bmp", ToastKind::Success); }, 30, "29_toast"},
        {[this] { stop_now(); }, 30, "30_library_after_stop"},
    };

    // Scene variant (ORBIT64_UITEST_SCENE=1): fast-forwards Super Mario 64 into the
    // castle-grounds intro and exercises the object tools on a real 3D scene.
    static const bool scene_mode = std::getenv("ORBIT64_UITEST_SCENE") != nullptr;
    static std::uint64_t wait_frame = 0;
    static const std::vector<Step> scene_steps = {
        {[this, resize] { settings_.pause_on_focus_loss = false; resize(1440, 900); }, 90, nullptr}, // library scan
        {[this, first_rom] {
             test_input_ = [](std::uint64_t f) -> std::uint16_t {
                 struct P { std::uint64_t at; std::uint16_t b; };
                 static const P kSeq[] = {{300, 0x1000}, {420, 0x8000}, {480, 0x8000}, {560, 0x1000},
                                          {700, 0x1000}, {900, 0x1000}, {1100, 0x1000}, {1300, 0x8000}};
                 for (const auto& p : kSeq) if (f >= p.at && f < p.at + 10) return p.b;
                 return 0;
             };
             launch(first_rom());
             settings_.limit_speed = false;
             wait_frame = 2380;
         }, 10, nullptr},
        {[this] { settings_.limit_speed = true; settings_.internal_scale = 4; }, 60, "59_scene_internal_4x"},
        {[this] { open_tool(DebugTool::Resolution); }, 30, "59a_scene_resolution_tool"},
        {[this] { dbg_.open[static_cast<int>(DebugTool::Resolution)] = false; open_settings(SettingsPage::Graphics); }, 30, "59b_scene_settings_graphics"},
        {[this] { settings_open_ = false; settings_.internal_scale = 1; open_tool(DebugTool::ObjectViewer); }, 40, "60_scene_object_viewer"},
        {[this] {
             const SceneObject* best = nullptr;
             for (const auto& o : dbg_.objects)
                 if (!o.reference && o.triangles > 20 && (!best || o.distance < best->distance)) best = &o;
             if (best) select_object(*best, true);
         }, 40, "61_scene_inspector"},
        {[this] { dbg_.cam.yaw += 2.0f; dbg_.cam.pitch = 0.6f; }, 30, "62_scene_inspector_rotated"},
        {[this] {
             for (const auto& o : dbg_.objects) if (o.reference) { select_object(o, true); break; }
         }, 40, "63_scene_level_mesh"},
        {[this] {
             dbg_.open[static_cast<int>(DebugTool::ObjectInspector)] = false;
             dbg_.open[static_cast<int>(DebugTool::ObjectViewer)] = false;
             const SceneObject* best = nullptr;
             for (const auto& o : dbg_.objects)
                 if (!o.reference && o.triangles > 20 && (!best || o.distance < best->distance)) best = &o;
             if (best) { dbg_.selected = {best->vtx_addr, best->instance, true}; dbg_.player = dbg_.selected; }
             open_tool(DebugTool::PlayerViewer);
         }, 120, "64_scene_player_viewer"},
        {[this] { stop_now(); test_input_ = nullptr; }, 20, nullptr},
    };
    const std::vector<Step>& active_steps = scene_mode ? scene_steps : steps;
    if (wait_frame && core_.loaded() && core_.stats().frame < wait_frame) return;
    wait_frame = 0;

    if (ui_test_step_ >= static_cast<int>(active_steps.size())) {
        running_ = false;
        return;
    }
    const Step& st = active_steps[ui_test_step_];
    if (ui_test_wait_ == 0) {
        st.action();
        ui_test_wait_ = st.wait + 1;
        return;
    }
    if (--ui_test_wait_ == 1) {
        if (st.shot) ui_test_capture(st.shot);
        ui_test_wait_ = 0;
        ui_test_step_++;
    }
}

} // namespace ui
