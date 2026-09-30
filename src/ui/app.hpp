#pragma once
// Orbit64 desktop frontend: window, main loop, views and actions.

#include "boxart.hpp"
#include "debug_state.hpp"
#include "emu_core.hpp"
#include "file_browser.hpp"
#include "input.hpp"
#include "library.hpp"
#include "settings.hpp"
#include "widgets.hpp"
#include "../gpu/device.hpp"

#include <SDL3/SDL.h>
#include <array>
#include <atomic>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ui {

constexpr const char* kAppName = "Orbit64";
constexpr const char* kAppVersion = "1.0.0";

enum class View { Library, Game };
enum class LibraryFilter { All, Favorites, Recent, Homebrew, RegionUSA, RegionEUR, RegionJPN, Folder };
enum class SettingsPage { General, Graphics, Audio, Controller, Emulation, Library, Shortcuts, Count };


class App {
public:
    int run(const std::string& initial_rom);
    // Scripted walkthrough of every screen that saves renderer captures to
    // `dir` (used for automated UI verification on any platform).
    void enable_ui_test(const std::string& dir) { ui_test_dir_ = dir; }

private:
    // ---- lifecycle (app.cpp)
    bool init();
    void shutdown();
    void main_loop();
    void process_event(const SDL_Event& e);
    bool handle_shortcut(const SDL_KeyboardEvent& k);
    bool run_hotkey(Hotkey h, bool modal_open);
    void capture_hotkey(const SDL_KeyboardEvent& k);
    int hotkey_capture_ = -1; // hotkey * kHotkeySlots + slot being rebound, -1 = none
    bool create_renderer();
    void destroy_renderer();
    void pump_video();
    void update_ui_scale(bool force);
    void update_input();
    void update_game_texture();
    // The frontend texture showing a frame the GPU renderer left in video
    // memory (nullptr if it can't be shown directly).
    SDL_Texture* gpu_frame_texture(const std::shared_ptr<GpuImage>& image);
    void release_gpu_frames();
    void apply_hires_renderer();
    void update_audio_volume();
    void open_audio();
    void close_audio();
    void save_settings();
    void apply_window_settings();
    // Shortcut labels for menus and hints: the first binding ("" when unbound),
    // or every binding joined with "or".
    std::string hotkey_label(Hotkey h) const;
    std::string hotkey_hint(Hotkey h) const;

    // ---- actions (app.cpp)
    void action_open_rom();
    void action_add_folder();
    // Transfer Pak: picks the Game Boy ROM for a port.
    void action_choose_gb_rom(int port);
    void on_gb_rom_picked(const std::filesystem::path& rom);
    void launch(const std::filesystem::path& rom);
    void request_stop(std::function<void()> then = nullptr);
    void stop_now();
    void toggle_pause();
    void reset_game();
    void toggle_fullscreen();
    void set_fullscreen(bool on);
    void take_screenshot();
    void save_thumbnail();
    void request_quit();
    void confirm(const std::string& title, const std::string& message, const std::string& ok_label, bool danger,
                 std::function<void()> on_ok);
    void open_settings(SettingsPage page);
    SDL_Texture* thumbnail_texture(const std::string& key);
    void invalidate_thumbnail(const std::string& key);
    // Card image: box art until the game has been played, then its screenshot.
    SDL_Texture* card_texture(const GameEntry& g);
    SDL_Texture* boxart_texture(const RomInfo& rom) { return images_.get(rom.boxart); }
    void rescan_library();
    void action_choose_boxart_folder();
    void on_folder_picked(const std::filesystem::path& dir);

    // ---- save states (app_states.cpp)
    static constexpr int kStateSlots = 9;
    std::filesystem::path states_dir() const; // the running game's folder
    std::filesystem::path state_path(int slot) const;
    void save_state(int slot);
    void load_state(int slot);
    void undo_load_state();
    void select_state_slot(int slot);
    void poll_state_events();
    void refresh_state_slots();
    void clear_state_slots();
    void state_slot_tooltip(int slot);
    void draw_state_menu();
    void reveal_states_folder();

    // ---- chrome (app_chrome.cpp)
    void draw_root();
    void draw_menubar();
    void draw_toolbar(ImVec2 pos, ImVec2 size);
    void draw_statusbar(ImVec2 pos, ImVec2 size);

    // ---- views
    void draw_library(ImVec2 pos, ImVec2 size);          // app_library.cpp
    void draw_library_sidebar(float height);
    void draw_library_header(float width);
    void draw_continue_row(const std::vector<const GameEntry*>& recent, float width);
    void draw_game_grid(const std::vector<const GameEntry*>& games, float width);
    void draw_game_list(const std::vector<const GameEntry*>& games);
    void draw_library_empty(ImVec2 size);
    void game_context_menu(const GameEntry& g);
    std::vector<const GameEntry*> filtered_games();

    void draw_game_view(ImVec2 pos, ImVec2 size);        // app_game.cpp
    void draw_no_game(ImVec2 pos, ImVec2 size);
    void draw_pause_overlay(ImVec2 mn, ImVec2 mx);
    void draw_info_panel(ImVec2 pos, ImVec2 size);

    void draw_settings();                                // app_settings.cpp
    void settings_general();
    void settings_graphics();
    void settings_audio();
    void settings_controller();
    void controller_profile_row(PortConfig& pc, float control_w);
    void scan_controller_profiles();
    void settings_emulation();
    void settings_library();
    void settings_shortcuts();
    void draw_controller_diagram(ImVec2 pos, float width, const ControllerSnapshot& s);

    void draw_dialogs();                                 // app_dialogs.cpp
    void draw_about();
    void draw_confirm();
    void draw_properties();
    void draw_native_wait();
    void draw_error();
    // File browser + "waiting for system dialog" popup. When Settings is open
    // they are drawn inside it (stacked modals); otherwise at the top level.
    void draw_picker_dialogs();
    bool picker_drawn_ = false;

    // ---- DEBUG / MEMORY tools
    void draw_debug_menu(bool icons);                    // app_debug.cpp
    void draw_debug_windows();
    void debug_update();
    void open_tool(DebugTool t);
    bool begin_tool(DebugTool t, ImVec2 default_size);
    void end_tool();
    void tool_registers();
    void tool_frame_control();
    void tool_resolution();
    void tool_object_viewer();                           // app_debug_objects.cpp
    void tool_object_inspector();
    void tool_player_viewer();
    void compute_scene_objects();
    const SceneObject* find_object(const ObjectKey& key) const;
    void select_object(const SceneObject& o, bool open_inspector);
    bool export_mesh_obj(const std::filesystem::path& obj_path); // OBJ + MTL + BMP textures
    void tool_memory_search();                           // app_debug_memory.cpp
    void tool_ram_watch();
    void tool_memory_editor();
    void tool_freeze_list();
    void add_watch(const std::string& name, std::uint32_t addr, MemType type);
    void add_freeze(const std::string& name, std::uint32_t addr, MemType type, const std::string& value);
    void open_in_editor(std::uint32_t addr, MemType type);
    void load_debug_lists();
    void save_debug_lists();
    // Screen resolution: the game image is resampled into a render target of
    // the chosen size before being presented.
    void update_scaled_texture();
    void internal_resolution_menu();
    SDL_Texture* display_texture() { return scaled_tex_ ? scaled_tex_ : game_tex_; }
    void output_resolution(int& w, int& h) const;

    void ui_test_tick();                                 // app_uitest.cpp
    void ui_audit_tick();                                // ui_audit.cpp
    void ui_test_capture(const std::string& name);

    // ---- state
    SDL_Window* window_ = nullptr;
    SDL_GPUDevice* gpu_ = nullptr; // the GPU renderer's device (nullptr: another SDL renderer)
    SDL_Renderer* renderer_ = nullptr;
    SDL_Texture* game_tex_ = nullptr;
    int game_tex_w_ = 0, game_tex_h_ = 0;
    int game_scale_ = 1; // internal resolution the frame was rendered at
    // The game's frame buffer size (the frame texture is game_scale_ times larger).
    int frame_native_w() const { return game_tex_w_ > 0 ? game_tex_w_ / game_scale_ : 320; }
    int frame_native_h() const { return game_tex_h_ > 0 ? game_tex_h_ / game_scale_ : 240; }
    int game_tex_filter_ = -1;
    VideoFrame frame_;                  // what game_tex_ shows
    SDL_Texture* stream_tex_ = nullptr; // frames that come as pixels
    int stream_w_ = 0, stream_h_ = 0;
    // Frontend textures wrapping the GPU renderer's frame images.
    struct WrappedImage {
        std::weak_ptr<GpuImage> image;
        SDL_Texture* tex;
    };
    std::vector<WrappedImage> wrapped_;
    std::shared_ptr<gpu::Device> gpu_dev_; // the GPU renderer's pipelines
    std::uint64_t frame_serial_ = 0;
    bool has_frame_ = false;
    std::string renderer_name_;

    Settings settings_;
    std::string settings_path_;
    Library library_;
    EmuCore core_;
    InputManager input_;
    FileBrowser browser_;

    bool running_ = true;
    View view_ = View::Library;
    bool fullscreen_ = false;
    bool window_focused_ = true;
    bool auto_paused_ = false;
    float fb_scale_ = 1.0f;
    float applied_scale_ = 0.0f;
    float applied_fb_scale_ = 0.0f;
    int applied_accent_ = -1;
    double last_mouse_move_ = 0.0;
    float chrome_reveal_ = 1.0f; // fullscreen auto-hide animation

    // current game session
    std::string current_key_;
    RomInfo current_rom_;
    double session_start_ = 0.0;
    double session_accum_ = 0.0;  // seconds of emulation in this session (excl. pause)
    double last_thumb_time_ = 0.0;
    std::deque<float> fps_history_;
    double fps_sample_time_ = 0.0;

    // save state slots of the running game (index = slot number, 0 unused)
    struct StateSlot {
        bool exists = false;
        std::filesystem::file_time_type mtime{};
        std::string when; // when it was saved, local time
        SDL_Texture* thumb = nullptr;
    };
    std::array<StateSlot, kStateSlots + 1> state_slots_{};
    int state_slot_ = 1;
    double state_slots_checked_ = -1.0;

    // audio
    SDL_AudioStream* audio_dev_ = nullptr; // bound to its own playback device
    int audio_freq_ = 0;  // rate the device actually opened at
    std::atomic<float> audio_volume_{1.0f};
    std::vector<std::string> audio_devices_;

    // library UI
    LibraryFilter lib_filter_ = LibraryFilter::All;
    std::string lib_folder_;
    char search_[128] = {};
    std::string selected_key_;
    bool rescan_pending_ = false;
    bool library_sidebar_ = true;

    // dialogs
    bool settings_open_ = false;
    SettingsPage settings_page_ = SettingsPage::General;
    int settings_port_ = 0;
    // Controller profiles (config_dir()/controller_profiles/<name>.ini),
    // rescanned whenever Settings opens.
    struct ControllerProfile {
        std::string name;
        PortConfig cfg;
    };
    std::vector<ControllerProfile> profiles_;
    bool profiles_scanned_ = false;
    char profile_name_[64] = {};
    std::string profile_to_delete_;
    int gb_pick_port_ = 0; // port whose Game Boy ROM the file picker chooses
    int ui_test_pak_ = 3; // UI walkthrough: port 2's accessory while it shows the Transfer Pak
    bool about_open_ = false;
    bool props_open_ = false;
    std::string props_key_;
    bool confirm_open_ = false;
    std::string confirm_title_, confirm_msg_, confirm_ok_;
    bool confirm_danger_ = false;
    std::function<void()> confirm_cb_;
    bool error_open_ = false;
    std::string error_title_, error_msg_;
    bool pause_before_modal_ = false;

    // native dialog running on a worker thread
    std::optional<std::future<std::optional<std::filesystem::path>>> native_job_;
    FileBrowser::Mode native_mode_ = FileBrowser::Mode::OpenRom;

    // Launch requests from inside library drawing are deferred to the end of
    // the frame so library entries aren't mutated while being iterated.
    std::filesystem::path pending_launch_;

    DebugState dbg_;
    // Debugger texture cache: decoded N64 textures keyed by content hash.
    std::unordered_map<std::uint64_t, SDL_Texture*> dbg_textures_;
    SDL_Texture* debug_texture(const CapturedTexture& t);
    SDL_Texture* scaled_tex_ = nullptr;
    int scaled_w_ = 0, scaled_h_ = 0;
    bool scaled_dirty_ = true;

    std::string ui_test_dir_;
    // UI test only: scripted controller buttons for port 1, keyed by emulated frame.
    std::function<std::uint16_t(std::uint64_t)> test_input_;
    int ui_test_step_ = 0;
    int ui_test_wait_ = 0;

    enum class FolderPurpose { RomDir, BoxArt } folder_purpose_ = FolderPurpose::RomDir;
    ImageCache images_;

    std::unordered_map<std::string, SDL_Texture*> thumbs_;
    std::unordered_map<std::string, bool> thumb_missing_;
};

} // namespace ui
