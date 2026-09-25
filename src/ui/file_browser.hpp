#pragma once
// Built-in ROM / folder picker.
//
// Works identically on every OS through std::filesystem, so the app never
// depends on a native dialog being present (e.g. minimal Linux setups
// without zenity/kdialog). Native dialogs are offered as an alternative.

#include "imgui.h"
#include "rom_info.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

struct SDL_Texture;

namespace ui {

class FileBrowser {
public:
    enum class Mode { OpenRom, PickFolder };

    void open(Mode mode, const std::filesystem::path& start_dir);
    bool is_open() const { return open_; }
    void close() { open_ = false; }

    // Draws the modal. Returns true on the frame the user confirmed a choice.
    bool draw();
    const std::filesystem::path& result() const { return result_; }
    Mode mode() const { return mode_; }
    // Selects the entry with the given file name (keyboard/automation helper).
    void select(const std::string& file_name);

    // Invoked when the user clicks "Use system dialog" (optional).
    std::function<void(Mode)> on_native_requested;
    bool native_available = false;
    std::vector<std::filesystem::path> library_dirs; // shown in the sidebar
    // Optional box art for the preview panel (returns 0 when none / still loading).
    std::function<ImTextureID(const RomInfo&)> cover_for;

private:
    struct Entry {
        std::filesystem::path path;
        std::string name;
        bool dir = false;
        bool rom = false;
        std::uintmax_t size = 0;
        std::string modified;
    };

    void navigate(const std::filesystem::path& dir, bool push_history = true);
    void refresh();
    void draw_sidebar(float height);
    void draw_toolbar(float width);
    void draw_list(float width, float height);
    void draw_preview(float width, float height);
    bool confirm(const Entry* e);

    Mode mode_ = Mode::OpenRom;
    bool open_ = false;
    std::filesystem::path cwd_;
    std::vector<Entry> entries_;
    std::string error_;
    int selected_ = -1;
    std::vector<std::filesystem::path> back_, forward_;
    char filter_[128] = {};
    bool show_all_ = false;
    bool editing_path_ = false;
    char path_buf_[1024] = {};
    RomInfo preview_;
    std::filesystem::path result_;
    bool confirmed_ = false;
    bool scroll_to_selected_ = false;
    bool scroll_to_top_ = false;
    int pending_confirm_ = -1; // double-clicked entry, handled after the list is drawn
};

// Procedurally generated "box art" used when no screenshot thumbnail exists.
// Draws a texture filling [mn,mx] ("cover" fit: keeps aspect ratio, crops overflow).
void draw_texture_cover(ImDrawList* dl, SDL_Texture* tex, ImVec2 mn, ImVec2 mx, float rounding);

void draw_cover_art(ImDrawList* dl, ImVec2 mn, ImVec2 mx, const RomInfo& rom, float rounding, bool with_title = true);

} // namespace ui
