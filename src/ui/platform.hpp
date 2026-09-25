#pragma once
// Platform abstraction layer.
//
// Everything that differs between Windows, macOS and Linux lives behind this
// interface: well-known folders, file-manager integration, native file
// dialogs, keyboard modifier conventions (Ctrl vs Cmd) and path <-> UTF-8
// conversion. The rest of the UI only ever talks to these functions, so no
// other file needs an #ifdef.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace platform {

namespace fs = std::filesystem;

enum class OS { Windows, MacOS, Linux, Other };

OS current_os();
const char* os_display_name();

// ---- Paths ---------------------------------------------------------------
// UTF-8 safe conversions (std::filesystem uses wchar_t on Windows).
std::string path_to_utf8(const fs::path& p);
fs::path utf8_to_path(const std::string& s);

// Per-user writable directory for settings, library cache, thumbnails.
// (%APPDATA%\Orbit64, ~/Library/Application Support/Orbit64, ~/.local/share/Orbit64)
fs::path config_dir();
fs::path screenshots_dir();

struct Place {
    std::string label;
    fs::path path;
    int icon; // ui::Icon value, kept as int to avoid a UI dependency here
};
// Quick-access locations for the built-in file browser (home, desktop,
// downloads, drive letters on Windows, /Volumes on macOS, /media on Linux).
std::vector<Place> quick_places();
fs::path home_dir();

// ---- Shell integration ---------------------------------------------------
// Reveal a file in Explorer / Finder / the default Linux file manager.
void reveal_in_file_manager(const fs::path& p);
// "Show in Explorer" / "Show in Finder" / "Open Containing Folder"
const char* reveal_action_label();
void open_url(const std::string& url);

// ---- Native dialogs --------------------------------------------------------
// Returns std::nullopt if no native dialog is available on this system (the
// caller then falls back to the built-in cross-platform browser) and an
// empty optional-with-value (empty path) if the user cancelled.
//  - Windows: comdlg32 GetOpenFileNameW / IFileDialog for folders
//  - macOS:   AppleScript "choose file" via osascript (no Obj-C needed)
//  - Linux:   zenity or kdialog when installed
bool native_dialogs_available();
std::optional<fs::path> native_open_file(const std::string& title,
                                         const std::vector<std::string>& extensions);
// Returns the chosen save path, an empty path if the user cancelled, or
// nullopt if no native dialog is available on this platform.
// `default_name` is the pre-filled filename (without directory).
std::optional<fs::path> native_save_file(const std::string& title,
                                         const std::string& default_name,
                                         const std::vector<std::string>& extensions);
std::optional<fs::path> native_pick_folder(const std::string& title);

// ---- Display ---------------------------------------------------------------
// Scale factor the OS asks applications to apply to logical UI sizes.
//  - Windows: per-monitor DPI / 96 (process is per-monitor DPI aware)
//  - macOS:   1.0 (Retina is handled through the framebuffer scale instead)
//  - Linux:   GDK_SCALE / QT_SCALE_FACTOR / Xft-style DPI hints, else 1.0
float system_ui_scale(int display_index);

// ---- Keyboard conventions --------------------------------------------------
// Primary shortcut modifier: Cmd on macOS, Ctrl elsewhere.
bool primary_mod_down(unsigned sdl_keymod);
const char* primary_mod_name(); // "Cmd" / "Ctrl"
const char* alt_mod_name();     // "Option" / "Alt"
// Formats a shortcut like "Ctrl+Shift+O" or "Cmd+Shift+O".
std::string shortcut_label(bool primary, bool shift, bool alt, const char* key);

} // namespace platform
