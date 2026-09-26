#include "platform.hpp"
#include "icons.hpp"

#include <SDL3/SDL.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <thread>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <commdlg.h>
#  include <shellapi.h>
#  include <shobjidl.h>
#else
#  include <spawn.h>
#  include <sys/wait.h>
#  include <unistd.h>
extern char** environ;
#endif

namespace platform {

void sleep_until_precise(std::chrono::steady_clock::time_point t) {
    using namespace std::chrono;
#if defined(_WIN32)
#  ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#    define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#  endif
    struct Timer {
        HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        ~Timer() { if (h) CloseHandle(h); }
    };
    thread_local Timer timer;
    constexpr auto spin = microseconds(600);
    for (;;) {
        const auto left = t - steady_clock::now();
        if (left <= steady_clock::duration::zero()) return;
        if (left > spin && timer.h) {
            LARGE_INTEGER due;
            due.QuadPart = -static_cast<LONGLONG>(duration_cast<nanoseconds>(left - spin).count() / 100);
            if (SetWaitableTimerEx(timer.h, &due, 0, nullptr, nullptr, nullptr, 0)) {
                WaitForSingleObject(timer.h, INFINITE);
                continue;
            }
        }
        if (left > spin) std::this_thread::sleep_for(left - spin);
        else SwitchToThread();
    }
#else
    std::this_thread::sleep_until(t);
#endif
}

OS current_os() {
#if defined(_WIN32)
    return OS::Windows;
#elif defined(__APPLE__)
    return OS::MacOS;
#elif defined(__linux__) || defined(__FreeBSD__)
    return OS::Linux;
#else
    return OS::Other;
#endif
}

const char* os_display_name() {
    switch (current_os()) {
        case OS::Windows: return "Windows";
        case OS::MacOS: return "macOS";
        case OS::Linux: return "Linux";
        default: return SDL_GetPlatform();
    }
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

std::string path_to_utf8(const fs::path& p) {
    auto u8 = p.u8string();
    return std::string(u8.begin(), u8.end());
}

fs::path utf8_to_path(const std::string& s) {
    return fs::path(std::u8string(s.begin(), s.end()));
}

static fs::path env_path(const char* name) {
#if defined(_WIN32)
    // _wgetenv keeps non-ASCII user names intact.
    std::wstring wname(name, name + std::strlen(name));
    if (const wchar_t* v = _wgetenv(wname.c_str())) return fs::path(v);
    return {};
#else
    if (const char* v = std::getenv(name)) return fs::path(v);
    return {};
#endif
}

fs::path home_dir() {
#if defined(_WIN32)
    fs::path p = env_path("USERPROFILE");
    if (p.empty()) p = env_path("HOMEDRIVE").string() + env_path("HOMEPATH").string();
    return p;
#else
    return env_path("HOME");
#endif
}

fs::path config_dir() {
    static fs::path cached;
    if (!cached.empty()) return cached;
    // ORBIT64_CONFIG_DIR allows portable installs (and isolated test runs).
    if (fs::path over = env_path("ORBIT64_CONFIG_DIR"); !over.empty()) {
        cached = over;
    } else if (char* pref = SDL_GetPrefPath("", "Orbit64")) {
        cached = utf8_to_path(pref);
        SDL_free(pref);
    } else {
        cached = home_dir() / ".orbit64";
    }
    std::error_code ec;
    fs::create_directories(cached, ec);
    return cached;
}

static fs::path xdg_user_dir(const char* key, const char* fallback) {
    // Honour ~/.config/user-dirs.dirs on Linux (localised folder names).
    fs::path cfg = env_path("XDG_CONFIG_HOME");
    if (cfg.empty()) cfg = home_dir() / ".config";
    if (FILE* f = std::fopen((cfg / "user-dirs.dirs").string().c_str(), "r")) {
        char line[512];
        std::string prefix = std::string(key) + "=\"";
        while (std::fgets(line, sizeof(line), f)) {
            std::string s(line);
            if (s.rfind(prefix, 0) == 0) {
                std::string v = s.substr(prefix.size());
                v = v.substr(0, v.find('"'));
                if (v.rfind("$HOME", 0) == 0) v = path_to_utf8(home_dir()) + v.substr(5);
                std::fclose(f);
                return utf8_to_path(v);
            }
        }
        std::fclose(f);
    }
    return home_dir() / fallback;
}

static fs::path user_folder(const char* xdg_key, const char* name) {
    if (current_os() == OS::Linux) return xdg_user_dir(xdg_key, name);
    return home_dir() / name;
}

fs::path screenshots_dir() {
    std::error_code ec;
    fs::path pictures = user_folder("XDG_PICTURES_DIR", "Pictures");
    fs::path dir = fs::is_directory(pictures, ec) ? pictures / "Orbit64" : config_dir() / "screenshots";
    fs::create_directories(dir, ec);
    return dir;
}

std::vector<Place> quick_places() {
    std::vector<Place> out;
    std::error_code ec;
    auto add = [&](const std::string& label, const fs::path& p, ui::Icon icon) {
        if (!p.empty() && fs::is_directory(p, ec)) out.push_back({label, p, static_cast<int>(icon)});
    };
    add("Home", home_dir(), ui::Icon::Home);
    add("Desktop", user_folder("XDG_DESKTOP_DIR", "Desktop"), ui::Icon::Monitor);
    add("Documents", user_folder("XDG_DOCUMENTS_DIR", "Documents"), ui::Icon::File);
    add("Downloads", user_folder("XDG_DOWNLOAD_DIR", "Downloads"), ui::Icon::Download);

#if defined(_WIN32)
    wchar_t buf[512];
    DWORD len = GetLogicalDriveStringsW(511, buf);
    for (wchar_t* d = buf; d < buf + len && *d; d += wcslen(d) + 1) {
        fs::path drive(d);
        std::string label = path_to_utf8(drive);
        if (!label.empty() && (label.back() == '\\' || label.back() == '/')) label.pop_back();
        out.push_back({label, drive, static_cast<int>(ui::Icon::Drive)});
    }
#else
    out.push_back({"File System", fs::path("/"), static_cast<int>(ui::Icon::Drive)});
    std::vector<fs::path> mount_roots;
    if (current_os() == OS::MacOS) {
        mount_roots = {"/Volumes"};
    } else {
        mount_roots = {"/media", "/mnt"};
        if (const char* user = std::getenv("USER")) mount_roots.push_back(fs::path("/run/media") / user);
    }
    for (const auto& root : mount_roots) {
        if (!fs::is_directory(root, ec)) continue;
        for (const auto& e : fs::directory_iterator(root, fs::directory_options::skip_permission_denied, ec)) {
            if (!e.is_directory(ec)) continue;
            // macOS lists the boot volume under /Volumes too; skip that symlink to "/".
            if (fs::is_symlink(e.path(), ec) && fs::equivalent(e.path(), "/", ec)) continue;
            out.push_back({path_to_utf8(e.path().filename()), e.path(), static_cast<int>(ui::Icon::Drive)});
        }
    }
#endif
    return out;
}

// ---------------------------------------------------------------------------
// Process helpers (POSIX)
// ---------------------------------------------------------------------------

#if !defined(_WIN32)
// Spawns a detached helper process without going through a shell, so paths
// with spaces/quotes need no escaping. The child is reaped on a background
// thread to avoid zombies.
static bool spawn_detached(const std::vector<std::string>& args) {
    std::vector<char*> argv;
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    pid_t pid = 0;
    if (posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), environ) != 0) return false;
    std::thread([pid] { int st = 0; waitpid(pid, &st, 0); }).detach();
    return true;
}

static std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out + "'";
}

[[maybe_unused]] static bool have_program(const char* name) {
    std::string cmd = std::string("command -v ") + name + " >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

// Runs a command and captures the first line of stdout. Returns false if the
// process failed (non-zero exit, e.g. user pressed Cancel).
static bool run_capture(const std::string& cmd, std::string& out) {
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return false;
    char buf[4096];
    out.clear();
    while (std::fgets(buf, sizeof(buf), p)) out += buf;
    int rc = pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return rc == 0;
}
#endif

#if defined(_WIN32)
static std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
    return w;
}
#endif

void reveal_in_file_manager(const fs::path& p) {
#if defined(_WIN32)
    std::wstring args = L"/select,\"" + p.wstring() + L"\"";
    ShellExecuteW(nullptr, L"open", L"explorer.exe", args.c_str(), nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
    spawn_detached({"open", "-R", path_to_utf8(p)});
#else
    std::error_code ec;
    fs::path dir = fs::is_directory(p, ec) ? p : p.parent_path();
    spawn_detached({"xdg-open", path_to_utf8(dir)});
#endif
}

const char* reveal_action_label() {
    switch (current_os()) {
        case OS::Windows: return "Show in Explorer";
        case OS::MacOS: return "Show in Finder";
        default: return "Open Containing Folder";
    }
}

void open_url(const std::string& url) {
#if defined(_WIN32)
    ShellExecuteW(nullptr, L"open", widen(url).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
    spawn_detached({"open", url});
#else
    spawn_detached({"xdg-open", url});
#endif
}

// ---------------------------------------------------------------------------
// Native dialogs
// ---------------------------------------------------------------------------

bool native_dialogs_available() {
#if defined(_WIN32) || defined(__APPLE__)
    return true;
#else
    static int cached = -1;
    if (cached < 0) cached = (have_program("zenity") || have_program("kdialog")) ? 1 : 0;
    return cached == 1;
#endif
}

std::optional<fs::path> native_open_file(const std::string& title, const std::vector<std::string>& extensions) {
#if defined(_WIN32)
    // Filter string: "N64 ROMs\0*.z64;*.n64\0All files\0*.*\0\0"
    std::wstring pattern;
    for (const auto& e : extensions) {
        if (!pattern.empty()) pattern += L";";
        pattern += L"*." + widen(e);
    }
    std::wstring filter = L"Nintendo 64 ROMs";
    filter.push_back(L'\0');
    filter += pattern;
    filter.push_back(L'\0');
    filter += L"All files";
    filter.push_back(L'\0');
    filter += L"*.*";
    filter.push_back(L'\0');
    filter.push_back(L'\0');

    wchar_t file[MAX_PATH * 4] = L"";
    std::wstring wtitle = widen(title);
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = filter.c_str();
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH * 4;
    ofn.lpstrTitle = wtitle.c_str();
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if (GetOpenFileNameW(&ofn)) return fs::path(file);
    return fs::path(); // cancelled
#elif defined(__APPLE__)
    (void)extensions; // ROM extensions have no registered UTI; validation happens on load.
    std::string script = "set f to choose file with prompt " + std::string("\"") + title + "\"";
    std::string cmd = "osascript -e " + shell_quote(script) + " -e 'POSIX path of f' 2>/dev/null";
    std::string out;
    if (run_capture(cmd, out) && !out.empty()) return utf8_to_path(out);
    return fs::path();
#else
    std::string out;
    std::string pattern;
    for (const auto& e : extensions) pattern += (pattern.empty() ? "*." : " *.") + e;
    if (have_program("zenity")) {
        std::string cmd = "zenity --file-selection --title=" + shell_quote(title) +
                          " --file-filter=" + shell_quote("Nintendo 64 ROMs | " + pattern) +
                          " --file-filter=" + shell_quote("All files | *") + " 2>/dev/null";
        if (run_capture(cmd, out) && !out.empty()) return utf8_to_path(out);
        return fs::path();
    }
    if (have_program("kdialog")) {
        std::string cmd = "kdialog --title " + shell_quote(title) + " --getopenfilename " +
                          shell_quote(path_to_utf8(home_dir())) + " " +
                          shell_quote(pattern + "|Nintendo 64 ROMs") + " 2>/dev/null";
        if (run_capture(cmd, out) && !out.empty()) return utf8_to_path(out);
        return fs::path();
    }
    return std::nullopt;
#endif
}

std::optional<fs::path> native_save_file(const std::string& title,
                                          const std::string& default_name,
                                          const std::vector<std::string>& extensions) {
#if defined(_WIN32)
    std::wstring pattern;
    for (const auto& e : extensions) {
        if (!pattern.empty()) pattern += L";";
        pattern += L"*." + widen(e);
    }
    std::wstring filter = L"3D Model";
    filter.push_back(L'\0');
    filter += pattern;
    filter.push_back(L'\0');
    filter += L"All files";
    filter.push_back(L'\0');
    filter += L"*.*";
    filter.push_back(L'\0');
    filter.push_back(L'\0');

    std::wstring wdefault = widen(default_name);
    wchar_t file[MAX_PATH * 4];
    wcsncpy_s(file, wdefault.c_str(), MAX_PATH * 4 - 1);
    std::wstring wtitle = widen(title);
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = filter.c_str();
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH * 4;
    ofn.lpstrTitle = wtitle.c_str();
    ofn.lpstrDefExt = extensions.empty() ? nullptr : widen(extensions[0]).c_str();
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if (GetSaveFileNameW(&ofn)) return fs::path(file);
    return fs::path(); // cancelled
#elif defined(__APPLE__)
    (void)extensions;
    std::string script = "set f to choose file name with prompt \"" + title +
                         "\" default name \"" + default_name + "\"";
    std::string cmd = "osascript -e " + shell_quote(script) + " -e 'POSIX path of f' 2>/dev/null";
    std::string out;
    if (run_capture(cmd, out) && !out.empty()) return utf8_to_path(out);
    return fs::path();
#else
    std::string out;
    std::string pattern;
    for (const auto& e : extensions) pattern += (pattern.empty() ? "*." : " *.") + e;
    if (have_program("zenity")) {
        std::string cmd = "zenity --file-selection --save --confirm-overwrite --title=" +
                          shell_quote(title) + " --filename=" + shell_quote(default_name) +
                          " --file-filter=" + shell_quote("3D Model | " + pattern) +
                          " --file-filter=" + shell_quote("All files | *") + " 2>/dev/null";
        if (run_capture(cmd, out) && !out.empty()) return utf8_to_path(out);
        return fs::path();
    }
    if (have_program("kdialog")) {
        std::string cmd = "kdialog --title " + shell_quote(title) + " --getsavefilename " +
                          shell_quote(path_to_utf8(home_dir()) + "/" + default_name) + " " +
                          shell_quote(pattern + "|3D Model") + " 2>/dev/null";
        if (run_capture(cmd, out) && !out.empty()) return utf8_to_path(out);
        return fs::path();
    }
    return std::nullopt;
#endif
}

std::optional<fs::path> native_pick_folder(const std::string& title) {
#if defined(_WIN32)
    fs::path result;
    HRESULT init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    IFileOpenDialog* dlg = nullptr;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL, IID_IFileOpenDialog,
                                   reinterpret_cast<void**>(&dlg)))) {
        DWORD opts = 0;
        dlg->GetOptions(&opts);
        dlg->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
        std::wstring wtitle = widen(title);
        dlg->SetTitle(wtitle.c_str());
        if (SUCCEEDED(dlg->Show(nullptr))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dlg->GetResult(&item))) {
                PWSTR p = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &p))) {
                    result = fs::path(p);
                    CoTaskMemFree(p);
                }
                item->Release();
            }
        }
        dlg->Release();
    }
    if (SUCCEEDED(init)) CoUninitialize();
    return result;
#elif defined(__APPLE__)
    std::string script = "set f to choose folder with prompt \"" + title + "\"";
    std::string cmd = "osascript -e " + shell_quote(script) + " -e 'POSIX path of f' 2>/dev/null";
    std::string out;
    if (run_capture(cmd, out) && !out.empty()) return utf8_to_path(out);
    return fs::path();
#else
    std::string out;
    if (have_program("zenity")) {
        std::string cmd = "zenity --file-selection --directory --title=" + shell_quote(title) + " 2>/dev/null";
        if (run_capture(cmd, out) && !out.empty()) return utf8_to_path(out);
        return fs::path();
    }
    if (have_program("kdialog")) {
        std::string cmd = "kdialog --title " + shell_quote(title) + " --getexistingdirectory " +
                          shell_quote(path_to_utf8(home_dir())) + " 2>/dev/null";
        if (run_capture(cmd, out) && !out.empty()) return utf8_to_path(out);
        return fs::path();
    }
    return std::nullopt;
#endif
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

float system_ui_scale(unsigned display_id) {
    switch (current_os()) {
        case OS::Windows: {
            const float s = SDL_GetDisplayContentScale(static_cast<SDL_DisplayID>(display_id));
            return s > 0.0f ? s : 1.0f;
        }
        case OS::MacOS: return 1.0f;
        default: {
            for (const char* var : {"ORBIT64_SCALE", "GDK_SCALE", "QT_SCALE_FACTOR"}) {
                if (const char* v = std::getenv(var)) {
                    float f = static_cast<float>(std::atof(v));
                    if (f >= 0.5f && f <= 4.0f) return f;
                }
            }
            if (const char* v = std::getenv("GDK_DPI_SCALE")) {
                float f = static_cast<float>(std::atof(v));
                if (f >= 0.5f && f <= 4.0f) return f;
            }
            // X11 reports the desktop's Xft.dpi as the content scale. (Wayland
            // scales windows through their pixel density instead.)
            if (const char* drv = SDL_GetCurrentVideoDriver(); drv && std::strcmp(drv, "x11") == 0) {
                const float s = SDL_GetDisplayContentScale(static_cast<SDL_DisplayID>(display_id));
                if (s >= 0.5f && s <= 4.0f) return s;
            }
            return 1.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Keyboard conventions
// ---------------------------------------------------------------------------

bool primary_mod_down(unsigned mod) {
    if (current_os() == OS::MacOS) return (mod & SDL_KMOD_GUI) != 0;
    return (mod & SDL_KMOD_CTRL) != 0;
}

const char* primary_mod_name() { return current_os() == OS::MacOS ? "Cmd" : "Ctrl"; }
const char* alt_mod_name() { return current_os() == OS::MacOS ? "Option" : "Alt"; }

std::string shortcut_label(bool primary, bool shift, bool alt, const char* key) {
    std::string s;
    if (primary) s += std::string(primary_mod_name()) + "+";
    if (shift) s += "Shift+";
    if (alt) s += std::string(alt_mod_name()) + "+";
    s += key;
    return s;
}

} // namespace platform
