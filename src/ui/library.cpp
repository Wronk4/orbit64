#include "library.hpp"
#include "platform.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>

namespace ui {

namespace fs = std::filesystem;

std::string key_for(const fs::path& p) {
    std::error_code ec;
    fs::path abs = fs::weakly_canonical(p, ec);
    if (ec) abs = fs::absolute(p, ec);
    return platform::path_to_utf8(abs);
}

static std::int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

Library::~Library() {
    cancel_ = true;
    if (worker_.joinable()) worker_.join();
}

// State file format (tab separated, one game per line):
//   G <tab> favorite <tab> last_played <tab> play_seconds <tab> launches <tab> path
//   X <tab> path          (ROM opened directly, outside library folders)
void Library::load(const fs::path& state_file) {
    state_file_ = state_file;
    std::ifstream f(state_file, std::ios::binary);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<std::string> parts;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, '\t')) parts.push_back(item);
        if (parts.size() == 6 && parts[0] == "G") {
            GameStats s;
            s.favorite = parts[1] == "1";
            s.last_played = std::strtoll(parts[2].c_str(), nullptr, 10);
            s.play_seconds = std::strtoll(parts[3].c_str(), nullptr, 10);
            s.launches = std::atoi(parts[4].c_str());
            stats_[parts[5]] = s;
        } else if (parts.size() == 2 && parts[0] == "X") {
            extra_paths_.push_back(parts[1]);
        }
    }
}

void Library::save() const {
    if (state_file_.empty()) return;
    std::ofstream f(state_file_, std::ios::binary);
    for (const auto& [key, s] : stats_) {
        if (!s.favorite && s.last_played == 0 && s.play_seconds == 0) continue;
        f << "G\t" << (s.favorite ? 1 : 0) << "\t" << s.last_played << "\t" << s.play_seconds << "\t" << s.launches
          << "\t" << key << "\n";
    }
    for (const auto& x : extra_paths_) f << "X\t" << x << "\n";
}

void Library::rescan(const std::vector<std::string>& dirs, bool recursive, const std::string& boxart_dir) {
    cancel_ = true;
    if (worker_.joinable()) worker_.join();
    cancel_ = false;
    scanning_ = true;
    scan_found_ = 0;
    std::vector<std::string> extras = extra_paths_;

    worker_ = std::thread([this, dirs, recursive, extras, boxart_dir] {
        std::vector<RomInfo> found;
        std::set<std::string> seen;
        auto consider = [&](const fs::path& p) {
            if (!is_rom_extension(p)) return;
            std::string k = key_for(p);
            if (!seen.insert(k).second) return;
            RomInfo info = inspect_rom(p);
            if (!info.valid) return;
            found.push_back(std::move(info));
            scan_found_ = static_cast<int>(found.size());
        };
        for (const auto& d : dirs) {
            std::error_code ec;
            fs::path root = platform::utf8_to_path(d);
            if (!fs::is_directory(root, ec)) continue;
            auto opts = fs::directory_options::skip_permission_denied;
            if (recursive) {
                fs::recursive_directory_iterator it(root, opts, ec), end;
                for (; it != end && !cancel_; it.increment(ec)) {
                    if (ec) break;
                    // Don't descend into huge unrelated trees (build dirs, VCS data).
                    if (it.depth() > 6) { it.disable_recursion_pending(); continue; }
                    std::string name = platform::path_to_utf8(it->path().filename());
                    if (it->is_directory(ec) && (!name.empty() && name[0] == '.')) { it.disable_recursion_pending(); continue; }
                    if (it->is_regular_file(ec)) consider(it->path());
                }
            } else {
                for (const auto& e : fs::directory_iterator(root, opts, ec)) {
                    if (cancel_) break;
                    if (e.is_regular_file(ec)) consider(e.path());
                }
            }
        }
        for (const auto& x : extras) {
            std::error_code ec;
            fs::path p = platform::utf8_to_path(x);
            if (fs::is_regular_file(p, ec)) consider(p);
        }
        // Box art: explicit folder from settings, or auto-detected inside the ROM folders.
        std::vector<fs::path> art_dirs;
        if (!boxart_dir.empty()) {
            art_dirs.push_back(platform::utf8_to_path(boxart_dir));
        } else {
            std::vector<fs::path> roots;
            for (const auto& d : dirs) roots.push_back(platform::utf8_to_path(d));
            art_dirs = BoxArtIndex::discover(roots);
        }
        BoxArtIndex index;
        index.build(art_dirs);
        for (auto& r : found) r.boxart = index.find(r.path, r.internal_name);

        std::lock_guard<std::mutex> lk(result_mutex_);
        result_boxart_ = std::move(index);
        result_boxart_dirs_ = std::move(art_dirs);
        result_ = std::move(found);
        result_ready_ = true;
        scanning_ = false;
    });
}

bool Library::poll() {
    std::lock_guard<std::mutex> lk(result_mutex_);
    if (!result_ready_) return false;
    result_ready_ = false;
    boxart_ = std::move(result_boxart_);
    boxart_dirs_ = std::move(result_boxart_dirs_);
    games_.clear();
    for (auto& r : result_) {
        GameEntry e;
        e.key = key_for(r.path);
        e.rom = std::move(r);
        auto it = stats_.find(e.key);
        if (it != stats_.end()) e.stats = it->second;
        games_.push_back(std::move(e));
    }
    result_.clear();
    std::sort(games_.begin(), games_.end(),
              [](const GameEntry& a, const GameEntry& b) { return a.rom.display_title < b.rom.display_title; });
    return true;
}

GameEntry* Library::find(const std::string& key) {
    for (auto& g : games_)
        if (g.key == key) return &g;
    return nullptr;
}

GameEntry& Library::ensure(const fs::path& rom_path) {
    std::string k = key_for(rom_path);
    if (GameEntry* g = find(k)) return *g;
    GameEntry e;
    e.key = k;
    e.rom = inspect_rom(rom_path);
    e.rom.boxart = boxart_.find(rom_path, e.rom.internal_name);
    auto it = stats_.find(k);
    if (it != stats_.end()) e.stats = it->second;
    if (std::find(extra_paths_.begin(), extra_paths_.end(), k) == extra_paths_.end()) extra_paths_.push_back(k);
    games_.push_back(std::move(e));
    return games_.back();
}

void Library::mark_launched(const std::string& key) {
    GameStats& s = stats_[key];
    s.last_played = now_unix();
    s.launches++;
    if (GameEntry* g = find(key)) g->stats = s;
    save();
}

void Library::add_play_time(const std::string& key, std::int64_t seconds) {
    if (seconds <= 0) return;
    GameStats& s = stats_[key];
    s.play_seconds += seconds;
    s.last_played = now_unix();
    if (GameEntry* g = find(key)) g->stats = s;
    save();
}

void Library::toggle_favorite(const std::string& key) {
    GameStats& s = stats_[key];
    s.favorite = !s.favorite;
    if (GameEntry* g = find(key)) g->stats = s;
    save();
}

void Library::forget_recent(const std::string& key) {
    GameStats& s = stats_[key];
    s.last_played = 0;
    if (GameEntry* g = find(key)) g->stats = s;
    save();
}

void Library::clear_recent() {
    for (auto& [k, s] : stats_) s.last_played = 0;
    for (auto& g : games_) g.stats.last_played = 0;
    save();
}

std::vector<const GameEntry*> Library::recent(size_t max_count) const {
    std::vector<const GameEntry*> out;
    for (const auto& g : games_)
        if (g.stats.last_played > 0) out.push_back(&g);
    std::sort(out.begin(), out.end(),
              [](const GameEntry* a, const GameEntry* b) { return a->stats.last_played > b->stats.last_played; });
    if (out.size() > max_count) out.resize(max_count);
    return out;
}

fs::path Library::boxart_for(const fs::path& rom, const std::string& internal_name) const {
    return boxart_.find(rom, internal_name);
}

fs::path Library::thumbnail_path(const std::string& key) const {
    // FNV-1a of the path: stable across compilers and platforms (std::hash is not).
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : key) { h ^= c; h *= 1099511628211ull; }
    char name[40];
    std::snprintf(name, sizeof name, "%016llx.bmp", static_cast<unsigned long long>(h));
    fs::path dir = platform::config_dir() / "thumbnails";
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir / name;
}

} // namespace ui
