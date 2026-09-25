#pragma once
// Game library: ROM discovery (background scan), recently played list,
// favourites and play-time statistics.

#include "boxart.hpp"
#include "rom_info.hpp"

#include <atomic>
#include <array>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ui {

struct GameStats {
    bool favorite = false;
    std::int64_t last_played = 0;   // unix time, 0 = never
    std::int64_t play_seconds = 0;
    int launches = 0;
};

struct GameEntry {
    RomInfo rom;
    GameStats stats;
    std::string key; // stable identity: UTF-8 absolute path
};

class Library {
public:
    ~Library();

    void load(const std::filesystem::path& state_file);
    void save() const;

    // Starts a background rescan of `dirs`. Entries become visible once the
    // scan finishes; `scanning()` / `scan_progress()` drive the UI spinner.
    // `boxart_dir` overrides box art auto-detection (empty = look inside the ROM folders).
    void rescan(const std::vector<std::string>& dirs, bool recursive, const std::string& boxart_dir = {});
    bool scanning() const { return scanning_.load(); }
    int scan_found() const { return scan_found_.load(); }
    // Must be called once per frame from the UI thread to publish scan results.
    bool poll();

    const std::vector<GameEntry>& games() const { return games_; }
    GameEntry* find(const std::string& key);

    // Adds a ROM opened directly (outside of library folders) so it can show
    // up in "Recently played".
    GameEntry& ensure(const std::filesystem::path& rom_path);
    void mark_launched(const std::string& key);
    void add_play_time(const std::string& key, std::int64_t seconds);
    void toggle_favorite(const std::string& key);
    void forget_recent(const std::string& key);
    void clear_recent();

    // Most recently played first.
    std::vector<const GameEntry*> recent(size_t max_count) const;

    std::filesystem::path thumbnail_path(const std::string& key) const;

    // Box art lookup for arbitrary ROM files (e.g. the ROM browser preview).
    std::filesystem::path boxart_for(const std::filesystem::path& rom, const std::string& internal_name) const;
    size_t boxart_count() const { return boxart_.size(); }
    const std::vector<std::filesystem::path>& boxart_dirs() const { return boxart_dirs_; }

private:
    std::filesystem::path state_file_;
    std::vector<GameEntry> games_;
    std::unordered_map<std::string, GameStats> stats_;
    std::vector<std::string> extra_paths_; // ROMs opened directly

    std::thread worker_;
    std::atomic<bool> scanning_{false};
    std::atomic<bool> cancel_{false};
    std::atomic<int> scan_found_{0};
    std::mutex result_mutex_;
    bool result_ready_ = false;
    std::vector<RomInfo> result_;
    BoxArtIndex result_boxart_;
    std::vector<std::filesystem::path> result_boxart_dirs_;
    BoxArtIndex boxart_;
    std::vector<std::filesystem::path> boxart_dirs_;
};

std::string key_for(const std::filesystem::path& p);

} // namespace ui
