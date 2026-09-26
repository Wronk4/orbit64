#pragma once
// Box art support.
//
// BoxArtIndex maps ROMs to cover images in a RetroArch / No-Intro style
// thumbnail folder ("Named_Boxarts/Super Mario 64 (USA).png"). Matching is
// tolerant: exact file-name stem first, then the title without tags
// ("(USA)", "(Rev 1)"), then the same set of words in any order (so
// "GoldenEye 007" finds "007 - GoldenEye").
//
// ImageCache decodes PNG/JPEG files on a worker thread (stb_image) and turns
// them into SDL textures on the UI thread, so large grids never stall a frame.

#include <SDL3/SDL.h>

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ui {

class BoxArtIndex {
public:
    // Indexes every image inside `dirs` (recursively, a few levels deep).
    void build(const std::vector<std::filesystem::path>& dirs);
    // Returns the best cover for a ROM file, or an empty path.
    std::filesystem::path find(const std::filesystem::path& rom, const std::string& internal_name) const;
    size_t size() const { return count_; }

    // Directories that look like box art folders inside `roots` (auto-detection).
    static std::vector<std::filesystem::path> discover(const std::vector<std::filesystem::path>& roots);

private:
    struct Candidate {
        std::filesystem::path path;
        std::string stem;
    };
    std::unordered_map<std::string, std::filesystem::path> by_stem_;          // lowercase full stem
    std::unordered_map<std::string, std::vector<Candidate>> by_title_;        // normalized title
    std::unordered_map<std::string, std::vector<Candidate>> by_words_;        // sorted word set
    size_t count_ = 0;
};

class ImageCache {
public:
    ImageCache();
    ~ImageCache();

    // Returns the texture if loaded; otherwise queues it and returns nullptr.
    SDL_Texture* get(const std::filesystem::path& path);
    // Uploads finished decodes. Call once per frame on the UI thread.
    void pump(SDL_Renderer* renderer);
    void clear();

private:
    struct Decoded {
        std::string key;
        int w = 0, h = 0;
        std::vector<unsigned char> rgba;
    };
    void worker();

    std::unordered_map<std::string, SDL_Texture*> textures_;
    std::unordered_set<std::string> requested_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::pair<std::string, std::filesystem::path>> queue_;
    std::vector<Decoded> done_;
    bool quit_ = false;
    std::thread thread_;
};

} // namespace ui
