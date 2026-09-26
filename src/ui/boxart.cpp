#include "boxart.hpp"
#include "platform.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO // files are read via std::filesystem (UTF-8 safe on Windows)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "stb_image.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace ui {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Name normalisation
// ---------------------------------------------------------------------------

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// Title without "(...)"/"[...]" tags.
static std::string strip_tags(const std::string& stem) {
    size_t p = stem.find_first_of("([");
    return p == std::string::npos ? stem : stem.substr(0, p);
}

// "Legend of Zelda, The - Ocarina of Time" -> "legendofzeldatheocarinaoftime"
static std::string norm_title(const std::string& s) {
    std::string out;
    for (unsigned char c : s)
        if (std::isalnum(c)) out.push_back(static_cast<char>(std::tolower(c)));
    return out;
}

// Order-independent word key: "007 - GoldenEye" and "GoldenEye 007" -> "007 goldeneye"
static std::string word_key(const std::string& s) {
    std::vector<std::string> words;
    std::string cur;
    for (unsigned char c : s) {
        if (std::isalnum(c)) cur.push_back(static_cast<char>(std::tolower(c)));
        else if (!cur.empty()) { words.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) words.push_back(cur);
    std::sort(words.begin(), words.end());
    words.erase(std::unique(words.begin(), words.end()), words.end());
    std::string out;
    for (const auto& w : words) out += (out.empty() ? "" : " ") + w;
    return out;
}

static bool is_image(const fs::path& p) {
    std::string e = lower(platform::path_to_utf8(p.extension()));
    return e == ".png" || e == ".jpg" || e == ".jpeg";
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

std::vector<fs::path> BoxArtIndex::discover(const std::vector<fs::path>& roots) {
    std::vector<fs::path> found;
    for (const auto& root : roots) {
        std::error_code ec;
        if (!fs::is_directory(root, ec)) continue;
        fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
        for (; it != end; it.increment(ec)) {
            if (ec) break;
            if (it.depth() > 3) { it.disable_recursion_pending(); continue; }
            if (!it->is_directory(ec)) continue;
            std::string name = lower(platform::path_to_utf8(it->path().filename()));
            if (!name.empty() && name[0] == '.') { it.disable_recursion_pending(); continue; }
            // RetroArch layout ("Named_Boxarts") or any folder called *box*/*cover*.
            if (name == "named_boxarts" || name.find("box") != std::string::npos || name.find("cover") != std::string::npos) {
                bool dup = std::any_of(found.begin(), found.end(), [&](const fs::path& f) {
                    return it->path().native().rfind(f.native(), 0) == 0;
                });
                if (!dup) {
                    found.push_back(it->path());
                    it.disable_recursion_pending(); // indexed recursively anyway
                }
            }
        }
    }
    return found;
}

void BoxArtIndex::build(const std::vector<fs::path>& dirs) {
    by_stem_.clear();
    by_title_.clear();
    by_words_.clear();
    count_ = 0;
    for (const auto& d : dirs) {
        std::error_code ec;
        if (!fs::is_directory(d, ec)) continue;
        fs::recursive_directory_iterator it(d, fs::directory_options::skip_permission_denied, ec), end;
        for (; it != end; it.increment(ec)) {
            if (ec) break;
            if (it.depth() > 4) { it.disable_recursion_pending(); continue; }
            if (!it->is_regular_file(ec) || !is_image(it->path())) continue;
            std::string stem = platform::path_to_utf8(it->path().stem());
            Candidate c{it->path(), stem};
            by_stem_.emplace(lower(stem), it->path());
            by_title_[norm_title(strip_tags(stem))].push_back(c);
            by_words_[word_key(strip_tags(stem))].push_back(c);
            ++count_;
        }
    }
}

fs::path BoxArtIndex::find(const fs::path& rom, const std::string& internal_name) const {
    if (count_ == 0) return {};
    const std::string stem = platform::path_to_utf8(rom.stem());
    if (auto it = by_stem_.find(lower(stem)); it != by_stem_.end()) return it->second;

    // Pick the most similar variant: same tags as the ROM, then USA, then no revision.
    const std::string tags = lower(stem.size() > strip_tags(stem).size() ? stem.substr(strip_tags(stem).size()) : "");
    auto best_of = [&](const std::vector<Candidate>& list) {
        const Candidate* best = nullptr;
        int best_score = -1;
        for (const auto& c : list) {
            std::string ct = lower(c.stem);
            int score = 0;
            if (!tags.empty() && ct.find(tags.substr(0, tags.find(')') + 1)) != std::string::npos) score += 4;
            if (ct.find("(usa)") != std::string::npos) score += 2;
            if (ct.find("(rev") == std::string::npos) score += 1;
            if (score > best_score || (score == best_score && best && c.stem.size() < best->stem.size())) {
                best = &c;
                best_score = score;
            }
        }
        return best ? best->path : fs::path();
    };
    const std::string title = strip_tags(stem);
    if (auto it = by_title_.find(norm_title(title)); it != by_title_.end()) return best_of(it->second);
    if (auto it = by_words_.find(word_key(title)); it != by_words_.end()) return best_of(it->second);
    // Last resort: the ROM header's internal name (e.g. a renamed file).
    if (!internal_name.empty()) {
        if (auto it = by_title_.find(norm_title(internal_name)); it != by_title_.end()) return best_of(it->second);
        if (auto it = by_words_.find(word_key(internal_name)); it != by_words_.end()) return best_of(it->second);
    }
    return {};
}

// ---------------------------------------------------------------------------
// Image cache
// ---------------------------------------------------------------------------

ImageCache::ImageCache() : thread_(&ImageCache::worker, this) {}

ImageCache::~ImageCache() {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        quit_ = true;
    }
    cv_.notify_all();
    thread_.join();
    for (auto& [k, t] : textures_) if (t) SDL_DestroyTexture(t);
}

SDL_Texture* ImageCache::get(const fs::path& path) {
    if (path.empty()) return nullptr;
    std::string key = platform::path_to_utf8(path);
    if (auto it = textures_.find(key); it != textures_.end()) return it->second;
    if (requested_.insert(key).second) {
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.emplace_back(key, path);
        cv_.notify_one();
    }
    return nullptr;
}

void ImageCache::pump(SDL_Renderer* renderer) {
    std::vector<Decoded> ready;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        ready.swap(done_);
    }
    for (auto& d : ready) {
        SDL_Texture* tex = nullptr;
        if (!d.rgba.empty()) {
            tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STATIC, d.w, d.h);
            if (tex) {
                SDL_UpdateTexture(tex, nullptr, d.rgba.data(), d.w * 4);
                SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_LINEAR);
                SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND);
            }
        }
        textures_[d.key] = tex; // nullptr marks a failed decode, so it isn't retried every frame
    }
}

void ImageCache::clear() {
    for (auto& [k, t] : textures_) if (t) SDL_DestroyTexture(t);
    textures_.clear();
    requested_.clear();
}

void ImageCache::worker() {
    for (;;) {
        std::pair<std::string, fs::path> job;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_.wait(lk, [&] { return quit_ || !queue_.empty(); });
            if (quit_) return;
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        Decoded d;
        d.key = job.first;
        std::ifstream f(job.second, std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string bytes = ss.str();
        int comp = 0;
        if (unsigned char* px = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(bytes.data()),
                                                      static_cast<int>(bytes.size()), &d.w, &d.h, &comp, 4)) {
            d.rgba.assign(px, px + static_cast<size_t>(d.w) * d.h * 4);
            stbi_image_free(px);
        }
        std::lock_guard<std::mutex> lk(mutex_);
        done_.push_back(std::move(d));
    }
}

} // namespace ui
