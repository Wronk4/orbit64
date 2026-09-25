#include "savestate.hpp"
#include <atomic>
#include <fstream>
#include <system_error>

// zlib (deflate) from the stb headers, compiled privately into this file:
// the frontend has its own copies of these implementations.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_STATIC
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace savestate {

namespace fs = std::filesystem;

void Writer::begin_section(const char (&tag)[5]) {
    bytes(tag, 4);
    section_ = buf_.size();
    const u32 len = 0; // patched by end_section()
    bytes(&len, 4);
}

void Writer::end_section() {
    const u32 len = static_cast<u32>(buf_.size() - section_ - 4);
    std::memcpy(buf_.data() + section_, &len, 4);
}

// "RDP " -> "RDP", for messages.
static std::string tag_name(const char* tag) {
    std::string s(tag, strnlen(tag, 4));
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

void Reader::begin_section(const char (&tag)[5]) {
    std::memcpy(section_tag_, tag, 5);
    char got[4] = {};
    u32 len = 0;
    bytes(got, 4);
    bytes(&len, 4);
    if (!ok()) return;
    if (std::memcmp(got, tag, 4) != 0) {
        fail("expected the " + tag_name(tag) + " section");
        return;
    }
    if (len > static_cast<size_t>(end_ - p_)) {
        fail("the " + tag_name(tag) + " section is truncated");
        return;
    }
    section_end_ = p_ + len;
}

void Reader::end_section() {
    if (ok() && p_ != section_end_) fail("the " + tag_name(section_tag_) + " section has an unexpected size");
}

// ---- File format --------------------------------------------------------------
//
//   offset size
//   0      8    magic "ORB64SAV"
//   8      4    file format version (kFileVersion)
//   12     4    state version (kStateVersion of the build that wrote it)
//   16     8    ROM CRC1, CRC2
//   24     4    game code
//   28     20   ROM title (internal name, NUL padded)
//   48     8    creation time (Unix seconds)
//   56     4    thumbnail width, height (u16 each)
//   60     4    thumbnail bytes (zlib)
//   64     4    state bytes, uncompressed
//   68     4    state bytes (zlib)
//   72     8    FNV-1a 64 of the uncompressed state
//   80          thumbnail, then state
//
// All little-endian.

namespace {

constexpr char kMagic[8] = {'O', 'R', 'B', '6', '4', 'S', 'A', 'V'};
constexpr u32 kFileVersion = 1;
constexpr size_t kHeaderSize = 80;
constexpr int kZlibLevel = 4; // stb's deflate barely gains past this, only gets slower
// Nothing the machine holds comes close: a larger size means a damaged file.
constexpr u32 kMaxStateBytes = 64u << 20;

void put_le(std::vector<u8>& b, u64 v, int n) {
    for (int i = 0; i < n; ++i) b.push_back(static_cast<u8>(v >> (8 * i)));
}
u64 get_le(const u8* p, int n) {
    u64 v = 0;
    for (int i = 0; i < n; ++i) v |= static_cast<u64>(p[i]) << (8 * i);
    return v;
}

u64 fnv1a(const u8* p, size_t n) {
    u64 h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

std::vector<u8> deflate(const void* data, size_t size) {
    int out_len = 0;
    unsigned char* out = stbi_zlib_compress(static_cast<unsigned char*>(const_cast<void*>(data)), static_cast<int>(size),
                                            &out_len, kZlibLevel);
    std::vector<u8> v;
    if (out) {
        v.assign(out, out + out_len);
        STBIW_FREE(out);
    }
    return v;
}

bool inflate(const u8* data, size_t size, size_t expected, std::vector<u8>& out) {
    int len = 0;
    char* raw = stbi_zlib_decode_malloc_guesssize_headerflag(reinterpret_cast<const char*>(data), static_cast<int>(size),
                                                             static_cast<int>(expected), &len, 1);
    if (!raw) return false;
    const bool ok = static_cast<size_t>(len) == expected;
    if (ok) out.assign(raw, raw + len);
    STBI_FREE(raw);
    return ok;
}

} // namespace

bool write_file(const fs::path& path, const std::vector<u8>& state, const FileInfo& info, std::string& error) {
    const std::vector<u8> thumb = info.thumb.empty() ? std::vector<u8>{} : deflate(info.thumb.data(), info.thumb.size() * 4);
    const std::vector<u8> body = deflate(state.data(), state.size());
    if (body.empty()) {
        error = "compressing the state failed";
        return false;
    }

    std::vector<u8> head;
    head.reserve(kHeaderSize);
    head.insert(head.end(), kMagic, kMagic + 8);
    put_le(head, kFileVersion, 4);
    put_le(head, info.state_version, 4);
    put_le(head, info.rom_crc1, 4);
    put_le(head, info.rom_crc2, 4);
    for (size_t i = 0; i < 4; ++i) head.push_back(i < info.game_code.size() ? static_cast<u8>(info.game_code[i]) : 0);
    for (size_t i = 0; i < 20; ++i) head.push_back(i < info.rom_title.size() ? static_cast<u8>(info.rom_title[i]) : 0);
    put_le(head, static_cast<u64>(info.created), 8);
    const bool has_thumb = !thumb.empty() && info.thumb_w <= 0xFFFF && info.thumb_h <= 0xFFFF;
    put_le(head, has_thumb ? info.thumb_w : 0, 2);
    put_le(head, has_thumb ? info.thumb_h : 0, 2);
    put_le(head, has_thumb ? thumb.size() : 0, 4);
    put_le(head, state.size(), 4);
    put_le(head, body.size(), 4);
    put_le(head, fnv1a(state.data(), state.size()), 8);

    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    // A temporary name of its own: two writes of the same path never share it.
    static std::atomic<u32> serial{0};
    fs::path tmp = path;
    tmp += ".tmp" + std::to_string(serial++);
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            error = "the file couldn't be created";
            return false;
        }
        f.write(reinterpret_cast<const char*>(head.data()), static_cast<std::streamsize>(head.size()));
        if (has_thumb) f.write(reinterpret_cast<const char*>(thumb.data()), static_cast<std::streamsize>(thumb.size()));
        f.write(reinterpret_cast<const char*>(body.data()), static_cast<std::streamsize>(body.size()));
        f.flush();
        if (!f) {
            f.close();
            fs::remove(tmp, ec);
            error = "writing the file failed (disk full?)";
            return false;
        }
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        error = "the file couldn't be replaced";
        return false;
    }
    return true;
}

bool read_file(const fs::path& path, std::vector<u8>* state, FileInfo& info, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = "the file couldn't be opened";
        return false;
    }
    u8 head[kHeaderSize];
    if (!f.read(reinterpret_cast<char*>(head), kHeaderSize) || std::memcmp(head, kMagic, 8) != 0) {
        error = "this is not an Orbit64 save state";
        return false;
    }
    if (get_le(head + 8, 4) != kFileVersion) {
        error = "the state was saved in an unsupported format";
        return false;
    }
    info.state_version = static_cast<u32>(get_le(head + 12, 4));
    info.rom_crc1 = static_cast<u32>(get_le(head + 16, 4));
    info.rom_crc2 = static_cast<u32>(get_le(head + 20, 4));
    info.game_code.assign(reinterpret_cast<const char*>(head + 24), strnlen(reinterpret_cast<const char*>(head + 24), 4));
    info.rom_title.assign(reinterpret_cast<const char*>(head + 28), strnlen(reinterpret_cast<const char*>(head + 28), 20));
    info.created = static_cast<s64>(get_le(head + 48, 8));
    info.thumb_w = static_cast<u32>(get_le(head + 56, 2));
    info.thumb_h = static_cast<u32>(get_le(head + 58, 2));
    const u32 thumb_bytes = static_cast<u32>(get_le(head + 60, 4));
    const u32 raw_bytes = static_cast<u32>(get_le(head + 64, 4));
    const u32 body_bytes = static_cast<u32>(get_le(head + 68, 4));
    const u64 checksum = get_le(head + 72, 8);

    info.thumb.clear();
    const size_t thumb_px = static_cast<size_t>(info.thumb_w) * info.thumb_h;
    if (thumb_bytes > 0 && thumb_px > 0 && thumb_px <= 1024 * 1024 && thumb_bytes <= kMaxStateBytes) {
        std::vector<u8> z(thumb_bytes), px;
        if (f.read(reinterpret_cast<char*>(z.data()), thumb_bytes) && inflate(z.data(), z.size(), thumb_px * 4, px)) {
            info.thumb.resize(thumb_px);
            std::memcpy(info.thumb.data(), px.data(), px.size());
        }
    }
    if (info.thumb.empty()) info.thumb_w = info.thumb_h = 0;
    if (!state) return true;

    if (info.state_version != kStateVersion) {
        error = "the state was saved by a different version of Orbit64 (state format " +
                std::to_string(info.state_version) + ", this build reads " + std::to_string(kStateVersion) + ")";
        return false;
    }
    if (raw_bytes == 0 || raw_bytes > kMaxStateBytes || body_bytes == 0 || body_bytes > kMaxStateBytes) {
        error = "the file is damaged";
        return false;
    }
    f.clear();
    f.seekg(static_cast<std::streamoff>(kHeaderSize) + thumb_bytes);
    std::vector<u8> z(body_bytes);
    if (!f.read(reinterpret_cast<char*>(z.data()), body_bytes) || !inflate(z.data(), z.size(), raw_bytes, *state) ||
        fnv1a(state->data(), state->size()) != checksum) {
        state->clear();
        error = "the file is damaged";
        return false;
    }
    return true;
}

} // namespace savestate
