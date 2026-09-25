#pragma once
// Save states: the whole machine at a frame boundary as one binary blob
// (Emulator::save_state / load_state), and the file it is stored in
// (write_file / read_file).
//
// Every component lists what it saves in a member
//
//     template <class S> void serialize(S& s) { s(a, b, c); }
//
// which writes those fields with a Writer and reads them back, in the same
// order, with a Reader. Only state that affects emulation belongs there;
// host-side caches (the audio ring, the JIT's compiled blocks, the RDP's
// high-resolution buffers and decoded textures) are rebuilt after a load.
// Values are stored in host byte order - every supported target is
// little-endian - and floats as their bit patterns, so a state restores
// exactly the same machine. Plain structs are stored as they are in memory,
// so a struct with padding gets a serialize() of its own: padding bytes
// aren't state, and the same machine must always give the same bytes.
//
// Adding, removing or reordering a serialized field changes the layout: bump
// kStateVersion with it. A state of another version is rejected, not misread.

#include "common.hpp"
#include <array>
#include <filesystem>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace savestate {

constexpr u32 kStateVersion = 2;

namespace detail {
template <class T> struct is_vector : std::false_type {};
template <class T, class A> struct is_vector<std::vector<T, A>> : std::true_type {};
template <class T> struct is_map : std::false_type {};
template <class K, class V, class H, class E, class A>
struct is_map<std::unordered_map<K, V, H, E, A>> : std::true_type {};
template <class T> struct is_std_array : std::false_type {};
template <class T, size_t N> struct is_std_array<std::array<T, N>> : std::true_type {};
} // namespace detail

class Writer {
public:
    static constexpr bool loading = false;

    // Plain values and arrays of them, structs with a serialize() member,
    // std::vector and std::unordered_map.
    template <class... T> void operator()(T&... v) { (put(v), ...); }
    void bytes(const void* p, size_t n) {
        const u8* b = static_cast<const u8*>(p);
        buf_.insert(buf_.end(), b, b + n);
    }
    // A buffer whose size the machine fixes (RDRAM, save memory): the size is
    // stored and, on load, has to match instead of resizing the buffer.
    template <class T> void fixed(std::vector<T>& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const u32 n = static_cast<u32>(v.size());
        put(n);
        bytes(v.data(), v.size() * sizeof(T));
    }
    // Sections carry a tag and their length, so a mismatch names the
    // component instead of misreading everything after it.
    void begin_section(const char (&tag)[5]);
    void end_section();
    void fail(const std::string&) {}

    std::vector<u8>& data() { return buf_; }

private:
    template <class T> void put(T& v) {
        if constexpr (requires { v.serialize(*this); }) {
            v.serialize(*this);
        } else if constexpr (detail::is_std_array<T>::value && requires { v[0].serialize(*this); }) {
            for (auto& e : v) put(e);
        } else if constexpr (detail::is_vector<T>::value) {
            const u32 n = static_cast<u32>(v.size());
            put(n);
            for (auto& e : v) put(e);
        } else if constexpr (detail::is_map<T>::value) {
            // Sorted by key: the same machine always gives the same bytes.
            std::vector<typename T::key_type> keys;
            keys.reserve(v.size());
            for (auto& kv : v) keys.push_back(kv.first);
            std::sort(keys.begin(), keys.end());
            const u32 n = static_cast<u32>(keys.size());
            put(n);
            for (auto& k : keys) {
                put(k);
                put(v.at(k));
            }
        } else {
            static_assert(std::is_trivially_copyable_v<T>, "serialize() takes plain values");
            bytes(&v, sizeof v);
        }
    }

    std::vector<u8> buf_;
    size_t section_ = 0; // offset of the open section's length field
};

class Reader {
public:
    static constexpr bool loading = true;

    Reader(const u8* data, size_t size) : p_(data), end_(data + size) {}

    template <class... T> void operator()(T&... v) { (get(v), ...); }
    void bytes(void* p, size_t n) {
        if (!ok() || static_cast<size_t>(end_ - p_) < n) {
            fail("the state is truncated");
            std::memset(p, 0, n);
            return;
        }
        std::memcpy(p, p_, n);
        p_ += n;
    }
    template <class T> void fixed(std::vector<T>& v) {
        u32 n = 0;
        get(n);
        if (n != v.size()) {
            fail("a memory block has the wrong size");
            return;
        }
        bytes(v.data(), v.size() * sizeof(T));
    }
    void begin_section(const char (&tag)[5]);
    void end_section();
    // The first error wins; everything read after it is zero.
    void fail(const std::string& why) {
        if (error_.empty()) error_ = why;
    }

    bool ok() const { return error_.empty(); }
    bool at_end() const { return p_ == end_; }
    const std::string& error() const { return error_; }

private:
    template <class T> void get(T& v) {
        if constexpr (requires { v.serialize(*this); }) {
            v.serialize(*this);
        } else if constexpr (detail::is_std_array<T>::value && requires { v[0].serialize(*this); }) {
            for (auto& e : v) get(e);
        } else if constexpr (detail::is_vector<T>::value) {
            u32 n = 0;
            get(n);
            // Every element takes at least one byte: a corrupt count can't
            // make it allocate more than the state holds.
            if (n > static_cast<size_t>(end_ - p_)) {
                fail("the state is truncated");
                n = 0;
            }
            v.clear();
            v.resize(n);
            for (auto& e : v) get(e);
        } else if constexpr (detail::is_map<T>::value) {
            u32 n = 0;
            get(n);
            if (n > static_cast<size_t>(end_ - p_)) {
                fail("the state is truncated");
                n = 0;
            }
            v.clear();
            for (u32 i = 0; i < n && ok(); ++i) {
                typename T::key_type k{};
                typename T::mapped_type m{};
                get(k);
                get(m);
                v[k] = m;
            }
        } else {
            static_assert(std::is_trivially_copyable_v<T>, "serialize() takes plain values");
            bytes(&v, sizeof v);
        }
    }

    const u8* p_;
    const u8* end_;
    const u8* section_end_ = nullptr;
    char section_tag_[5] = {};
    std::string error_;
};

// ---- Files ------------------------------------------------------------------
// A small header (which game, when, and a picture of the screen) followed by
// the zlib-compressed state.

struct FileInfo {
    u32 state_version = kStateVersion;
    u32 rom_crc1 = 0, rom_crc2 = 0; // the cartridge header's checksums: which game this is
    std::string game_code;          // e.g. "NSME"
    std::string rom_title;          // internal name from the ROM header
    s64 created = 0;                // Unix time
    // The screen when the state was made, ARGB8888 at the game's own resolution.
    u32 thumb_w = 0, thumb_h = 0;
    std::vector<u32> thumb;
};

// Writes atomically (a temporary file renamed over `path`), so a crash never
// leaves a half-written state behind. When two writes of one path overlap,
// whichever renames last wins; each file is complete either way.
bool write_file(const std::filesystem::path& path, const std::vector<u8>& state, const FileInfo& info,
                std::string& error);
// `state` receives the uncompressed state; with a null `state` only the header
// and thumbnail are read (slot menus).
bool read_file(const std::filesystem::path& path, std::vector<u8>* state, FileInfo& info, std::string& error);

} // namespace savestate
