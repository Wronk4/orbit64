#pragma once
// Writes a file so a crash, a full disk or a power loss can't leave it truncated:
// the data goes to a temporary file next to it, which then replaces the target.
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace atomic_file {

inline bool write(const std::filesystem::path& path, const void* data, size_t size) {
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        if (size) f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        f.flush();
        if (!f) {
            f.close();
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

} // namespace atomic_file
