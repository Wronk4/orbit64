#include "rom_info.hpp"
#include "platform.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace ui {

namespace fs = std::filesystem;

const char* format_name(RomFormat f) {
    switch (f) {
        case RomFormat::Z64: return "Big-endian (.z64)";
        case RomFormat::V64: return "Byte-swapped (.v64)";
        case RomFormat::N64: return "Little-endian (.n64)";
        default: return "Unknown";
    }
}

std::string format_size(std::uintmax_t bytes) {
    char buf[32];
    if (bytes >= 1024ull * 1024ull) std::snprintf(buf, sizeof buf, "%.1f MB", bytes / (1024.0 * 1024.0));
    else if (bytes >= 1024ull) std::snprintf(buf, sizeof buf, "%.0f KB", bytes / 1024.0);
    else std::snprintf(buf, sizeof buf, "%llu B", static_cast<unsigned long long>(bytes));
    return buf;
}

bool is_rom_extension(const fs::path& p) {
    std::string ext = platform::path_to_utf8(p.extension());
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext == ".z64" || ext == ".n64" || ext == ".v64";
}

static std::string trim(std::string s) {
    auto ns = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), ns));
    s.erase(std::find_if(s.rbegin(), s.rend(), ns).base(), s.end());
    return s;
}

static void region_from_code(char c, std::string& full, std::string& shrt) {
    switch (c) {
        case 'E': full = "North America"; shrt = "USA"; break;
        case 'J': full = "Japan"; shrt = "JPN"; break;
        case 'P': case 'X': case 'Y': full = "Europe"; shrt = "EUR"; break;
        case 'D': full = "Germany"; shrt = "GER"; break;
        case 'F': full = "France"; shrt = "FRA"; break;
        case 'I': full = "Italy"; shrt = "ITA"; break;
        case 'S': full = "Spain"; shrt = "SPA"; break;
        case 'U': full = "Australia"; shrt = "AUS"; break;
        case 'A': full = "Asia / Multi-region"; shrt = "ALL"; break;
        case 'B': full = "Brazil"; shrt = "BRA"; break;
        case 'C': full = "China"; shrt = "CHN"; break;
        case 'K': full = "Korea"; shrt = "KOR"; break;
        default: full = "Unknown"; shrt = "???"; break;
    }
}

// Splits "Legend of Zelda, The - Ocarina of Time (USA) (Rev 1)" into a
// presentable title ("The Legend of Zelda - Ocarina of Time") and tags.
static void clean_title(const std::string& stem, std::string& title, std::string& tags, std::string& region_tag) {
    std::string base = stem;
    size_t paren = std::string::npos;
    for (size_t i = 0; i < stem.size(); ++i) {
        if (stem[i] == '(' || stem[i] == '[') { paren = i; break; }
    }
    if (paren != std::string::npos) {
        base = stem.substr(0, paren);
        tags = trim(stem.substr(paren));
        size_t close = stem.find_first_of(")]", paren);
        if (close != std::string::npos) region_tag = stem.substr(paren + 1, close - paren - 1);
    }
    base = trim(base);
    // Move trailing article: "Legend of Zelda, The - Ocarina" -> "The Legend of Zelda - Ocarina"
    for (const char* art : {", The", ", A"}) {
        size_t pos = base.find(art);
        if (pos != std::string::npos) {
            std::string article = std::string(art).substr(2);
            size_t end = pos + std::strlen(art);
            base = article + " " + base.substr(0, pos) + base.substr(end);
            break;
        }
    }
    title = base;
}

RomInfo inspect_rom(const fs::path& p) {
    RomInfo info;
    info.path = p;
    info.file_name = platform::path_to_utf8(p.filename());
    std::string ext = platform::path_to_utf8(p.extension());
    info.extension = ext.empty() ? "" : ext.substr(1);
    std::error_code ec;
    info.size = fs::file_size(p, ec);
    if (ec) return info;

    std::ifstream f(p, std::ios::binary);
    unsigned char h[0x40] = {};
    if (!f.read(reinterpret_cast<char*>(h), sizeof h)) return info;

    const std::uint32_t magic = (h[0] << 24) | (h[1] << 16) | (h[2] << 8) | h[3];
    if (magic == 0x80371240) info.format = RomFormat::Z64;
    else if (magic == 0x37804012) info.format = RomFormat::V64;
    else if (magic == 0x40123780) info.format = RomFormat::N64;
    else return info;

    // Normalise to big-endian.
    for (int i = 0; i < 0x40; i += 4) {
        unsigned char b[4] = {h[i], h[i + 1], h[i + 2], h[i + 3]};
        if (info.format == RomFormat::V64) { h[i] = b[1]; h[i + 1] = b[0]; h[i + 2] = b[3]; h[i + 3] = b[2]; }
        if (info.format == RomFormat::N64) { h[i] = b[3]; h[i + 1] = b[2]; h[i + 2] = b[1]; h[i + 3] = b[0]; }
    }
    info.valid = true;
    info.crc1 = (h[0x10] << 24) | (h[0x11] << 16) | (h[0x12] << 8) | h[0x13];
    info.crc2 = (h[0x14] << 24) | (h[0x15] << 16) | (h[0x16] << 8) | h[0x17];

    std::string name;
    for (int i = 0x20; i < 0x34; ++i) {
        unsigned char c = h[i];
        name.push_back((c >= 0x20 && c < 0x7F) ? static_cast<char>(c) : ' ');
    }
    info.internal_name = trim(name);
    for (int i = 0x3B; i < 0x3F; ++i) {
        unsigned char c = h[i];
        info.game_code.push_back((c >= 0x20 && c < 0x7F) ? static_cast<char>(c) : '-');
    }
    info.version = h[0x3F];

    std::string tags, region_tag;
    clean_title(platform::path_to_utf8(p.stem()), info.display_title, tags, region_tag);
    info.tags = tags;
    info.homebrew = tags.find("(PD)") != std::string::npos || tags.find("by ") != std::string::npos;
    region_from_code(h[0x3E], info.region, info.region_short);

    if (info.homebrew) {
        info.region = "Public Domain";
        info.region_short = "PD";
    } else if (region_tag == "USA") {
        info.region = "North America"; info.region_short = "USA";
    } else if (region_tag == "Europe") {
        info.region = "Europe"; info.region_short = "EUR";
    } else if (region_tag == "Japan") {
        info.region = "Japan"; info.region_short = "JPN";
    }
    if (info.display_title.empty()) info.display_title = info.internal_name;
    return info;
}

} // namespace ui
