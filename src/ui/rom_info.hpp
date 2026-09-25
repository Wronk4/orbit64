#pragma once
// Lightweight ROM header inspection used by the library and the ROM picker.
// Only the first 64 bytes are read, so scanning large folders stays cheap.

#include <cstdint>
#include <filesystem>
#include <string>

namespace ui {

enum class RomFormat { Unknown, Z64 /*big endian*/, V64 /*byte swapped*/, N64 /*little endian*/ };

struct RomInfo {
    bool valid = false;
    std::filesystem::path path;
    std::string file_name;     // "Super Mario 64 (USA).z64"
    std::string display_title; // "Super Mario 64" (cleaned file name / header name)
    std::string internal_name; // header name, e.g. "SUPER MARIO 64"
    std::string game_code;     // "NSME"
    std::string region;        // "USA", "Europe", "Japan", ...
    std::string region_short;  // "USA", "EUR", "JPN", "PD"
    std::string tags;          // "(Rev 1) (En,Fr)" – remaining No-Intro tags
    std::string extension;     // "z64"
    RomFormat format = RomFormat::Unknown;
    std::uint32_t crc1 = 0, crc2 = 0;
    std::uint8_t version = 0;
    std::uintmax_t size = 0;
    bool homebrew = false;
    std::filesystem::path boxart; // matched cover image (empty = none)
};

const char* format_name(RomFormat f);
std::string format_size(std::uintmax_t bytes);
bool is_rom_extension(const std::filesystem::path& p);
RomInfo inspect_rom(const std::filesystem::path& p);

} // namespace ui
