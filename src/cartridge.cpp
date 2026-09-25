#include "cartridge.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

Cartridge::Cartridge() = default;

Cartridge::~Cartridge() {
    save_backup();
}

// Paths are UTF-8 strings; going through std::filesystem::path keeps
// non-ASCII folder names working on Windows (where a narrow std::string
// would be interpreted in the ANSI code page).
static std::filesystem::path utf8_path(const std::string& s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

bool Cartridge::load_rom(const std::string& filepath) {
    std::ifstream file(utf8_path(filepath), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[Cartridge] Failed to open ROM: " << filepath << "\n";
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size < 0x1000) {
        std::cerr << "[Cartridge] ROM file too small: " << size << " bytes\n";
        return false;
    }

    rom.resize(size);
    if (!file.read(reinterpret_cast<char*>(rom.data()), size)) {
        std::cerr << "[Cartridge] Failed to read ROM data\n";
        return false;
    }

    // Check endianness and convert to big-endian (z64 format)
    u8 b0 = rom[0], b1 = rom[1], b2 = rom[2], b3 = rom[3];
    if (b0 == 0x80 && b1 == 0x37 && b2 == 0x12 && b3 == 0x40) {
        // Native .z64 big-endian
    } else if (b0 == 0x37 && b1 == 0x80 && b2 == 0x40 && b3 == 0x12) {
        // .v64 byte-swapped
        for (size_t i = 0; i < rom.size(); i += 2) {
            std::swap(rom[i], rom[i + 1]);
        }
    } else if (b0 == 0x40 && b1 == 0x12 && b2 == 0x37 && b3 == 0x80) {
        // .n64 little-endian
        for (size_t i = 0; i < rom.size(); i += 4) {
            std::swap(rom[i], rom[i + 3]);
            std::swap(rom[i + 1], rom[i + 2]);
        }
    } else {
        std::cerr << "[Cartridge] Unknown ROM format magic: " 
                  << std::hex << (int)b0 << " " << (int)b1 << " " 
                  << (int)b2 << " " << (int)b3 << std::dec << "\n";
        return false;
    }

    // Parse header fields
    entry_point = (static_cast<u32>(rom[8]) << 24) |
                  (static_cast<u32>(rom[9]) << 16) |
                  (static_cast<u32>(rom[10]) << 8) |
                  static_cast<u32>(rom[11]);

    crc1 = (static_cast<u32>(rom[16]) << 24) |
           (static_cast<u32>(rom[17]) << 16) |
           (static_cast<u32>(rom[18]) << 8) |
           static_cast<u32>(rom[19]);

    crc2 = (static_cast<u32>(rom[20]) << 24) |
           (static_cast<u32>(rom[21]) << 16) |
           (static_cast<u32>(rom[22]) << 8) |
           static_cast<u32>(rom[23]);

    title.clear();
    for (int i = 32; i < 52; ++i) {
        char c = static_cast<char>(rom[i]);
        if (c >= 32 && c <= 126) title.push_back(c);
    }
    // Trim trailing spaces
    while (!title.empty() && title.back() == ' ') title.pop_back();

    game_code.clear();
    for (int i = 59; i < 63; ++i) {
        game_code.push_back(static_cast<char>(rom[i]));
    }

    detect_cic_and_save();

    // Set save file path (none: save_backup() then has nothing to write to)
    size_t dot = filepath.find_last_of('.');
    if (!use_save_file_) {
        save_filepath.clear();
    } else if (dot != std::string::npos) {
        save_filepath = filepath.substr(0, dot) + ".sav";
    } else {
        save_filepath = filepath + ".sav";
    }

    // Load existing save file if present
    std::ifstream save_file;
    if (!save_filepath.empty()) save_file.open(utf8_path(save_filepath), std::ios::binary);
    if (save_file.is_open()) {
        if (save_type == SaveType::SRAM_32K) {
            save_file.read(reinterpret_cast<char*>(sram.data()), sram.size());
        } else if (save_type == SaveType::EEPROM_4K || save_type == SaveType::EEPROM_16K) {
            save_file.read(reinterpret_cast<char*>(eeprom.data()), eeprom.size());
        }
    }

    std::cout << "[Cartridge] Loaded: \"" << title << "\" [" << game_code << "]\n";
    std::cout << "[Cartridge] Size: " << (rom.size() / (1024 * 1024)) << " MB, Entry: 0x" 
              << std::hex << entry_point << ", CRC1: 0x" << crc1 << ", CRC2: 0x" << crc2 << std::dec << "\n";
    std::cout << "[Cartridge] CIC: " << (cic_type == CICType::CIC_6105 ? "6105" : "6102") << "\n";

    return true;
}

void Cartridge::detect_cic_and_save() {
    // Check game code
    if (game_code == "CZLE" || game_code == "NZLE" || game_code == "NZLP" || game_code == "NZLJ") {
        cic_type = CICType::CIC_6105;
        save_type = SaveType::SRAM_32K;
    } else {
        cic_type = CICType::CIC_6102;
        save_type = SaveType::EEPROM_4K;
    }

    if (save_type == SaveType::SRAM_32K) {
        sram.resize(32768, 0xFF);
    } else if (save_type == SaveType::EEPROM_4K) {
        eeprom.resize(512, 0x00);
    } else if (save_type == SaveType::EEPROM_16K) {
        eeprom.resize(2048, 0x00);
    }
}

void Cartridge::save_backup() {
    if (!sram_dirty && !eeprom_dirty) return;
    if (save_filepath.empty()) return;

    std::ofstream file(utf8_path(save_filepath), std::ios::binary);
    if (!file.is_open()) return;

    if (save_type == SaveType::SRAM_32K && !sram.empty()) {
        file.write(reinterpret_cast<const char*>(sram.data()), sram.size());
        sram_dirty = false;
    } else if ((save_type == SaveType::EEPROM_4K || save_type == SaveType::EEPROM_16K) && !eeprom.empty()) {
        file.write(reinterpret_cast<const char*>(eeprom.data()), eeprom.size());
        eeprom_dirty = false;
    }
}

u8 Cartridge::read_rom(u32 addr) const {
    if (addr < rom.size()) return rom[addr];
    return 0;
}

u16 Cartridge::read_rom16(u32 addr) const {
    if (addr + 1 < rom.size()) {
        return (static_cast<u16>(rom[addr]) << 8) | static_cast<u16>(rom[addr + 1]);
    }
    return 0;
}

u32 Cartridge::read_rom32(u32 addr) const {
    if (addr + 3 < rom.size()) {
        return (static_cast<u32>(rom[addr]) << 24) |
               (static_cast<u32>(rom[addr + 1]) << 16) |
               (static_cast<u32>(rom[addr + 2]) << 8) |
               static_cast<u32>(rom[addr + 3]);
    }
    return 0;
}

u8 Cartridge::read_sram(u32 addr) const {
    if (sram.empty()) return 0;
    return sram[addr % sram.size()];
}

void Cartridge::write_sram(u32 addr, u8 val) {
    if (sram.empty()) return;
    sram[addr % sram.size()] = val;
    sram_dirty = true;
}

void Cartridge::read_eeprom(u8 block, u8* out_data) {
    size_t offset = static_cast<size_t>(block) * 8;
    if (offset + 8 <= eeprom.size()) {
        std::memcpy(out_data, eeprom.data() + offset, 8);
    } else {
        std::memset(out_data, 0, 8);
    }
}

void Cartridge::write_eeprom(u8 block, const u8* in_data) {
    size_t offset = static_cast<size_t>(block) * 8;
    if (offset + 8 <= eeprom.size()) {
        std::memcpy(eeprom.data() + offset, in_data, 8);
        eeprom_dirty = true;
    }
}
