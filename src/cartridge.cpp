#include "cartridge.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace { void format_mempak(std::vector<u8>& pak); }

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
        if (save_type == SaveType::SRAM_32K || save_type == SaveType::FLASHRAM_128K) {
            save_file.read(reinterpret_cast<char*>(sram.data()), sram.size());
        } else if (save_type == SaveType::EEPROM_4K || save_type == SaveType::EEPROM_16K) {
            save_file.read(reinterpret_cast<char*>(eeprom.data()), eeprom.size());
        }
    }

    // Controller Pak for controller 1.
    mempak_filepath.clear();
    mempak_dirty = false;
    format_mempak(mempak);
    if (use_save_file_) {
        mempak_filepath = (dot != std::string::npos ? filepath.substr(0, dot) : filepath) + ".mpk";
        std::ifstream pak_file(utf8_path(mempak_filepath), std::ios::binary);
        if (pak_file.is_open()) pak_file.read(reinterpret_cast<char*>(mempak.data()), mempak.size());
    }

    std::cout << "[Cartridge] Loaded: \"" << title << "\" [" << game_code << "]\n";
    std::cout << "[Cartridge] Size: " << (rom.size() / (1024 * 1024)) << " MB, Entry: 0x" 
              << std::hex << entry_point << ", CRC1: 0x" << crc1 << ", CRC2: 0x" << crc2 << std::dec << "\n";
    static const char* cic_names[] = {"6101", "6102", "6103", "6105", "6106", "unknown"};
    static const char* save_names[] = {"none", "EEPROM 4K", "EEPROM 16K", "SRAM", "FlashRAM"};
    std::cout << "[Cartridge] CIC: " << cic_names[static_cast<int>(cic_type)]
              << ", save: " << save_names[static_cast<int>(save_type)] << "\n";

    return true;
}

namespace {
// CRC-32 (zlib polynomial) of the boot code, ROM 0x40-0xFFF: it tells which
// CIC chip the cartridge was made for.
u32 crc32(const u8* p, size_t n) {
    u32 c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

// Save chip by game (the two-letter id in the middle of the game code, so
// all regions and the 64DD "C" variants match). Games not listed get a 4 Kbit
// EEPROM, the most common chip.
struct SaveEntry { const char id[3]; SaveType type; };
constexpr SaveEntry kSaveTypes[] = {
    // 16 Kbit EEPROM
    {"DO", SaveType::EEPROM_16K}, // Donkey Kong 64
    {"B7", SaveType::EEPROM_16K}, // Banjo-Tooie
    {"PD", SaveType::EEPROM_16K}, // Perfect Dark
    {"YS", SaveType::EEPROM_16K}, // Yoshi's Story
    {"EV", SaveType::EEPROM_16K}, // Excitebike 64
    {"FU", SaveType::EEPROM_16K}, // Conker's Bad Fur Day
    {"JF", SaveType::EEPROM_16K}, // Jet Force Gemini
    {"M8", SaveType::EEPROM_16K}, // Mario Tennis
    // SRAM
    {"ZL", SaveType::SRAM_32K}, // Zelda: Ocarina of Time
    {"FZ", SaveType::SRAM_32K}, // F-Zero X
    {"AL", SaveType::SRAM_32K}, // Super Smash Bros.
    {"RE", SaveType::SRAM_32K}, // Resident Evil 2
    {"YW", SaveType::SRAM_32K}, // Harvest Moon 64
    {"MF", SaveType::SRAM_32K}, // Mario Golf
    {"OB", SaveType::SRAM_32K}, // Ogre Battle 64
    // FlashRAM
    {"PF", SaveType::FLASHRAM_128K}, // Pokemon Snap
    {"ZS", SaveType::FLASHRAM_128K}, // Zelda: Majora's Mask
    {"MQ", SaveType::FLASHRAM_128K}, // Paper Mario
    {"PO", SaveType::FLASHRAM_128K}, // Pokemon Stadium
    {"PN", SaveType::FLASHRAM_128K}, // Pokemon Puzzle League
    // No save chip (Controller Pak only)
    {"QK", SaveType::NONE}, // Quake
    {"R6", SaveType::NONE}, // Rainbow Six
    {"HW", SaveType::NONE}, // Hot Wheels Turbo Racing
    {"SL", SaveType::NONE}, // Spider-Man
};
} // namespace

namespace {
// An empty, formatted Controller Pak file system (libultra's layout): the ID
// block (with its checksums) four times on page 0, then the inode table and
// its backup on pages 1-2 with every data page free, empty note table.
void format_mempak(std::vector<u8>& pak) {
    std::fill(pak.begin(), pak.end(), 0);
    for (int i = 0; i < 32; ++i) pak[i] = static_cast<u8>(i == 0 ? 0x81 : i); // label area
    u8 id[32] = {0xFF, 0xFF, 0xFF, 0xFF, 0x05, 0x1A, 0x5F, 0x13, 0, 0, 0, 0, 0, 0, 0, 0,
                 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0xFF, 0, 0, 0, 0};
    u32 sum = 0;
    for (int i = 0; i < 28; i += 2) sum += (id[i] << 8) | id[i + 1];
    const u16 chk = static_cast<u16>(sum), inv = static_cast<u16>(0xFFF2 - chk);
    id[28] = chk >> 8; id[29] = chk & 0xFF; id[30] = inv >> 8; id[31] = inv & 0xFF;
    for (u32 off : {0x20u, 0x60u, 0x80u, 0xC0u}) std::copy(id, id + 32, pak.begin() + off);
    for (u32 page : {1u, 2u}) {
        u8* p = pak.data() + page * 256;
        for (int e = 1; e < 128; ++e) { p[e * 2] = 0x00; p[e * 2 + 1] = 0x03; } // free
        u32 c = 0;
        for (int b = 10; b < 256; ++b) c += p[b]; // checksum of the data pages' entries
        p[0] = 0x00; p[1] = static_cast<u8>(c);
    }
}
} // namespace

void Cartridge::detect_cic_and_save() {
    switch (crc32(rom.data() + 0x40, 0x1000 - 0x40)) {
        case 0x6170A4A1: cic_type = CICType::CIC_6101; break;
        case 0x009E9EA3: cic_type = CICType::CIC_6101; break; // 7102 (Lylat Wars)
        case 0x90BB6CB5: cic_type = CICType::CIC_6102; break;
        case 0x0B050EE0: cic_type = CICType::CIC_6103; break;
        case 0x98BC2C86: cic_type = CICType::CIC_6105; break;
        case 0xACC8580A: cic_type = CICType::CIC_6106; break;
        default: cic_type = CICType::CIC_6102; break; // homebrew / unknown boot code
    }

    save_type = SaveType::EEPROM_4K;
    if (game_code.size() >= 3) {
        for (const SaveEntry& e : kSaveTypes) {
            if (game_code[1] == e.id[0] && game_code[2] == e.id[1]) {
                save_type = e.type;
                break;
            }
        }
    }

    sram.clear();
    eeprom.clear();
    if (save_type == SaveType::SRAM_32K) {
        sram.resize(32768, 0xFF);
    } else if (save_type == SaveType::FLASHRAM_128K) {
        sram.resize(131072, 0xFF);
    } else if (save_type == SaveType::EEPROM_4K) {
        eeprom.resize(512, 0x00);
    } else if (save_type == SaveType::EEPROM_16K) {
        eeprom.resize(2048, 0x00);
    }
    flash_mode = FLASH_READ_ARRAY;
    flash_status = 0;
    flash_erase_offset = 0;
    flash_erase_chip = false;
}

u8 Cartridge::get_cic_seed() const {
    switch (cic_type) {
        case CICType::CIC_6103: return 0x78;
        case CICType::CIC_6105: return 0x91;
        case CICType::CIC_6106: return 0x85;
        default: return 0x3F; // 6101, 6102
    }
}

u32 Cartridge::get_cic_id() const {
    switch (cic_type) {
        case CICType::CIC_6101: return 6101;
        case CICType::CIC_6103: return 6103;
        case CICType::CIC_6105: return 6105;
        case CICType::CIC_6106: return 6106;
        default: return 6102;
    }
}

u32 Cartridge::get_boot_address() const {
    u32 entry = entry_point ? entry_point : 0x80000400;
    if (cic_type == CICType::CIC_6103) entry -= 0x100000;
    if (cic_type == CICType::CIC_6106) entry -= 0x200000;
    return entry;
}

// ---- Cartridge domain 2: SRAM / FlashRAM --------------------------------------

void Cartridge::flash_command(u32 cmd) {
    switch (cmd >> 24) {
        case 0x3C: // chip erase (armed; 0x78 runs it)
            flash_erase_chip = true;
            break;
        case 0x4B: // sector erase: 128 pages of 128 bytes
            flash_erase_chip = false;
            flash_erase_offset = ((cmd & 0xFFFF) * 128) & ~0x3FFFu;
            break;
        case 0x78: // run the armed erase
            if (flash_erase_chip) {
                std::fill(sram.begin(), sram.end(), 0xFF);
            } else if (flash_erase_offset + 0x4000 <= sram.size()) {
                std::fill(sram.begin() + flash_erase_offset, sram.begin() + flash_erase_offset + 0x4000, 0xFF);
            }
            flash_status = 0x08;
            sram_dirty = true;
            break;
        case 0xA5: { // program the page buffer into a page
            const u32 off = (cmd & 0xFFFF) * 128;
            if (off + 128 <= sram.size()) std::copy(flash_buf.begin(), flash_buf.end(), sram.begin() + off);
            flash_status = 0x04;
            sram_dirty = true;
            break;
        }
        case 0xB4: flash_mode = FLASH_WRITE_BUFFER; break;
        case 0xD2: flash_mode = FLASH_STATUS; break;
        case 0xE1: flash_mode = FLASH_ID; break;
        case 0xF0: flash_mode = FLASH_READ_ARRAY; break;
        default: break;
    }
}

u32 Cartridge::read_bus32(u32 off) {
    if (save_type == SaveType::FLASHRAM_128K) return 0x11118000u | flash_status; // status register
    return (static_cast<u32>(read_sram(off)) << 24) | (static_cast<u32>(read_sram(off + 1)) << 16) |
           (static_cast<u32>(read_sram(off + 2)) << 8) | read_sram(off + 3);
}

void Cartridge::write_bus32(u32 off, u32 val) {
    if (save_type == SaveType::FLASHRAM_128K) {
        if (off == 0x10000) flash_command(val);
        else if (off == 0) flash_status = 0; // clear status
        return;
    }
    for (int i = 0; i < 4; ++i) write_sram(off + i, static_cast<u8>(val >> (24 - 8 * i)));
}

void Cartridge::dma_to_rdram(u32 off, u8* dst, u32 len) {
    if (save_type == SaveType::FLASHRAM_128K) {
        if (flash_mode == FLASH_READ_ARRAY) {
            // The flash is addressed in 16-bit units: libultra passes page * 64.
            const u32 base = off * 2;
            for (u32 i = 0; i < len; ++i) dst[i] = sram[(base + i) % sram.size()];
        } else {
            const u8 id[8] = {0x11, 0x11, 0x80, static_cast<u8>(flash_mode == FLASH_ID ? 0x01 : flash_status),
                              0x00, 0xC2, 0x00, 0x1E}; // Macronix MX29L1100
            for (u32 i = 0; i < len; ++i) dst[i] = id[i & 7];
        }
        return;
    }
    for (u32 i = 0; i < len; ++i) dst[i] = read_sram(off + i);
}

void Cartridge::dma_from_rdram(u32 off, const u8* src, u32 len) {
    if (save_type == SaveType::FLASHRAM_128K) {
        if (flash_mode == FLASH_WRITE_BUFFER)
            for (u32 i = 0; i < len && i < flash_buf.size(); ++i) flash_buf[i] = src[i];
        return;
    }
    for (u32 i = 0; i < len; ++i) write_sram(off + i, src[i]);
}

void Cartridge::save_backup() {
    if (mempak_dirty && !mempak_filepath.empty()) {
        std::ofstream pak_file(utf8_path(mempak_filepath), std::ios::binary);
        if (pak_file.is_open()) {
            pak_file.write(reinterpret_cast<const char*>(mempak.data()), mempak.size());
            mempak_dirty = false;
        }
    }
    if (!sram_dirty && !eeprom_dirty) return;
    if (save_filepath.empty()) return;

    std::ofstream file(utf8_path(save_filepath), std::ios::binary);
    if (!file.is_open()) return;

    if ((save_type == SaveType::SRAM_32K || save_type == SaveType::FLASHRAM_128K) && !sram.empty()) {
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
