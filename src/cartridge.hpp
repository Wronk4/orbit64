#pragma once

#include "common.hpp"
#include <array>
#include <string>
#include <vector>

enum class CICType {
    CIC_6101,
    CIC_6102,
    CIC_6103,
    CIC_6105,
    CIC_6106,
    UNKNOWN
};

enum class SaveType {
    NONE,
    EEPROM_4K,
    EEPROM_16K,
    SRAM_32K,
    FLASHRAM_128K
};

class Cartridge {
public:
    Cartridge();
    ~Cartridge();

    bool load_rom(const std::string& filepath);
    void save_backup();
    // Off: load_rom() neither reads nor later writes the .sav next to the ROM,
    // so a run always starts from blank save memory (reproducible test runs).
    void set_use_save_file(bool on) { use_save_file_ = on; }

    u8 read_rom(u32 addr) const;
    u16 read_rom16(u32 addr) const;
    u32 read_rom32(u32 addr) const;

    u8 read_sram(u32 addr) const;
    void write_sram(u32 addr, u8 val);

    // Cartridge domain 2 (0x08000000): SRAM, or FlashRAM and its command
    // interface. `off` is relative to 0x08000000.
    u32 read_bus32(u32 off);
    void write_bus32(u32 off, u32 val);
    void dma_to_rdram(u32 off, u8* dst, u32 len);      // PI read (cart -> RDRAM)
    void dma_from_rdram(u32 off, const u8* src, u32 len); // PI write (RDRAM -> cart)

    // What the CIC chip's boot code (IPL3) does differently per chip: the
    // seed it passes on, and where it copies the game's boot code and jumps
    // (6103 and 6106 subtract 1 or 2 MB from the header's entry point).
    u8 get_cic_seed() const;
    u32 get_boot_address() const;

    // EEPROM access
    void read_eeprom(u8 block, u8* out_data);
    void write_eeprom(u8 block, const u8* in_data);

    const std::vector<u8>& get_rom_data() const { return rom; }
    size_t get_rom_size() const { return rom.size(); }

    CICType get_cic_type() const { return cic_type; }
    SaveType get_save_type() const { return save_type; }

    u32 get_entry_point() const { return entry_point; }
    u32 get_crc1() const { return crc1; }
    u32 get_crc2() const { return crc2; }
    std::string get_title() const { return title; }
    std::string get_game_code() const { return game_code; }

    // Save states (savestate.hpp): the save memory, so the game finds what it
    // last wrote. A loaded state's save memory goes to the .sav file too.
    template <class S> void serialize(S& s) {
        s.fixed(sram);
        s.fixed(eeprom);
        s(flash_mode, flash_status, flash_erase_offset, flash_erase_chip, flash_buf);
        if constexpr (S::loading) sram_dirty = eeprom_dirty = true;
    }

private:
    std::string save_filepath;
    bool use_save_file_{true};
    std::vector<u8> rom;
    std::vector<u8> sram;
    std::vector<u8> eeprom;
    bool sram_dirty{false};
    bool eeprom_dirty{false};

    // FlashRAM (128 KB, kept in `sram`): the libultra osFlash* protocol.
    enum FlashMode : u8 { FLASH_READ_ARRAY, FLASH_STATUS, FLASH_ID, FLASH_WRITE_BUFFER };
    u8 flash_mode{FLASH_READ_ARRAY};
    u8 flash_status{0};        // low byte of the status register: 0x04 programmed, 0x08 erased
    u32 flash_erase_offset{0};
    bool flash_erase_chip{false};
    std::array<u8, 128> flash_buf{};
    void flash_command(u32 cmd);

    u32 entry_point{0};
    u32 crc1{0};
    u32 crc2{0};
    std::string title;
    std::string game_code;
    CICType cic_type{CICType::CIC_6102};
    SaveType save_type{SaveType::EEPROM_4K};

    void detect_cic_and_save();
};
