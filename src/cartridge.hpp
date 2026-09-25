#pragma once

#include "common.hpp"
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

private:
    std::string save_filepath;
    bool use_save_file_{true};
    std::vector<u8> rom;
    std::vector<u8> sram;
    std::vector<u8> eeprom;
    bool sram_dirty{false};
    bool eeprom_dirty{false};

    u32 entry_point{0};
    u32 crc1{0};
    u32 crc2{0};
    std::string title;
    std::string game_code;
    CICType cic_type{CICType::CIC_6102};
    SaveType save_type{SaveType::EEPROM_4K};

    void detect_cic_and_save();
};
