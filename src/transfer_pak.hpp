#pragma once
// Transfer Pak: a Game Boy / Game Boy Color cartridge in the controller's
// accessory slot (Pokémon Stadium 1 & 2, Mario Golf, Mario Tennis, Perfect
// Dark...). The N64 sees the cartridge through a 16 KB window, so only the
// cartridge itself is emulated - its ROM, battery RAM, memory bank
// controller (MBC1/2/3/5, MBC3's real-time clock) - not a Game Boy.
//
// The cartridge's save is the .sav next to its ROM, the file Game Boy
// emulators use; an MBC3 clock is kept in the common 48-byte footer after the
// RAM (VBA-M / mGBA / SameBoy layout).

#include "common.hpp"
#include <memory>
#include <string>
#include <vector>

class GbCart {
public:
    // Loads a .gb / .gbc image and its save. False (with a reason) if it
    // can't be used.
    bool load(const std::string& rom_path, std::string& error);
    // Writes the save if the game changed it.
    void save();

    u8 read(u16 addr);
    void write(u16 addr, u8 val);
    void reset();

    const std::string& title() const { return title_; }

    // Save states: the bank controller's registers. The RAM is the
    // cartridge's own battery memory, like a real cartridge it isn't part
    // of the N64's state.
    template <class S> void serialize(S& s) {
        s(rom_bank_, ram_bank_, ram_enabled_, mbc1_mode_, latch_prev_, rtc_latched_);
    }

private:
    enum class Mbc : u8 { None, Mbc1, Mbc2, Mbc3, Mbc5 };

    // MBC3 clock: seconds since day 0, running unless halted.
    void rtc_update();
    void rtc_fields(u32 out[5]) const; // S, M, H, DL, DH from rtc_seconds_
    u8 rtc_read(u8 reg);
    void rtc_write(u8 reg, u8 val);

    std::string rom_path_, save_path_, title_;
    std::vector<u8> rom_, ram_;
    std::vector<u8> save_tail_; // bytes after the RAM in the .sav (kept as they were, or the clock)
    Mbc mbc_ = Mbc::None;
    bool battery_ = false, has_rtc_ = false, dirty_ = false;

    u16 rom_bank_ = 1;
    u8 ram_bank_ = 0;
    u8 ram_enabled_ = 0;
    u8 mbc1_mode_ = 0;
    u8 latch_prev_ = 0xFF;
    u32 rtc_latched_[5] = {};
    u64 rtc_seconds_ = 0;
    bool rtc_halted_ = false, rtc_carry_ = false;
    s64 rtc_updated_ = 0; // host Unix time rtc_seconds_ was brought up to
};

class TransferPak {
public:
    TransferPak() = default;
    TransferPak(const TransferPak&) = delete;
    TransferPak& operator=(const TransferPak&) = delete;
    ~TransferPak() { flush(); }

    // Puts a Game Boy cartridge in ("" takes it out). The previous one's save
    // is written first. False (with a reason) if the image can't be used; the
    // pak is then empty.
    bool insert(const std::string& gb_rom_path, std::string& error);
    const std::string& rom_path() const { return path_; }
    bool has_cart() const { return cart_ != nullptr; }
    void flush() { if (cart_) cart_->save(); }
    void reset();

    // Accessory slot accesses: 32-byte blocks at `addr` (0x0000-0xFFE0).
    void read(u16 addr, u8* out);
    void write(u16 addr, const u8* in);

    template <class S> void serialize(S& s) {
        s(enabled_, bank_, access_mode_, access_changed_);
        GbCart none;
        if (cart_) cart_->serialize(s); else none.serialize(s);
    }

private:
    // Status byte at 0xB000 (libultra's OS_GBPAK_*): cartridge in and access
    // mode 0 / 1 (power + reset released), or no cartridge.
    static constexpr u8 kMode0 = 0x80, kMode1 = 0x89, kNoCart = 0x40;

    std::string path_;
    std::unique_ptr<GbCart> cart_;
    u8 enabled_ = 0;
    u8 bank_ = 0;             // which 16 KB of the Game Boy's address space 0xC000 shows
    u8 access_mode_ = kNoCart;
    u8 access_changed_ = 0x44; // reported once: reset detected (0x04)
};
