#include "transfer_pak.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

std::filesystem::path utf8_path(const std::string& s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

s64 unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

u64 get_le(const u8* p, int n) {
    u64 v = 0;
    for (int i = n - 1; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

void put_le(u8* p, u64 v, int n) {
    for (int i = 0; i < n; ++i) p[i] = static_cast<u8>(v >> (8 * i));
}

} // namespace

// ---- Game Boy cartridge ------------------------------------------------------

bool GbCart::load(const std::string& rom_path, std::string& error) {
    std::ifstream f(utf8_path(rom_path), std::ios::binary);
    if (!f.is_open()) {
        error = "The Game Boy ROM could not be opened.";
        return false;
    }
    rom_.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    if (rom_.size() < 0x8000 || rom_.size() > (8u << 20)) {
        error = "The file is not a Game Boy ROM.";
        return false;
    }

    const u8 type = rom_[0x147];
    switch (type) {
        case 0x00: case 0x08: case 0x09: mbc_ = Mbc::None; break;
        case 0x01: case 0x02: case 0x03: mbc_ = Mbc::Mbc1; break;
        case 0x05: case 0x06: mbc_ = Mbc::Mbc2; break;
        case 0x0F: case 0x10: case 0x11: case 0x12: case 0x13: mbc_ = Mbc::Mbc3; break;
        case 0x19: case 0x1A: case 0x1B: case 0x1C: case 0x1D: case 0x1E: mbc_ = Mbc::Mbc5; break;
        default: {
            char buf[96];
            std::snprintf(buf, sizeof buf, "This Game Boy cartridge type (0x%02X) isn't supported by the Transfer Pak yet.", type);
            error = buf;
            return false;
        }
    }
    battery_ = type == 0x03 || type == 0x06 || type == 0x09 || type == 0x0F || type == 0x10 || type == 0x13 ||
               type == 0x1B || type == 0x1E;
    has_rtc_ = type == 0x0F || type == 0x10;

    static const u32 kRamSizes[] = {0, 0x800, 0x2000, 0x8000, 0x20000, 0x10000};
    u32 ram_size = rom_[0x149] < 6 ? kRamSizes[rom_[0x149]] : 0;
    if (mbc_ == Mbc::Mbc2) ram_size = 512; // 512 x 4 bits, one per byte here
    ram_.assign(ram_size, 0xFF);

    title_.clear();
    for (int i = 0x134; i < 0x144 && rom_[i] >= 0x20 && rom_[i] < 0x7F; ++i) title_.push_back(static_cast<char>(rom_[i]));

    rom_path_ = rom_path;
    const size_t dot = rom_path.find_last_of('.');
    const size_t slash = rom_path.find_last_of("/\\");
    save_path_ = (dot != std::string::npos && (slash == std::string::npos || dot > slash) ? rom_path.substr(0, dot) : rom_path) + ".sav";

    rtc_seconds_ = 0;
    rtc_halted_ = rtc_carry_ = false;
    rtc_updated_ = unix_now();
    save_tail_.clear();
    if (battery_ && (!ram_.empty() || has_rtc_)) {
        std::ifstream sf(utf8_path(save_path_), std::ios::binary);
        if (sf.is_open()) {
            std::vector<u8> data((std::istreambuf_iterator<char>(sf)), std::istreambuf_iterator<char>());
            std::copy_n(data.begin(), std::min(data.size(), ram_.size()), ram_.begin());
            if (data.size() > ram_.size()) save_tail_.assign(data.begin() + ram_.size(), data.end());
        }
        // The clock footer: live S M H DL DH, latched S M H DL DH (u32
        // each), then the Unix time it was saved at (u64, or u32 in older files).
        if (has_rtc_ && (save_tail_.size() == 48 || save_tail_.size() == 44)) {
            const u8* t = save_tail_.data();
            const u64 days = get_le(t + 12, 4) | ((get_le(t + 16, 4) & 1) << 8);
            rtc_seconds_ = get_le(t, 4) % 60 + (get_le(t + 4, 4) % 60) * 60 + (get_le(t + 8, 4) % 24) * 3600 + days * 86400;
            rtc_halted_ = (get_le(t + 16, 4) & 0x40) != 0;
            rtc_carry_ = (get_le(t + 16, 4) & 0x80) != 0;
            for (int i = 0; i < 5; ++i) rtc_latched_[i] = static_cast<u32>(get_le(t + 20 + i * 4, 4));
            rtc_updated_ = static_cast<s64>(get_le(t + 40, save_tail_.size() == 48 ? 8 : 4));
            rtc_update();
        } else if (has_rtc_ && save_tail_.empty()) {
            save_tail_.assign(48, 0);
        }
    }
    dirty_ = false;
    reset();
    std::cout << "[TransferPak] Game Boy cartridge \"" << title_ << "\", type 0x" << std::hex << int(type) << std::dec
              << ", " << rom_.size() / 1024 << " KB ROM, " << ram_.size() << " bytes RAM\n";
    return true;
}

void GbCart::save() {
    if (!dirty_ || !battery_ || save_path_.empty()) return;
    std::vector<u8> data = ram_;
    if (has_rtc_ && (save_tail_.size() == 48 || save_tail_.size() == 44)) {
        rtc_update();
        u32 live[5];
        rtc_fields(live);
        u8* t = save_tail_.data();
        for (int i = 0; i < 5; ++i) put_le(t + i * 4, live[i], 4);
        for (int i = 0; i < 5; ++i) put_le(t + 20 + i * 4, rtc_latched_[i], 4);
        put_le(t + 40, static_cast<u64>(rtc_updated_), save_tail_.size() == 48 ? 8 : 4);
    }
    data.insert(data.end(), save_tail_.begin(), save_tail_.end());
    std::ofstream f(utf8_path(save_path_), std::ios::binary);
    if (!f.is_open()) return;
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
    dirty_ = false;
}

void GbCart::reset() {
    rom_bank_ = 1;
    ram_bank_ = 0;
    ram_enabled_ = 0;
    mbc1_mode_ = 0;
    latch_prev_ = 0xFF;
}

void GbCart::rtc_update() {
    const s64 now = unix_now();
    if (!rtc_halted_ && now > rtc_updated_) rtc_seconds_ += static_cast<u64>(now - rtc_updated_);
    rtc_updated_ = now;
    if (rtc_seconds_ >= 512ull * 86400) {
        rtc_seconds_ %= 512ull * 86400;
        rtc_carry_ = true;
    }
}

void GbCart::rtc_fields(u32 out[5]) const {
    const u64 days = rtc_seconds_ / 86400;
    out[0] = static_cast<u32>(rtc_seconds_ % 60);
    out[1] = static_cast<u32>(rtc_seconds_ / 60 % 60);
    out[2] = static_cast<u32>(rtc_seconds_ / 3600 % 24);
    out[3] = static_cast<u32>(days & 0xFF);
    out[4] = static_cast<u32>(((days >> 8) & 1) | (rtc_halted_ ? 0x40 : 0) | (rtc_carry_ ? 0x80 : 0));
}

u8 GbCart::rtc_read(u8 reg) { return static_cast<u8>(rtc_latched_[reg - 0x08]); }

void GbCart::rtc_write(u8 reg, u8 val) {
    rtc_update();
    u32 f[5];
    rtc_fields(f);
    f[reg - 0x08] = val;
    rtc_halted_ = (f[4] & 0x40) != 0;
    rtc_carry_ = (f[4] & 0x80) != 0;
    const u64 days = f[3] | ((f[4] & 1u) << 8);
    rtc_seconds_ = (f[0] % 60) + (f[1] % 60) * 60 + (f[2] % 24) * 3600 + days * 86400;
    dirty_ = true;
}

u8 GbCart::read(u16 addr) {
    const size_t banks = rom_.size() / 0x4000;
    if (addr < 0x4000) {
        size_t bank = 0;
        if (mbc_ == Mbc::Mbc1 && mbc1_mode_) bank = (ram_bank_ & 3u) << 5;
        return rom_[(bank % banks) * 0x4000 + addr];
    }
    if (addr < 0x8000) {
        size_t bank = rom_bank_;
        if (mbc_ == Mbc::Mbc1) bank = (rom_bank_ & 0x1F) | ((ram_bank_ & 3u) << 5);
        return rom_[(bank % banks) * 0x4000 + (addr - 0x4000)];
    }
    if (addr >= 0xA000 && addr < 0xC000) {
        if (mbc_ != Mbc::None && !ram_enabled_) return 0xFF;
        if (mbc_ == Mbc::Mbc3 && ram_bank_ >= 0x08 && ram_bank_ <= 0x0C) return has_rtc_ ? rtc_read(ram_bank_) : 0xFF;
        if (mbc_ == Mbc::Mbc2) return static_cast<u8>(0xF0 | (ram_[addr & 0x1FF] & 0x0F));
        if (ram_.empty()) return 0xFF;
        size_t bank = (mbc_ == Mbc::Mbc1 && !mbc1_mode_) ? 0 : ram_bank_;
        return ram_[(bank * 0x2000 + (addr - 0xA000)) % ram_.size()];
    }
    return 0xFF; // not the cartridge's
}

void GbCart::write(u16 addr, u8 val) {
    switch (mbc_) {
        case Mbc::None: break;
        case Mbc::Mbc1:
            if (addr < 0x2000) ram_enabled_ = (val & 0x0F) == 0x0A;
            else if (addr < 0x4000) rom_bank_ = (val & 0x1F) ? (val & 0x1F) : 1;
            else if (addr < 0x6000) ram_bank_ = val & 3;
            else if (addr < 0x8000) mbc1_mode_ = val & 1;
            break;
        case Mbc::Mbc2:
            if (addr < 0x4000) {
                if (addr & 0x100) rom_bank_ = (val & 0x0F) ? (val & 0x0F) : 1;
                else ram_enabled_ = (val & 0x0F) == 0x0A;
            }
            break;
        case Mbc::Mbc3:
            if (addr < 0x2000) ram_enabled_ = (val & 0x0F) == 0x0A;
            else if (addr < 0x4000) rom_bank_ = (val & 0x7F) ? (val & 0x7F) : 1;
            else if (addr < 0x6000) ram_bank_ = val & 0x0F;
            else if (addr < 0x8000) {
                // Latch the clock on a 0 -> 1 write.
                if (has_rtc_ && latch_prev_ == 0 && val == 1) {
                    rtc_update();
                    rtc_fields(rtc_latched_);
                }
                latch_prev_ = val;
            }
            break;
        case Mbc::Mbc5:
            if (addr < 0x2000) ram_enabled_ = (val & 0x0F) == 0x0A;
            else if (addr < 0x3000) rom_bank_ = static_cast<u16>((rom_bank_ & 0x100) | val);
            else if (addr < 0x4000) rom_bank_ = static_cast<u16>((rom_bank_ & 0xFF) | ((val & 1) << 8));
            else if (addr < 0x6000) ram_bank_ = val & 0x0F;
            break;
    }
    if (addr >= 0xA000 && addr < 0xC000) {
        if (mbc_ != Mbc::None && !ram_enabled_) return;
        if (mbc_ == Mbc::Mbc3 && ram_bank_ >= 0x08 && ram_bank_ <= 0x0C) {
            if (has_rtc_) rtc_write(ram_bank_, val);
            return;
        }
        if (ram_.empty()) return;
        if (mbc_ == Mbc::Mbc2) {
            ram_[addr & 0x1FF] = val & 0x0F;
        } else {
            size_t bank = (mbc_ == Mbc::Mbc1 && !mbc1_mode_) ? 0 : ram_bank_;
            ram_[(bank * 0x2000 + (addr - 0xA000)) % ram_.size()] = val;
        }
        dirty_ = true;
    }
}

// ---- Transfer Pak -------------------------------------------------------------

bool TransferPak::insert(const std::string& gb_rom_path, std::string& error) {
    flush();
    cart_.reset();
    path_ = gb_rom_path;
    bool ok = true;
    if (!gb_rom_path.empty()) {
        auto cart = std::make_unique<GbCart>();
        if (cart->load(gb_rom_path, error)) cart_ = std::move(cart);
        else ok = false;
    }
    reset();
    return ok;
}

void TransferPak::reset() {
    enabled_ = 0;
    bank_ = 0;
    access_mode_ = cart_ ? kMode0 : kNoCart;
    access_changed_ = 0x44;
    if (cart_) cart_->reset();
}

// The pak's address space (libultra's osGbpak*): 0x8000 power (0x84 on,
// 0xFE off), 0xA000 which 16 KB of the Game Boy's addresses 0xC000-0xFFFF
// show, 0xB000 status / access mode, 0xC000 that window.
void TransferPak::read(u16 addr, u8* out) {
    if (addr >= 0x8000 && addr < 0x9000) {
        std::fill_n(out, 32, enabled_ ? 0x84 : 0x00);
    } else if (addr >= 0xB000 && addr < 0xC000 && enabled_) {
        std::fill_n(out, 32, access_mode_);
        if (access_mode_ != kNoCart) out[0] |= access_changed_;
        access_changed_ = 0;
    } else if (addr >= 0xC000 && enabled_ && cart_) {
        const u16 gb = static_cast<u16>((addr - 0xC000) + (bank_ & 3) * 0x4000);
        for (int i = 0; i < 32; ++i) out[i] = cart_->read(static_cast<u16>(gb + i));
    } else {
        std::fill_n(out, 32, 0x00);
    }
}

void TransferPak::write(u16 addr, const u8* in) {
    if (addr >= 0x8000 && addr < 0x9000) {
        if (in[0] == 0xFE) enabled_ = 0;
        else if (in[0] == 0x84) enabled_ = 1;
    } else if (!enabled_) {
        return;
    } else if (addr >= 0xA000 && addr < 0xB000) {
        bank_ = in[0];
    } else if (addr >= 0xB000 && addr < 0xC000) {
        if (!cart_) return;
        access_mode_ = (in[0] & 1) ? kMode1 : kMode0;
        if (access_mode_ == kMode1) access_changed_ = 0x04;
    } else if (addr >= 0xC000 && cart_) {
        const u16 gb = static_cast<u16>((addr - 0xC000) + (bank_ & 3) * 0x4000);
        for (int i = 0; i < 32; ++i) cart_->write(static_cast<u16>(gb + i), in[i]);
    }
}
