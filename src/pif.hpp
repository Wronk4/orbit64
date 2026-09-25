#pragma once

#include "common.hpp"
#include "controller.hpp"
#include "cartridge.hpp"

class PIF {
public:
    PIF();

    void reset(CICType cic);

    u8 read_ram(u32 addr) const;
    void write_ram(u32 addr, u8 val);

    u8 read_rom(u32 addr) const;

    u8* get_ram() { return ram.data(); }
    const u8* get_ram() const { return ram.data(); }

    void process_commands(Controller controllers[4], Cartridge& cartridge);

    // Save states (savestate.hpp).
    template <class S> void serialize(S& s) { s(ram, cic_type); }

private:
    std::array<u8, PIF_ROM_SIZE> rom{};
    std::array<u8, PIF_RAM_SIZE> ram{};
    CICType cic_type{CICType::CIC_6102};

    void process_cic_challenge(Cartridge& cartridge);
    void solve_cic_6105_challenge();
};
