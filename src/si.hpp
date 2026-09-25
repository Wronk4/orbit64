#pragma once

#include "common.hpp"

class MI;
class PIF;
class Cartridge;
class Controller;

class SI {
public:
    SI();

    void reset();

    u32 read_reg(u32 addr) const;
    void write_reg(u32 addr, u32 val, MI& mi, PIF& pif, Controller controllers[4], Cartridge& cart, u8* rdram, size_t rdram_size);

    u32 get_status() const { return status; }

    // Save states (savestate.hpp).
    template <class S> void serialize(S& s) { s(dram_addr, pif_addr_rd64b, pif_addr_wr64b, status); }

private:
    u32 dram_addr{0};
    u32 pif_addr_rd64b{0};
    u32 pif_addr_wr64b{0};
    u32 status{0};
};
