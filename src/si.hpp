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

    // CPU cycles until the running DMA ends (0: none is running). The emulator never runs the CPU past it.
    u32 cycles_left() const { return dma_cycles_left; }
    // Time has passed: ends a DMA whose time has run out (clears the busy bit, sets the interrupt flag, raises the
    // SI interrupt). Returns true if one ended.
    bool step(u32 cycles, MI& mi);

    // Save states (savestate.hpp).
    template <class S> void serialize(S& s) { s(dram_addr, pif_addr_rd64b, pif_addr_wr64b, status, dma_cycles_left); }

private:
    u32 dram_addr{0};
    u32 pif_addr_rd64b{0};
    u32 pif_addr_wr64b{0};
    u32 status{0};
    u32 dma_cycles_left{0}; // the data is copied when the DMA starts; the interrupt comes when this runs out
};
