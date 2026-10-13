#pragma once

#include "common.hpp"

class MI;
class Cartridge;

class PI {
public:
    PI();

    void reset();

    u32 read_reg(u32 addr) const;
    void write_reg(u32 addr, u32 val, MI& mi, Cartridge& cart, u8* rdram, size_t rdram_size);

    u32 get_status() const { return status; }

    // CPU cycles until the running DMA ends (0: none is running). The emulator never runs the CPU past it.
    u32 cycles_left() const { return dma_cycles_left; }
    // Time has passed: ends a DMA whose time has run out (clears the busy bit, raises the PI interrupt).
    // Returns true if one ended.
    bool step(u32 cycles, MI& mi);

    // Save states (savestate.hpp).
    template <class S> void serialize(S& s) {
        s(dram_addr, cart_addr, rd_len, wr_len, status, dom1_lat, dom1_pwd, dom1_pgs, dom1_rls, dom2_lat, dom2_pwd,
          dom2_pgs, dom2_rls, dma_cycles_left);
    }

private:
    u32 dram_addr{0};
    u32 cart_addr{0};
    u32 rd_len{0};
    u32 wr_len{0};
    u32 status{0};

    u32 dom1_lat{0};
    u32 dom1_pwd{0};
    u32 dom1_pgs{0};
    u32 dom1_rls{0};
    u32 dom2_lat{0};
    u32 dom2_pwd{0};
    u32 dom2_pgs{0};
    u32 dom2_rls{0};

    // CPU cycles until the running DMA is over; 0 when none is. The data is copied when the DMA starts.
    u32 dma_cycles_left{0};
    void begin_dma(u32 len);

    void execute_dma_read(MI& mi, Cartridge& cart, u8* rdram, size_t rdram_size);
    void execute_dma_write(MI& mi, Cartridge& cart, const u8* rdram, size_t rdram_size);
};
