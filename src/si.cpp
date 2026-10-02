#include "si.hpp"
#include "mi.hpp"
#include "pif.hpp"
#include "jit/jit_invalidate.hpp"
#include <cstring>
#include <iostream>

SI::SI() {
    reset();
}

void SI::reset() {
    dram_addr = 0;
    pif_addr_rd64b = 0;
    pif_addr_wr64b = 0;
    status = 0;
}

u32 SI::read_reg(u32 addr) const {
    u32 reg = (addr & 0x1F) >> 2;
    switch (reg) {
        case 0: return dram_addr;
        case 1: return pif_addr_rd64b;
        case 4: return pif_addr_wr64b;
        case 6: return status;
        default: return 0;
    }
}

void SI::write_reg(u32 addr, u32 val, MI& mi, PIF& pif, Controller controllers[4], Cartridge& cart, u8* rdram, size_t rdram_size) {
    u32 reg = (addr & 0x1F) >> 2;
    switch (reg) {
        case 0: // SI_DRAM_ADDR_REG
            dram_addr = val & 0x00FFFFFE;
            break;
        case 1: { // SI_PIF_ADDR_RD64B_REG (PIF to RDRAM)
            pif_addr_rd64b = val;
            // Execute handshakes before copying to RDRAM
            pif.process_commands(controllers, cart);

            u32 cur_dram = dram_addr & (rdram_size - 1);
            const u8* pif_ram = pif.get_ram();
            for (u32 i = 0; i < 64 && cur_dram + i < rdram_size; ++i) {
                rdram[cur_dram + i] = pif_ram[i];
            }
            jit::notify_code_write(cur_dram, static_cast<u32>(std::min<size_t>(64, rdram_size - cur_dram)));

            status |= (1 << 12); // Interrupt set
            mi.raise_interrupt(MIInterrupt::SI);
            break;
        }
        case 4: { // SI_PIF_ADDR_WR64B_REG (RDRAM to PIF)
            pif_addr_wr64b = val;

            u32 cur_dram = dram_addr & (rdram_size - 1);
            u8* pif_ram = pif.get_ram();
            jit::notify_read(cur_dram, static_cast<u32>(std::min<size_t>(64, rdram_size - cur_dram)));
            for (u32 i = 0; i < 64 && cur_dram + i < rdram_size; ++i) {
                pif_ram[i] = rdram[cur_dram + i];
            }

            // Process commands immediately
            pif.process_commands(controllers, cart);

            status |= (1 << 12); // Interrupt set
            mi.raise_interrupt(MIInterrupt::SI);
            break;
        }
        case 6: // SI_STATUS_REG
            // Writing any value clears SI interrupt
            status &= ~(1 << 12);
            mi.clear_interrupt(MIInterrupt::SI);
            break;
    }
}
