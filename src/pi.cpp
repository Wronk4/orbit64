#include "pi.hpp"
#include "mi.hpp"
#include "cartridge.hpp"
#include "jit/jit_invalidate.hpp"
#include <iostream>

PI::PI() {
    reset();
}

void PI::reset() {
    dram_addr = 0;
    cart_addr = 0;
    rd_len = 0;
    wr_len = 0;
    status = 0;
    dom1_lat = 0;
    dom1_pwd = 0;
    dom1_pgs = 0;
    dom1_rls = 0;
    dom2_lat = 0;
    dom2_pwd = 0;
    dom2_pgs = 0;
    dom2_rls = 0;
}

u32 PI::read_reg(u32 addr) const {
    u32 reg = (addr & 0x3F) >> 2;
    switch (reg) {
        case 0: return dram_addr;
        case 1: return cart_addr;
        case 2: return rd_len;
        case 3: return wr_len;
        case 4: return status;
        case 5: return dom1_lat;
        case 6: return dom1_pwd;
        case 7: return dom1_pgs;
        case 8: return dom1_rls;
        case 9: return dom2_lat;
        case 10: return dom2_pwd;
        case 11: return dom2_pgs;
        case 12: return dom2_rls;
        default: return 0;
    }
}

void PI::write_reg(u32 addr, u32 val, MI& mi, Cartridge& cart, u8* rdram, size_t rdram_size) {
    u32 reg = (addr & 0x3F) >> 2;
    switch (reg) {
        case 0: // PI_DRAM_ADDR_REG
            dram_addr = val & 0x00FFFFFE;
            break;
        case 1: // PI_CART_ADDR_REG
            cart_addr = val & 0xFFFFFFFE;
            break;
        case 2: // PI_RD_LEN_REG: RDRAM -> Cartridge (SRAM write)
            rd_len = val;
            execute_dma_write(mi, cart, rdram, rdram_size);
            break;
        case 3: // PI_WR_LEN_REG: Cartridge -> RDRAM (ROM/SRAM read)
            wr_len = val;
            execute_dma_read(mi, cart, rdram, rdram_size);
            break;
        case 4: // PI_STATUS_REG
            if (val & (1 << 1)) {
                // Clear PI interrupt
                mi.clear_interrupt(MIInterrupt::PI);
            }
            if (val & (1 << 0)) {
                // Reset PI controller
                status = 0;
            }
            break;
        case 5: dom1_lat = val & 0xFF; break;
        case 6: dom1_pwd = val & 0xFF; break;
        case 7: dom1_pgs = val & 0x0F; break;
        case 8: dom1_rls = val & 0x03; break;
        case 9: dom2_lat = val & 0xFF; break;
        case 10: dom2_pwd = val & 0xFF; break;
        case 11: dom2_pgs = val & 0x0F; break;
        case 12: dom2_rls = val & 0x03; break;
    }
}

void PI::execute_dma_read(MI& mi, Cartridge& cart, u8* rdram, size_t rdram_size) {
    u32 len = (wr_len & 0x00FFFFFF) + 1;
    if (len & 1) len++;

    u32 cur_dram = dram_addr & (rdram_size - 1);
    const auto& rom = cart.get_rom_data();

    if (cart_addr >= 0x08000000 && cart_addr < 0x10000000) {
        // Read from Cartridge SRAM / FlashRAM
        u32 sram_offset = cart_addr - 0x08000000;
        for (u32 i = 0; i < len; ++i) {
            u8 byte = cart.read_sram(sram_offset + i);
            if (cur_dram + i < rdram_size) {
                rdram[cur_dram + i] = byte;
            }
        }
    } else {
        // Read from Cartridge ROM
        u32 rom_offset = (cart_addr >= 0x10000000) ? (cart_addr - 0x10000000) : cart_addr;
        for (u32 i = 0; i < len; ++i) {
            u8 byte = (rom_offset + i < rom.size()) ? rom[rom_offset + i] : 0;
            if (cur_dram + i < rdram_size) {
                rdram[cur_dram + i] = byte;
            }
        }
    }
    // Overlay loads come through here: drop JIT blocks compiled from what
    // was just overwritten.
    jit::notify_code_write(cur_dram, static_cast<u32>(std::min<size_t>(len, rdram_size - cur_dram)));

    dram_addr += len;
    cart_addr += len;

    // Raise PI interrupt in MI
    mi.raise_interrupt(MIInterrupt::PI);
}

void PI::execute_dma_write(MI& mi, Cartridge& cart, const u8* rdram, size_t rdram_size) {
    u32 len = (rd_len & 0x00FFFFFF) + 1;
    if (len & 1) len++;

    u32 cur_dram = dram_addr & (rdram_size - 1);

    u32 sram_offset = (cart_addr >= 0x08000000) ? (cart_addr - 0x08000000) : cart_addr;
    for (u32 i = 0; i < len; ++i) {
        u8 byte = (cur_dram + i < rdram_size) ? rdram[cur_dram + i] : 0;
        cart.write_sram(sram_offset + i, byte);
    }

    dram_addr += len;
    cart_addr += len;

    // Raise PI interrupt in MI
    mi.raise_interrupt(MIInterrupt::PI);
}
