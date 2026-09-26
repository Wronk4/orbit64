#include "bus.hpp"
#include "cartridge.hpp"
#include "pif.hpp"
#include "controller.hpp"
#include "mi.hpp"
#include "vi.hpp"
#include "ai.hpp"
#include "pi.hpp"
#include "si.hpp"
#include "rsp.hpp"
#include "rdp.hpp"
#include "jit/jit_invalidate.hpp"
#include <iostream>

Bus::Bus(Cartridge& cart, PIF& pif, Controller controllers[4],
         MI& mi, VI& vi, AI& ai, PI& pi, SI& si, RSP& rsp, RDP& rdp)
    : cart(cart), pif(pif), controllers(controllers),
      mi(mi), vi(vi), ai(ai), pi(pi), si(si), rsp(rsp), rdp(rdp) {
    rdram.resize(RDRAM_SIZE, 0);
    reset();
}

void Bus::reset() {
    std::fill(rdram.begin(), rdram.end(), 0);
    for (auto& entry : tlb_entries) entry = {};
    tlb_gen_++;
    ri_mode = 0;
    ri_config = 0;
    ri_current_load = 0;
    ri_select = 0x0000000E;
    ri_refresh = 0;
    ri_latency = 0;
    ri_error = 0;
    ri_werror = 0;
}

void Bus::set_tlb_entry(size_t index, const TLBEntry& entry) {
    tlb_gen_++;
    if (index < tlb_entries.size()) {
        tlb_entries[index] = entry;
    }
}

const TLBEntry& Bus::get_tlb_entry(size_t index) const {
    static TLBEntry dummy{};
    if (index < tlb_entries.size()) return tlb_entries[index];
    return dummy;
}

TLBResult Bus::translate_vaddr(u64 vaddr, u32& paddr, bool is_write, u8 current_asid) {
    u32 va = static_cast<u32>(vaddr);

    // KSEG0: Direct unmapped cached (0x80000000 - 0x9FFFFFFF)
    if (va >= 0x80000000 && va < 0xA0000000) {
        paddr = va & 0x1FFFFFFF;
        return TLBResult::SUCCESS;
    }

    // KSEG1: Direct unmapped uncached (0xA0000000 - 0xBFFFFFFF)
    if (va >= 0xA0000000 && va < 0xC0000000) {
        paddr = va & 0x1FFFFFFF;
        return TLBResult::SUCCESS;
    }

    // Mapped segments: KUSEG (0x00000000-0x7FFFFFFF), KSSEG (0xC0000000-0xDFFFFFFF), KSEG3 (0xE0000000-0xFFFFFFFF) via TLB
    for (const auto& entry : tlb_entries) {
        if (!entry.initialized) continue;
        u32 mask = 0xFFFFE000 & ~entry.page_mask;
        if ((va & mask) == (static_cast<u32>(entry.entry_hi) & mask)) {
            bool is_global = (entry.entry_lo0 & 1) && (entry.entry_lo1 & 1);
            if (!is_global && ((entry.entry_hi & 0xFF) != (current_asid & 0xFF))) {
                continue;
            }

            u32 page_size = ((entry.page_mask >> 13) + 1) * 4096;
            bool odd = (va & page_size) != 0;
            u64 lo = odd ? entry.entry_lo1 : entry.entry_lo0;

            if (!(lo & 2)) {
                // Invalid page
                return TLBResult::INVALID;
            }
            if (is_write && !(lo & 4)) {
                // Dirty bit not set (write protection)
                return TLBResult::MODIFIED;
            }

            paddr = static_cast<u32>(((lo >> 6) << 12) & ~(page_size - 1)) | (va & (page_size - 1));
            return TLBResult::SUCCESS;
        }
    }

    return TLBResult::MISS;
}

u8 Bus::read8(u32 paddr) {
    if (paddr < ram_limit_) {
        return rdram[paddr];
    } else if (paddr >= 0x04000000 && paddr < 0x04001000) {
        return rsp.read_dmem(paddr);
    } else if (paddr >= 0x04001000 && paddr < 0x04002000) {
        return rsp.read_imem(paddr);
    } else if (paddr >= 0x08000000 && paddr < 0x08800000) {
        return cart.read_sram(paddr - 0x08000000);
    } else if (paddr >= 0x10000000 && paddr < 0x1FBFFFFF) {
        return cart.read_rom(paddr - 0x10000000);
    } else if (paddr >= 0x1FC00000 && paddr < 0x1FC007C0) {
        return pif.read_rom(paddr - 0x1FC00000);
    } else if (paddr >= 0x1FC007C0 && paddr < 0x1FC00800) {
        return pif.read_ram(paddr - 0x1FC007C0);
    } else {
        // Handle 32-bit MMIO byte read
        u32 aligned_addr = paddr & ~3;
        u32 shift = (3 - (paddr & 3)) * 8;
        return (read32(aligned_addr) >> shift) & 0xFF;
    }
}

u16 Bus::read16(u32 paddr) {
    if (paddr + 1 < ram_limit_) {
        return (static_cast<u16>(rdram[paddr]) << 8) | static_cast<u16>(rdram[paddr + 1]);
    }
    return (static_cast<u16>(read8(paddr)) << 8) | static_cast<u16>(read8(paddr + 1));
}

u32 Bus::read32(u32 paddr) {
    if (paddr + 3 < ram_limit_) {
        return (static_cast<u32>(rdram[paddr + 0]) << 24) |
               (static_cast<u32>(rdram[paddr + 1]) << 16) |
               (static_cast<u32>(rdram[paddr + 2]) << 8)  |
                static_cast<u32>(rdram[paddr + 3]);
    } else if (paddr >= 0x04000000 && paddr < 0x04001000) {
        u32 off = paddr & 0xFFC;
        return (static_cast<u32>(rsp.read_dmem(off + 0)) << 24) |
               (static_cast<u32>(rsp.read_dmem(off + 1)) << 16) |
               (static_cast<u32>(rsp.read_dmem(off + 2)) << 8)  |
                static_cast<u32>(rsp.read_dmem(off + 3));
    } else if (paddr >= 0x04001000 && paddr < 0x04002000) {
        u32 off = paddr & 0xFFC;
        return (static_cast<u32>(rsp.read_imem(off + 0)) << 24) |
               (static_cast<u32>(rsp.read_imem(off + 1)) << 16) |
               (static_cast<u32>(rsp.read_imem(off + 2)) << 8)  |
                static_cast<u32>(rsp.read_imem(off + 3));
    } else if (paddr >= 0x04040000 && paddr < 0x040FFFFF) {
        return rsp.read_reg(paddr);
    } else if (paddr >= 0x04100000 && paddr < 0x041FFFFF) {
        return rdp.read_dpc_reg(paddr);
    } else if (paddr >= 0x04200000 && paddr < 0x042FFFFF) {
        return rdp.read_dps_reg(paddr);
    } else if (paddr >= 0x04300000 && paddr < 0x043FFFFF) {
        return mi.read_reg(paddr);
    } else if (paddr >= 0x04400000 && paddr < 0x044FFFFF) {
        return vi.read_reg(paddr);
    } else if (paddr >= 0x04500000 && paddr < 0x045FFFFF) {
        return ai.read_reg(paddr);
    } else if (paddr >= 0x04600000 && paddr < 0x046FFFFF) {
        return pi.read_reg(paddr);
    } else if (paddr >= 0x04700000 && paddr < 0x047FFFFF) {
        u32 reg = (paddr & 0x1F) >> 2;
        switch (reg) {
            case 0: return ri_mode;
            case 1: return ri_config;
            case 2: return ri_current_load;
            case 3: return ri_select;
            case 4: return ri_refresh;
            case 5: return ri_latency;
            case 6: return ri_error;
            case 7: return ri_werror;
            default: return 0;
        }
    } else if (paddr >= 0x04800000 && paddr < 0x048FFFFF) {
        return si.read_reg(paddr);
    } else if (paddr >= 0x05000000 && paddr < 0x08000000) {
        // Cartridge domain 2 below SRAM is the 64DD, which isn't there: open
        // bus, i.e. the address's low half twice. 64DD-aware games (F-Zero X)
        // take anything else for a drive and wait for it forever.
        return ((paddr & 0xFFFF) << 16) | (paddr & 0xFFFF);
    } else if (paddr >= 0x08000000 && paddr < 0x10000000) {
        return cart.read_bus32(paddr - 0x08000000); // SRAM / FlashRAM status
    } else if (paddr >= 0x10000000 && paddr < 0x1FBFFFFF) {
        return cart.read_rom32(paddr - 0x10000000);
    } else if (paddr >= 0x1FC007C0 && paddr < 0x1FC00800) {
        u32 off = paddr - 0x1FC007C0;
        return (static_cast<u32>(pif.read_ram(off + 0)) << 24) |
               (static_cast<u32>(pif.read_ram(off + 1)) << 16) |
               (static_cast<u32>(pif.read_ram(off + 2)) << 8)  |
                static_cast<u32>(pif.read_ram(off + 3));
    }
    return 0;
}

u64 Bus::read64(u32 paddr) {
    return (static_cast<u64>(read32(paddr)) << 32) | static_cast<u64>(read32(paddr + 4));
}

void Bus::write8(u32 paddr, u8 val) {
    if (paddr < ram_limit_) {
        rdram[paddr] = val;
        jit::notify_code_write(paddr, 1); // this CPU store may be self-modifying code
        return;
    } else if (paddr >= 0x04000000 && paddr < 0x04001000) {
        rsp.write_dmem(paddr & 0xFFF, val);
    } else if (paddr >= 0x04001000 && paddr < 0x04002000) {
        rsp.write_imem(paddr & 0xFFF, val);
    } else if (paddr >= 0x1FC007C0 && paddr < 0x1FC00800) {
        pif.write_ram(paddr - 0x1FC007C0, val);
    }
}

void Bus::write16(u32 paddr, u16 val) {
    if (paddr + 1 < ram_limit_) {
        rdram[paddr + 0] = (val >> 8) & 0xFF;
        rdram[paddr + 1] = val & 0xFF;
        jit::notify_code_write(paddr, 2);
    } else {
        write8(paddr + 0, (val >> 8) & 0xFF);
        write8(paddr + 1, val & 0xFF);
    }
}

void Bus::write32(u32 paddr, u32 val) {
    if (paddr + 3 < ram_limit_) {
        rdram[paddr + 0] = (val >> 24) & 0xFF;
        rdram[paddr + 1] = (val >> 16) & 0xFF;
        rdram[paddr + 2] = (val >> 8)  & 0xFF;
        rdram[paddr + 3] = val & 0xFF;
        jit::notify_code_write(paddr, 4);
    } else if (paddr >= 0x04000000 && paddr < 0x04001000) {
        u32 off = paddr & 0xFFC;
        rsp.write_dmem(off + 0, (val >> 24) & 0xFF);
        rsp.write_dmem(off + 1, (val >> 16) & 0xFF);
        rsp.write_dmem(off + 2, (val >> 8)  & 0xFF);
        rsp.write_dmem(off + 3, val & 0xFF);
    } else if (paddr >= 0x04001000 && paddr < 0x04002000) {
        u32 off = paddr & 0xFFC;
        rsp.write_imem(off + 0, (val >> 24) & 0xFF);
        rsp.write_imem(off + 1, (val >> 16) & 0xFF);
        rsp.write_imem(off + 2, (val >> 8)  & 0xFF);
        rsp.write_imem(off + 3, val & 0xFF);
    } else if (paddr >= 0x04040000 && paddr < 0x040FFFFF) {
        // An SP DMA into RDRAM reports the exact range it wrote to the JIT
        // itself (RSP::execute_sp_dma_write) - no blanket invalidation here.
        rsp.write_reg(paddr, val, mi, rdp, rdram.data(), rdram.size());
    } else if (paddr >= 0x04100000 && paddr < 0x041FFFFF) {
        rdp.write_dpc_reg(paddr, val, mi, rdram.data(), rdram.size());
    } else if (paddr >= 0x04200000 && paddr < 0x042FFFFF) {
        rdp.write_dps_reg(paddr, val);
    } else if (paddr >= 0x04300000 && paddr < 0x043FFFFF) {
        mi.write_reg(paddr, val);
    } else if (paddr >= 0x04400000 && paddr < 0x044FFFFF) {
        vi.write_reg(paddr, val, mi);
    } else if (paddr >= 0x04500000 && paddr < 0x045FFFFF) {
        ai.write_reg(paddr, val, mi, rdram.data(), rdram.size());
    } else if (paddr >= 0x04600000 && paddr < 0x046FFFFF) {
        // Likewise for cartridge DMAs into RDRAM (PI::execute_dma_read).
        pi.write_reg(paddr, val, mi, cart, rdram.data(), rdram.size());
    } else if (paddr >= 0x04700000 && paddr < 0x047FFFFF) {
        u32 reg = (paddr & 0x1F) >> 2;
        switch (reg) {
            case 0: ri_mode = val; break;
            case 1: ri_config = val; break;
            case 2: ri_current_load = val; break;
            case 3: ri_select = val; break;
            case 4: ri_refresh = val; break;
            case 5: ri_latency = val; break;
            case 6: ri_error = val; break;
            case 7: ri_werror = val; break;
        }
    } else if (paddr >= 0x04800000 && paddr < 0x048FFFFF) {
        // Likewise for PIF RAM -> RDRAM DMAs (SI::write_reg).
        si.write_reg(paddr, val, mi, pif, controllers, cart, rdram.data(), rdram.size());
    } else if (paddr >= 0x08000000 && paddr < 0x10000000) {
        cart.write_bus32(paddr - 0x08000000, val); // SRAM / FlashRAM commands
    } else if (paddr >= 0x1FC007C0 && paddr < 0x1FC00800) {
        u32 off = paddr - 0x1FC007C0;
        pif.write_ram(off + 0, (val >> 24) & 0xFF);
        pif.write_ram(off + 1, (val >> 16) & 0xFF);
        pif.write_ram(off + 2, (val >> 8)  & 0xFF);
        pif.write_ram(off + 3, val & 0xFF);
    }
}

void Bus::write64(u32 paddr, u64 val) {
    write32(paddr, static_cast<u32>(val >> 32));
    write32(paddr + 4, static_cast<u32>(val & 0xFFFFFFFF));
}

u8 Bus::read_v8(u64 vaddr, TLBResult& result, u8 current_asid) {
    u32 paddr = 0;
    result = translate_vaddr(vaddr, paddr, false, current_asid);
    if (result != TLBResult::SUCCESS) {
        return 0;
    }
    return read8(paddr);
}

u16 Bus::read_v16(u64 vaddr, TLBResult& result, u8 current_asid) {
    u32 paddr = 0;
    result = translate_vaddr(vaddr, paddr, false, current_asid);
    if (result != TLBResult::SUCCESS) {
        return 0;
    }
    return read16(paddr);
}

u32 Bus::read_v32(u64 vaddr, TLBResult& result, u8 current_asid) {
    u32 paddr = 0;
    result = translate_vaddr(vaddr, paddr, false, current_asid);
    if (result != TLBResult::SUCCESS) {
        return 0;
    }
    return read32(paddr);
}

u64 Bus::read_v64(u64 vaddr, TLBResult& result, u8 current_asid) {
    u32 paddr = 0;
    result = translate_vaddr(vaddr, paddr, false, current_asid);
    if (result != TLBResult::SUCCESS) {
        return 0;
    }
    return read64(paddr);
}

void Bus::write_v8(u64 vaddr, u8 val, TLBResult& result, u8 current_asid) {
    u32 paddr = 0;
    result = translate_vaddr(vaddr, paddr, true, current_asid);
    if (result != TLBResult::SUCCESS) {
        return;
    }
    write8(paddr, val);
}

void Bus::write_v16(u64 vaddr, u16 val, TLBResult& result, u8 current_asid) {
    u32 paddr = 0;
    result = translate_vaddr(vaddr, paddr, true, current_asid);
    if (result != TLBResult::SUCCESS) {
        return;
    }
    write16(paddr, val);
}

void Bus::write_v32(u64 vaddr, u32 val, TLBResult& result, u8 current_asid) {
    u32 paddr = 0;
    result = translate_vaddr(vaddr, paddr, true, current_asid);
    if (result != TLBResult::SUCCESS) {
        return;
    }
    write32(paddr, val);
}

void Bus::write_v64(u64 vaddr, u64 val, TLBResult& result, u8 current_asid) {
    u32 paddr = 0;
    result = translate_vaddr(vaddr, paddr, true, current_asid);
    if (result != TLBResult::SUCCESS) {
        return;
    }
    write64(paddr, val);
}
