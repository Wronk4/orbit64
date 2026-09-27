#include "rsp.hpp"
#include "mi.hpp"
#include "rdp.hpp"
#include "jit/jit_invalidate.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

RSP::RSP() {
    reset();
}

namespace {
// Some of Rare's graphics microcodes carry no credit string to recognize them
// by, so they are recognized by a CRC-32 of the start of their code; Auto
// (no match) leaves the banner search to decide.
MicrocodeType ucode_by_crc(u32 ucode_ptr, const u8* rdram, size_t rdram_size) {
    constexpr u32 kLen = 0x800;
    const u32 phys = ucode_ptr & static_cast<u32>(rdram_size - 1);
    if (phys + kLen > rdram_size) return MicrocodeType::Auto;
    u32 c = 0xFFFFFFFFu;
    for (u32 i = 0; i < kLen; ++i) {
        c ^= rdram[phys + i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    switch (~c) {
        case 0xF295D221u: return MicrocodeType::F3DPD;  // Perfect Dark
        case 0xE434110Du: return MicrocodeType::F3DDKR; // Diddy Kong Racing
        case 0x248DCED9u: return MicrocodeType::F3DJFG; // Jet Force Gemini
        default: return MicrocodeType::Auto;
    }
}
} // namespace

void RSP::reset() {
    std::fill(dmem.begin(), dmem.end(), 0);
    std::fill(imem.begin(), imem.end(), 0);
    mem_addr = 0;
    dram_addr = 0;
    rd_len = 0;
    wr_len = 0;
    status = SPStatus::HALT;
    semaphore = 0;
    task_pending = false;
    task_delay_cycles = 0;
    core.reset();
    lle_running = false;
    lle_cycle_debt = 0;
    ahle.reset();
    if (const char* e = std::getenv("ORBIT64_RSP")) {
        force_lle_ = std::strcmp(e, "lle") == 0;
        force_lle_audio_ = std::strcmp(e, "lle-audio") == 0;
    }
}

u32 RSP::read_reg(u32 addr) const {
    if (addr >= 0x04080000 && addr <= 0x04080004) {
        return core.pc;
    }
    u32 reg = (addr & 0x1F) >> 2;
    switch (reg) {
        case 0: return mem_addr;
        case 1: return dram_addr;
        case 2: return rd_len;
        case 3: return wr_len;
        case 4: return status;
        case 5: return 0; // DMA_FULL
        case 6: return 0; // DMA_BUSY
        case 7: {
            u32 res = semaphore;
            const_cast<RSP*>(this)->semaphore = 1; // Read sets to 1
            return res;
        }
        default: return 0;
    }
}

void RSP::write_status(u32 val, MI& mi) {
    if (val & (1 << 0)) status &= ~SPStatus::HALT;        // Clr Halt
    if (val & (1 << 1)) status |= SPStatus::HALT;         // Set Halt
    if (val & (1 << 2)) status &= ~SPStatus::BROKE;       // Clr Broke
    if (val & (1 << 3)) mi.clear_interrupt(MIInterrupt::SP); // Clr Intr
    if (val & (1 << 4)) mi.raise_interrupt(MIInterrupt::SP); // Set Intr
    if (val & (1 << 5)) status &= ~SPStatus::SINGLE_STEP;
    if (val & (1 << 6)) status |= SPStatus::SINGLE_STEP;
    if (val & (1 << 7)) status &= ~SPStatus::INTR_ON_BREAK;
    if (val & (1 << 8)) status |= SPStatus::INTR_ON_BREAK;
    for (int s = 0; s < 8; ++s) {
        if (val & (1 << (9 + s * 2)))     status &= ~(1 << (7 + s)); // Clr Sig
        if (val & (1 << (10 + s * 2)))    status |= (1 << (7 + s));  // Set Sig
    }
}

void RSP::write_reg(u32 addr, u32 val, MI& mi, RDP& rdp, u8* rdram, size_t rdram_size) {
    if (addr >= 0x04080000 && addr <= 0x04080004) {
        core.pc = val & 0xFFC;
        core.npc = (core.pc + 4) & 0xFFC;
        return;
    }

    u32 reg = (addr & 0x1F) >> 2;
    switch (reg) {
        case 0: // SP_MEM_ADDR_REG
            mem_addr = val & 0x1FF8;
            break;
        case 1: // SP_DRAM_ADDR_REG
            dram_addr = val & 0x00FFFFF8;
            break;
        case 2: // SP_RD_LEN_REG (RDRAM to SP)
            rd_len = val;
            execute_sp_dma(false, val, rdram, rdram_size);
            break;
        case 3: // SP_WR_LEN_REG (SP to RDRAM)
            wr_len = val;
            execute_sp_dma(true, val, rdram, rdram_size);
            break;
        case 4: { // SP_STATUS_REG
            const bool was_halted = status & SPStatus::HALT;
            write_status(val, mi);
            if (was_halted && !(status & SPStatus::HALT) && !task_pending && !lle_running)
                start(mi, rdp, rdram, rdram_size);
            break;
        }
        case 7: // SP_SEMAPHORE_REG
            semaphore = 0;
            break;
    }
}

// The CPU has just let the RSP run. libultra's tasks (an OSTask at DMEM
// 0xFC0) of the graphics and audio microcodes are emulated at a high level;
// anything else runs on the low-level RSP.
void RSP::start(MI& mi, RDP& rdp, u8* rdram, size_t rdram_size) {
    (void)mi; (void)rdp; (void)rdram; (void)rdram_size;
    const u32 task_type = (static_cast<u32>(dmem[0xFC0]) << 24) | (static_cast<u32>(dmem[0xFC1]) << 16) |
                          (static_cast<u32>(dmem[0xFC2]) << 8) | static_cast<u32>(dmem[0xFC3]);
    if (force_lle_ || (force_lle_audio_ && task_type == 2) || (task_type != 1 && task_type != 2)) {
        lle_running = true;
        lle_cycle_debt = 0;
        ++lle_task_count;
        if (const char* d = std::getenv("ORBIT64_RSP_DUMP")) { char n[512]; std::snprintf(n, sizeof n, "%s/t%05llu_pc%03x.bin", d, (unsigned long long)lle_task_count, core.pc); if (FILE* f = std::fopen(n, "wb")) { std::fwrite(imem.data(), 1, 4096, f); std::fwrite(dmem.data(), 1, 4096, f); std::fclose(f); } }
        return;
    }
    // Simulate RSP run time instead of completing the task synchronously
    // inside this register write. Some games' message-queue schedulers
    // expect the CPU to reach its wait state before the "task done" signal
    // arrives.
    task_pending = true;
    if (task_type == 2) {
        // Audio tasks now do real decode/mix work (ahle.process),
        // proportional to the command list size, not a flat
        // worst-case constant: a fixed multi-thousand-cycle
        // penalty per task starves audio throughput when a game
        // submits several small audio tasks per frame (measured
        // effective output well below the target sample rate).
        // Scale with data_size instead, with a small floor so a
        // tiny task still yields the CPU once.
        u32 data_size = (static_cast<u32>(dmem[0xFC0 + 0x34]) << 24) |
                        (static_cast<u32>(dmem[0xFC0 + 0x35]) << 16) |
                        (static_cast<u32>(dmem[0xFC0 + 0x36]) << 8)  |
                         static_cast<u32>(dmem[0xFC0 + 0x37]);
        task_delay_cycles = std::max<u32>(500, data_size * 4);
    } else {
        task_delay_cycles = 2976;
    }
}

u32 RSP::cop0_read(u32 reg, RDP& rdp) {
    if (reg < 8) return read_reg(0x04040000 + reg * 4);
    return rdp.read_dpc_reg(0x04100000 + (reg - 8) * 4);
}

void RSP::cop0_write(u32 reg, u32 val, MI& mi, RDP& rdp, u8* rdram, size_t rdram_size) {
    if (reg < 8) {
        if (reg == 4) write_status(val, mi); // no task start: the RSP is running
        else write_reg(0x04040000 + reg * 4, val, mi, rdp, rdram, rdram_size);
    } else {
        rdp.write_dpc_reg(0x04100000 + (reg - 8) * 4, val, mi, rdram, rdram_size);
    }
}

void RSP::core_break(MI& mi) {
    status |= SPStatus::HALT | SPStatus::BROKE;
    if (status & SPStatus::INTR_ON_BREAK) mi.raise_interrupt(MIInterrupt::SP);
}

u8 RSP::read_dmem(u32 addr) const {
    return dmem[addr & (DMEM_SIZE - 1)];
}

void RSP::write_dmem(u32 addr, u8 val) {
    dmem[addr & (DMEM_SIZE - 1)] = val;
}

u8 RSP::read_imem(u32 addr) const {
    return imem[addr & (IMEM_SIZE - 1)];
}

void RSP::write_imem(u32 addr, u8 val) {
    imem[addr & (IMEM_SIZE - 1)] = val;
}

// SP DMA: count + 1 rows of length + 1 bytes (a multiple of 8), with `skip`
// bytes between rows in RDRAM. Both addresses advance, and the length
// register reads back as 0xFF8 afterwards, as on hardware.
void RSP::execute_sp_dma(bool to_rdram, u32 len_reg, u8* rdram, size_t rdram_size) {
    const u32 length = (len_reg & 0xFFF) | 7;
    const u32 count = (len_reg >> 12) & 0xFF;
    const u32 skip = (len_reg >> 20) & 0xFFF;
    u8* mem = (mem_addr & 0x1000) ? imem.data() : dmem.data();
    u32 maddr = mem_addr & 0xFF8;
    u32 daddr = dram_addr & 0xFFFFF8;
    const u32 mask = static_cast<u32>(rdram_size - 1);
    for (u32 row = 0; row <= count; ++row) {
        for (u32 i = 0; i <= length; ++i) {
            const u32 m = (maddr + i) & 0xFFF, d = (daddr + i) & mask;
            if (to_rdram) rdram[d] = mem[m];
            else mem[m] = rdram[d];
        }
        if (to_rdram) {
            // Drop JIT blocks compiled from what was just overwritten
            // (wrapping at the end of RDRAM like the copy above).
            const u32 start = daddr & mask;
            const u32 first = std::min<u32>(length + 1, static_cast<u32>(rdram_size) - start);
            jit::notify_code_write(start, first);
            if (first < length + 1) jit::notify_code_write(0, length + 1 - first);
        }
        maddr += length + 1;
        daddr += length + 1 + skip;
    }
    mem_addr = (mem_addr & 0x1000) | (maddr & 0xFF8);
    dram_addr = daddr & 0xFFFFF8;
    (to_rdram ? wr_len : rd_len) = (len_reg & 0xFFF00000u) | 0xFF8;
}

void RSP::step(u32 cycles, MI& mi, RDP& rdp, u8* rdram, size_t rdram_size) {
    if (lle_running) {
        // The RSP runs at 2/3 of the CPU clock.
        lle_cycle_debt += static_cast<s64>(cycles) * 2 / 3;
        if (lle_cycle_debt > 0 && !(status & SPStatus::HALT))
            lle_cycle_debt -= core.run(static_cast<u32>(lle_cycle_debt), *this, mi, rdp, rdram, rdram_size);
        if (status & SPStatus::HALT) {
            lle_running = false;
            lle_cycle_debt = 0;
        }
        return;
    }
    if (!task_pending) return;
    task_delay_cycles -= cycles;
    if (task_delay_cycles <= 0) {
        task_pending = false;
        check_and_run_task(mi, rdp, rdram, rdram_size);
    }
}

void RSP::check_and_run_task(MI& mi, RDP& rdp, u8* rdram, size_t rdram_size) {
    // Look for OSTask at 0xFC0 or 0x000 in DMEM
    u32 task_offsets[] = { 0xFC0, 0x000 };
    for (u32 offset : task_offsets) {
        u32 task_type = (static_cast<u32>(dmem[offset + 0]) << 24) |
                        (static_cast<u32>(dmem[offset + 1]) << 16) |
                        (static_cast<u32>(dmem[offset + 2]) << 8)  |
                         static_cast<u32>(dmem[offset + 3]);

        if (task_type == 1) { // M_GFXTASK
            u32 ucode_ptr = (static_cast<u32>(dmem[offset + 0x10]) << 24) |
                            (static_cast<u32>(dmem[offset + 0x11]) << 16) |
                            (static_cast<u32>(dmem[offset + 0x12]) << 8)  |
                             static_cast<u32>(dmem[offset + 0x13]);
            u32 ucode_data_ptr = (static_cast<u32>(dmem[offset + 0x18]) << 24) |
                                 (static_cast<u32>(dmem[offset + 0x19]) << 16) |
                                 (static_cast<u32>(dmem[offset + 0x1A]) << 8)  |
                                  static_cast<u32>(dmem[offset + 0x1B]);

            auto detect_banner = [&](u32 ptr) -> bool {
                u32 phys = ptr & (rdram_size - 1);
                size_t check_len = std::min<size_t>(2048, rdram_size > phys ? rdram_size - phys : 0);
                if (check_len < 16) return false;
                std::string header(reinterpret_cast<const char*>(&rdram[phys]), check_len);
                {
                    static int count = 0;
                    if (count < 4) {
                        count++;
                        std::string clean;
                        for (char c : header) clean += (c >= 32 && c < 127) ? c : '.';
                        std::cout << "[UCODE-BANNER-RAW] ptr=0x" << std::hex << ptr << std::dec << " text=\"" << clean.substr(0, 200) << "\"\n";
                    }
                }
                if (const MicrocodeType t = identify_ucode_banner(&rdram[phys], check_len); t != MicrocodeType::Auto) {
                    rdp.set_ucode_type(t);
                    rdp.set_cbfd(ucode_banner_is_cbfd(&rdram[phys], check_len));
                    rdp.set_no_near_clip(ucode_banner_is_non(&rdram[phys], check_len));
                    return true;
                }
                // Require the longer "ucode S2DEX" anchor (part of the real credit
                // string "RSP Gfx ucode S2DEX ...") rather than a bare "S2DEX"/
                // "S2DEX2" substring: this region isn't always a text banner (it can
                // be an unrelated task's data), and a short ASCII pattern like
                // "S2DEX2" can appear by pure chance in binary/opcode bytes, which
                // was mis-detecting plain F3DEX2 games (e.g. Dr. Mario 64) as S2DEX2
                // and permanently wrecking their rendering.
                size_t s2dex_anchor = header.find("ucode S2DEX");
                if (s2dex_anchor != std::string::npos) {
                    // Determine GBI-1 vs GBI-2 from text local to *this* banner match
                    // only (e.g. "...S2DEX       fifo 2.04..." vs "...S2DEX  1.06...").
                    // Searching the whole 2048-byte window for "fifo 2" independently
                    // can pick up an unrelated, adjacent ucode's banner (ROMs often
                    // bundle several microcodes together) and misclassify a real
                    // GBI-1 S2DEX banner as S2DEX2.
                    std::string local = header.substr(s2dex_anchor, 40);
                    bool is_gbi2 = local.find("S2DEX2") != std::string::npos || local.find("fifo 2") != std::string::npos;
                    rdp.set_ucode_type(is_gbi2 ? MicrocodeType::S2DEX2 : MicrocodeType::S2DEX);
                    return true;
                } else if (header.find("F3DEX 2") != std::string::npos ||
                    header.find("F3DEX2") != std::string::npos ||
                    header.find("fifo 2") != std::string::npos ||
                    header.find("F3DZEX") != std::string::npos) {
                    rdp.set_ucode_type(MicrocodeType::F3DEX2);
                    return true;
                } else if (header.find("F3DEX") != std::string::npos ||
                           header.find("F3DLX") != std::string::npos) {
                    rdp.set_ucode_type(MicrocodeType::F3DEX);
                    return true;
                } else if (header.find("2.0G") != std::string::npos) {
                    rdp.set_ucode_type(MicrocodeType::F3DGOLDEN);
                    return true;
                } else if (header.find("Fast3D") != std::string::npos ||
                           header.find("RSP SW Version") != std::string::npos ||
                           header.find("SGI U64 GFX") != std::string::npos) {
                    rdp.set_ucode_type(MicrocodeType::Fast3D);
                    return true;
                }
                return false;
            };

            if (!detect_banner(ucode_data_ptr)) {
                if (const MicrocodeType t = ucode_by_crc(ucode_ptr, rdram, rdram_size); t != MicrocodeType::Auto) {
                    rdp.set_ucode_type(t);
                } else if (!detect_banner(ucode_ptr)) {
                    static int dbg_fail = 0;
                    if (dbg_fail < 5) {
                        dbg_fail++;
                        u32 phys = ucode_data_ptr & (rdram_size - 1);
                        std::string header(reinterpret_cast<const char*>(&rdram[phys]), std::min<size_t>(256, rdram_size - phys));
                        std::cout << "[UCODE-DETECT-FAIL] data_ptr=0x" << std::hex << ucode_data_ptr
                                  << " code_ptr=0x" << ucode_ptr << std::dec << " header=\"" << header << "\"\n";
                    }
                }
            }
            {
                static MicrocodeType last_ucode = MicrocodeType::Auto;
                if (rdp.get_ucode_type() != last_ucode) {
                    last_ucode = rdp.get_ucode_type();
                    std::cout << "[UCODE-CHANGE] now=" << static_cast<int>(last_ucode) << "\n";
                }
            }

            u32 data_ptr = (static_cast<u32>(dmem[offset + 0x30]) << 24) |
                           (static_cast<u32>(dmem[offset + 0x31]) << 16) |
                           (static_cast<u32>(dmem[offset + 0x32]) << 8)  |
                            static_cast<u32>(dmem[offset + 0x33]);

            gfx_task_count++;
            if (data_ptr != 0) {
                rdp.process_display_list(data_ptr, rdram, rdram_size, mi);
            }
            break;
        } else if (task_type == 2) { // M_AUDTASK
            u32 ucode_data_ptr = (static_cast<u32>(dmem[offset + 0x18]) << 24) |
                                 (static_cast<u32>(dmem[offset + 0x19]) << 16) |
                                 (static_cast<u32>(dmem[offset + 0x1A]) << 8)  |
                                  static_cast<u32>(dmem[offset + 0x1B]);
            u32 data_ptr = (static_cast<u32>(dmem[offset + 0x30]) << 24) |
                           (static_cast<u32>(dmem[offset + 0x31]) << 16) |
                           (static_cast<u32>(dmem[offset + 0x32]) << 8)  |
                            static_cast<u32>(dmem[offset + 0x33]);
            u32 data_size = (static_cast<u32>(dmem[offset + 0x34]) << 24) |
                            (static_cast<u32>(dmem[offset + 0x35]) << 16) |
                            (static_cast<u32>(dmem[offset + 0x36]) << 8)  |
                             static_cast<u32>(dmem[offset + 0x37]);
            audio_task_count++;
            if (data_ptr != 0 && data_size != 0) {
                ahle.process(rdram, rdram_size, data_ptr, data_size, ucode_data_ptr);
            }
            break;
        }
    }

    // Mark task complete: halt RSP, set broke and signals, trigger interrupt
    status |= SPStatus::HALT | SPStatus::BROKE | (1 << 9) /* SIG2 */;
    // Always fire SP interrupt; only fire DP interrupt for GFX tasks (handled inside G_ENDDL)
    mi.raise_interrupt(MIInterrupt::SP);
}
