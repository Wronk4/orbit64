#include "rsp.hpp"
#include "mi.hpp"
#include "rdp.hpp"
#include "jit/jit_invalidate.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

RSP::RSP() {
    reset();
}

namespace {
// Perfect Dark's graphics microcode carries no credit string to recognize it
// by, so it is recognized by a CRC-32 of the start of its code.
bool is_perfect_dark_ucode(u32 ucode_ptr, const u8* rdram, size_t rdram_size) {
    constexpr u32 kLen = 0x800;
    const u32 phys = ucode_ptr & static_cast<u32>(rdram_size - 1);
    if (phys + kLen > rdram_size) return false;
    u32 c = 0xFFFFFFFFu;
    for (u32 i = 0; i < kLen; ++i) {
        c ^= rdram[phys + i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c == 0xF295D221u;
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
    pc = 0;
    task_pending = false;
    task_delay_cycles = 0;
    ahle.reset();
}

u32 RSP::read_reg(u32 addr) const {
    if (addr >= 0x04080000 && addr <= 0x04080004) {
        return pc;
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

void RSP::write_reg(u32 addr, u32 val, MI& mi, RDP& rdp, u8* rdram, size_t rdram_size) {
    (void)rdp;
    if (addr >= 0x04080000 && addr <= 0x04080004) {
        pc = val & 0xFFC;
        return;
    }

    u32 reg = (addr & 0x1F) >> 2;
    switch (reg) {
        case 0: // SP_MEM_ADDR_REG
            mem_addr = val & 0x1FFF;
            break;
        case 1: // SP_DRAM_ADDR_REG
            dram_addr = val & 0x00FFFFFF;
            break;
        case 2: // SP_RD_LEN_REG (RDRAM to SP)
            rd_len = val;
            execute_sp_dma_read(rdram, rdram_size);
            break;
        case 3: // SP_WR_LEN_REG (SP to RDRAM)
            wr_len = val;
            execute_sp_dma_write(rdram, rdram_size);
            break;
        case 4: { // SP_STATUS_REG
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

            if (!(status & SPStatus::HALT) && !task_pending) {
                // Simulate RSP run time instead of completing the task
                // synchronously inside this register write. Some games'
                // message-queue schedulers expect the CPU to reach its wait
                // state before the "task done" signal arrives.
                u32 task_type = (static_cast<u32>(dmem[0xFC0]) << 24) |
                                (static_cast<u32>(dmem[0xFC1]) << 16) |
                                (static_cast<u32>(dmem[0xFC2]) << 8)  |
                                 static_cast<u32>(dmem[0xFC3]);
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
            break;
        }
        case 7: // SP_SEMAPHORE_REG
            semaphore = 0;
            break;
    }
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

void RSP::execute_sp_dma_read(u8* rdram, size_t rdram_size) {
    u32 len = (rd_len & 0xFFF) + 1;
    len = (len + 7) & ~7; // 8-byte aligned

    u32 sp_target = mem_addr & 0x1FFF;
    u8* target_buf = (sp_target & 0x1000) ? imem.data() : dmem.data();
    u32 offset = sp_target & 0xFFF;

    for (u32 i = 0; i < len; ++i) {
        u32 dram_idx = (dram_addr + i) & (rdram_size - 1);
        u32 sp_idx = (offset + i) & 0xFFF;
        target_buf[sp_idx] = rdram[dram_idx];
    }
}

void RSP::execute_sp_dma_write(const u8* rdram, size_t rdram_size) {
    u32 len = (wr_len & 0xFFF) + 1;
    len = (len + 7) & ~7;

    u32 sp_target = mem_addr & 0x1FFF;
    const u8* src_buf = (sp_target & 0x1000) ? imem.data() : dmem.data();
    u32 offset = sp_target & 0xFFF;

    for (u32 i = 0; i < len; ++i) {
        u32 dram_idx = (dram_addr + i) & (rdram_size - 1);
        u32 sp_idx = (offset + i) & 0xFFF;
        const_cast<u8*>(rdram)[dram_idx] = src_buf[sp_idx];
    }

    // Report the written range (wrapping at the end of RDRAM like the copy
    // above) so JIT blocks compiled from it are dropped.
    u32 start = dram_addr & (rdram_size - 1);
    u32 first = static_cast<u32>(std::min<size_t>(len, rdram_size - start));
    jit::notify_code_write(start, first);
    if (first < len) jit::notify_code_write(0, len - first);
}

void RSP::step(u32 cycles, MI& mi, RDP& rdp, u8* rdram, size_t rdram_size) {
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
                if (is_perfect_dark_ucode(ucode_ptr, rdram, rdram_size)) {
                    rdp.set_ucode_type(MicrocodeType::F3DPD);
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
