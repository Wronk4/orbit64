#pragma once

#include "common.hpp"
#include "ahle.hpp"

class MI;
class RDP;

namespace SPStatus {
    constexpr u32 HALT           = 1 << 0;
    constexpr u32 BROKE          = 1 << 1;
    constexpr u32 DMA_BUSY       = 1 << 2;
    constexpr u32 DMA_FULL       = 1 << 3;
    constexpr u32 IO_FULL        = 1 << 4;
    constexpr u32 SINGLE_STEP    = 1 << 5;
    constexpr u32 INTR_ON_BREAK  = 1 << 6;
    constexpr u32 SIG0           = 1 << 7;
    constexpr u32 SIG1           = 1 << 8;
    constexpr u32 SIG2           = 1 << 9;
}

class RSP {
public:
    RSP();

    void reset();

    u32 read_reg(u32 addr) const;
    void write_reg(u32 addr, u32 val, MI& mi, RDP& rdp, u8* rdram, size_t rdram_size);

    u8 read_dmem(u32 addr) const;
    void write_dmem(u32 addr, u8 val);

    u8 read_imem(u32 addr) const;
    void write_imem(u32 addr, u8 val);

    u8* get_dmem() { return dmem.data(); }
    u8* get_imem() { return imem.data(); }
    const u8* get_dmem() const { return dmem.data(); }
    const u8* get_imem() const { return imem.data(); }

    u32 get_status() const { return status; }
    void check_and_run_task(MI& mi, RDP& rdp, u8* rdram, size_t rdram_size);

    // Advance emulated RSP task latency; completes a pending task once its
    // simulated run time has elapsed (see task_pending in write_reg).
    void step(u32 cycles, MI& mi, RDP& rdp, u8* rdram, size_t rdram_size);

    // Frontend status queries (read-only).
    u64 get_gfx_task_count() const { return gfx_task_count; }
    u64 get_audio_task_count() const { return audio_task_count; }
    const AudioHLE& get_audio_hle() const { return ahle; }

private:
    std::array<u8, DMEM_SIZE> dmem{};
    std::array<u8, IMEM_SIZE> imem{};

    u32 mem_addr{0};
    u32 dram_addr{0};
    u32 rd_len{0};
    u32 wr_len{0};
    u32 status{SPStatus::HALT};
    u32 semaphore{0};
    u32 pc{0};

    // A task's completion (interrupt + status update) is deferred by this many
    // cycles instead of firing synchronously inside the SP_STATUS write that
    // starts it. Some games' message-queue schedulers rely on the CPU having
    // a chance to reach its wait state before the "done" signal arrives; an
    // instant same-instruction completion can deliver it too early.
    bool task_pending{false};
    s64 task_delay_cycles{0};

    void execute_sp_dma_read(u8* rdram, size_t rdram_size);
    void execute_sp_dma_write(const u8* rdram, size_t rdram_size);

    AudioHLE ahle;
    u64 gfx_task_count{0};
    u64 audio_task_count{0};
};
