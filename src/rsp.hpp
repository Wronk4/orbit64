#pragma once

#include "common.hpp"
#include "ahle.hpp"
#include "rsp_core.hpp"

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

// Which tasks run on the low-level RSP (rsp_core.hpp, a real RSP running the
// game's own microcode, with the RDP drawing the commands it sends) instead
// of the high-level emulation. Tasks the high-level emulation doesn't know -
// including graphics microcodes it can't identify - always run low-level.
enum class RspMode : u8 {
    HLE,         // graphics and audio emulated at a high level (fastest)
    LLEGraphics, // graphics low-level, audio high-level
    LLE,         // everything low-level (most accurate, slowest)
    LLEAudio,    // audio low-level only (debugging)
};

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

    // The low-level RSP (rsp_core.hpp) reaches the SP and DP registers
    // through its COP0 registers 0-7 and 8-15.
    u32 cop0_read(u32 reg, RDP& rdp);
    void cop0_write(u32 reg, u32 val, MI& mi, RDP& rdp, u8* rdram, size_t rdram_size);
    void core_break(MI& mi); // BREAK: halt, and interrupt the CPU if asked to
    // Which tasks run on the low-level RSP (RspMode); ORBIT64_RSP=hle, lle,
    // lle-gfx or lle-audio overrides it. Takes effect with the next task.
    void set_mode(RspMode m) { mode_ = m; }
    RspMode mode() const { return env_mode_ >= 0 ? static_cast<RspMode>(env_mode_) : mode_; }
    void set_force_lle(bool on) { mode_ = on ? RspMode::LLE : RspMode::HLE; env_mode_ = -1; }
    // Whether the last graphics task ran low-level.
    bool gfx_lle() const { return gfx_lle_; }
    bool lle_active() const { return lle_running; }
    u64 get_lle_task_count() const { return lle_task_count; }
    const RspCore& debug_core() const { return core; }
    void check_and_run_task(MI& mi, RDP& rdp, u8* rdram, size_t rdram_size);

    // Advance emulated RSP task latency; completes a pending task once its
    // simulated run time has elapsed (see task_pending in write_reg).
    void step(u32 cycles, MI& mi, RDP& rdp, u8* rdram, size_t rdram_size);

    // Frontend status queries (read-only).
    u64 get_gfx_task_count() const { return gfx_task_count; }
    u64 get_audio_task_count() const { return audio_task_count; }
    const AudioHLE& get_audio_hle() const { return ahle; }

    // Save states (savestate.hpp).
    template <class S> void serialize(S& s) {
        s(dmem, imem, mem_addr, dram_addr, rd_len, wr_len, status, semaphore, task_pending, task_delay_cycles, ahle,
          gfx_task_count, audio_task_count, core, lle_running, lle_task_count, lle_cycle_debt);
    }

private:
    std::array<u8, DMEM_SIZE> dmem{};
    std::array<u8, IMEM_SIZE> imem{};

    u32 mem_addr{0};
    u32 dram_addr{0};
    u32 rd_len{0};
    u32 wr_len{0};
    u32 status{SPStatus::HALT};
    u32 semaphore{0};

    RspCore core;
    bool lle_running{false}; // the halted RSP was started on a task run by `core`
    RspMode mode_{RspMode::HLE};
    int env_mode_{-1}; // ORBIT64_RSP's RspMode, -1 = not set
    bool gfx_lle_{false};
    u64 lle_task_count{0};
    s64 lle_cycle_debt{0};   // RSP cycles owed from the last step (2 RSP : 3 CPU)

    // A task's completion (interrupt + status update) is deferred by this many
    // cycles instead of firing synchronously inside the SP_STATUS write that
    // starts it. Some games' message-queue schedulers rely on the CPU having
    // a chance to reach its wait state before the "done" signal arrives; an
    // instant same-instruction completion can deliver it too early.
    bool task_pending{false};
    s64 task_delay_cycles{0};

    void execute_sp_dma(bool to_rdram, u32 len_reg, u8* rdram, size_t rdram_size);
    void write_status(u32 val, MI& mi);
    void start(MI& mi, RDP& rdp, u8* rdram, size_t rdram_size);
    // Identifies the graphics microcode of the task at DMEM `offset` and
    // tells the RDP; false when it is none the high-level emulation knows.
    bool detect_gfx_ucode(u32 offset, RDP& rdp, const u8* rdram, size_t rdram_size);

    AudioHLE ahle;
    u64 gfx_task_count{0};
    u64 audio_task_count{0};
};
