#pragma once

#include "common.hpp"

class Bus;
enum class TLBResult;

namespace CP0Reg {
    constexpr int INDEX       = 0;
    constexpr int RANDOM      = 1;
    constexpr int ENTRY_LO0   = 2;
    constexpr int ENTRY_LO1   = 3;
    constexpr int CONTEXT     = 4;
    constexpr int PAGE_MASK   = 5;
    constexpr int WIRED       = 6;
    constexpr int BAD_VADDR   = 8;
    constexpr int COUNT       = 9;
    constexpr int ENTRY_HI    = 10;
    constexpr int COMPARE     = 11;
    constexpr int STATUS      = 12;
    constexpr int CAUSE       = 13;
    constexpr int EPC         = 14;
    constexpr int PRID        = 15;
    constexpr int CONFIG      = 16;
    constexpr int LLADDR      = 17;
    constexpr int WATCH_LO    = 18;
    constexpr int WATCH_HI    = 19;
    constexpr int X_CONTEXT   = 20;
    constexpr int TAG_LO      = 28;
    constexpr int TAG_HI      = 29;
    constexpr int ERROR_EPC   = 30;
}

class CPU {
public:
    explicit CPU(Bus& bus);

    void reset(u32 entry_point = 0x80000400);

    // Step a single instruction; returns number of CPU cycles consumed (typically 1 or 2)
    u32 step();

    // Check interrupts and timer
    void step_timer(u32 cycles);
    void check_interrupts();

    // Register access
    u64 get_pc() const { return pc; }
    void set_pc(u64 val) { pc = val; }

    u64 get_gpr(size_t reg) const { return gpr[reg & 31]; }
    void set_gpr(size_t reg, u64 val) { if (reg != 0) gpr[reg & 31] = val; }

    u64 get_cp0(size_t reg) const { return cp0[reg & 31]; }
    void set_cp0(size_t reg, u64 val) { cp0[reg & 31] = val; }

    u64 get_hi() const { return hi; }
    u64 get_lo() const { return lo; }

    // Debugger queries (read-only).
    u64 get_fpr_raw(size_t reg) const { return fpr[reg & 31]; }
    u32 get_fcsr() const { return fcsr; }
    bool is_in_delay_slot() const { return delay_slot_active; }

    // --- Recompiler support -------------------------------------------------
    // Raw pointers to the register file, handed to compiled native code so it
    // can load/store guest registers directly instead of going through
    // get_gpr()/set_gpr() per access. Compiled blocks themselves never touch
    // pc/cp0/delay-slot state directly (they only ever contain straight-line,
    // non-branching, non-trapping instructions) - but the recompiler's driver
    // does need to consult delay-slot state *before* deciding whether the
    // instruction at the current pc may even start a block; see
    // jit_pending_delay_slot().
    u64* jit_gpr_ptr() { return gpr.data(); }
    u64* jit_hi_ptr() { return &hi; }
    u64* jit_lo_ptr() { return &lo; }
    u8 jit_asid() const { return static_cast<u8>(cp0[CP0Reg::ENTRY_HI] & 0xFF); }

    // True when the instruction at the *current* pc is sitting in a branch
    // delay slot (i.e. the previous step/block ended in a taken branch and
    // this is the one instruction that always executes before control
    // transfers to the branch target). Such an instruction must never be the
    // first instruction of a JIT block: a compiled block always computes its
    // own next pc as start+4*n, which is wrong here - after this single
    // instruction, pc must become the pending branch's target instead. The
    // interpreter's step() already gets this right (it's exactly what
    // in_delay_slot/branch_target exist for), so the recompiler's driver
    // just needs to route this one instruction through cpu.step().
    bool jit_pending_delay_slot() const { return in_delay_slot; }
    // Valid only when jit_pending_delay_slot() is true: the pc that must
    // apply *after* the delay-slot instruction finishes (since in_delay_slot
    // is only ever true for a *taken* branch's delay slot - see the trigger
    // sites in execute() - this is unconditionally where control goes next,
    // never "maybe branch_target, maybe not").
    u64 jit_branch_target() const { return branch_target; }
    // Must be called after successfully running a delay-slot instruction
    // through jit_branch_target()'s dedicated path (never through
    // cpu.step()): clears the pending flag exactly like the top of
    // CPU::step() does for every instruction, so the *next* instruction
    // isn't mistaken for another delay slot.
    void jit_clear_pending_delay_slot() { in_delay_slot = false; }

    // Raises a TLB exception on behalf of a JIT-compiled load/store (mirrors
    // the interpreter's LW/SW/etc. handling). Always returns true so call
    // sites can write `return cpu->jit_raise_tlb_exception(...);`.
    bool jit_raise_tlb_exception(u64 fault_vaddr, TLBResult result, bool is_write) {
        trigger_tlb_exception(fault_vaddr, result, is_write);
        return true;
    }

    // Raises the integer-overflow exception (EXC_OV) on behalf of a
    // JIT-compiled trapping ADD/SUB/DADD/DSUB/ADDI/DADDI whose native
    // overflow flag came back set.
    void jit_raise_overflow_exception() { trigger_exception(12); }

    // FPU register access for JIT-compiled COP1 instructions. These forward
    // to the exact same get_fpr32/set_fpr32/get_fpr64/set_fpr64 the
    // interpreter uses (including their FR-mode-dependent odd/even pairing
    // when STATUS.FR=0), so compiled code never has to duplicate that
    // logic - it always goes through a call, in exchange for never being
    // able to disagree with the interpreter about it.
    u32 jit_get_fpr32(size_t reg) const { return get_fpr32(reg); }
    void jit_set_fpr32(size_t reg, u32 val) { set_fpr32(reg, val); }
    u64 jit_get_fpr64(size_t reg) const { return get_fpr64(reg); }
    void jit_set_fpr64(size_t reg, u64 val) { set_fpr64(reg, val); }

    // C.cond.S/C.cond.D: mirrors execute_fpu_op's C.cond case exactly
    // (including the NaN/"unordered" handling in fp_condition, which is why
    // this stays a call into the interpreter's own logic rather than being
    // reimplemented as native compare+branch) - reads fs/ft, writes the
    // FCSR condition bit (23).
    void jit_fpu_compare_s(size_t fs, size_t ft, u32 cond);
    void jit_fpu_compare_d(size_t fs, size_t ft, u32 cond);

private:
    Bus& bus;

    // Registers
    std::array<u64, 32> gpr{};
    u64 pc{0x80000400};
    u64 hi{0};
    u64 lo{0};

    // Delay slot state machine
    u64 cur_pc{0};
    bool in_delay_slot{false};
    bool delay_slot_active{false};
    u64 branch_target{0};
    bool pc_overridden{false};

    // CP0 (System Control Coprocessor)
    std::array<u64, 32> cp0{};

    // CP1 (Floating Point Coprocessor)
    std::array<u64, 32> fpr{};
    u32 fcsr{0};

    // Exception handling
    void trigger_exception(u32 exc_code, u64 vector = 0x80000180);
    void trigger_tlb_exception(u64 fault_vaddr, TLBResult result, bool is_write);
    void trigger_interrupt();

    // Instruction execution
    void execute(u32 instr, u64 cur_pc);

    // FPU operations
    void execute_fpu_op(u32 instr);
    u32 get_fpr32(size_t reg) const;
    void set_fpr32(size_t reg, u32 val);
    u64 get_fpr64(size_t reg) const;
    void set_fpr64(size_t reg, u64 val);

    // TLB instructions
    void execute_tlb(u32 funct);
};
