#pragma once
// C++ thunks called (by absolute address) from JIT-compiled code for every
// load/store. They do the exact same TLB translation + MMIO dispatch +
// exception raising as the interpreter (see CPU::execute case 0x20..0x3F),
// just parameterized instead of switched on. Keeping these in C++ instead of
// hand-rolling TLB walks in assembly is what keeps the recompiler's risk
// bounded to straight-line ALU codegen.
//
// Every load/store helper returns 0 on success, 1 if it raised a TLB
// exception (the generated code must stop executing the block immediately
// when it sees 1 - cpu->pc has already been redirected to the exception
// vector, and ctx->faulted is set).

#include "jit_abi.hpp"

extern "C" {
// `site` identifies the instruction for exception reporting - see
// kSiteDelaySlot in jit_abi.hpp. It is only looked at on the fault path.
u32 jit_load_lb(JitCtx* ctx, u64 addr, u32 rt, u32 site);
u32 jit_load_lbu(JitCtx* ctx, u64 addr, u32 rt, u32 site);
u32 jit_load_lh(JitCtx* ctx, u64 addr, u32 rt, u32 site);
u32 jit_load_lhu(JitCtx* ctx, u64 addr, u32 rt, u32 site);
u32 jit_load_lw(JitCtx* ctx, u64 addr, u32 rt, u32 site);
u32 jit_load_lwu(JitCtx* ctx, u64 addr, u32 rt, u32 site);
u32 jit_load_ld(JitCtx* ctx, u64 addr, u32 rt, u32 site);

u32 jit_store_sb(JitCtx* ctx, u64 addr, u32 val, u32 site);
u32 jit_store_sh(JitCtx* ctx, u64 addr, u32 val, u32 site);
u32 jit_store_sw(JitCtx* ctx, u64 addr, u32 val, u32 site);
u32 jit_store_sd(JitCtx* ctx, u64 addr, u64 val, u32 site);

// Raises EXC_OV on behalf of a trapping ADD/SUB/DADD/DSUB/ADDI/DADDI whose
// native overflow flag came back set. The generated code itself then returns
// the instruction count it baked in for that site.
void jit_overflow(JitCtx* ctx, u32 site);

// A store compiled inline (fastmem) hit a page compiled code was read from:
// forwards to jit::notify_code_write, exactly what Bus::write* would do.
void jit_notify_code_write(u32 paddr, u32 len);

// Runs `instr` - one of is_interp_callable()'s instructions - on the
// interpreter, in place, as part of a compiled block. Returns 0, or 1 if it
// raised an exception (cpu->pc is then the vector and ctx->faulted is set).
u32 jit_interp(JitCtx* ctx, u32 instr, u32 site);

// FPU register access, transporting values as raw bit patterns (u32/u64)
// rather than float/double - the generated code moves those bits into/out
// of a host FP register itself (or just leaves them as integer bits for a
// pure copy/transfer instruction), so nothing here ever needs to cross the
// C++ ABI as an actual floating-point value.
u32 jit_fpr_get32(JitCtx* ctx, u32 reg);
void jit_fpr_set32(JitCtx* ctx, u32 reg, u32 bits);
u64 jit_fpr_get64(JitCtx* ctx, u32 reg);
void jit_fpr_set64(JitCtx* ctx, u32 reg, u64 bits);

// COP0 moves and TLB instructions (MFC0/DMFC0/MTC0/DMTC0/TLBR/TLBWI/TLBWR/TLBP)
// in a chain of compiled blocks (AArch64). `remaining` is the chain's cycle
// budget left at this instruction (JitCtx::cycles - it = cycles executed
// before it). COUNT and RANDOM are first brought up to date, then the
// instruction runs exactly as CPU::step() would run it (RANDOM steps before,
// COUNT after). Returns 1 if the chain has to end right after it: an
// interrupt can now be taken, COUNT/COMPARE changed, STATUS.FR did (the rest
// of the block assumed the old COP1 register mode), or the TLB or ASID did.
u32 jit_cop0(JitCtx* ctx, u32 instr, s64 remaining);
// ERET, the same way; the pc it returns to is left in JitCtx::next_pc.
u32 jit_eret(JitCtx* ctx, s64 remaining);
// A block compiled for the other STATUS.FR setting was entered (see JitCtx::fr_mismatch).
void jit_fr_mismatch(JitCtx* ctx);

// C.cond.S/C.cond.D: updates FCSR's condition bit; see CPU::jit_fpu_compare_s/d.
void jit_fpu_ccond_s(JitCtx* ctx, u32 fs, u32 ft, u32 cond);
void jit_fpu_ccond_d(JitCtx* ctx, u32 fs, u32 ft, u32 cond);
}
