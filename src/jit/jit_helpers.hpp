#pragma once
// C++ thunks called (by absolute address) from JIT-compiled code for every
// load/store. They do the exact same TLB translation + MMIO dispatch +
// exception raising as the interpreter (see CPU::execute case 0x20..0x3F),
// just parameterized instead of switched on. Keeping these in C++ instead of
// hand-rolling TLB walks in assembly is what keeps the recompiler's risk
// bounded to straight-line ALU codegen.
//
// Every helper returns 0 on success, 1 if it raised a TLB exception (the
// generated code must stop executing the block immediately when it sees 1 -
// cpu->pc has already been redirected to the exception vector).

#include "jit_abi.hpp"

extern "C" {
u32 jit_load_lb(JitCtx* ctx, u64 addr, u32 rt);
u32 jit_load_lbu(JitCtx* ctx, u64 addr, u32 rt);
u32 jit_load_lh(JitCtx* ctx, u64 addr, u32 rt);
u32 jit_load_lhu(JitCtx* ctx, u64 addr, u32 rt);
u32 jit_load_lw(JitCtx* ctx, u64 addr, u32 rt);
u32 jit_load_lwu(JitCtx* ctx, u64 addr, u32 rt);
u32 jit_load_ld(JitCtx* ctx, u64 addr, u32 rt);

u32 jit_store_sb(JitCtx* ctx, u64 addr, u32 val);
u32 jit_store_sh(JitCtx* ctx, u64 addr, u32 val);
u32 jit_store_sw(JitCtx* ctx, u64 addr, u32 val);
u32 jit_store_sd(JitCtx* ctx, u64 addr, u64 val);

// Raises EXC_OV on behalf of a trapping ADD/SUB/DADD/DSUB/ADDI/DADDI whose
// native overflow flag came back set. Always returns 1 (guest instructions
// executed by the aborted block, filled in by the caller via a fixed
// argument would be simpler, but the recompiler already bakes the "n"
// return value into the generated code the same way it does for load/store
// faults, so this helper only needs to perform the side effect).
void jit_overflow(JitCtx* ctx);

// FPU register access, transporting values as raw bit patterns (u32/u64)
// rather than float/double - the generated code moves those bits into/out
// of a host FP register itself (or just leaves them as integer bits for a
// pure copy/transfer instruction), so nothing here ever needs to cross the
// C++ ABI as an actual floating-point value.
u32 jit_fpr_get32(JitCtx* ctx, u32 reg);
void jit_fpr_set32(JitCtx* ctx, u32 reg, u32 bits);
u64 jit_fpr_get64(JitCtx* ctx, u32 reg);
void jit_fpr_set64(JitCtx* ctx, u32 reg, u64 bits);

// C.cond.S/C.cond.D: updates FCSR's condition bit; see CPU::jit_fpu_compare_s/d.
void jit_fpu_ccond_s(JitCtx* ctx, u32 fs, u32 ft, u32 cond);
void jit_fpu_ccond_d(JitCtx* ctx, u32 fs, u32 ft, u32 cond);
}
