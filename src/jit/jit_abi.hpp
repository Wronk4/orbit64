#pragma once
// ABI shared between the C++ driver and JIT-compiled native code.
//
// A compiled block is a plain C function pointer:
//     u32 block_fn(JitCtx* ctx);
// returning how many guest instructions it actually executed. That is
// normally the full length the block was compiled for; it is shorter only
// when a load/store inside the block raised a TLB exception, in which case
// cpu->pc has already been set by CPU::jit_raise_tlb_exception() and the
// driver must not touch pc itself.
//
// Compiled code never touches CPU/Bus internals directly beyond what these
// pointers expose - all "interesting" work (TLB translation, MMIO, raising
// exceptions) still goes through the exact same C++ paths the interpreter
// uses, via the jit_helpers thunks. The recompiler only inlines guest
// register file (gpr/hi/lo) access and pure ALU/shift arithmetic.

#include "../common.hpp"

class CPU;
class Bus;

struct JitCtx {
    CPU* cpu;
    Bus* bus;
    u64* gpr; // 32 entries
    u64* hi;
    u64* lo;
};

using JitBlockFn = u32 (*)(JitCtx* ctx);

// Compile-time sanity: the block prologue loads these via fixed byte offsets.
static_assert(offsetof(JitCtx, cpu) == 0, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, bus) == 8, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, gpr) == 16, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, hi) == 24, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, lo) == 32, "JitCtx layout is baked into codegen");
