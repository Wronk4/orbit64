#pragma once
// ABI shared between the C++ driver and JIT-compiled native code.
//
// A compiled block is a plain C function pointer:
//     u32 block_fn(JitCtx* ctx);
// returning how many guest instructions it actually executed (for COUNT and
// the scanline budget). Where execution continues is reported separately:
//   - normal exit: ctx->next_pc. The driver presets it to the fall-through
//     pc (start_pc + 4 * length); a block that ends in a branch/jump
//     overwrites it with wherever that branch went. (x64 blocks also store
//     the fall-through pc themselves: a chained block isn't entered by the
//     driver.)
//   - an instruction raised an exception (TLB fault, integer overflow):
//     the jit_helpers thunk that raised it set ctx->faulted, and cpu->pc is
//     already the exception vector - the driver must not touch pc itself.
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
    u64 start_pc;     // guest pc the block was entered at (either KSEG0/KSEG1 alias, sign- or zero-extended)
    u64 next_pc;      // see above
    u32 branch_taken; // written by a block's terminating branch before its delay slot runs
    u32 faulted;      // set by jit_helpers when an instruction raised an exception
    u8* rdram;        // RDRAM_SIZE bytes, for inline KSEG0/KSEG1 loads/stores
    const u8* code_pages; // Recompiler::code_pages_: nonzero = a compiled block was read from this 64-byte page
    // Block chaining (see Recompiler::run): blocks
    // jump straight to each other while the cycle budget lasts.
    s64 cycles;       // in: cycles the chain may run; out: what is left (<= 0 once used up)
    const void* jcache; // Recompiler::jcache_, the dispatcher's pc -> block lookup table
    u32 exit_req;     // set by C++ (an MMIO store, a write to compiled code) to end the chain early
    u32 fr_mismatch;  // a block compiled for the other COP1 register mode (STATUS.FR) was entered
    s64 synced;       // cycles of this chain already applied to COUNT/RANDOM (by jit_cop0)
};

// Identifies the guest instruction a jit_helpers call is made for, so an
// exception it raises gets the right EPC/BD: the instruction's index in the
// block (pc = start_pc + 4 * index) plus kSiteDelaySlot if it is the delay
// slot of the block's terminating branch (BD is then set iff that branch was
// taken - a not-taken branch's delay slot runs as an ordinary instruction,
// exactly as in the interpreter).
constexpr u32 kSiteDelaySlot = 0x80000000u;

using JitBlockFn = u32 (*)(JitCtx* ctx);

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__)
// Compile-time sanity: the block prologue loads these via fixed byte offsets.
static_assert(offsetof(JitCtx, cpu) == 0, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, bus) == 8, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, gpr) == 16, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, hi) == 24, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, lo) == 32, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, start_pc) == 40, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, next_pc) == 48, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, branch_taken) == 56, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, rdram) == 64, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, code_pages) == 72, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, cycles) == 80, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, jcache) == 88, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, exit_req) == 96, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, fr_mismatch) == 100, "JitCtx layout is baked into codegen");
static_assert(offsetof(JitCtx, synced) == 104, "JitCtx layout is baked into codegen");
#endif
