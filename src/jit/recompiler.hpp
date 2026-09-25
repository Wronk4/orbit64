#pragma once
// Dynamic recompiler for the CPU's hot path.
//
// Scope (see docs/JIT.md for the full write-up): only pure, non-trapping
// ALU/shift/immediate instructions and loads/stores are ever translated to
// native code. Branches, jumps, MULT/DIV, COP0/COP1/TLB, trap-on-overflow
// arithmetic and anything else stay on the existing, already-correct
// interpreter - a compiled block is simply a maximal straight-line run of
// whitelisted instructions starting at some PC, and the moment a
// non-whitelisted instruction is reached the block ends (without including
// it). This bounds the recompiler's correctness risk to a small, well-tested
// slice while still eliminating fetch/decode/switch-dispatch overhead for
// the instruction mix that dominates real game code.
//
// Blocks are only compiled for code living in KSEG0/KSEG1 (0x80000000-
// 0xBFFFFFFF, or its sign-extended 64-bit form), i.e. the fixed, TLB-free
// direct mapping to physical RDRAM that virtually all N64 game code runs
// from. Anything else always falls back to the interpreter.

#include "../common.hpp"
#include "code_buffer.hpp"
#include "jit_abi.hpp"
#include <vector>

class CPU;
class Bus;

class Recompiler {
public:
    // Exposed so the (free-function) per-architecture codegen backends in
    // recompiler.cpp can build and return one; not part of the public API
    // otherwise.
    struct Block {
        JitBlockFn fn = nullptr; // null = "known not JIT-able at this PC"
        u32 length = 0;          // guest instructions covered
    };

    Recompiler();
    ~Recompiler();

    // Runs one interpreter step's worth of work starting at cpu.get_pc():
    // either a cached/newly-compiled native block (which may cover many
    // guest instructions) or a single interpreted instruction when the PC
    // isn't JIT-able. Returns the same "cycles consumed" convention as
    // CPU::step() (2 per instruction, 1 on an instruction-fetch TLB fault).
    u32 run_step(CPU& cpu, Bus& bus);

    // Drops every cached block immediately. Safe to call from plain C++ (e.g.
    // Emulator::reset()) but never while a compiled block might be on the
    // call stack - see request_invalidate().
    void invalidate_all();

    // Requests that every cached block be dropped because [paddr, paddr+len)
    // was just written - but only if that range overlaps a page any cached
    // block was compiled from (see code_pages_); RDRAM writes to ordinary
    // data (by far the common case - heap, BSS clears, stacks, ...) are then
    // free. The actual work, when needed, is *deferred* to the start of the
    // next run_step() call rather than done synchronously: this is what the
    // global jit::notify_code_write() hook calls, and it can run while a
    // guest store executed *inside* a compiled block (e.g. SW to RDRAM,
    // nothing to do with code) is still routing through Bus::write32 with
    // that very block's native code still on the call stack.
    // invalidate_all() would mprotect away exec permission on the buffer out
    // from under the return address about to be used, which reliably
    // crashes. run_step()'s entry is only ever reached from ordinary C++, so
    // deferring to there avoids that reentrancy hazard entirely.
    void request_invalidate(u32 paddr, u32 len);

private:
    // Granularity of code_pages_: 64-byte "pages", deliberately much finer
    // than a 4 KB MMU page. Games keep hot globals right next to code (OoT
    // stores ~18 times a frame into .data sharing a 4 KB page with its boot
    // code), and at 4 KB granularity every such store dropped the whole cache.
    static constexpr u32 kPageShift = 6;
    // Direct-mapped cache instead of a hash map: run_step() looks this up on
    // *every* instruction, including ones that will never be JIT-able (see
    // the big comment on run_step() - a lot of real game code is FPU-heavy
    // and none of that is whitelisted yet, so most lookups miss the JIT
    // entirely and just want a cheap "no" on the way to cpu.step()).
    // std::unordered_map's hashing + bucket walk was measurably more
    // expensive than the whole rest of the interpreter fallback path
    // combined; a plain array indexed by a few bits of the address is a
    // single cache-friendly access. Collisions just evict - correctness is
    // unaffected either way, a compiled block gets rebuilt (or a
    // known-not-JIT-able verdict re-derived) on the next miss.
    struct CacheEntry {
        u32 tag = 0xFFFFFFFFu; // physical address this slot currently holds, or the sentinel for "empty"
        Block block;
    };
    static constexpr u32 kCacheBits = 16;
    static constexpr u32 kCacheSize = 1u << kCacheBits;
    std::vector<CacheEntry> cache_;

    CodeBuffer code_;
    std::vector<bool> code_pages_; // page index -> "a cached block covers this page"
    bool pending_invalidate_ = false;

    // Single-entry cache in front of cache_: a tight loop bounces between a
    // handful of fixed addresses (its body, its branch's fallback path,
    // ...), so the immediately-preceding lookup is very often an exact
    // repeat, cheaper still than the array index+tag check.
    u32 last_paddr_ = 0xFFFFFFFFu;
    const Block* last_block_ = nullptr;

    // A *taken* branch's delay-slot instruction always executes next no
    // matter what, with pc landing on the branch's target right after (see
    // CPU::jit_pending_delay_slot()) - never on delay_slot_addr+4. A normal
    // block compiled there would be wrong the moment it covered more than
    // one instruction (everything after the first would be instructions
    // that must *not* run next). So delay slots get their own compiler
    // entry point that always caps the block at exactly one instruction,
    // and - since a block compiled that way would silently become wrong if
    // reused for an ordinary (non-delay-slot) visit to the same address,
    // where more of the following code is a legal continuation - their own
    // same-sized cache, kept separate from cache_ rather than tagged into
    // it.
    std::vector<CacheEntry> ds_cache_;
    u32 last_ds_paddr_ = 0xFFFFFFFFu;
    const Block* last_ds_block_ = nullptr;

    static u32 cache_index(u32 paddr) { return (paddr >> 2) & (kCacheSize - 1); }
    const Block* lookup_or_compile_delay_slot(CPU& cpu, Bus& bus, u32 paddr);

    const Block* lookup_or_compile(CPU& cpu, Bus& bus, u64 pc, u32 paddr);
    Block compile_block(CPU& cpu, Bus& bus, u32 start_paddr, u32 max_len);
    void mark_code_pages(u32 start_paddr, u32 end_paddr);
};
