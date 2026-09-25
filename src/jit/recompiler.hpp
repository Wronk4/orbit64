#pragma once
// Dynamic recompiler for the CPU's hot path.
//
// A compiled block is a straight-line run of whitelisted instructions
// starting at some PC, optionally ended by a branch or jump together with
// its delay slot (the branch hands its outcome back through
// JitCtx::next_pc - blocks never jump to each other directly). The moment
// a non-whitelisted instruction is reached the block ends without including
// it, and that instruction runs on the existing, already-correct
// interpreter. Compiled to native code: integer ALU/shift/immediate
// arithmetic (including the trapping ADD/SUB forms), aligned loads/stores
// (through the same C++ memory paths the interpreter uses), a subset of COP1
// arithmetic, and conditional branches/jumps other than BC1T/BC1F. Kept
// inside a block as a call to the interpreter for that one instruction (see
// is_interp_callable in recompiler.cpp): MULT/DIV, FPU and unaligned
// loads/stores, LL/SC, CFC1/CTC1 and the remaining COP1 arithmetic. COP0/TLB,
// ERET, SYSCALL/BREAK/traps and BC1T/BC1F end a block.
//
// Blocks are only compiled for code living in KSEG0/KSEG1 (0x80000000-
// 0xBFFFFFFF, or its sign-extended 64-bit form), i.e. the fixed, TLB-free
// direct mapping to physical RDRAM that virtually all N64 game code runs
// from. Anything else always falls back to the interpreter. A block is keyed
// by physical address only and never bakes in its own virtual pc - branch
// targets and link values are computed from JitCtx::start_pc at run time -
// so the KSEG0 and KSEG1 aliases of the same code share one block.

#include "../common.hpp"
#include "code_buffer.hpp"
#include "jit_abi.hpp"
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

class CPU;
class Bus;

class Recompiler {
public:
    // Exposed so the (free-function) per-architecture codegen backends in
    // recompiler.cpp can build and return one; not part of the public API
    // otherwise.
    struct Block {
        // null (with valid) = "known not JIT-able at this PC". On AArch64 this
        // is not a C function but a chain entry point (see run()).
        JitBlockFn fn = nullptr;
        u32 length = 0;          // guest instructions covered (incl. a terminating branch + delay slot)
        bool valid = false;      // false = not compiled yet
        bool link_ok = false;    // AArch64: other blocks may jump straight here
    };

    // AArch64 chaining runtime, emitted at the start of the code arena (see
    // backend_a64.cpp): `enter` sets up a chain's registers and jumps to a
    // block; blocks leave through `exit`, or look their successor up through
    // `dispatch` when it isn't known at compile time.
    struct ChainRuntime {
        void (*enter)(JitCtx* ctx, const void* entry) = nullptr;
        const u8* exit = nullptr;
        const u8* dispatch = nullptr;
    };
    // A branch in compiled code that should jump straight to the block for
    // `target_paddr` once that is compiled; until then it leaves the chain.
    struct PendingLink {
        u32 target_paddr;
        u32 offset; // of the B instruction, from the start of its block's code
    };

    Recompiler();
    ~Recompiler();

    // Runs one interpreter step's worth of work starting at cpu.get_pc():
    // either a cached/newly-compiled native block (which may cover many
    // guest instructions) or a single interpreted instruction when the PC
    // isn't JIT-able. Returns the same "cycles consumed" convention as
    // CPU::step() (2 per instruction, 1 on an instruction-fetch TLB fault).
    u32 run_step(CPU& cpu, Bus& bus);

    // Like run_step(), but may run up to `budget` cycles' worth of compiled
    // code in one go: on AArch64, blocks chain directly into each other until
    // the budget is used up, the COUNT/COMPARE timer is about to fire, a store
    // hits MMIO or compiled code, or execution reaches code that has to be
    // interpreted. Interrupts are then checked exactly as after a single block
    // (nothing that could raise one happens mid-chain). Returns the cycles
    // consumed. Elsewhere this is run_step().
    u32 run(CPU& cpu, Bus& bus, u32 budget);

    // Chaining on/off (AArch64). Off, every block returns to the driver - the
    // same code and state, just slower, which makes it the reference to check
    // chaining against.
    void set_chaining(bool on) { chain_ = on; }

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

    // Optional coverage counters (off by default; `--jit-stats` in headless
    // mode). Answers "how much of what the game runs goes through compiled
    // code, and which instructions send it back to the interpreter".
    struct Stats {
        u64 jit_instrs = 0;      // guest instructions executed inside compiled blocks
        u64 jit_entries = 0;     // compiled-block calls (normal + delay slot)
        u64 interp_instrs = 0;   // cpu.step() fallbacks
        u64 interp_unmapped = 0; //   ... of which pc was outside KSEG0/KSEG1
        u64 compiles = 0;        // blocks compiled (incl. "not JIT-able" verdicts)
        u64 invalidations = 0;   // whole-cache drops
    };
    void set_stats_enabled(bool on) { stats_on_ = on; }
    void print_stats(std::ostream& out) const;

private:
    // Granularity of code_pages_: 64-byte "pages", deliberately much finer
    // than a 4 KB MMU page. Games keep hot globals right next to code (OoT
    // stores ~18 times a frame into .data sharing a 4 KB page with its boot
    // code), and at 4 KB granularity every such store dropped the whole cache.
    static constexpr u32 kPageShift = 6;

    // Block lookup: a two-level table with one slot per instruction address
    // of RDRAM, whose second level (4 KB of guest code each) is only
    // allocated once something in it is looked up. run_step() does this on
    // every step, including ones that end up interpreted, so it has to be a
    // couple of plain array accesses; unlike the direct-mapped cache this
    // replaced, nothing ever evicts anything, so a large game (OoT) no longer
    // keeps recompiling code whose slot a distant address had taken over.
    static constexpr u32 kTableShift = 12;
    static constexpr u32 kSlotsPerTable = (1u << kTableShift) / 4;
    using BlockTable = std::array<Block, kSlotsPerTable>;
    struct BlockMap {
        std::vector<std::unique_ptr<BlockTable>> tables = std::vector<std::unique_ptr<BlockTable>>(RDRAM_SIZE >> kTableShift);
        // Single-entry cache in front of the table: a tight loop bounces
        // between a handful of fixed addresses, so the immediately-preceding
        // lookup is very often an exact repeat.
        u32 last_paddr = 0xFFFFFFFFu;
        Block* last_block = nullptr;
    };

    // A *taken* branch's delay-slot instruction always executes next no
    // matter what, with pc landing on the branch's target right after (see
    // CPU::jit_pending_delay_slot()) - never on delay_slot_addr+4. This only
    // happens when the branch itself ran on the interpreter (a branch inside
    // a compiled block always brings its delay slot along). A normal block
    // compiled there would be wrong the moment it covered more than one
    // instruction, so delay slots get their own compiler entry point that
    // always caps the block at exactly one instruction - and, since a block
    // compiled that way would silently become wrong if reused for an
    // ordinary visit to the same address, their own map.
    BlockMap blocks_;
    BlockMap ds_blocks_;
    // AArch64: code reached through the TLB (e.g. GoldenEye runs at
    // 0x7000xxxx), keyed by the physical address it translated to. Such a
    // block never runs past its 4 KB page - the next virtual page may map
    // anywhere - and only links to blocks in the same page.
    BlockMap map_blocks_;

    CodeBuffer code_;
    std::vector<u8> code_pages_; // page index -> nonzero if a cached block covers this page (read by compiled stores)
    bool pending_invalidate_ = false;

    // --- AArch64 chaining (see run()) ---
    bool chain_ = true;
    ChainRuntime rt_;          // valid while rt_ready_; re-emitted after every arena reset
    bool rt_ready_ = false;
    // Dispatcher lookup: direct-mapped on the (virtual) pc, entries only for
    // blocks that allow linking. A miss just leaves the chain.
    struct JCacheEntry {
        u64 pc;
        const void* entry;
    };
    static constexpr u32 kJCacheBits = 13;
    std::vector<JCacheEntry> jcache_;
    // Branches waiting for their target block: target paddr -> B instructions
    // (for blocks_ and map_blocks_ respectively).
    std::unordered_map<u32, std::vector<u8*>> pending_links_;
    std::unordered_map<u32, std::vector<u8*>> map_pending_links_;
    // TLB contents / ASID the dispatcher table's entries for mapped code were
    // made under; any change drops the table.
    u32 tlb_gen_seen_ = 0;
    u8 asid_seen_ = 0;
    JitCtx* active_ctx_ = nullptr; // the running chain's context, for request_invalidate()

    const Block* lookup_or_compile(CPU& cpu, Bus& bus, u32 paddr, bool delay_slot, bool mapped = false);
    Block compile_block(CPU& cpu, Bus& bus, u32 start_paddr, bool delay_slot);
    // AArch64: compile_block() with linking; branches to blocks not compiled
    // yet are returned in `links`.
    Block compile_block_linked(CPU& cpu, Bus& bus, u32 start_paddr, bool delay_slot, bool mapped,
                               std::vector<PendingLink>& links);
    void mark_code_pages(u32 start_paddr, u32 end_paddr);
    // Entry of the compiled, linkable block at `paddr` (in map_blocks_ if
    // `mapped`), or null.
    const void* link_target(u32 paddr, bool mapped) const;
    void clear_jcache();

    bool stats_on_ = false;
    Stats stats_;
    std::vector<u64> fallback_counts_; // indexed by fallback_key(), sized lazily
    void count_fallback(Bus& bus, u64 pc);
};
