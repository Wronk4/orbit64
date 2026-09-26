#include "recompiler.hpp"
#include "jit_helpers.hpp"
#include "jit_invalidate.hpp"
#include "jit_decode.hpp"
#include "../cpu.hpp"
#include "../bus.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include "assembler_x64.hpp"
#define ORBIT64_JIT_X64 1
#elif defined(__aarch64__)
#include "backend_a64.hpp"
#include "assembler_a64.hpp"
#define ORBIT64_JIT_A64 1
#endif

using namespace jit_detail;

Recompiler::Recompiler() : code_pages_(RDRAM_SIZE >> kPageShift, 0), jcache_(1u << kJCacheBits) {
    clear_jcache();
    jit::set_invalidate_hook(this, [](void* owner, u32 paddr, u32 len) {
        static_cast<Recompiler*>(owner)->request_invalidate(paddr, len);
    });
}

Recompiler::~Recompiler() {
    jit::clear_invalidate_hook_if(this);
}

void Recompiler::invalidate_all() {
    for (BlockMap* map : {&blocks_, &ds_blocks_, &map_blocks_}) {
        for (auto& table : map->tables) table.reset();
        map->last_paddr = 0xFFFFFFFFu;
        map->last_block = nullptr;
    }
    code_.reset();
    std::fill(code_pages_.begin(), code_pages_.end(), 0);
    page_blocks_.clear();
    dirty_pages_.clear();
    pending_links_.clear();
    map_pending_links_.clear();
    clear_jcache();
    rt_ready_ = false; // the chaining runtime lived in the arena too
}

void Recompiler::clear_jcache() {
    // No pc is ever odd, so this never matches.
    std::fill(jcache_.begin(), jcache_.end(), JCacheEntry{~0ULL, nullptr});
}

const void* Recompiler::link_target(u32 paddr, bool mapped) const {
    if (paddr >= RDRAM_SIZE) return nullptr;
    const auto& table = (mapped ? map_blocks_ : blocks_).tables[paddr >> kTableShift];
    if (!table) return nullptr;
    const Block& b = (*table)[(paddr >> 2) & (kSlotsPerTable - 1)];
    return (b.valid && b.fn && b.link_ok) ? reinterpret_cast<const void*>(b.fn) : nullptr;
}

void Recompiler::request_invalidate(u32 paddr, u32 len) {
    if (len == 0) return;
    u32 first_page = paddr >> kPageShift;
    u32 last_page = (paddr + len - 1) >> kPageShift;
    for (u32 p = first_page; p <= last_page && p < code_pages_.size(); ++p) {
        if (code_pages_[p]) {
#if defined(ORBIT64_JIT_A64)
            // Chained blocks jump into each other: drop everything.
            pending_invalidate_ = true;
            if (active_ctx_) active_ctx_->exit_req = 1; // leave the chain after this store
            return;
#else
            dirty_pages_.push_back(p);
            if (active_ctx_) active_ctx_->exit_req = 1; // the chain may not run a dropped block
#endif
        }
    }
}

void Recompiler::mark_code_pages(u32 start_paddr, u32 end_paddr, bool delay_slot, [[maybe_unused]] bool mapped) {
    u32 first_page = start_paddr >> kPageShift;
    u32 last_page = (end_paddr == start_paddr) ? first_page : (end_paddr - 1) >> kPageShift;
    for (u32 p = first_page; p <= last_page && p < code_pages_.size(); ++p) {
        code_pages_[p] = 1;
#if !defined(ORBIT64_JIT_A64)
        const u32 entry = start_paddr | (delay_slot ? 1u : mapped ? 2u : 0u);
        std::vector<u32>& list = page_blocks_[p];
        if (list.empty() || list.back() != entry) list.push_back(entry);
#endif
    }
}

void Recompiler::drop_dirty_pages() {
    for (u32 p : dirty_pages_) {
        if (!code_pages_[p]) continue; // already dropped
        code_pages_[p] = 0;
        auto it = page_blocks_.find(p);
        if (it == page_blocks_.end()) continue;
        // A block that also covers other pages stays listed there; dropping
        // whatever sits at its address again later only costs a recompile.
        for (u32 entry : it->second) {
            BlockMap& map = (entry & 1) ? ds_blocks_ : (entry & 2) ? map_blocks_ : blocks_;
            const u32 paddr = entry & ~3u;
            if (auto& table = map.tables[paddr >> kTableShift]) (*table)[(paddr >> 2) & (kSlotsPerTable - 1)] = Block{};
        }
        page_blocks_.erase(it);
        if (stats_on_) stats_.page_drops++;
    }
    dirty_pages_.clear();
    for (BlockMap* map : {&blocks_, &ds_blocks_, &map_blocks_}) {
        map->last_paddr = 0xFFFFFFFFu;
        map->last_block = nullptr;
    }
    clear_jcache(); // it may point into dropped blocks
}

#if defined(ORBIT64_JIT_A64)
u32 Recompiler::run_step(CPU& cpu, Bus& bus) { return run(cpu, bus, 1); }

u32 Recompiler::run(CPU& cpu, Bus& bus, u32 budget) {
    if (pending_invalidate_) {
        pending_invalidate_ = false;
        invalidate_all();
        if (stats_on_) stats_.invalidations++;
    }

    // Translations cached in the dispatcher table for TLB-mapped code are only
    // good for the TLB contents and ASID they were made under.
    if (bus.tlb_generation() != tlb_gen_seen_ || cpu.jit_asid() != asid_seen_) {
        tlb_gen_seen_ = bus.tlb_generation();
        asid_seen_ = cpu.jit_asid();
        clear_jcache();
    }

    const u64 pc = cpu.get_pc();
    // KSEG0/KSEG1, in either its zero- or sign-extended 64-bit form (see the
    // other run_step() below), is RDRAM directly. Anything else goes through
    // the TLB exactly as an instruction fetch would; code outside RDRAM, or a
    // fetch that faults, is left to the interpreter.
    const u64 pc_hi = pc >> 32;
    const u32 pc32 = static_cast<u32>(pc);
    u32 paddr = pc32 & 0x1FFFFFFFu;
    bool mapped = false;
    if ((pc_hi != 0 && pc_hi != 0xFFFFFFFFull) || pc32 < 0x80000000u || pc32 > 0xBFFFFFFFu) {
        mapped = (pc & 3) == 0 && bus.translate_vaddr(pc, paddr, false, cpu.jit_asid()) == TLBResult::SUCCESS &&
                 paddr < RDRAM_SIZE;
        if (!mapped) {
            if (stats_on_) {
                stats_.interp_instrs++;
                stats_.interp_unmapped++;
            }
            return cpu.step();
        }
    }
    const bool delay_slot = cpu.jit_pending_delay_slot();
    const Block* blk = lookup_or_compile(cpu, bus, paddr, delay_slot, mapped);
    // COP1 code with the FPU disabled goes to the interpreter (see run_step()).
    // TODO: a chain can still jump straight into such a block.
    if (blk && blk->fpu && !(cpu.get_cp0(CP0Reg::STATUS) & (1u << 29))) blk = nullptr;
    if (!blk || !blk->fn || !rt_ready_) {
        if (stats_on_) count_fallback(bus, paddr);
        return cpu.step();
    }
    if (blk->link_ok) jcache_[(pc >> 2) & ((1u << kJCacheBits) - 1)] = {pc, reinterpret_cast<const void*>(blk->fn)};

    // The chain stops at the block that reaches COUNT == COMPARE, so the timer
    // interrupt is raised after exactly the block it would be with one block
    // per call (COUNT ticks once every 2 cycles; 0 ticks away = 2^32).
    const u32 ticks = static_cast<u32>(cpu.get_cp0(CP0Reg::COMPARE)) - static_cast<u32>(cpu.get_cp0(CP0Reg::COUNT));
    const u64 to_timer = ticks ? 2ULL * ticks : (2ULL << 32);
    const s64 limit = static_cast<s64>(std::min<u64>(std::max<u32>(budget, 1), to_timer));

    JitCtx ctx{&cpu, &bus, cpu.jit_gpr_ptr(), cpu.jit_hi_ptr(), cpu.jit_lo_ptr(),
               /*start_pc*/ pc,
               /*next_pc*/ delay_slot ? cpu.jit_branch_target() : 0,
               /*branch_taken*/ 1, // delay-slot sites are only ever flagged on the taken path
               /*faulted*/ 0,
               bus.get_rdram(),
               code_pages_.data(),
               /*cycles*/ limit,
               /*jcache*/ jcache_.data(),
               /*exit_req*/ 0,
               /*fr_mismatch*/ 0,
               /*synced*/ 0};
    active_ctx_ = &ctx;
    rt_.enter(&ctx, reinterpret_cast<const void*>(blk->fn));
    active_ctx_ = nullptr;
    const u32 consumed = static_cast<u32>(limit - ctx.cycles);
    if (stats_on_) {
        stats_.jit_entries++;
        stats_.jit_instrs += consumed / 2;
    }

    if (!ctx.faulted) {
        cpu.set_pc(ctx.next_pc);
        if (delay_slot) cpu.jit_clear_pending_delay_slot(); // see run_step() below
    }
    // A block compiled for the other STATUS.FR mode was reached: recompile
    // everything for the current one (FR hardly ever changes after boot).
    if (ctx.fr_mismatch) pending_invalidate_ = true;
    // Everything a single block does at its end, for the whole chain: nothing
    // in it could have raised an interrupt before its last block (see run()).
    // COP0 instructions in the chain have applied the cycles up to them already.
    if (consumed > 0) {
        const u32 rest = consumed - static_cast<u32>(ctx.synced);
        cpu.jit_advance_random(rest / 2);
        cpu.step_timer(rest);
        cpu.check_interrupts();
    }
    return consumed;
}
#else
u32 Recompiler::run_step(CPU& cpu, Bus& bus) { return run(cpu, bus, 1); }

u32 Recompiler::run(CPU& cpu, Bus& bus, u32 budget) {
    // Always reached from plain C++ (never from inside a compiled block), so
    // it's the one safe place to actually drop the cache - see
    // request_invalidate() for why this can't happen synchronously.
    if (pending_invalidate_) {
        pending_invalidate_ = false;
        invalidate_all();
        if (stats_on_) stats_.invalidations++;
    }
    if (!dirty_pages_.empty()) drop_dirty_pages();
    // The chain's pc -> block table holds TLB-mapped code under its virtual
    // pc, which is only right for the TLB contents and ASID it was made under.
    if (bus.tlb_generation() != tlb_gen_seen_ || cpu.jit_asid() != asid_seen_) {
        tlb_gen_seen_ = bus.tlb_generation();
        asid_seen_ = cpu.jit_asid();
        clear_jcache();
    }

    u64 pc = cpu.get_pc();
    // KSEG0/KSEG1 (the fixed, TLB-free window most game code runs from) is
    // RDRAM directly. In 32-bit mode addresses are sign-extended to 64 bits,
    // so the same window just as often shows up as 0xFFFFFFFF80000000-
    // 0xFFFFFFFFBFFFFFFF (any JR through a register loaded with LW, ERET to an
    // EPC saved from such a pc, ...) - in some games that is half of
    // everything executed, so both forms count. Anything else goes through the
    // TLB as an instruction fetch would (GoldenEye and Perfect Dark run almost
    // entirely from 0x7Fxxxxxx); code outside RDRAM, or a fetch that faults,
    // is left to the interpreter.
    const u64 pc_hi = pc >> 32;
    const u32 pc32 = static_cast<u32>(pc);
    u32 paddr = pc32 & 0x1FFFFFFFu;
    bool mapped = false;
    if ((pc_hi != 0 && pc_hi != 0xFFFFFFFFull) || pc32 < 0x80000000u || pc32 > 0xBFFFFFFFu) {
        mapped = (pc & 3) == 0 && translate_pc(cpu, bus, pc, paddr);
        if (!mapped) {
            if (stats_on_) {
                stats_.interp_instrs++;
                stats_.interp_unmapped++;
            }
            return cpu.step();
        }
    }

    // A taken branch's delay slot whose branch ran on the interpreter: pc
    // after this one instruction is fixed (jit_branch_target(), not pc+4),
    // so it gets its own capped-at-1-instruction compile instead of the
    // normal multi-instruction one - see ds_blocks_.
    const bool delay_slot = cpu.jit_pending_delay_slot();
    const Block* blk = lookup_or_compile(cpu, bus, paddr, delay_slot, mapped);
    // COP1 code with the FPU disabled: the interpreter raises Coprocessor
    // Unusable at the right instruction (compiled COP1 code doesn't check).
    if (blk && blk->fpu && !(cpu.get_cp0(CP0Reg::STATUS) & (1u << 29))) blk = nullptr;
    if (!blk || !blk->fn) {
        if (stats_on_) count_fallback(bus, paddr);
        return cpu.step();
    }

    // Other blocks' exits may jump straight into this one from now on - not
    // into a lone delay slot (where it continues is fixed by its branch) nor
    // an idle loop, which has to come back here to be skipped (see
    // Emulator::skip_idle_loop()).
    if (!delay_slot && blk->chain_entry && !is_idle_loop_at(bus.get_rdram(), bus.get_rdram_size(), paddr))
        jcache_[(pc >> 2) & ((1u << kJCacheBits) - 1)] = {pc, blk->chain_entry};

    // Blocks run on, each exit looking the next pc up in jcache_, until the
    // budget is used up or the COUNT == COMPARE timer is due (so the timer
    // interrupt comes after exactly the block it would with one block per
    // call; COUNT ticks once every 2 cycles, 0 ticks away = 2^32), or an
    // exception, an MMIO store or a write to compiled code ends the chain
    // (JitCtx::faulted / exit_req). Nothing else in a chain can raise an
    // interrupt: COP0 instructions end a block and run on the interpreter.
    const u32 ticks = static_cast<u32>(cpu.get_cp0(CP0Reg::COMPARE)) - static_cast<u32>(cpu.get_cp0(CP0Reg::COUNT));
    const u64 to_timer = ticks ? 2ULL * ticks : (2ULL << 32);
    // A lone delay slot runs by itself: the CPU's pending-delay-slot state is
    // only cleared once it's done (below), and the interpreter, which the
    // blocks call into, must not see it while other blocks run.
    const s64 limit = (chain_ && !delay_slot) ? static_cast<s64>(std::min<u64>(std::max<u32>(budget, 1), to_timer)) : 1;

    JitCtx ctx{&cpu, &bus, cpu.jit_gpr_ptr(), cpu.jit_hi_ptr(), cpu.jit_lo_ptr(),
               /*start_pc*/ pc,
               /*next_pc*/ delay_slot ? cpu.jit_branch_target() : pc + 4ULL * blk->length,
               /*branch_taken*/ delay_slot ? 1u : 0u,
               /*faulted*/ 0,
               bus.get_rdram(),
               code_pages_.data(),
               /*cycles*/ limit,
               /*jcache*/ jcache_.data(),
               /*exit_req*/ 0,
               /*fr_mismatch*/ 0,
               /*synced*/ 0};
    active_ctx_ = &ctx;
    blk->fn(&ctx);
    active_ctx_ = nullptr;
    const u32 consumed = static_cast<u32>(limit - ctx.cycles);
    const u32 executed = consumed / 2;
    if (stats_on_) {
        stats_.jit_entries++;
        stats_.jit_instrs += executed;
    }

    if (!ctx.faulted) {
        cpu.set_pc(ctx.next_pc);
        // CPU::step() always clears this at its own entry, before possibly
        // setting it again for a *new* branch; since this path never goes
        // through step(), it has to do that part explicitly too, or the next
        // instruction gets mistaken for another pending delay slot forever.
        if (delay_slot) cpu.jit_clear_pending_delay_slot();
    }
    // else: an instruction inside the block raised an exception, which
    // already redirected cpu's pc to the vector (and cleared any pending
    // delay slot - see CPU::trigger_exception).

    // Batched COUNT/compare-timer update + a single interrupt check for the
    // whole chain. This is coarser than the interpreter (which re-checks
    // after every instruction): an interrupt that becomes pending mid-block
    // is taken at the chain's end instead, i.e. a few instructions late -
    // never between a branch and its delay slot, since a block always
    // contains both. CP0 RANDOM (TLB-replacement index) is caught up the
    // same way: nothing inside a block reads it.
    if (executed > 0) {
        cpu.jit_advance_random(executed);
        cpu.step_timer(consumed);
        cpu.check_interrupts();
    }

    return consumed;
}
#endif

bool Recompiler::translate_pc(CPU& cpu, Bus& bus, u64 pc, u32& paddr) {
    const u8 asid = cpu.jit_asid();
    if ((pc >> 12) == tpc_vpage_ && bus.tlb_generation() == tpc_gen_ && asid == tpc_asid_) {
        paddr = tpc_ppage_ | static_cast<u32>(pc & 0xFFF);
        return true;
    }
    if (bus.translate_vaddr(pc, paddr, false, asid) != TLBResult::SUCCESS || paddr >= RDRAM_SIZE) return false;
    tpc_vpage_ = pc >> 12;
    tpc_ppage_ = paddr & ~0xFFFu;
    tpc_gen_ = bus.tlb_generation();
    tpc_asid_ = asid;
    return true;
}

const Recompiler::Block* Recompiler::lookup_or_compile(CPU& cpu, Bus& bus, u32 paddr, bool delay_slot, bool mapped) {
    // A delay slot block is a single instruction, so it doesn't matter how it
    // was reached.
    BlockMap& map = delay_slot ? ds_blocks_ : mapped ? map_blocks_ : blocks_;
    if (paddr == map.last_paddr) return map.last_block;
    if (paddr >= RDRAM_SIZE) return nullptr; // e.g. code running straight from cartridge ROM

    std::unique_ptr<BlockTable>* table = &map.tables[paddr >> kTableShift];
    Block* slot = *table ? &(**table)[(paddr >> 2) & (kSlotsPerTable - 1)] : nullptr;
    if (!slot || !slot->valid) {
        if (code_.remaining() < kArenaHeadroom) invalidate_all(); // also drops *table
        if (!*table) *table = std::make_unique<BlockTable>();
        slot = &(**table)[(paddr >> 2) & (kSlotsPerTable - 1)];
#if defined(ORBIT64_JIT_A64)
        std::vector<PendingLink> links;
        *slot = compile_block_linked(cpu, bus, paddr, delay_slot, mapped && !delay_slot, links);
#else
        *slot = compile_block(cpu, bus, paddr, delay_slot, mapped);
#endif
        slot->valid = true;
        if (slot->fn) {
            for (u32 i = 0; i < slot->length && !slot->fpu; ++i) {
                const u32 op = fetch_instr(bus.get_rdram(), paddr + i * 4) >> 26;
                slot->fpu = op == 0x11 || op == 0x31 || op == 0x35 || op == 0x39 || op == 0x3D;
            }
        }
        if (stats_on_) stats_.compiles++;
        if (slot->fn) mark_code_pages(paddr, paddr + slot->length * 4, delay_slot, mapped);
#if defined(ORBIT64_JIT_A64)
        if (slot->fn) {
            u8* code = reinterpret_cast<u8*>(slot->fn);
            auto& pending = (mapped && !delay_slot) ? map_pending_links_ : pending_links_;
            // Branches that were waiting for this block now jump straight to it.
            if (slot->link_ok) {
                if (auto it = pending.find(paddr); it != pending.end()) {
                    for (u8* site : it->second) code_.patch32(site, a64::Assembler::b_word(site, code));
                    pending.erase(it);
                }
            }
            for (const PendingLink& l : links) pending[l.target_paddr].push_back(code + l.offset);
        }
#endif
    }
    map.last_paddr = paddr;
    map.last_block = slot;
    return slot;
}

// ---------------------------------------------------------------------------
// Coverage statistics
// ---------------------------------------------------------------------------
namespace {

// Dense index for an instruction's "kind" (major opcode plus whichever
// sub-field selects the operation), so fallbacks can be histogrammed in a
// flat array. Layout: [0,64) major op, [64,128) SPECIAL funct, [128,160)
// REGIMM rt, [160,192) COP0 rs, [192,256) COP0 CO funct, [256,288) COP1 rs,
// [288,544) COP1 fmt S/D/W/L x funct.
constexpr u32 kFallbackKeys = 544;

u32 fallback_key(u32 raw) {
    const u32 op = raw >> 26, rs = (raw >> 21) & 31, rt = (raw >> 16) & 31, funct = raw & 63;
    switch (op) {
    case 0x00: return 64 + funct;
    case 0x01: return 128 + rt;
    case 0x10: return (rs == 0x10) ? 192 + funct : 160 + rs;
    case 0x11:
        if (rs == 0x10 || rs == 0x11) return 288 + (rs - 0x10) * 64 + funct;
        if (rs == 0x14 || rs == 0x15) return 288 + (rs - 0x12) * 64 + funct;
        return 256 + rs;
    default: return op;
    }
}

std::string fallback_key_name(u32 key) {
    static const char* const kOps[64] = {
        "SPECIAL", "REGIMM", "J", "JAL", "BEQ", "BNE", "BLEZ", "BGTZ",
        "ADDI", "ADDIU", "SLTI", "SLTIU", "ANDI", "ORI", "XORI", "LUI",
        "COP0", "COP1", "COP2", "COP3", "BEQL", "BNEL", "BLEZL", "BGTZL",
        "DADDI", "DADDIU", "LDL", "LDR", "op1C", "op1D", "op1E", "op1F",
        "LB", "LH", "LWL", "LW", "LBU", "LHU", "LWR", "LWU",
        "SB", "SH", "SWL", "SW", "SDL", "SDR", "SWR", "CACHE",
        "LL", "LWC1", "LWC2", "op33", "LLD", "LDC1", "LDC2", "LD",
        "SC", "SWC1", "SWC2", "op3B", "SCD", "SDC1", "SDC2", "SD"};
    static const char* const kSpecial[64] = {
        "SLL", "sp01", "SRL", "SRA", "SLLV", "sp05", "SRLV", "SRAV",
        "JR", "JALR", "MOVZ", "MOVN", "SYSCALL", "BREAK", "sp0E", "SYNC",
        "MFHI", "MTHI", "MFLO", "MTLO", "DSLLV", "sp15", "DSRLV", "DSRAV",
        "MULT", "MULTU", "DIV", "DIVU", "DMULT", "DMULTU", "DDIV", "DDIVU",
        "ADD", "ADDU", "SUB", "SUBU", "AND", "OR", "XOR", "NOR",
        "sp28", "sp29", "SLT", "SLTU", "DADD", "DADDU", "DSUB", "DSUBU",
        "TGE", "TGEU", "TLT", "TLTU", "TEQ", "sp35", "TNE", "sp37",
        "DSLL", "sp39", "DSRL", "DSRA", "DSLL32", "sp3D", "DSRL32", "DSRA32"};
    static const char* const kRegimm[32] = {
        "BLTZ", "BGEZ", "BLTZL", "BGEZL", "ri04", "ri05", "ri06", "ri07",
        "TGEI", "TGEIU", "TLTI", "TLTIU", "TEQI", "ri0D", "TNEI", "ri0F",
        "BLTZAL", "BGEZAL", "BLTZALL", "BGEZALL", "ri14", "ri15", "ri16", "ri17",
        "ri18", "ri19", "ri1A", "ri1B", "ri1C", "ri1D", "ri1E", "ri1F"};
    static const char* const kCop1Fn[64] = {
        "ADD", "SUB", "MUL", "DIV", "SQRT", "ABS", "MOV", "NEG",
        "ROUND.L", "TRUNC.L", "CEIL.L", "FLOOR.L", "ROUND.W", "TRUNC.W", "CEIL.W", "FLOOR.W",
        "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17",
        "f18", "f19", "f1A", "f1B", "f1C", "f1D", "f1E", "f1F",
        "CVT.S", "CVT.D", "f22", "f23", "CVT.W", "CVT.L", "f26", "f27",
        "f28", "f29", "f2A", "f2B", "f2C", "f2D", "f2E", "f2F",
        "C.F", "C.UN", "C.EQ", "C.UEQ", "C.OLT", "C.ULT", "C.OLE", "C.ULE",
        "C.SF", "C.NGLE", "C.SEQ", "C.NGL", "C.LT", "C.NGE", "C.LE", "C.NGT"};
    char buf[32];
    if (key < 64) return kOps[key];
    if (key < 128) return kSpecial[key - 64];
    if (key < 160) return kRegimm[key - 128];
    if (key < 192) {
        switch (key - 160) {
        case 0x00: return "MFC0";
        case 0x01: return "DMFC0";
        case 0x04: return "MTC0";
        case 0x05: return "DMTC0";
        default: std::snprintf(buf, sizeof buf, "COP0 rs=%02X", key - 160); return buf;
        }
    }
    if (key < 256) {
        switch (key - 192) {
        case 0x01: return "TLBR";
        case 0x02: return "TLBWI";
        case 0x06: return "TLBWR";
        case 0x08: return "TLBP";
        case 0x18: return "ERET";
        default: std::snprintf(buf, sizeof buf, "COP0.CO f=%02X", key - 192); return buf;
        }
    }
    if (key < 288) {
        switch (key - 256) {
        case 0x00: return "MFC1";
        case 0x01: return "DMFC1";
        case 0x02: return "CFC1";
        case 0x04: return "MTC1";
        case 0x05: return "DMTC1";
        case 0x06: return "CTC1";
        case 0x08: return "BC1";
        default: std::snprintf(buf, sizeof buf, "COP1 rs=%02X", key - 256); return buf;
        }
    }
    static const char kFmt[4] = {'S', 'D', 'W', 'L'};
    const u32 k = key - 288;
    std::snprintf(buf, sizeof buf, "%s.%c", kCop1Fn[k & 63], kFmt[k >> 6]);
    return buf;
}

} // namespace

void Recompiler::count_fallback(Bus& bus, u32 paddr) {
    stats_.interp_instrs++;
    if (static_cast<size_t>(paddr) + 4 > bus.get_rdram_size()) return;
    const u8* p = bus.get_rdram() + paddr;
    const u32 raw = (static_cast<u32>(p[0]) << 24) | (static_cast<u32>(p[1]) << 16) |
                    (static_cast<u32>(p[2]) << 8) | static_cast<u32>(p[3]);
    if (fallback_counts_.empty()) fallback_counts_.assign(kFallbackKeys, 0);
    fallback_counts_[fallback_key(raw)]++;
}

void Recompiler::print_stats(std::ostream& out) const {
    const Stats& s = stats_;
    const u64 total = s.jit_instrs + s.interp_instrs;
    auto pct = [total](u64 v) { return total ? 100.0 * static_cast<double>(v) / static_cast<double>(total) : 0.0; };
    char line[160];
    out << "[JIT] ---- coverage ----\n";
    std::snprintf(line, sizeof line, "[JIT] guest instructions : %llu\n", static_cast<unsigned long long>(total));
    out << line;
    std::snprintf(line, sizeof line, "[JIT]   compiled         : %llu (%.1f%%)\n",
                  static_cast<unsigned long long>(s.jit_instrs), pct(s.jit_instrs));
    out << line;
    std::snprintf(line, sizeof line, "[JIT]   interpreted      : %llu (%.1f%%), outside KSEG0/1: %llu (%.1f%%)\n",
                  static_cast<unsigned long long>(s.interp_instrs), pct(s.interp_instrs),
                  static_cast<unsigned long long>(s.interp_unmapped), pct(s.interp_unmapped));
    out << line;
    std::snprintf(line, sizeof line, "[JIT] block entries: %llu, avg %.2f instr/entry, compiles: %llu, cache drops: %llu, page drops: %llu\n",
                  static_cast<unsigned long long>(s.jit_entries),
                  s.jit_entries ? static_cast<double>(s.jit_instrs) / static_cast<double>(s.jit_entries) : 0.0,
                  static_cast<unsigned long long>(s.compiles), static_cast<unsigned long long>(s.invalidations),
                  static_cast<unsigned long long>(s.page_drops));
    out << line;

    std::vector<std::pair<u64, u32>> top;
    for (u32 k = 0; k < fallback_counts_.size(); ++k) {
        if (fallback_counts_[k]) top.emplace_back(fallback_counts_[k], k);
    }
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    out << "[JIT] top interpreted instructions (% of all executed):\n";
    for (size_t i = 0; i < top.size() && i < 30; ++i) {
        std::snprintf(line, sizeof line, "[JIT]   %-12s %12llu  %5.2f%%\n", fallback_key_name(top[i].second).c_str(),
                      static_cast<unsigned long long>(top[i].first), pct(top[i].first));
        out << line;
    }
}

// ===========================================================================
// x86-64 backend
// ===========================================================================
#if defined(ORBIT64_JIT_X64)
namespace {
using namespace x64;

// Context registers, fixed for the lifetime of a compiled block. None of
// these ever holds anything else, so per-instruction codegen can assume them.
constexpr int R_GPR = RBX;   // u64 gpr[32]
constexpr int R_PAGES = R12; // JitCtx::code_pages
constexpr int R_HI = R13;    // u64* (single u64)
constexpr int R_LO = R14;    // u64* (single u64)
constexpr int R_MEM = R15;   // JitCtx::rdram
constexpr int R_CTX = RBP;   // JitCtx* (original argument, for helper calls)
// x64::RSP collides with the (unrelated) forward-declared N64 `class RSP` -
// alias it locally so this file can stay unqualified everywhere else.
constexpr int R_SP = x64::RSP;

// Native calling convention for blocks (called from C++) and for the
// jit_helpers thunks they call. Everything register-level that differs
// between the two x86-64 ABIs is confined to these constants:
//   SysV (Linux/macOS): args RDI, RSI, RDX, RCX; no shadow space.
//   Win64 (Windows):    args RCX, RDX, R8, R9; the caller must reserve 32
//                       bytes of "shadow space" at [rsp] for every call,
//                       which the callee may freely overwrite. RDI/RSI and
//                       XMM6-15 are callee-saved there - codegen never
//                       touches RDI/RSI on Win64 and only ever uses XMM0/1.
// ARG1..ARG3 are also used as ordinary scratch while setting up a call, so
// codegen must never keep a live value in RCX/RDX across one either way.
#if defined(_WIN32)
constexpr int ARG0 = RCX, ARG1 = RDX, ARG2 = R8, ARG3 = R9;
constexpr u32 kShadowSpace = 32;
#else
constexpr int ARG0 = RDI, ARG1 = RSI, ARG2 = RDX, ARG3 = RCX;
constexpr u32 kShadowSpace = 0;
#endif

// Extra stack space reserved beyond the 6 pushes: after them (plus the
// return address) rsp is 8 mod 16, so this must be 8 mod 16 for every CALL
// to see an aligned stack. It holds the Win64 shadow space (if any) at
// [rsp], then a fixed, always-valid scratch slot above it that COP1 codegen
// uses to stash one FP register's raw bits across a second helper call -
// XMM registers (and every general-purpose scratch register) are
// caller-saved, so nothing survives a call except what's explicitly parked
// on the stack or in one of the context registers. The slot must sit above
// the shadow space, which a callee is allowed to overwrite.
constexpr s32 kFpScratchSlot = static_cast<s32>(kShadowSpace) + 8;
constexpr u32 kFrameExtra = kShadowSpace + 24; // 8 (alignment) + 16 (scratch slot, room to spare)
static_assert(kFrameExtra % 16 == 8, "CALLs from a block need a 16-byte aligned stack");

void emit_prologue(Assembler& x) {
    x.push(RBP); x.push(RBX); x.push(R12); x.push(R13); x.push(R14); x.push(R15);
    x.alu_ri(true, /*SUB*/ 5, R_SP, kFrameExtra);
    x.mov_reg_reg64(R_CTX, ARG0);
    x.load_mem64(R_MEM, ARG0, 64);
    x.load_mem64(R_PAGES, ARG0, 72);
    x.load_mem64(R_GPR, ARG0, 16);
    x.load_mem64(R_HI, ARG0, 24);
    x.load_mem64(R_LO, ARG0, 32);
}

void emit_epilogue(Assembler& x) {
    x.alu_ri(true, /*ADD*/ 0, R_SP, kFrameExtra);
    x.pop(R15); x.pop(R14); x.pop(R13); x.pop(R12); x.pop(RBX); x.pop(RBP);
    x.ret();
}

// Calls a jit_helpers thunk of the form `u32 fn(JitCtx*, u64 addr, u32 arg, u32 site)`.
// `addr` must already be in ARG1 and `arg` in ARG2. On return, if RAX != 0 the
// block aborts (a TLB fault redirected cpu->pc already); this bakes in the
// instructions up to and including this one as the returned "instructions
// executed" count (COUNT/RANDOM advance for the faulting one too, as in
// CPU::step()).
void emit_helper_call(Assembler& x, void* fn, u32 site, std::vector<size_t>& fault_patches) {
    x.mov_reg_reg64(ARG0, R_CTX);
    x.mov_reg_imm32(ARG3, site);
    x.mov_reg_imm64(RAX, reinterpret_cast<u64>(fn));
    x.call_reg(RAX);
    x.test_rr(false, RAX, RAX);
    size_t skip = x.jcc_rel32(Cc::E); // RAX==0 -> success, skip the fault path
    x.mov_reg_imm32(RAX, site_index(site) + 1); // the faulting instruction counts, as in the interpreter
    fault_patches.push_back(x.jmp_rel32());
    x.patch_rel32(skip, x.pos());
}

// Records a pending overflow branch (the native ADD/SUB/etc. just emitted
// already set OF; the Jcc(O) placeholder is resolved later).
void emit_overflow_check(Assembler& x, u32 site, std::vector<OverflowSite>& overflow_sites) {
    overflow_sites.push_back({x.jcc_rel32(Cc::O), site});
}

// ---- COP1 helpers: fetch/store an FPU register's raw bits via jit_fpr_*.
// These never fault (no TLB, no exceptions), so unlike loads/stores there's
// nothing to check on return - the result is just wherever the ABI puts it.
void emit_fpr_get(Assembler& x, void* fn, u8 reg) {
    x.mov_reg_reg64(ARG0, R_CTX);
    x.mov_reg_imm32(ARG1, reg);
    x.mov_reg_imm64(RAX, reinterpret_cast<u64>(fn));
    x.call_reg(RAX); // result in EAX/RAX
}
// `value_reg` holds the bits to store (RAX/EAX by convention here, matching
// the register emit_fpr_get() just left them in).
void emit_fpr_set(Assembler& x, void* fn, u8 reg, int value_reg) {
    x.mov_reg_reg64(ARG2, value_reg);
    x.mov_reg_imm32(ARG1, reg);
    x.mov_reg_reg64(ARG0, R_CTX);
    x.mov_reg_imm64(RAX, reinterpret_cast<u64>(fn));
    x.call_reg(RAX);
}

void emit_load(Assembler& x, const Decoded& d, void* fn, u32 site, std::vector<size_t>& fault_patches) {
    x.load_mem64(ARG1, R_GPR, d.rs * 8);
    x.alu_ri(true, /*ADD*/ 0, ARG1, static_cast<u32>(d.simm));
    x.mov_reg_imm32(ARG2, d.rt);
    emit_helper_call(x, fn, site, fault_patches);
}

void emit_store(Assembler& x, const Decoded& d, void* fn, u32 site, std::vector<size_t>& fault_patches) {
    x.load_mem64(ARG1, R_GPR, d.rs * 8);
    x.alu_ri(true, /*ADD*/ 0, ARG1, static_cast<u32>(d.simm));
    x.load_mem64(ARG2, R_GPR, d.rt * 8);
    emit_helper_call(x, fn, site, fault_patches);
}

// ---- Fastmem: KSEG0/KSEG1 loads/stores that hit RDRAM, inline ----
// Bus::translate_vaddr maps the low 32 bits of any such address straight to
// RDRAM, and Bus::read*/write* access RDRAM as plain big-endian bytes (a
// write's only side effect being jit::notify_code_write). So when the
// address is in either window, inside RDRAM and naturally aligned, the
// compiled code does exactly that itself; anything else - TLB-mapped, MMIO,
// cartridge, unaligned - takes the helper call as before.
static_assert(RDRAM_SIZE == 0x800000, "fastmem masks assume 8 MB of RDRAM");
// Bits 31, 30 and 28 down to log2(RDRAM the CPU sees): KSEG0/KSEG1 (bit 29
// free) within 8 MB (0xDF800000), or 4 MB without the Expansion Pak. Set per
// block from Bus::get_ram_limit (compile_block_x64).
u32 g_fastmem_mask = 0xDF800000u;

// RAX = guest address rs + simm; ECX = its RDRAM offset. Returns the Jcc to
// the slow path, taken when the fast path doesn't apply.
size_t emit_fastmem_address(Assembler& x, const Decoded& d, u32 size) {
    x.load_mem64(RAX, R_GPR, d.rs * 8);
    x.add_ri(true, RAX, static_cast<u32>(d.simm));
    x.mov_reg_reg32(RCX, RAX);
    x.and_ri(false, RCX, g_fastmem_mask | (size - 1));
    x.cmp_ri(false, RCX, 0x80000000u);
    const size_t slow = x.jcc_rel32(Cc::NE);
    x.mov_reg_reg32(RCX, RAX);
    x.and_ri(false, RCX, RDRAM_SIZE - 1);
    return slow;
}

void emit_fast_load(Assembler& x, const Decoded& d, void* fn, u32 site, std::vector<size_t>& fault_patches) {
    u32 size = 4;
    switch (d.op) {
    case 0x20: case 0x24: size = 1; break; // LB, LBU
    case 0x21: case 0x25: size = 2; break; // LH, LHU
    case 0x37: size = 8; break;            // LD
    default: break;                        // LW, LWU
    }
    const size_t slow = emit_fastmem_address(x, d, size);
    switch (d.op) {
    case 0x20: // LB
        x.load_idx8_zx(RAX, R_MEM, RCX);
        x.shl_ri(false, RAX, 24);
        x.sar_ri(false, RAX, 24);
        x.movsxd(RAX, RAX);
        break;
    case 0x24: // LBU
        x.load_idx8_zx(RAX, R_MEM, RCX);
        break;
    case 0x21: // LH
        x.load_idx16_zx(RAX, R_MEM, RCX);
        x.bswap(false, RAX);
        x.sar_ri(false, RAX, 16);
        x.movsxd(RAX, RAX);
        break;
    case 0x25: // LHU
        x.load_idx16_zx(RAX, R_MEM, RCX);
        x.bswap(false, RAX);
        x.shr_ri(false, RAX, 16);
        break;
    case 0x23: // LW
        x.load_idx32(RAX, R_MEM, RCX);
        x.bswap(false, RAX);
        x.movsxd(RAX, RAX);
        break;
    case 0x27: // LWU
        x.load_idx32(RAX, R_MEM, RCX);
        x.bswap(false, RAX);
        break;
    default: // LD
        x.load_idx64(RAX, R_MEM, RCX);
        x.bswap(true, RAX);
        break;
    }
    if (d.rt != 0) x.store_mem64(R_GPR, d.rt * 8, RAX);
    const size_t done = x.jmp_rel32();
    x.patch_rel32(slow, x.pos());
    emit_load(x, d, fn, site, fault_patches);
    x.patch_rel32(done, x.pos());
}

void emit_fast_store(Assembler& x, const Decoded& d, void* fn, u32 site, std::vector<size_t>& fault_patches) {
    u32 size = 4;
    switch (d.op) {
    case 0x28: size = 1; break; // SB
    case 0x29: size = 2; break; // SH
    case 0x3F: size = 8; break; // SD
    default: break;             // SW
    }
    const size_t slow = emit_fastmem_address(x, d, size);
    x.load_mem64(RDX, R_GPR, d.rt * 8);
    switch (size) {
    case 1:
        x.store_idx8(R_MEM, RCX, RDX);
        break;
    case 2:
        x.bswap(false, RDX);
        x.shr_ri(false, RDX, 16);
        x.store_idx16(R_MEM, RCX, RDX);
        break;
    case 4:
        x.bswap(false, RDX);
        x.store_idx32(R_MEM, RCX, RDX);
        break;
    default:
        x.bswap(true, RDX);
        x.store_idx64(R_MEM, RCX, RDX);
        break;
    }
    // Self-modifying code check, as Bus::write* does via notify_code_write:
    // only a store into a page some compiled block came from needs the call
    // (an aligned store never spans two 64-byte pages).
    x.mov_reg_reg32(RAX, RCX);
    x.shr_ri(false, RAX, 6);
    x.load_idx8_zx(RAX, R_PAGES, RAX);
    x.test_rr(false, RAX, RAX);
    const size_t no_code = x.jcc_rel32(Cc::E);
    x.mov_reg_reg32(ARG0, RCX);
    x.mov_reg_imm32(ARG1, size);
    x.mov_reg_imm64(RAX, reinterpret_cast<u64>(&jit_notify_code_write));
    x.call_reg(RAX);
    x.patch_rel32(no_code, x.pos());
    const size_t done = x.jmp_rel32();
    x.patch_rel32(slow, x.pos());
    emit_store(x, d, fn, site, fault_patches);
    x.patch_rel32(done, x.pos());
}

// Returns false if `d` isn't in the native whitelist (caller stops the block
// before this instruction; it is not included in the compiled block).
// Branches/jumps are never compiled here - see emit_branch().
bool compile_one(Assembler& x, const Decoded& d, u32 site, std::vector<size_t>& fault_patches,
                  std::vector<OverflowSite>& overflow_sites) {
    auto st_if = [&](u8 reg, int src) { if (reg != 0) x.store_mem64(R_GPR, reg * 8, src); };

    switch (d.op) {
    case 0x00: // SPECIAL
        switch (d.funct) {
        case 0x00: // SLL
            if (d.rd != 0) {
                x.load_mem32(RAX, R_GPR, d.rt * 8);
                x.shl_ri(false, RAX, d.shamt);
                x.movsxd(RAX, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x02: // SRL
            if (d.rd != 0) {
                x.load_mem32(RAX, R_GPR, d.rt * 8);
                x.shr_ri(false, RAX, d.shamt);
                x.movsxd(RAX, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x03: // SRA
            if (d.rd != 0) {
                x.load_mem32(RAX, R_GPR, d.rt * 8);
                x.sar_ri(false, RAX, d.shamt);
                x.movsxd(RAX, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x04: // SLLV
            if (d.rd != 0) {
                x.load_mem32(RAX, R_GPR, d.rt * 8);
                x.load_mem32(RCX, R_GPR, d.rs * 8);
                x.and_ri(false, RCX, 0x1F);
                x.shl_cl(false, RAX);
                x.movsxd(RAX, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x06: // SRLV
            if (d.rd != 0) {
                x.load_mem32(RAX, R_GPR, d.rt * 8);
                x.load_mem32(RCX, R_GPR, d.rs * 8);
                x.and_ri(false, RCX, 0x1F);
                x.shr_cl(false, RAX);
                x.movsxd(RAX, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x07: // SRAV
            if (d.rd != 0) {
                x.load_mem32(RAX, R_GPR, d.rt * 8);
                x.load_mem32(RCX, R_GPR, d.rs * 8);
                x.and_ri(false, RCX, 0x1F);
                x.sar_cl(false, RAX);
                x.movsxd(RAX, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x0A: // MOVZ (copy rs->rd when rt==0)
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rt * 8);
                x.test_rr(true, RAX, RAX);
                size_t skip = x.jcc_rel32(Cc::NE);
                x.load_mem64(RAX, R_GPR, d.rs * 8);
                x.store_mem64(R_GPR, d.rd * 8, RAX);
                x.patch_rel32(skip, x.pos());
            }
            return true;
        case 0x0B: // MOVN (copy rs->rd when rt!=0)
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rt * 8);
                x.test_rr(true, RAX, RAX);
                size_t skip = x.jcc_rel32(Cc::E);
                x.load_mem64(RAX, R_GPR, d.rs * 8);
                x.store_mem64(R_GPR, d.rd * 8, RAX);
                x.patch_rel32(skip, x.pos());
            }
            return true;
        case 0x0F: // SYNC (no-op)
            return true;
        case 0x10: // MFHI
            if (d.rd != 0) { x.load_mem64(RAX, R_HI, 0); st_if(d.rd, RAX); }
            return true;
        case 0x11: // MTHI
            x.load_mem64(RAX, R_GPR, d.rs * 8);
            x.store_mem64(R_HI, 0, RAX);
            return true;
        case 0x12: // MFLO
            if (d.rd != 0) { x.load_mem64(RAX, R_LO, 0); st_if(d.rd, RAX); }
            return true;
        case 0x13: // MTLO
            x.load_mem64(RAX, R_GPR, d.rs * 8);
            x.store_mem64(R_LO, 0, RAX);
            return true;
        case 0x14: // DSLLV
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rt * 8);
                x.load_mem32(RCX, R_GPR, d.rs * 8);
                x.and_ri(false, RCX, 0x3F);
                x.shl_cl(true, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x16: // DSRLV
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rt * 8);
                x.load_mem32(RCX, R_GPR, d.rs * 8);
                x.and_ri(false, RCX, 0x3F);
                x.shr_cl(true, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x17: // DSRAV
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rt * 8);
                x.load_mem32(RCX, R_GPR, d.rs * 8);
                x.and_ri(false, RCX, 0x3F);
                x.sar_cl(true, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x20: { // ADD (traps on 32-bit signed overflow - always checked, even if rd==$0)
            x.load_mem32(RAX, R_GPR, d.rs * 8);
            x.load_mem32(RCX, R_GPR, d.rt * 8);
            x.add_rr(false, RAX, RCX);
            emit_overflow_check(x, site, overflow_sites);
            x.movsxd(RAX, RAX);
            st_if(d.rd, RAX);
            return true;
        }
        case 0x21: // ADDU
            if (d.rd != 0) {
                x.load_mem32(RAX, R_GPR, d.rs * 8);
                x.load_mem32(RCX, R_GPR, d.rt * 8);
                x.add_rr(false, RAX, RCX);
                x.movsxd(RAX, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x22: { // SUB (traps on 32-bit signed overflow)
            x.load_mem32(RAX, R_GPR, d.rs * 8);
            x.load_mem32(RCX, R_GPR, d.rt * 8);
            x.sub_rr(false, RAX, RCX);
            emit_overflow_check(x, site, overflow_sites);
            x.movsxd(RAX, RAX);
            st_if(d.rd, RAX);
            return true;
        }
        case 0x23: // SUBU
            if (d.rd != 0) {
                x.load_mem32(RAX, R_GPR, d.rs * 8);
                x.load_mem32(RCX, R_GPR, d.rt * 8);
                x.sub_rr(false, RAX, RCX);
                x.movsxd(RAX, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x24: // AND
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rs * 8);
                x.load_mem64(RCX, R_GPR, d.rt * 8);
                x.and_rr(true, RAX, RCX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x25: // OR
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rs * 8);
                x.load_mem64(RCX, R_GPR, d.rt * 8);
                x.or_rr(true, RAX, RCX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x26: // XOR
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rs * 8);
                x.load_mem64(RCX, R_GPR, d.rt * 8);
                x.xor_rr(true, RAX, RCX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x27: // NOR
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rs * 8);
                x.load_mem64(RCX, R_GPR, d.rt * 8);
                x.or_rr(true, RAX, RCX);
                x.not_r(true, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x2A: // SLT
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rs * 8);
                x.load_mem64(RCX, R_GPR, d.rt * 8);
                x.cmp_rr(true, RAX, RCX);
                x.setcc(Cc::L, RAX);
                x.movzx8(RAX, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x2B: // SLTU
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rs * 8);
                x.load_mem64(RCX, R_GPR, d.rt * 8);
                x.cmp_rr(true, RAX, RCX);
                x.setcc(Cc::B, RAX);
                x.movzx8(RAX, RAX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x2C: { // DADD (traps on 64-bit signed overflow)
            x.load_mem64(RAX, R_GPR, d.rs * 8);
            x.load_mem64(RCX, R_GPR, d.rt * 8);
            x.add_rr(true, RAX, RCX);
            emit_overflow_check(x, site, overflow_sites);
            st_if(d.rd, RAX);
            return true;
        }
        case 0x2D: // DADDU
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rs * 8);
                x.load_mem64(RCX, R_GPR, d.rt * 8);
                x.add_rr(true, RAX, RCX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x2E: { // DSUB (traps on 64-bit signed overflow)
            x.load_mem64(RAX, R_GPR, d.rs * 8);
            x.load_mem64(RCX, R_GPR, d.rt * 8);
            x.sub_rr(true, RAX, RCX);
            emit_overflow_check(x, site, overflow_sites);
            st_if(d.rd, RAX);
            return true;
        }
        case 0x2F: // DSUBU
            if (d.rd != 0) {
                x.load_mem64(RAX, R_GPR, d.rs * 8);
                x.load_mem64(RCX, R_GPR, d.rt * 8);
                x.sub_rr(true, RAX, RCX);
                st_if(d.rd, RAX);
            }
            return true;
        case 0x38: // DSLL
            if (d.rd != 0) { x.load_mem64(RAX, R_GPR, d.rt * 8); x.shl_ri(true, RAX, d.shamt); st_if(d.rd, RAX); }
            return true;
        case 0x3A: // DSRL
            if (d.rd != 0) { x.load_mem64(RAX, R_GPR, d.rt * 8); x.shr_ri(true, RAX, d.shamt); st_if(d.rd, RAX); }
            return true;
        case 0x3B: // DSRA
            if (d.rd != 0) { x.load_mem64(RAX, R_GPR, d.rt * 8); x.sar_ri(true, RAX, d.shamt); st_if(d.rd, RAX); }
            return true;
        case 0x3C: // DSLL32
            if (d.rd != 0) { x.load_mem64(RAX, R_GPR, d.rt * 8); x.shl_ri(true, RAX, d.shamt + 32); st_if(d.rd, RAX); }
            return true;
        case 0x3E: // DSRL32
            if (d.rd != 0) { x.load_mem64(RAX, R_GPR, d.rt * 8); x.shr_ri(true, RAX, d.shamt + 32); st_if(d.rd, RAX); }
            return true;
        case 0x3F: // DSRA32
            if (d.rd != 0) { x.load_mem64(RAX, R_GPR, d.rt * 8); x.sar_ri(true, RAX, d.shamt + 32); st_if(d.rd, RAX); }
            return true;
        default:
            return false; // JR/JALR/traps/MULT/DIV/etc - interpreter handles it
        }

    case 0x08: { // ADDI (traps on 32-bit signed overflow)
        x.load_mem32(RAX, R_GPR, d.rs * 8);
        x.add_ri(false, RAX, static_cast<u32>(d.simm));
        emit_overflow_check(x, site, overflow_sites);
        x.movsxd(RAX, RAX);
        st_if(d.rt, RAX);
        return true;
    }
    case 0x09: // ADDIU
        if (d.rt != 0) {
            x.load_mem32(RAX, R_GPR, d.rs * 8);
            x.add_ri(false, RAX, static_cast<u32>(d.simm));
            x.movsxd(RAX, RAX);
            st_if(d.rt, RAX);
        }
        return true;
    case 0x0A: // SLTI
        if (d.rt != 0) {
            x.load_mem64(RAX, R_GPR, d.rs * 8);
            x.cmp_ri(true, RAX, static_cast<u32>(d.simm));
            x.setcc(Cc::L, RAX);
            x.movzx8(RAX, RAX);
            st_if(d.rt, RAX);
        }
        return true;
    case 0x0B: // SLTIU
        if (d.rt != 0) {
            x.load_mem64(RAX, R_GPR, d.rs * 8);
            x.cmp_ri(true, RAX, static_cast<u32>(d.simm));
            x.setcc(Cc::B, RAX);
            x.movzx8(RAX, RAX);
            st_if(d.rt, RAX);
        }
        return true;
    case 0x0C: // ANDI
        if (d.rt != 0) { x.load_mem64(RAX, R_GPR, d.rs * 8); x.and_ri(true, RAX, d.imm); st_if(d.rt, RAX); }
        return true;
    case 0x0D: // ORI
        if (d.rt != 0) { x.load_mem64(RAX, R_GPR, d.rs * 8); x.or_ri(true, RAX, d.imm); st_if(d.rt, RAX); }
        return true;
    case 0x0E: // XORI
        if (d.rt != 0) { x.load_mem64(RAX, R_GPR, d.rs * 8); x.xor_ri(true, RAX, d.imm); st_if(d.rt, RAX); }
        return true;
    case 0x0F: // LUI
        if (d.rt != 0) {
            s64 val = sign_extend_32_64(static_cast<s32>(static_cast<u32>(d.imm) << 16));
            x.mov_reg_imm64(RAX, static_cast<u64>(val));
            st_if(d.rt, RAX);
        }
        return true;
    case 0x18: { // DADDI (traps on 64-bit signed overflow)
        x.load_mem64(RAX, R_GPR, d.rs * 8);
        x.add_ri(true, RAX, static_cast<u32>(d.simm));
        emit_overflow_check(x, site, overflow_sites);
        st_if(d.rt, RAX);
        return true;
    }
    case 0x19: // DADDIU
        if (d.rt != 0) { x.load_mem64(RAX, R_GPR, d.rs * 8); x.add_ri(true, RAX, static_cast<u32>(d.simm)); st_if(d.rt, RAX); }
        return true;

    case 0x11: // COP1 - fmt/sub-op lives in d.rs, ft in d.rt, fs in d.rd, fd in d.shamt
        switch (d.rs) {
        case 0x00: // MFC1 rt, fs
            emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
            x.movsxd(RAX, RAX);
            st_if(d.rt, RAX);
            return true;
        case 0x01: // DMFC1 rt, fs
            emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
            st_if(d.rt, RAX);
            return true;
        case 0x04: // MTC1 rt, fs
            x.load_mem32(RAX, R_GPR, d.rt * 8);
            emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.rd, RAX);
            return true;
        case 0x05: // DMTC1 rt, fs
            x.load_mem64(RAX, R_GPR, d.rt * 8);
            emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.rd, RAX);
            return true;
        case 0x10: // .S (single precision)
            switch (d.funct) {
            case 0x06: // MOV.S
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, RAX);
                return true;
            case 0x05: // ABS.S - clear the sign bit, pure integer op
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                x.and_ri(false, RAX, 0x7FFFFFFFu);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, RAX);
                return true;
            case 0x07: // NEG.S - flip the sign bit, pure integer op
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                x.xor_ri(false, RAX, 0x80000000u);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, RAX);
                return true;
            case 0x00: case 0x01: case 0x02: { // ADD.S / SUB.S / MUL.S
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd); // fs
                x.store_mem32(R_SP, kFpScratchSlot, RAX);
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rt); // ft
                x.movd_to_xmm(1, RAX);
                x.load_mem32(RAX, R_SP, kFpScratchSlot);
                x.movd_to_xmm(0, RAX);
                if (d.funct == 0x00) x.addss(0, 1);
                else if (d.funct == 0x01) x.subss(0, 1);
                else x.mulss(0, 1);
                x.movd_from_xmm(RAX, 0);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, RAX);
                return true;
            }
            case 0x03: { // DIV.S - hardware divide, except the interpreter
                // special-cases a zero divisor to 0.0f instead of IEEE Inf/NaN.
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd); // fs
                x.store_mem32(R_SP, kFpScratchSlot, RAX);
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rt); // ft (divisor)
                x.mov_reg_reg64(RCX, RAX);
                x.and_ri(false, RCX, 0x7FFFFFFFu); // ignore sign: +0.0 and -0.0 both count as zero
                x.test_rr(false, RCX, RCX);
                size_t nz = x.jcc_rel32(Cc::NE);
                x.mov_reg_imm32(RAX, 0); // divisor is zero -> result is 0.0f
                size_t skip = x.jmp_rel32();
                x.patch_rel32(nz, x.pos());
                x.movd_to_xmm(1, RAX);
                x.load_mem32(RAX, R_SP, kFpScratchSlot);
                x.movd_to_xmm(0, RAX);
                x.divss(0, 1);
                x.movd_from_xmm(RAX, 0);
                x.patch_rel32(skip, x.pos());
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, RAX);
                return true;
            }
            case 0x04: // SQRT.S
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                x.movd_to_xmm(0, RAX);
                x.sqrtss(0, 0);
                x.movd_from_xmm(RAX, 0);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, RAX);
                return true;
            case 0x21: // CVT.D.S: float -> double
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                x.movd_to_xmm(0, RAX);
                x.cvtss2sd(0, 0);
                x.movq_from_xmm(RAX, 0);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, RAX);
                return true;
            case 0x24: // CVT.W.S: float -> 32-bit int (truncating)
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                x.movd_to_xmm(0, RAX);
                x.cvttss2si(RAX, 0, false);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, RAX);
                return true;
            case 0x25: // CVT.L.S: float -> 64-bit int (truncating)
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                x.movd_to_xmm(0, RAX);
                x.cvttss2si(RAX, 0, true);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, RAX);
                return true;
            default:
                if ((d.funct & 0x30) == 0x30) { // C.cond.S
                    x.mov_reg_reg64(ARG0, R_CTX);
                    x.mov_reg_imm32(ARG1, d.rd);
                    x.mov_reg_imm32(ARG2, d.rt);
                    x.mov_reg_imm32(ARG3, d.funct & 0x07);
                    x.mov_reg_imm64(RAX, reinterpret_cast<u64>(&jit_fpu_ccond_s));
                    x.call_reg(RAX);
                    return true;
                }
                return false; // ROUND/TRUNC/CEIL/FLOOR - interpreter handles it
            }
        case 0x11: // .D (double precision)
            switch (d.funct) {
            case 0x06: // MOV.D
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, RAX);
                return true;
            case 0x05: // ABS.D
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                x.mov_reg_imm64(RCX, 0x7FFFFFFFFFFFFFFFull);
                x.and_rr(true, RAX, RCX);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, RAX);
                return true;
            case 0x07: // NEG.D
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                x.mov_reg_imm64(RCX, 0x8000000000000000ull);
                x.xor_rr(true, RAX, RCX);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, RAX);
                return true;
            case 0x00: case 0x01: case 0x02: { // ADD.D / SUB.D / MUL.D
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd); // fs
                x.store_mem64(R_SP, kFpScratchSlot, RAX);
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rt); // ft
                x.movq_to_xmm(1, RAX);
                x.load_mem64(RAX, R_SP, kFpScratchSlot);
                x.movq_to_xmm(0, RAX);
                if (d.funct == 0x00) x.addsd(0, 1);
                else if (d.funct == 0x01) x.subsd(0, 1);
                else x.mulsd(0, 1);
                x.movq_from_xmm(RAX, 0);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, RAX);
                return true;
            }
            case 0x03: { // DIV.D - same zero-divisor special case as DIV.S, 64-bit
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd); // fs
                x.store_mem64(R_SP, kFpScratchSlot, RAX);
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rt); // ft (divisor)
                x.mov_reg_reg64(RCX, RAX);
                x.mov_reg_imm64(RDX, 0x7FFFFFFFFFFFFFFFull);
                x.and_rr(true, RCX, RDX);
                x.test_rr(true, RCX, RCX);
                size_t nz = x.jcc_rel32(Cc::NE);
                x.mov_reg_imm32(RAX, 0); // divisor is zero -> result is 0.0 (RAX zero-extends to 64 bits)
                size_t skip = x.jmp_rel32();
                x.patch_rel32(nz, x.pos());
                x.movq_to_xmm(1, RAX);
                x.load_mem64(RAX, R_SP, kFpScratchSlot);
                x.movq_to_xmm(0, RAX);
                x.divsd(0, 1);
                x.movq_from_xmm(RAX, 0);
                x.patch_rel32(skip, x.pos());
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, RAX);
                return true;
            }
            case 0x04: // SQRT.D
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                x.movq_to_xmm(0, RAX);
                x.sqrtsd(0, 0);
                x.movq_from_xmm(RAX, 0);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, RAX);
                return true;
            case 0x20: // CVT.S.D: double -> float
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                x.movq_to_xmm(0, RAX);
                x.cvtsd2ss(0, 0);
                x.movd_from_xmm(RAX, 0);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, RAX);
                return true;
            case 0x24: // CVT.W.D: double -> 32-bit int (truncating)
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                x.movq_to_xmm(0, RAX);
                x.cvttsd2si(RAX, 0, false);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, RAX);
                return true;
            case 0x25: // CVT.L.D: double -> 64-bit int (truncating)
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                x.movq_to_xmm(0, RAX);
                x.cvttsd2si(RAX, 0, true);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, RAX);
                return true;
            default:
                if ((d.funct & 0x30) == 0x30) { // C.cond.D
                    x.mov_reg_reg64(ARG0, R_CTX);
                    x.mov_reg_imm32(ARG1, d.rd);
                    x.mov_reg_imm32(ARG2, d.rt);
                    x.mov_reg_imm32(ARG3, d.funct & 0x07);
                    x.mov_reg_imm64(RAX, reinterpret_cast<u64>(&jit_fpu_ccond_d));
                    x.call_reg(RAX);
                    return true;
                }
                return false;
            }
        case 0x14: // .W (word) - fs holds a raw int32 stored in the FPR, not float bits
            switch (d.funct) {
            case 0x20: // CVT.S.W
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                x.cvtsi2ss(0, RAX, false);
                x.movd_from_xmm(RAX, 0);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, RAX);
                return true;
            case 0x21: // CVT.D.W
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                x.movsxd(RAX, RAX); // sign-extend the int32 before treating it as a 64-bit source width
                x.cvtsi2sd(0, RAX, true);
                x.movq_from_xmm(RAX, 0);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, RAX);
                return true;
            default: return false;
            }
        case 0x15: // .L (long) - fs holds a raw int64 stored in the FPR
            switch (d.funct) {
            case 0x20: // CVT.S.L
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                x.cvtsi2ss(0, RAX, true);
                x.movd_from_xmm(RAX, 0);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, RAX);
                return true;
            case 0x21: // CVT.D.L
                emit_fpr_get(x, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                x.cvtsi2sd(0, RAX, true);
                x.movq_from_xmm(RAX, 0);
                emit_fpr_set(x, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, RAX);
                return true;
            default: return false;
            }
        default: return false; // CFC1/CTC1/BC1 - interpreter handles it
        }

    case 0x20: emit_fast_load(x, d, reinterpret_cast<void*>(&jit_load_lb), site, fault_patches); return true;
    case 0x21: emit_fast_load(x, d, reinterpret_cast<void*>(&jit_load_lh), site, fault_patches); return true;
    case 0x23: emit_fast_load(x, d, reinterpret_cast<void*>(&jit_load_lw), site, fault_patches); return true;
    case 0x24: emit_fast_load(x, d, reinterpret_cast<void*>(&jit_load_lbu), site, fault_patches); return true;
    case 0x25: emit_fast_load(x, d, reinterpret_cast<void*>(&jit_load_lhu), site, fault_patches); return true;
    case 0x27: emit_fast_load(x, d, reinterpret_cast<void*>(&jit_load_lwu), site, fault_patches); return true;
    case 0x37: emit_fast_load(x, d, reinterpret_cast<void*>(&jit_load_ld), site, fault_patches); return true;

    case 0x28: emit_fast_store(x, d, reinterpret_cast<void*>(&jit_store_sb), site, fault_patches); return true;
    case 0x29: emit_fast_store(x, d, reinterpret_cast<void*>(&jit_store_sh), site, fault_patches); return true;
    case 0x2B: emit_fast_store(x, d, reinterpret_cast<void*>(&jit_store_sw), site, fault_patches); return true;
    case 0x3F: emit_fast_store(x, d, reinterpret_cast<void*>(&jit_store_sd), site, fault_patches); return true;

    case 0x2F: return true; // CACHE: a no-op, as in the interpreter

    default:
        return false; // branches/jumps/COP0/COP1/TLB/unaligned loads/etc.
    }
}

// Runs `d` through jit_interp (see is_interp_callable), with the same
// fault-exit convention as a load/store helper call.
void emit_interp_call(Assembler& x, const Decoded& d, u32 site, std::vector<size_t>& fault_patches) {
    x.mov_reg_reg64(ARG0, R_CTX);
    x.mov_reg_imm32(ARG1, d.raw);
    x.mov_reg_imm32(ARG2, site);
    x.mov_reg_imm64(RAX, reinterpret_cast<u64>(&jit_interp));
    x.call_reg(RAX);
    x.test_rr(false, RAX, RAX);
    size_t skip = x.jcc_rel32(Cc::E);
    x.mov_reg_imm32(RAX, site_index(site) + 1); // the faulting instruction counts, as in the interpreter
    fault_patches.push_back(x.jmp_rel32());
    x.patch_rel32(skip, x.pos());
}

// compile_one(), or failing that an in-place interpreter call; false if `d`
// has to end the block.
bool compile_or_call(Assembler& x, const Decoded& d, u32 site, std::vector<size_t>& fault_patches,
                     std::vector<OverflowSite>& overflow_sites) {
    if (compile_one(x, d, site, fault_patches, overflow_sites)) return true;
    if (!is_interp_callable(d)) return false;
    emit_interp_call(x, d, site, fault_patches);
    return true;
}

// JitCtx fields a block's terminating branch writes (see jit_abi.hpp).
constexpr s32 kCtxStartPc = 40;
constexpr s32 kCtxNextPc = 48;
constexpr s32 kCtxBranchTaken = 56;
constexpr s32 kCtxFaulted = 60;
constexpr s32 kCtxCycles = 80;
constexpr s32 kCtxJcache = 88;
constexpr s32 kCtxExitReq = 96;

// RAX = start_pc + offset: the pc of guest instruction offset/4 (or, past the
// end, the pc it falls through to) in whichever virtual alias the block was
// entered through.
void emit_pc_plus(Assembler& x, s32 offset) {
    x.load_mem64(RAX, R_CTX, kCtxStartPc);
    if (offset != 0) x.add_ri(true, RAX, static_cast<u32>(offset));
}

// Emits the branch/jump `d` at block index `n` (is_block_branch(d) must hold),
// up to - not including - its delay slot, which the caller compiles right
// after. Mirrors CPU::execute() exactly, including its order of operations:
// the link register is written before the branch's own operands are read.
// On return every path that reaches the delay slot has stored ctx->next_pc
// (and ctx->branch_taken = 1 if taken); a likely branch's not-taken path
// skips the delay slot and exits the block itself (its jump to the epilogue
// is added to exit_patches, with EAX = n + 1 instructions executed).
void emit_branch(Assembler& x, const Decoded& d, u32 n, std::vector<size_t>& exit_patches) {
    const s32 fallthrough = static_cast<s32>(4 * n + 8);
    auto store_next_pc = [&]() { x.store_mem64(R_CTX, kCtxNextPc, RAX); };
    auto mark_taken = [&]() {
        x.mov_reg_imm32(RCX, 1);
        x.store_mem32(R_CTX, kCtxBranchTaken, RCX);
    };
    auto link = [&](u8 reg) {
        if (reg == 0) return;
        emit_pc_plus(x, fallthrough);
        x.store_mem64(R_GPR, reg * 8, RAX);
    };

    // Unconditional jumps.
    if (d.op == 0x02 || d.op == 0x03) { // J, JAL
        if (d.op == 0x03) link(31);
        x.load_mem64(RAX, R_CTX, kCtxStartPc);
        x.mov_reg_imm64(RCX, 0xFFFFFFFFF0000000ull);
        x.and_rr(true, RAX, RCX);
        x.or_ri(true, RAX, (d.raw & 0x03FFFFFFu) << 2);
        store_next_pc();
        mark_taken();
        return;
    }
    if (d.op == 0x00) { // JR, JALR
        if (d.funct == 0x09) link(d.rd);
        x.load_mem64(RAX, R_GPR, d.rs * 8);
        store_next_pc();
        mark_taken();
        return;
    }

    // Conditional branches: set the flags, pick the "taken" condition.
    Cc taken_cc;
    bool likely = false;
    if (d.op == 0x01) { // REGIMM: BLTZ/BGEZ, +L, +AL, +ALL
        if (d.rt & 0x10) link(31);
        likely = (d.rt & 0x02) != 0;
        taken_cc = (d.rt & 0x01) ? Cc::GE : Cc::L;
        x.load_mem64(RAX, R_GPR, d.rs * 8);
        x.cmp_ri(true, RAX, 0);
    } else {
        likely = d.op >= 0x14;
        switch (d.op & 0x07) {
        case 0x04: taken_cc = Cc::E; break;  // BEQ
        case 0x05: taken_cc = Cc::NE; break; // BNE
        case 0x06: taken_cc = Cc::LE; break; // BLEZ
        default: taken_cc = Cc::G; break;    // BGTZ
        }
        x.load_mem64(RAX, R_GPR, d.rs * 8);
        if ((d.op & 0x07) <= 0x05) {
            x.load_mem64(RCX, R_GPR, d.rt * 8);
            x.cmp_rr(true, RAX, RCX);
        } else {
            x.cmp_ri(true, RAX, 0);
        }
    }

    const size_t not_taken = x.jcc_rel32(static_cast<Cc>(static_cast<u8>(taken_cc) ^ 1));
    emit_pc_plus(x, static_cast<s32>(4 * n + 4) + d.simm * 4);
    store_next_pc();
    mark_taken();
    const size_t to_delay_slot = x.jmp_rel32();

    x.patch_rel32(not_taken, x.pos());
    emit_pc_plus(x, fallthrough);
    store_next_pc();
    if (likely) {
        // Not taken: the delay slot is nullified - only the branch itself ran.
        x.mov_reg_imm32(RAX, n + 1);
        exit_patches.push_back(x.jmp_rel32());
    }
    // (Not taken, ordinary branch: branch_taken stays 0 - the driver preset
    // it - and the delay slot runs as a plain instruction, as it does in the
    // interpreter.)
    x.patch_rel32(to_delay_slot, x.pos());
}

} // namespace

// A block's exit, with EAX = guest instructions it executed: takes their
// cycles off JitCtx::cycles and, unless that ran out or the chain has to end
// (faulted / exit_req), looks next_pc up in the jcache (Recompiler::JCacheEntry
// {pc, entry}, 16 bytes, indexed by bits of pc >> 2) and jumps straight into
// that block - which runs on in this very stack frame, with the same context
// registers. Otherwise it returns to Recompiler::run(); the offset of that
// return is what this gives back.
static size_t emit_block_exit(Assembler& x, u32 jcache_bits) {
    size_t to_ret[3];
    x.add_rr(false, RAX, RAX);                    // cycles (zero-extends)
    x.load_mem64(RCX, R_CTX, kCtxCycles);
    x.sub_rr(true, RCX, RAX);
    x.store_mem64(R_CTX, kCtxCycles, RCX);
    to_ret[0] = x.jcc_rel32(Cc::LE);
    x.load_mem32(RCX, R_CTX, kCtxExitReq);
    x.load_mem32(RDX, R_CTX, kCtxFaulted);
    x.or_rr(false, RCX, RDX);
    to_ret[1] = x.jcc_rel32(Cc::NE);
    x.load_mem64(RCX, R_CTX, kCtxNextPc);
    x.mov_reg_reg64(RDX, RCX);
    x.shr_ri(true, RDX, 2);
    x.and_ri(false, RDX, (1u << jcache_bits) - 1);
    x.shl_ri(false, RDX, 4);
    x.load_mem64(RAX, R_CTX, kCtxJcache);
    x.add_rr(true, RDX, RAX);
    x.load_mem64(RAX, RDX, 0);
    x.cmp_rr(true, RAX, RCX);
    to_ret[2] = x.jcc_rel32(Cc::NE);
    x.store_mem64(R_CTX, kCtxStartPc, RCX);       // the next block starts here
    x.mov_reg_imm32(RAX, 0);
    x.store_mem32(R_CTX, kCtxBranchTaken, RAX);
    x.load_mem64(RAX, RDX, 8);
    x.jmp_reg(RAX);
    const size_t ret_pos = x.pos();
    for (int i = 0; i < 3; ++i) x.patch_rel32(to_ret[i], ret_pos);
    emit_epilogue(x);
    return ret_pos;
}

static Recompiler::Block compile_block_x64(CPU& cpu, Bus& bus, u32 start_paddr, CodeBuffer& code, bool delay_slot,
                                           bool mapped, u32 jcache_bits) {
    Recompiler::Block blk{};
    // Code reached through the TLB stops at the end of its 4 KB page: the
    // next virtual page may map anywhere.
    const size_t rdram_size = mapped ? std::min<size_t>((start_paddr & ~0xFFFu) + 0x1000, bus.get_rdram_size())
                                     : bus.get_rdram_size();
    const u8* rdram = bus.get_rdram();
    const u32 max_len = delay_slot ? 1 : kMaxBlockLen;
    g_fastmem_mask = 0xDFFFFFFFu & ~(bus.get_ram_limit() - 1);

    Assembler x;
    emit_prologue(x);
    const size_t body_pos = x.pos(); // where a chain enters (see emit_block_exit())
    // Jumps to the exit with EAX already holding the executed count:
    // load/store faults, and a likely branch's not-taken exit.
    std::vector<size_t> fault_patches;
    std::vector<OverflowSite> overflow_sites;

    u32 len = 0;
    bool branch_end = false;
    while (len < max_len) {
        const u32 paddr = start_paddr + len * 4;
        if (static_cast<size_t>(paddr) + 4 > rdram_size) break;
        const Decoded d = decode(fetch_instr(rdram, paddr));
        const u32 site = delay_slot ? (len | kSiteDelaySlot) : len;
        if (compile_or_call(x, d, site, fault_patches, overflow_sites)) {
            len++;
            continue;
        }
        // A branch ends the block, but only together with its delay slot -
        // so only if that is compilable too (dry run into a scratch
        // assembler); otherwise the branch is left to the interpreter.
        if (delay_slot || !is_block_branch(d) || static_cast<size_t>(paddr) + 8 > rdram_size) break;
        const Decoded ds = decode(fetch_instr(rdram, paddr + 4));
        {
            Assembler scratch;
            std::vector<size_t> scratch_faults;
            std::vector<OverflowSite> scratch_overflows;
            if (!compile_or_call(scratch, ds, 0, scratch_faults, scratch_overflows)) break;
        }
        emit_branch(x, d, len, fault_patches);
        compile_or_call(x, ds, (len + 1) | kSiteDelaySlot, fault_patches, overflow_sites);
        len += 2;
        branch_end = true;
        break;
    }
    if (len == 0) return blk;

    // Falling off the end continues right after the block. (A lone delay slot
    // continues at its branch's target, which the driver put in next_pc.)
    if (!branch_end && !delay_slot) {
        x.load_mem64(RAX, R_CTX, kCtxStartPc);
        x.add_ri(true, RAX, 4 * len);
        x.store_mem64(R_CTX, kCtxNextPc, RAX);
    }
    x.mov_reg_imm32(RAX, len);
    size_t epilogue_pos = x.pos();
    const size_t ret_pos = emit_block_exit(x, jcache_bits);
    for (size_t site : fault_patches) x.patch_rel32(site, epilogue_pos);
    // Out-of-line stubs for trapping arithmetic that overflowed: raise EXC_OV
    // via the interpreter's own trigger_exception() (through jit_overflow),
    // report the instructions up to and including it as executed, then fall into the same
    // epilogue as everything else.
    for (const OverflowSite& s : overflow_sites) {
        x.patch_rel32(s.patch, x.pos());
        x.mov_reg_reg64(ARG0, R_CTX);
        x.mov_reg_imm32(ARG1, s.site);
        x.mov_reg_imm64(RAX, reinterpret_cast<u64>(&jit_overflow));
        x.call_reg(RAX);
        x.mov_reg_imm32(RAX, site_index(s.site) + 1);
        x.patch_rel32(x.jmp_rel32(), epilogue_pos);
    }

    // A block with COP1 code may only be chained into while STATUS.CU1 is
    // set; otherwise the chain ends before it, with next_pc still pointing
    // at it, and Recompiler::run() leaves it to the interpreter, which raises
    // Coprocessor Unusable.
    bool fpu = false;
    for (u32 i = 0; i < len && !fpu; ++i) {
        const u32 op = fetch_instr(rdram, start_paddr + i * 4) >> 26;
        fpu = op == 0x11 || op == 0x31 || op == 0x35 || op == 0x39 || op == 0x3D;
    }
    size_t chain_pos = body_pos;
    if (fpu) {
        chain_pos = x.pos();
        x.mov_reg_imm64(RAX, reinterpret_cast<u64>(cpu.jit_cp0_ptr() + CP0Reg::STATUS));
        x.load_mem32(RAX, RAX, 0);
        x.and_ri(false, RAX, 1u << 29);
        const size_t to_body = x.jcc_rel32(Cc::NE);
        x.patch_rel32(to_body, body_pos);
        x.patch_rel32(x.jmp_rel32(), ret_pos); // straight back: nothing ran
    }

    // lookup_or_compile() starts the arena over (dropping every cached block)
    // long before it gets this full, so this only triggers if a block ever
    // outgrows kArenaHeadroom. Resetting the arena here instead would leave
    // the block caches pointing into code about to be overwritten.
    if (code.remaining() < x.size()) return blk; // left to the interpreter
    code.make_writable(x.size());
    u8* dst = code.write_ptr();
    std::memcpy(dst, x.data(), x.size());
    code.commit(x.size());
    code.make_executable();

    blk.fn = reinterpret_cast<JitBlockFn>(code.exec_ptr(dst));
    blk.chain_entry = code.exec_ptr(dst + chain_pos);
    blk.length = len;
    return blk;
}
#endif // ORBIT64_JIT_X64


#if defined(ORBIT64_JIT_A64)
Recompiler::Block Recompiler::compile_block_linked(CPU& cpu, Bus& bus, u32 start_paddr, bool delay_slot, bool mapped,
                                                   std::vector<PendingLink>& links) {
    if (!rt_ready_) {
        rt_ready_ = emit_a64_runtime(code_, rt_, kJCacheBits);
        if (!rt_ready_) return Block{};
    }
    A64BlockEnv env;
    env.rt = &rt_;
    env.chain = chain_;
    env.mapped = mapped;
    env.link_target = [this, mapped](u32 target) { return link_target(target, mapped); };
    return compile_block_a64(cpu, bus, start_paddr, code_, delay_slot, env, links);
}
#endif

Recompiler::Block Recompiler::compile_block([[maybe_unused]] CPU& cpu, Bus& bus, u32 start_paddr, bool delay_slot,
                                            [[maybe_unused]] bool mapped) {
#if defined(ORBIT64_JIT_X64)
    return compile_block_x64(cpu, bus, start_paddr, code_, delay_slot, mapped, kJCacheBits);
#elif defined(ORBIT64_JIT_A64)
    std::vector<PendingLink> links; // dropped: nothing will patch them
    A64BlockEnv env;
    if (!rt_ready_) rt_ready_ = emit_a64_runtime(code_, rt_, kJCacheBits);
    env.rt = &rt_;
    env.chain = false;
    env.link_target = [](u32) { return static_cast<const void*>(nullptr); };
    return compile_block_a64(cpu, bus, start_paddr, code_, delay_slot, env, links);
#else
    (void)bus; (void)start_paddr; (void)delay_slot;
    return Block{}; // unsupported host architecture: interpreter-only
#endif
}
