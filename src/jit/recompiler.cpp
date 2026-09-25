#include "recompiler.hpp"
#include "jit_helpers.hpp"
#include "jit_invalidate.hpp"
#include "../cpu.hpp"
#include "../bus.hpp"
#include <cstdio>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include "assembler_x64.hpp"
#define ORBIT64_JIT_X64 1
#elif defined(__aarch64__)
#include "assembler_a64.hpp"
#define ORBIT64_JIT_A64 1
#endif

namespace {

constexpr u32 kMaxBlockLen = 512;
// Free arena space lookup_or_compile() insists on before compiling: once less
// is left, every cached block is dropped and the arena starts over. Well above
// the largest block either backend can emit (kMaxBlockLen instructions, each
// well under 256 bytes even for the COP1 helper-call sequences).
constexpr size_t kArenaHeadroom = 256 * 1024;

// A single decoded MIPS instruction; shared by both backends so the opcode
// table only has to be reasoned about once. Whitelisting happens inside each
// backend's compile_one() (it returns false the moment it sees something it
// won't inline), not here.
struct Decoded {
    u32 raw;
    u8 op, rs, rt, rd, shamt, funct;
    u16 imm;   // zero-extended 16-bit immediate (ANDI/ORI/XORI)
    s32 simm;  // sign-extended 16-bit immediate (everything else)
};

// A pending "trapping arithmetic overflowed" branch: `patch` is where the
// native overflow-branch instruction was emitted (still pointing nowhere -
// patched once the out-of-line stub for it is emitted, same idea as the
// load/store fault_patches list), `site` identifies the guest instruction
// (see kSiteDelaySlot); its index is the instruction count to report.
struct OverflowSite {
    size_t patch;
    u32 site;
};

// Guest instructions that completed before the one `site` names.
constexpr u32 site_index(u32 site) { return site & ~kSiteDelaySlot; }

// Branches/jumps a block may end with (together with their delay slot).
// BC1T/BC1F are not among them: they read the FPU condition bit, which only
// the interpreter maintains.
[[maybe_unused]] bool is_block_branch(const Decoded& d) {
    switch (d.op) {
    case 0x00: return d.funct == 0x08 || d.funct == 0x09; // JR, JALR
    case 0x01: return d.rt <= 0x03 || (d.rt >= 0x10 && d.rt <= 0x13); // BLTZ..BGEZL, BLTZAL..BGEZALL
    case 0x02: case 0x03: // J, JAL
    case 0x04: case 0x05: case 0x06: case 0x07: // BEQ, BNE, BLEZ, BGTZ
    case 0x14: case 0x15: case 0x16: case 0x17: // ...likely
        return true;
    default: return false;
    }
}

// Instructions a block runs by calling the interpreter for just that one
// instruction (jit_interp) instead of ending there: a helper call costs about
// what interpreting it would, but the block - and everything after it -
// stays compiled. Only instructions that never branch, never depend on how
// far COUNT has advanced mid-block and never change interrupt state qualify
// (so no COP0), and only the forms each backend's compile_one() doesn't
// already inline - it is always tried first.
[[maybe_unused]] bool is_interp_callable(const Decoded& d) {
    switch (d.op) {
    case 0x00: return d.funct >= 0x18 && d.funct <= 0x1F; // MULT, MULTU, DIV, DIVU, DMULT, DMULTU, DDIV, DDIVU
    case 0x11: // COP1: CFC1/CTC1 and S/D/W/L arithmetic (ROUND/TRUNC/CEIL/FLOOR, ...)
        return d.rs == 0x02 || d.rs == 0x06 || d.rs == 0x10 || d.rs == 0x11 || d.rs == 0x14 || d.rs == 0x15;
    case 0x1A: case 0x1B: // LDL, LDR
    case 0x22: case 0x26: // LWL, LWR
    case 0x2A: case 0x2C: case 0x2D: case 0x2E: // SWL, SDL, SDR, SWR
    case 0x30: case 0x34: case 0x38: case 0x3C: // LL, LLD, SC, SCD
    case 0x31: case 0x35: case 0x39: case 0x3D: // LWC1, LDC1, SWC1, SDC1
        return true;
    default: return false;
    }
}

u32 fetch_instr(const u8* rdram, u32 paddr) {
    return (static_cast<u32>(rdram[paddr]) << 24) | (static_cast<u32>(rdram[paddr + 1]) << 16) |
           (static_cast<u32>(rdram[paddr + 2]) << 8) | static_cast<u32>(rdram[paddr + 3]);
}

Decoded decode(u32 instr) {
    Decoded d{};
    d.raw = instr;
    d.op = (instr >> 26) & 0x3F;
    d.rs = (instr >> 21) & 0x1F;
    d.rt = (instr >> 16) & 0x1F;
    d.rd = (instr >> 11) & 0x1F;
    d.shamt = (instr >> 6) & 0x1F;
    d.funct = instr & 0x3F;
    d.imm = instr & 0xFFFF;
    d.simm = static_cast<s16>(d.imm);
    return d;
}

} // namespace

Recompiler::Recompiler() : code_pages_(RDRAM_SIZE >> kPageShift, 0) {
    jit::set_invalidate_hook(this, [](void* owner, u32 paddr, u32 len) {
        static_cast<Recompiler*>(owner)->request_invalidate(paddr, len);
    });
}

Recompiler::~Recompiler() {
    jit::clear_invalidate_hook_if(this);
}

void Recompiler::invalidate_all() {
    for (BlockMap* map : {&blocks_, &ds_blocks_}) {
        for (auto& table : map->tables) table.reset();
        map->last_paddr = 0xFFFFFFFFu;
        map->last_block = nullptr;
    }
    code_.reset();
    std::fill(code_pages_.begin(), code_pages_.end(), 0);
}

void Recompiler::request_invalidate(u32 paddr, u32 len) {
    if (len == 0) return;
    u32 first_page = paddr >> kPageShift;
    u32 last_page = (paddr + len - 1) >> kPageShift;
    for (u32 p = first_page; p <= last_page && p < code_pages_.size(); ++p) {
        if (code_pages_[p]) {
            pending_invalidate_ = true;
            return;
        }
    }
}

void Recompiler::mark_code_pages(u32 start_paddr, u32 end_paddr) {
    u32 first_page = start_paddr >> kPageShift;
    u32 last_page = (end_paddr == start_paddr) ? first_page : (end_paddr - 1) >> kPageShift;
    for (u32 p = first_page; p <= last_page && p < code_pages_.size(); ++p) {
        code_pages_[p] = 1;
    }
}

u32 Recompiler::run_step(CPU& cpu, Bus& bus) {
    // Always reached from plain C++ (never from inside a compiled block), so
    // it's the one safe place to actually drop the cache - see
    // request_invalidate() for why this can't happen synchronously.
    if (pending_invalidate_) {
        pending_invalidate_ = false;
        invalidate_all();
        if (stats_on_) stats_.invalidations++;
    }

    u64 pc = cpu.get_pc();
    // Only KSEG0/KSEG1 (the fixed, TLB-free window almost all game code runs
    // from) is ever JIT-compiled; everything else - mapped segments, boot ROM
    // shenanigans, etc. - just uses the interpreter. In 32-bit mode addresses
    // are sign-extended to 64 bits, so the same window just as often shows up
    // as 0xFFFFFFFF80000000-0xFFFFFFFFBFFFFFFF (any JR through a register
    // loaded with LW, ERET to an EPC saved from such a pc, ...) - in some
    // games that is half of everything executed, so both forms count.
    const u64 pc_hi = pc >> 32;
    const u32 pc32 = static_cast<u32>(pc);
    if ((pc_hi != 0 && pc_hi != 0xFFFFFFFFull) || pc32 < 0x80000000u || pc32 > 0xBFFFFFFFu) {
        if (stats_on_) {
            stats_.interp_instrs++;
            stats_.interp_unmapped++;
        }
        return cpu.step();
    }
    const u32 paddr = pc32 & 0x1FFFFFFFu;

    // A taken branch's delay slot whose branch ran on the interpreter: pc
    // after this one instruction is fixed (jit_branch_target(), not pc+4),
    // so it gets its own capped-at-1-instruction compile instead of the
    // normal multi-instruction one - see ds_blocks_.
    const bool delay_slot = cpu.jit_pending_delay_slot();
    const Block* blk = lookup_or_compile(cpu, bus, paddr, delay_slot);
    if (!blk || !blk->fn) {
        if (stats_on_) count_fallback(bus, pc);
        return cpu.step();
    }

    JitCtx ctx{&cpu, &bus, cpu.jit_gpr_ptr(), cpu.jit_hi_ptr(), cpu.jit_lo_ptr(),
               /*start_pc*/ pc,
               /*next_pc*/ delay_slot ? cpu.jit_branch_target() : pc + 4ULL * blk->length,
               /*branch_taken*/ delay_slot ? 1u : 0u,
               /*faulted*/ 0,
               bus.get_rdram(),
               code_pages_.data()};
    const u32 executed = blk->fn(&ctx);
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
    // whole block. This is coarser than the interpreter (which re-checks
    // after every instruction): an interrupt that becomes pending mid-block
    // is taken at the block's end instead, i.e. a few instructions late -
    // never between a branch and its delay slot, since a block always
    // contains both. CP0 RANDOM (TLB-replacement index) is deliberately
    // *not* decremented per compiled instruction - only the interpreter does
    // that precisely - since whitelisted blocks never touch the TLB
    // themselves.
    if (executed > 0) {
        cpu.step_timer(2 * executed);
        cpu.check_interrupts();
    }

    return executed * 2;
}

const Recompiler::Block* Recompiler::lookup_or_compile(CPU& cpu, Bus& bus, u32 paddr, bool delay_slot) {
    BlockMap& map = delay_slot ? ds_blocks_ : blocks_;
    if (paddr == map.last_paddr) return map.last_block;
    if (paddr >= RDRAM_SIZE) return nullptr; // e.g. code running straight from cartridge ROM

    std::unique_ptr<BlockTable>* table = &map.tables[paddr >> kTableShift];
    Block* slot = *table ? &(**table)[(paddr >> 2) & (kSlotsPerTable - 1)] : nullptr;
    if (!slot || !slot->valid) {
        if (code_.remaining() < kArenaHeadroom) invalidate_all(); // also drops *table
        if (!*table) *table = std::make_unique<BlockTable>();
        slot = &(**table)[(paddr >> 2) & (kSlotsPerTable - 1)];
        *slot = compile_block(cpu, bus, paddr, delay_slot);
        slot->valid = true;
        if (stats_on_) stats_.compiles++;
        if (slot->fn) mark_code_pages(paddr, paddr + slot->length * 4);
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

void Recompiler::count_fallback(Bus& bus, u64 pc) {
    stats_.interp_instrs++;
    const u32 paddr = static_cast<u32>(pc) & 0x1FFFFFFFu;
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
    std::snprintf(line, sizeof line, "[JIT] block entries: %llu, avg %.2f instr/entry, compiles: %llu, cache drops: %llu\n",
                  static_cast<unsigned long long>(s.jit_entries),
                  s.jit_entries ? static_cast<double>(s.jit_instrs) / static_cast<double>(s.jit_entries) : 0.0,
                  static_cast<unsigned long long>(s.compiles), static_cast<unsigned long long>(s.invalidations));
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
// instruction's index as the returned "instructions executed" count.
void emit_helper_call(Assembler& x, void* fn, u32 site, std::vector<size_t>& fault_patches) {
    x.mov_reg_reg64(ARG0, R_CTX);
    x.mov_reg_imm32(ARG3, site);
    x.mov_reg_imm64(RAX, reinterpret_cast<u64>(fn));
    x.call_reg(RAX);
    x.test_rr(false, RAX, RAX);
    size_t skip = x.jcc_rel32(Cc::E); // RAX==0 -> success, skip the fault path
    x.mov_reg_imm32(RAX, site_index(site));
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
constexpr u32 kFastmemMask = 0xDF800000u; // bit 31, 30, 28..23: KSEG0/KSEG1 (bit 29 free) within 8 MB

// RAX = guest address rs + simm; ECX = its RDRAM offset. Returns the Jcc to
// the slow path, taken when the fast path doesn't apply.
size_t emit_fastmem_address(Assembler& x, const Decoded& d, u32 size) {
    x.load_mem64(RAX, R_GPR, d.rs * 8);
    x.add_ri(true, RAX, static_cast<u32>(d.simm));
    x.mov_reg_reg32(RCX, RAX);
    x.and_ri(false, RCX, kFastmemMask | (size - 1));
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
    x.mov_reg_imm32(RAX, site_index(site));
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

static Recompiler::Block compile_block_x64(Bus& bus, u32 start_paddr, CodeBuffer& code, bool delay_slot) {
    Recompiler::Block blk{};
    const size_t rdram_size = bus.get_rdram_size();
    const u8* rdram = bus.get_rdram();
    const u32 max_len = delay_slot ? 1 : kMaxBlockLen;

    Assembler x;
    emit_prologue(x);
    // Jumps to the epilogue with EAX already holding the executed count:
    // load/store faults, and a likely branch's not-taken exit.
    std::vector<size_t> fault_patches;
    std::vector<OverflowSite> overflow_sites;

    u32 len = 0;
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
        break;
    }
    if (len == 0) return blk;

    x.mov_reg_imm32(RAX, len);
    size_t epilogue_pos = x.pos();
    emit_epilogue(x);
    for (size_t site : fault_patches) x.patch_rel32(site, epilogue_pos);
    // Out-of-line stubs for trapping arithmetic that overflowed: raise EXC_OV
    // via the interpreter's own trigger_exception() (through jit_overflow),
    // report the instructions before it as executed, then fall into the same
    // epilogue as everything else.
    for (const OverflowSite& s : overflow_sites) {
        x.patch_rel32(s.patch, x.pos());
        x.mov_reg_reg64(ARG0, R_CTX);
        x.mov_reg_imm32(ARG1, s.site);
        x.mov_reg_imm64(RAX, reinterpret_cast<u64>(&jit_overflow));
        x.call_reg(RAX);
        x.mov_reg_imm32(RAX, site_index(s.site));
        x.patch_rel32(x.jmp_rel32(), epilogue_pos);
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
    blk.length = len;
    return blk;
}
#endif // ORBIT64_JIT_X64

// ===========================================================================
// AArch64 backend
// ===========================================================================
#if defined(ORBIT64_JIT_A64)
namespace {
using namespace a64;

// Context registers (AAPCS64 callee-saved X19-X24), fixed for the lifetime
// of a compiled block.
constexpr int X_CPU = 19; // CPU* (unused directly by codegen; kept for symmetry)
constexpr int X_BUS = 20; // Bus* (unused directly; helpers take ctx instead)
constexpr int X_GPR = 21; // u64 gpr[32]
constexpr int X_HI = 22;  // u64*
constexpr int X_LO = 23;  // u64*
constexpr int X_CTX = 24; // JitCtx*
// Scratch: X0-X17 freely, except while a helper call is being set up (X0-X2
// are that call's arguments) - codegen never needs a scratch value to survive
// across a call.
constexpr int S0 = 0, S1 = 1, S2 = 2, S3 = 9;
// COP1 arithmetic needs one integer scratch slot to stash an FPU register's
// raw bits across a second helper call (X0-X17 are all caller-saved, same
// reasoning as the x64 backend's kFpScratchSlot). The 64-byte frame already
// reserved in emit_prologue only uses its first 56 bytes (LR + X19-X24);
// this borrows the free 8 bytes at the end instead of growing the frame.
constexpr s32 kFpScratchSlot = 56;

void emit_prologue(Assembler& a) {
    a.sub_sp(64);
    a.str64(30, SP, 0);  // LR
    a.str64(19, SP, 8);
    a.str64(20, SP, 16);
    a.str64(21, SP, 24);
    a.str64(22, SP, 32);
    a.str64(23, SP, 40);
    a.str64(24, SP, 48);
    a.ldr64(X_CPU, 0, 0);
    a.ldr64(X_BUS, 0, 8);
    a.ldr64(X_GPR, 0, 16);
    a.ldr64(X_HI, 0, 24);
    a.ldr64(X_LO, 0, 32);
    a.mov_reg(X_CTX, 0);
}

void emit_epilogue(Assembler& a) {
    // LR is reloaded but never branched to directly - RET always targets X30.
    a.ldr64(30, SP, 0);
    a.ldr64(19, SP, 8);
    a.ldr64(20, SP, 16);
    a.ldr64(21, SP, 24);
    a.ldr64(22, SP, 32);
    a.ldr64(23, SP, 40);
    a.ldr64(24, SP, 48);
    a.add_sp(64);
    a.ret();
}

// Calls a jit_helpers thunk `u32 fn(JitCtx*, u64 addr, u32 arg, u32 site)` with
// X1=addr, X2=arg already set. On return, W0!=0 means a TLB fault redirected
// cpu->pc; the block then reports the instruction's index as the number of
// guest instructions executed and returns.
void emit_helper_call(Assembler& a, void* fn, u32 site, std::vector<size_t>& fault_patches) {
    a.mov_reg(0, X_CTX);
    a.mov_imm64(3, site);
    a.mov_imm64(S3, reinterpret_cast<u64>(fn));
    a.blr(S3);
    size_t skip = a.cbz(0, false); // W0==0 -> success, skip the fault path
    a.movz(0, static_cast<u16>(site_index(site)), 0, false);
    fault_patches.push_back(a.b());
    a.patch_branch(skip, a.pos());
}

void emit_load(Assembler& a, const Decoded& d, void* fn, u32 site, std::vector<size_t>& fault_patches) {
    a.ldr64(1, X_GPR, d.rs * 8);
    a.mov_imm64_sext32(S3, d.simm);
    a.add_reg(1, 1, S3, true);
    a.movz(2, d.rt, 0, false);
    emit_helper_call(a, fn, site, fault_patches);
}

void emit_store(Assembler& a, const Decoded& d, void* fn, u32 site, std::vector<size_t>& fault_patches) {
    a.ldr64(1, X_GPR, d.rs * 8);
    a.mov_imm64_sext32(S3, d.simm);
    a.add_reg(1, 1, S3, true);
    a.ldr64(2, X_GPR, d.rt * 8); // value to store
    emit_helper_call(a, fn, site, fault_patches);
}

// ---- COP1 helpers: fetch/store an FPU register's raw bits via jit_fpr_*.
// These never fault, so - unlike load/store - there's nothing to check on
// return; the result is just wherever the ABI puts it (X0).
void emit_fpr_get(Assembler& a, void* fn, u8 reg) {
    a.mov_reg(0, X_CTX);
    a.movz(1, reg, 0, false);
    a.mov_imm64(S3, reinterpret_cast<u64>(fn));
    a.blr(S3); // result in X0/W0
}
// `value_reg` holds the bits to store (X0 by convention, matching where
// emit_fpr_get() just left them).
void emit_fpr_set(Assembler& a, void* fn, u8 reg, int value_reg) {
    a.mov_reg(2, value_reg);
    a.movz(1, reg, 0, false);
    a.mov_reg(0, X_CTX);
    a.mov_imm64(S3, reinterpret_cast<u64>(fn));
    a.blr(S3);
}

// Records a pending overflow branch (the native ADDS/SUBS just emitted
// already set the V flag; the B.VS placeholder is resolved later).
void emit_overflow_check(Assembler& a, u32 site, std::vector<OverflowSite>& overflow_sites) {
    overflow_sites.push_back({a.bcond(Cond::VS), site});
}

bool compile_one(Assembler& a, const Decoded& d, u32 site, std::vector<size_t>& fault_patches,
                  std::vector<OverflowSite>& overflow_sites) {
    auto st_if = [&](u8 reg, int src) { if (reg != 0) a.str64(src, X_GPR, reg * 8); };
    // shamt materialized into S2 for the variable-shift instructions.
    auto shift32 = [&](void (Assembler::*op)(int, int, int, bool), u8 rt, u8 shamt, u8 rd) {
        a.ldr32z(S0, X_GPR, rt * 8);
        a.movz(S2, shamt, 0, false);
        (a.*op)(S0, S0, S2, false);
        a.sxtw(S0, S0);
        st_if(rd, S0);
    };
    auto shiftv32 = [&](void (Assembler::*op)(int, int, int, bool), u8 rt, u8 rs, u8 rd) {
        a.ldr32z(S0, X_GPR, rt * 8);
        a.ldr32z(S1, X_GPR, rs * 8);
        a.movz(S2, 0x1F, 0, false);
        a.and_reg(S1, S1, S2, false);
        (a.*op)(S0, S0, S1, false);
        a.sxtw(S0, S0);
        st_if(rd, S0);
    };
    auto shift64 = [&](void (Assembler::*op)(int, int, int, bool), u8 rt, u8 shamt, u8 rd) {
        a.ldr64(S0, X_GPR, rt * 8);
        a.movz(S2, shamt, 0, false);
        (a.*op)(S0, S0, S2, true);
        st_if(rd, S0);
    };
    auto shiftv64 = [&](void (Assembler::*op)(int, int, int, bool), u8 rt, u8 rs, u8 rd) {
        a.ldr64(S0, X_GPR, rt * 8);
        a.ldr64(S1, X_GPR, rs * 8);
        a.movz(S2, 0x3F, 0, false);
        a.and_reg(S1, S1, S2, true);
        (a.*op)(S0, S0, S1, true);
        st_if(rd, S0);
    };

    switch (d.op) {
    case 0x00: // SPECIAL
        switch (d.funct) {
        case 0x00: if (d.rd != 0) shift32(&Assembler::lslv, d.rt, d.shamt, d.rd); return true; // SLL
        case 0x02: if (d.rd != 0) shift32(&Assembler::lsrv, d.rt, d.shamt, d.rd); return true; // SRL
        case 0x03: if (d.rd != 0) shift32(&Assembler::asrv, d.rt, d.shamt, d.rd); return true; // SRA
        case 0x04: if (d.rd != 0) shiftv32(&Assembler::lslv, d.rt, d.rs, d.rd); return true;   // SLLV
        case 0x06: if (d.rd != 0) shiftv32(&Assembler::lsrv, d.rt, d.rs, d.rd); return true;   // SRLV
        case 0x07: if (d.rd != 0) shiftv32(&Assembler::asrv, d.rt, d.rs, d.rd); return true;   // SRAV
        case 0x0A: // MOVZ
            if (d.rd != 0) {
                a.ldr64(S0, X_GPR, d.rt * 8);
                size_t skip = a.cbnz(S0, true);
                a.ldr64(S0, X_GPR, d.rs * 8);
                a.str64(S0, X_GPR, d.rd * 8);
                a.patch_branch(skip, a.pos());
            }
            return true;
        case 0x0B: // MOVN
            if (d.rd != 0) {
                a.ldr64(S0, X_GPR, d.rt * 8);
                size_t skip = a.cbz(S0, true);
                a.ldr64(S0, X_GPR, d.rs * 8);
                a.str64(S0, X_GPR, d.rd * 8);
                a.patch_branch(skip, a.pos());
            }
            return true;
        case 0x0F: return true; // SYNC (no-op)
        case 0x10: if (d.rd != 0) { a.ldr64(S0, X_HI, 0); st_if(d.rd, S0); } return true; // MFHI
        case 0x11: a.ldr64(S0, X_GPR, d.rs * 8); a.str64(S0, X_HI, 0); return true;       // MTHI
        case 0x12: if (d.rd != 0) { a.ldr64(S0, X_LO, 0); st_if(d.rd, S0); } return true; // MFLO
        case 0x13: a.ldr64(S0, X_GPR, d.rs * 8); a.str64(S0, X_LO, 0); return true;       // MTLO
        case 0x14: if (d.rd != 0) shiftv64(&Assembler::lslv, d.rt, d.rs, d.rd); return true; // DSLLV
        case 0x16: if (d.rd != 0) shiftv64(&Assembler::lsrv, d.rt, d.rs, d.rd); return true; // DSRLV
        case 0x17: if (d.rd != 0) shiftv64(&Assembler::asrv, d.rt, d.rs, d.rd); return true; // DSRAV
        case 0x20: { // ADD (traps on 32-bit signed overflow - always checked, even if rd==$0)
            a.ldr32z(S0, X_GPR, d.rs * 8); a.ldr32z(S1, X_GPR, d.rt * 8);
            a.adds_reg(S0, S0, S1, false);
            emit_overflow_check(a, site, overflow_sites);
            a.sxtw(S0, S0); st_if(d.rd, S0);
            return true;
        }
        case 0x21: // ADDU
            if (d.rd != 0) {
                a.ldr32z(S0, X_GPR, d.rs * 8); a.ldr32z(S1, X_GPR, d.rt * 8);
                a.add_reg(S0, S0, S1, false); a.sxtw(S0, S0); st_if(d.rd, S0);
            }
            return true;
        case 0x22: { // SUB (traps on 32-bit signed overflow)
            a.ldr32z(S0, X_GPR, d.rs * 8); a.ldr32z(S1, X_GPR, d.rt * 8);
            a.subs_reg(S0, S0, S1, false);
            emit_overflow_check(a, site, overflow_sites);
            a.sxtw(S0, S0); st_if(d.rd, S0);
            return true;
        }
        case 0x23: // SUBU
            if (d.rd != 0) {
                a.ldr32z(S0, X_GPR, d.rs * 8); a.ldr32z(S1, X_GPR, d.rt * 8);
                a.sub_reg(S0, S0, S1, false); a.sxtw(S0, S0); st_if(d.rd, S0);
            }
            return true;
        case 0x24: // AND
            if (d.rd != 0) { a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8); a.and_reg(S0, S0, S1, true); st_if(d.rd, S0); }
            return true;
        case 0x25: // OR
            if (d.rd != 0) { a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8); a.orr_reg(S0, S0, S1, true); st_if(d.rd, S0); }
            return true;
        case 0x26: // XOR
            if (d.rd != 0) { a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8); a.eor_reg(S0, S0, S1, true); st_if(d.rd, S0); }
            return true;
        case 0x27: // NOR
            if (d.rd != 0) {
                a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8);
                a.orr_reg(S0, S0, S1, true); a.mvn_reg(S0, S0, true); st_if(d.rd, S0);
            }
            return true;
        case 0x2A: // SLT
            if (d.rd != 0) {
                a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8);
                a.cmp_reg(S0, S1, true); a.cset(S0, Cond::LT, true); st_if(d.rd, S0);
            }
            return true;
        case 0x2B: // SLTU
            if (d.rd != 0) {
                a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8);
                a.cmp_reg(S0, S1, true); a.cset(S0, Cond::CC, true); st_if(d.rd, S0);
            }
            return true;
        case 0x2C: { // DADD (traps on 64-bit signed overflow)
            a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8);
            a.adds_reg(S0, S0, S1, true);
            emit_overflow_check(a, site, overflow_sites);
            st_if(d.rd, S0);
            return true;
        }
        case 0x2D: // DADDU
            if (d.rd != 0) { a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8); a.add_reg(S0, S0, S1, true); st_if(d.rd, S0); }
            return true;
        case 0x2E: { // DSUB (traps on 64-bit signed overflow)
            a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8);
            a.subs_reg(S0, S0, S1, true);
            emit_overflow_check(a, site, overflow_sites);
            st_if(d.rd, S0);
            return true;
        }
        case 0x2F: // DSUBU
            if (d.rd != 0) { a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8); a.sub_reg(S0, S0, S1, true); st_if(d.rd, S0); }
            return true;
        case 0x38: if (d.rd != 0) shift64(&Assembler::lslv, d.rt, d.shamt, d.rd); return true;      // DSLL
        case 0x3A: if (d.rd != 0) shift64(&Assembler::lsrv, d.rt, d.shamt, d.rd); return true;      // DSRL
        case 0x3B: if (d.rd != 0) shift64(&Assembler::asrv, d.rt, d.shamt, d.rd); return true;      // DSRA
        case 0x3C: if (d.rd != 0) shift64(&Assembler::lslv, d.rt, d.shamt + 32, d.rd); return true; // DSLL32
        case 0x3E: if (d.rd != 0) shift64(&Assembler::lsrv, d.rt, d.shamt + 32, d.rd); return true; // DSRL32
        case 0x3F: if (d.rd != 0) shift64(&Assembler::asrv, d.rt, d.shamt + 32, d.rd); return true; // DSRA32
        default: return false;
        }

    case 0x08: { // ADDI (traps on 32-bit signed overflow)
        a.ldr32z(S0, X_GPR, d.rs * 8); a.mov_imm64_sext32(S1, d.simm);
        a.adds_reg(S0, S0, S1, false);
        emit_overflow_check(a, site, overflow_sites);
        a.sxtw(S0, S0); st_if(d.rt, S0);
        return true;
    }
    case 0x09: // ADDIU
        if (d.rt != 0) {
            a.ldr32z(S0, X_GPR, d.rs * 8); a.mov_imm64_sext32(S1, d.simm);
            a.add_reg(S0, S0, S1, false); a.sxtw(S0, S0); st_if(d.rt, S0);
        }
        return true;
    case 0x0A: // SLTI
        if (d.rt != 0) {
            a.ldr64(S0, X_GPR, d.rs * 8); a.mov_imm64_sext32(S1, d.simm);
            a.cmp_reg(S0, S1, true); a.cset(S0, Cond::LT, true); st_if(d.rt, S0);
        }
        return true;
    case 0x0B: // SLTIU
        if (d.rt != 0) {
            a.ldr64(S0, X_GPR, d.rs * 8); a.mov_imm64_sext32(S1, d.simm);
            a.cmp_reg(S0, S1, true); a.cset(S0, Cond::CC, true); st_if(d.rt, S0);
        }
        return true;
    case 0x0C: // ANDI
        if (d.rt != 0) { a.ldr64(S0, X_GPR, d.rs * 8); a.movz(S1, d.imm, 0); a.and_reg(S0, S0, S1, true); st_if(d.rt, S0); }
        return true;
    case 0x0D: // ORI
        if (d.rt != 0) { a.ldr64(S0, X_GPR, d.rs * 8); a.movz(S1, d.imm, 0); a.orr_reg(S0, S0, S1, true); st_if(d.rt, S0); }
        return true;
    case 0x0E: // XORI
        if (d.rt != 0) { a.ldr64(S0, X_GPR, d.rs * 8); a.movz(S1, d.imm, 0); a.eor_reg(S0, S0, S1, true); st_if(d.rt, S0); }
        return true;
    case 0x0F: // LUI
        if (d.rt != 0) {
            s64 val = sign_extend_32_64(static_cast<s32>(static_cast<u32>(d.imm) << 16));
            a.mov_imm64(S0, static_cast<u64>(val));
            st_if(d.rt, S0);
        }
        return true;
    case 0x18: { // DADDI (traps on 64-bit signed overflow)
        a.ldr64(S0, X_GPR, d.rs * 8); a.mov_imm64_sext32(S1, d.simm);
        a.adds_reg(S0, S0, S1, true);
        emit_overflow_check(a, site, overflow_sites);
        st_if(d.rt, S0);
        return true;
    }
    case 0x19: // DADDIU
        if (d.rt != 0) {
            a.ldr64(S0, X_GPR, d.rs * 8); a.mov_imm64_sext32(S1, d.simm);
            a.add_reg(S0, S0, S1, true); st_if(d.rt, S0);
        }
        return true;

    case 0x11: // COP1 - fmt/sub-op lives in d.rs, ft in d.rt, fs in d.rd, fd in d.shamt
        switch (d.rs) {
        case 0x00: // MFC1 rt, fs
            emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
            a.sxtw(S0, 0);
            st_if(d.rt, S0);
            return true;
        case 0x01: // DMFC1 rt, fs
            emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
            st_if(d.rt, 0);
            return true;
        case 0x04: // MTC1 rt, fs
            a.ldr32z(0, X_GPR, d.rt * 8);
            emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.rd, 0);
            return true;
        case 0x05: // DMTC1 rt, fs
            a.ldr64(0, X_GPR, d.rt * 8);
            emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.rd, 0);
            return true;
        case 0x10: // .S (single precision)
            switch (d.funct) {
            case 0x06: // MOV.S
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, 0);
                return true;
            case 0x05: // ABS.S - clear the sign bit, pure integer op
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                a.mov_imm64_sext32(1, 0x7FFFFFFF); // low 32 bits: 0x7FFFFFFF regardless of upper bits
                a.and_reg(0, 0, 1, false);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, 0);
                return true;
            case 0x07: // NEG.S - flip the sign bit, pure integer op
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                a.mov_imm64_sext32(1, static_cast<s32>(0x80000000u)); // low 32 bits: 0x80000000
                a.eor_reg(0, 0, 1, false);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, 0);
                return true;
            case 0x00: case 0x01: case 0x02: { // ADD.S / SUB.S / MUL.S
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd); // fs
                a.str64(0, SP, kFpScratchSlot);
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rt); // ft
                a.fmov_to_fp(1, 0, false);
                a.ldr64(0, SP, kFpScratchSlot);
                a.fmov_to_fp(0, 0, false);
                if (d.funct == 0x00) a.fadd(0, 0, 1, false);
                else if (d.funct == 0x01) a.fsub(0, 0, 1, false);
                else a.fmul(0, 0, 1, false);
                a.fmov_from_fp(0, 0, false);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, 0);
                return true;
            }
            case 0x03: { // DIV.S - hardware divide, except the interpreter
                // special-cases a zero divisor to 0.0f instead of IEEE Inf/NaN.
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd); // fs
                a.str64(0, SP, kFpScratchSlot);
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rt); // ft (divisor)
                a.mov_reg(2, 0);
                a.mov_imm64_sext32(S3, 0x7FFFFFFF);
                a.and_reg(2, 2, S3, false); // ignore sign: +0.0 and -0.0 both count as zero
                size_t nz = a.cbnz(2, false);
                a.movz(0, 0, 0, false); // divisor is zero -> result is 0.0f
                size_t skip = a.b();
                a.patch_branch(nz, a.pos());
                a.fmov_to_fp(1, 0, false);
                a.ldr64(0, SP, kFpScratchSlot);
                a.fmov_to_fp(0, 0, false);
                a.fdiv(0, 0, 1, false);
                a.fmov_from_fp(0, 0, false);
                a.patch_branch(skip, a.pos());
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, 0);
                return true;
            }
            case 0x04: // SQRT.S
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                a.fmov_to_fp(0, 0, false);
                a.fsqrt(0, 0, false);
                a.fmov_from_fp(0, 0, false);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, 0);
                return true;
            case 0x21: // CVT.D.S: float -> double
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                a.fmov_to_fp(0, 0, false);
                a.fcvt_s_to_d(0, 0);
                a.fmov_from_fp(0, 0, true);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, 0);
                return true;
            case 0x24: // CVT.W.S: float -> 32-bit int (truncating)
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                a.fmov_to_fp(0, 0, false);
                a.fcvtzs(0, 0, false, false);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, 0);
                return true;
            case 0x25: // CVT.L.S: float -> 64-bit int (truncating)
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                a.fmov_to_fp(0, 0, false);
                a.fcvtzs(0, 0, true, false);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, 0);
                return true;
            default:
                if ((d.funct & 0x30) == 0x30) { // C.cond.S
                    a.mov_reg(0, X_CTX);
                    a.movz(1, d.rd, 0, false);
                    a.movz(2, d.rt, 0, false);
                    a.movz(3, d.funct & 0x07, 0, false);
                    a.mov_imm64(S3, reinterpret_cast<u64>(&jit_fpu_ccond_s));
                    a.blr(S3);
                    return true;
                }
                return false; // ROUND/TRUNC/CEIL/FLOOR - interpreter handles it
            }
        case 0x11: // .D (double precision)
            switch (d.funct) {
            case 0x06: // MOV.D
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, 0);
                return true;
            case 0x05: // ABS.D
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                a.mov_imm64(1, 0x7FFFFFFFFFFFFFFFull);
                a.and_reg(0, 0, 1, true);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, 0);
                return true;
            case 0x07: // NEG.D
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                a.mov_imm64(1, 0x8000000000000000ull);
                a.eor_reg(0, 0, 1, true);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, 0);
                return true;
            case 0x00: case 0x01: case 0x02: { // ADD.D / SUB.D / MUL.D
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd); // fs
                a.str64(0, SP, kFpScratchSlot);
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rt); // ft
                a.fmov_to_fp(1, 0, true);
                a.ldr64(0, SP, kFpScratchSlot);
                a.fmov_to_fp(0, 0, true);
                if (d.funct == 0x00) a.fadd(0, 0, 1, true);
                else if (d.funct == 0x01) a.fsub(0, 0, 1, true);
                else a.fmul(0, 0, 1, true);
                a.fmov_from_fp(0, 0, true);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, 0);
                return true;
            }
            case 0x03: { // DIV.D - same zero-divisor special case as DIV.S, 64-bit
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd); // fs
                a.str64(0, SP, kFpScratchSlot);
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rt); // ft (divisor)
                a.mov_reg(2, 0);
                a.mov_imm64(S3, 0x7FFFFFFFFFFFFFFFull);
                a.and_reg(2, 2, S3, true);
                size_t nz = a.cbnz(2, true);
                a.movz(0, 0, 0, false); // divisor is zero -> result is 0.0 (W0=0 zero-extends X0)
                size_t skip = a.b();
                a.patch_branch(nz, a.pos());
                a.fmov_to_fp(1, 0, true);
                a.ldr64(0, SP, kFpScratchSlot);
                a.fmov_to_fp(0, 0, true);
                a.fdiv(0, 0, 1, true);
                a.fmov_from_fp(0, 0, true);
                a.patch_branch(skip, a.pos());
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, 0);
                return true;
            }
            case 0x04: // SQRT.D
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                a.fmov_to_fp(0, 0, true);
                a.fsqrt(0, 0, true);
                a.fmov_from_fp(0, 0, true);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, 0);
                return true;
            case 0x20: // CVT.S.D: double -> float
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                a.fmov_to_fp(0, 0, true);
                a.fcvt_d_to_s(0, 0);
                a.fmov_from_fp(0, 0, false);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, 0);
                return true;
            case 0x24: // CVT.W.D: double -> 32-bit int (truncating)
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                a.fmov_to_fp(0, 0, true);
                a.fcvtzs(0, 0, false, true);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, 0);
                return true;
            case 0x25: // CVT.L.D: double -> 64-bit int (truncating)
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                a.fmov_to_fp(0, 0, true);
                a.fcvtzs(0, 0, true, true);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, 0);
                return true;
            default:
                if ((d.funct & 0x30) == 0x30) { // C.cond.D
                    a.mov_reg(0, X_CTX);
                    a.movz(1, d.rd, 0, false);
                    a.movz(2, d.rt, 0, false);
                    a.movz(3, d.funct & 0x07, 0, false);
                    a.mov_imm64(S3, reinterpret_cast<u64>(&jit_fpu_ccond_d));
                    a.blr(S3);
                    return true;
                }
                return false;
            }
        // SCVTF writes its result to an FP register (S0/D0), while emit_fpr_set
        // takes the bits from X0 - hence the FMOV back after every conversion.
        case 0x14: // .W (word) - fs holds a raw int32 stored in the FPR, not float bits
            switch (d.funct) {
            case 0x20: // CVT.S.W
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                a.scvtf(0, 0, false, false);
                a.fmov_from_fp(0, 0, false);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, 0);
                return true;
            case 0x21: // CVT.D.W
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get32), d.rd);
                a.sxtw(0, 0); // sign-extend the int32 before treating it as a 64-bit source width
                a.scvtf(0, 0, true, true);
                a.fmov_from_fp(0, 0, true);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, 0);
                return true;
            default: return false;
            }
        case 0x15: // .L (long) - fs holds a raw int64 stored in the FPR
            switch (d.funct) {
            case 0x20: // CVT.S.L
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                a.scvtf(0, 0, true, false);
                a.fmov_from_fp(0, 0, false);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set32), d.shamt, 0);
                return true;
            case 0x21: // CVT.D.L
                emit_fpr_get(a, reinterpret_cast<void*>(&jit_fpr_get64), d.rd);
                a.scvtf(0, 0, true, true);
                a.fmov_from_fp(0, 0, true);
                emit_fpr_set(a, reinterpret_cast<void*>(&jit_fpr_set64), d.shamt, 0);
                return true;
            default: return false;
            }
        default: return false; // CFC1/CTC1/BC1 - interpreter handles it
        }

    case 0x20: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lb), site, fault_patches); return true;
    case 0x21: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lh), site, fault_patches); return true;
    case 0x23: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lw), site, fault_patches); return true;
    case 0x24: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lbu), site, fault_patches); return true;
    case 0x25: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lhu), site, fault_patches); return true;
    case 0x27: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lwu), site, fault_patches); return true;
    case 0x37: emit_load(a, d, reinterpret_cast<void*>(&jit_load_ld), site, fault_patches); return true;

    case 0x28: emit_store(a, d, reinterpret_cast<void*>(&jit_store_sb), site, fault_patches); return true;
    case 0x29: emit_store(a, d, reinterpret_cast<void*>(&jit_store_sh), site, fault_patches); return true;
    case 0x2B: emit_store(a, d, reinterpret_cast<void*>(&jit_store_sw), site, fault_patches); return true;
    case 0x3F: emit_store(a, d, reinterpret_cast<void*>(&jit_store_sd), site, fault_patches); return true;

    case 0x2F: return true; // CACHE: a no-op, as in the interpreter

    default:
        return false;
    }
}

// Runs `d` through jit_interp (see is_interp_callable), with the same
// fault-exit convention as a load/store helper call.
void emit_interp_call(Assembler& a, const Decoded& d, u32 site, std::vector<size_t>& fault_patches) {
    a.mov_reg(0, X_CTX);
    a.mov_imm64(1, d.raw);
    a.mov_imm64(2, site);
    a.mov_imm64(S3, reinterpret_cast<u64>(&jit_interp));
    a.blr(S3);
    size_t skip = a.cbz(0, false);
    a.movz(0, static_cast<u16>(site_index(site)), 0, false);
    fault_patches.push_back(a.b());
    a.patch_branch(skip, a.pos());
}

} // namespace

// Branches are not compiled on this backend yet (no emit_branch()): a block
// ends right before one and the interpreter runs it.
static Recompiler::Block compile_block_a64(Bus& bus, u32 start_paddr, CodeBuffer& code, bool delay_slot) {
    Recompiler::Block blk{};
    size_t rdram_size = bus.get_rdram_size();
    const u8* rdram = bus.get_rdram();
    const u32 max_len = delay_slot ? 1 : kMaxBlockLen;

    Assembler a;
    emit_prologue(a);
    std::vector<size_t> fault_patches;
    std::vector<OverflowSite> overflow_sites;

    u32 len = 0;
    while (len < max_len) {
        u32 paddr = start_paddr + len * 4;
        if (static_cast<size_t>(paddr) + 4 > rdram_size) break;
        Decoded d = decode(fetch_instr(rdram, paddr));
        const u32 site = delay_slot ? (len | kSiteDelaySlot) : len;
        if (!compile_one(a, d, site, fault_patches, overflow_sites)) {
            if (!is_interp_callable(d)) break;
            emit_interp_call(a, d, site, fault_patches);
        }
        len++;
    }
    if (len == 0) return blk;

    a.movz(0, static_cast<u16>(len), 0, false);
    size_t epilogue_pos = a.pos();
    emit_epilogue(a);
    for (size_t site : fault_patches) a.patch_branch(site, epilogue_pos);
    // Out-of-line stubs for trapping arithmetic that overflowed - see the
    // x64 backend's identical comment above compile_block_x64's equivalent.
    for (const OverflowSite& s : overflow_sites) {
        a.patch_branch(s.patch, a.pos());
        a.mov_reg(0, X_CTX);
        a.mov_imm64(1, s.site);
        a.mov_imm64(S3, reinterpret_cast<u64>(&jit_overflow));
        a.blr(S3);
        a.movz(0, static_cast<u16>(site_index(s.site)), 0, false);
        a.patch_branch(a.b(), epilogue_pos);
    }

    // See compile_block_x64: lookup_or_compile() keeps kArenaHeadroom free.
    if (code.remaining() < a.size()) return blk; // left to the interpreter
    code.make_writable(a.size());
    u8* dst = code.write_ptr();
    std::memcpy(dst, a.data(), a.size());
    code.commit(a.size());
    code.make_executable();

    blk.fn = reinterpret_cast<JitBlockFn>(code.exec_ptr(dst));
    blk.length = len;
    return blk;
}
#endif // ORBIT64_JIT_A64

Recompiler::Block Recompiler::compile_block(CPU&, Bus& bus, u32 start_paddr, bool delay_slot) {
#if defined(ORBIT64_JIT_X64)
    return compile_block_x64(bus, start_paddr, code_, delay_slot);
#elif defined(ORBIT64_JIT_A64)
    return compile_block_a64(bus, start_paddr, code_, delay_slot);
#else
    (void)bus; (void)start_paddr; (void)delay_slot;
    return Block{}; // unsupported host architecture: interpreter-only
#endif
}
