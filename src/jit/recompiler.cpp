#include "recompiler.hpp"
#include "jit_helpers.hpp"
#include "jit_invalidate.hpp"
#include "../cpu.hpp"
#include "../bus.hpp"
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

// A pending "trapping arithmetic overflowed" branch site: `site` is where the
// native overflow-branch instruction was emitted (still pointing nowhere -
// patched once the out-of-line stub for it is emitted, same idea as the
// load/store fault_patches list), `n` is the guest instruction count to
// report if it fires.
struct OverflowSite {
    size_t site;
    u32 n;
};

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

Recompiler::Recompiler() : cache_(kCacheSize), code_pages_(RDRAM_SIZE >> kPageShift, false), ds_cache_(kCacheSize) {
    jit::set_invalidate_hook(this, [](void* owner, u32 paddr, u32 len) {
        static_cast<Recompiler*>(owner)->request_invalidate(paddr, len);
    });
}

Recompiler::~Recompiler() {
    jit::clear_invalidate_hook_if(this);
}

void Recompiler::invalidate_all() {
    for (CacheEntry& e : cache_) e.tag = 0xFFFFFFFFu;
    for (CacheEntry& e : ds_cache_) e.tag = 0xFFFFFFFFu;
    code_.reset();
    std::fill(code_pages_.begin(), code_pages_.end(), false);
    last_paddr_ = 0xFFFFFFFFu;
    last_block_ = nullptr;
    last_ds_paddr_ = 0xFFFFFFFFu;
    last_ds_block_ = nullptr;
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
        code_pages_[p] = true;
    }
}

u32 Recompiler::run_step(CPU& cpu, Bus& bus) {
    // Always reached from plain C++ (never from inside a compiled block), so
    // it's the one safe place to actually drop the cache - see
    // request_invalidate() for why this can't happen synchronously.
    if (pending_invalidate_) {
        pending_invalidate_ = false;
        invalidate_all();
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
        return cpu.step();
    }
    u32 paddr = pc32 & 0x1FFFFFFFu;

    // A taken branch's delay slot: pc after this one instruction is fixed
    // (jit_branch_target(), not pc+4), so it gets its own capped-at-1-
    // instruction compile instead of the normal multi-instruction one - see
    // lookup_or_compile_delay_slot()/ds_cache_.
    if (cpu.jit_pending_delay_slot()) {
        const Block* dblk = lookup_or_compile_delay_slot(cpu, bus, paddr);
        if (!dblk || !dblk->fn) return cpu.step();

        u64 target = cpu.jit_branch_target();
        JitCtx dctx{&cpu, &bus, cpu.jit_gpr_ptr(), cpu.jit_hi_ptr(), cpu.jit_lo_ptr()};
        u32 dexecuted = dblk->fn(&dctx);
        if (dexecuted == dblk->length) {
            cpu.set_pc(target);
            // CPU::step() always clears this at its own entry, before
            // possibly setting it again for a *new* branch; since this path
            // never goes through step(), it has to do that part explicitly
            // too, or the next instruction gets mistaken for another
            // pending delay slot forever (target never changes once stuck).
            cpu.jit_clear_pending_delay_slot();
        }
        // else: this one instruction was a load/store that raised a TLB
        // exception, which already cleared this flag as part of redirecting
        // cpu's pc to the vector (see CPU::trigger_tlb_exception) - exactly
        // like the normal block path below.
        if (dexecuted > 0) {
            cpu.step_timer(2 * dexecuted);
            cpu.check_interrupts();
        }
        return dexecuted * 2;
    }

    const Block* blk = lookup_or_compile(cpu, bus, pc, paddr);
    if (!blk || !blk->fn) {
        return cpu.step();
    }

    JitCtx ctx{&cpu, &bus, cpu.jit_gpr_ptr(), cpu.jit_hi_ptr(), cpu.jit_lo_ptr()};
    u32 executed = blk->fn(&ctx);

    if (executed == blk->length) {
        cpu.set_pc(pc + 4ULL * executed);
    }
    // else: a load/store inside the block raised a TLB exception, which
    // already redirected cpu's pc to the exception vector.

    // Batched COUNT/compare-timer update + a single interrupt check for the
    // whole block. This is coarser than the interpreter (which re-checks
    // after every instruction) but observationally equivalent here: compiled
    // blocks never contain a branch, so there's no mid-block point control
    // flow could have been redirected to. CP0 RANDOM (TLB-replacement index)
    // is deliberately *not* decremented per compiled instruction - only the
    // interpreter does that precisely - since whitelisted blocks never touch
    // the TLB themselves; documented simplification for this first version.
    if (executed > 0) {
        cpu.step_timer(2 * executed);
        cpu.check_interrupts();
    }

    return executed * 2;
}

const Recompiler::Block* Recompiler::lookup_or_compile(CPU& cpu, Bus& bus, u64, u32 paddr) {
    if (paddr == last_paddr_) return last_block_;

    CacheEntry& slot = cache_[cache_index(paddr)];
    if (slot.tag != paddr) {
        if (code_.remaining() < kArenaHeadroom) invalidate_all();
        slot.tag = paddr;
        slot.block = compile_block(cpu, bus, paddr, kMaxBlockLen);
        if (slot.block.fn) mark_code_pages(paddr, paddr + slot.block.length * 4);
    }
    last_paddr_ = paddr;
    last_block_ = &slot.block;
    return last_block_;
}

const Recompiler::Block* Recompiler::lookup_or_compile_delay_slot(CPU& cpu, Bus& bus, u32 paddr) {
    if (paddr == last_ds_paddr_) return last_ds_block_;

    CacheEntry& slot = ds_cache_[cache_index(paddr)];
    if (slot.tag != paddr) {
        if (code_.remaining() < kArenaHeadroom) invalidate_all();
        slot.tag = paddr;
        slot.block = compile_block(cpu, bus, paddr, 1);
        if (slot.block.fn) mark_code_pages(paddr, paddr + 4);
    }
    last_ds_paddr_ = paddr;
    last_ds_block_ = &slot.block;
    return last_ds_block_;
}

// ===========================================================================
// x86-64 backend
// ===========================================================================
#if defined(ORBIT64_JIT_X64)
namespace {
using namespace x64;

// Context registers, fixed for the lifetime of a compiled block. None of
// these ever holds anything else, so per-instruction codegen can assume them.
constexpr int R_GPR = RBX; // u64 gpr[32]
constexpr int R_BUS = R12; // Bus*
constexpr int R_HI = R13;  // u64* (single u64)
constexpr int R_LO = R14;  // u64* (single u64)
constexpr int R_CPU = R15; // CPU* (unused by codegen directly; kept for helpers via ctx)
constexpr int R_CTX = RBP; // JitCtx* (original argument, for helper calls)
// x64::RSP collides with the (unrelated) forward-declared N64 `class RSP` -
// alias it locally so this file can stay unqualified everywhere else.
constexpr int R_SP = x64::RSP;

// Extra stack space reserved beyond the 8 bytes needed to reach 16-byte
// alignment for CALLs (6 pushes then -8 lands on a 16-byte boundary; any
// further reservation must stay a multiple of 16 to preserve that). This
// gives COP1 codegen a fixed, always-valid scratch slot at [rsp+8] to stash
// one FP register's raw bits across a second helper call - XMM registers
// (and every general-purpose scratch register) are caller-saved, so nothing
// survives a call except what's explicitly parked on the stack or in one of
// the context registers.
constexpr u32 kFrameExtra = 24; // 8 (alignment) + 16 (one scratch slot, room to spare)
constexpr s32 kFpScratchSlot = 8;

void emit_prologue(Assembler& x) {
    x.push(RBP); x.push(RBX); x.push(R12); x.push(R13); x.push(R14); x.push(R15);
    x.alu_ri(true, /*SUB*/ 5, R_SP, kFrameExtra);
    x.mov_reg_reg64(R_CTX, RDI);
    x.load_mem64(R_CPU, RDI, 0);
    x.load_mem64(R_BUS, RDI, 8);
    x.load_mem64(R_GPR, RDI, 16);
    x.load_mem64(R_HI, RDI, 24);
    x.load_mem64(R_LO, RDI, 32);
}

void emit_epilogue(Assembler& x) {
    x.alu_ri(true, /*ADD*/ 0, R_SP, kFrameExtra);
    x.pop(R15); x.pop(R14); x.pop(R13); x.pop(R12); x.pop(RBX); x.pop(RBP);
    x.ret();
}

// Calls a jit_helpers thunk of the form `u32 fn(JitCtx*, u64 addr, u32 arg)`.
// `addr` must already be in RSI and `arg` in RDX. On return, if RAX != 0 the
// block aborts (a TLB fault redirected cpu->pc already); this bakes in the
// static instruction index `n` as the returned "instructions executed" count.
void emit_helper_call(Assembler& x, void* fn, u32 n, std::vector<size_t>& fault_patches) {
    x.mov_reg_reg64(RDI, R_CTX);
    x.mov_reg_imm64(RAX, reinterpret_cast<u64>(fn));
    x.call_reg(RAX);
    x.test_rr(false, RAX, RAX);
    size_t skip = x.jcc_rel32(Cc::E); // RAX==0 -> success, skip the fault path
    x.mov_reg_imm32(RAX, n);
    fault_patches.push_back(x.jmp_rel32());
    x.patch_rel32(skip, x.pos());
}

// Records a pending overflow branch (the native ADD/SUB/etc. just emitted
// already set OF; `site` is the Jcc(O) placeholder to resolve later).
void emit_overflow_check(Assembler& x, u32 n, std::vector<OverflowSite>& overflow_sites) {
    overflow_sites.push_back({x.jcc_rel32(Cc::O), n});
}

// ---- COP1 helpers: fetch/store an FPU register's raw bits via jit_fpr_*.
// These never fault (no TLB, no exceptions), so unlike loads/stores there's
// nothing to check on return - the result is just wherever the ABI puts it.
void emit_fpr_get(Assembler& x, void* fn, u8 reg) {
    x.mov_reg_reg64(RDI, R_CTX);
    x.mov_reg_imm32(RSI, reg);
    x.mov_reg_imm64(RAX, reinterpret_cast<u64>(fn));
    x.call_reg(RAX); // result in EAX/RAX
}
// `value_reg` holds the bits to store (RAX/EAX by convention here, matching
// the register emit_fpr_get() just left them in).
void emit_fpr_set(Assembler& x, void* fn, u8 reg, int value_reg) {
    x.mov_reg_reg64(RDX, value_reg);
    x.mov_reg_imm32(RSI, reg);
    x.mov_reg_reg64(RDI, R_CTX);
    x.mov_reg_imm64(RAX, reinterpret_cast<u64>(fn));
    x.call_reg(RAX);
}

void emit_load(Assembler& x, const Decoded& d, void* fn, u32 n, std::vector<size_t>& fault_patches) {
    x.load_mem64(RSI, R_GPR, d.rs * 8);
    x.alu_ri(true, /*ADD*/ 0, RSI, static_cast<u32>(d.simm));
    x.mov_reg_imm32(RDX, d.rt);
    emit_helper_call(x, fn, n, fault_patches);
}

void emit_store(Assembler& x, const Decoded& d, void* fn, u32 n, std::vector<size_t>& fault_patches) {
    x.load_mem64(RSI, R_GPR, d.rs * 8);
    x.alu_ri(true, /*ADD*/ 0, RSI, static_cast<u32>(d.simm));
    x.load_mem64(RDX, R_GPR, d.rt * 8);
    emit_helper_call(x, fn, n, fault_patches);
}

// Returns false if `d` isn't in the native whitelist (caller stops the block
// before this instruction; it is not included in the compiled block).
bool compile_one(Assembler& x, const Decoded& d, u32 n, std::vector<size_t>& fault_patches,
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
            emit_overflow_check(x, n, overflow_sites);
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
            emit_overflow_check(x, n, overflow_sites);
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
            emit_overflow_check(x, n, overflow_sites);
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
            emit_overflow_check(x, n, overflow_sites);
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
        emit_overflow_check(x, n, overflow_sites);
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
        emit_overflow_check(x, n, overflow_sites);
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
                    x.mov_reg_reg64(RDI, R_CTX);
                    x.mov_reg_imm32(RSI, d.rd);
                    x.mov_reg_imm32(RDX, d.rt);
                    x.mov_reg_imm32(RCX, d.funct & 0x07);
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
                    x.mov_reg_reg64(RDI, R_CTX);
                    x.mov_reg_imm32(RSI, d.rd);
                    x.mov_reg_imm32(RDX, d.rt);
                    x.mov_reg_imm32(RCX, d.funct & 0x07);
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

    case 0x20: emit_load(x, d, reinterpret_cast<void*>(&jit_load_lb), n, fault_patches); return true;
    case 0x21: emit_load(x, d, reinterpret_cast<void*>(&jit_load_lh), n, fault_patches); return true;
    case 0x23: emit_load(x, d, reinterpret_cast<void*>(&jit_load_lw), n, fault_patches); return true;
    case 0x24: emit_load(x, d, reinterpret_cast<void*>(&jit_load_lbu), n, fault_patches); return true;
    case 0x25: emit_load(x, d, reinterpret_cast<void*>(&jit_load_lhu), n, fault_patches); return true;
    case 0x27: emit_load(x, d, reinterpret_cast<void*>(&jit_load_lwu), n, fault_patches); return true;
    case 0x37: emit_load(x, d, reinterpret_cast<void*>(&jit_load_ld), n, fault_patches); return true;

    case 0x28: emit_store(x, d, reinterpret_cast<void*>(&jit_store_sb), n, fault_patches); return true;
    case 0x29: emit_store(x, d, reinterpret_cast<void*>(&jit_store_sh), n, fault_patches); return true;
    case 0x2B: emit_store(x, d, reinterpret_cast<void*>(&jit_store_sw), n, fault_patches); return true;
    case 0x3F: emit_store(x, d, reinterpret_cast<void*>(&jit_store_sd), n, fault_patches); return true;

    default:
        return false; // branches/jumps/COP0/COP1/TLB/unaligned loads/etc.
    }
}

} // namespace

static Recompiler::Block compile_block_x64(Bus& bus, u32 start_paddr, CodeBuffer& code, u32 max_len) {
    Recompiler::Block blk{};
    size_t rdram_size = bus.get_rdram_size();
    const u8* rdram = bus.get_rdram();

    Assembler x;
    emit_prologue(x);
    std::vector<size_t> fault_patches;
    std::vector<OverflowSite> overflow_sites;

    u32 len = 0;
    while (len < max_len) {
        u32 paddr = start_paddr + len * 4;
        if (static_cast<size_t>(paddr) + 4 > rdram_size) break;
        u32 raw = (static_cast<u32>(rdram[paddr]) << 24) | (static_cast<u32>(rdram[paddr + 1]) << 16) |
                  (static_cast<u32>(rdram[paddr + 2]) << 8) | static_cast<u32>(rdram[paddr + 3]);
        Decoded d = decode(raw);
        if (!compile_one(x, d, len, fault_patches, overflow_sites)) break;
        len++;
    }
    if (len == 0) return blk;

    x.mov_reg_imm32(RAX, len);
    size_t epilogue_pos = x.pos();
    emit_epilogue(x);
    for (size_t site : fault_patches) x.patch_rel32(site, epilogue_pos);
    // Out-of-line stubs for trapping arithmetic that overflowed: raise EXC_OV
    // via the interpreter's own trigger_exception() (through jit_overflow),
    // report `n` guest instructions executed, then fall into the same
    // epilogue as everything else.
    for (const OverflowSite& s : overflow_sites) {
        x.patch_rel32(s.site, x.pos());
        x.mov_reg_reg64(RDI, R_CTX);
        x.mov_reg_imm64(RAX, reinterpret_cast<u64>(&jit_overflow));
        x.call_reg(RAX);
        x.mov_reg_imm32(RAX, s.n);
        x.patch_rel32(x.jmp_rel32(), epilogue_pos);
    }

    // lookup_or_compile() starts the arena over (dropping every cached block)
    // long before it gets this full, so this only triggers if a block ever
    // outgrows kArenaHeadroom. Resetting the arena here instead would leave
    // the block caches pointing into code about to be overwritten.
    if (code.remaining() < x.size()) return blk; // left to the interpreter
    code.make_writable();
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

// Calls a jit_helpers thunk `u32 fn(JitCtx*, u64 addr, u32 arg)` with X1=addr,
// X2=arg already set. On return, W0!=0 means a TLB fault redirected cpu->pc;
// the block then reports `n` guest instructions executed and returns.
void emit_helper_call(Assembler& a, void* fn, u32 n, std::vector<size_t>& fault_patches) {
    a.mov_reg(0, X_CTX);
    a.mov_imm64(S3, reinterpret_cast<u64>(fn));
    a.blr(S3);
    size_t skip = a.cbz(0, false); // W0==0 -> success, skip the fault path
    a.movz(0, static_cast<u16>(n), 0, false);
    fault_patches.push_back(a.b());
    a.patch_branch(skip, a.pos());
}

void emit_load(Assembler& a, const Decoded& d, void* fn, u32 n, std::vector<size_t>& fault_patches) {
    a.ldr64(1, X_GPR, d.rs * 8);
    a.mov_imm64_sext32(S3, d.simm);
    a.add_reg(1, 1, S3, true);
    a.movz(2, d.rt, 0, false);
    emit_helper_call(a, fn, n, fault_patches);
}

void emit_store(Assembler& a, const Decoded& d, void* fn, u32 n, std::vector<size_t>& fault_patches) {
    a.ldr64(1, X_GPR, d.rs * 8);
    a.mov_imm64_sext32(S3, d.simm);
    a.add_reg(1, 1, S3, true);
    a.ldr64(2, X_GPR, d.rt * 8); // value to store
    emit_helper_call(a, fn, n, fault_patches);
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
// already set the V flag; `site` is the B.VS placeholder to resolve later).
void emit_overflow_check(Assembler& a, u32 n, std::vector<OverflowSite>& overflow_sites) {
    overflow_sites.push_back({a.bcond(Cond::VS), n});
}

bool compile_one(Assembler& a, const Decoded& d, u32 n, std::vector<size_t>& fault_patches,
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
            emit_overflow_check(a, n, overflow_sites);
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
            emit_overflow_check(a, n, overflow_sites);
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
            emit_overflow_check(a, n, overflow_sites);
            st_if(d.rd, S0);
            return true;
        }
        case 0x2D: // DADDU
            if (d.rd != 0) { a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8); a.add_reg(S0, S0, S1, true); st_if(d.rd, S0); }
            return true;
        case 0x2E: { // DSUB (traps on 64-bit signed overflow)
            a.ldr64(S0, X_GPR, d.rs * 8); a.ldr64(S1, X_GPR, d.rt * 8);
            a.subs_reg(S0, S0, S1, true);
            emit_overflow_check(a, n, overflow_sites);
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
        emit_overflow_check(a, n, overflow_sites);
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
        emit_overflow_check(a, n, overflow_sites);
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

    case 0x20: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lb), n, fault_patches); return true;
    case 0x21: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lh), n, fault_patches); return true;
    case 0x23: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lw), n, fault_patches); return true;
    case 0x24: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lbu), n, fault_patches); return true;
    case 0x25: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lhu), n, fault_patches); return true;
    case 0x27: emit_load(a, d, reinterpret_cast<void*>(&jit_load_lwu), n, fault_patches); return true;
    case 0x37: emit_load(a, d, reinterpret_cast<void*>(&jit_load_ld), n, fault_patches); return true;

    case 0x28: emit_store(a, d, reinterpret_cast<void*>(&jit_store_sb), n, fault_patches); return true;
    case 0x29: emit_store(a, d, reinterpret_cast<void*>(&jit_store_sh), n, fault_patches); return true;
    case 0x2B: emit_store(a, d, reinterpret_cast<void*>(&jit_store_sw), n, fault_patches); return true;
    case 0x3F: emit_store(a, d, reinterpret_cast<void*>(&jit_store_sd), n, fault_patches); return true;

    default:
        return false;
    }
}

} // namespace

static Recompiler::Block compile_block_a64(Bus& bus, u32 start_paddr, CodeBuffer& code, u32 max_len) {
    Recompiler::Block blk{};
    size_t rdram_size = bus.get_rdram_size();
    const u8* rdram = bus.get_rdram();

    Assembler a;
    emit_prologue(a);
    std::vector<size_t> fault_patches;
    std::vector<OverflowSite> overflow_sites;

    u32 len = 0;
    while (len < max_len) {
        u32 paddr = start_paddr + len * 4;
        if (static_cast<size_t>(paddr) + 4 > rdram_size) break;
        u32 raw = (static_cast<u32>(rdram[paddr]) << 24) | (static_cast<u32>(rdram[paddr + 1]) << 16) |
                  (static_cast<u32>(rdram[paddr + 2]) << 8) | static_cast<u32>(rdram[paddr + 3]);
        Decoded d = decode(raw);
        if (!compile_one(a, d, len, fault_patches, overflow_sites)) break;
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
        a.patch_branch(s.site, a.pos());
        a.mov_reg(0, X_CTX);
        a.mov_imm64(S3, reinterpret_cast<u64>(&jit_overflow));
        a.blr(S3);
        a.movz(0, static_cast<u16>(s.n), 0, false);
        a.patch_branch(a.b(), epilogue_pos);
    }

    // See compile_block_x64: lookup_or_compile() keeps kArenaHeadroom free.
    if (code.remaining() < a.size()) return blk; // left to the interpreter
    code.make_writable();
    u8* dst = code.write_ptr();
    std::memcpy(dst, a.data(), a.size());
    code.commit(a.size());
    code.make_executable();

    blk.fn = reinterpret_cast<JitBlockFn>(code.exec_ptr(dst));
    blk.length = len;
    return blk;
}
#endif // ORBIT64_JIT_A64

Recompiler::Block Recompiler::compile_block(CPU&, Bus& bus, u32 start_paddr, u32 max_len) {
#if defined(ORBIT64_JIT_X64)
    return compile_block_x64(bus, start_paddr, code_, max_len);
#elif defined(ORBIT64_JIT_A64)
    return compile_block_a64(bus, start_paddr, code_, max_len);
#else
    (void)bus; (void)start_paddr; (void)max_len;
    return Block{}; // unsupported host architecture: interpreter-only
#endif
}
