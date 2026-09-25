#include "jit_helpers.hpp"
#include "../cpu.hpp"
#include "../bus.hpp"
#include "jit_invalidate.hpp"

// rt == 0 is handled by CPU::set_gpr() itself (it no-ops writes to $zero),
// matching the interpreter exactly.

namespace {

// Points the CPU's exception bookkeeping (EPC/BD source) at the guest
// instruction `site` names - compiled code never updates it per instruction
// the way CPU::step() does - and flags the block as aborted.
void enter_exception_site(JitCtx* ctx, u32 site) {
    const u32 index = site & ~kSiteDelaySlot;
    const bool delay_slot = (site & kSiteDelaySlot) != 0 && ctx->branch_taken != 0;
    ctx->cpu->jit_set_exception_site(ctx->start_pc + 4ULL * index, delay_slot);
    ctx->faulted = 1;
}

u32 raise_tlb(JitCtx* ctx, u32 site, u64 addr, TLBResult res, bool is_write) {
    enter_exception_site(ctx, site);
    return ctx->cpu->jit_raise_tlb_exception(addr, res, is_write);
}

} // namespace

extern "C" u32 jit_load_lb(JitCtx* ctx, u64 addr, u32 rt, u32 site) {
    TLBResult res = TLBResult::SUCCESS;
    u8 byte = ctx->bus->read_v8(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return raise_tlb(ctx, site, addr, res, false);
    ctx->cpu->set_gpr(rt, static_cast<u64>(sign_extend_8_64(static_cast<s8>(byte))));
    return 0;
}

extern "C" u32 jit_load_lbu(JitCtx* ctx, u64 addr, u32 rt, u32 site) {
    TLBResult res = TLBResult::SUCCESS;
    u8 byte = ctx->bus->read_v8(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return raise_tlb(ctx, site, addr, res, false);
    ctx->cpu->set_gpr(rt, byte);
    return 0;
}

extern "C" u32 jit_load_lh(JitCtx* ctx, u64 addr, u32 rt, u32 site) {
    TLBResult res = TLBResult::SUCCESS;
    u16 hword = ctx->bus->read_v16(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return raise_tlb(ctx, site, addr, res, false);
    ctx->cpu->set_gpr(rt, static_cast<u64>(sign_extend_16_64(static_cast<s16>(hword))));
    return 0;
}

extern "C" u32 jit_load_lhu(JitCtx* ctx, u64 addr, u32 rt, u32 site) {
    TLBResult res = TLBResult::SUCCESS;
    u16 hword = ctx->bus->read_v16(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return raise_tlb(ctx, site, addr, res, false);
    ctx->cpu->set_gpr(rt, hword);
    return 0;
}

extern "C" u32 jit_load_lw(JitCtx* ctx, u64 addr, u32 rt, u32 site) {
    TLBResult res = TLBResult::SUCCESS;
    u32 word = ctx->bus->read_v32(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return raise_tlb(ctx, site, addr, res, false);
    ctx->cpu->set_gpr(rt, static_cast<u64>(sign_extend_32_64(static_cast<s32>(word))));
    return 0;
}

extern "C" u32 jit_load_lwu(JitCtx* ctx, u64 addr, u32 rt, u32 site) {
    TLBResult res = TLBResult::SUCCESS;
    u32 word = ctx->bus->read_v32(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return raise_tlb(ctx, site, addr, res, false);
    ctx->cpu->set_gpr(rt, word);
    return 0;
}

extern "C" u32 jit_load_ld(JitCtx* ctx, u64 addr, u32 rt, u32 site) {
    TLBResult res = TLBResult::SUCCESS;
    u64 dword = ctx->bus->read_v64(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return raise_tlb(ctx, site, addr, res, false);
    ctx->cpu->set_gpr(rt, dword);
    return 0;
}

namespace {

// Stores translate first (exactly what Bus::write_v* does) so they know
// whether they hit MMIO: a chain of compiled blocks is then ended right after
// the store, so that an interrupt it raised is taken as soon as it would be
// with one block per run (see Recompiler::run).
template <class T, void (Bus::*Write)(u32, T)>
u32 store_helper(JitCtx* ctx, u64 addr, T val, u32 site) {
    u32 paddr = 0;
    const TLBResult res = ctx->bus->translate_vaddr(addr, paddr, true, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return raise_tlb(ctx, site, addr, res, true);
    (ctx->bus->*Write)(paddr, val);
    if (paddr >= RDRAM_SIZE) ctx->exit_req = 1;
    return 0;
}

} // namespace

extern "C" u32 jit_store_sb(JitCtx* ctx, u64 addr, u32 val, u32 site) {
    return store_helper<u8, &Bus::write8>(ctx, addr, static_cast<u8>(val), site);
}

extern "C" u32 jit_store_sh(JitCtx* ctx, u64 addr, u32 val, u32 site) {
    return store_helper<u16, &Bus::write16>(ctx, addr, static_cast<u16>(val), site);
}

extern "C" u32 jit_store_sw(JitCtx* ctx, u64 addr, u32 val, u32 site) {
    return store_helper<u32, &Bus::write32>(ctx, addr, val, site);
}

extern "C" u32 jit_store_sd(JitCtx* ctx, u64 addr, u64 val, u32 site) {
    return store_helper<u64, &Bus::write64>(ctx, addr, val, site);
}

extern "C" void jit_overflow(JitCtx* ctx, u32 site) {
    enter_exception_site(ctx, site);
    ctx->cpu->jit_raise_overflow_exception();
}

extern "C" void jit_notify_code_write(u32 paddr, u32 len) { jit::notify_code_write(paddr, len); }

extern "C" u32 jit_interp(JitCtx* ctx, u32 instr, u32 site) {
    const u32 index = site & ~kSiteDelaySlot;
    const bool delay_slot = (site & kSiteDelaySlot) != 0 && ctx->branch_taken != 0;
    // A store (SWL/SWR/SDL/SDR/SC/SCD/SWC1/SDC1) to MMIO ends a block chain,
    // like the store helpers above; the address is worked out before the
    // instruction runs (SC overwrites rt, which may be rs).
    const u32 op = instr >> 26;
    bool mmio = false;
    if (op >= 0x28 && op != 0x2F) {
        const u64 addr = ctx->cpu->get_gpr((instr >> 21) & 31) + static_cast<u64>(static_cast<s64>(static_cast<s16>(instr & 0xFFFF)));
        u32 paddr = 0;
        mmio = ctx->bus->translate_vaddr(addr, paddr, true, ctx->cpu->jit_asid()) == TLBResult::SUCCESS && paddr >= RDRAM_SIZE;
    }
    if (!ctx->cpu->jit_execute(instr, ctx->start_pc + 4ULL * index, delay_slot)) {
        if (mmio) ctx->exit_req = 1;
        return 0;
    }
    ctx->faulted = 1;
    return 1;
}

extern "C" u32 jit_fpr_get32(JitCtx* ctx, u32 reg) { return ctx->cpu->jit_get_fpr32(reg); }
extern "C" void jit_fpr_set32(JitCtx* ctx, u32 reg, u32 bits) { ctx->cpu->jit_set_fpr32(reg, bits); }
extern "C" u64 jit_fpr_get64(JitCtx* ctx, u32 reg) { return ctx->cpu->jit_get_fpr64(reg); }
extern "C" void jit_fpr_set64(JitCtx* ctx, u32 reg, u64 bits) { ctx->cpu->jit_set_fpr64(reg, bits); }

extern "C" void jit_fpu_ccond_s(JitCtx* ctx, u32 fs, u32 ft, u32 cond) { ctx->cpu->jit_fpu_compare_s(fs, ft, cond); }
extern "C" void jit_fpu_ccond_d(JitCtx* ctx, u32 fs, u32 ft, u32 cond) { ctx->cpu->jit_fpu_compare_d(fs, ft, cond); }

namespace {
// Applies the chain's cycles up to (not including) the instruction at
// `remaining` to COUNT and RANDOM, then this instruction's own RANDOM step.
void sync_timers(JitCtx* ctx, s64 remaining) {
    const s64 before = ctx->cycles - remaining;
    const s64 delta = before - ctx->synced;
    ctx->cpu->jit_advance_random(static_cast<u64>(delta) / 2 + 1);
    ctx->cpu->step_timer(static_cast<u32>(delta));
    ctx->synced = before + 2; // including this instruction, whose COUNT step follows
}
} // namespace

extern "C" u32 jit_cop0(JitCtx* ctx, u32 instr, s64 remaining) {
    sync_timers(ctx, remaining);
    const u64 status = ctx->cpu->get_cp0(CP0Reg::STATUS);
    const u8 asid = ctx->cpu->jit_asid();
    const u32 tlb_gen = ctx->bus->tlb_generation();
    ctx->cpu->jit_execute(instr, 0, false); // none of these raise exceptions
    ctx->cpu->step_timer(2);
    const u32 rs = (instr >> 21) & 31, rd = (instr >> 11) & 31;
    if ((rs == 0x04 || rs == 0x05) && (rd == CP0Reg::COUNT || rd == CP0Reg::COMPARE)) return 1; // the timer moved
    // STATUS.FR changed: the rest of this block was compiled for the old mode.
    if ((status ^ ctx->cpu->get_cp0(CP0Reg::STATUS)) & (1u << 26)) return 1;
    // TLB-mapped code the chain might jump to is found through translations
    // the driver cached for the old TLB contents / ASID.
    if (ctx->cpu->jit_asid() != asid || ctx->bus->tlb_generation() != tlb_gen) return 1;
    return ctx->cpu->jit_interrupt_deliverable() ? 1 : 0;
}

extern "C" u32 jit_eret(JitCtx* ctx, s64 remaining) {
    sync_timers(ctx, remaining);
    ctx->cpu->jit_execute(0x42000018u, 0, false);
    ctx->cpu->step_timer(2);
    ctx->next_pc = ctx->cpu->get_pc();
    return ctx->cpu->jit_interrupt_deliverable() ? 1 : 0;
}

extern "C" void jit_fr_mismatch(JitCtx* ctx) { ctx->fr_mismatch = 1; }
