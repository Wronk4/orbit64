#include "jit_helpers.hpp"
#include "../cpu.hpp"
#include "../bus.hpp"

// rt == 0 is handled by CPU::set_gpr() itself (it no-ops writes to $zero),
// matching the interpreter exactly.

extern "C" u32 jit_load_lb(JitCtx* ctx, u64 addr, u32 rt) {
    TLBResult res = TLBResult::SUCCESS;
    u8 byte = ctx->bus->read_v8(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return ctx->cpu->jit_raise_tlb_exception(addr, res, false);
    ctx->cpu->set_gpr(rt, static_cast<u64>(sign_extend_8_64(static_cast<s8>(byte))));
    return 0;
}

extern "C" u32 jit_load_lbu(JitCtx* ctx, u64 addr, u32 rt) {
    TLBResult res = TLBResult::SUCCESS;
    u8 byte = ctx->bus->read_v8(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return ctx->cpu->jit_raise_tlb_exception(addr, res, false);
    ctx->cpu->set_gpr(rt, byte);
    return 0;
}

extern "C" u32 jit_load_lh(JitCtx* ctx, u64 addr, u32 rt) {
    TLBResult res = TLBResult::SUCCESS;
    u16 hword = ctx->bus->read_v16(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return ctx->cpu->jit_raise_tlb_exception(addr, res, false);
    ctx->cpu->set_gpr(rt, static_cast<u64>(sign_extend_16_64(static_cast<s16>(hword))));
    return 0;
}

extern "C" u32 jit_load_lhu(JitCtx* ctx, u64 addr, u32 rt) {
    TLBResult res = TLBResult::SUCCESS;
    u16 hword = ctx->bus->read_v16(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return ctx->cpu->jit_raise_tlb_exception(addr, res, false);
    ctx->cpu->set_gpr(rt, hword);
    return 0;
}

extern "C" u32 jit_load_lw(JitCtx* ctx, u64 addr, u32 rt) {
    TLBResult res = TLBResult::SUCCESS;
    u32 word = ctx->bus->read_v32(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return ctx->cpu->jit_raise_tlb_exception(addr, res, false);
    ctx->cpu->set_gpr(rt, static_cast<u64>(sign_extend_32_64(static_cast<s32>(word))));
    return 0;
}

extern "C" u32 jit_load_lwu(JitCtx* ctx, u64 addr, u32 rt) {
    TLBResult res = TLBResult::SUCCESS;
    u32 word = ctx->bus->read_v32(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return ctx->cpu->jit_raise_tlb_exception(addr, res, false);
    ctx->cpu->set_gpr(rt, word);
    return 0;
}

extern "C" u32 jit_load_ld(JitCtx* ctx, u64 addr, u32 rt) {
    TLBResult res = TLBResult::SUCCESS;
    u64 dword = ctx->bus->read_v64(addr, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return ctx->cpu->jit_raise_tlb_exception(addr, res, false);
    ctx->cpu->set_gpr(rt, dword);
    return 0;
}

extern "C" u32 jit_store_sb(JitCtx* ctx, u64 addr, u32 val) {
    TLBResult res = TLBResult::SUCCESS;
    ctx->bus->write_v8(addr, static_cast<u8>(val), res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return ctx->cpu->jit_raise_tlb_exception(addr, res, true);
    return 0;
}

extern "C" u32 jit_store_sh(JitCtx* ctx, u64 addr, u32 val) {
    TLBResult res = TLBResult::SUCCESS;
    ctx->bus->write_v16(addr, static_cast<u16>(val), res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return ctx->cpu->jit_raise_tlb_exception(addr, res, true);
    return 0;
}

extern "C" u32 jit_store_sw(JitCtx* ctx, u64 addr, u32 val) {
    TLBResult res = TLBResult::SUCCESS;
    ctx->bus->write_v32(addr, val, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return ctx->cpu->jit_raise_tlb_exception(addr, res, true);
    return 0;
}

extern "C" u32 jit_store_sd(JitCtx* ctx, u64 addr, u64 val) {
    TLBResult res = TLBResult::SUCCESS;
    ctx->bus->write_v64(addr, val, res, ctx->cpu->jit_asid());
    if (res != TLBResult::SUCCESS) return ctx->cpu->jit_raise_tlb_exception(addr, res, true);
    return 0;
}

extern "C" void jit_overflow(JitCtx* ctx) {
    ctx->cpu->jit_raise_overflow_exception();
}

extern "C" u32 jit_fpr_get32(JitCtx* ctx, u32 reg) { return ctx->cpu->jit_get_fpr32(reg); }
extern "C" void jit_fpr_set32(JitCtx* ctx, u32 reg, u32 bits) { ctx->cpu->jit_set_fpr32(reg, bits); }
extern "C" u64 jit_fpr_get64(JitCtx* ctx, u32 reg) { return ctx->cpu->jit_get_fpr64(reg); }
extern "C" void jit_fpr_set64(JitCtx* ctx, u32 reg, u64 bits) { ctx->cpu->jit_set_fpr64(reg, bits); }

extern "C" void jit_fpu_ccond_s(JitCtx* ctx, u32 fs, u32 ft, u32 cond) { ctx->cpu->jit_fpu_compare_s(fs, ft, cond); }
extern "C" void jit_fpu_ccond_d(JitCtx* ctx, u32 fs, u32 ft, u32 cond) { ctx->cpu->jit_fpu_compare_d(fs, ft, cond); }
