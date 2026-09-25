#pragma once
// Minimal hand-rolled x86-64 (SysV) instruction encoder. Only the handful of
// forms the recompiler actually needs - just enough to inline pure ALU /
// shift / immediate arithmetic on the guest register file and to call out to
// the jit_helpers thunks for memory access.

#include "../common.hpp"
#include <vector>
#include <cstring>

namespace x64 {

// Standard x86-64 register encoding (0..15 = RAX,RCX,RDX,RBX,RSP,RBP,RSI,RDI,R8..R15).
enum Reg : int {
    RAX = 0, RCX = 1, RDX = 2, RBX = 3, RSP = 4, RBP = 5, RSI = 6, RDI = 7,
    R8 = 8, R9 = 9, R10 = 10, R11 = 11, R12 = 12, R13 = 13, R14 = 14, R15 = 15
};

enum class Cc : u8 { O = 0x0, NO = 0x1, B = 0x2, AE = 0x3, E = 0x4, NE = 0x5, BE = 0x6, A = 0x7,
                      L = 0xC, GE = 0xD, LE = 0xE, G = 0xF };

class Assembler {
public:
    // ---- raw byte helpers ----
    void db(u8 b) { buf_.push_back(b); }
    void d32(u32 v) { u8 t[4]; std::memcpy(t, &v, 4); buf_.insert(buf_.end(), t, t + 4); }
    void d64(u64 v) { u8 t[8]; std::memcpy(t, &v, 8); buf_.insert(buf_.end(), t, t + 8); }

    size_t pos() const { return buf_.size(); }
    const u8* data() const { return buf_.data(); }
    size_t size() const { return buf_.size(); }

    // ---- REX / ModRM ----
    // wide=REX.W, reg/rm are the two register operands (reg field vs r/m field).
    void rex(bool wide, int reg, int rm, bool force = false) {
        u8 r = (reg >> 3) & 1, b = (rm >> 3) & 1;
        u8 byte = 0x40 | (wide ? 8 : 0) | (r << 2) | b;
        if (byte != 0x40 || force) db(byte);
    }
    void modrm_reg(int reg, int rm) { db(0xC0 | ((reg & 7) << 3) | (rm & 7)); }
    // [rm + disp32] addressing. rm&7==100 (RSP or R12) is special-cased in
    // ModRM: that encoding always means "read a SIB byte next", regardless
    // of which of the two registers it is, so a plain rm field can never
    // address through either - a SIB byte encoding "no index, this base" is
    // required. Every other register (including RBP, which needs the
    // disp32 form here since disp=0/mod=00/rm=101 means RIP-relative) is a
    // plain ModRM.
    void modrm_mem(int reg, int base, s32 disp) {
        bool needs_sib = (base & 7) == 4; // RSP or R12
        if (disp == 0 && (base & 7) != RBP) {
            db(0x00 | ((reg & 7) << 3) | (needs_sib ? 4 : (base & 7)));
            if (needs_sib) db(0x20 | (base & 7)); // scale=0, index=100 (none), base
        } else {
            db(0x80 | ((reg & 7) << 3) | (needs_sib ? 4 : (base & 7)));
            if (needs_sib) db(0x20 | (base & 7));
            d32(static_cast<u32>(disp));
        }
    }

    // ---- moves ----
    void mov_reg_imm64(int dst, u64 imm) {
        rex(true, 0, dst);
        db(0xB8 | (dst & 7));
        d64(imm);
    }
    // 32-bit immediate move (zero-extends dst to 64 bits); used for small
    // compile-time constants (register indices, block lengths) where a full
    // 10-byte movabs would be wasteful.
    void mov_reg_imm32(int dst, u32 imm) {
        rex(false, 0, dst);
        db(0xB8 | (dst & 7));
        d32(imm);
    }
    // mov dst64, [base+disp]
    void load_mem64(int dst, int base, s32 disp) {
        rex(true, dst, base);
        db(0x8B);
        modrm_mem(dst, base, disp);
    }
    // mov dst32, [base+disp] (zero-extends dst to 64 bits)
    void load_mem32(int dst, int base, s32 disp) {
        rex(false, dst, base);
        db(0x8B);
        modrm_mem(dst, base, disp);
    }
    void load_mem16_zx(int dst, int base, s32 disp) {
        rex(false, dst, base);
        db(0x0F); db(0xB7);
        modrm_mem(dst, base, disp);
    }
    void load_mem8_zx(int dst, int base, s32 disp) {
        rex(false, dst, base);
        db(0x0F); db(0xB6);
        modrm_mem(dst, base, disp);
    }
    // mov [base+disp], src64
    void store_mem64(int base, s32 disp, int src) {
        rex(true, src, base);
        db(0x89);
        modrm_mem(src, base, disp);
    }
    void store_mem32(int base, s32 disp, int src) {
        rex(false, src, base);
        db(0x89);
        modrm_mem(src, base, disp);
    }
    void mov_reg_reg64(int dst, int src) {
        rex(true, src, dst);
        db(0x89);
        modrm_reg(src, dst);
    }
    void mov_reg_reg32(int dst, int src) {
        rex(false, src, dst);
        db(0x89);
        modrm_reg(src, dst);
    }
    // movsxd dst64, src32 (sign-extend low 32 bits of src into dst)
    void movsxd(int dst, int src) {
        rex(true, dst, src);
        db(0x63);
        modrm_reg(dst, src);
    }

    // ---- ALU reg,reg (dst = dst OP src); opc is the "r/m,reg" opcode (01 add,29 sub,21 and,09 or,31 xor,39 cmp) ----
    void alu_rr(bool wide, u8 opc, int dst, int src) {
        rex(wide, src, dst);
        db(opc);
        modrm_reg(src, dst);
    }
    void add_rr(bool wide, int dst, int src) { alu_rr(wide, 0x01, dst, src); }
    void sub_rr(bool wide, int dst, int src) { alu_rr(wide, 0x29, dst, src); }
    void and_rr(bool wide, int dst, int src) { alu_rr(wide, 0x21, dst, src); }
    void or_rr(bool wide, int dst, int src) { alu_rr(wide, 0x09, dst, src); }
    void xor_rr(bool wide, int dst, int src) { alu_rr(wide, 0x31, dst, src); }
    void cmp_rr(bool wide, int a, int b) { alu_rr(wide, 0x39, a, b); }
    void test_rr(bool wide, int a, int b) { alu_rr(wide, 0x85, a, b); }

    void not_r(bool wide, int dst) {
        rex(wide, 0, dst);
        db(0xF7);
        modrm_reg(2, dst); // /2 = NOT
    }
    void neg_r(bool wide, int dst) {
        rex(wide, 0, dst);
        db(0xF7);
        modrm_reg(3, dst); // /3 = NEG
    }

    // ---- ALU reg,imm32 (opcode 0x81 group; ext selects the operation) ----
    void alu_ri(bool wide, u8 ext, int dst, u32 imm) {
        rex(wide, 0, dst);
        db(0x81);
        modrm_reg(ext, dst);
        d32(imm);
    }
    void add_ri(bool wide, int dst, u32 imm) { alu_ri(wide, 0, dst, imm); }
    void and_ri(bool wide, int dst, u32 imm) { alu_ri(wide, 4, dst, imm); }
    void or_ri(bool wide, int dst, u32 imm) { alu_ri(wide, 1, dst, imm); }
    void xor_ri(bool wide, int dst, u32 imm) { alu_ri(wide, 6, dst, imm); }
    void cmp_ri(bool wide, int dst, u32 imm) { alu_ri(wide, 7, dst, imm); }

    // ---- shifts by immediate (C1 group) / by CL (D3 group) ----
    void shift_ri(bool wide, u8 ext, int dst, u8 imm) {
        rex(wide, 0, dst);
        db(0xC1);
        modrm_reg(ext, dst);
        db(imm);
    }
    void shl_ri(bool wide, int dst, u8 imm) { shift_ri(wide, 4, dst, imm); }
    void shr_ri(bool wide, int dst, u8 imm) { shift_ri(wide, 5, dst, imm); }
    void sar_ri(bool wide, int dst, u8 imm) { shift_ri(wide, 7, dst, imm); }

    // shift dst by CL
    void shift_cl(bool wide, u8 ext, int dst) {
        rex(wide, 0, dst);
        db(0xD3);
        modrm_reg(ext, dst);
    }
    void shl_cl(bool wide, int dst) { shift_cl(wide, 4, dst); }
    void shr_cl(bool wide, int dst) { shift_cl(wide, 5, dst); }
    void sar_cl(bool wide, int dst) { shift_cl(wide, 7, dst); }

    void setcc(Cc cc, int dst) {
        // SETcc r/m8; needs REX if dst is a "new" byte-register (r8-r15) or to
        // access sil/dil/bpl/spl - always emit REX for uniformity/safety.
        rex(false, 0, dst, true);
        db(0x0F); db(0x90 | static_cast<u8>(cc));
        modrm_reg(0, dst);
    }
    void movzx8(int dst, int src) {
        // Forced REX: without it, ModRM.rm in [4..7] means AH/CH/DH/BH, not
        // SPL/BPL/SIL/DIL - src here is always one of our scratch regs (which
        // include RSI/RDI), so this must never be silently omitted.
        rex(false, dst, src, true);
        db(0x0F); db(0xB6);
        modrm_reg(dst, src);
    }

    // ---- scalar SSE2: move raw bits GPR<->XMM, and single/double
    // add/sub/mul. `xmm`/`gpr` register numbers use the same 0-15 encoding;
    // REX.R/B extension bits apply identically to either register file, so
    // the existing rex()/modrm_reg() helpers work unchanged. ----
    void movd_to_xmm(int xmm, int gpr) { db(0x66); rex(false, xmm, gpr); db(0x0F); db(0x6E); modrm_reg(xmm, gpr); }
    void movd_from_xmm(int gpr, int xmm) { db(0x66); rex(false, xmm, gpr); db(0x0F); db(0x7E); modrm_reg(xmm, gpr); }
    void movq_to_xmm(int xmm, int gpr) { db(0x66); rex(true, xmm, gpr); db(0x0F); db(0x6E); modrm_reg(xmm, gpr); }
    void movq_from_xmm(int gpr, int xmm) { db(0x66); rex(true, xmm, gpr); db(0x0F); db(0x7E); modrm_reg(xmm, gpr); }
    void addss(int dst, int src) { db(0xF3); rex(false, dst, src); db(0x0F); db(0x58); modrm_reg(dst, src); }
    void subss(int dst, int src) { db(0xF3); rex(false, dst, src); db(0x0F); db(0x5C); modrm_reg(dst, src); }
    void mulss(int dst, int src) { db(0xF3); rex(false, dst, src); db(0x0F); db(0x59); modrm_reg(dst, src); }
    void addsd(int dst, int src) { db(0xF2); rex(false, dst, src); db(0x0F); db(0x58); modrm_reg(dst, src); }
    void subsd(int dst, int src) { db(0xF2); rex(false, dst, src); db(0x0F); db(0x5C); modrm_reg(dst, src); }
    void mulsd(int dst, int src) { db(0xF2); rex(false, dst, src); db(0x0F); db(0x59); modrm_reg(dst, src); }
    void divss(int dst, int src) { db(0xF3); rex(false, dst, src); db(0x0F); db(0x5E); modrm_reg(dst, src); }
    void divsd(int dst, int src) { db(0xF2); rex(false, dst, src); db(0x0F); db(0x5E); modrm_reg(dst, src); }
    void sqrtss(int dst, int src) { db(0xF3); rex(false, dst, src); db(0x0F); db(0x51); modrm_reg(dst, src); }
    void sqrtsd(int dst, int src) { db(0xF2); rex(false, dst, src); db(0x0F); db(0x51); modrm_reg(dst, src); }
    // integer(32/64,truncating)<->float/double, and float<->double conversions.
    void cvttss2si(int gpr, int xmm, bool wide) { db(0xF3); rex(wide, gpr, xmm); db(0x0F); db(0x2C); modrm_reg(gpr, xmm); }
    void cvttsd2si(int gpr, int xmm, bool wide) { db(0xF2); rex(wide, gpr, xmm); db(0x0F); db(0x2C); modrm_reg(gpr, xmm); }
    void cvtsi2ss(int xmm, int gpr, bool wide) { db(0xF3); rex(wide, xmm, gpr); db(0x0F); db(0x2A); modrm_reg(xmm, gpr); }
    void cvtsi2sd(int xmm, int gpr, bool wide) { db(0xF2); rex(wide, xmm, gpr); db(0x0F); db(0x2A); modrm_reg(xmm, gpr); }
    void cvtss2sd(int dst, int src) { db(0xF3); rex(false, dst, src); db(0x0F); db(0x5A); modrm_reg(dst, src); }
    void cvtsd2ss(int dst, int src) { db(0xF2); rex(false, dst, src); db(0x0F); db(0x5A); modrm_reg(dst, src); }
    // scalar compare (sets ZF/PF/CF the same way as integer CMP; PF=1 means unordered/NaN).
    void ucomiss(int a, int b) { rex(false, a, b); db(0x0F); db(0x2E); modrm_reg(a, b); }
    void ucomisd(int a, int b) { db(0x66); rex(false, a, b); db(0x0F); db(0x2E); modrm_reg(a, b); }

    void push(int r) { rex(false, 0, r); db(0x50 | (r & 7)); }
    void pop(int r) { rex(false, 0, r); db(0x58 | (r & 7)); }
    void ret() { db(0xC3); }
    void call_reg(int r) { rex(false, 0, r); db(0xFF); modrm_reg(2, r); }

    // ---- branches with rel32, backpatched later ----
    size_t jmp_rel32() {
        db(0xE9);
        size_t site = pos();
        d32(0);
        return site;
    }
    size_t jcc_rel32(Cc cc) {
        db(0x0F); db(0x80 | static_cast<u8>(cc));
        size_t site = pos();
        d32(0);
        return site;
    }
    // `site` is the offset returned by jmp_rel32/jcc_rel32 (points at the imm32).
    void patch_rel32(size_t site, size_t target) {
        s32 rel = static_cast<s32>(static_cast<s64>(target) - static_cast<s64>(site + 4));
        std::memcpy(buf_.data() + site, &rel, 4);
    }

    std::vector<u8>& buffer() { return buf_; }

private:
    std::vector<u8> buf_;
};

} // namespace x64
