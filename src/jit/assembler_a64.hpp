#pragma once
// Minimal hand-rolled AArch64 (AAPCS64) instruction encoder - the ARM64
// counterpart to assembler_x64.hpp, covering the same small instruction set
// needed to inline guest ALU/shift/immediate arithmetic and call out to the
// jit_helpers thunks for memory access.
//
// Every encoder function derives its instruction word from the documented
// AArch64 bitfield layout instead of using opaque magic constants, so the
// encoding can be checked field by field against the ARM Architecture
// Reference Manual.

#include "../common.hpp"
#include <vector>
#include <cstring>

namespace a64 {

// X0..X30, plus 31 used contextually as XZR/WZR (zero register) or SP
// depending on instruction class - we only ever use it as the zero register.
enum Reg : int { XZR = 31, SP = 31 };

enum class Cond : u32 { EQ = 0, NE = 1, CS = 2, CC = 3, MI = 4, PL = 5, VS = 6, VC = 7,
                         HI = 8, LS = 9, GE = 10, LT = 11, GT = 12, LE = 13, AL = 14 };

inline Cond invert(Cond c) { return static_cast<Cond>(static_cast<u32>(c) ^ 1u); }

class Assembler {
public:
    void emit(u32 word) { u8 t[4]; std::memcpy(t, &word, 4); buf_.insert(buf_.end(), t, t + 4); }
    size_t pos() const { return buf_.size(); }
    const u8* data() const { return buf_.data(); }
    size_t size() const { return buf_.size(); }
    std::vector<u8>& buffer() { return buf_; }

    // ---- move immediate: MOVZ + up to 3 MOVK, always emits all 4 words for
    // simplicity (dead MOVKs of #0 are harmless; codegen isn't the bottleneck). ----
    void movz(int rd, u16 imm16, int hw /*0..3*/, bool sf = true) {
        u32 w = (sf ? (1u << 31) : 0) | (0b10u << 29) | (0b100101u << 23) | (static_cast<u32>(hw) << 21)
              | (static_cast<u32>(imm16) << 5) | (rd & 31);
        emit(w);
    }
    void movk(int rd, u16 imm16, int hw, bool sf = true) {
        u32 w = (sf ? (1u << 31) : 0) | (0b11u << 29) | (0b100101u << 23) | (static_cast<u32>(hw) << 21)
              | (static_cast<u32>(imm16) << 5) | (rd & 31);
        emit(w);
    }
    void mov_imm64(int rd, u64 imm) {
        movz(rd, static_cast<u16>(imm), 0);
        for (int hw = 1; hw < 4; ++hw) {
            u16 chunk = static_cast<u16>(imm >> (hw * 16));
            movk(rd, chunk, hw);
        }
    }
    // Small helper for compile-time-known 32-bit values that don't need the
    // full 4-word sequence (branch targets aren't materialized this way, but
    // guest immediates are - keeping it uniform with mov_imm64 is simplest
    // and correctness-first, so this just forwards.
    void mov_imm64_sext32(int rd, s32 imm) { mov_imm64(rd, static_cast<u64>(static_cast<s64>(imm))); }

    // ---- register move (alias for ORR Xd, XZR, Xm) ----
    void mov_reg(int rd, int rn, bool sf = true) { orr_reg(rd, XZR, rn, sf); }

    // ---- load/store immediate (unsigned offset). `off` is a BYTE offset,
    // must be a multiple of the access size. ----
    void ldr64(int rt, int rn, s32 off) { ldst_uimm(0b11, 0b01, rt, rn, off, 8); }
    void str64(int rt, int rn, s32 off) { ldst_uimm(0b11, 0b00, rt, rn, off, 8); }
    void ldr32z(int rt, int rn, s32 off) { ldst_uimm(0b10, 0b01, rt, rn, off, 4); } // zero-extends to Xt
    void str32(int rt, int rn, s32 off) { ldst_uimm(0b10, 0b00, rt, rn, off, 4); }
    void ldrh_z(int rt, int rn, s32 off) { ldst_uimm(0b01, 0b01, rt, rn, off, 2); }
    void strh(int rt, int rn, s32 off) { ldst_uimm(0b01, 0b00, rt, rn, off, 2); }
    void ldrb_z(int rt, int rn, s32 off) { ldst_uimm(0b00, 0b01, rt, rn, off, 1); }
    void strb(int rt, int rn, s32 off) { ldst_uimm(0b00, 0b00, rt, rn, off, 1); }
    void ldrsw(int rt, int rn, s32 off) { ldst_uimm(0b10, 0b10, rt, rn, off, 4); }  // sign-extends to Xt
    void ldrsh64(int rt, int rn, s32 off) { ldst_uimm(0b01, 0b10, rt, rn, off, 2); } // sign-extends to Xt
    void ldrsb64(int rt, int rn, s32 off) { ldst_uimm(0b00, 0b10, rt, rn, off, 1); } // sign-extends to Xt

    // ---- add/sub/logical (register, no shift) ----
    void add_reg(int rd, int rn, int rm, bool sf = true) { dp3reg(sf, 0, 0, rd, rn, rm); }
    void sub_reg(int rd, int rn, int rm, bool sf = true) { dp3reg(sf, 1, 0, rd, rn, rm); }
    void adds_reg(int rd, int rn, int rm, bool sf = true) { dp3reg(sf, 0, 1, rd, rn, rm); } // sets NZCV incl. V (signed overflow)
    void subs_reg(int rd, int rn, int rm, bool sf = true) { dp3reg(sf, 1, 1, rd, rn, rm); } // also CMP when rd=XZR
    void cmp_reg(int rn, int rm, bool sf = true) { subs_reg(XZR, rn, rm, sf); }

    void and_reg(int rd, int rn, int rm, bool sf = true) { logical_reg(sf, 0b00, 0, rd, rn, rm); }
    void orr_reg(int rd, int rn, int rm, bool sf = true) { logical_reg(sf, 0b01, 0, rd, rn, rm); }
    void eor_reg(int rd, int rn, int rm, bool sf = true) { logical_reg(sf, 0b10, 0, rd, rn, rm); }
    void orn_reg(int rd, int rn, int rm, bool sf = true) { logical_reg(sf, 0b01, 1, rd, rn, rm); } // rd = rn | ~rm
    void mvn_reg(int rd, int rm, bool sf = true) { orn_reg(rd, XZR, rm, sf); }

    // ---- raw-bit move between a GPR and an FP scalar register (Sd<->Wn or
    // Dd<->Xn, no conversion - exactly the "reinterpret these bits as the
    // other kind of register" AArch64 provides for), and scalar FP
    // add/sub/mul. FP register numbers (0-31) are a separate file from the
    // Xn/Wn ones; callers just need to not confuse the two. ----
    void fmov_to_fp(int vd, int rn, bool dbl) {   // Sd,Wn or Dd,Xn
        u32 w = (dbl ? (1u << 31) : 0) | (0b11110u << 24) | (dbl ? (1u << 22) : 0) | (1u << 21) | (0b111u << 16) |
                (static_cast<u32>(rn & 31) << 5) | (vd & 31);
        emit(w);
    }
    void fmov_from_fp(int rd, int vn, bool dbl) { // Wd,Sn or Xd,Dn
        u32 w = (dbl ? (1u << 31) : 0) | (0b11110u << 24) | (dbl ? (1u << 22) : 0) | (1u << 21) | (0b110u << 16) |
                (static_cast<u32>(vn & 31) << 5) | (rd & 31);
        emit(w);
    }
    void fadd(int vd, int vn, int vm, bool dbl) { fp2src(dbl, 0b0010, vd, vn, vm); }
    void fsub(int vd, int vn, int vm, bool dbl) { fp2src(dbl, 0b0011, vd, vn, vm); }
    void fmul(int vd, int vn, int vm, bool dbl) { fp2src(dbl, 0b0000, vd, vn, vm); }
    void fdiv(int vd, int vn, int vm, bool dbl) { fp2src(dbl, 0b0001, vd, vn, vm); }
    // "Floating-point data-processing (1 source)" - FSQRT/FCVT(precision change).
    void fsqrt(int vd, int vn, bool dbl) { fp1src(dbl, 0b000011, vd, vn); }
    void fcvt_d_to_s(int vd, int vn) { fp1src(true, 0b000100, vd, vn); }  // source is double, result single
    void fcvt_s_to_d(int vd, int vn) { fp1src(false, 0b000101, vd, vn); } // source is single, result double
    // "Conversion between floating-point and integer": truncating float->int
    // (round toward zero, signed) and int->float (signed), both directions
    // parameterized by the *integer* register's width (32/64) and the FP
    // register's precision (single/double).
    void fcvtzs(int rd, int vn, bool int64, bool dbl) { fcvt_int(int64, dbl, 0b11, 0b000, rd, vn); }
    void scvtf(int vd, int rn, bool int64, bool dbl) { fcvt_int(int64, dbl, 0b00, 0b010, vd, rn); }

    // ---- add/sub (immediate, unsigned 12-bit, unshifted) ----
    void add_imm(int rd, int rn, u32 imm12, bool sf = true) { addsub_imm(sf, 0, 0, rd, rn, imm12); }
    void sub_imm(int rd, int rn, u32 imm12, bool sf = true) { addsub_imm(sf, 1, 0, rd, rn, imm12); }

    // ---- variable shifts (register) ----
    void lslv(int rd, int rn, int rm, bool sf = true) { dp2src(sf, 0b001000, rd, rn, rm); }
    void lsrv(int rd, int rn, int rm, bool sf = true) { dp2src(sf, 0b001001, rd, rn, rm); }
    void asrv(int rd, int rn, int rm, bool sf = true) { dp2src(sf, 0b001010, rd, rn, rm); }

    // ---- sign-extend word -> doubleword (SXTW Xd, Wn = SBFM Xd,Xn,#0,#31) ----
    void sxtw(int rd, int rn) {
        u32 w = (1u << 31) | (0b100110u << 23) | (1u << 22) | (0b011111u << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31);
        emit(w);
    }

    // ---- conditional select: CSET Xd,cond ----
    void csinc(int rd, int rn, int rm, Cond cond, bool sf = true) {
        u32 w = (sf ? (1u << 31) : 0) | (0b11010100u << 21) | (static_cast<u32>(rm & 31) << 16)
              | (static_cast<u32>(cond) << 12) | (0b01u << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31);
        emit(w);
    }
    void cset(int rd, Cond cond, bool sf = true) { csinc(rd, XZR, XZR, invert(cond), sf); }

    // ---- branches ----
    void ret() { emit(0xD65F03C0u); }
    void blr(int rn) { emit(0xD63F0000u | (static_cast<u32>(rn & 31) << 5)); }

    // Returns the byte offset of the instruction word to patch later.
    size_t b() { size_t site = pos(); emit(0x14000000u); return site; }
    size_t bcond(Cond cond) { size_t site = pos(); emit(0x54000000u | static_cast<u32>(cond)); return site; }
    size_t cbz(int rt, bool sf = true) { size_t site = pos(); emit((sf ? 0xB4000000u : 0x34000000u) | static_cast<u32>(rt & 31)); return site; }
    size_t cbnz(int rt, bool sf = true) { size_t site = pos(); emit((sf ? 0xB5000000u : 0x35000000u) | static_cast<u32>(rt & 31)); return site; }

    // `site` is the byte offset returned above; patches the branch's
    // relative-immediate field to point at `target` (both byte offsets into
    // the same buffer).
    void patch_branch(size_t site, size_t target) {
        s64 diff = static_cast<s64>(target) - static_cast<s64>(site);
        u32 word;
        std::memcpy(&word, buf_.data() + site, 4);
        u32 top11 = word & 0xFFE00000u; // preserves opcode/cond/Rt bits outside the immediate field
        if ((word & 0xFC000000u) == 0x14000000u) { // B: imm26
            u32 imm26 = static_cast<u32>(diff / 4) & 0x03FFFFFFu;
            word = (word & 0xFC000000u) | imm26;
        } else if ((word & 0x7E000000u) == 0x34000000u) { // CBZ/CBNZ: imm19 << 5 | Rt
            u32 imm19 = static_cast<u32>(diff / 4) & 0x7FFFFu;
            word = (word & 0xFF00001Fu) | (imm19 << 5);
        } else { // B.cond: imm19 << 5 | cond
            (void)top11;
            u32 imm19 = static_cast<u32>(diff / 4) & 0x7FFFFu;
            word = (word & 0xFF00000Fu) | (imm19 << 5);
        }
        std::memcpy(buf_.data() + site, &word, 4);
    }

    // ---- SP-relative prologue/epilogue helpers (frame_bytes must be a
    // multiple of 16) ----
    void sub_sp(u32 frame_bytes) { sub_imm(SP, SP, frame_bytes); }
    void add_sp(u32 frame_bytes) { add_imm(SP, SP, frame_bytes); }

private:
    void ldst_uimm(u32 size, u32 opc, int rt, int rn, s32 off, u32 access_size) {
        u32 uoff = static_cast<u32>(off);
        u32 imm12 = uoff / access_size;
        u32 w = (size << 30) | (0b111u << 27) | (0b01u << 24) | (opc << 22) | ((imm12 & 0xFFFu) << 10)
              | (static_cast<u32>(rn & 31) << 5) | (rt & 31);
        emit(w);
    }
    // "Add/subtract (shifted register)" with shift=0.
    void dp3reg(bool sf, u32 op, u32 s, int rd, int rn, int rm) {
        u32 w = (sf ? (1u << 31) : 0) | (op << 30) | (s << 29) | (0b01011u << 24)
              | (static_cast<u32>(rm & 31) << 16) | (static_cast<u32>(rn & 31) << 5) | (rd & 31);
        emit(w);
    }
    // "Logical (shifted register)" with shift=0.
    void logical_reg(bool sf, u32 opc, u32 n, int rd, int rn, int rm) {
        u32 w = (sf ? (1u << 31) : 0) | (opc << 29) | (0b01010u << 24) | (n << 21)
              | (static_cast<u32>(rm & 31) << 16) | (static_cast<u32>(rn & 31) << 5) | (rd & 31);
        emit(w);
    }
    // "Add/subtract (immediate)", shift=0.
    void addsub_imm(bool sf, u32 op, u32 s, int rd, int rn, u32 imm12) {
        u32 w = (sf ? (1u << 31) : 0) | (op << 30) | (s << 29) | (0b100010u << 23)
              | ((imm12 & 0xFFFu) << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31);
        emit(w);
    }
    // "Data-processing (2 source)" - used for LSLV/LSRV/ASRV.
    void dp2src(bool sf, u32 opcode6, int rd, int rn, int rm) {
        u32 w = (sf ? (1u << 31) : 0) | (0b11010110u << 21) | (static_cast<u32>(rm & 31) << 16)
              | (opcode6 << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31);
        emit(w);
    }
    // "Floating-point data-processing (2 source)" - used for FADD/FSUB/FMUL.
    void fp2src(bool dbl, u32 opcode4, int vd, int vn, int vm) {
        u32 w = 0x1E200000u | (dbl ? (1u << 22) : 0) | (static_cast<u32>(vm & 31) << 16)
              | (opcode4 << 12) | (0b10u << 10) | (static_cast<u32>(vn & 31) << 5) | (vd & 31);
        emit(w);
    }
    // "Floating-point data-processing (1 source)" - FSQRT/FABS/FNEG/FCVT(precision).
    // `dbl` selects the *source* precision (for FCVT this is what makes the
    // two conversion directions distinct opcodes rather than a shared one).
    void fp1src(bool dbl, u32 opcode6, int vd, int vn) {
        u32 w = 0x1E204000u | (dbl ? (1u << 22) : 0) | (opcode6 << 15) | (static_cast<u32>(vn & 31) << 5) | (vd & 31);
        emit(w);
    }
    // "Conversion between floating-point and integer" (FCVTZS/SCVTF/...).
    void fcvt_int(bool int64, bool dbl, u32 rmode, u32 opcode3, int rd_or_vd, int rn_or_vn) {
        u32 w = (int64 ? (1u << 31) : 0) | 0x1E200000u | (dbl ? (1u << 22) : 0) | (rmode << 19) | (opcode3 << 16)
              | (static_cast<u32>(rn_or_vn & 31) << 5) | (rd_or_vd & 31);
        emit(w);
    }

    std::vector<u8> buf_;
};

} // namespace a64
