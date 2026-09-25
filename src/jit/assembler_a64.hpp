#pragma once
// Minimal hand-rolled AArch64 (AAPCS64) instruction encoder - the ARM64
// counterpart to assembler_x64.hpp, covering the instructions the recompiler
// emits: integer ALU in its register/immediate/bitfield forms, loads and
// stores (immediate and register offset), branches, the multiply/divide
// family and the scalar FP operations COP1 maps onto.
//
// Every encoder function derives its instruction word from the documented
// AArch64 bitfield layout instead of using opaque magic constants, so the
// encoding can be checked field by field against the ARM Architecture
// Reference Manual. tools/a64_encoding_check.sh assembles the same
// instructions with the system assembler and compares the words.
//
// Register number 31 means XZR/WZR or SP depending on the instruction class,
// exactly as in the architecture: ZR for the (shifted-register) ALU forms,
// multiply/divide, conditional select and the Rt of loads/stores; SP for the
// base register of loads/stores and for the (immediate) ADD/SUB forms. The
// helpers below never guess - callers pass XZR only where it is valid.

#include "../common.hpp"
#include <vector>
#include <cstring>

namespace a64 {

enum Reg : int { XZR = 31, SP = 31 };

enum class Cond : u32 { EQ = 0, NE = 1, CS = 2, CC = 3, MI = 4, PL = 5, VS = 6, VC = 7,
                         HI = 8, LS = 9, GE = 10, LT = 11, GT = 12, LE = 13, AL = 14 };
constexpr Cond HS = Cond::CS;
constexpr Cond LO = Cond::CC;

inline Cond invert(Cond c) { return static_cast<Cond>(static_cast<u32>(c) ^ 1u); }

// Register-offset addressing: how Rm is extended before being added (and
// optionally shifted left by the access size).
enum class Extend : u32 { UXTW = 0b010, LSL = 0b011, SXTW = 0b110, SXTX = 0b111 };

// Encodes `imm` as an AArch64 "bitmask immediate" (logical instructions):
// a repeating element of 2/4/8/16/32/64 bits holding a rotated run of ones.
// Returns false if it has no such encoding (0 and all-ones never do).
inline bool encode_logical_imm(u64 imm, bool sf, u32& n, u32& immr, u32& imms) {
    if (!sf) {
        imm &= 0xFFFFFFFFull;
        imm |= imm << 32; // a 32-bit pattern is the 64-bit one with a 32-bit (or smaller) element
    }
    if (imm == 0 || imm == ~0ull) return false;
    // Smallest element size the value repeats with.
    u32 size = 64;
    while (size > 2) {
        const u32 half = size / 2;
        const u64 mask = (1ull << half) - 1;
        if ((imm & mask) != ((imm >> half) & mask)) break;
        size = half;
    }
    const u64 emask = size == 64 ? ~0ull : ((1ull << size) - 1);
    const u64 elem = imm & emask;
    // Rotate right until the run of ones starts at bit 0 (the element is then
    // 0...01...1): count how far that is and how many ones there are.
    u32 rot = 0;
    u64 e = elem;
    auto ror = [&](u64 v, u32 r) { return r == 0 ? v : (((v >> r) | (v << (size - r))) & emask); };
    // Find a rotation where bit 0 is 1 and bit size-1 is 0.
    for (rot = 0; rot < size; ++rot) {
        e = ror(elem, rot);
        if ((e & 1) && !(e >> (size - 1) & 1)) break;
    }
    if (rot == size) return false;
    // e must now be a contiguous run of ones from bit 0.
    const u64 ones = e + 1;
    if (ones & e) return false; // not of the form 2^k - 1
    u32 count = 0;
    while (e) { e >>= 1; ++count; }
    // ROR(pattern, immr) of the canonical run gives the value: the canonical
    // run rotated right by `rot` is `e`, so the value is the run rotated right
    // by (size - rot) mod size.
    immr = (size - rot) % size;
    const u32 size_bits = (size == 64) ? 0 : (~(size * 2 - 1) & 0x3F); // 0b0xxxxx, 0b10xxxx, ...
    imms = (size_bits | (count - 1)) & 0x3F;
    n = (size == 64) ? 1 : 0;
    return true;
}

class Assembler {
public:
    void emit(u32 word) { u8 t[4]; std::memcpy(t, &word, 4); buf_.insert(buf_.end(), t, t + 4); }
    size_t pos() const { return buf_.size(); }
    const u8* data() const { return buf_.data(); }
    size_t size() const { return buf_.size(); }
    std::vector<u8>& buffer() { return buf_; }
    u32 word_at(size_t off) const { u32 w; std::memcpy(&w, buf_.data() + off, 4); return w; }
    void set_word(size_t off, u32 w) { std::memcpy(buf_.data() + off, &w, 4); }

    // ---- move wide immediate ----
    void movz(int rd, u16 imm16, int hw /*0..3*/, bool sf = true) { movw(sf, 0b10, rd, imm16, hw); }
    void movk(int rd, u16 imm16, int hw, bool sf = true) { movw(sf, 0b11, rd, imm16, hw); }
    void movn(int rd, u16 imm16, int hw, bool sf = true) { movw(sf, 0b00, rd, imm16, hw); }

    // Shortest sequence that puts `imm` in rd: a single ORR with a bitmask
    // immediate, or MOVZ/MOVN plus MOVKs for the 16-bit chunks that differ.
    void mov_imm(int rd, u64 imm, bool sf = true) {
        if (!sf) imm &= 0xFFFFFFFFull;
        const int chunks = sf ? 4 : 2;
        int zero = 0, ones = 0;
        for (int hw = 0; hw < chunks; ++hw) {
            const u16 c = static_cast<u16>(imm >> (hw * 16));
            zero += c == 0;
            ones += c == 0xFFFF;
        }
        if (zero < chunks - 1 && ones < chunks - 1) {
            u32 n, immr, imms;
            if (encode_logical_imm(imm, sf, n, immr, imms)) {
                logical_imm_raw(sf, 0b01, rd, XZR, n, immr, imms); // ORR rd, zr, #imm
                return;
            }
        }
        const bool use_movn = ones > zero;
        bool first = true;
        for (int hw = 0; hw < chunks; ++hw) {
            const u16 c = static_cast<u16>(imm >> (hw * 16));
            if (use_movn ? c == 0xFFFF : c == 0) continue;
            if (first) {
                if (use_movn) movn(rd, static_cast<u16>(~c), hw, sf);
                else movz(rd, c, hw, sf);
                first = false;
            } else {
                movk(rd, c, hw, sf);
            }
        }
        if (first) { // all chunks 0 (or all 0xFFFF)
            if (use_movn) movn(rd, 0, 0, sf);
            else movz(rd, 0, 0, sf);
        }
    }
    void mov_imm64(int rd, u64 imm) { mov_imm(rd, imm, true); }
    void mov_imm64_sext32(int rd, s32 imm) { mov_imm(rd, static_cast<u64>(static_cast<s64>(imm)), true); }

    // ---- register move (alias for ORR Xd, XZR, Xm) ----
    void mov_reg(int rd, int rm, bool sf = true) { orr_reg(rd, XZR, rm, sf); }
    // MOV to/from SP (ADD Xd, Xn, #0) - the ORR form can't address SP.
    void mov_sp(int rd, int rn) { add_imm(rd, rn, 0); }

    // ---- load/store, unsigned scaled immediate offset. `off` is a BYTE
    // offset, a multiple of the access size and < 4096 * size. ----
    void ldr64(int rt, int rn, s32 off) { ldst_uimm(0b11, 0, 0b01, rt, rn, off, 8); }
    void str64(int rt, int rn, s32 off) { ldst_uimm(0b11, 0, 0b00, rt, rn, off, 8); }
    void ldr32z(int rt, int rn, s32 off) { ldst_uimm(0b10, 0, 0b01, rt, rn, off, 4); } // zero-extends to Xt
    void str32(int rt, int rn, s32 off) { ldst_uimm(0b10, 0, 0b00, rt, rn, off, 4); }
    void ldrh_z(int rt, int rn, s32 off) { ldst_uimm(0b01, 0, 0b01, rt, rn, off, 2); }
    void strh(int rt, int rn, s32 off) { ldst_uimm(0b01, 0, 0b00, rt, rn, off, 2); }
    void ldrb_z(int rt, int rn, s32 off) { ldst_uimm(0b00, 0, 0b01, rt, rn, off, 1); }
    void strb(int rt, int rn, s32 off) { ldst_uimm(0b00, 0, 0b00, rt, rn, off, 1); }
    void ldrsw(int rt, int rn, s32 off) { ldst_uimm(0b10, 0, 0b10, rt, rn, off, 4); }  // sign-extends to Xt
    void ldrsh64(int rt, int rn, s32 off) { ldst_uimm(0b01, 0, 0b10, rt, rn, off, 2); } // sign-extends to Xt
    void ldrsb64(int rt, int rn, s32 off) { ldst_uimm(0b00, 0, 0b10, rt, rn, off, 1); } // sign-extends to Xt
    // FP/SIMD scalar registers S (32-bit) / D (64-bit).
    void ldr_s(int vt, int rn, s32 off) { ldst_uimm(0b10, 1, 0b01, vt, rn, off, 4); }
    void str_s(int vt, int rn, s32 off) { ldst_uimm(0b10, 1, 0b00, vt, rn, off, 4); }
    void ldr_d(int vt, int rn, s32 off) { ldst_uimm(0b11, 1, 0b01, vt, rn, off, 8); }
    void str_d(int vt, int rn, s32 off) { ldst_uimm(0b11, 1, 0b00, vt, rn, off, 8); }

    // ---- load/store, register offset: address = Xn + extend(Rm) << (shift ? log2(size) : 0) ----
    void ldr64_r(int rt, int rn, int rm, Extend ext = Extend::LSL, bool shift = false) { ldst_reg(0b11, 0b01, rt, rn, rm, ext, shift); }
    void str64_r(int rt, int rn, int rm, Extend ext = Extend::LSL, bool shift = false) { ldst_reg(0b11, 0b00, rt, rn, rm, ext, shift); }
    void ldr32_r(int rt, int rn, int rm, Extend ext = Extend::LSL, bool shift = false) { ldst_reg(0b10, 0b01, rt, rn, rm, ext, shift); }
    void str32_r(int rt, int rn, int rm, Extend ext = Extend::LSL, bool shift = false) { ldst_reg(0b10, 0b00, rt, rn, rm, ext, shift); }
    void ldrh_r(int rt, int rn, int rm, Extend ext = Extend::LSL, bool shift = false) { ldst_reg(0b01, 0b01, rt, rn, rm, ext, shift); }
    void strh_r(int rt, int rn, int rm, Extend ext = Extend::LSL, bool shift = false) { ldst_reg(0b01, 0b00, rt, rn, rm, ext, shift); }
    void ldrb_r(int rt, int rn, int rm, Extend ext = Extend::LSL, bool shift = false) { ldst_reg(0b00, 0b01, rt, rn, rm, ext, shift); }
    void strb_r(int rt, int rn, int rm, Extend ext = Extend::LSL, bool shift = false) { ldst_reg(0b00, 0b00, rt, rn, rm, ext, shift); }
    void ldrsb64_r(int rt, int rn, int rm, Extend ext = Extend::LSL, bool shift = false) { ldst_reg(0b00, 0b10, rt, rn, rm, ext, shift); }
    void ldrsh64_r(int rt, int rn, int rm, Extend ext = Extend::LSL, bool shift = false) { ldst_reg(0b01, 0b10, rt, rn, rm, ext, shift); }

    // ---- load/store pair (64-bit), signed offset / pre-index / post-index ----
    void stp64(int rt, int rt2, int rn, s32 off) { ldst_pair(0b10, 0, rt, rt2, rn, off); }
    void ldp64(int rt, int rt2, int rn, s32 off) { ldst_pair(0b10, 1, rt, rt2, rn, off); }
    void stp64_pre(int rt, int rt2, int rn, s32 off) { ldst_pair(0b11, 0, rt, rt2, rn, off); }
    void ldp64_post(int rt, int rt2, int rn, s32 off) { ldst_pair(0b01, 1, rt, rt2, rn, off); }

    // ---- add/sub (shifted register, LSL #0) ----
    void add_reg(int rd, int rn, int rm, bool sf = true) { dp3reg(sf, 0, 0, rd, rn, rm, 0, 0); }
    void sub_reg(int rd, int rn, int rm, bool sf = true) { dp3reg(sf, 1, 0, rd, rn, rm, 0, 0); }
    void adds_reg(int rd, int rn, int rm, bool sf = true) { dp3reg(sf, 0, 1, rd, rn, rm, 0, 0); } // sets NZCV incl. V (signed overflow)
    void subs_reg(int rd, int rn, int rm, bool sf = true) { dp3reg(sf, 1, 1, rd, rn, rm, 0, 0); } // also CMP when rd=XZR
    void cmp_reg(int rn, int rm, bool sf = true) { subs_reg(XZR, rn, rm, sf); }
    void neg_reg(int rd, int rm, bool sf = true) { sub_reg(rd, XZR, rm, sf); }
    // rd = rn + (rm << sh)
    void add_reg_lsl(int rd, int rn, int rm, u32 sh, bool sf = true) { dp3reg(sf, 0, 0, rd, rn, rm, 0b00, sh); }
    // ---- add/sub (extended register): rd = rn + extend(Wm/Xm). Rd/Rn=31 is SP here. ----
    void add_ext(int rd, int rn, int rm, Extend ext, u32 lsl = 0) { addsub_ext(1, 0, 0, rd, rn, rm, static_cast<u32>(ext), lsl); }

    // ---- logical (shifted register, LSL #0) ----
    void and_reg(int rd, int rn, int rm, bool sf = true) { logical_reg(sf, 0b00, 0, rd, rn, rm); }
    void orr_reg(int rd, int rn, int rm, bool sf = true) { logical_reg(sf, 0b01, 0, rd, rn, rm); }
    void eor_reg(int rd, int rn, int rm, bool sf = true) { logical_reg(sf, 0b10, 0, rd, rn, rm); }
    void ands_reg(int rd, int rn, int rm, bool sf = true) { logical_reg(sf, 0b11, 0, rd, rn, rm); }
    void tst_reg(int rn, int rm, bool sf = true) { ands_reg(XZR, rn, rm, sf); }
    void orn_reg(int rd, int rn, int rm, bool sf = true) { logical_reg(sf, 0b01, 1, rd, rn, rm); } // rd = rn | ~rm
    void mvn_reg(int rd, int rm, bool sf = true) { orn_reg(rd, XZR, rm, sf); }

    // ---- logical (immediate). Only valid if encode_logical_imm() accepts
    // the value; the *_imm_or() forms fall back to materializing it in
    // `scratch` first. Rd=31 would be SP, so rd must be a real register. ----
    static bool is_logical_imm(u64 imm, bool sf) { u32 n, r, s; return encode_logical_imm(imm, sf, n, r, s); }
    void and_imm(int rd, int rn, u64 imm, bool sf = true) { logical_imm(sf, 0b00, rd, rn, imm); }
    void orr_imm(int rd, int rn, u64 imm, bool sf = true) { logical_imm(sf, 0b01, rd, rn, imm); }
    void eor_imm(int rd, int rn, u64 imm, bool sf = true) { logical_imm(sf, 0b10, rd, rn, imm); }
    void tst_imm(int rn, u64 imm, bool sf = true) { logical_imm(sf, 0b11, XZR, rn, imm); } // ANDS XZR (Rd=31 is ZR for ANDS)
    void and_imm_or(int rd, int rn, u64 imm, int scratch, bool sf = true) { logical_any(sf, 0b00, rd, rn, imm, scratch); }
    void orr_imm_or(int rd, int rn, u64 imm, int scratch, bool sf = true) { logical_any(sf, 0b01, rd, rn, imm, scratch); }
    void eor_imm_or(int rd, int rn, u64 imm, int scratch, bool sf = true) { logical_any(sf, 0b10, rd, rn, imm, scratch); }

    // ---- raw-bit move between a GPR and an FP scalar register (Sd<->Wn or
    // Dd<->Xn, no conversion), and scalar FP arithmetic. FP register numbers
    // (0-31) are a separate file from the Xn/Wn ones. ----
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
    // "Floating-point data-processing (1 source)" - FMOV/FABS/FNEG/FSQRT/FCVT.
    void fmov_fp(int vd, int vn, bool dbl) { fp1src(dbl, 0b000000, vd, vn); }
    void fabs_(int vd, int vn, bool dbl) { fp1src(dbl, 0b000001, vd, vn); }
    void fneg(int vd, int vn, bool dbl) { fp1src(dbl, 0b000010, vd, vn); }
    void fsqrt(int vd, int vn, bool dbl) { fp1src(dbl, 0b000011, vd, vn); }
    void fcvt_d_to_s(int vd, int vn) { fp1src(true, 0b000100, vd, vn); }  // source is double, result single
    void fcvt_s_to_d(int vd, int vn) { fp1src(false, 0b000101, vd, vn); } // source is single, result double
    // FCMP Sn, Sm / Dn, Dm (quiet compare; sets NZCV, unordered = C and V set).
    void fcmp(int vn, int vm, bool dbl) {
        emit(0x1E202000u | (dbl ? (1u << 22) : 0) | (static_cast<u32>(vm & 31) << 16) | (static_cast<u32>(vn & 31) << 5));
    }
    // FCMP Sn, #0.0 / Dn, #0.0
    void fcmp_zero(int vn, bool dbl) {
        emit(0x1E202008u | (dbl ? (1u << 22) : 0) | (static_cast<u32>(vn & 31) << 5));
    }
    // FCSEL: vd = cond ? vn : vm
    void fcsel(int vd, int vn, int vm, Cond cond, bool dbl) {
        emit(0x1E200C00u | (dbl ? (1u << 22) : 0) | (static_cast<u32>(vm & 31) << 16) | (static_cast<u32>(cond) << 12)
             | (static_cast<u32>(vn & 31) << 5) | (vd & 31));
    }
    // "Conversion between floating-point and integer", parameterized by the
    // *integer* register's width (32/64) and the FP register's precision.
    void fcvtzs(int rd, int vn, bool int64, bool dbl) { fcvt_int(int64, dbl, 0b11, 0b000, rd, vn); } // toward zero
    void fcvtas(int rd, int vn, bool int64, bool dbl) { fcvt_int(int64, dbl, 0b00, 0b100, rd, vn); } // nearest, ties away
    void fcvtps(int rd, int vn, bool int64, bool dbl) { fcvt_int(int64, dbl, 0b01, 0b000, rd, vn); } // toward +inf
    void fcvtms(int rd, int vn, bool int64, bool dbl) { fcvt_int(int64, dbl, 0b10, 0b000, rd, vn); } // toward -inf
    void scvtf(int vd, int rn, bool int64, bool dbl) { fcvt_int(int64, dbl, 0b00, 0b010, vd, rn); }

    // ---- add/sub (immediate). imm12, optionally shifted left by 12. Rd/Rn=31 is SP. ----
    void add_imm(int rd, int rn, u32 imm12, bool sf = true, bool lsl12 = false) { addsub_imm(sf, 0, 0, rd, rn, imm12, lsl12); }
    void sub_imm(int rd, int rn, u32 imm12, bool sf = true, bool lsl12 = false) { addsub_imm(sf, 1, 0, rd, rn, imm12, lsl12); }
    void adds_imm(int rd, int rn, u32 imm12, bool sf = true, bool lsl12 = false) { addsub_imm(sf, 0, 1, rd, rn, imm12, lsl12); }
    void subs_imm(int rd, int rn, u32 imm12, bool sf = true, bool lsl12 = false) { addsub_imm(sf, 1, 1, rd, rn, imm12, lsl12); }
    void cmp_imm(int rn, u32 imm12, bool sf = true, bool lsl12 = false) { subs_imm(XZR, rn, imm12, sf, lsl12); }
    void cmn_imm(int rn, u32 imm12, bool sf = true, bool lsl12 = false) { adds_imm(XZR, rn, imm12, sf, lsl12); }
    // True if `v` fits an (optionally LSL #12-shifted) imm12.
    static bool is_addsub_imm(u64 v) { return v < 4096 || ((v & 0xFFF) == 0 && v < (4096ull << 12)); }
    // rd = rn + imm for any 64-bit constant (rn/rd may not be SP), using `scratch` if needed.
    void add_any(int rd, int rn, s64 imm, int scratch, bool sf = true) {
        const bool neg = imm < 0;
        const u64 mag = neg ? static_cast<u64>(0) - static_cast<u64>(imm) : static_cast<u64>(imm);
        if (!sf && (mag >> 32)) { // only the low 32 bits matter
            add_any(rd, rn, static_cast<s32>(static_cast<u32>(imm)), scratch, false);
            return;
        }
        if (mag < 4096) {
            if (mag == 0 && rd == rn) return;
            neg ? sub_imm(rd, rn, static_cast<u32>(mag), sf) : add_imm(rd, rn, static_cast<u32>(mag), sf);
        } else if ((mag & 0xFFF) == 0 && mag < (4096ull << 12)) {
            neg ? sub_imm(rd, rn, static_cast<u32>(mag >> 12), sf, true) : add_imm(rd, rn, static_cast<u32>(mag >> 12), sf, true);
        } else if (mag < (4096ull << 12)) {
            if (neg) { sub_imm(rd, rn, static_cast<u32>(mag >> 12), sf, true); sub_imm(rd, rd, static_cast<u32>(mag & 0xFFF), sf); }
            else { add_imm(rd, rn, static_cast<u32>(mag >> 12), sf, true); add_imm(rd, rd, static_cast<u32>(mag & 0xFFF), sf); }
        } else {
            mov_imm(scratch, static_cast<u64>(imm), sf);
            add_reg(rd, rn, scratch, sf);
        }
    }
    // Sets flags for rn - imm (CMP), any constant.
    void cmp_any(int rn, s64 imm, int scratch, bool sf = true) {
        if (imm >= 0 && imm < 4096) cmp_imm(rn, static_cast<u32>(imm), sf);
        else if (imm < 0 && imm > -4096) cmn_imm(rn, static_cast<u32>(-imm), sf);
        else { mov_imm(scratch, static_cast<u64>(imm), sf); cmp_reg(rn, scratch, sf); }
    }

    // ---- variable shifts (register; the amount is taken mod 32/64 by hardware) ----
    void lslv(int rd, int rn, int rm, bool sf = true) { dp2src(sf, 0b001000, rd, rn, rm); }
    void lsrv(int rd, int rn, int rm, bool sf = true) { dp2src(sf, 0b001001, rd, rn, rm); }
    void asrv(int rd, int rn, int rm, bool sf = true) { dp2src(sf, 0b001010, rd, rn, rm); }
    void udiv(int rd, int rn, int rm, bool sf = true) { dp2src(sf, 0b000010, rd, rn, rm); }
    void sdiv(int rd, int rn, int rm, bool sf = true) { dp2src(sf, 0b000011, rd, rn, rm); }

    // ---- bitfield moves (SBFM/UBFM) and their aliases ----
    void sbfm(int rd, int rn, u32 immr, u32 imms, bool sf = true) { bitfield(sf, 0b00, rd, rn, immr, imms); }
    void bfm(int rd, int rn, u32 immr, u32 imms, bool sf = true) { bitfield(sf, 0b01, rd, rn, immr, imms); }
    // Inserts the low `width` bits of rn at bit `lsb` of rd, the other bits of rd unchanged.
    void bfi(int rd, int rn, u32 lsb, u32 width, bool sf = true) { const u32 w = sf ? 64 : 32; bfm(rd, rn, (w - lsb) % w, width - 1, sf); }
    void ubfm(int rd, int rn, u32 immr, u32 imms, bool sf = true) { bitfield(sf, 0b10, rd, rn, immr, imms); }
    void lsl_imm(int rd, int rn, u32 sh, bool sf = true) { const u32 w = sf ? 64 : 32; ubfm(rd, rn, (w - sh) % w, w - 1 - sh, sf); }
    void lsr_imm(int rd, int rn, u32 sh, bool sf = true) { ubfm(rd, rn, sh, sf ? 63 : 31, sf); }
    void asr_imm(int rd, int rn, u32 sh, bool sf = true) { sbfm(rd, rn, sh, sf ? 63 : 31, sf); }
    void sbfx(int rd, int rn, u32 lsb, u32 width, bool sf = true) { sbfm(rd, rn, lsb, lsb + width - 1, sf); }
    void ubfx(int rd, int rn, u32 lsb, u32 width, bool sf = true) { ubfm(rd, rn, lsb, lsb + width - 1, sf); }
    void sbfiz(int rd, int rn, u32 lsb, u32 width, bool sf = true) { const u32 w = sf ? 64 : 32; sbfm(rd, rn, (w - lsb) % w, width - 1, sf); }
    void sxtw(int rd, int rn) { sbfm(rd, rn, 0, 31, true); }
    void sxth(int rd, int rn) { sbfm(rd, rn, 0, 15, true); }
    void sxtb(int rd, int rn) { sbfm(rd, rn, 0, 7, true); }
    void uxtw(int rd, int rn) { ubfm(rd, rn, 0, 31, true); } // == MOV Wd, Wn
    // ROR (immediate) = EXTR Rd, Rn, Rn, #sh
    void ror_imm(int rd, int rn, u32 sh, bool sf = true) {
        emit((sf ? (1u << 31) : 0) | (0b00100111u << 23) | (sf ? (1u << 22) : 0) | (static_cast<u32>(rn & 31) << 16) |
             ((sh & 63) << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31));
    }

    // ---- byte reversal ----
    void rev64(int rd, int rn) { emit(0xDAC00C00u | (static_cast<u32>(rn & 31) << 5) | (rd & 31)); }
    void rev32(int rd, int rn) { emit(0x5AC00800u | (static_cast<u32>(rn & 31) << 5) | (rd & 31)); }   // REV Wd, Wn
    void rev16_w(int rd, int rn) { emit(0x5AC00400u | (static_cast<u32>(rn & 31) << 5) | (rd & 31)); } // REV16 Wd, Wn

    // ---- multiply (MADD/MSUB and the widening forms) ----
    void madd(int rd, int rn, int rm, int ra, bool sf = true) { dp3src(sf, 0b000, 0, rd, rn, rm, ra); }
    void msub(int rd, int rn, int rm, int ra, bool sf = true) { dp3src(sf, 0b000, 1, rd, rn, rm, ra); }
    void mul(int rd, int rn, int rm, bool sf = true) { madd(rd, rn, rm, XZR, sf); }
    void smull(int rd, int rn, int rm) { dp3src(true, 0b001, 0, rd, rn, rm, XZR); } // Xd = Wn * Wm (signed)
    void umull(int rd, int rn, int rm) { dp3src(true, 0b101, 0, rd, rn, rm, XZR); } // Xd = Wn * Wm (unsigned)
    void smulh(int rd, int rn, int rm) { dp3src(true, 0b010, 0, rd, rn, rm, XZR); }
    void umulh(int rd, int rn, int rm) { dp3src(true, 0b110, 0, rd, rn, rm, XZR); }

    // ---- conditional select ----
    void csel(int rd, int rn, int rm, Cond cond, bool sf = true) { condsel(sf, 0, 0b00, rd, rn, rm, cond); }
    void csinc(int rd, int rn, int rm, Cond cond, bool sf = true) { condsel(sf, 0, 0b01, rd, rn, rm, cond); }
    void csinv(int rd, int rn, int rm, Cond cond, bool sf = true) { condsel(sf, 1, 0b00, rd, rn, rm, cond); }
    void csneg(int rd, int rn, int rm, Cond cond, bool sf = true) { condsel(sf, 1, 0b01, rd, rn, rm, cond); }
    void cset(int rd, Cond cond, bool sf = true) { csinc(rd, XZR, XZR, invert(cond), sf); }

    // ---- branches ----
    void ret() { emit(0xD65F03C0u); }
    void blr(int rn) { emit(0xD63F0000u | (static_cast<u32>(rn & 31) << 5)); }
    void br(int rn) { emit(0xD61F0000u | (static_cast<u32>(rn & 31) << 5)); }

    // Each returns the byte offset of the instruction word, for patch_branch().
    size_t b() { size_t site = pos(); emit(0x14000000u); return site; }
    size_t bl() { size_t site = pos(); emit(0x94000000u); return site; }
    size_t bcond(Cond cond) { size_t site = pos(); emit(0x54000000u | static_cast<u32>(cond)); return site; }
    size_t cbz(int rt, bool sf = true) { size_t site = pos(); emit((sf ? 0xB4000000u : 0x34000000u) | static_cast<u32>(rt & 31)); return site; }
    size_t cbnz(int rt, bool sf = true) { size_t site = pos(); emit((sf ? 0xB5000000u : 0x35000000u) | static_cast<u32>(rt & 31)); return site; }
    size_t tbz(int rt, u32 bit) { size_t site = pos(); emit(tb_word(0x36000000u, rt, bit)); return site; }
    size_t tbnz(int rt, u32 bit) { size_t site = pos(); emit(tb_word(0x37000000u, rt, bit)); return site; }

    // Retargets the branch at `site` (a byte offset returned above) to
    // `target` (a byte offset into the same buffer).
    void patch_branch(size_t site, size_t target) {
        u32 word = word_at(site);
        set_word(site, retarget(word, static_cast<s64>(target) - static_cast<s64>(site)));
    }
    // The same, for a branch instruction word living anywhere: `diff` is the
    // byte distance from the branch to its new target.
    static u32 retarget(u32 word, s64 diff) {
        if ((word & 0x7C000000u) == 0x14000000u) { // B / BL: imm26
            return (word & 0xFC000000u) | (static_cast<u32>(diff / 4) & 0x03FFFFFFu);
        }
        if ((word & 0x7E000000u) == 0x36000000u) { // TBZ/TBNZ: imm14 << 5
            return (word & 0xFFF8001Fu) | ((static_cast<u32>(diff / 4) & 0x3FFFu) << 5);
        }
        // CBZ/CBNZ (imm19 << 5 | Rt) and B.cond (imm19 << 5 | cond) share the field.
        return (word & 0xFF00001Fu) | ((static_cast<u32>(diff / 4) & 0x7FFFFu) << 5);
    }
    // An unconditional B from `from` to `to` (absolute addresses, +-128 MB).
    static u32 b_word(const void* from, const void* to) {
        const s64 diff = reinterpret_cast<const u8*>(to) - reinterpret_cast<const u8*>(from);
        return 0x14000000u | (static_cast<u32>(diff / 4) & 0x03FFFFFFu);
    }

    // ---- SP-relative prologue/epilogue helpers (frame_bytes must be a
    // multiple of 16) ----
    void sub_sp(u32 frame_bytes) { sub_imm(SP, SP, frame_bytes); }
    void add_sp(u32 frame_bytes) { add_imm(SP, SP, frame_bytes); }

private:
    void movw(bool sf, u32 opc, int rd, u16 imm16, int hw) {
        emit((sf ? (1u << 31) : 0) | (opc << 29) | (0b100101u << 23) | (static_cast<u32>(hw) << 21)
             | (static_cast<u32>(imm16) << 5) | (rd & 31));
    }
    void ldst_uimm(u32 size, u32 v, u32 opc, int rt, int rn, s32 off, u32 access_size) {
        const u32 imm12 = static_cast<u32>(off) / access_size;
        emit((size << 30) | (0b111u << 27) | (v << 26) | (0b01u << 24) | (opc << 22) | ((imm12 & 0xFFFu) << 10)
             | (static_cast<u32>(rn & 31) << 5) | (rt & 31));
    }
    void ldst_reg(u32 size, u32 opc, int rt, int rn, int rm, Extend ext, bool shift) {
        emit((size << 30) | (0b111u << 27) | (opc << 22) | (1u << 21) | (static_cast<u32>(rm & 31) << 16)
             | (static_cast<u32>(ext) << 13) | ((shift ? 1u : 0u) << 12) | (0b10u << 10)
             | (static_cast<u32>(rn & 31) << 5) | (rt & 31));
    }
    // type: 0b01 post-index, 0b10 signed offset, 0b11 pre-index. 64-bit (opc=10).
    void ldst_pair(u32 type, u32 load, int rt, int rt2, int rn, s32 off) {
        const u32 imm7 = static_cast<u32>(off / 8) & 0x7Fu;
        emit((0b10u << 30) | (0b101u << 27) | (type << 23) | (load << 22) | (imm7 << 15)
             | (static_cast<u32>(rt2 & 31) << 10) | (static_cast<u32>(rn & 31) << 5) | (rt & 31));
    }
    // "Add/subtract (shifted register)".
    void dp3reg(bool sf, u32 op, u32 s, int rd, int rn, int rm, u32 shift, u32 imm6) {
        emit((sf ? (1u << 31) : 0) | (op << 30) | (s << 29) | (0b01011u << 24) | (shift << 22)
             | (static_cast<u32>(rm & 31) << 16) | ((imm6 & 63) << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31));
    }
    // "Add/subtract (extended register)" (64-bit).
    void addsub_ext(u32 sf, u32 op, u32 s, int rd, int rn, int rm, u32 option, u32 imm3) {
        emit((sf << 31) | (op << 30) | (s << 29) | (0b01011u << 24) | (1u << 21) | (static_cast<u32>(rm & 31) << 16)
             | (option << 13) | ((imm3 & 7) << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31));
    }
    // "Logical (shifted register)" with shift=0.
    void logical_reg(bool sf, u32 opc, u32 n, int rd, int rn, int rm) {
        emit((sf ? (1u << 31) : 0) | (opc << 29) | (0b01010u << 24) | (n << 21)
             | (static_cast<u32>(rm & 31) << 16) | (static_cast<u32>(rn & 31) << 5) | (rd & 31));
    }
    void logical_imm_raw(bool sf, u32 opc, int rd, int rn, u32 n, u32 immr, u32 imms) {
        emit((sf ? (1u << 31) : 0) | (opc << 29) | (0b100100u << 23) | (n << 22) | (immr << 16) | (imms << 10)
             | (static_cast<u32>(rn & 31) << 5) | (rd & 31));
    }
    void logical_imm(bool sf, u32 opc, int rd, int rn, u64 imm) {
        u32 n = 0, immr = 0, imms = 0;
        encode_logical_imm(imm, sf, n, immr, imms); // caller checked is_logical_imm()
        logical_imm_raw(sf, opc, rd, rn, n, immr, imms);
    }
    void logical_any(bool sf, u32 opc, int rd, int rn, u64 imm, int scratch) {
        u32 n, immr, imms;
        if (encode_logical_imm(imm, sf, n, immr, imms)) {
            logical_imm_raw(sf, opc, rd, rn, n, immr, imms);
        } else {
            mov_imm(scratch, imm, sf);
            logical_reg(sf, opc, 0, rd, rn, scratch);
        }
    }
    // "Add/subtract (immediate)".
    void addsub_imm(bool sf, u32 op, u32 s, int rd, int rn, u32 imm12, bool lsl12) {
        emit((sf ? (1u << 31) : 0) | (op << 30) | (s << 29) | (0b100010u << 23) | ((lsl12 ? 1u : 0u) << 22)
             | ((imm12 & 0xFFFu) << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31));
    }
    // "Bitfield" (SBFM/BFM/UBFM); N must equal sf.
    void bitfield(bool sf, u32 opc, int rd, int rn, u32 immr, u32 imms) {
        emit((sf ? (1u << 31) : 0) | (opc << 29) | (0b100110u << 23) | (sf ? (1u << 22) : 0) | ((immr & 63) << 16)
             | ((imms & 63) << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31));
    }
    // "Data-processing (2 source)" - LSLV/LSRV/ASRV/UDIV/SDIV.
    void dp2src(bool sf, u32 opcode6, int rd, int rn, int rm) {
        emit((sf ? (1u << 31) : 0) | (0b11010110u << 21) | (static_cast<u32>(rm & 31) << 16)
             | (opcode6 << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31));
    }
    // "Data-processing (3 source)" - MADD/MSUB/SMADDL/UMADDL/SMULH/UMULH.
    void dp3src(bool sf, u32 op31, u32 o0, int rd, int rn, int rm, int ra) {
        emit((sf ? (1u << 31) : 0) | (0b11011u << 24) | (op31 << 21) | (static_cast<u32>(rm & 31) << 16) | (o0 << 15)
             | (static_cast<u32>(ra & 31) << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31));
    }
    void condsel(bool sf, u32 op, u32 op2, int rd, int rn, int rm, Cond cond) {
        emit((sf ? (1u << 31) : 0) | (op << 30) | (0b11010100u << 21) | (static_cast<u32>(rm & 31) << 16)
             | (static_cast<u32>(cond) << 12) | (op2 << 10) | (static_cast<u32>(rn & 31) << 5) | (rd & 31));
    }
    static u32 tb_word(u32 base, int rt, u32 bit) {
        return base | ((bit >> 5) << 31) | ((bit & 31) << 19) | static_cast<u32>(rt & 31);
    }
    // "Floating-point data-processing (2 source)" - used for FADD/FSUB/FMUL/FDIV.
    void fp2src(bool dbl, u32 opcode4, int vd, int vn, int vm) {
        emit(0x1E200000u | (dbl ? (1u << 22) : 0) | (static_cast<u32>(vm & 31) << 16)
             | (opcode4 << 12) | (0b10u << 10) | (static_cast<u32>(vn & 31) << 5) | (vd & 31));
    }
    // "Floating-point data-processing (1 source)". `dbl` selects the *source*
    // precision (for FCVT this is what makes the two directions distinct).
    void fp1src(bool dbl, u32 opcode6, int vd, int vn) {
        emit(0x1E204000u | (dbl ? (1u << 22) : 0) | (opcode6 << 15) | (static_cast<u32>(vn & 31) << 5) | (vd & 31));
    }
    // "Conversion between floating-point and integer" (FCVT*S/SCVTF/...).
    void fcvt_int(bool int64, bool dbl, u32 rmode, u32 opcode3, int rd_or_vd, int rn_or_vn) {
        emit((int64 ? (1u << 31) : 0) | 0x1E200000u | (dbl ? (1u << 22) : 0) | (rmode << 19) | (opcode3 << 16)
             | (static_cast<u32>(rn_or_vn & 31) << 5) | (rd_or_vd & 31));
    }

    std::vector<u8> buf_;
};

} // namespace a64
