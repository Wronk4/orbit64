#pragma once

// Low-level RSP: an interpreter for the RSP's scalar unit (a MIPS R4000
// subset with 32-bit registers, 4 KB of IMEM and DMEM) and its vector unit
// (COP2: 32 registers of eight 16-bit lanes, a 48-bit accumulator per lane,
// the VCO/VCC/VCE flags and the reciprocal / inverse square root unit).
//
// It runs whatever microcode a game starts that the high-level emulation in
// rsp.cpp doesn't recognise (see RSP::start_task), or every task with
// ORBIT64_RSP=lle. The vector unit follows the behaviour documented by
// ares / cen64 and the hardware tests they were checked against.

#include "common.hpp"

class RSP;
class MI;
class RDP;

class RspCore {
public:
    struct VReg {
        u16 e[8];
        u8 byte(u32 b) const { return (b & 1) ? static_cast<u8>(e[(b >> 1) & 7]) : static_cast<u8>(e[(b >> 1) & 7] >> 8); }
        void set_byte(u32 b, u8 v) {
            u16& w = e[(b >> 1) & 7];
            w = (b & 1) ? static_cast<u16>((w & 0xFF00) | v) : static_cast<u16>((w & 0x00FF) | (v << 8));
        }
    };

    void reset();

    // Runs up to `cycles` instructions, or until the RSP halts (BREAK, or a
    // write to SP_STATUS). Returns the number of cycles used.
    u32 run(u32 cycles, RSP& rsp, MI& mi, RDP& rdp, u8* rdram, size_t rdram_size);

    u32 gpr(int i) const { return r[i & 31]; }
    const VReg& vreg(int i) const { return vr[i & 31]; }

    u32 pc{0};   // address of the next instruction to run
    u32 npc{4};  // and of the one after it (a branch target, in a delay slot)

    template <class S> void serialize(S& s) {
        s(r, pc, npc, vr, acc_h, acc_m, acc_l, vcol, vcoh, vccl, vcch, vce, div_in, div_out, div_dp);
    }

private:
    u32 r[32]{};
    VReg vr[32]{};
    u16 acc_h[8]{}, acc_m[8]{}, acc_l[8]{};
    // VCO: carry (low byte) / not-equal (high byte), VCC: compare / clip, VCE.
    u8 vcol{0}, vcoh{0}, vccl{0}, vcch{0}, vce{0};
    s16 div_in{0}, div_out{0};
    bool div_dp{false};

    void vector_op(u32 instr);
    void lwc2(u32 instr, const u8* dmem);
    void swc2(u32 instr, u8* dmem);

    s64 acc(int i) const;
    void set_acc(int i, s64 v);
};
