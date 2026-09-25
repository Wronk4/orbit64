#pragma once
// Guest instruction decoding and block-shape rules shared by the recompiler's
// driver (recompiler.cpp) and its per-architecture backends. Internal to
// src/jit - nothing outside the recompiler includes this.

#include "../common.hpp"
#include "jit_abi.hpp"

namespace jit_detail {

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
inline bool is_block_branch(const Decoded& d) {
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
inline bool is_interp_callable(const Decoded& d) {
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

inline u32 fetch_instr(const u8* rdram, u32 paddr) {
    return (static_cast<u32>(rdram[paddr]) << 24) | (static_cast<u32>(rdram[paddr + 1]) << 16) |
           (static_cast<u32>(rdram[paddr + 2]) << 8) | static_cast<u32>(rdram[paddr + 3]);
}

inline Decoded decode(u32 instr) {
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

// "BEQ/BNE rs, rt, . / NOP": the idle loop Emulator::skip_idle_loop()
// fast-forwards. Compiled code never chains into it, so the driver sees the
// loop come around and can skip it.
inline bool is_idle_loop_at(const u8* rdram, size_t rdram_size, u32 paddr) {
    if (static_cast<size_t>(paddr) + 8 > rdram_size) return false;
    const u32 br = fetch_instr(rdram, paddr);
    const u32 op = br >> 26;
    return (op == 0x04 || op == 0x05) && (br & 0xFFFF) == 0xFFFF && fetch_instr(rdram, paddr + 4) == 0;
}

} // namespace jit_detail
