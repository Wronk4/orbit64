#include "rsp_core.hpp"
#include "rsp.hpp"
#include "mi.hpp"
#include "rdp.hpp"

namespace {

// The reciprocal and inverse square root tables of the vector unit's divider.
struct DivTables {
    u16 rcp[512];
    u16 rsq[512];
    DivTables() {
        for (u32 i = 0; i < 512; ++i) {
            const u64 a = i + 512;
            const u64 b = (u64(1) << 34) / a;
            // Stored without the leading 1 (0x1xxxx); the first entry, 0x20000, is 0xFFFF in the ROM.
            rcp[i] = i == 0 ? 0xFFFF : static_cast<u16>((b + 1) >> 8);
        }
        for (u32 i = 0; i < 512; ++i) {
            const u64 a = (i + 512) >> ((i % 2 == 1) ? 1 : 0);
            u64 b = u64(1) << 17;
            // The largest b with b < 1 / sqrt(a) (in fixed point).
            while (a * (b + 1) * (b + 1) < (u64(1) << 44)) ++b;
            rsq[i] = static_cast<u16>(b >> 1);
        }
    }
};
const DivTables& div_tables() {
    static const DivTables t;
    return t;
}

inline s16 clamp_s16(s64 v) {
    return static_cast<s16>(v < -32768 ? -32768 : v > 32767 ? 32767 : v);
}

// Lane i of vt as operand element e selects it.
inline u32 elem_index(u32 e, u32 i) {
    if (e < 2) return i;
    if (e < 4) return (i & 6) | (e & 1);
    if (e < 8) return (i & 4) | (e & 3);
    return e & 7;
}

inline u8 dm_read(const u8* dmem, u32 a) { return dmem[a & 0xFFF]; }
inline void dm_write(u8* dmem, u32 a, u8 v) { dmem[a & 0xFFF] = v; }

} // namespace

void RspCore::reset() {
    std::memset(r, 0, sizeof r);
    std::memset(vr, 0, sizeof vr);
    std::memset(acc_h, 0, sizeof acc_h);
    std::memset(acc_m, 0, sizeof acc_m);
    std::memset(acc_l, 0, sizeof acc_l);
    vcol = vcoh = vccl = vcch = vce = 0;
    div_in = div_out = 0;
    div_dp = false;
    pc = 0;
    npc = 4;
}

s64 RspCore::acc(int i) const {
    const u64 v = (u64(acc_h[i]) << 32) | (u64(acc_m[i]) << 16) | acc_l[i];
    return static_cast<s64>(v << 16) >> 16; // 48-bit signed
}

void RspCore::set_acc(int i, s64 v) {
    acc_h[i] = static_cast<u16>(v >> 32);
    acc_m[i] = static_cast<u16>(v >> 16);
    acc_l[i] = static_cast<u16>(v);
}

u32 RspCore::run(u32 cycles, RSP& rsp, MI& mi, RDP& rdp, u8* rdram, size_t rdram_size) {
    u8* imem = rsp.get_imem();
    u8* dmem = rsp.get_dmem();
    u32 done = 0;
    while (done < cycles) {
        if (rsp.get_status() & SPStatus::HALT) break;
        ++done;
        const u32 cur = pc & 0xFFC;
        const u32 instr = (u32(imem[cur]) << 24) | (u32(imem[cur + 1]) << 16) | (u32(imem[cur + 2]) << 8) | imem[cur + 3];
        pc = npc & 0xFFC;
        npc = (pc + 4) & 0xFFC;

        const u32 op = instr >> 26;
        const u32 rs = (instr >> 21) & 31, rt = (instr >> 16) & 31, rd = (instr >> 11) & 31;
        const u32 sa = (instr >> 6) & 31;
        const u32 imm = instr & 0xFFFF;
        const s32 simm = static_cast<s16>(imm);
        auto branch = [&](bool taken) {
            if (taken) npc = (cur + 4 + (simm << 2)) & 0xFFC;
        };
        auto set = [&](u32 reg, u32 v) {
            if (reg) r[reg] = v;
        };
        const u32 addr = r[rs] + simm;

        switch (op) {
            case 0x00: // SPECIAL
                switch (instr & 63) {
                    case 0x00: set(rd, r[rt] << sa); break;                              // SLL
                    case 0x02: set(rd, r[rt] >> sa); break;                              // SRL
                    case 0x03: set(rd, static_cast<u32>(static_cast<s32>(r[rt]) >> sa)); break; // SRA
                    case 0x04: set(rd, r[rt] << (r[rs] & 31)); break;                    // SLLV
                    case 0x06: set(rd, r[rt] >> (r[rs] & 31)); break;                    // SRLV
                    case 0x07: set(rd, static_cast<u32>(static_cast<s32>(r[rt]) >> (r[rs] & 31))); break; // SRAV
                    case 0x08: npc = r[rs] & 0xFFC; break;                               // JR
                    case 0x09: {                                                          // JALR
                        const u32 target = r[rs] & 0xFFC;
                        set(rd, (cur + 8) & 0xFFC);
                        npc = target;
                        break;
                    }
                    case 0x0D: // BREAK
                        rsp.core_break(mi);
                        return done;
                    case 0x20: case 0x21: set(rd, r[rs] + r[rt]); break;               // ADD(U)
                    case 0x22: case 0x23: set(rd, r[rs] - r[rt]); break;               // SUB(U)
                    case 0x24: set(rd, r[rs] & r[rt]); break;
                    case 0x25: set(rd, r[rs] | r[rt]); break;
                    case 0x26: set(rd, r[rs] ^ r[rt]); break;
                    case 0x27: set(rd, ~(r[rs] | r[rt])); break;
                    case 0x2A: set(rd, static_cast<s32>(r[rs]) < static_cast<s32>(r[rt])); break;
                    case 0x2B: set(rd, r[rs] < r[rt]); break;
                    default: break;
                }
                break;
            case 0x01: { // REGIMM
                const bool lt = static_cast<s32>(r[rs]) < 0;
                switch (rt) {
                    case 0x00: branch(lt); break;                                   // BLTZ
                    case 0x01: branch(!lt); break;                                  // BGEZ
                    case 0x10: set(31, (cur + 8) & 0xFFC); branch(lt); break;       // BLTZAL
                    case 0x11: set(31, (cur + 8) & 0xFFC); branch(!lt); break;      // BGEZAL
                    default: break;
                }
                break;
            }
            case 0x02: npc = (instr << 2) & 0xFFC; break;                           // J
            case 0x03: set(31, (cur + 8) & 0xFFC); npc = (instr << 2) & 0xFFC; break; // JAL
            case 0x04: branch(r[rs] == r[rt]); break;
            case 0x05: branch(r[rs] != r[rt]); break;
            case 0x06: branch(static_cast<s32>(r[rs]) <= 0); break;
            case 0x07: branch(static_cast<s32>(r[rs]) > 0); break;
            case 0x08: case 0x09: set(rt, r[rs] + simm); break;                     // ADDI(U)
            case 0x0A: set(rt, static_cast<s32>(r[rs]) < simm); break;
            case 0x0B: set(rt, r[rs] < static_cast<u32>(simm)); break;
            case 0x0C: set(rt, r[rs] & imm); break;
            case 0x0D: set(rt, r[rs] | imm); break;
            case 0x0E: set(rt, r[rs] ^ imm); break;
            case 0x0F: set(rt, imm << 16); break;
            case 0x10: // COP0
                if (rs == 0x00) set(rt, rsp.cop0_read(rd & 15, rdp));
                else if (rs == 0x04) {
                    rsp.cop0_write(rd & 15, r[rt], mi, rdp, rdram, rdram_size);
                    if (rsp.get_status() & SPStatus::HALT) return done;
                }
                break;
            case 0x12: // COP2
                if (instr & (1u << 25)) {
                    vector_op(instr);
                } else {
                    const u32 e = (instr >> 7) & 15;
                    VReg& v = vr[rd];
                    switch (rs) {
                        case 0x00: // MFC2
                            set(rt, static_cast<u32>(static_cast<s16>((v.byte(e) << 8) | v.byte((e + 1) & 15))));
                            break;
                        case 0x02: // CFC2
                            if ((rd & 3) == 0) set(rt, static_cast<u32>(static_cast<s16>((vcoh << 8) | vcol)));
                            else if ((rd & 3) == 1) set(rt, static_cast<u32>(static_cast<s16>((vcch << 8) | vccl)));
                            else set(rt, vce);
                            break;
                        case 0x04: // MTC2
                            v.set_byte(e, static_cast<u8>(r[rt] >> 8));
                            if (e != 15) v.set_byte(e + 1, static_cast<u8>(r[rt]));
                            break;
                        case 0x06: // CTC2
                            if ((rd & 3) == 0) { vcol = static_cast<u8>(r[rt]); vcoh = static_cast<u8>(r[rt] >> 8); }
                            else if ((rd & 3) == 1) { vccl = static_cast<u8>(r[rt]); vcch = static_cast<u8>(r[rt] >> 8); }
                            else vce = static_cast<u8>(r[rt]);
                            break;
                        default: break;
                    }
                }
                break;
            case 0x20: set(rt, static_cast<u32>(static_cast<s8>(dm_read(dmem, addr)))); break; // LB
            case 0x21: set(rt, static_cast<u32>(static_cast<s16>((dm_read(dmem, addr) << 8) | dm_read(dmem, addr + 1)))); break;
            case 0x23: case 0x27: // LW (LWU)
                set(rt, (u32(dm_read(dmem, addr)) << 24) | (u32(dm_read(dmem, addr + 1)) << 16) |
                            (u32(dm_read(dmem, addr + 2)) << 8) | dm_read(dmem, addr + 3));
                break;
            case 0x24: set(rt, dm_read(dmem, addr)); break;                          // LBU
            case 0x25: set(rt, (dm_read(dmem, addr) << 8) | dm_read(dmem, addr + 1)); break; // LHU
            case 0x28: dm_write(dmem, addr, static_cast<u8>(r[rt])); break;          // SB
            case 0x29:                                                               // SH
                dm_write(dmem, addr, static_cast<u8>(r[rt] >> 8));
                dm_write(dmem, addr + 1, static_cast<u8>(r[rt]));
                break;
            case 0x2B:                                                               // SW
                dm_write(dmem, addr, static_cast<u8>(r[rt] >> 24));
                dm_write(dmem, addr + 1, static_cast<u8>(r[rt] >> 16));
                dm_write(dmem, addr + 2, static_cast<u8>(r[rt] >> 8));
                dm_write(dmem, addr + 3, static_cast<u8>(r[rt]));
                break;
            case 0x32: lwc2(instr, dmem); break;
            case 0x3A: swc2(instr, dmem); break;
            default: break;
        }
    }
    return done;
}

// --- Vector loads and stores --------------------------------------------------

void RspCore::lwc2(u32 instr, const u8* dmem) {
    const u32 base = (instr >> 21) & 31, vt = (instr >> 16) & 31, sub = (instr >> 11) & 31;
    const u32 e = (instr >> 7) & 15;
    const s32 off = static_cast<s32>(instr << 25) >> 25; // 7-bit signed
    VReg& v = vr[vt];
    auto load_bytes = [&](u32 size) {
        u32 a = r[base] + off * static_cast<s32>(size);
        for (u32 o = e; o < std::min<u32>(e + size, 16); ++o) v.set_byte(o, dm_read(dmem, a++));
    };
    switch (sub) {
        case 0x00: load_bytes(1); break; // LBV
        case 0x01: load_bytes(2); break; // LSV
        case 0x02: load_bytes(4); break; // LLV
        case 0x03: load_bytes(8); break; // LDV
        case 0x04: { // LQV
            u32 a = r[base] + off * 16;
            const u32 end = std::min<u32>(16, e + (16 - (a & 15)));
            for (u32 o = e; o < end; ++o) v.set_byte(o, dm_read(dmem, a++));
            break;
        }
        case 0x05: { // LRV
            u32 a = r[base] + off * 16;
            const u32 index = e + (16 - (a & 15));
            a &= ~15u;
            for (u32 o = index; o < 16; ++o) v.set_byte(o, dm_read(dmem, a++));
            break;
        }
        case 0x06: case 0x07: { // LPV, LUV
            u32 a = r[base] + off * 8;
            const u32 index = (a & 7) - e;
            a &= ~7u;
            const int shift = sub == 0x06 ? 8 : 7;
            for (u32 i = 0; i < 8; ++i) v.e[i] = static_cast<u16>(dm_read(dmem, a + ((index + i) & 15)) << shift);
            break;
        }
        case 0x08: { // LHV
            u32 a = r[base] + off * 16;
            const u32 index = (a & 7) - e;
            a &= ~7u;
            for (u32 i = 0; i < 8; ++i) v.e[i] = static_cast<u16>(dm_read(dmem, a + ((index + i * 2) & 15)) << 7);
            break;
        }
        case 0x09: { // LFV
            u32 a = r[base] + off * 16;
            const u32 index = (a & 7) - e;
            a &= ~7u;
            VReg tmp{};
            for (u32 i = 0; i < 4; ++i) {
                tmp.e[i] = static_cast<u16>(dm_read(dmem, a + ((index + i * 4) & 15)) << 7);
                tmp.e[i + 4] = static_cast<u16>(dm_read(dmem, a + ((index + i * 4 + 8) & 15)) << 7);
            }
            for (u32 o = e; o < std::min<u32>(e + 8, 16); ++o) v.set_byte(o, tmp.byte(o));
            break;
        }
        case 0x0A: { // LWV
            u32 a = r[base] + off * 16;
            for (u32 o = 16 - e; o < e + 16; ++o) {
                v.set_byte(o & 15, dm_read(dmem, a));
                a += 4;
            }
            break;
        }
        case 0x0B: { // LTV
            u32 a = r[base] + off * 16;
            const u32 begin = a & ~7u;
            a = begin + ((e + (a & 8)) & 15);
            const u32 vtbase = vt & ~7u;
            u32 vtoff = e >> 1;
            for (u32 i = 0; i < 8; ++i) {
                vr[vtbase + vtoff].set_byte(i * 2, dm_read(dmem, a++));
                if (a == begin + 16) a = begin;
                vr[vtbase + vtoff].set_byte(i * 2 + 1, dm_read(dmem, a++));
                if (a == begin + 16) a = begin;
                vtoff = (vtoff + 1) & 7;
            }
            break;
        }
        default: break;
    }
}

void RspCore::swc2(u32 instr, u8* dmem) {
    const u32 base = (instr >> 21) & 31, vt = (instr >> 16) & 31, sub = (instr >> 11) & 31;
    const u32 e = (instr >> 7) & 15;
    const s32 off = static_cast<s32>(instr << 25) >> 25;
    const VReg& v = vr[vt];
    auto store_bytes = [&](u32 size) {
        u32 a = r[base] + off * static_cast<s32>(size);
        for (u32 o = e; o < e + size; ++o) dm_write(dmem, a++, v.byte(o & 15));
    };
    switch (sub) {
        case 0x00: store_bytes(1); break; // SBV
        case 0x01: store_bytes(2); break; // SSV
        case 0x02: store_bytes(4); break; // SLV
        case 0x03: store_bytes(8); break; // SDV
        case 0x04: { // SQV
            u32 a = r[base] + off * 16;
            const u32 end = e + (16 - (a & 15));
            for (u32 o = e; o < end; ++o) dm_write(dmem, a++, v.byte(o & 15));
            break;
        }
        case 0x05: { // SRV
            u32 a = r[base] + off * 16;
            const u32 end = e + (a & 15);
            const u32 b = 16 - (a & 15);
            a &= ~15u;
            for (u32 o = e; o < end; ++o) dm_write(dmem, a++, v.byte((o + b) & 15));
            break;
        }
        case 0x06: case 0x07: { // SPV, SUV
            u32 a = r[base] + off * 8;
            for (u32 o = e; o < e + 8; ++o) {
                const bool packed = ((o & 15) < 8) == (sub == 0x06);
                dm_write(dmem, a++, packed ? v.byte((o & 7) << 1) : static_cast<u8>(v.e[o & 7] >> 7));
            }
            break;
        }
        case 0x08: { // SHV
            u32 a = r[base] + off * 16;
            const u32 index = a & 7;
            a &= ~7u;
            for (u32 i = 0; i < 8; ++i) {
                const u32 b = e + i * 2;
                const u8 value = static_cast<u8>((v.byte(b & 15) << 1) | (v.byte((b + 1) & 15) >> 7));
                dm_write(dmem, a + ((index + i * 2) & 15), value);
            }
            break;
        }
        case 0x09: { // SFV
            u32 a = r[base] + off * 16;
            const u32 b = a & 7;
            a &= ~7u;
            static const s8 table[16][4] = {
                {0, 1, 2, 3}, {6, 7, 4, 5}, {-1}, {-1}, {1, 2, 3, 0}, {7, 4, 5, 6}, {-1}, {-1},
                {4, 5, 6, 7}, {-1}, {-1}, {3, 0, 1, 2}, {5, 6, 7, 4}, {-1}, {-1}, {0, 1, 2, 3}};
            for (u32 i = 0; i < 4; ++i) {
                const s8 el = table[e][0] < 0 ? -1 : table[e][i];
                dm_write(dmem, a + ((b + (i << 2)) & 15), el < 0 ? 0 : static_cast<u8>(v.e[el] >> 7));
            }
            break;
        }
        case 0x0A: { // SWV
            u32 a = r[base] + off * 16;
            u32 b = a & 7;
            a &= ~7u;
            for (u32 o = e; o < e + 16; ++o) dm_write(dmem, a + (b++ & 15), v.byte(o & 15));
            break;
        }
        case 0x0B: { // STV
            u32 a = r[base] + off * 16;
            const u32 start = vt & ~7u;
            u32 element = 16 - (e & ~1u);
            u32 b = (a & 7) - (e & ~1u);
            a &= ~7u;
            for (u32 o = start; o < start + 8; ++o) {
                dm_write(dmem, a + (b++ & 15), vr[o].byte(element++ & 15));
                dm_write(dmem, a + (b++ & 15), vr[o].byte(element++ & 15));
            }
            break;
        }
        default: break;
    }
}

// --- Vector computational instructions ---------------------------------------

void RspCore::vector_op(u32 instr) {
    const u32 e = (instr >> 21) & 15;
    const u32 vt = (instr >> 16) & 31, vs = (instr >> 11) & 31, vdi = (instr >> 6) & 31;
    const u32 funct = instr & 63;
    const VReg s = vr[vs];
    VReg t;
    for (u32 i = 0; i < 8; ++i) t.e[i] = vr[vt].e[elem_index(e, i)];
    VReg d;

    auto carry = [&](u32 i) { return (vcol >> i) & 1; };
    auto ne = [&](u32 i) { return (vcoh >> i) & 1; };

    switch (funct) {
        case 0x00: case 0x01: case 0x08: case 0x09: { // VMULF VMULU VMACF VMACU
            const bool mac = funct >= 0x08, uns = funct & 1;
            for (int i = 0; i < 8; ++i) {
                const s64 prod = s64(static_cast<s16>(s.e[i])) * static_cast<s16>(t.e[i]) * 2;
                const s64 a = mac ? acc(i) + prod : prod + 0x8000;
                set_acc(i, a);
                if (!uns) {
                    d.e[i] = static_cast<u16>(clamp_s16(acc(i) >> 16));
                } else {
                    const s16 h = static_cast<s16>(acc_h[i]);
                    const s16 m = static_cast<s16>(acc_m[i]);
                    if (h < 0) d.e[i] = 0;
                    else if (mac ? (h != 0 || m < 0) : ((h ^ m) < 0)) d.e[i] = 0xFFFF;
                    else d.e[i] = acc_m[i];
                }
            }
            break;
        }
        case 0x02: case 0x0A: { // VRNDP VRNDN
            for (int i = 0; i < 8; ++i) {
                s64 prod = static_cast<s16>(t.e[i]);
                if (vs & 1) prod <<= 16;
                s64 a = acc(i);
                if (funct == 0x02 ? a >= 0 : a < 0) a += prod;
                set_acc(i, a);
                d.e[i] = static_cast<u16>(clamp_s16(acc(i) >> 16));
            }
            break;
        }
        case 0x03: { // VMULQ
            for (int i = 0; i < 8; ++i) {
                s32 prod = s32(static_cast<s16>(s.e[i])) * static_cast<s16>(t.e[i]);
                if (prod < 0) prod += 31;
                acc_h[i] = static_cast<u16>(prod >> 16);
                acc_m[i] = static_cast<u16>(prod);
                acc_l[i] = 0;
                d.e[i] = static_cast<u16>(clamp_s16(prod >> 1) & ~15);
            }
            break;
        }
        case 0x0B: { // VMACQ
            for (int i = 0; i < 8; ++i) {
                s32 prod = static_cast<s32>((u32(acc_h[i]) << 16) | acc_m[i]);
                if (prod < 0 && !(prod & (1 << 5))) prod += 32;
                else if (prod >= 32 && !(prod & (1 << 5))) prod -= 32;
                acc_h[i] = static_cast<u16>(prod >> 16);
                acc_m[i] = static_cast<u16>(prod);
                d.e[i] = static_cast<u16>(clamp_s16(prod >> 1) & ~15);
            }
            break;
        }
        case 0x04: case 0x0C: { // VMUDL VMADL
            for (int i = 0; i < 8; ++i) {
                const s64 prod = (u32(s.e[i]) * u32(t.e[i])) >> 16;
                set_acc(i, funct == 0x0C ? acc(i) + prod : prod);
                const s32 hm = static_cast<s32>((u32(acc_h[i]) << 16) | acc_m[i]);
                d.e[i] = hm < -32768 ? 0 : hm > 32767 ? 0xFFFF : acc_l[i];
            }
            break;
        }
        case 0x05: case 0x0D: { // VMUDM VMADM
            for (int i = 0; i < 8; ++i) {
                const s64 prod = s64(static_cast<s16>(s.e[i])) * t.e[i];
                set_acc(i, funct == 0x0D ? acc(i) + prod : prod);
                d.e[i] = static_cast<u16>(clamp_s16(acc(i) >> 16));
            }
            break;
        }
        case 0x06: case 0x0E: { // VMUDN VMADN
            for (int i = 0; i < 8; ++i) {
                const s64 prod = s64(s.e[i]) * static_cast<s16>(t.e[i]);
                set_acc(i, funct == 0x0E ? acc(i) + prod : prod);
                const s32 hm = static_cast<s32>((u32(acc_h[i]) << 16) | acc_m[i]);
                d.e[i] = hm < -32768 ? 0 : hm > 32767 ? 0xFFFF : acc_l[i];
            }
            break;
        }
        case 0x07: case 0x0F: { // VMUDH VMADH
            for (int i = 0; i < 8; ++i) {
                const s64 prod = (s64(static_cast<s16>(s.e[i])) * static_cast<s16>(t.e[i])) * 65536;
                set_acc(i, funct == 0x0F ? acc(i) + prod : prod);
                d.e[i] = static_cast<u16>(clamp_s16(acc(i) >> 16));
            }
            break;
        }
        case 0x10: case 0x11: { // VADD VSUB
            for (int i = 0; i < 8; ++i) {
                const s32 a = static_cast<s16>(s.e[i]), b = static_cast<s16>(t.e[i]);
                const s32 r = funct == 0x10 ? a + b + carry(i) : a - b - carry(i);
                acc_l[i] = static_cast<u16>(r);
                d.e[i] = static_cast<u16>(clamp_s16(r));
            }
            vcol = vcoh = 0;
            break;
        }
        case 0x13: { // VABS
            for (int i = 0; i < 8; ++i) {
                const s16 a = static_cast<s16>(s.e[i]), b = static_cast<s16>(t.e[i]);
                if (a < 0) {
                    if (b == -32768) { acc_l[i] = 0x8000; d.e[i] = 0x7FFF; }
                    else { acc_l[i] = d.e[i] = static_cast<u16>(-b); }
                } else if (a == 0) {
                    acc_l[i] = d.e[i] = 0;
                } else {
                    acc_l[i] = d.e[i] = t.e[i];
                }
            }
            break;
        }
        case 0x14: { // VADDC
            vcol = vcoh = 0;
            for (int i = 0; i < 8; ++i) {
                const u32 sum = u32(s.e[i]) + t.e[i];
                acc_l[i] = d.e[i] = static_cast<u16>(sum);
                vcol |= static_cast<u8>((sum >> 16) << i);
            }
            break;
        }
        case 0x15: { // VSUBC
            vcol = vcoh = 0;
            for (int i = 0; i < 8; ++i) {
                const s32 diff = s32(s.e[i]) - s32(t.e[i]);
                acc_l[i] = d.e[i] = static_cast<u16>(diff);
                vcol |= static_cast<u8>((diff < 0) << i);
                vcoh |= static_cast<u8>((diff != 0) << i);
            }
            break;
        }
        case 0x1D: { // VSAR
            for (int i = 0; i < 8; ++i)
                d.e[i] = e == 8 ? acc_h[i] : e == 9 ? acc_m[i] : e == 10 ? acc_l[i] : 0;
            break;
        }
        case 0x20: case 0x21: case 0x22: case 0x23: { // VLT VEQ VNE VGE
            u8 cc = 0;
            for (int i = 0; i < 8; ++i) {
                const s16 a = static_cast<s16>(s.e[i]), b = static_cast<s16>(t.e[i]);
                bool c;
                switch (funct) {
                    case 0x20: c = a < b || (a == b && carry(i) && ne(i)); break;
                    case 0x21: c = a == b && !ne(i); break;
                    case 0x22: c = a != b || ne(i); break;
                    default: c = a > b || (a == b && !(carry(i) && ne(i))); break;
                }
                cc |= static_cast<u8>(c << i);
                acc_l[i] = d.e[i] = c ? s.e[i] : t.e[i];
            }
            vccl = cc;
            vcch = 0;
            vcol = vcoh = 0;
            break;
        }
        case 0x24: { // VCL
            for (int i = 0; i < 8; ++i) {
                const u16 a = s.e[i], b = t.e[i];
                const u8 bit = static_cast<u8>(1 << i);
                if (carry(i)) {
                    if (!ne(i)) {
                        const u32 sum = u32(a) + b;
                        const bool lo_zero = static_cast<u16>(sum) == 0, cout = sum > 0xFFFF;
                        const bool le = (vce & bit) ? (lo_zero || !cout) : (lo_zero && !cout);
                        vccl = static_cast<u8>((vccl & ~bit) | (le ? bit : 0));
                    }
                    acc_l[i] = (vccl & bit) ? static_cast<u16>(-b) : a;
                } else {
                    if (!ne(i)) {
                        const bool ge = s32(a) - s32(b) >= 0;
                        vcch = static_cast<u8>((vcch & ~bit) | (ge ? bit : 0));
                    }
                    acc_l[i] = (vcch & bit) ? b : a;
                }
                d.e[i] = acc_l[i];
            }
            vcol = vcoh = 0;
            vce = 0;
            break;
        }
        case 0x25: { // VCH
            vcol = vcoh = vccl = vcch = vce = 0;
            for (int i = 0; i < 8; ++i) {
                const s16 a = static_cast<s16>(s.e[i]), b = static_cast<s16>(t.e[i]);
                bool le, ge, c, n, ce;
                if ((a ^ b) < 0) {
                    const s16 r = static_cast<s16>(a + b);
                    le = r <= 0;
                    ge = b < 0;
                    c = true;
                    n = r != 0 && static_cast<u16>(a) != static_cast<u16>(b ^ 0xFFFF);
                    ce = r == -1;
                    acc_l[i] = le ? static_cast<u16>(-b) : static_cast<u16>(a);
                } else {
                    const s16 r = static_cast<s16>(a - b);
                    le = b < 0;
                    ge = r >= 0;
                    c = false;
                    n = r != 0 && static_cast<u16>(a) != static_cast<u16>(b ^ 0xFFFF);
                    ce = false;
                    acc_l[i] = ge ? static_cast<u16>(b) : static_cast<u16>(a);
                }
                vccl |= static_cast<u8>(le << i);
                vcch |= static_cast<u8>(ge << i);
                vcol |= static_cast<u8>(c << i);
                vcoh |= static_cast<u8>(n << i);
                vce |= static_cast<u8>(ce << i);
                d.e[i] = acc_l[i];
            }
            break;
        }
        case 0x26: { // VCR
            vccl = vcch = 0;
            for (int i = 0; i < 8; ++i) {
                const s16 a = static_cast<s16>(s.e[i]), b = static_cast<s16>(t.e[i]);
                bool le, ge;
                if ((a ^ b) < 0) {
                    ge = b < 0;
                    le = s32(a) + s32(b) + 1 <= 0;
                    acc_l[i] = le ? static_cast<u16>(~b) : static_cast<u16>(a);
                } else {
                    le = b < 0;
                    ge = s32(a) - s32(b) >= 0;
                    acc_l[i] = ge ? static_cast<u16>(b) : static_cast<u16>(a);
                }
                vccl |= static_cast<u8>(le << i);
                vcch |= static_cast<u8>(ge << i);
                d.e[i] = acc_l[i];
            }
            vcol = vcoh = 0;
            vce = 0;
            break;
        }
        case 0x27: // VMRG
            for (int i = 0; i < 8; ++i) acc_l[i] = d.e[i] = ((vccl >> i) & 1) ? s.e[i] : t.e[i];
            vcol = vcoh = 0;
            break;
        case 0x28: for (int i = 0; i < 8; ++i) acc_l[i] = d.e[i] = s.e[i] & t.e[i]; break;
        case 0x29: for (int i = 0; i < 8; ++i) acc_l[i] = d.e[i] = static_cast<u16>(~(s.e[i] & t.e[i])); break;
        case 0x2A: for (int i = 0; i < 8; ++i) acc_l[i] = d.e[i] = s.e[i] | t.e[i]; break;
        case 0x2B: for (int i = 0; i < 8; ++i) acc_l[i] = d.e[i] = static_cast<u16>(~(s.e[i] | t.e[i])); break;
        case 0x2C: for (int i = 0; i < 8; ++i) acc_l[i] = d.e[i] = s.e[i] ^ t.e[i]; break;
        case 0x2D: for (int i = 0; i < 8; ++i) acc_l[i] = d.e[i] = static_cast<u16>(~(s.e[i] ^ t.e[i])); break;
        case 0x30: case 0x31: case 0x34: case 0x35: { // VRCP VRCPL VRSQ VRSQL
            const bool low = funct & 1, sqrt = funct >= 0x34;
            const u16 in16 = vr[vt].e[e & 7];
            const s32 input = (low && div_dp) ? static_cast<s32>((u32(static_cast<u16>(div_in)) << 16) | in16)
                                              : static_cast<s16>(in16);
            const s32 mask = input >> 31;
            s32 data = input ^ mask;
            if (input > -32768) data -= mask;
            s32 result;
            if (data == 0) {
                result = 0x7FFFFFFF;
            } else if (input == -32768) {
                result = static_cast<s32>(0xFFFF0000u);
            } else {
                const u32 shift = static_cast<u32>(__builtin_clz(static_cast<u32>(data)));
                const u32 index = static_cast<u32>((u64(static_cast<u32>(data)) << shift) & 0x7FC00000u) >> 22;
                u32 res;
                if (!sqrt) {
                    res = div_tables().rcp[index];
                    res = (0x10000 | res) << 14;
                    res = res >> (31 - shift);
                } else {
                    res = div_tables().rsq[(index & 0x1FE) | (shift & 1)];
                    res = (0x10000 | res) << 14;
                    res = res >> ((31 - shift) >> 1);
                }
                result = static_cast<s32>(res) ^ mask;
            }
            div_dp = false;
            div_out = static_cast<s16>(result >> 16);
            for (int i = 0; i < 8; ++i) acc_l[i] = t.e[i];
            d = vr[vdi];
            d.e[vs & 7] = static_cast<u16>(result);
            break;
        }
        case 0x32: case 0x36: // VRCPH VRSQH
            for (int i = 0; i < 8; ++i) acc_l[i] = t.e[i];
            div_dp = true;
            div_in = static_cast<s16>(vr[vt].e[e & 7]);
            d = vr[vdi];
            d.e[vs & 7] = static_cast<u16>(div_out);
            break;
        case 0x33: // VMOV
            for (int i = 0; i < 8; ++i) acc_l[i] = t.e[i];
            d = vr[vdi];
            d.e[vs & 7] = vr[vt].e[e & 7];
            break;
        case 0x37: // VNOP
        case 0x3F: // VNULL
            return;
        default: // reserved: the accumulator gets vs + vt, the result is 0
            for (int i = 0; i < 8; ++i) {
                acc_l[i] = static_cast<u16>(s.e[i] + t.e[i]);
                d.e[i] = 0;
            }
            break;
    }
    vr[vdi] = d;
}
