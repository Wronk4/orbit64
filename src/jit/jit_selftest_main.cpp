// Standalone correctness check for the recompiler: runs the same guest
// instruction stream through the interpreter and through the JIT from
// identical starting states and diffs every GPR/HI/LO. Not part of the app
// build (see build_jit_selftest.sh); intended to be compiled and run once
// per host architecture while working on the recompiler.
#include "../cpu.hpp"
#include "../bus.hpp"
#include "../cartridge.hpp"
#include "../pif.hpp"
#include "../controller.hpp"
#include "../mi.hpp"
#include "../vi.hpp"
#include "../ai.hpp"
#include "../pi.hpp"
#include "../si.hpp"
#include "../rsp.hpp"
#include "../rdp.hpp"
#include "recompiler.hpp"
#include <cstdio>
#include <vector>
#include <cstring>

namespace {

u32 rtype(u8 op, u8 rs, u8 rt, u8 rd, u8 shamt, u8 funct) {
    return (static_cast<u32>(op) << 26) | (static_cast<u32>(rs) << 21) | (static_cast<u32>(rt) << 16) |
           (static_cast<u32>(rd) << 11) | (static_cast<u32>(shamt) << 6) | funct;
}
u32 itype(u8 op, u8 rs, u8 rt, u16 imm) {
    return (static_cast<u32>(op) << 26) | (static_cast<u32>(rs) << 21) | (static_cast<u32>(rt) << 16) | imm;
}

struct World {
    Cartridge cart;
    PIF pif;
    Controller controllers[4];
    MI mi; VI vi; AI ai; PI pi; SI si; RSP rsp; RDP rdp;
    Bus bus;
    CPU cpu;
    World() : bus(cart, pif, controllers, mi, vi, ai, pi, si, rsp, rdp), cpu(bus) {}
};

constexpr u64 kEntry = 0x80001000ULL;
constexpr u32 kPaddr = 0x1000;

// Seed values chosen to exercise sign bits, 32/64-bit truncation and shifts.
const u64 kSeed[32] = {
    /*0*/ 0, /*1*/ 0x0000000080000000ULL, /*2*/ 0xFFFFFFFF80000000ULL, /*3*/ 5,
    /*4*/ 0xFFFFFFFFFFFFFFFFULL, /*5*/ 0x100000000ULL, /*6*/ 7, /*7*/ 0xFFFFFFFFFFFFFFF8ULL,
    /*8*/ 12345, /*9*/ 0x8000000000000000ULL, /*10*/ 2, /*11*/ 3,
    /*12*/ 0x7FFFFFFF, /*13*/ 0xFFFFFFFF80000001ULL, /*14*/ 42, /*15*/ 0,
    /*16*/ 0x123456789ABCDEF0ULL, /*17*/ 9, /*18*/ 0xFFFFFFFFFFFF0000ULL, /*19*/ 0x10,
    /*20*/ 7, /*21*/ 0xFFFFFFFF89ABCDEFULL, /*22*/ 0xFFFFFFFFFFFFFFFEULL, /*23*/ 100,
    /*24*/ 200, /*25*/ 0, /*26*/ 0xDEADBEEFULL, /*27*/ 0xCAFEBABEULL,
    /*28*/ 300, /*29*/ 0xFFFFFFFF7FFFFFFFULL, /*30*/ 0, /*31*/ 0x8033B400ULL
};

std::vector<u32> build_program() {
    std::vector<u32> p;
    auto R = [&](u8 op, u8 rs, u8 rt, u8 rd, u8 sh, u8 f) { p.push_back(rtype(op, rs, rt, rd, sh, f)); };
    auto I = [&](u8 op, u8 rs, u8 rt, u16 imm) { p.push_back(itype(op, rs, rt, imm)); };

    // Immediate ALU
    I(0x0F, 0, 8, 0x1234);        // LUI t0, 0x1234
    I(0x0D, 8, 8, 0x5678);        // ORI t0, t0, 0x5678
    I(0x09, 4, 9, static_cast<u16>(-100));  // ADDIU t1, a0, -100
    I(0x0C, 5, 10, 0x00FF);       // ANDI t2, a1, 0xFF
    I(0x0E, 6, 11, 0xFFFF);       // XORI t3, a2, 0xFFFF
    I(0x0A, 3, 12, static_cast<u16>(-1));   // SLTI t4, v1, -1
    I(0x0B, 3, 13, static_cast<u16>(-1));   // SLTIU t5, v1, -1
    I(0x19, 2, 14, static_cast<u16>(-7));   // DADDIU t6, v0, -7

    // Shifts (immediate)
    R(0x00, 0, 16, 15, 4, 0x00);  // SLL t7, s0, 4
    R(0x00, 0, 16, 15, 4, 0x02);  // SRL t7, s0, 4
    R(0x00, 0, 2, 15, 3, 0x03);   // SRA t7, v0, 3
    R(0x00, 0, 16, 15, 4, 0x38);  // DSLL t7, s0, 4
    R(0x00, 0, 16, 15, 4, 0x3A);  // DSRL t7, s0, 4
    R(0x00, 0, 4, 15, 3, 0x3B);   // DSRA t7, a0, 3
    R(0x00, 0, 16, 15, 4, 0x3C);  // DSLL32 t7, s0, 4
    R(0x00, 0, 16, 15, 4, 0x3E);  // DSRL32 t7, s0, 4
    R(0x00, 0, 4, 15, 3, 0x3F);   // DSRA32 t7, a0, 3

    // Shifts (variable)
    R(0x00, 20, 21, 22, 0, 0x04); // SLLV s6, s5, s4
    R(0x00, 20, 21, 22, 0, 0x06); // SRLV s6, s5, s4
    R(0x00, 20, 21, 22, 0, 0x07); // SRAV s6, s5, s4
    R(0x00, 20, 21, 22, 0, 0x14); // DSLLV s6, s5, s4
    R(0x00, 20, 21, 22, 0, 0x16); // DSRLV s6, s5, s4
    R(0x00, 20, 21, 22, 0, 0x17); // DSRAV s6, s5, s4

    // Trapping arithmetic (non-overflowing operands - the overflow path
    // itself is exercised separately in build_overflow_program()).
    I(0x08, 8, 24, 100);          // ADDI s8, t0, 100
    I(0x18, 8, 24, static_cast<u16>(-100)); // DADDI s8, t0, -100
    R(0x00, 8, 11, 24, 0, 0x20);  // ADD s8, t0, t3
    R(0x00, 8, 11, 24, 0, 0x22);  // SUB s8, t0, t3
    R(0x00, 8, 11, 24, 0, 0x2C);  // DADD s8, t0, t3
    R(0x00, 8, 11, 24, 0, 0x2E);  // DSUB s8, t0, t3

    // Register ALU
    R(0x00, 4, 5, 0, 0, 0x21);    // ADDU $0, a0, a1  (write to $zero must be a no-op)
    R(0x00, 4, 5, 23, 0, 0x21);   // ADDU s7, a0, a1
    R(0x00, 4, 5, 23, 0, 0x23);   // SUBU s7, a0, a1
    R(0x00, 4, 5, 23, 0, 0x24);   // AND
    R(0x00, 4, 5, 23, 0, 0x25);   // OR
    R(0x00, 4, 5, 23, 0, 0x26);   // XOR
    R(0x00, 4, 5, 23, 0, 0x27);   // NOR
    R(0x00, 4, 5, 23, 0, 0x2A);   // SLT
    R(0x00, 4, 5, 23, 0, 0x2B);   // SLTU
    R(0x00, 4, 5, 23, 0, 0x2D);   // DADDU
    R(0x00, 4, 5, 23, 0, 0x2F);   // DSUBU

    // MOVZ/MOVN (both taken and not taken)
    R(0x00, 6, 0, 24, 0, 0x0A);   // MOVZ s8, a2, $0   (rt=$0 -> taken)
    R(0x00, 6, 1, 24, 0, 0x0A);   // MOVZ s8, a2, at    (rt!=0 -> not taken)
    R(0x00, 6, 1, 25, 0, 0x0B);   // MOVN t9, a2, at    (rt!=0 -> taken)
    R(0x00, 6, 0, 25, 0, 0x0B);   // MOVN t9, a2, $0    (rt==0 -> not taken)

    // HI/LO
    R(0x00, 26, 0, 0, 0, 0x11);   // MTHI k0
    R(0x00, 0, 0, 27, 0, 0x10);   // MFHI k1
    R(0x00, 26, 0, 0, 0, 0x13);   // MTLO k0
    R(0x00, 0, 0, 27, 0, 0x12);   // MFLO k1

    // Loads/stores (round-trip through a scratch RDRAM address in a0-relative form)
    I(0x0F, 0, 4, 0x8000);        // LUI a0, 0x8000        (vaddr 0x80000000, well within RDRAM)
    I(0x09, 0, 5, 0x0C00);        // ADDIU a1, $0, 0xC00   (below program start, unused scratch)
    R(0x00, 4, 5, 4, 0, 0x21);    // ADDU a0, a0, a1  -> a0 = 0x80000C00
    I(0x29, 4, 8, 0);             // SH t0, 0(a0)
    I(0x21, 4, 28, 0);            // LH gp, 0(a0)
    I(0x25, 4, 29, 0);            // LHU sp, 0(a0)
    I(0x28, 4, 8, 4);             // SB t0, 4(a0)
    I(0x20, 4, 30, 4);            // LB fp, 4(a0)
    I(0x24, 4, 31, 4);            // LBU ra, 4(a0)
    I(0x2B, 4, 9, 8);             // SW t1, 8(a0)
    I(0x23, 4, 20, 8);            // LW s4, 8(a0)
    I(0x27, 4, 21, 8);            // LWU s5, 8(a0)
    I(0x3F, 4, 16, 16);           // SD s0, 16(a0)
    I(0x37, 4, 22, 16);           // LD s6, 16(a0)

    // FPU: build known bit patterns in GPRs, move into FPU registers, do
    // arithmetic, move results back out to GPRs for comparison (State only
    // captures GPR/HI/LO/PC, not FPRs).
    I(0x0F, 0, 20, 0x3F80); I(0x0D, 20, 20, 0x0000);           // t8=1.0f bits (0x3F800000)
    I(0x0F, 0, 21, 0x4000); I(0x0D, 21, 21, 0x0000);           // t9=2.0f bits (0x40000000)
    R(0x11, 0x04, 20, 0, 0, 0);   // MTC1 t8, f0   (fmt-field=rs=0x04 selects MTC1; fd encoded in "fs"=rd)
    R(0x11, 0x04, 21, 1, 0, 0);   // MTC1 t9, f1
    R(0x11, 0x10, 1, 0, 2, 0x00); // ADD.S f2, f0, f1  -> 3.0f
    R(0x11, 0x10, 0, 1, 3, 0x01); // SUB.S f3, f1, f0  -> 1.0f
    R(0x11, 0x10, 1, 0, 4, 0x02); // MUL.S f4, f0, f1  -> 2.0f
    R(0x11, 0x10, 0, 2, 5, 0x06); // MOV.S f5, f2      -> 3.0f
    R(0x11, 0x10, 0, 0, 6, 0x07); // NEG.S f6, f0      -> -1.0f
    R(0x11, 0x10, 0, 6, 7, 0x05); // ABS.S f7, f6      -> 1.0f
    R(0x11, 0x04, 0, 8, 0, 0);    // MTC1 $0, f8 -> f8 = 0.0f bits
    R(0x11, 0x10, 0, 1, 19, 0x03); // DIV.S f19, f1, f0  -> 2.0/1.0 = 2.0f
    R(0x11, 0x10, 8, 0, 20, 0x03); // DIV.S f20, f0, f8  -> 1.0/0.0 -> 0.0f (special-cased, not Inf)
    R(0x11, 0x10, 0, 4, 21, 0x04); // SQRT.S f21, f4     -> sqrt(2.0f)
    R(0x11, 0x10, 0, 21, 22, 0x24); // CVT.W.S f22, f21  -> (int)sqrt(2.0f) == 1
    R(0x11, 0x10, 0, 0, 23, 0x21);  // CVT.D.S f23, f0   -> 1.0f -> 1.0 double
    R(0x11, 0x10, 0, 0, 9, 0x32);   // C.EQ.S f0,f0 (true: equal to itself)
    R(0x11, 0x02, 26, 0, 0, 0);     // CFC1 k0, (fcsr) - interpreter fallback, reads what C.EQ.S just set
    R(0x11, 0x00, 22, 2, 0, 0);   // MFC1 s6, f2
    R(0x11, 0x00, 23, 3, 0, 0);   // MFC1 t7, f3
    R(0x11, 0x00, 24, 4, 0, 0);   // MFC1 t8, f4
    R(0x11, 0x00, 25, 5, 0, 0);   // MFC1 t9, f5
    R(0x11, 0x00, 27, 7, 0, 0);   // MFC1 k1, f7
    R(0x11, 0x00, 6, 19, 0, 0);   // MFC1 a2, f19 (DIV.S result)
    R(0x11, 0x00, 7, 20, 0, 0);   // MFC1 a3(t0 slot reused via reg7), f20 (DIV.S/0 result)
    R(0x11, 0x00, 8, 22, 0, 0);   // MFC1 t0, f22 (CVT.W.S result)

    I(0x0F, 0, 28, 0x3FF0);                                    // gp = 0x3FF00000
    R(0x00, 0, 28, 28, 0, 0x3C);                               // DSLL32 gp,gp,0 -> 1.0 double bits
    I(0x0F, 0, 29, 0x4000);                                    // sp = 0x40000000
    R(0x00, 0, 29, 29, 0, 0x3C);                               // DSLL32 sp,sp,0 -> 2.0 double bits
    R(0x11, 0x05, 28, 10, 0, 0);  // DMTC1 gp, f10
    R(0x11, 0x05, 29, 11, 0, 0);  // DMTC1 sp, f11
    R(0x11, 0x11, 11, 10, 12, 0x00); // ADD.D f12, f10, f11 -> 3.0
    R(0x11, 0x11, 10, 11, 13, 0x01); // SUB.D f13, f11, f10 -> 1.0
    R(0x11, 0x11, 11, 10, 14, 0x02); // MUL.D f14, f10, f11 -> 2.0
    R(0x11, 0x11, 0, 12, 15, 0x06);  // MOV.D f15, f12      -> 3.0
    R(0x11, 0x11, 0, 10, 17, 0x07);  // NEG.D f17, f10      -> -1.0
    R(0x11, 0x11, 0, 17, 18, 0x05);  // ABS.D f18, f17      -> 1.0
    R(0x11, 0x05, 0, 25, 0, 0);      // DMTC1 $0, f25 -> f25 = 0.0 double bits
    R(0x11, 0x11, 11, 10, 26, 0x03); // DIV.D f26, f10, f11 -> 1.0/2.0 = 0.5
    R(0x11, 0x11, 25, 10, 27, 0x03); // DIV.D f27, f10, f25 -> 1.0/0.0 -> 0.0 (special-cased)
    R(0x11, 0x11, 0, 11, 28, 0x04);  // SQRT.D f28, f11     -> sqrt(2.0)
    R(0x11, 0x11, 0, 12, 29, 0x20);  // CVT.S.D f29, f12    -> 3.0 double -> 3.0f
    R(0x11, 0x11, 0, 12, 30, 0x24);  // CVT.W.D f30, f12    -> (int)3.0 == 3
    R(0x11, 0x01, 28, 12, 0, 0);  // DMFC1 gp, f12
    R(0x11, 0x01, 29, 13, 0, 0);  // DMFC1 sp, f13
    R(0x11, 0x01, 30, 14, 0, 0);  // DMFC1 fp, f14
    R(0x11, 0x01, 31, 15, 0, 0);  // DMFC1 ra, f15
    R(0x11, 0x01, 9, 26, 0, 0);   // DMFC1 t1, f26 (DIV.D result)
    R(0x11, 0x01, 10, 27, 0, 0);  // DMFC1 t2, f27 (DIV.D/0 result)
    R(0x11, 0x00, 11, 30, 0, 0);  // MFC1 t3, f30 (CVT.W.D result, 32-bit)
    R(0x11, 0x01, 4, 17, 0, 0);   // DMFC1 a0, f17
    R(0x11, 0x01, 5, 18, 0, 0);   // DMFC1 a1, f18

    // Integer -> float conversions (the ARM64 backend once stored the
    // unconverted integer bits here: SCVTF leaves its result in an FP register).
    I(0x09, 0, 12, static_cast<u16>(-7));  // ADDIU t4, $0, -7
    R(0x11, 0x04, 12, 16, 0, 0);            // MTC1 t4, f16
    R(0x11, 0x14, 0, 16, 24, 0x20);         // CVT.S.W f24, f16 -> -7.0f
    R(0x11, 0x14, 0, 16, 31, 0x21);         // CVT.D.W f31, f16 -> -7.0
    R(0x11, 0x00, 13, 24, 0, 0);            // MFC1 t5, f24
    R(0x11, 0x01, 14, 31, 0, 0);            // DMFC1 t6, f31
    I(0x09, 0, 15, static_cast<u16>(-3));  // ADDIU t7, $0, -3
    R(0x00, 0, 15, 15, 0, 0x3C);            // DSLL32 t7, t7, 0 -> -3 * 2^32 (needs all 64 bits)
    R(0x11, 0x05, 15, 16, 0, 0);            // DMTC1 t7, f16
    R(0x11, 0x15, 0, 16, 24, 0x20);         // CVT.S.L f24, f16
    R(0x11, 0x15, 0, 16, 31, 0x21);         // CVT.D.L f31, f16
    R(0x11, 0x00, 1, 24, 0, 0);             // MFC1 at, f24
    R(0x11, 0x01, 2, 31, 0, 0);             // DMFC1 v0, f31

    return p;
}

void write_word(u8* rd, u32 addr, u32 v) {
    rd[addr + 0] = (v >> 24) & 0xFF;
    rd[addr + 1] = (v >> 16) & 0xFF;
    rd[addr + 2] = (v >> 8) & 0xFF;
    rd[addr + 3] = v & 0xFF;
}

void write_program(World& w, const std::vector<u32>& prog) {
    u8* rd = w.bus.get_rdram();
    for (size_t i = 0; i < prog.size(); ++i) {
        write_word(rd, kPaddr + static_cast<u32>(i) * 4, prog[i]);
    }
    // BREAK - not in the recompiler's whitelist, so it deterministically ends
    // the compiled block exactly at the end of the program instead of
    // running on into whatever (zero-initialized, i.e. all-NOP) RDRAM
    // follows it.
    write_word(rd, kPaddr + static_cast<u32>(prog.size()) * 4, rtype(0, 0, 0, 0, 0, 0x0D));
}

void seed_regs(CPU& cpu) {
    for (int i = 1; i < 32; ++i) cpu.set_gpr(i, kSeed[i]);
}

struct State {
    u64 gpr[32];
    u64 hi, lo, pc;
};

State capture(CPU& cpu) {
    State s{};
    for (int i = 0; i < 32; ++i) s.gpr[i] = cpu.get_gpr(i);
    s.hi = cpu.get_hi(); s.lo = cpu.get_lo(); s.pc = cpu.get_pc();
    return s;
}

std::vector<u32> build_overflow_program() {
    std::vector<u32> p;
    // ADDI t1, t4, 1 - t4 seeded to 0x7FFFFFFF (INT32_MAX), so this
    // overflows 32-bit signed addition and must raise EXC_OV.
    p.push_back(itype(0x08, 12, 9, 1));
    return p;
}

// Runs `prog` for exactly one core-step (interpreter, or as many JIT
// instructions as one run_step call covers) and returns the resulting state.
// Used for the overflow check, where the trap redirects pc to the exception
// vector after a single instruction - there's no fixed instruction count to
// loop for the way build_program()'s straight-line test has.
State run_one_and_capture(World& w, const std::vector<u32>& prog, bool use_jit) {
    write_program(w, prog);
    w.cpu.reset(static_cast<u32>(kEntry));
    w.cpu.set_pc(kEntry);
    seed_regs(w.cpu);
    if (use_jit) {
        Recompiler rec;
        rec.run_step(w.cpu, w.bus);
    } else {
        w.cpu.step();
    }
    return capture(w.cpu);
}

} // namespace

int main() {
    auto prog = build_program();

    World interp;
    write_program(interp, prog);
    interp.cpu.reset(static_cast<u32>(kEntry));
    interp.cpu.set_pc(kEntry);
    seed_regs(interp.cpu);
    for (size_t i = 0; i < prog.size(); ++i) interp.cpu.step();
    State a = capture(interp.cpu);

    World jitw;
    write_program(jitw, prog);
    jitw.cpu.reset(static_cast<u32>(kEntry));
    jitw.cpu.set_pc(kEntry);
    seed_regs(jitw.cpu);
    Recompiler rec;
    u64 target_pc = kEntry + prog.size() * 4;
    int guard = 0;
    while (jitw.cpu.get_pc() != target_pc && guard++ < 10000) {
        rec.run_step(jitw.cpu, jitw.bus);
    }
    State b = capture(jitw.cpu);

    bool ok = true;
    for (int i = 0; i < 32; ++i) {
        if (a.gpr[i] != b.gpr[i]) {
            std::printf("MISMATCH gpr[%d]: interp=0x%016llx jit=0x%016llx\n", i,
                        static_cast<unsigned long long>(a.gpr[i]), static_cast<unsigned long long>(b.gpr[i]));
            ok = false;
        }
    }
    if (a.hi != b.hi) { std::printf("MISMATCH hi: interp=0x%016llx jit=0x%016llx\n", (unsigned long long)a.hi, (unsigned long long)b.hi); ok = false; }
    if (a.lo != b.lo) { std::printf("MISMATCH lo: interp=0x%016llx jit=0x%016llx\n", (unsigned long long)a.lo, (unsigned long long)b.lo); ok = false; }
    if (a.pc != b.pc) { std::printf("MISMATCH pc: interp=0x%016llx jit=0x%016llx\n", (unsigned long long)a.pc, (unsigned long long)b.pc); ok = false; }

    // Overflow trap: ADDI whose 32-bit result overflows must raise EXC_OV
    // identically (same exception vector, CAUSE, EPC) whether it was
    // interpreted or compiled.
    {
        auto oprog = build_overflow_program();
        World oi, oj;
        State si = run_one_and_capture(oi, oprog, false);
        State sj = run_one_and_capture(oj, oprog, true);
        bool oflow_ok = si.pc == sj.pc;
        u64 cause_i = oi.cpu.get_cp0(CP0Reg::CAUSE), cause_j = oj.cpu.get_cp0(CP0Reg::CAUSE);
        u64 epc_i = oi.cpu.get_cp0(CP0Reg::EPC), epc_j = oj.cpu.get_cp0(CP0Reg::EPC);
        if (cause_i != cause_j) { std::printf("MISMATCH overflow CAUSE: interp=0x%llx jit=0x%llx\n", (unsigned long long)cause_i, (unsigned long long)cause_j); oflow_ok = false; }
        if (epc_i != epc_j) { std::printf("MISMATCH overflow EPC: interp=0x%llx jit=0x%llx\n", (unsigned long long)epc_i, (unsigned long long)epc_j); oflow_ok = false; }
        if (si.pc != sj.pc) { std::printf("MISMATCH overflow pc: interp=0x%llx jit=0x%llx\n", (unsigned long long)si.pc, (unsigned long long)sj.pc); oflow_ok = false; }
        // gpr[9] must NOT have been written (the overflowing ADDI's
        // destination is discarded on trap, same as the interpreter).
        if (oi.cpu.get_gpr(9) != oj.cpu.get_gpr(9)) { std::printf("MISMATCH overflow gpr9\n"); oflow_ok = false; }
        if (!oflow_ok) ok = false;
        else std::printf("OK: overflow trap matches (pc=0x%llx).\n", (unsigned long long)si.pc);
    }

    if (ok) { std::printf("OK: interpreter and JIT agree on all %d instructions.\n", (int)prog.size()); return 0; }
    return 1;
}
