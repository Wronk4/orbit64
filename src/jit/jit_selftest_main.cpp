// Standalone correctness check for the recompiler: runs the same guest
// programs through the interpreter and through the JIT from identical
// starting states and diffs the resulting machine state (GPRs, HI/LO, pc,
// the exception registers and a scratch area of RDRAM). Not part of the app
// build (`make jit_selftest`); run it whenever the recompiler changes.
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
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {

u32 rtype(u8 op, u8 rs, u8 rt, u8 rd, u8 shamt, u8 funct) {
    return (static_cast<u32>(op) << 26) | (static_cast<u32>(rs) << 21) | (static_cast<u32>(rt) << 16) |
           (static_cast<u32>(rd) << 11) | (static_cast<u32>(shamt) << 6) | funct;
}
u32 itype(u8 op, u8 rs, u8 rt, u16 imm) {
    return (static_cast<u32>(op) << 26) | (static_cast<u32>(rs) << 21) | (static_cast<u32>(rt) << 16) | imm;
}
u32 jtype(u8 op, u32 target_addr) { return (static_cast<u32>(op) << 26) | ((target_addr >> 2) & 0x03FFFFFFu); }

constexpr u32 kBreak = 0x0000000Du;
constexpr u32 kProgPaddr = 0x1F00; // programs run across the 4 KB page boundary at 0x2000
constexpr u32 kScratchPaddr = 0x0C00; // below the program; stores land here
constexpr u32 kScratchLen = 0x100;
constexpr u32 kStrayPaddr = kProgPaddr + 4 * 200; // past any program
// Virtual address TLB entry 0 maps to physical 0 (so kProgPaddr runs at
// kMappedBase + kProgPaddr); KUSEG, so it is reached only through the TLB.
constexpr u64 kMappedBase = 0x00400000;

struct World {
    Cartridge cart;
    PIF pif;
    Controller controllers[4];
    MI mi; VI vi; AI ai; PI pi; SI si; RSP rsp; RDP rdp;
    Bus bus;
    CPU cpu;
    World() : bus(cart, pif, controllers, mi, vi, ai, pi, si, rsp, rdp), cpu(bus) {}
};

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

struct State {
    u64 gpr[32];
    u64 fpr[32];
    u64 hi, lo, pc;
    u64 epc, cause, badvaddr, count, random, status;
    u32 fcsr;
    u8 scratch[kScratchLen];
};

State capture(World& w) {
    State s{};
    for (int i = 0; i < 32; ++i) s.gpr[i] = w.cpu.get_gpr(i);
    s.hi = w.cpu.get_hi();
    s.lo = w.cpu.get_lo();
    s.pc = w.cpu.get_pc();
    s.epc = w.cpu.get_cp0(CP0Reg::EPC);
    s.cause = w.cpu.get_cp0(CP0Reg::CAUSE) & 0x8000007CULL; // BD + ExcCode (IP bits depend on timing)
    s.badvaddr = w.cpu.get_cp0(CP0Reg::BAD_VADDR);
    s.count = w.cpu.get_cp0(CP0Reg::COUNT);
    s.random = w.cpu.get_cp0(CP0Reg::RANDOM);
    s.status = w.cpu.get_cp0(CP0Reg::STATUS);
    for (int i = 0; i < 32; ++i) s.fpr[i] = w.cpu.get_fpr_raw(i);
    s.fcsr = w.cpu.get_fcsr();
    std::memcpy(s.scratch, w.bus.get_rdram() + kScratchPaddr, kScratchLen);
    return s;
}

void write_word(u8* rd, u32 addr, u32 v) {
    rd[addr + 0] = (v >> 24) & 0xFF;
    rd[addr + 1] = (v >> 16) & 0xFF;
    rd[addr + 2] = (v >> 8) & 0xFF;
    rd[addr + 3] = v & 0xFF;
}

// How a program is run: on the interpreter, one compiled block per call
// (Recompiler::run_step), or with blocks chained into each other for as long
// as possible (Recompiler::run with a large budget).
enum class Mode { Interp, JitStep, JitChain };

// Runs `prog` from `entry` (a KSEG0 or KSEG1 address of kProgPaddr) until pc
// reaches the instruction right after the program, or either exception
// vector. Every one of those holds a BREAK, which is never reached/run.
State run(const std::vector<u32>& prog, u64 entry, Mode mode, bool* timed_out, const u64* seeds = kSeed,
          u64 status = 0x34000000, const u64* fpr_seeds = nullptr) {
    World w;
    u8* rd = w.bus.get_rdram();
    for (size_t i = 0; i < prog.size(); ++i) write_word(rd, kProgPaddr + static_cast<u32>(i) * 4, prog[i]);
    write_word(rd, kProgPaddr + static_cast<u32>(prog.size()) * 4, kBreak);
    write_word(rd, kStrayPaddr, kBreak); // where a random program's stray JR lands
    write_word(rd, 0x000, kBreak); // TLB refill vector
    write_word(rd, 0x180, kBreak); // general exception vector
    w.cpu.reset(static_cast<u32>(entry));
    w.cpu.set_pc(entry);
    // TLB entries 0 and 1 (wired) map kMappedBase..+16K to physical 0..16K, so
    // a program can also run TLB-mapped (see kMappedBase).
    TLBEntry map{};
    map.page_mask = 0;
    map.entry_hi = kMappedBase;
    map.entry_lo0 = (0x0000u >> 12 << 6) | 0x7; // PFN 0, dirty, valid, global
    map.entry_lo1 = (0x1000u >> 12 << 6) | 0x7; // PFN 1
    map.initialized = true;
    w.bus.set_tlb_entry(0, map);
    map.entry_hi = kMappedBase + 0x2000; // ... and the next 8K to physical 8K..16K
    map.entry_lo0 = (0x2000u >> 12 << 6) | 0x7;
    map.entry_lo1 = (0x3000u >> 12 << 6) | 0x7;
    w.bus.set_tlb_entry(1, map);
    w.cpu.set_cp0(CP0Reg::WIRED, 2);
    w.cpu.set_cp0(CP0Reg::RANDOM, 31);
    for (int i = 1; i < 32; ++i) w.cpu.set_gpr(i, seeds[i]);
    w.cpu.set_cp0(CP0Reg::STATUS, status);
    if (fpr_seeds) {
        // FPU registers only become reachable through MTC1/DMTC1, so they are
        // seeded by running those on the interpreter first.
        for (int i = 0; i < 32; ++i) {
            w.cpu.set_gpr(1, fpr_seeds[i]);
            const u64 pc = w.cpu.get_pc();
            write_word(rd, 0x3F00, rtype(0x11, (status & (1u << 26)) ? 0x05 : 0x04, 1, static_cast<u8>(i), 0, 0));
            w.cpu.set_pc(0xFFFFFFFF80003F00ULL);
            w.cpu.step();
            w.cpu.set_pc(pc);
        }
        w.cpu.set_gpr(1, seeds[1]);
        w.cpu.set_cp0(CP0Reg::COUNT, 0);
        w.cpu.set_cp0(CP0Reg::RANDOM, 31);
    }

    const u64 exit_pc = entry + prog.size() * 4;
    auto stopped = [&](u64 pc) {
        const u32 pc32 = static_cast<u32>(pc);
        return pc == exit_pc || pc32 == 0x80000180u || pc32 == 0x80000000u;
    };
    Recompiler rec;
    u64 cycles = 0; // every program here ends well within this
    constexpr u64 kLimit = 200000;
    while (!stopped(w.cpu.get_pc()) && cycles < kLimit) {
        switch (mode) {
        case Mode::Interp: cycles += w.cpu.step(); break;
        case Mode::JitStep: cycles += rec.run_step(w.cpu, w.bus); break;
        case Mode::JitChain: cycles += rec.run(w.cpu, w.bus, 1u << 16); break;
        }
    }
    *timed_out = cycles >= kLimit;
    return capture(w);
}

bool compare(const char* name, const State& a, const State& b) {
    bool ok = true;
    auto diff = [&](const char* what, u64 x, u64 y) {
        if (x == y) return;
        std::printf("  [%s] MISMATCH %s: interp=0x%016llx jit=0x%016llx\n", name, what,
                    static_cast<unsigned long long>(x), static_cast<unsigned long long>(y));
        ok = false;
    };
    for (int i = 0; i < 32; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "gpr[%d]", i);
        diff(buf, a.gpr[i], b.gpr[i]);
    }
    diff("hi", a.hi, b.hi);
    diff("lo", a.lo, b.lo);
    diff("pc", a.pc, b.pc);
    diff("EPC", a.epc, b.epc);
    diff("CAUSE", a.cause, b.cause);
    diff("BadVAddr", a.badvaddr, b.badvaddr);
    diff("COUNT", a.count, b.count);
    diff("RANDOM", a.random, b.random);
    diff("STATUS", a.status, b.status);
    diff("FCSR", a.fcsr, b.fcsr);
    for (int i = 0; i < 32; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "fpr[%d]", i);
        diff(buf, a.fpr[i], b.fpr[i]);
    }
    if (std::memcmp(a.scratch, b.scratch, kScratchLen) != 0) {
        std::printf("  [%s] MISMATCH scratch RDRAM\n", name);
        ok = false;
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------
using Prog = std::vector<u32>;
// Helpers appending common instructions.
void addiu(Prog& p, u8 rt, u8 rs, s16 imm) { p.push_back(itype(0x09, rs, rt, static_cast<u16>(imm))); }
void li32(Prog& p, u8 rt, u32 v) {
    p.push_back(itype(0x0F, 0, rt, static_cast<u16>(v >> 16)));
    p.push_back(itype(0x0D, rt, rt, static_cast<u16>(v & 0xFFFF)));
}

Prog alu_program() {
    Prog p;
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

    // Trapping arithmetic (non-overflowing operands).
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

    // Loads/stores (round-trip through the scratch area)
    I(0x0F, 0, 4, 0x8000);        // LUI a0, 0x8000
    I(0x09, 0, 5, kScratchPaddr); // ADDIU a1, $0, scratch
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
    // arithmetic, move results back out to GPRs for comparison.
    I(0x0F, 0, 20, 0x3F80); I(0x0D, 20, 20, 0x0000);           // 1.0f bits
    I(0x0F, 0, 21, 0x4000); I(0x0D, 21, 21, 0x0000);           // 2.0f bits
    R(0x11, 0x04, 20, 0, 0, 0);   // MTC1 s4, f0
    R(0x11, 0x04, 21, 1, 0, 0);   // MTC1 s5, f1
    R(0x11, 0x10, 1, 0, 2, 0x00); // ADD.S f2, f0, f1  -> 3.0f
    R(0x11, 0x10, 0, 1, 3, 0x01); // SUB.S f3, f1, f0  -> 1.0f
    R(0x11, 0x10, 1, 0, 4, 0x02); // MUL.S f4, f0, f1  -> 2.0f
    R(0x11, 0x10, 0, 2, 5, 0x06); // MOV.S f5, f2      -> 3.0f
    R(0x11, 0x10, 0, 0, 6, 0x07); // NEG.S f6, f0      -> -1.0f
    R(0x11, 0x10, 0, 6, 7, 0x05); // ABS.S f7, f6      -> 1.0f
    R(0x11, 0x04, 0, 8, 0, 0);    // MTC1 $0, f8 -> 0.0f
    R(0x11, 0x10, 0, 1, 19, 0x03); // DIV.S f19, f1, f0  -> 2.0f
    R(0x11, 0x10, 8, 0, 20, 0x03); // DIV.S f20, f0, f8  -> 1.0/0.0 -> 0.0f (special-cased, not Inf)
    R(0x11, 0x10, 0, 4, 21, 0x04); // SQRT.S f21, f4
    R(0x11, 0x10, 0, 21, 22, 0x24); // CVT.W.S f22, f21
    R(0x11, 0x10, 0, 0, 23, 0x21);  // CVT.D.S f23, f0
    R(0x11, 0x10, 0, 0, 9, 0x32);   // C.EQ.S f0,f0
    R(0x11, 0x02, 26, 0, 0, 0);     // CFC1 k0, fcsr (interpreter)
    R(0x11, 0x00, 22, 2, 0, 0);   // MFC1 s6, f2
    R(0x11, 0x00, 23, 3, 0, 0);   // MFC1 s7, f3
    R(0x11, 0x00, 24, 4, 0, 0);   // MFC1 t8, f4
    R(0x11, 0x00, 25, 5, 0, 0);   // MFC1 t9, f5
    R(0x11, 0x00, 27, 7, 0, 0);   // MFC1 k1, f7
    R(0x11, 0x00, 6, 19, 0, 0);   // MFC1 a2, f19
    R(0x11, 0x00, 7, 20, 0, 0);   // MFC1 a3, f20
    R(0x11, 0x00, 8, 22, 0, 0);   // MFC1 t0, f22

    I(0x0F, 0, 28, 0x3FF0);                                    // gp = 0x3FF00000
    R(0x00, 0, 28, 28, 0, 0x3C);                               // 1.0 double bits
    I(0x0F, 0, 29, 0x4000);                                    // sp = 0x40000000
    R(0x00, 0, 29, 29, 0, 0x3C);                               // 2.0 double bits
    R(0x11, 0x05, 28, 10, 0, 0);  // DMTC1 gp, f10
    R(0x11, 0x05, 29, 11, 0, 0);  // DMTC1 sp, f11
    R(0x11, 0x11, 11, 10, 12, 0x00); // ADD.D f12, f10, f11
    R(0x11, 0x11, 10, 11, 13, 0x01); // SUB.D f13, f11, f10
    R(0x11, 0x11, 11, 10, 14, 0x02); // MUL.D f14, f10, f11
    R(0x11, 0x11, 0, 12, 15, 0x06);  // MOV.D f15, f12
    R(0x11, 0x11, 0, 10, 17, 0x07);  // NEG.D f17, f10
    R(0x11, 0x11, 0, 17, 18, 0x05);  // ABS.D f18, f17
    R(0x11, 0x05, 0, 25, 0, 0);      // DMTC1 $0, f25
    R(0x11, 0x11, 11, 10, 26, 0x03); // DIV.D f26, f10, f11
    R(0x11, 0x11, 25, 10, 27, 0x03); // DIV.D f27, f10, f25 -> 0.0 (special-cased)
    R(0x11, 0x11, 0, 11, 28, 0x04);  // SQRT.D f28, f11
    R(0x11, 0x11, 0, 12, 29, 0x20);  // CVT.S.D f29, f12
    R(0x11, 0x11, 0, 12, 30, 0x24);  // CVT.W.D f30, f12
    R(0x11, 0x01, 28, 12, 0, 0);  // DMFC1 gp, f12
    R(0x11, 0x01, 29, 13, 0, 0);  // DMFC1 sp, f13
    R(0x11, 0x01, 30, 14, 0, 0);  // DMFC1 fp, f14
    R(0x11, 0x01, 31, 15, 0, 0);  // DMFC1 ra, f15
    R(0x11, 0x01, 9, 26, 0, 0);   // DMFC1 t1, f26
    R(0x11, 0x01, 10, 27, 0, 0);  // DMFC1 t2, f27
    R(0x11, 0x00, 11, 30, 0, 0);  // MFC1 t3, f30
    R(0x11, 0x01, 4, 17, 0, 0);   // DMFC1 a0, f17
    R(0x11, 0x01, 5, 18, 0, 0);   // DMFC1 a1, f18

    // Integer -> float conversions.
    I(0x09, 0, 12, static_cast<u16>(-7));  // ADDIU t4, $0, -7
    R(0x11, 0x04, 12, 16, 0, 0);            // MTC1 t4, f16
    R(0x11, 0x14, 0, 16, 24, 0x20);         // CVT.S.W f24, f16
    R(0x11, 0x14, 0, 16, 31, 0x21);         // CVT.D.W f31, f16
    R(0x11, 0x00, 13, 24, 0, 0);            // MFC1 t5, f24
    R(0x11, 0x01, 14, 31, 0, 0);            // DMFC1 t6, f31
    I(0x09, 0, 15, static_cast<u16>(-3));  // ADDIU t7, $0, -3
    R(0x00, 0, 15, 15, 0, 0x3C);            // DSLL32 t7, t7, 0
    R(0x11, 0x05, 15, 16, 0, 0);            // DMTC1 t7, f16
    R(0x11, 0x15, 0, 16, 24, 0x20);         // CVT.S.L f24, f16
    R(0x11, 0x15, 0, 16, 31, 0x21);         // CVT.D.L f31, f16
    R(0x11, 0x00, 1, 24, 0, 0);             // MFC1 at, f24
    R(0x11, 0x01, 2, 31, 0, 0);             // DMFC1 v0, f31
    return p;
}

// One conditional branch with markers on both paths:
//   b:   <branch> +2          (target = b+3)
//   b+1: ADDIU s0, s0, 1      delay slot (also clobbers the compared register
//                             when `ds_clobbers` - the branch must still use
//                             the value from before it)
//   b+2: ADDIU s1, s1, 1      fall-through only
//   b+3: ADDIU s2, s2, 1      both
struct BranchKind { const char* name; u32 (*encode)(u8 rs, u8 rt, s16 off); bool two_regs; };
u32 enc_op(u8 op, u8 rs, u8 rt, s16 off) { return itype(op, rs, rt, static_cast<u16>(off)); }
u32 enc_regimm(u8 sub, u8 rs, s16 off) { return itype(0x01, rs, sub, static_cast<u16>(off)); }

const BranchKind kBranches[] = {
    {"BEQ", [](u8 rs, u8 rt, s16 o) { return enc_op(0x04, rs, rt, o); }, true},
    {"BNE", [](u8 rs, u8 rt, s16 o) { return enc_op(0x05, rs, rt, o); }, true},
    {"BLEZ", [](u8 rs, u8, s16 o) { return enc_op(0x06, rs, 0, o); }, false},
    {"BGTZ", [](u8 rs, u8, s16 o) { return enc_op(0x07, rs, 0, o); }, false},
    {"BEQL", [](u8 rs, u8 rt, s16 o) { return enc_op(0x14, rs, rt, o); }, true},
    {"BNEL", [](u8 rs, u8 rt, s16 o) { return enc_op(0x15, rs, rt, o); }, true},
    {"BLEZL", [](u8 rs, u8, s16 o) { return enc_op(0x16, rs, 0, o); }, false},
    {"BGTZL", [](u8 rs, u8, s16 o) { return enc_op(0x17, rs, 0, o); }, false},
    {"BLTZ", [](u8 rs, u8, s16 o) { return enc_regimm(0x00, rs, o); }, false},
    {"BGEZ", [](u8 rs, u8, s16 o) { return enc_regimm(0x01, rs, o); }, false},
    {"BLTZL", [](u8 rs, u8, s16 o) { return enc_regimm(0x02, rs, o); }, false},
    {"BGEZL", [](u8 rs, u8, s16 o) { return enc_regimm(0x03, rs, o); }, false},
    {"BLTZAL", [](u8 rs, u8, s16 o) { return enc_regimm(0x10, rs, o); }, false},
    {"BGEZAL", [](u8 rs, u8, s16 o) { return enc_regimm(0x11, rs, o); }, false},
    {"BLTZALL", [](u8 rs, u8, s16 o) { return enc_regimm(0x12, rs, o); }, false},
    {"BGEZALL", [](u8 rs, u8, s16 o) { return enc_regimm(0x13, rs, o); }, false},
};

Prog branch_program(const BranchKind& k, s64 rs_val, s64 rt_val, bool ds_clobbers, u8 rs_reg = 8) {
    Prog p;
    li32(p, rs_reg, static_cast<u32>(rs_val)); // LUI sign-extends, so negative values come out right
    li32(p, 9, static_cast<u32>(rt_val));
    p.push_back(k.encode(rs_reg, 9, 2));
    p.push_back(ds_clobbers ? itype(0x09, 0, rs_reg, 0x55) : itype(0x09, 16, 16, 1)); // delay slot
    addiu(p, 17, 17, 1);
    addiu(p, 18, 18, 1);
    return p;
}

Prog loop_program() {
    // t0 = 10; do { t0--; t1 += 2 (delay slot) } while (t0 != 0)
    Prog p;
    addiu(p, 8, 0, 10);
    addiu(p, 9, 0, 0);
    addiu(p, 8, 8, -1);                          // loop:
    p.push_back(itype(0x05, 8, 0, static_cast<u16>(-2))); // BNE t0, $0, loop
    addiu(p, 9, 9, 2);                           // delay slot
    // Backwards BEQL that falls through: its delay slot must not run.
    p.push_back(itype(0x14, 8, 17, static_cast<u16>(-4))); // BEQL t0, s1 (t0=0, s1=9 -> not taken)
    addiu(p, 10, 10, 100);                       // nullified
    return p;
}

// J/JAL forward, JR/JALR through a computed register, JALR rd==rs.
Prog jump_program(u64 entry) {
    Prog p;
    const u32 base = static_cast<u32>(entry);
    // 0: J to 3
    p.push_back(jtype(0x02, base + 3 * 4));
    addiu(p, 8, 8, 1);                // 1: delay slot
    addiu(p, 9, 9, 1);                // 2: skipped
    // 3: JAL to 6 (ra = entry + 5*4)
    p.push_back(jtype(0x03, base + 6 * 4));
    addiu(p, 10, 10, 1);              // 4: delay slot
    addiu(p, 11, 11, 1);              // 5: skipped
    // 6..7: t4 = address of 12 (sign-extended KSEG0/KSEG1 address)
    li32(p, 12, base + 12 * 4);
    // 8: JR t4
    p.push_back(rtype(0, 12, 0, 0, 0, 0x08));
    addiu(p, 12, 12, 4);              // 9: delay slot modifies the jump register (must not matter)
    addiu(p, 13, 13, 1);              // 10: skipped
    addiu(p, 13, 13, 1);              // 11: skipped
    // 12..13: t6 = address of 17
    li32(p, 14, base + 17 * 4);
    // 14: JALR s3, t6
    p.push_back(rtype(0, 14, 0, 19, 0, 0x09));
    addiu(p, 15, 15, 1);              // 15: delay slot
    addiu(p, 15, 15, 100);            // 16: skipped
    addiu(p, 20, 0, 7);               // 17
    return p;
}

// Exceptions raised inside compiled code must report the faulting
// instruction's own pc (or the branch's, with BD, for a taken branch's delay
// slot) in EPC - regardless of what the interpreter last executed.
Prog overflow_mid_block() {
    Prog p;
    addiu(p, 8, 8, 1);
    addiu(p, 9, 9, 1);
    p.push_back(itype(0x08, 12, 10, 1)); // ADDI t2, t4(=INT32_MAX), 1 -> EXC_OV
    addiu(p, 11, 11, 1);
    return p;
}
Prog overflow_in_delay_slot(u8 branch_op, bool taken) {
    Prog p;
    addiu(p, 8, 8, 1);
    // BEQ/BNE/BEQL/BNEL $0, $0 -> taken iff "equal" flavour
    p.push_back(itype(branch_op, 0, 0, 2));
    p.push_back(itype(0x08, 12, 10, 1)); // ADDI overflow in the delay slot
    addiu(p, 17, 17, 1);
    addiu(p, 18, 18, 1);
    (void)taken;
    return p;
}
Prog tlb_miss_in_delay_slot() {
    Prog p;
    addiu(p, 8, 0, 0x100);                         // t0 = 0x100 (kuseg, TLB-mapped)
    p.push_back(itype(0x04, 0, 0, 2));             // BEQ $0,$0 (taken)
    p.push_back(itype(0x23, 8, 9, 0));             // LW t1, 0(t0) -> TLB miss
    addiu(p, 17, 17, 1);
    addiu(p, 18, 18, 1);
    return p;
}
Prog tlb_miss_store_mid_block() {
    Prog p;
    addiu(p, 8, 0, 0x200);
    addiu(p, 9, 9, 1);
    p.push_back(itype(0x2B, 8, 9, 4));             // SW t1, 4(t0) -> TLB miss (store)
    addiu(p, 10, 10, 1);
    return p;
}
// A branch the JIT doesn't compile (BC1T) runs on the interpreter; its delay
// slot then goes through the dedicated one-instruction delay-slot block.
Prog bc1t_program(bool overflow_in_ds) {
    Prog p;
    p.push_back(rtype(0x11, 0x10, 0, 0, 0, 0x32)); // C.EQ.S f0, f0 -> condition true
    p.push_back(itype(0x11, 0x08, 0x01, 2));       // BC1T +2
    p.push_back(overflow_in_ds ? itype(0x08, 12, 10, 1) : itype(0x09, 16, 16, 1));
    addiu(p, 17, 17, 1);
    addiu(p, 18, 18, 1);
    return p;
}
// Stores inside the delay slot, and a store to code the block itself came
// from - the JIT must see the new instruction next time around.
Prog store_in_delay_slot() {
    Prog p;
    li32(p, 4, 0x80000000u | kScratchPaddr);
    p.push_back(itype(0x04, 0, 0, 2));             // BEQ $0,$0 (taken)
    p.push_back(itype(0x2B, 4, 21, 0x10));         // SW s5, 16(a0) in the delay slot
    addiu(p, 17, 17, 1);
    p.push_back(itype(0x23, 4, 22, 0x10));         // LW s6, 16(a0)
    return p;
}

// Instructions compiled as in-place interpreter calls (jit_interp).
Prog muldiv_program(u8 funct, u32 a, u32 b) {
    Prog p;
    li32(p, 8, a);
    li32(p, 9, b);
    p.push_back(rtype(0, 8, 9, 0, 0, funct));
    p.push_back(rtype(0, 0, 0, 10, 0, 0x10)); // MFHI t2
    p.push_back(rtype(0, 0, 0, 11, 0, 0x12)); // MFLO t3
    p.push_back(rtype(0, 16, 21, 0, 0, funct)); // 64-bit seeds s0 x s5
    return p;
}
Prog fpu_memory_program() {
    Prog p;
    li32(p, 4, 0x80000000u | kScratchPaddr);
    li32(p, 8, 0x3FC00000u);                 // 1.5f
    p.push_back(rtype(0x11, 0x04, 8, 0, 0, 0)); // MTC1 t0, f0
    p.push_back(itype(0x39, 4, 0, 0x20));    // SWC1 f0, 0x20(a0)
    p.push_back(itype(0x31, 4, 2, 0x20));    // LWC1 f2, 0x20(a0)
    p.push_back(rtype(0x11, 0x00, 9, 2, 0, 0)); // MFC1 t1, f2
    p.push_back(itype(0x3F, 4, 16, 0x28));   // SD s0, 0x28(a0)
    p.push_back(itype(0x35, 4, 4 + 0, 0x28)); // LDC1 f4, 0x28(a0)
    p.push_back(itype(0x3D, 4, 4, 0x30));    // SDC1 f4, 0x30(a0)
    p.push_back(itype(0x37, 4, 10, 0x30));   // LD t2, 0x30(a0)
    // Unaligned word/doubleword loads and stores.
    p.push_back(itype(0x22, 4, 11, 0x29));   // LWL t3, 0x29(a0)
    p.push_back(itype(0x26, 4, 11, 0x2C));   // LWR t3, 0x2C(a0)
    p.push_back(itype(0x1A, 4, 12, 0x2B));   // LDL t4, 0x2B(a0)
    p.push_back(itype(0x1B, 4, 12, 0x32));   // LDR t4, 0x32(a0)
    p.push_back(itype(0x2A, 4, 21, 0x41));   // SWL s5, 0x41(a0)
    p.push_back(itype(0x2E, 4, 21, 0x44));   // SWR s5, 0x44(a0)
    p.push_back(itype(0x2C, 4, 16, 0x4B));   // SDL s0, 0x4B(a0)
    p.push_back(itype(0x2D, 4, 16, 0x52));   // SDR s0, 0x52(a0)
    p.push_back(itype(0x30, 4, 13, 0x20));   // LL t5, 0x20(a0)
    p.push_back(itype(0x38, 4, 14, 0x60));   // SC t6, 0x60(a0)
    p.push_back(itype(0x2F, 4, 0, 0));       // CACHE
    return p;
}
Prog fpu_convert_program() {
    Prog p;
    li32(p, 8, 0xC0200000u); // -2.5f
    p.push_back(rtype(0x11, 0x04, 8, 0, 0, 0)); // MTC1 t0, f0
    const u8 ops[] = {0x0C, 0x0D, 0x0E, 0x0F, 0x08, 0x09, 0x0A, 0x0B}; // ROUND/TRUNC/CEIL/FLOOR .W and .L
    u8 out = 9;
    for (u8 op : ops) {
        p.push_back(rtype(0x11, 0x10, 0, 0, 2, op));  // op.S f2, f0
        p.push_back(rtype(0x11, 0x01, out++, 2, 0, 0)); // DMFC1 out, f2
    }
    p.push_back(rtype(0x11, 0x02, 20, 31, 0, 0));   // CFC1 s4, fcsr
    li32(p, 21, 0x00000003u);
    p.push_back(rtype(0x11, 0x06, 21, 31, 0, 0));   // CTC1 s5, fcsr (round toward -inf)
    p.push_back(rtype(0x11, 0x02, 22, 31, 0, 0));   // CFC1 s6, fcsr
    return p;
}
Prog lwc1_in_likely_delay_slot(bool taken) {
    Prog p;
    li32(p, 4, 0x80000000u | kScratchPaddr);
    p.push_back(itype(taken ? 0x14 : 0x15, 0, 0, 2)); // BEQL/BNEL $0,$0
    p.push_back(itype(0x31, 4, 6, 0x10));             // LWC1 f6, 0x10(a0)
    addiu(p, 17, 17, 1);
    p.push_back(rtype(0x11, 0x00, 18, 6, 0, 0));      // MFC1 s2, f6
    return p;
}
Prog lwc1_tlb_miss_in_delay_slot() {
    Prog p;
    addiu(p, 8, 0, 0x300);
    p.push_back(itype(0x05, 8, 0, 2));    // BNE t0, $0 (taken)
    p.push_back(itype(0x31, 8, 2, 0));    // LWC1 f2, 0(t0) -> TLB miss
    addiu(p, 17, 17, 1);
    addiu(p, 18, 18, 1);
    return p;
}

// Patches an instruction of a block that already ran, then runs it again:
// the (inline) store must invalidate the compiled copy.
Prog self_modifying_program(u64 entry) {
    Prog p;
    const u32 base = static_cast<u32>(entry);
    const u32 target = base + 7 * 4;
    li32(p, 4, target);                               // 0-1: a0 = &L
    li32(p, 11, itype(0x09, 10, 10, 100));            // 2-3: t3 = "ADDIU t2, t2, 100"
    addiu(p, 9, 0, 0);                                // 4: t1 = 0 (first pass)
    p.push_back(jtype(0x02, target));                 // 5: J L - so L starts a block on both passes
    p.push_back(0);                                   // 6
    addiu(p, 10, 10, 1);                              // 7 (L): ADDIU t2, t2, 1 - patched below
    p.push_back(itype(0x05, 9, 0, 5));                // 8: BNE t1, $0, END (second pass leaves)
    p.push_back(0);                                   // 9
    p.push_back(itype(0x2B, 4, 11, 0));               // 10: SW t3, 0(a0)
    addiu(p, 9, 0, 1);                                // 11: t1 = 1
    p.push_back(jtype(0x02, target));                 // 12: J L
    p.push_back(0);                                   // 13
    addiu(p, 12, 10, 0);                              // 14 (END)
    return p;
}
// Loads and stores through the uncached KSEG1 window and at the RDRAM edge.
Prog kseg1_memory_program() {
    Prog p;
    li32(p, 4, 0xA0000000u | kScratchPaddr);
    p.push_back(itype(0x2B, 4, 21, 0x70));   // SW s5, 0x70(a0)
    p.push_back(itype(0x3F, 4, 16, 0x78));   // SD s0, 0x78(a0)
    p.push_back(itype(0x29, 4, 26, 0x80));   // SH k0, 0x80(a0)
    p.push_back(itype(0x28, 4, 27, 0x83));   // SB k1, 0x83(a0)
    li32(p, 5, 0x80000000u | kScratchPaddr);
    p.push_back(itype(0x23, 5, 8, 0x70));    // LW t0, 0x70(a1) - same bytes via KSEG0
    p.push_back(itype(0x37, 5, 9, 0x78));    // LD t1
    p.push_back(itype(0x21, 5, 10, 0x80));   // LH t2
    p.push_back(itype(0x25, 5, 11, 0x80));   // LHU t3
    p.push_back(itype(0x20, 5, 12, 0x83));   // LB t4
    p.push_back(itype(0x24, 5, 13, 0x83));   // LBU t5
    p.push_back(itype(0x27, 5, 14, 0x70));   // LWU t6
    p.push_back(itype(0x23, 5, 15, 0x72));   // LW t7, unaligned (slow path, like the interpreter)
    li32(p, 6, 0x807FFFFCu);
    p.push_back(itype(0x23, 6, 17, 0));      // LW s1, last word of RDRAM
    return p;
}

// ---------------------------------------------------------------------------
// Random programs
// ---------------------------------------------------------------------------
// Straight-line mixes of ALU, shift, multiply/divide, trapping arithmetic,
// loads and stores (aligned and not) and every kind of branch/jump, with
// random registers and immediates. Control flow only ever goes forward, except
// for counted loops (at most 4 times around), so every program ends. Register 20 always holds the
// scratch area's address (loads/stores go through it), 22 is the loop counter,
// 21/23 serve JR/JALR (see kJumpReg); none of them is ever written otherwise.
struct Rng {
    u64 x;
    u64 next() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; }
    u32 below(u32 n) { return static_cast<u32>(next() % n); }
    bool chance(u32 pct) { return below(100) < pct; }
};

// 21 is the JR/JALR target, 23 the program's entry address and 24 a STATUS
// value for MTC0 (FR either way, interrupts disabled).
constexpr u8 kBaseReg = 20, kJumpReg = 21, kLoopReg = 22, kEntryReg = 23, kStatusReg = 24;

u8 rand_src(Rng& r) {
    static const u8 regs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 31};
    return regs[r.below(sizeof regs)];
}
u8 rand_dst(Rng& r) {
    static const u8 regs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 31};
    return regs[r.below(sizeof regs)];
}
u16 rand_imm(Rng& r) {
    static const u16 edge[] = {0, 1, 2, 0x7FFF, 0x8000, 0xFFFF, 0xFFFE, 0x1F, 0x20, 0x3F, 0x100, 0xF000};
    return r.chance(40) ? edge[r.below(sizeof edge / sizeof edge[0])] : static_cast<u16>(r.next());
}

// One non-branch instruction (also used for delay slots).
u32 rand_plain(Rng& r) {
    const u8 rs = rand_src(r), rt = rand_src(r), rd = rand_dst(r), sa = static_cast<u8>(r.below(32));
    switch (r.below(14)) {
    case 0: { // SPECIAL ALU
        static const u8 f[] = {0x00, 0x02, 0x03, 0x04, 0x06, 0x07, 0x0A, 0x0B, 0x10, 0x12, 0x14, 0x16, 0x17,
                               0x21, 0x23, 0x24, 0x25, 0x26, 0x27, 0x2A, 0x2B, 0x2D, 0x2F, 0x38, 0x3A, 0x3B,
                               0x3C, 0x3E, 0x3F};
        return rtype(0, rs, rt, rd, sa, f[r.below(sizeof f)]);
    }
    case 1: { // trapping arithmetic
        static const u8 f[] = {0x20, 0x22, 0x2C, 0x2E};
        return rtype(0, rs, rt, rd, 0, f[r.below(sizeof f)]);
    }
    case 2: return rtype(0, rs, 0, 0, 0, r.chance(50) ? 0x11 : 0x13); // MTHI / MTLO
    case 3: { // multiply / divide
        return rtype(0, rs, rt, 0, 0, static_cast<u8>(0x18 + r.below(8)));
    }
    case 4: case 5: { // I-type ALU
        static const u8 ops[] = {0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x18, 0x19};
        return itype(ops[r.below(sizeof ops)], rs, rand_dst(r), rand_imm(r));
    }
    case 6: case 7: { // aligned load/store via the scratch base
        static const u8 ops[] = {0x20, 0x21, 0x23, 0x24, 0x25, 0x27, 0x37, 0x28, 0x29, 0x2B, 0x3F};
        const u8 op = ops[r.below(sizeof ops)];
        const u32 align = (op == 0x37 || op == 0x3F) ? 8 : (op == 0x23 || op == 0x27 || op == 0x2B) ? 4
                        : (op == 0x21 || op == 0x25 || op == 0x29) ? 2 : 1;
        const u16 off = static_cast<u16>(r.below(kScratchLen / align) * align);
        return itype(op, kBaseReg, (op >= 0x28) ? rand_src(r) : rand_dst(r), off);
    }
    case 8: { // unaligned / partial-word memory ops (slow paths and interpreter calls)
        static const u8 ops[] = {0x23, 0x21, 0x37, 0x2B, 0x29, 0x3F, 0x22, 0x26, 0x2A, 0x2E, 0x1A, 0x1B, 0x2C, 0x2D};
        const u8 op = ops[r.below(sizeof ops)];
        const u16 off = static_cast<u16>(r.below(kScratchLen - 8));
        return itype(op, kBaseReg, (op >= 0x28) ? rand_src(r) : rand_dst(r), off);
    }
    case 9: { // FPU moves and arithmetic through GPR bits
        const u8 f1 = static_cast<u8>(r.below(8)), f2 = static_cast<u8>(r.below(8)), f3 = static_cast<u8>(r.below(8));
        switch (r.below(10)) {
        case 0: return rtype(0x11, 0x04, rs, f1, 0, 0);          // MTC1
        case 1: return rtype(0x11, 0x00, rand_dst(r), f1, 0, 0); // MFC1
        case 2: return rtype(0x11, 0x05, rs, f1, 0, 0);          // DMTC1
        case 3: return rtype(0x11, 0x01, rand_dst(r), f1, 0, 0); // DMFC1
        case 4: return rtype(0x11, 0x02, rand_dst(r), static_cast<u8>(r.below(32)), 0, 0); // CFC1
        case 5: // CVT.S/CVT.D from .W/.L (and an invalid one now and then)
            return rtype(0x11, r.chance(50) ? 0x14 : 0x15, 0, f1, f2, static_cast<u8>(r.chance(90) ? 0x20 + r.below(2) : 0x24));
        case 6: // C.cond, all 16
            return rtype(0x11, r.chance(50) ? 0x10 : 0x11, f1, f2, 0, static_cast<u8>(0x30 + r.below(16)));
        default: {
            static const u8 f[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
                                   0x0C, 0x0D, 0x0E, 0x0F, 0x20, 0x21, 0x24, 0x25, 0x13};
            return rtype(0x11, r.chance(50) ? 0x10 : 0x11, f1, f2, f3, f[r.below(sizeof f)]);
        }
        }
    }
    case 10: { // COP0: MFC0/MTC0 (STATUS toggles FR, COMPARE/COUNT move the timer) and TLB instructions
        switch (r.below(6)) {
        case 0: case 1: {
            static const u8 regs[] = {1, 4, 6, 8, 9, 10, 12, 13, 14, 11, 18};
            return rtype(0x10, r.chance(80) ? 0x00 : 0x01, rand_dst(r), regs[r.below(sizeof regs)], 0, 0);
        }
        case 2: case 3: {
            static const u8 regs[] = {4, 10, 14, 18, 9, 11, 6};
            return rtype(0x10, 0x04, rs, regs[r.below(sizeof regs)], 0, 0);
        }
        case 4: return rtype(0x10, 0x04, kStatusReg, 12, 0, 0); // MTC0 STATUS from a prepared value
        default: {
            static const u8 fn[] = {0x01, 0x06, 0x08, 0x02};
            return rtype(0x10, 0x10, 0, 0, 0, fn[r.below(sizeof fn)]);
        }
        }
    }
    case 12: return 0; // NOP
    case 11: { // FPU loads/stores (aligned: inline; the interpreter does the rest)
        static const u8 ops[] = {0x31, 0x35, 0x39, 0x3D};
        const u8 op = ops[r.below(sizeof ops)];
        const u32 align = (op == 0x35 || op == 0x3D) ? 8 : 4;
        const u16 off = r.chance(90) ? static_cast<u16>(r.below(kScratchLen / align) * align) : static_cast<u16>(r.below(kScratchLen - 8));
        return itype(op, kBaseReg, static_cast<u8>(r.below(8)), off);
    }
    default: return rtype(0, rs, rt, rd, 0, 0x21); // ADDU
    }
}

Prog random_program(Rng& r, u64 entry) {
    const u32 n = 20 + r.below(40);
    Prog p;
    p.reserve(n + 8);
    while (p.size() < n) {
        const u32 here = static_cast<u32>(p.size());
        const u32 room = n - here; // instructions left, so forward targets stay inside
        const u32 kind = r.below(100);
        if (kind < 55 || room < 4) {
            p.push_back(rand_plain(r));
        } else if (kind < 80) { // conditional branch forward (target within the program or its exit)
            static const u8 ops[] = {0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17};
            const s16 off = static_cast<s16>(1 + r.below(room - 1)); // lands on here+1+off <= n
            if (r.chance(25)) {
                static const u8 sub[] = {0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13};
                p.push_back(itype(0x01, rand_src(r), sub[r.below(sizeof sub)], static_cast<u16>(off)));
            } else if (r.chance(20)) { // BC1F/BC1T/BC1FL/BC1TL
                p.push_back(itype(0x11, 0x08, static_cast<u8>(r.below(4)), static_cast<u16>(off)));
            } else {
                p.push_back(itype(ops[r.below(sizeof ops)], rand_src(r), rand_src(r), static_cast<u16>(off)));
            }
            p.push_back(rand_plain(r)); // delay slot
        } else if (kind < 87) { // J / JAL forward
            const u32 target = here + 2 + r.below(room - 1);
            p.push_back(jtype(r.chance(50) ? 0x02 : 0x03, static_cast<u32>(entry) + target * 4));
            p.push_back(rand_plain(r));
        } else if (kind < 92 && room >= 6) { // JR / JALR through a register set to a forward address
            const u32 target = here + 3 + r.below(room - 2);
            addiu(p, kJumpReg, kEntryReg, static_cast<s16>(target * 4));
            p.push_back(r.chance(50) ? rtype(0, kJumpReg, 0, 0, 0, 0x08) : rtype(0, kJumpReg, 0, rand_dst(r), 0, 0x09));
            // The delay slot points the register back at the stray BREAK (the
            // jump has read it already), so jumping straight onto a later JR
            // can't go backwards.
            addiu(p, kJumpReg, kEntryReg, static_cast<s16>(kStrayPaddr - kProgPaddr));
        } else if (room >= 8) { // counted loop: the body runs 1-4 times
            addiu(p, kLoopReg, 0, static_cast<s16>(1 + r.below(4)));
            const u32 top = static_cast<u32>(p.size());
            const u32 body = 1 + r.below(3);
            for (u32 i = 0; i < body; ++i) p.push_back(rand_plain(r));
            addiu(p, kLoopReg, kLoopReg, -1);
            const s16 back = static_cast<s16>(static_cast<s32>(top) - static_cast<s32>(p.size()) - 1);
            // BGTZ/BGTZL: a forward branch into the middle of the body (past
            // the counter's set-up) then just leaves the loop.
            p.push_back(itype(r.chance(70) ? 0x07 : 0x17, kLoopReg, 0, static_cast<u16>(back)));
            p.push_back(rand_plain(r));
        } else {
            p.push_back(rand_plain(r));
        }
    }
    return p;
}

} // namespace

int main(int argc, char** argv) {
    int failures = 0, cases = 0, skipped = 0;
    // `may_not_end`: a random program the interpreter itself doesn't finish
    // (e.g. a branch into the middle of a JR's set-up) proves nothing - skip it.
    auto check = [&](const std::string& name, const Prog& prog, u64 entry = 0xFFFFFFFF80000000ULL | kProgPaddr,
                     const u64* seeds = kSeed, bool may_not_end = false, u64 status = 0x34000000,
                     const u64* fpr_seeds = nullptr) {
        bool t1 = false, t2 = false, t3 = false;
        const State a = run(prog, entry, Mode::Interp, &t1, seeds, status, fpr_seeds);
        if (t1 && may_not_end) {
            ++skipped;
            return;
        }
        ++cases;
        const State b = run(prog, entry, Mode::JitStep, &t2, seeds, status, fpr_seeds);
        const State c = run(prog, entry, Mode::JitChain, &t3, seeds, status, fpr_seeds);
        bool ok = compare(name.c_str(), a, b);
        ok = compare((name + " (chained)").c_str(), a, c) && ok;
        if (t1 || t2 || t3) {
            std::printf("  [%s] did not reach the exit (interp %s, jit %s, chained %s)\n", name.c_str(),
                        t1 ? "timed out" : "ok", t2 ? "timed out" : "ok", t3 ? "timed out" : "ok");
            ok = false;
        }
        if (!ok) ++failures;
    };

    const u64 kseg0 = 0xFFFFFFFF80000000ULL | kProgPaddr;
    const u64 kseg1 = 0xFFFFFFFFA0000000ULL | kProgPaddr;

    check("alu", alu_program());

    const s64 values[] = {-5, 0, 7};
    for (const BranchKind& k : kBranches) {
        for (s64 rs : values) {
            for (s64 rt : values) {
                if (!k.two_regs && rt != 0) continue;
                for (bool clobber : {false, true}) {
                    char name[64];
                    std::snprintf(name, sizeof name, "%s rs=%lld rt=%lld%s", k.name, static_cast<long long>(rs),
                                  static_cast<long long>(rt), clobber ? " ds-clobbers-rs" : "");
                    check(name, branch_program(k, rs, rt, clobber));
                }
            }
        }
        // Branch-and-link reading $ra itself: the link is written first.
        std::string n = std::string(k.name) + " rs=$ra";
        check(n, branch_program(k, -1, 0, false, 31));
        check(std::string(k.name) + " kseg1", branch_program(k, 7, 7, false), kseg1);
    }
    check("loop", loop_program());
    check("jumps kseg0", jump_program(kseg0), kseg0);
    check("jumps kseg1", jump_program(kseg1), kseg1);
    check("overflow mid-block", overflow_mid_block());
    check("overflow in BEQ delay slot (taken)", overflow_in_delay_slot(0x04, true));
    check("overflow in BNE delay slot (not taken)", overflow_in_delay_slot(0x05, false));
    check("overflow in BEQL delay slot (taken)", overflow_in_delay_slot(0x14, true));
    check("overflow in BNEL delay slot (nullified)", overflow_in_delay_slot(0x15, false));
    check("TLB miss (load) in delay slot", tlb_miss_in_delay_slot());
    check("TLB miss (store) mid-block", tlb_miss_store_mid_block());
    check("BC1T delay slot", bc1t_program(false));
    check("overflow in BC1T delay slot", bc1t_program(true));
    check("store in delay slot", store_in_delay_slot());

    const char* muldiv_names[] = {"MULT", "MULTU", "DIV", "DIVU", "DMULT", "DMULTU", "DDIV", "DDIVU"};
    const u32 operands[][2] = {{7, 3}, {0xFFFFFFF9u, 3}, {0x80000000u, 0xFFFFFFFFu}, {12345, 0}, {0xFFFFFFFFu, 0xFFFFFFFFu}};
    for (u8 f = 0x18; f <= 0x1F; ++f) {
        for (const auto& ab : operands) {
            char name[64];
            std::snprintf(name, sizeof name, "%s 0x%x,0x%x", muldiv_names[f - 0x18], ab[0], ab[1]);
            check(name, muldiv_program(f, ab[0], ab[1]));
        }
    }
    check("FPU/unaligned loads and stores", fpu_memory_program());
    check("FPU conversions + CFC1/CTC1", fpu_convert_program());
    check("LWC1 in BEQL delay slot (taken)", lwc1_in_likely_delay_slot(true));
    check("LWC1 in BNEL delay slot (nullified)", lwc1_in_likely_delay_slot(false));
    check("LWC1 TLB miss in delay slot", lwc1_tlb_miss_in_delay_slot());
    check("self-modifying code", self_modifying_program(kseg0), kseg0);
    check("KSEG1 / edge-of-RDRAM loads and stores", kseg1_memory_program());
    check("jumps TLB-mapped", jump_program(kMappedBase | kProgPaddr), kMappedBase | kProgPaddr);
    check("loop TLB-mapped", loop_program(), kMappedBase | kProgPaddr);
    check("alu TLB-mapped", alu_program(), kMappedBase | kProgPaddr);

    // Random programs: `jit_selftest [count] [seed]`.
    const int fuzz = argc > 1 ? std::atoi(argv[1]) : 2000;
    Rng rng{argc > 2 ? std::strtoull(argv[2], nullptr, 0) : 0x9E3779B97F4A7C15ULL};
    int fuzz_failures = 0;
    for (int i = 0; i < fuzz; ++i) {
        const u32 kind = rng.below(10);
        const u64 entry = (kind < 5 ? 0xFFFFFFFF80000000ULL : kind < 7 ? 0xFFFFFFFFA0000000ULL : kMappedBase) | kProgPaddr;
        u64 seeds[32];
        for (int k = 0; k < 32; ++k) {
            switch (rng.below(4)) {
            case 0: seeds[k] = rng.next(); break;
            case 1: seeds[k] = static_cast<u64>(static_cast<s64>(static_cast<s32>(rng.next()))); break;
            case 2: seeds[k] = rng.below(8); break;
            default: seeds[k] = kSeed[k]; break;
            }
        }
        seeds[kBaseReg] = (rng.chance(80) ? 0xFFFFFFFF80000000ULL : 0xFFFFFFFFA0000000ULL) | kScratchPaddr;
        seeds[kLoopReg] = 0;
        seeds[kEntryReg] = entry;
        const u64 status = rng.chance(70) ? 0x30000000 : 0x34000000; // FR=0 is what games run with
        seeds[kStatusReg] = rng.chance(50) ? 0x30000000 : 0x34000000;
        u64 fprs[32];
        for (int k = 0; k < 32; ++k) {
            static const u64 fp_edge[] = {0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001,
                                          0x00000001, 0x80000000, 0x3FF0000000000000ULL, 0x7FF8000000000000ULL,
                                          0x4F000000, 0xCF000000, 0x41DFFFFFFFC00000ULL, 0x40490FDB};
            switch (rng.below(4)) {
            case 0: fprs[k] = fp_edge[rng.below(sizeof fp_edge / sizeof fp_edge[0])]; break;
            case 1: fprs[k] = rng.next(); break;
            case 2: fprs[k] = rng.next() & 0xFFFFFFFFull; break;
            default: { // an ordinary float/double value
                const float f = static_cast<float>(static_cast<s32>(rng.next() % 200000) - 100000) / 64.0f;
                u32 b; std::memcpy(&b, &f, 4);
                fprs[k] = rng.chance(50) ? b : 0x40A0000000000000ULL + (rng.next() & 0xFFFFFFFFFFFFull);
                break;
            }
            }
        }
        seeds[kJumpReg] = (entry & ~0xFFFFULL) | kStrayPaddr; // a stray JR hits a BREAK
        const Prog prog = random_program(rng, entry);
        const int before = failures;
        char name[48];
        std::snprintf(name, sizeof name, "random #%d", i);
        check(name, prog, entry, seeds, /*may_not_end*/ true, status, fprs);
        if (failures != before && ++fuzz_failures <= 3) {
            std::printf("  program (entry 0x%llx):\n", static_cast<unsigned long long>(entry));
            for (size_t k = 0; k < prog.size(); ++k) std::printf("    %3zu: %08x\n", k, prog[k]);
        }
    }

    if (skipped) std::printf("(%d random programs skipped: the interpreter doesn't finish them either)\n", skipped);
    if (failures == 0) {
        std::printf("OK: interpreter and JIT agree on all %d programs.\n", cases);
        return 0;
    }
    std::printf("FAILED: %d of %d programs differ.\n", failures, cases);
    return 1;
}
