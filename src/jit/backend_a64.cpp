// AArch64 backend of the recompiler: turns a run of guest instructions into
// native code. See recompiler.hpp for what a block is.
//
// Execution model. Blocks are not C functions: Recompiler::run() calls the
// `enter` trampoline (emit_a64_runtime), which saves the callee-saved
// registers, loads the context registers below and jumps to the block. A
// block ends with its branch and delay slot (or before an instruction it
// can't compile), subtracts the cycles it used from X_CYC and, while budget
// is left, jumps straight to the next block: a direct B for a static target
// (patched in once that block is compiled), the dispatcher's pc lookup for
// JR/JALR. Everything else leaves through `exit`, which hands X_PC/X_CYC
// back in the JitCtx.
//
// A conditional branch's delay slot is compiled twice, once on each path, so
// each copy knows statically whether it is a taken branch's delay slot (BD for
// exceptions) and where execution goes after it.
//
// Guest registers live in the CPU's register file (X_GPR) and are loaded and
// stored around every instruction. Loads and stores that hit RDRAM through
// KSEG0/KSEG1 are done inline ("fastmem"); everything else - TLB-mapped
// addresses, MMIO, cartridge space, unaligned accesses - takes an out-of-line
// slow path into the same C++ helpers the interpreter's semantics live in.
#include "backend_a64.hpp"

#if defined(__aarch64__)
#include "assembler_a64.hpp"
#include "code_buffer.hpp"
#include "jit_decode.hpp"
#include "jit_helpers.hpp"
#include "../bus.hpp"
#include "../cpu.hpp"
#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

using namespace jit_detail;
using namespace a64;

namespace {

// Context registers (AAPCS64 callee-saved), fixed while a chain runs.
constexpr int X_CTX = 19;   // JitCtx*
constexpr int X_GPR = 20;   // u64 gpr[32]
constexpr int X_MEM = 21;   // RDRAM base (JitCtx::rdram)
constexpr int X_PAGES = 22; // JitCtx::code_pages
constexpr int X_PC = 23;    // guest pc of the running block's first instruction
constexpr int X_CYC = 24;   // cycle budget left (signed)
constexpr int X_JC = 25;    // JitCtx::jcache
constexpr int X_TGT = 26;   // JR/JALR target, kept across the delay slot
// Scratch: X0-X17. X18 is the platform register on Apple and is never used;
// nothing survives a helper call except the context registers above.
constexpr int X_CALL = 9; // holds a helper's address for BLR

// The trampoline's frame: X29/X30, X19-X28.
constexpr u32 kFrameSize = 96;

// Byte offsets of JitCtx fields (static_asserted in jit_abi.hpp).
constexpr s32 kCtxGpr = 16, kCtxHi = 24, kCtxLo = 32, kCtxStartPc = 40, kCtxNextPc = 48, kCtxRdram = 64,
              kCtxPages = 72, kCtxCycles = 80, kCtxJCache = 88, kCtxExitReq = 96, kCtxFrMismatch = 100;

static_assert(RDRAM_SIZE == 0x800000, "fastmem range check assumes 8 MB of RDRAM");

// Where execution goes after the instruction being compiled, for leaving the
// chain right after it (an MMIO store asked for that).
struct Cont {
    enum Kind { Static, Target, CtxNext } kind = Static;
    s64 off = 0;      // Static: X_PC += off. Target: X_PC = X_TGT. CtxNext: X_PC = JitCtx::next_pc.
    u32 executed = 0; // guest instructions completed by then
};

struct Gen {
    Assembler a;
    const A64BlockEnv* env = nullptr;
    const u8* rdram = nullptr;
    size_t rdram_size = 0;
    u32 ram_limit = RDRAM_SIZE; // Bus::get_ram_limit: the fastmem bound
    u32 start_paddr = 0;
    u32 code_end = 0;   // physical address the block may not reach into
    bool mapped = false; // reached through the TLB: see A64BlockEnv::mapped
    bool chain = false; // this block's exits may jump to other blocks
    // CPU state relative to X_GPR (all members of the same CPU object; see
    // compile_block_a64). HI/LO fall back to the JitCtx pointers if unusable;
    // without the others COP1 falls back to the interpreter.
    s32 hi_off = -1, lo_off = -1;
    s32 fpr_off = -1, fcsr_off = -1, status_off = -1;
    bool fpu_ok() const { return fpr_off >= 0 && fcsr_off >= 0 && status_off >= 0; }
    // STATUS.FR this block is compiled for, and whether anything in it depends
    // on it (64-bit FPR access) - then the block checks it on entry.
    bool fr = false;
    bool uses_fr = false;
    Cont cont; // of the instruction being compiled

    std::vector<size_t> le_exits;                          // B.LE -> the block's local "B exit"
    std::vector<std::pair<size_t, const void*>> abs_branches; // B -> absolute address
    std::vector<size_t> self_links;                        // B -> this block's own entry
    std::vector<OverflowSite> overflows;
    std::vector<std::function<void()>> stubs; // out-of-line code, emitted after the body
    std::vector<Recompiler::PendingLink> pending;
};

// ---------------------------------------------------------------------------
// Small building blocks
// ---------------------------------------------------------------------------

// Guest register -> host register holding its value: `tmp` after loading it,
// or XZR for $zero (only valid where the instruction reads 31 as ZR).
int load64(Gen& g, u8 r, int tmp) {
    if (r == 0) return XZR;
    g.a.ldr64(tmp, X_GPR, r * 8);
    return tmp;
}
// Only the low 32 bits (zero-extended into the X register).
int load32(Gen& g, u8 r, int tmp) {
    if (r == 0) return XZR;
    g.a.ldr32z(tmp, X_GPR, r * 8);
    return tmp;
}
void store(Gen& g, u8 r, int src) {
    if (r != 0) g.a.str64(src, X_GPR, r * 8);
}

void call(Gen& g, const void* fn) {
    g.a.mov_imm(X_CALL, reinterpret_cast<u64>(fn));
    g.a.blr(X_CALL);
}

void b_abs(Gen& g, const void* target) {
    g.abs_branches.emplace_back(g.a.b(), target);
}

// Leaves the chain after `executed` instructions of this block (X_PC is
// irrelevant: an exception already moved the guest pc to its vector).
void exit_fault(Gen& g, u32 executed) {
    g.a.sub_imm(X_CYC, X_CYC, 2 * executed);
    b_abs(g, g.env->rt->exit);
}

// Leaves the chain at continuation `c`.
void exit_at(Gen& g, const Cont& c) {
    switch (c.kind) {
    case Cont::Static: g.a.add_any(X_PC, X_PC, c.off, 16); break;
    case Cont::Target: g.a.mov_reg(X_PC, X_TGT); break;
    case Cont::CtxNext: g.a.ldr64(X_PC, X_CTX, kCtxNextPc); break;
    }
    g.a.sub_imm(X_CYC, X_CYC, 2 * c.executed);
    b_abs(g, g.env->rt->exit);
}

// After a helper that may have set JitCtx::exit_req (a store that hit MMIO or
// compiled code): leave the chain right after this instruction if it did.
void check_exit_req(Gen& g, const Cont& c) {
    g.a.ldr32z(16, X_CTX, kCtxExitReq);
    const size_t to = g.a.cbnz(16, false);
    g.stubs.push_back([&g, to, c]() {
        g.a.patch_branch(to, g.a.pos());
        exit_at(g, c);
    });
}

// Helpers that can raise an exception compute the faulting pc from
// JitCtx::start_pc, which only the block itself knows.
void publish_pc(Gen& g) { g.a.str64(X_PC, X_CTX, kCtxStartPc); }

// After a helper returning 0 = ok / 1 = raised an exception: leaves the chain
// on a fault (the faulting instruction counted, as in the interpreter), else
// branches on. Returns that CBZ, for the caller to point wherever execution
// continues.
size_t check_fault(Gen& g, u32 site) {
    const size_t ok = g.a.cbz(0, false);
    exit_fault(g, site_index(site) + 1);
    return ok;
}

void emit_overflow_check(Gen& g, u32 site) {
    g.overflows.push_back({g.a.bcond(Cond::VS), site});
}

void load_hilo(Gen& g, int dst, bool hi) {
    const s32 off = hi ? g.hi_off : g.lo_off;
    if (off >= 0) {
        g.a.ldr64(dst, X_GPR, off);
    } else {
        g.a.ldr64(dst, X_CTX, hi ? kCtxHi : kCtxLo);
        g.a.ldr64(dst, dst, 0);
    }
}
void store_hilo(Gen& g, int src, bool hi) {
    const s32 off = hi ? g.hi_off : g.lo_off;
    if (off >= 0) {
        g.a.str64(src, X_GPR, off);
    } else {
        g.a.ldr64(17, X_CTX, hi ? kCtxHi : kCtxLo);
        g.a.str64(src, 17, 0);
    }
}

// FPR access (see the COP1 section below).
s32 fpr_at(const Gen& g, u8 r) { return g.fpr_off + (r & 31) * 8; }

// get_fpr32 -> S register.
void fget32(Gen& g, int vd, u8 r) { g.a.ldr_s(vd, X_GPR, fpr_at(g, r)); }
// set_fpr32 from an S register holding a scalar result (its upper bits are
// zero, so storing the whole D register is the zero-extended store).
void fset32(Gen& g, int vs, u8 r) { g.a.str_d(vs, X_GPR, fpr_at(g, r)); }
// set_fpr32 from a W register (any W write zero-extends the X register).
void fset32_w(Gen& g, int ws, u8 r) { g.a.str64(ws, X_GPR, fpr_at(g, r)); }
// get_fpr64 -> X register (X17 scratch).
void fget64_x(Gen& g, int xd, u8 r) {
    g.uses_fr = true;
    if (g.fr) {
        g.a.ldr64(xd, X_GPR, fpr_at(g, r));
        return;
    }
    g.a.ldr32z(xd, X_GPR, fpr_at(g, r & ~1));
    g.a.ldr32z(17, X_GPR, fpr_at(g, r | 1));
    g.a.bfi(xd, 17, 32, 32);
}
// get_fpr64 -> D register (X16/X17 scratch).
void fget64_d(Gen& g, int vd, u8 r) {
    g.uses_fr = true;
    if (g.fr) {
        g.a.ldr_d(vd, X_GPR, fpr_at(g, r));
        return;
    }
    fget64_x(g, 16, r);
    g.a.fmov_to_fp(vd, 16, true);
}
// set_fpr64 from an X register other than X16/X17 (XZR allowed).
void fset64_x(Gen& g, int xs, u8 r) {
    g.uses_fr = true;
    if (g.fr) {
        g.a.str64(xs, X_GPR, fpr_at(g, r));
        return;
    }
    g.a.lsr_imm(17, xs, 32);
    g.a.str64(17, X_GPR, fpr_at(g, r | 1));
    g.a.uxtw(16, xs);
    g.a.str64(16, X_GPR, fpr_at(g, r & ~1));
}
// set_fpr64 from a D register (X15 scratch).
void fset64_d(Gen& g, int vs, u8 r) {
    g.uses_fr = true;
    if (g.fr) {
        g.a.str_d(vs, X_GPR, fpr_at(g, r));
        return;
    }
    g.a.fmov_from_fp(15, vs, true);
    fset64_x(g, 15, r);
}

// ---------------------------------------------------------------------------
// Block exits
// ---------------------------------------------------------------------------

// Continues at X_PC + off, which is the block at `target_paddr` (-1: not a
// block that may be linked to), after `executed` instructions of this block.
void exit_static(Gen& g, s64 off, u32 executed, s64 target_paddr) {
    Assembler& a = g.a;
    a.add_any(X_PC, X_PC, off, 16);
    // TLB-mapped code only knows where the target is physically when it is in
    // the same page; elsewhere the dispatcher looks the virtual pc up.
    if (g.mapped && target_paddr >= 0 && (static_cast<u64>(target_paddr) >> 12) != (g.start_paddr >> 12)) {
        if (!g.chain) {
            a.sub_imm(X_CYC, X_CYC, 2 * executed);
            b_abs(g, g.env->rt->exit);
            return;
        }
        a.subs_imm(X_CYC, X_CYC, 2 * executed);
        g.le_exits.push_back(a.bcond(Cond::LE));
        b_abs(g, g.env->rt->dispatch);
        return;
    }
    const bool linkable = g.chain && target_paddr >= 0 && static_cast<size_t>(target_paddr) + 4 <= g.rdram_size &&
                          !is_idle_loop_at(g.rdram, g.rdram_size, static_cast<u32>(target_paddr));
    if (!linkable) {
        a.sub_imm(X_CYC, X_CYC, 2 * executed);
        b_abs(g, g.env->rt->exit);
        return;
    }
    a.subs_imm(X_CYC, X_CYC, 2 * executed);
    g.le_exits.push_back(a.bcond(Cond::LE));
    const u32 target = static_cast<u32>(target_paddr);
    if (target == g.start_paddr) {
        g.self_links.push_back(a.b());
    } else if (const void* entry = g.env->link_target(target)) {
        b_abs(g, entry);
    } else {
        // Not compiled yet: leave the chain (X_PC is already the target) until
        // the driver compiles it and retargets this B.
        g.pending.push_back({target, static_cast<u32>(a.pos())});
        b_abs(g, g.env->rt->exit);
    }
}

// Continues at the pc in `reg` (JR/JALR) after `executed` instructions.
void exit_dynamic(Gen& g, int reg, u32 executed) {
    Assembler& a = g.a;
    a.mov_reg(X_PC, reg);
    if (!g.chain) {
        a.sub_imm(X_CYC, X_CYC, 2 * executed);
        b_abs(g, g.env->rt->exit);
        return;
    }
    a.subs_imm(X_CYC, X_CYC, 2 * executed);
    g.le_exits.push_back(a.bcond(Cond::LE));
    b_abs(g, g.env->rt->dispatch);
}

// ---------------------------------------------------------------------------
// Loads and stores
// ---------------------------------------------------------------------------

// Call into a jit_load_*/jit_store_* helper: recomputes the full 64-bit
// address (rs + simm), passes rt (load: the register number; store: its value)
// and the site. Returns the branch taken on success (see check_fault).
size_t emit_mem_helper(Gen& g, const Decoded& d, bool is_store, const void* helper, u32 site) {
    Assembler& a = g.a;
    publish_pc(g);
    if (d.rs == 0) a.mov_imm(1, static_cast<u64>(static_cast<s64>(d.simm)));
    else { a.ldr64(1, X_GPR, d.rs * 8); a.add_any(1, 1, d.simm, 3); }
    if (is_store) {
        if (d.rt == 0) a.movz(2, 0, 0);
        else a.ldr64(2, X_GPR, d.rt * 8);
    } else {
        a.movz(2, d.rt, 0, false);
    }
    a.mov_reg(0, X_CTX);
    a.mov_imm(3, site, false);
    call(g, helper);
    return check_fault(g, site);
}

// KSEG0/KSEG1 RDRAM access inline. The guest address (low 32 bits - the only
// ones Bus::translate_vaddr looks at for these segments) is checked with
//   W1 = ror((addr ^ 0x80000000) & ~0x20000000, log2(size))
// which is the physical address divided by the access size when the address
// is in KSEG0 or KSEG1 and aligned; anything else (TLB-mapped segments, a
// misaligned address - its low bits rotate into the top) makes it at least
// RDRAM_SIZE >> log2(size), and the access takes the helper instead. The
// bound is the RDRAM the CPU sees (Bus::get_ram_limit), fixed per block.
size_t emit_interp_slow(Gen& g, const Decoded& d, u32 site);

// `helper` is the jit_load_*/jit_store_* slow path; null for the COP1 forms
// (LWC1/LDC1/SWC1/SDC1), whose slow path runs the whole instruction on the
// interpreter instead.
void emit_load_store(Gen& g, const Decoded& d, u32 site, const void* helper) {
    Assembler& a = g.a;
    bool is_store = false, sign = false;
    u32 size = 4;
    const bool fpu = helper == nullptr;
    switch (d.op) {
    case 0x31: break;                            // LWC1
    case 0x35: size = 8; break;                  // LDC1
    case 0x39: is_store = true; break;           // SWC1
    case 0x3D: size = 8; is_store = true; break; // SDC1
    case 0x20: size = 1; sign = true; break; // LB
    case 0x24: size = 1; break;              // LBU
    case 0x21: size = 2; sign = true; break; // LH
    case 0x25: size = 2; break;              // LHU
    case 0x23: sign = true; break;           // LW
    case 0x27: break;                        // LWU
    case 0x37: size = 8; break;              // LD
    case 0x28: size = 1; is_store = true; break; // SB
    case 0x29: size = 2; is_store = true; break; // SH
    case 0x2B: is_store = true; break;           // SW
    case 0x3F: size = 8; is_store = true; break; // SD
    default: break;
    }
    const Cont c = g.cont;
    auto slow_call = [&g, d, is_store, helper, site]() {
        return helper ? emit_mem_helper(g, d, is_store, helper, site) : emit_interp_slow(g, d, site);
    };
    // An address from $zero alone (0xFFFFxxxx or 0x0000xxxx) is never RDRAM.
    if (d.rs == 0) {
        a.patch_branch(slow_call(), a.pos());
        if (is_store) check_exit_req(g, c);
        return;
    }
    const u32 k = size == 8 ? 3 : size == 4 ? 2 : size == 2 ? 1 : 0;
    a.ldr32z(0, X_GPR, d.rs * 8);
    a.add_any(0, 0, d.simm, 3, false);
    a.eor_imm(1, 0, 0x80000000u, false);
    a.and_imm(1, 1, 0xDFFFFFFFu, false);
    if (k) a.ror_imm(1, 1, k, false);
    a.cmp_imm(1, (g.ram_limit >> k) >> 12, false, true); // 4 or 8 MB (Bus::get_ram_limit)
    const size_t to_slow = a.bcond(HS);
    // A load from memory the RDP is drawing on the GPU reports it first
    // (jit::notify_read, through the slow path's Bus::read*).
    size_t to_slow_read = ~size_t{0};
    if (!is_store && jit::g_read_checks) {
        a.lsr_imm(3, 1, 6 - k, false);
        a.ldrb_r(3, X_PAGES, 3, Extend::UXTW, false);
        a.tst_imm(3, jit::kReadWatched, false);
        to_slow_read = a.bcond(Cond::NE);
    }

    if (is_store) {
        int v;
        if (!fpu) {
            v = load64(g, d.rt, 2);
        } else if (size == 4) {
            a.ldr32z(2, X_GPR, fpr_at(g, d.rt));
            v = 2;
        } else {
            fget64_x(g, 2, d.rt);
            v = 2;
        }
        switch (size) {
        case 1: a.strb_r(v, X_MEM, 1, Extend::UXTW, false); break;
        case 2:
            if (v != XZR) { a.rev16_w(2, v); v = 2; }
            a.strh_r(v, X_MEM, 1, Extend::UXTW, true);
            break;
        case 4:
            if (v != XZR) { a.rev32(2, v); v = 2; }
            a.str32_r(v, X_MEM, 1, Extend::UXTW, true);
            break;
        default:
            if (v != XZR) { a.rev64(2, v); v = 2; }
            a.str64_r(v, X_MEM, 1, Extend::UXTW, true);
            break;
        }
        // Self-modifying code: a store into a 64-byte page some compiled block
        // was read from reports it, exactly as Bus::write* would (an aligned
        // store never spans two pages).
        a.lsr_imm(3, 1, 6 - k, false);
        a.ldrb_r(3, X_PAGES, 3, Extend::UXTW, false);
        const size_t to_smc = a.cbnz(3, false);
        const size_t cont = a.pos();
        g.stubs.push_back([&g, to_smc, cont, k, size, c]() {
            Assembler& s = g.a;
            s.patch_branch(to_smc, s.pos());
            if (k) s.lsl_imm(0, 1, k, false); else s.mov_reg(0, 1, false);
            s.movz(1, static_cast<u16>(size), 0, false);
            call(g, reinterpret_cast<const void*>(&jit_notify_code_write));
            check_exit_req(g, c);
            s.patch_branch(s.b(), cont);
        });
        g.stubs.push_back([&g, to_slow, cont, c, slow_call]() {
            g.a.patch_branch(to_slow, g.a.pos());
            g.a.patch_branch(slow_call(), g.a.pos());
            check_exit_req(g, c);
            g.a.patch_branch(g.a.b(), cont);
        });
        return;
    }

    if (fpu) {
        if (size == 4) {
            a.ldr32_r(2, X_MEM, 1, Extend::UXTW, true);
            a.rev32(2, 2);
            fset32_w(g, 2, d.rt);
        } else {
            a.ldr64_r(2, X_MEM, 1, Extend::UXTW, true);
            a.rev64(2, 2);
            fset64_x(g, 2, d.rt);
        }
    } else if (d.rt != 0) {
        switch (size) {
        case 1:
            if (sign) a.ldrsb64_r(2, X_MEM, 1, Extend::UXTW, false);
            else a.ldrb_r(2, X_MEM, 1, Extend::UXTW, false);
            break;
        case 2:
            a.ldrh_r(2, X_MEM, 1, Extend::UXTW, true);
            a.rev16_w(2, 2);
            if (sign) a.sxth(2, 2);
            break;
        case 4:
            a.ldr32_r(2, X_MEM, 1, Extend::UXTW, true);
            a.rev32(2, 2);
            if (sign) a.sxtw(2, 2);
            break;
        default:
            a.ldr64_r(2, X_MEM, 1, Extend::UXTW, true);
            a.rev64(2, 2);
            break;
        }
        store(g, d.rt, 2);
    }
    const size_t cont = a.pos();
    g.stubs.push_back([&g, to_slow, to_slow_read, cont, slow_call]() {
        g.a.patch_branch(to_slow, g.a.pos());
        if (to_slow_read != ~size_t{0}) g.a.patch_branch(to_slow_read, g.a.pos());
        g.a.patch_branch(slow_call(), cont);
    });
}

// ---------------------------------------------------------------------------
// COP1, inline. Mirrors CPU::get_fpr32/set_fpr32/get_fpr64/set_fpr64: a 32-bit
// access is the low word of fpr[r] (written zero-extended); a 64-bit one is
// fpr[r] with STATUS.FR set, else the even/odd pair (odd = high word). Only
// the latter depends on FR, which the block then checks on entry.
// ---------------------------------------------------------------------------

// C.cond.S/D: the relations the low 3 bits of cond select (bit 0 unordered,
// bit 1 equal, bit 2 less - see fp_condition in cpu.cpp) as one host condition
// after FCMP (less: N; equal: Z,C; greater: C; unordered: C,V), written to
// FCSR bit 23.
void emit_ccond(Gen& g, const Decoded& d, bool dbl) {
    Assembler& a = g.a;
    if (dbl) { fget64_d(g, 0, d.rd); fget64_d(g, 1, d.rt); }
    else { fget32(g, 0, d.rd); fget32(g, 1, d.rt); }
    a.fcmp(0, 1, dbl);
    switch (d.funct & 7) {
    case 0: a.movz(0, 0, 0, false); break;          // F
    case 1: a.cset(0, Cond::VS, false); break;      // UN
    case 2: a.cset(0, Cond::EQ, false); break;      // EQ
    case 3:                                         // UEQ: equal or unordered
        a.cset(1, Cond::VS, false);
        a.csinc(0, 1, XZR, Cond::NE, false);
        break;
    case 4: a.cset(0, Cond::MI, false); break;      // OLT
    case 5: a.cset(0, Cond::LT, false); break;      // ULT
    case 6: a.cset(0, Cond::LS, false); break;      // OLE
    default: a.cset(0, Cond::LE, false); break;     // ULE
    }
    a.ldr32z(1, X_GPR, g.fcsr_off);
    a.bfi(1, 0, 23, 1, false);
    a.str32(1, X_GPR, g.fcsr_off);
}

// .S / .D arithmetic and conversions (CPU::execute_fpu_op).
bool emit_fp_arith(Gen& g, const Decoded& d) {
    Assembler& a = g.a;
    const bool dbl = d.rs == 0x11;
    const u8 fs = d.rd, ft = d.rt, fd = d.shamt;
    auto ld = [&](int v, u8 r) { if (dbl) fget64_d(g, v, r); else fget32(g, v, r); };
    auto st = [&](int v, u8 r) { if (dbl) fset64_d(g, v, r); else fset32(g, v, r); };
    switch (d.funct) {
    case 0x00: case 0x01: case 0x02: // ADD / SUB / MUL
        ld(0, fs);
        ld(1, ft);
        if (d.funct == 0x00) a.fadd(0, 0, 1, dbl);
        else if (d.funct == 0x01) a.fsub(0, 0, 1, dbl);
        else a.fmul(0, 0, 1, dbl);
        st(0, fd);
        return true;
    case 0x03: // DIV: a divisor equal to (+-)0.0 gives 0.0 instead of Inf/NaN
        ld(0, fs);
        ld(1, ft);
        a.fdiv(2, 0, 1, dbl);
        a.fcmp_zero(1, dbl);
        a.fmov_to_fp(3, XZR, dbl);
        a.fcsel(0, 3, 2, Cond::EQ, dbl);
        st(0, fd);
        return true;
    case 0x04: ld(0, fs); a.fsqrt(0, 0, dbl); st(0, fd); return true; // SQRT
    case 0x05: ld(0, fs); a.fabs_(0, 0, dbl); st(0, fd); return true; // ABS
    case 0x06: ld(0, fs); st(0, fd); return true;                     // MOV
    case 0x07: ld(0, fs); a.fneg(0, 0, dbl); st(0, fd); return true;  // NEG
    case 0x08: case 0x09: case 0x0A: case 0x0B:   // ROUND/TRUNC/CEIL/FLOOR.L
    case 0x0C: case 0x0D: case 0x0E: case 0x0F: { // ... .W
        const bool to64 = d.funct < 0x0C;
        ld(0, fs);
        switch (d.funct & 3) {
        case 0: a.fcvtas(0, 0, to64, dbl); break; // std::round: nearest, ties away from zero
        case 1: a.fcvtzs(0, 0, to64, dbl); break;
        case 2: a.fcvtps(0, 0, to64, dbl); break;
        default: a.fcvtms(0, 0, to64, dbl); break;
        }
        if (to64) fset64_x(g, 0, fd); else fset32_w(g, 0, fd);
        return true;
    }
    case 0x20: // CVT.S.D
        if (!dbl) return false;
        fget64_d(g, 0, fs);
        a.fcvt_d_to_s(0, 0);
        fset32(g, 0, fd);
        return true;
    case 0x21: // CVT.D.S
        if (dbl) return false;
        fget32(g, 0, fs);
        a.fcvt_s_to_d(0, 0);
        fset64_d(g, 0, fd);
        return true;
    case 0x24: // CVT.W
        ld(0, fs);
        a.fcvtzs(0, 0, false, dbl);
        fset32_w(g, 0, fd);
        return true;
    case 0x25: // CVT.L
        ld(0, fs);
        a.fcvtzs(0, 0, true, dbl);
        fset64_x(g, 0, fd);
        return true;
    default:
        if ((d.funct & 0x30) == 0x30) { // C.cond
            emit_ccond(g, d, dbl);
            return true;
        }
        return false; // reserved: the interpreter raises the exception
    }
}

bool compile_cop1(Gen& g, const Decoded& d) {
    if (!g.fpu_ok()) return false;
    Assembler& a = g.a;
    switch (d.rs) {
    case 0x00: // MFC1: sign-extended low word
        if (d.rt != 0) {
            a.ldrsw(0, X_GPR, fpr_at(g, d.rd));
            store(g, d.rt, 0);
        }
        return true;
    case 0x01: // DMFC1
        if (d.rt != 0) {
            fget64_x(g, 0, d.rd);
            store(g, d.rt, 0);
        }
        return true;
    case 0x02: // CFC1 (every fs reads FCSR, as in the interpreter)
        if (d.rt != 0) {
            a.ldrsw(0, X_GPR, g.fcsr_off);
            store(g, d.rt, 0);
        }
        return true;
    case 0x04: // MTC1
        fset32_w(g, load32(g, d.rt, 0), d.rd);
        return true;
    case 0x05: // DMTC1
        fset64_x(g, load64(g, d.rt, 0), d.rd);
        return true;
    case 0x06: // CTC1 (every fs writes FCSR)
        a.str32(load32(g, d.rt, 0), X_GPR, g.fcsr_off);
        return true;
    case 0x10: case 0x11:
        return emit_fp_arith(g, d);
    case 0x14: case 0x15: { // CVT.S/CVT.D from .W (int32 in the low word) / .L (int64)
        if (d.funct != 0x20 && d.funct != 0x21) return false;
        const bool src64 = d.rs == 0x15, to_dbl = d.funct == 0x21;
        if (src64) fget64_x(g, 0, d.rd);
        else a.ldr32z(0, X_GPR, fpr_at(g, d.rd));
        a.scvtf(0, 0, src64, to_dbl);
        if (to_dbl) fset64_d(g, 0, d.shamt); else fset32(g, 0, d.shamt);
        return true;
    }
    default:
        return false; // BC1 is a branch; the rest is reserved
    }
}

// ---------------------------------------------------------------------------
// Integer instructions
// ---------------------------------------------------------------------------

// MULT/MULTU/DIV/DIVU and their 64-bit forms, exactly as CPU::execute: 32-bit
// results are sign-extended; division by zero gives lo = (num >= 0 ? -1 : 1)
// (unsigned: all ones) and hi = num; MIN / -1 gives lo = MIN, hi = 0 - which
// is also what SDIV and MSUB produce.
void emit_muldiv(Gen& g, const Decoded& d) {
    Assembler& a = g.a;
    const bool dw = d.funct >= 0x1C;
    const int vs = dw ? load64(g, d.rs, 0) : load32(g, d.rs, 0);
    const int vt = dw ? load64(g, d.rt, 1) : load32(g, d.rt, 1);
    switch (d.funct & 3) {
    case 0: case 1: // (D)MULT(U)
        if (!dw) {
            if (d.funct & 1) a.umull(2, vs, vt); else a.smull(2, vs, vt);
            a.sxtw(3, 2);       // lo
            a.asr_imm(4, 2, 32); // hi (= sign extension of the product's top word)
        } else {
            a.mul(3, vs, vt);
            if (d.funct & 1) a.umulh(4, vs, vt); else a.smulh(4, vs, vt);
        }
        break;
    default: { // (D)DIV(U)
        const bool sgn = (d.funct & 1) == 0;
        if (sgn) a.sdiv(2, vs, vt, dw); else a.udiv(2, vs, vt, dw);
        a.msub(4, 2, vt, vs, dw); // remainder (= num when den == 0, since q is 0 then)
        a.cmp_reg(vt, XZR, dw);
        if (sgn) {
            // den == 0: lo = num >= 0 ? -1 : 1  ==  ~(num >> 63) | 1
            a.asr_imm(5, vs, dw ? 63 : 31, dw);
            a.mvn_reg(5, 5, dw);
            a.orr_imm(5, 5, 1, dw);
            a.csel(2, 5, 2, Cond::EQ, dw);
        } else {
            a.csinv(2, 2, XZR, Cond::NE, dw); // den == 0: all ones
        }
        if (!dw) {
            a.sxtw(3, 2);
            a.sxtw(4, 4);
        } else {
            a.mov_reg(3, 2);
        }
        break;
    }
    }
    store_hilo(g, 3, false);
    store_hilo(g, 4, true);
}

// COP0 moves and TLB instructions through jit_cop0, which brings COUNT/RANDOM
// up to this instruction first; it asks to leave the chain when an interrupt
// could now be taken.
void emit_cop0(Gen& g, const Decoded& d) {
    Assembler& a = g.a;
    const u32 before = g.cont.executed - 1; // instructions of this block before this one
    a.mov_reg(0, X_CTX);
    a.mov_imm(1, d.raw, false);
    a.sub_imm(2, X_CYC, 2 * before);
    call(g, reinterpret_cast<const void*>(&jit_cop0));
    const size_t to = a.cbnz(0, false);
    const Cont c = g.cont;
    g.stubs.push_back([&g, to, c]() {
        g.a.patch_branch(to, g.a.pos());
        exit_at(g, c);
    });
}

// Returns false if `d` isn't compiled natively (the caller then tries an
// interpreter call, or ends the block before it). Branches are handled by
// emit_branch(), not here.
bool compile_one(Gen& g, const Decoded& d, u32 site) {
    Assembler& a = g.a;
    const u8 rs = d.rs, rt = d.rt, rd = d.rd, sa = d.shamt;

    switch (d.op) {
    case 0x00: // SPECIAL
        switch (d.funct) {
        case 0x00: // SLL (SLL $0,$0,0 is NOP)
            if (rd == 0) return true;
            if (rt == 0) { store(g, rd, XZR); return true; }
            a.ldr32z(0, X_GPR, rt * 8);
            a.sbfiz(0, 0, sa, 32 - sa); // sext32(rt << sa)
            store(g, rd, 0);
            return true;
        case 0x02: // SRL
            if (rd == 0) return true;
            if (rt == 0) { store(g, rd, XZR); return true; }
            a.ldr32z(0, X_GPR, rt * 8);
            if (sa == 0) a.sxtw(0, 0);
            else a.ubfx(0, 0, sa, 32 - sa); // bit 31 of the result is 0
            store(g, rd, 0);
            return true;
        case 0x03: // SRA
            if (rd == 0) return true;
            if (rt == 0) { store(g, rd, XZR); return true; }
            a.ldr32z(0, X_GPR, rt * 8);
            a.sbfx(0, 0, sa, 32 - sa);
            store(g, rd, 0);
            return true;
        case 0x04: case 0x06: case 0x07: { // SLLV / SRLV / SRAV (hardware takes the amount mod 32)
            if (rd == 0) return true;
            const int vt = load32(g, rt, 0), vs = load32(g, rs, 1);
            if (d.funct == 0x04) a.lslv(0, vt, vs, false);
            else if (d.funct == 0x06) a.lsrv(0, vt, vs, false);
            else a.asrv(0, vt, vs, false);
            a.sxtw(0, 0);
            store(g, rd, 0);
            return true;
        }
        case 0x14: case 0x16: case 0x17: { // DSLLV / DSRLV / DSRAV (mod 64)
            if (rd == 0) return true;
            const int vt = load64(g, rt, 0), vs = load64(g, rs, 1);
            if (d.funct == 0x14) a.lslv(0, vt, vs);
            else if (d.funct == 0x16) a.lsrv(0, vt, vs);
            else a.asrv(0, vt, vs);
            store(g, rd, 0);
            return true;
        }
        case 0x0A: case 0x0B: { // MOVZ / MOVN
            if (rd == 0) return true;
            const int vt = load64(g, rt, 0);
            if (vt == XZR) { // condition known: MOVZ always moves, MOVN never does
                if (d.funct == 0x0A) store(g, rd, load64(g, rs, 1));
                return true;
            }
            const size_t skip = d.funct == 0x0A ? a.cbnz(vt) : a.cbz(vt);
            store(g, rd, load64(g, rs, 1));
            a.patch_branch(skip, a.pos());
            return true;
        }
        case 0x0F: return true; // SYNC
        case 0x18: case 0x19: case 0x1A: case 0x1B:   // MULT / MULTU / DIV / DIVU
        case 0x1C: case 0x1D: case 0x1E: case 0x1F: { // DMULT / DMULTU / DDIV / DDIVU
            emit_muldiv(g, d);
            return true;
        }
        case 0x10: case 0x12: // MFHI / MFLO
            if (rd == 0) return true;
            load_hilo(g, 0, d.funct == 0x10);
            store(g, rd, 0);
            return true;
        case 0x11: case 0x13: // MTHI / MTLO
            store_hilo(g, load64(g, rs, 0), d.funct == 0x11);
            return true;
        case 0x20: case 0x22: { // ADD / SUB (trap on 32-bit signed overflow, even with rd == $0)
            const int vs = load32(g, rs, 0), vt = load32(g, rt, 1);
            if (d.funct == 0x20) a.adds_reg(0, vs, vt, false);
            else a.subs_reg(0, vs, vt, false);
            emit_overflow_check(g, site);
            if (rd != 0) { a.sxtw(0, 0); store(g, rd, 0); }
            return true;
        }
        case 0x21: case 0x23: { // ADDU / SUBU
            if (rd == 0) return true;
            const int vs = load32(g, rs, 0), vt = load32(g, rt, 1);
            if (d.funct == 0x21) a.add_reg(0, vs, vt, false);
            else a.sub_reg(0, vs, vt, false);
            a.sxtw(0, 0);
            store(g, rd, 0);
            return true;
        }
        case 0x24: case 0x25: case 0x26: case 0x27: { // AND / OR / XOR / NOR
            if (rd == 0) return true;
            const int vs = load64(g, rs, 0), vt = load64(g, rt, 1);
            switch (d.funct) {
            case 0x24: a.and_reg(0, vs, vt); break;
            case 0x25: a.orr_reg(0, vs, vt); break;
            case 0x26: a.eor_reg(0, vs, vt); break;
            default: a.orr_reg(0, vs, vt); a.mvn_reg(0, 0); break;
            }
            store(g, rd, 0);
            return true;
        }
        case 0x2A: case 0x2B: { // SLT / SLTU
            if (rd == 0) return true;
            const int vs = load64(g, rs, 0), vt = load64(g, rt, 1);
            a.cmp_reg(vs, vt);
            a.cset(0, d.funct == 0x2A ? Cond::LT : LO);
            store(g, rd, 0);
            return true;
        }
        case 0x2C: case 0x2E: { // DADD / DSUB (trap on 64-bit signed overflow)
            const int vs = load64(g, rs, 0), vt = load64(g, rt, 1);
            if (d.funct == 0x2C) a.adds_reg(0, vs, vt);
            else a.subs_reg(0, vs, vt);
            emit_overflow_check(g, site);
            store(g, rd, 0);
            return true;
        }
        case 0x2D: case 0x2F: { // DADDU / DSUBU
            if (rd == 0) return true;
            const int vs = load64(g, rs, 0), vt = load64(g, rt, 1);
            if (d.funct == 0x2D) a.add_reg(0, vs, vt);
            else a.sub_reg(0, vs, vt);
            store(g, rd, 0);
            return true;
        }
        case 0x38: case 0x3A: case 0x3B: case 0x3C: case 0x3E: case 0x3F: { // DSLL/DSRL/DSRA(+32)
            if (rd == 0) return true;
            if (rt == 0) { store(g, rd, XZR); return true; }
            const u32 s = sa + ((d.funct & 0x04) ? 32 : 0);
            a.ldr64(0, X_GPR, rt * 8);
            switch (d.funct & 0x03) {
            case 0x00: if (s) a.lsl_imm(0, 0, s); break;
            case 0x02: if (s) a.lsr_imm(0, 0, s); break;
            default: if (s) a.asr_imm(0, 0, s); break;
            }
            store(g, rd, 0);
            return true;
        }
        default:
            return false; // JR/JALR, traps, SYSCALL/BREAK, MULT/DIV...
        }

    case 0x08: case 0x18: { // ADDI / DADDI (trap on signed overflow, even with rt == $0)
        const bool dw = d.op == 0x18;
        const s32 v = d.simm;
        if (rs == 0) { // 0 + simm never overflows
            if (rt != 0) { a.mov_imm(0, static_cast<u64>(static_cast<s64>(v))); store(g, rt, 0); }
            return true;
        }
        if (dw) a.ldr64(0, X_GPR, rs * 8); else a.ldr32z(0, X_GPR, rs * 8);
        if (v >= 0 && v < 4096) a.adds_imm(0, 0, static_cast<u32>(v), dw);
        else if (v < 0 && v > -4096) a.subs_imm(0, 0, static_cast<u32>(-v), dw);
        else { a.mov_imm(1, static_cast<u64>(static_cast<s64>(v)), dw); a.adds_reg(0, 0, 1, dw); }
        emit_overflow_check(g, site);
        if (rt != 0) {
            if (!dw) a.sxtw(0, 0);
            store(g, rt, 0);
        }
        return true;
    }
    case 0x09: case 0x19: { // ADDIU / DADDIU
        if (rt == 0) return true;
        const bool dw = d.op == 0x19;
        if (rs == 0) {
            a.mov_imm(0, static_cast<u64>(static_cast<s64>(d.simm)));
        } else {
            if (dw) a.ldr64(0, X_GPR, rs * 8); else a.ldr32z(0, X_GPR, rs * 8);
            a.add_any(0, 0, d.simm, 1, dw);
            if (!dw) a.sxtw(0, 0);
        }
        store(g, rt, 0);
        return true;
    }
    case 0x0A: case 0x0B: { // SLTI / SLTIU (the immediate is sign-extended for both)
        if (rt == 0) return true;
        if (rs == 0) { // 0 < simm, signed or unsigned
            const bool r = d.op == 0x0A ? 0 < d.simm : 0 < static_cast<u64>(static_cast<s64>(d.simm));
            a.movz(0, r ? 1 : 0, 0);
            store(g, rt, 0);
            return true;
        }
        a.ldr64(0, X_GPR, rs * 8);
        a.cmp_any(0, d.simm, 1);
        a.cset(0, d.op == 0x0A ? Cond::LT : LO);
        store(g, rt, 0);
        return true;
    }
    case 0x0C: case 0x0D: case 0x0E: { // ANDI / ORI / XORI (zero-extended immediate)
        if (rt == 0) return true;
        if (rs == 0) { // $zero op imm
            if (d.op == 0x0C || d.imm == 0) store(g, rt, XZR);
            else { a.mov_imm(0, d.imm); store(g, rt, 0); }
            return true;
        }
        a.ldr64(0, X_GPR, rs * 8);
        if (d.imm == 0) {
            if (d.op == 0x0C) { store(g, rt, XZR); return true; }
        } else if (d.op == 0x0C) {
            a.and_imm_or(0, 0, d.imm, 1);
        } else if (d.op == 0x0D) {
            a.orr_imm_or(0, 0, d.imm, 1);
        } else {
            a.eor_imm_or(0, 0, d.imm, 1);
        }
        store(g, rt, 0);
        return true;
    }
    case 0x0F: // LUI
        if (rt == 0) return true;
        if (d.imm == 0) { store(g, rt, XZR); return true; }
        a.mov_imm(0, static_cast<u64>(sign_extend_32_64(static_cast<s32>(static_cast<u32>(d.imm) << 16))));
        store(g, rt, 0);
        return true;

    case 0x10: // COP0: moves and TLB instructions (ERET ends a block - see compile_block_a64)
        if (d.rs == 0x00 || d.rs == 0x01 || d.rs == 0x04 || d.rs == 0x05 ||
            (d.rs == 0x10 && (d.funct == 0x01 || d.funct == 0x02 || d.funct == 0x06 || d.funct == 0x08))) {
            emit_cop0(g, d);
            return true;
        }
        return false;

    case 0x11: // COP1
        return compile_cop1(g, d);
    case 0x31: case 0x35: case 0x39: case 0x3D: // LWC1 / LDC1 / SWC1 / SDC1
        if (!g.fpu_ok()) return false;
        emit_load_store(g, d, site, nullptr);
        return true;

    case 0x20: emit_load_store(g, d, site, reinterpret_cast<const void*>(&jit_load_lb)); return true;
    case 0x21: emit_load_store(g, d, site, reinterpret_cast<const void*>(&jit_load_lh)); return true;
    case 0x23: emit_load_store(g, d, site, reinterpret_cast<const void*>(&jit_load_lw)); return true;
    case 0x24: emit_load_store(g, d, site, reinterpret_cast<const void*>(&jit_load_lbu)); return true;
    case 0x25: emit_load_store(g, d, site, reinterpret_cast<const void*>(&jit_load_lhu)); return true;
    case 0x27: emit_load_store(g, d, site, reinterpret_cast<const void*>(&jit_load_lwu)); return true;
    case 0x37: emit_load_store(g, d, site, reinterpret_cast<const void*>(&jit_load_ld)); return true;
    case 0x28: emit_load_store(g, d, site, reinterpret_cast<const void*>(&jit_store_sb)); return true;
    case 0x29: emit_load_store(g, d, site, reinterpret_cast<const void*>(&jit_store_sh)); return true;
    case 0x2B: emit_load_store(g, d, site, reinterpret_cast<const void*>(&jit_store_sw)); return true;
    case 0x3F: emit_load_store(g, d, site, reinterpret_cast<const void*>(&jit_store_sd)); return true;

    case 0x2F: return true; // CACHE: a no-op, as in the interpreter

    default:
        return false;
    }
}

// Calls jit_interp for `d`; returns the branch taken when it didn't fault
// (see check_fault).
size_t emit_interp_slow(Gen& g, const Decoded& d, u32 site) {
    Assembler& a = g.a;
    publish_pc(g);
    a.mov_reg(0, X_CTX);
    a.mov_imm(1, d.raw, false);
    a.mov_imm(2, site, false);
    call(g, reinterpret_cast<const void*>(&jit_interp));
    return check_fault(g, site);
}

// Runs `d` through jit_interp (see is_interp_callable), with the same
// fault-exit convention as a load/store helper call.
void emit_interp_call(Gen& g, const Decoded& d, u32 site) {
    g.a.patch_branch(emit_interp_slow(g, d, site), g.a.pos());
    if (d.op >= 0x28 && d.op != 0x2F) check_exit_req(g, g.cont); // a store: may have hit MMIO
}

bool compile_or_call(Gen& g, const Decoded& d, u32 site) {
    if (compile_one(g, d, site)) return true;
    if (!is_interp_callable(d)) return false;
    emit_interp_call(g, d, site);
    return true;
}

// Whether `d` can be compiled at all (dry run into a scratch buffer).
bool compilable(const Gen& g, const Decoded& d) {
    Gen scratch;
    scratch.env = g.env;
    scratch.rdram = g.rdram;
    scratch.rdram_size = g.rdram_size;
    scratch.ram_limit = g.ram_limit;
    scratch.hi_off = g.hi_off;
    scratch.lo_off = g.lo_off;
    scratch.fpr_off = g.fpr_off;
    scratch.fcsr_off = g.fcsr_off;
    scratch.status_off = g.status_off;
    scratch.fr = g.fr;
    return compile_or_call(scratch, d, 0);
}

// ---------------------------------------------------------------------------
// Branches
// ---------------------------------------------------------------------------

// Emits the branch `d` at block index `n` together with its delay slot `ds`
// (compilable() must hold for it), and every exit that follows. Mirrors
// CPU::execute(): a link register is written before the branch's operands are
// read, and a likely branch that isn't taken skips its delay slot.
void emit_branch(Gen& g, const Decoded& d, const Decoded& ds, u32 n) {
    Assembler& a = g.a;
    const u32 done = n + 2; // instructions executed through the delay slot
    const s64 fallthrough = 4 * static_cast<s64>(n + 2);
    auto link = [&](u8 reg) {
        if (reg == 0) return;
        a.add_any(0, X_PC, fallthrough, 16);
        a.str64(0, X_GPR, reg * 8);
    };
    auto delay_slot = [&](bool bd, const Cont& after) {
        g.cont = after;
        compile_or_call(g, ds, (n + 1) | (bd ? kSiteDelaySlot : 0));
    };

    if (d.op == 0x02 || d.op == 0x03) { // J / JAL
        if (d.op == 0x03) link(31);
        const u32 target28 = (d.raw & 0x03FFFFFFu) << 2;
        if (g.mapped) { // the target's page is unknown: through the dispatcher
            a.and_imm(X_TGT, X_PC, 0xFFFFFFFFF0000000ull);
            a.orr_imm_or(X_TGT, X_TGT, target28, 16);
            delay_slot(true, Cont{Cont::Target, 0, done});
            exit_dynamic(g, X_TGT, done);
            return;
        }
        // X_PC's low 28 bits are start_paddr (KSEG0/KSEG1 bases have none set),
        // so replacing them with the target is a constant add.
        const s64 off = static_cast<s64>(target28) - static_cast<s64>(g.start_paddr);
        delay_slot(true, Cont{Cont::Static, off, done});
        exit_static(g, off, done, target28);
        return;
    }
    if (d.op == 0x00) { // JR / JALR
        if (d.funct == 0x09) link(d.rd);
        if (d.rs == 0) a.movz(X_TGT, 0, 0);
        else a.ldr64(X_TGT, X_GPR, d.rs * 8);
        delay_slot(true, Cont{Cont::Target, 0, done});
        exit_dynamic(g, X_TGT, done);
        return;
    }

    // Conditional branches.
    bool likely;
    enum { Never, Always, Test } kind = Test;
    size_t to_taken = 0;
    if (d.op == 0x11) { // BC1F/BC1T(L): FCSR bit 23
        likely = (d.rt & 0x02) != 0;
        a.ldr32z(0, X_GPR, g.fcsr_off);
        to_taken = (d.rt & 0x01) ? a.tbnz(0, 23) : a.tbz(0, 23);
    } else if (d.op == 0x01) { // REGIMM: BLTZ/BGEZ, +L, +AL, +ALL
        if (d.rt & 0x10) link(31);
        likely = (d.rt & 0x02) != 0;
        const bool gez = (d.rt & 0x01) != 0;
        if (d.rs == 0) kind = gez ? Always : Never;
        else {
            a.ldr64(0, X_GPR, d.rs * 8);
            to_taken = gez ? a.tbz(0, 63) : a.tbnz(0, 63);
        }
    } else {
        likely = d.op >= 0x14;
        switch (d.op & 0x07) {
        case 0x04: case 0x05: { // BEQ / BNE
            const bool eq = (d.op & 0x07) == 0x04;
            if (d.rs == d.rt) { kind = eq ? Always : Never; break; }
            const u8 r = d.rs == 0 ? d.rt : d.rs; // compare against zero?
            if (d.rs == 0 || d.rt == 0) {
                a.ldr64(0, X_GPR, r * 8);
                to_taken = eq ? a.cbz(0) : a.cbnz(0);
            } else {
                a.ldr64(0, X_GPR, d.rs * 8);
                a.ldr64(1, X_GPR, d.rt * 8);
                a.cmp_reg(0, 1);
                to_taken = a.bcond(eq ? Cond::EQ : Cond::NE);
            }
            break;
        }
        default: { // BLEZ / BGTZ
            const bool lez = (d.op & 0x07) == 0x06;
            if (d.rs == 0) { kind = lez ? Always : Never; break; }
            a.ldr64(0, X_GPR, d.rs * 8);
            a.cmp_imm(0, 0);
            to_taken = a.bcond(lez ? Cond::LE : Cond::GT);
            break;
        }
        }
    }

    const s64 taken_off = 4 * static_cast<s64>(n + 1) + 4 * static_cast<s64>(d.simm);
    const s64 taken_paddr = static_cast<s64>(g.start_paddr) + taken_off;
    const u32 fall_paddr = g.start_paddr + static_cast<u32>(fallthrough);

    if (kind != Always) { // not taken
        if (likely) {
            exit_static(g, fallthrough, n + 1, fall_paddr);
        } else {
            delay_slot(false, Cont{Cont::Static, fallthrough, done});
            exit_static(g, fallthrough, done, fall_paddr);
        }
    }
    if (kind != Never) { // taken
        if (kind == Test) a.patch_branch(to_taken, a.pos());
        delay_slot(true, Cont{Cont::Static, taken_off, done});
        exit_static(g, taken_off, done, taken_paddr);
    }
}

// ERET ends a block: jit_eret runs it (COUNT/RANDOM brought up to date as for
// jit_cop0) and leaves the new pc in JitCtx::next_pc; the chain goes on there
// unless an interrupt can now be taken.
void emit_eret(Gen& g, u32 index) {
    Assembler& a = g.a;
    a.mov_reg(0, X_CTX);
    a.sub_imm(1, X_CYC, 2 * index);
    call(g, reinterpret_cast<const void*>(&jit_eret));
    a.ldr64(X_TGT, X_CTX, kCtxNextPc);
    const size_t to = a.cbnz(0, false);
    exit_dynamic(g, X_TGT, index + 1);
    const Cont c{Cont::Target, 0, index + 1};
    g.stubs.push_back([&g, to, c]() {
        g.a.patch_branch(to, g.a.pos());
        exit_at(g, c);
    });
}

} // namespace

// ---------------------------------------------------------------------------
// Runtime: entry trampoline, chain exit, dispatcher
// ---------------------------------------------------------------------------

bool emit_a64_runtime(CodeBuffer& code, Recompiler::ChainRuntime& rt, u32 jcache_bits) {
    Assembler a;
    // void enter(JitCtx* ctx /*X0*/, const void* entry /*X1*/)
    a.stp64_pre(29, 30, SP, -static_cast<s32>(kFrameSize));
    a.mov_sp(29, SP);
    a.stp64(19, 20, SP, 16);
    a.stp64(21, 22, SP, 32);
    a.stp64(23, 24, SP, 48);
    a.stp64(25, 26, SP, 64);
    a.stp64(27, 28, SP, 80);
    a.mov_reg(X_CTX, 0);
    a.ldr64(X_GPR, 0, kCtxGpr);
    a.ldr64(X_MEM, 0, kCtxRdram);
    a.ldr64(X_PAGES, 0, kCtxPages);
    a.ldr64(X_PC, 0, kCtxStartPc);
    a.ldr64(X_CYC, 0, kCtxCycles);
    a.ldr64(X_JC, 0, kCtxJCache);
    a.br(1);

    // exit: hands the next pc and the budget left back, returns from enter().
    const size_t exit_off = a.pos();
    a.str64(X_PC, X_CTX, kCtxNextPc);
    a.str64(X_CYC, X_CTX, kCtxCycles);
    a.ldp64(27, 28, SP, 80);
    a.ldp64(25, 26, SP, 64);
    a.ldp64(23, 24, SP, 48);
    a.ldp64(21, 22, SP, 32);
    a.ldp64(19, 20, SP, 16);
    a.ldp64_post(29, 30, SP, static_cast<s32>(kFrameSize));
    a.ret();

    // dispatch: X_PC -> jcache[(X_PC >> 2) & mask] = {pc, entry}; a miss leaves.
    const size_t dispatch_off = a.pos();
    a.ubfx(16, X_PC, 2, jcache_bits);
    a.add_reg_lsl(16, X_JC, 16, 4);
    a.ldp64(16, 17, 16, 0);
    a.cmp_reg(16, X_PC);
    a.patch_branch(a.bcond(Cond::NE), exit_off);
    a.br(17);

    if (code.remaining() < a.size()) return false;
    code.make_writable(a.size());
    u8* dst = code.write_ptr();
    std::memcpy(dst, a.data(), a.size());
    code.commit(a.size());
    code.make_executable();
    rt.enter = reinterpret_cast<void (*)(JitCtx*, const void*)>(code.exec_ptr(dst));
    rt.exit = code.exec_ptr(dst) + exit_off;
    rt.dispatch = code.exec_ptr(dst) + dispatch_off;
    return true;
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

Recompiler::Block compile_block_a64(CPU& cpu, Bus& bus, u32 start_paddr, CodeBuffer& code, bool delay_slot,
                                    const A64BlockEnv& env, std::vector<Recompiler::PendingLink>& pending) {
    Recompiler::Block blk{};
    Gen g;
    g.env = &env;
    g.rdram = bus.get_rdram();
    g.rdram_size = bus.get_rdram_size();
    g.ram_limit = bus.get_ram_limit();
    g.start_paddr = start_paddr;
    g.mapped = env.mapped;
    g.code_end = env.mapped ? std::min<u32>((start_paddr & ~0xFFFu) + 0x1000, static_cast<u32>(g.rdram_size))
                            : static_cast<u32>(g.rdram_size);
    // A taken branch's delay slot run on its own (its branch was interpreted)
    // continues at JitCtx::next_pc; the idle loop must come back to the driver
    // every time around. Neither ever links.
    const bool idle = !delay_slot && is_idle_loop_at(g.rdram, g.rdram_size, start_paddr);
    g.chain = env.chain && !delay_slot && !idle;
    {
        // Offsets of other CPU state from the register file: usable with a
        // scaled 12-bit LDR/STR offset (for 4-byte ones the smallest limit).
        const auto* gpr = reinterpret_cast<const u8*>(cpu.jit_gpr_ptr());
        auto off = [gpr](const void* p, s64 align) -> s32 {
            const s64 o = reinterpret_cast<const u8*>(p) - gpr;
            return (o >= 0 && o < 16384 && o % align == 0) ? static_cast<s32>(o) : -1;
        };
        g.hi_off = off(cpu.jit_hi_ptr(), 8);
        g.lo_off = off(cpu.jit_lo_ptr(), 8);
        g.fpr_off = off(cpu.jit_fpr_ptr(), 8);
        g.fcsr_off = off(cpu.jit_fcsr_ptr(), 4);
        g.status_off = off(cpu.jit_cp0_ptr() + CP0Reg::STATUS, 8);
        g.fr = (cpu.get_cp0(CP0Reg::STATUS) & (1u << 26)) != 0;
    }
    Assembler& a = g.a;
    const u32 max_len = delay_slot ? 1 : kMaxBlockLen;
    // Room for the STATUS.FR check, filled in once the body shows it is needed.
    constexpr u32 kGuardWords = 3;
    for (u32 i = 0; i < kGuardWords; ++i) a.emit(0xD503201Fu); // NOP

    u32 len = 0;
    bool ended = false;    // the last thing emitted was a branch with all its exits
    bool cut = false;      // stopped by the length limit or the page end, not an instruction
    while (true) {
        const u32 paddr = start_paddr + len * 4;
        if (len >= max_len || paddr + 4 > g.code_end) {
            cut = paddr + 4 <= g.rdram_size;
            break;
        }
        const Decoded d = decode(fetch_instr(g.rdram, paddr));
        const u32 site = delay_slot ? (len | kSiteDelaySlot) : len;
        g.cont = delay_slot ? Cont{Cont::CtxNext, 0, len + 1} : Cont{Cont::Static, 4 * static_cast<s64>(len + 1), len + 1};
        if (compile_or_call(g, d, site)) {
            len++;
            continue;
        }
        if (!delay_slot && d.op == 0x10 && d.rs == 0x10 && d.funct == 0x18) { // ERET
            emit_eret(g, len);
            len++;
            ended = true;
            break;
        }
        // A branch ends the block, together with its delay slot - so only if
        // that is compilable too; otherwise the interpreter runs the branch.
        const bool bc1 = d.op == 0x11 && d.rs == 0x08 && d.rt <= 3 && g.fpu_ok();
        if (delay_slot || !(is_block_branch(d) || bc1) || paddr + 8 > g.code_end) break;
        const Decoded ds = decode(fetch_instr(g.rdram, paddr + 4));
        if (!compilable(g, ds)) break;
        emit_branch(g, d, ds, len);
        len += 2;
        ended = true;
        break;
    }
    if (len == 0) return blk;
    // Entry: at the STATUS.FR check if the code depends on FR, else past it.
    const u32 entry = g.uses_fr ? 0 : 4 * kGuardWords;
    if (g.uses_fr) {
        a.set_word(0, 0); // placeholders, rewritten below
        Assembler guard;
        guard.ldr32z(16, X_GPR, g.status_off);
        guard.tst_imm(16, 1u << 26, false);
        for (u32 i = 0; i < 2; ++i) a.set_word(4 * i, guard.word_at(4 * i));
        // B.cond to the mismatch stub: taken when FR differs from g.fr.
        a.set_word(8, 0x54000000u | static_cast<u32>(g.fr ? Cond::EQ : Cond::NE));
        g.stubs.push_back([&g]() {
            // Compiled for the other FR mode: leave before the first
            // instruction; the driver drops the code cache and recompiles.
            g.a.patch_branch(8, g.a.pos());
            g.a.movz(16, 1, 0, false);
            g.a.str32(16, X_CTX, kCtxFrMismatch);
            b_abs(g, g.env->rt->exit);
        });
    }

    if (!ended) {
        if (delay_slot) {
            exit_at(g, Cont{Cont::CtxNext, 0, len});
        } else {
            // Stopped at the length limit or the page end (the next block can
            // be linked to) or before an instruction only the interpreter runs
            // (it can't).
            exit_static(g, 4 * static_cast<s64>(len), len,
                        cut && !delay_slot ? static_cast<s64>(start_paddr) + 4 * static_cast<s64>(len) : -1);
        }
    }

    // Out-of-line code. A stub may queue further stubs, so each is moved out
    // of the vector before it runs.
    for (size_t i = 0; i < g.stubs.size(); ++i) {
        auto stub = std::move(g.stubs[i]);
        stub();
    }
    // Trapping arithmetic that overflowed: raise EXC_OV through the
    // interpreter's own trigger_exception() (jit_overflow).
    for (const OverflowSite& s : g.overflows) {
        a.patch_branch(s.patch, a.pos());
        publish_pc(g);
        a.mov_reg(0, X_CTX);
        a.mov_imm(1, s.site, false);
        call(g, reinterpret_cast<const void*>(&jit_overflow));
        exit_fault(g, site_index(s.site) + 1);
    }
    // Budget used up: every exit's B.LE lands on one shared B to the chain exit.
    if (!g.le_exits.empty()) {
        const size_t local_exit = a.pos();
        b_abs(g, env.rt->exit);
        for (size_t site : g.le_exits) a.patch_branch(site, local_exit);
    }
    for (size_t site : g.self_links) a.patch_branch(site, entry);

    // lookup_or_compile() keeps kArenaHeadroom free, so this only triggers if
    // a block ever outgrows it; the interpreter then runs that code.
    if (code.remaining() < a.size()) return blk;
    u8* dst = code.write_ptr();
    const u8* exec = code.exec_ptr(dst);
    for (const auto& [site, target] : g.abs_branches) a.set_word(site, Assembler::b_word(exec + site, target));
    code.make_writable(a.size());
    std::memcpy(dst, a.data(), a.size());
    code.commit(a.size());
    code.make_executable();

    blk.fn = reinterpret_cast<JitBlockFn>(const_cast<u8*>(exec + entry));
    blk.length = len;
    blk.link_ok = g.chain;
    for (Recompiler::PendingLink l : g.pending) {
        l.offset -= entry; // relative to fn
        pending.push_back(l);
    }
    return blk;
}

#endif // __aarch64__
