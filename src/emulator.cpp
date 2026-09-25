#include "emulator.hpp"
#include "savestate.hpp"
#include <chrono>
#include <iostream>

Emulator::Emulator()
    : bus(cart, pif, controllers, mi, vi, ai, pi, si, rsp, rdp),
      cpu(bus) {
}

Emulator::~Emulator() {
    cart.save_backup();
}

bool Emulator::load_rom(const std::string& rom_path) {
    if (!cart.load_rom(rom_path)) {
        return false;
    }

    reset();
    return true;
}

void Emulator::reset() {
    bus.reset();
    mi.reset();
    vi.reset();
    ai.reset();
    pi.reset();
    si.reset();
    rsp.reset();
    rdp.reset();
    pif.reset(cart.get_cic_type());
    jit.invalidate_all(); // RDRAM contents are about to be rewritten below
    for (auto& ctrl : controllers) ctrl.reset();

    // Copy IPL3/payload into RDRAM and RSP DMEM as IPL3 leaves it
    const auto& rom = cart.get_rom_data();
    u32 entry = cart.get_entry_point();
    if (entry == 0) entry = 0x80000400;
    u32 phys_entry = entry & 0x1FFFFFFF;

    if (rom.size() >= 0x1000) {
        size_t copy_size = std::min(static_cast<size_t>(0x100000), rom.size() - 0x1000);
        u8* rdram = bus.get_rdram();
        
        // Copy to entry point destination in RDRAM
        if (phys_entry + copy_size <= bus.get_rdram_size()) {
            std::memcpy(rdram + phys_entry, rom.data() + 0x1000, copy_size);
        }
        // Also copy to 0x00000400
        if (0x400 + copy_size <= bus.get_rdram_size()) {
            std::memcpy(rdram + 0x400, rom.data() + 0x1000, copy_size);
        }

        // Copy header and IPL3 into RSP DMEM (0xA4000000)
        std::memcpy(rsp.get_dmem(), rom.data(), 0x1000);
    }

    // Set up standard IPL3 / OS parameters in RDRAM (0x80000300 - 0x80000320)
    u8* rdram = bus.get_rdram();
    auto write_u32 = [&](u32 a, u32 v) {
        rdram[a + 0] = (v >> 24) & 0xFF;
        rdram[a + 1] = (v >> 16) & 0xFF;
        rdram[a + 2] = (v >> 8) & 0xFF;
        rdram[a + 3] = v & 0xFF;
    };
    write_u32(0x300, 1);          // osTvType (1 = NTSC)
    write_u32(0x304, 0);          // osRomType (0 = cartridge)
    write_u32(0x308, 0x10000000); // osRomBase (Cartridge Domain 1)
    write_u32(0x30C, 0);          // osResetType
    write_u32(0x310, (cart.get_cic_type() == CICType::CIC_6105) ? 0x91 : 0x3F); // osCicId
    write_u32(0x314, 0);          // osVersion
    write_u32(0x318, bus.get_rdram_size()); // osMemSize (8MB = 0x00800000)

    // Reset CPU to start at entry point
    cpu.reset(entry);

    // Initialise CP0 Status, Config and Count
    cpu.set_cp0(CP0Reg::STATUS, 0x34000000); // FR=1, CU1=1, CU0=1
    cpu.set_cp0(CP0Reg::CONFIG, 0x0006E463);
    cpu.set_cp0(CP0Reg::COUNT, 0x02000000);  // Advance past 0.5s boot delay for osContInit

    // Initial GPR setup according to IPL3 specification
    cpu.set_gpr(20, 0x00000000); // s4
    cpu.set_gpr(22, (cart.get_cic_type() == CICType::CIC_6105) ? 0x91 : 0x3F); // s6 (CIC seed)
    cpu.set_gpr(29, (entry >= 0x80200000) ? 0x80200600 : 0x8033B400); // sp
    cpu.set_gpr(31, entry); // ra
}

int g_current_frame = 0;

void Emulator::step_frame() {
    g_current_frame++;
    constexpr u32 scanlines = 525;
    constexpr u32 cycles_per_scanline = CYCLES_PER_FRAME / scanlines;

    using Clock = std::chrono::steady_clock;
    for (u32 line = 0; line < scanlines; ++line) {
        const Clock::time_point t0 = profiling_ ? Clock::now() : Clock::time_point{};
        u32 executed = 0;
        // An idle loop alternates between two pcs, so skip_idle_loop() only
        // runs when pc repeats the one two steps back - and a two-pc loop it
        // has turned down isn't asked about again for the rest of the scanline.
        u64 prev_pc = ~0ULL, prev_prev_pc = ~0ULL, rejected_pc = ~0ULL;
        while (executed < cycles_per_scanline) {
            const u64 pc = cpu.get_pc();
            if (pc == prev_prev_pc && pc != rejected_pc) {
                if (u32 skipped = skip_idle_loop(cycles_per_scanline - executed)) {
                    executed += skipped;
                    continue;
                }
                rejected_pc = pc;
            }
            prev_prev_pc = prev_pc;
            prev_pc = pc;
            executed += (cpu_core_ == CpuCore::Recompiler) ? jit.run_step(cpu, bus) : cpu.step();
        }
        const Clock::time_point t1 = profiling_ ? Clock::now() : Clock::time_point{};

        ai.step(cycles_per_scanline, mi, bus.get_rdram(), bus.get_rdram_size());
        vi.step_scanline(mi);
        rsp.step(cycles_per_scanline, mi, rdp, bus.get_rdram(), bus.get_rdram_size());
        cpu.check_interrupts();

        if (profiling_) {
            prof_cpu_seconds_ += std::chrono::duration<double>(t1 - t0).count();
            prof_other_seconds_ += std::chrono::duration<double>(Clock::now() - t1).count();
        }
    }

    static int f_count = 0;
    f_count++;
    if (f_count <= 5 || f_count % 60 == 0) {
        std::cout << "[Frame " << std::dec << f_count
                  << "] PC=0x" << std::hex << cpu.get_pc()
                  << " VI_origin=0x" << vi.get_origin()
                  << " VI_status=0x" << vi.get_status()
                  << " SP_status=0x" << rsp.get_status()
                  << " MI_intr=0x" << mi.get_intr()
                  << std::dec << "\n";
    }
}

// Idle loops - libultra's idle thread spinning on "b . / nop" while every
// other thread is blocked, or busy-waits like DK64's "bne v1, v0, . / nop" -
// are often more than half of everything a game executes. Nothing in such a
// loop changes machine state, so only an interrupt can end it, and between
// scanline boundaries (where AI/VI/RSP are stepped) the only one that can
// come up is the COUNT/COMPARE timer. So the rest of the scanline, or
// everything up to the timer if that comes first, can be consumed at once,
// ending in the same state as running the loop instruction by instruction.
u32 Emulator::skip_idle_loop(u32 budget) {
    if (cpu.jit_pending_delay_slot()) return 0; // must be on the branch, not its delay slot
    // Fetched exactly the way CPU::step() does, so loops in TLB-mapped code
    // (GoldenEye runs its idle thread at 0x7000xxxx) are recognised too.
    const u64 pc = cpu.get_pc();
    const u8 asid = static_cast<u8>(cpu.get_cp0(CP0Reg::ENTRY_HI) & 0xFF);
    TLBResult branch_res = TLBResult::SUCCESS, slot_res = TLBResult::SUCCESS;
    const u32 branch = bus.read_v32(pc, branch_res, asid);
    const u32 slot = bus.read_v32(pc + 4, slot_res, asid);
    if (branch_res != TLBResult::SUCCESS || slot_res != TLBResult::SUCCESS) return 0;
    // BEQ/BNE to itself (offset -1) with a NOP in the delay slot: nothing in
    // the loop touches the registers it compares, so if it's taken now it
    // stays taken until an interrupt.
    const u32 op = branch >> 26;
    if (slot != 0 || (op != 0x04 && op != 0x05) || (branch & 0xFFFF) != 0xFFFF) return 0;
    const bool equal = cpu.get_gpr((branch >> 21) & 0x1F) == cpu.get_gpr((branch >> 16) & 0x1F);
    if (equal != (op == 0x04)) return 0; // not taken: the loop is exiting

    // COUNT ticks once per 2 cycles; stop exactly where it reaches COMPARE so
    // the timer interrupt is raised on the same instruction as it would be.
    // (COUNT == COMPARE now means that match was just raised; the next one is
    // 2^32 ticks away.)
    const u32 ticks = static_cast<u32>(cpu.get_cp0(CP0Reg::COMPARE)) - static_cast<u32>(cpu.get_cp0(CP0Reg::COUNT));
    const u64 to_timer = ticks ? 2ULL * ticks : (2ULL << 32);
    // Whole loop iterations only (2 instructions x 2 cycles): pc stays on the branch.
    const u32 cycles = static_cast<u32>(std::min<u64>(budget, to_timer)) & ~3u;
    if (cycles == 0) return 0;

    // CP0 RANDOM steps once per instruction, from 31 down to WIRED and then
    // back to 31 (see CPU::step).
    const u32 wired = static_cast<u32>(cpu.get_cp0(CP0Reg::WIRED)) & 0x1F;
    const u32 random = static_cast<u32>(cpu.get_cp0(CP0Reg::RANDOM)) & 0x1F;
    const u32 span = 32 - wired;
    const u32 pos = random > wired ? random - wired : 0;
    cpu.set_cp0(CP0Reg::RANDOM, wired + (pos + span - (cycles / 2) % span) % span);

    cpu.step_timer(cycles);
    cpu.check_interrupts();
    return cycles;
}

void Emulator::render_frame(std::vector<u32>& out_pixels, int& out_w, int& out_h, int* out_scale) {
    const u8* rdram = bus.get_rdram();
    const size_t rdram_size = bus.get_rdram_size();
    HiResRenderer* hr = rdp.hires();
    if (!hr) {
        vi.render_frame(rdram, rdram_size, out_pixels, out_w, out_h);
    } else if (!hr->compose(vi.scanout(rdram_size), rdram, rdram_size, out_pixels, out_w, out_h)) {
        // The VI shows memory the RDP never drew into (a screen the CPU
        // drew): enlarge the native image so the output size stays the same.
        int w = 0, h = 0;
        vi.render_frame(rdram, rdram_size, native_frame_, w, h);
        const int s = static_cast<int>(hr->scale());
        out_w = w * s;
        out_h = h * s;
        out_pixels.resize(static_cast<size_t>(out_w) * out_h);
        for (int y = 0; y < out_h; ++y) {
            const u32* src = native_frame_.data() + static_cast<size_t>(y / s) * w;
            u32* dst = out_pixels.data() + static_cast<size_t>(y) * out_w;
            for (int x = 0; x < out_w; ++x) dst[x] = src[x / s];
        }
    }
    if (hr) hr->end_frame();
    if (out_scale) *out_scale = static_cast<int>(rdp.hires_scale());
    rdp.clear_zbuffer();
}

template <class S> void Emulator::serialize(S& s) {
    // A state only fits the cartridge it was made with.
    u32 crc1 = cart.get_crc1(), crc2 = cart.get_crc2();
    u64 rom_size = cart.get_rom_size();
    s.begin_section("ROM ");
    s(crc1, crc2, rom_size);
    s.end_section();
    if constexpr (S::loading) {
        if (crc1 != cart.get_crc1() || crc2 != cart.get_crc2() || rom_size != cart.get_rom_size()) {
            s.fail("the state belongs to a different game");
            return;
        }
    }
    // Controllers are left out: they follow the player's input, not the state.
    auto section = [&s](const char (&tag)[5], auto& part) {
        s.begin_section(tag);
        part.serialize(s);
        s.end_section();
    };
    section("CPU ", cpu);
    section("BUS ", bus);
    section("MI  ", mi);
    section("VI  ", vi);
    section("AI  ", ai);
    section("PI  ", pi);
    section("SI  ", si);
    section("PIF ", pif);
    section("RSP ", rsp);
    section("RDP ", rdp);
    section("CART", cart);
}

std::vector<u8> Emulator::save_state() {
    savestate::Writer w;
    w.data().reserve(bus.get_rdram_size() + (4u << 20));
    serialize(w);
    return std::move(w.data());
}

bool Emulator::load_state(const std::vector<u8>& state, std::string& error) {
    // The state is read straight into the components, so keep the current
    // machine to go back to if it turns out not to fit.
    std::vector<u8> current = save_state();
    savestate::Reader r(state.data(), state.size());
    serialize(r);
    if (r.ok() && !r.at_end()) r.fail("the state has unexpected data at the end");
    const bool ok = r.ok();
    if (!ok) {
        error = r.error();
        savestate::Reader back(current.data(), current.size());
        serialize(back);
    }
    jit.invalidate_all(); // RDRAM holds other code now
    rdp.state_loaded();
    return ok;
}

void Emulator::get_audio_samples(float* out_stream, size_t count) {
    ai.get_samples(out_stream, count);
}
