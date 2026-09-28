#include "emulator.hpp"
#include "savestate.hpp"
#include <chrono>
#include <iostream>

Emulator::Emulator()
    : bus(cart, pif, controllers, mi, vi, ai, pi, si, rsp, rdp),
      cpu(bus) {
    // Controller 1 has a Controller Pak unless the front end says otherwise.
    controllers[0].set_accessory(Accessory::ControllerPak);
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
    bus.set_ram_limit(expansion_pak_ ? RDRAM_SIZE : 0x400000u);
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
    // IPL3 copies the first MB after the boot code to where the game starts
    // (the header's entry point, minus 1/2 MB for CIC 6103/6106) and jumps there.
    const u32 entry = cart.get_boot_address();
    u32 phys_entry = entry & 0x1FFFFFFF;

    if (rom.size() >= 0x1000) {
        size_t copy_size = std::min(static_cast<size_t>(0x100000), rom.size() - 0x1000);
        u8* rdram = bus.get_rdram();
        
        // Copy to entry point destination in RDRAM
        if (phys_entry + copy_size <= bus.get_rdram_size()) {
            std::memcpy(rdram + phys_entry, rom.data() + 0x1000, copy_size);
        }

        // Copy header and IPL3 into RSP DMEM (0xA4000000)
        std::memcpy(rsp.get_dmem(), rom.data(), 0x1000);

        // 6105's IPL3 copies the tail of itself (DMEM 0x554-0x887) to RDRAM
        // 0x4 and finishes running from there; Perfect Dark checks one of
        // those words (0x2E8) and hangs on purpose if it isn't there. The OS
        // parameters written below land on top of it, as they do on hardware.
        if (cart.get_cic_type() == CICType::CIC_6105)
            std::memcpy(rdram + 0x4, rom.data() + 0x554, 0x888 - 0x554);

        // Just before jumping to the game, the 6101/6102 IPL3 clears SP DMEM
        // and IMEM, and the 6103/6106 ones fill them with 0xFF and store
        // their osCicId (below) in the first IMEM word. Diddy Kong Racing
        // checks that the first DMEM word is -1 and, if not, burns ~10
        // million cycles every frame (a new picture only every 11 VIs);
        // Yoshi's Story checks it too and never starts its graphics thread.
        // (6105 is left as it was: not checked.)
        const CICType cic = cart.get_cic_type();
        if (cic == CICType::CIC_6101 || cic == CICType::CIC_6102) {
            std::memset(rsp.get_dmem(), 0x00, 0x1000);
            std::memset(rsp.get_imem(), 0x00, 0x1000);
        } else if (cic == CICType::CIC_6103 || cic == CICType::CIC_6106) {
            std::memset(rsp.get_dmem(), 0xFF, 0x1000);
            std::memset(rsp.get_imem(), 0xFF, 0x1000);
            const u32 id = cart.get_cic_id();
            const u8 cic_number[4] = {static_cast<u8>(id >> 24), static_cast<u8>(id >> 16),
                                      static_cast<u8>(id >> 8), static_cast<u8>(id)};
            std::memcpy(rsp.get_imem(), cic_number, 4);
        }
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
    write_u32(0x308, 0xB0000000); // osRomBase: the cartridge via KSEG1, as IPL3 sets it (Turok reads through it)
    write_u32(0x30C, 0);          // osResetType
    write_u32(0x310, cart.get_cic_id());   // osCicId (e.g. 6105 = 0x17D9, 6106 = 0x17D8)
    write_u32(0x314, 0);          // osVersion
    // osMemSize: 8 MB with the Expansion Pak, 4 MB without.
    const u32 mem_size = bus.get_ram_limit();
    write_u32(0x318, mem_size);
    if (cart.get_cic_type() == CICType::CIC_6105) {
        // 6105's IPL3 stores the memory size at 0x3F0 instead, and leaves a
        // few instructions at the start of SP IMEM that its games check.
        write_u32(0x3F0, mem_size);
        static const u32 imem_words[] = {0x3C0DBFC0, 0x8DA807FC, 0x25AD07C0, 0x31080080,
                                         0x5500FFFC, 0x3C0DBFC0, 0x8DA80024, 0x3C0BB000};
        u8* imem = rsp.get_imem();
        for (size_t i = 0; i < 8; ++i)
            for (int b = 0; b < 4; ++b) imem[i * 4 + b] = static_cast<u8>(imem_words[i] >> (24 - 8 * b));
        // One of its instructions (`sw s7, 0x14(t0)`) also ends up in RDRAM
        // at 0x2FE1C0; Donkey Kong 64 hangs on purpose if it isn't there.
        write_u32(0x2FE1C0, 0xAD170014);
    }

    // Reset CPU to start at entry point
    cpu.reset(entry);

    // Initialise CP0 Status, Config and Count
    cpu.set_cp0(CP0Reg::STATUS, 0x34000000); // FR=1, CU1=1, CU0=1
    cpu.set_cp0(CP0Reg::CONFIG, 0x0006E463);
    cpu.set_cp0(CP0Reg::COUNT, 0x02000000);  // Advance past 0.5s boot delay for osContInit

    // Initial GPR setup according to IPL3 specification
    cpu.set_gpr(20, 0x00000001); // s4: TV type (NTSC)
    cpu.set_gpr(22, cart.get_cic_seed()); // s6: CIC seed
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
            executed += (cpu_core_ == CpuCore::Recompiler) ? jit.run(cpu, bus, cycles_per_scanline - executed) : cpu.step();
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

void Emulator::render_frame(VideoFrame& out) {
    const u8* rdram = bus.get_rdram();
    const size_t rdram_size = bus.get_rdram_size();
    HiResRenderer* hr = rdp.hires();
    const VIScanout so = vi.scanout(rdram_size);
    if (!hr || so.blank || so.lines > kFbLines || !hr->present(so, rdram, rdram_size, out)) {
        // Native resolution, or the VI shows memory the RDP never drew into
        // (a screen the CPU drew, or a 480-line screen, which the
        // high-resolution buffers - 240 lines tall - do not hold): the
        // native image.
        out.gpu.reset();
        vi.render_frame(rdram, rdram_size, out.pixels, out.w, out.h);
        out.scale = 1;
    }
    if (hr) hr->end_frame();
    rdp.clear_zbuffer();
}

void Emulator::render_frame(std::vector<u32>& out_pixels, int& out_w, int& out_h, int* out_scale) {
    VideoFrame f;
    f.pixels.swap(out_pixels);
    render_frame(f);
    if (f.gpu) f.gpu->read(f.pixels);
    out_pixels.swap(f.pixels);
    out_w = f.w;
    out_h = f.h;
    if (out_scale) *out_scale = f.scale;
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
    // Added after state version 6 was out, at the end, so its states (which
    // stop here: 8 MB, one Controller Pak) still load. The RDRAM size belongs
    // to the machine in the state, not to the current setting.
    u32 ram_limit = bus.get_ram_limit();
    if constexpr (S::loading) {
        ram_limit = RDRAM_SIZE;
        if (s.at_end()) {
            bus.set_ram_limit(ram_limit);
            return;
        }
    }
    // Accessories: controllers 2-4's Controller Paks, the Transfer Paks.
    s.begin_section("PAKS");
    s(ram_limit);
    cart.serialize_extra_paks(s);
    for (auto& c : controllers) c.transfer_pak().serialize(s);
    s.end_section();
    if constexpr (S::loading) bus.set_ram_limit(ram_limit == 0x400000u ? ram_limit : RDRAM_SIZE);
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

