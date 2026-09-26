#pragma once

#include "common.hpp"
#include "cartridge.hpp"
#include "pif.hpp"
#include "controller.hpp"
#include "mi.hpp"
#include "vi.hpp"
#include "ai.hpp"
#include "pi.hpp"
#include "si.hpp"
#include "rsp.hpp"
#include "rdp.hpp"
#include "bus.hpp"
#include "cpu.hpp"
#include "jit/recompiler.hpp"

enum class CpuCore { Interpreter, Recompiler };

class Emulator {
public:
    Emulator();
    ~Emulator();

    bool load_rom(const std::string& rom_path);
    void reset();

    void step_frame();
    // Runs one core-step (interpreter instruction, or one recompiled block)
    // using whichever core is currently selected. Exposed mainly for
    // diagnostics/tests that need finer granularity than step_frame().
    u32 step_one() { return (cpu_core_ == CpuCore::Recompiler) ? jit.run_step(cpu, bus) : cpu.step(); }

    void set_cpu_core(CpuCore core) { cpu_core_ = core; }
    CpuCore cpu_core() const { return cpu_core_; }

    // Wall-clock split of step_frame() between the CPU (interpreter/JIT,
    // incl. whatever MMIO/DMA its accesses trigger) and the per-scanline
    // AI/VI/RSP(+RDP) stepping. Off by default; headless --jit-stats.
    void set_profiling(bool on) { profiling_ = on; }
    double prof_cpu_seconds() const { return prof_cpu_seconds_; }
    double prof_other_seconds() const { return prof_other_seconds_; }

    // The displayed image. With an internal resolution above 1
    // (RDP::set_hires_scale) it is usually that many times the frame buffer
    // size (VideoFrame::scale), and with the GPU renderer it stays in video
    // memory (VideoFrame::gpu).
    void render_frame(VideoFrame& out);
    // The same as ARGB8888 in memory; `out_scale` receives the factor.
    void render_frame(std::vector<u32>& out_pixels, int& out_w, int& out_h, int* out_scale = nullptr);

    // Save states (savestate.hpp): the whole machine between two frames,
    // i.e. after step_frame() (and render_frame()). load_state() leaves the
    // machine as it was when the state doesn't fit - another game, another
    // state version, damaged - and says why in `error`.
    std::vector<u8> save_state();
    bool load_state(const std::vector<u8>& state, std::string& error);

    Controller& get_controller(int index = 0) {
        return controllers[index & 3];
    }

    Cartridge& get_cartridge() { return cart; }
    CPU& get_cpu() { return cpu; }
    Recompiler& get_jit() { return jit; }
    Bus& get_bus() { return bus; }
    VI& get_vi() { return vi; }
    AI& get_ai() { return ai; }
    RSP& get_rsp() { return rsp; }
    RDP& get_rdp() { return rdp; }

private:
    // Consumes up to `budget` cycles at once while the CPU sits in an idle
    // loop; returns the cycles consumed (0 = not idle). See emulator.cpp.
    u32 skip_idle_loop(u32 budget);
    template <class S> void serialize(S& s);

    Cartridge cart;
    PIF pif;
    Controller controllers[4];
    MI mi;
    VI vi;
    AI ai;
    PI pi;
    SI si;
    RSP rsp;
    RDP rdp;
    Bus bus;
    CPU cpu;
    Recompiler jit;
    CpuCore cpu_core_ = CpuCore::Recompiler;
    bool profiling_ = false;
    double prof_cpu_seconds_ = 0.0;
    double prof_other_seconds_ = 0.0;
};
