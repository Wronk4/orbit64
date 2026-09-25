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

    // The displayed image, ARGB8888. With an internal resolution above 1
    // (RDP::set_hires_scale) it is that many times the frame buffer size;
    // `out_scale` receives the factor.
    void render_frame(std::vector<u32>& out_pixels, int& out_w, int& out_h, int* out_scale = nullptr);
    void get_audio_samples(float* out_stream, size_t count);
    void set_audio_output_rate(u32 rate) { ai.set_output_sample_rate(rate); }

    Controller& get_controller(int index = 0) {
        return controllers[index & 3];
    }

    Cartridge& get_cartridge() { return cart; }
    CPU& get_cpu() { return cpu; }
    Bus& get_bus() { return bus; }
    VI& get_vi() { return vi; }
    AI& get_ai() { return ai; }
    RSP& get_rsp() { return rsp; }
    RDP& get_rdp() { return rdp; }

private:
    // Consumes up to `budget` cycles at once while the CPU sits in an idle
    // loop; returns the cycles consumed (0 = not idle). See emulator.cpp.
    u32 skip_idle_loop(u32 budget);

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
    std::vector<u32> native_frame_; // VI output before upscaling (internal resolution > 1)
};
