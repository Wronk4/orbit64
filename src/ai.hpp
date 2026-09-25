#pragma once

#include "common.hpp"
#include <vector>

class MI;
class AudioStream;

struct AudioBuffer {
    u32 dram_addr{0};
    u32 length{0};
    u32 remaining{0};
};

class AI {
public:
    AI();

    void reset();

    u32 read_reg(u32 addr) const;
    void write_reg(u32 addr, u32 val, MI& mi, const u8* rdram, size_t rdram_size);

    void step(u32 cycles, MI& mi, const u8* rdram, size_t rdram_size);

    // Where the samples go: every buffer the game hands to the DAC is pushed
    // to the sink at the native rate (AudioStream resamples it for the host
    // device). nullptr = discard (headless runs).
    void set_sink(AudioStream* sink) { sink_ = sink; }

    // Frontend status queries (read-only). 0 until the game programs AI_DACRATE.
    u32 get_native_sample_rate() const { return dacrate_reg ? native_sample_rate() : 0; }

    // Test tap (headless --wav): every buffer the game hands to the DAC is
    // appended here as interleaved s16 L,R at the native rate. nullptr = off.
    void set_capture(std::vector<s16>* out) { capture = out; }

    // Save states (savestate.hpp). The DMA side only: the host's queued
    // output is flushed on a load (EmuCore).
    template <class S> void serialize(S& s) {
        s(dram_addr_reg, control_reg, dacrate_reg, bitrate_reg, current_buffer, next_buffer, busy, full,
          cycle_byte_accum);
    }

private:
    u32 native_sample_rate() const;

    u32 dram_addr_reg{0};
    u32 control_reg{0};
    u32 dacrate_reg{0};
    u32 bitrate_reg{0};

    AudioBuffer current_buffer{};
    AudioBuffer next_buffer{};
    bool busy{false};
    bool full{false};

    double cycle_byte_accum{0.0};

    AudioStream* sink_{nullptr};
    std::vector<s16> frames_; // scratch for one buffer
    std::vector<s16>* capture{nullptr};
};
