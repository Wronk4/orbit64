#pragma once

#include "common.hpp"
#include <array>
#include <mutex>

class MI;

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

    void get_samples(float* out_stream, size_t count);

    // Sample rate the host audio device is opened at; decoded samples are
    // resampled from the N64's native AI_DACRATE-derived rate to this rate
    // before being queued, since AI_DACRATE varies per game.
    void set_output_sample_rate(u32 rate) { output_sample_rate = rate; }

    // Frontend status queries (read-only). 0 until the game programs AI_DACRATE.
    u32 get_native_sample_rate() const { return dacrate_reg ? native_sample_rate() : 0; }
    u32 get_output_sample_rate() const { return output_sample_rate; }

    // Save states (savestate.hpp). The DMA side only: the host's resampler
    // position and the queued output samples carry on across a load.
    template <class S> void serialize(S& s) {
        s(dram_addr_reg, control_reg, dacrate_reg, bitrate_reg, current_buffer, next_buffer, busy, full,
          cycle_byte_accum);
    }

private:
    u32 native_sample_rate() const;

    // Pushes one interleaved stereo frame into the ring buffer, overwriting
    // the oldest frame if the buffer is full.
    void push_frame(float l, float r);

    u32 dram_addr_reg{0};
    u32 control_reg{0};
    u32 dacrate_reg{0};
    u32 bitrate_reg{0};

    AudioBuffer current_buffer{};
    AudioBuffer next_buffer{};
    bool busy{false};
    bool full{false};

    u32 output_sample_rate{44100};
    double resample_pos{0.0};
    float prev_left{0.0f};
    float prev_right{0.0f};
    double cycle_byte_accum{0.0};

    // Ring buffer of interleaved (L,R) stereo frames. get_samples() runs on
    // the real-time SDL audio callback thread, so it must never do anything
    // proportional to the whole buffer's size (a vector::erase from the
    // front, as this used to do, is exactly that and can itself cause
    // audible glitches independent of emulation speed).
    static constexpr size_t FIFO_CAPACITY_FRAMES = 48000; // ~1s at 48kHz
    std::array<float, FIFO_CAPACITY_FRAMES * 2> ring{};
    size_t ring_read{0};   // frame index
    size_t ring_write{0};  // frame index
    size_t ring_count{0};  // frames currently buffered

    // Last frame actually output, used to fade smoothly across underrun
    // boundaries instead of jumping to/from hard silence (which sounds like
    // clicks/pops on top of the underrun itself).
    float carry_l{0.0f};
    float carry_r{0.0f};

    std::mutex audio_mutex;
};
