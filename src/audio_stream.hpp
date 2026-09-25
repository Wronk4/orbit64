#pragma once

#include "common.hpp"
#include <atomic>
#include <vector>

// Host side of the emulated sound. The AI hands every buffer the game plays
// to push() (emulation thread) at the DAC's native rate, and the audio
// device callback pulls it resampled to the device rate with pull(). The two
// threads share a lock-free single-producer/single-consumer queue, so the
// callback never waits on emulation.
//
// Resampling is a windowed-sinc polyphase filter (32 taps, Kaiser window,
// interpolated between 256 phases): flat up to close to the native Nyquist
// frequency, without the images and dull top end a linear interpolator
// leaves.
//
// Clock drift between the emulated DAC and the host device is not bent into
// the pitch. Instead the emulator paces itself with pacing_adjust(), running
// up to 3% faster or slower to keep the queue at target_seconds(). When the
// queue does run dry (emulation slower than real time) the output fades out
// and resumes, faded in, once the queue is back at its target, rather than
// crackling on every callback; each such underrun also raises the target a
// little.
class AudioStream {
public:
    AudioStream();

    // Producer (emulation thread): `count` interleaved s16 L,R frames.
    void push(const s16* frames, size_t count, u32 native_rate);

    // Consumer (audio device callback): always fills `frames` stereo frames.
    void pull(float* out, size_t frames);

    // Setup and control (any thread). `device_frames` is the device's
    // callback size. Not playing: fade out and hold what is queued.
    void configure(u32 output_rate, u32 device_frames);
    void set_playing(bool on) { playing_.store(on, std::memory_order_relaxed); }
    // Forgets everything queued so far (reset, state load, new game).
    void flush();

    // Emulation thread, once per emulated frame: relative change of the
    // frame period that steers the queue towards its target (>0 = slower).
    double pacing_adjust();

    double queued_seconds() const;
    double target_seconds() const;

    struct Stats {
        u64 underruns = 0;       // times the queue ran dry while playing
        u64 dropped_frames = 0;  // input discarded because the queue was full
        double queued_ms = 0.0;  // input waiting to be played
        double latency_ms = 0.0; // queued input + the device buffer
        double target_ms = 0.0;
    };
    Stats stats() const;

private:
    static constexpr size_t kCapacity = 1 << 15; // frames (~0.7-1.5 s)
    static constexpr int kTaps = 32;
    static constexpr int kPhases = 256;
    static constexpr int kFadeFrames = 96;

    void rebuild_kernel(u32 in_rate, u32 out_rate);
    bool next_frame(float& l, float& r);

    // Queue: frames, and the native rate each one was played at.
    std::vector<s16> buf_;
    std::vector<u32> rates_;
    std::atomic<u64> head_{0}; // written by the producer
    std::atomic<u64> tail_{0}; // written by the consumer
    std::atomic<u64> flush_to_{0};
    std::atomic<u32> flush_serial_{0};
    std::atomic<u32> last_rate_{32000};
    std::atomic<u32> chunk_frames_{0}; // largest recent push, native frames

    std::atomic<u32> out_rate_{48000};
    std::atomic<u32> device_frames_{1024};
    std::atomic<bool> playing_{false};
    std::atomic<u32> margin_us_{10000};
    std::atomic<u64> underruns_{0};
    std::atomic<u64> dropped_{0};

    // Consumer-only state.
    u32 seen_flush_ = 0;
    u64 read_ = 0;       // next frame to take from the queue
    u64 head_snap_ = 0;
    std::vector<float> kernel_; // (kPhases + 1) rows of kTaps
    std::vector<float> window_; // Kaiser window on the same grid
    u32 kernel_in_ = 0, kernel_out_ = 0;
    double step_ = 1.0;
    double frac_ = 1.0;
    float win_l_[kTaps * 2] = {};
    float win_r_[kTaps * 2] = {};
    int win_pos_ = 0;
    bool starving_ = true;
    float gain_ = 0.0f;
    float hold_l_ = 0.0f, hold_r_ = 0.0f;
    u64 clean_frames_ = 0; // output since the last underrun

    // Pacing (emulation thread).
    double fill_avg_ = -1.0;
};
