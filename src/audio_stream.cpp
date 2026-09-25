#include "audio_stream.hpp"
#include <algorithm>
#include <cmath>

namespace {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kKaiserBeta = 8.0;       // ~80 dB stop band
    constexpr double kCutoff = 0.92;          // of the lower Nyquist frequency
    // Emulation speed trim: 1% per target's worth of queue error, at most
    // 3%. It changes how fast sound is made, not its pitch.
    constexpr double kPacingGain = 0.01;
    constexpr double kMaxPacing = 0.03;
    constexpr u32 kBaseMarginUs = 10000;
    constexpr u32 kMaxMarginUs = 40000;

    double bessel_i0(double x) {
        double sum = 1.0, term = 1.0;
        for (int k = 1; k < 40; ++k) {
            term *= (x / (2.0 * k)) * (x / (2.0 * k));
            sum += term;
            if (term < sum * 1e-12) break;
        }
        return sum;
    }
}

AudioStream::AudioStream()
    : buf_(kCapacity * 2), rates_(kCapacity), kernel_((kPhases + 1) * kTaps), window_((kPhases + 1) * kTaps) {
    // Tap k of phase p sits at x = k - (kTaps/2 - 1) - p/kPhases input
    // frames from the output position (see next_frame()).
    const double half = kTaps / 2.0;
    const double norm = bessel_i0(kKaiserBeta);
    for (int p = 0; p <= kPhases; ++p) {
        for (int k = 0; k < kTaps; ++k) {
            const double x = k - (half - 1.0) - static_cast<double>(p) / kPhases;
            const double u = std::clamp(x / half, -1.0, 1.0);
            window_[p * kTaps + k] = static_cast<float>(bessel_i0(kKaiserBeta * std::sqrt(1.0 - u * u)) / norm);
        }
    }
}

void AudioStream::configure(u32 output_rate, u32 device_frames) {
    if (output_rate) out_rate_.store(output_rate, std::memory_order_relaxed);
    if (device_frames) device_frames_.store(device_frames, std::memory_order_relaxed);
}

void AudioStream::flush() {
    flush_to_.store(head_.load(std::memory_order_acquire), std::memory_order_relaxed);
    flush_serial_.fetch_add(1, std::memory_order_release);
}

double AudioStream::queued_seconds() const {
    const u64 used = head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
    return static_cast<double>(used) / std::max<u32>(1, last_rate_.load(std::memory_order_relaxed));
}

double AudioStream::target_seconds() const {
    const double device = static_cast<double>(device_frames_.load(std::memory_order_relaxed)) /
                          std::max<u32>(1, out_rate_.load(std::memory_order_relaxed));
    const double chunk = static_cast<double>(chunk_frames_.load(std::memory_order_relaxed)) /
                         std::max<u32>(1, last_rate_.load(std::memory_order_relaxed));
    return device + chunk + margin_us_.load(std::memory_order_relaxed) * 1e-6;
}

AudioStream::Stats AudioStream::stats() const {
    Stats s;
    s.underruns = underruns_.load(std::memory_order_relaxed);
    s.dropped_frames = dropped_.load(std::memory_order_relaxed);
    s.queued_ms = queued_seconds() * 1000.0;
    s.latency_ms = s.queued_ms + 1000.0 * device_frames_.load(std::memory_order_relaxed) /
                                     std::max<u32>(1, out_rate_.load(std::memory_order_relaxed));
    s.target_ms = target_seconds() * 1000.0;
    return s;
}

// -----------------------------------------------------------------------------
// Producer
// -----------------------------------------------------------------------------

void AudioStream::push(const s16* frames, size_t count, u32 native_rate) {
    if (count == 0 || native_rate == 0) return;
    last_rate_.store(native_rate, std::memory_order_relaxed);
    // The target covers one push: remember the largest recent one (decaying
    // slowly, so a one-off burst is forgotten again).
    const u32 prev = chunk_frames_.load(std::memory_order_relaxed);
    chunk_frames_.store(std::max(static_cast<u32>(count), prev - prev / 64), std::memory_order_relaxed);

    const u64 head = head_.load(std::memory_order_relaxed);
    const u64 used = head - tail_.load(std::memory_order_acquire);
    // Beyond four targets' worth (fast-forward, paused output, no device)
    // the new input is dropped: latency stays bounded.
    const double cap_s = std::max(4.0 * target_seconds(), 0.25);
    const u64 cap = std::min<u64>(kCapacity - 1, static_cast<u64>(cap_s * native_rate));
    size_t n = count;
    if (used + n > cap) {
        const size_t room = used < cap ? static_cast<size_t>(cap - used) : 0;
        dropped_.fetch_add(n - room, std::memory_order_relaxed);
        n = room;
    }
    for (size_t i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>((head + i) & (kCapacity - 1));
        buf_[idx * 2] = frames[i * 2];
        buf_[idx * 2 + 1] = frames[i * 2 + 1];
        rates_[idx] = native_rate;
    }
    head_.store(head + n, std::memory_order_release);
}

double AudioStream::pacing_adjust() {
    const double q = queued_seconds();
    fill_avg_ = fill_avg_ < 0.0 ? q : fill_avg_ + (q - fill_avg_) * 0.05;
    const double t = target_seconds();
    return std::clamp(kPacingGain * (fill_avg_ - t) / t, -kMaxPacing, kMaxPacing);
}

// -----------------------------------------------------------------------------
// Consumer
// -----------------------------------------------------------------------------

void AudioStream::rebuild_kernel(u32 in_rate, u32 out_rate) {
    kernel_in_ = in_rate;
    kernel_out_ = out_rate;
    step_ = static_cast<double>(in_rate) / out_rate;
    // Low-pass at the lower of the two Nyquist frequencies, in cycles per
    // input frame.
    const double fc = 0.5 * kCutoff * std::min(1.0, static_cast<double>(out_rate) / in_rate);
    const double half = kTaps / 2.0;
    for (int p = 0; p <= kPhases; ++p) {
        float* row = &kernel_[p * kTaps];
        double sum = 0.0;
        for (int k = 0; k < kTaps; ++k) {
            const double x = k - (half - 1.0) - static_cast<double>(p) / kPhases;
            const double a = 2.0 * kPi * fc * x;
            const double sinc = std::fabs(a) < 1e-9 ? 1.0 : std::sin(a) / a;
            const double v = sinc * window_[p * kTaps + k];
            row[k] = static_cast<float>(v);
            sum += v;
        }
        // Unity gain at DC in every phase.
        for (int k = 0; k < kTaps; ++k) row[k] = static_cast<float>(row[k] / sum);
    }
}

bool AudioStream::next_frame(float& l, float& r) {
    while (frac_ >= 1.0) {
        if (read_ == head_snap_) {
            head_snap_ = head_.load(std::memory_order_acquire);
            if (read_ == head_snap_) return false;
        }
        const size_t idx = static_cast<size_t>(read_ & (kCapacity - 1));
        const u32 rate = rates_[idx];
        const u32 orate = out_rate_.load(std::memory_order_relaxed);
        if (rate != kernel_in_ || orate != kernel_out_) rebuild_kernel(rate, orate);
        win_pos_ = (win_pos_ + 1) % kTaps;
        win_l_[win_pos_] = win_l_[win_pos_ + kTaps] = buf_[idx * 2] * (1.0f / 32768.0f);
        win_r_[win_pos_] = win_r_[win_pos_ + kTaps] = buf_[idx * 2 + 1] * (1.0f / 32768.0f);
        ++read_;
        frac_ -= 1.0;
    }
    // The last kTaps input frames, oldest first; the output lies frac_ past
    // the middle of them.
    const double ph = frac_ * kPhases;
    const int p = static_cast<int>(ph);
    const float t = static_cast<float>(ph - p);
    const float* k0 = &kernel_[p * kTaps];
    const float* k1 = k0 + kTaps;
    const float* wl = &win_l_[win_pos_ + 1];
    const float* wr = &win_r_[win_pos_ + 1];
    float al = 0.0f, ar = 0.0f;
    for (int k = 0; k < kTaps; ++k) {
        const float c = k0[k] + t * (k1[k] - k0[k]);
        al += c * wl[k];
        ar += c * wr[k];
    }
    l = al;
    r = ar;
    frac_ += step_;
    return true;
}

void AudioStream::pull(float* out, size_t frames) {
    const u32 serial = flush_serial_.load(std::memory_order_acquire);
    if (serial != seen_flush_) {
        seen_flush_ = serial;
        read_ = std::max(read_, flush_to_.load(std::memory_order_relaxed));
        std::fill(std::begin(win_l_), std::end(win_l_), 0.0f);
        std::fill(std::begin(win_r_), std::end(win_r_), 0.0f);
        frac_ = 1.0;
        starving_ = true;
        gain_ = 0.0f;
        hold_l_ = hold_r_ = 0.0f;
    }
    head_snap_ = head_.load(std::memory_order_acquire);
    const bool playing = playing_.load(std::memory_order_relaxed);

    // After an underrun (and at the start) wait for a full cushion.
    if (starving_ && playing) {
        const double queued = static_cast<double>(head_snap_ - read_) /
                              std::max<u32>(1, last_rate_.load(std::memory_order_relaxed));
        if (queued >= target_seconds()) starving_ = false;
    }

    constexpr float fade = 1.0f / kFadeFrames;
    for (size_t i = 0; i < frames; ++i) {
        float l = hold_l_, r = hold_r_;
        if (!starving_ && (playing || gain_ > 0.0f)) {
            if (next_frame(l, r)) {
                hold_l_ = l;
                hold_r_ = r;
            } else {
                // Ran dry: hold the last value while fading out (no click).
                starving_ = true;
                if (playing) {
                    clean_frames_ = 0;
                    underruns_.fetch_add(1, std::memory_order_relaxed);
                    const u32 m = margin_us_.load(std::memory_order_relaxed);
                    margin_us_.store(std::min(kMaxMarginUs, m + 4000), std::memory_order_relaxed);
                }
            }
        }
        const float target = (playing && !starving_) ? 1.0f : 0.0f;
        gain_ = gain_ < target ? std::min(target, gain_ + fade) : std::max(target, gain_ - fade);
        out[i * 2] = l * gain_;
        out[i * 2 + 1] = r * gain_;
    }
    if (gain_ == 0.0f) hold_l_ = hold_r_ = 0.0f;
    // A raised target comes back down after 30 s without an underrun.
    if (playing && !starving_ && (clean_frames_ += frames) > 30ull * kernel_out_) {
        clean_frames_ = 0;
        const u32 m = margin_us_.load(std::memory_order_relaxed);
        if (m > kBaseMarginUs) margin_us_.store(std::max(kBaseMarginUs, m - 2000), std::memory_order_relaxed);
    }
    tail_.store(read_, std::memory_order_release);
}
