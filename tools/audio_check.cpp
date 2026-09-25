// Headless checks of the host audio path (src/audio_stream.cpp):
//
//   make audio_check  -> bin/audio_check
//   bin/audio_check                      resampler quality + pacing simulation
//   bin/audio_check in.wav out.wav [rate] resample a --wav capture to the device rate
//
// Resampler: pure tones at a native rate go through push()/pull() in device
// sized chunks; the output must hold the tone's level, and everything else
// (images, aliasing, noise) must stay far below it.
//
// Pacing: an emulated 60 Hz game produces a frame of sound per frame, paced
// by pacing_adjust() like EmuCore does, while a simulated device whose clock
// is off by a few hundred ppm drains it in callbacks. After the start the
// queue must never run dry and has to settle at its target.
#include "audio_stream.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
constexpr double kPi = 3.14159265358979323846;

// Goertzel-free tone analysis: least-squares fit of a sine at `freq`, the
// residual is "everything else".
void tone_error(const std::vector<float>& x, double rate, double freq, double& level_db, double& rest_db) {
    double ss = 0, cc = 0, sc = 0, xs = 0, xc = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double s = std::sin(2 * kPi * freq * i / rate), c = std::cos(2 * kPi * freq * i / rate);
        ss += s * s; cc += c * c; sc += s * c; xs += x[i] * s; xc += x[i] * c;
    }
    const double det = ss * cc - sc * sc;
    const double a = (xs * cc - xc * sc) / det, b = (xc * ss - xs * sc) / det;
    double err = 0, sig = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double fit = a * std::sin(2 * kPi * freq * i / rate) + b * std::cos(2 * kPi * freq * i / rate);
        err += (x[i] - fit) * (x[i] - fit);
        sig += fit * fit;
    }
    level_db = 10 * std::log10(sig / x.size() + 1e-30);
    rest_db = 10 * std::log10(err / x.size() + 1e-30);
}

bool check_tone(u32 in_rate, u32 out_rate, double freq, double min_snr_db, double tol_db = 0.1) {
    AudioStream st;
    st.configure(out_rate, 1024);
    st.set_playing(true);
    const size_t n_in = in_rate * 2;
    std::vector<s16> in(n_in * 2);
    const double amp = 0.5;
    for (size_t i = 0; i < n_in; ++i)
        in[i * 2] = in[i * 2 + 1] = static_cast<s16>(std::lround(amp * 32767 * std::sin(2 * kPi * freq * i / in_rate)));
    std::vector<float> outl;
    std::vector<float> chunk(1024 * 2);
    size_t pushed = 0;
    while (pushed < n_in) {
        const size_t n = std::min<size_t>(in_rate / 60, n_in - pushed);
        st.push(in.data() + pushed * 2, n, in_rate);
        pushed += n;
        // Drain whenever the queue holds more than its target.
        while (st.queued_seconds() > st.target_seconds()) {
            st.pull(chunk.data(), 1024);
            for (size_t i = 0; i < 1024; ++i) outl.push_back(chunk[i * 2]);
        }
    }
    // Skip the fade-in, analyse one second.
    if (outl.size() < out_rate + out_rate / 4) return false;
    std::vector<float> seg(outl.begin() + out_rate / 4, outl.begin() + out_rate / 4 + out_rate);
    double level, rest;
    tone_error(seg, out_rate, freq, level, rest);
    const double want = 20 * std::log10(amp * 32767 / 32768.0) - 3.0103; // sine power
    const bool ok = std::fabs(level - want) < tol_db && level - rest >= min_snr_db;
    std::printf("  %5u -> %5u Hz, %7.1f Hz tone: level %+.2f dB (want %+.2f), rest %.1f dB below  %s\n", in_rate, out_rate,
                freq, level, want, level - rest, ok ? "ok" : "FAIL");
    return ok;
}

// A device whose clock runs `ppm` fast drains a 60 Hz producer paced by
// pacing_adjust(); returns false on an underrun after the start.
bool check_pacing(u32 in_rate, u32 out_rate, u32 device_frames, double ppm, double seconds) {
    AudioStream st;
    st.configure(out_rate, device_frames);
    st.set_playing(true);
    const double dev_period = device_frames / (out_rate * (1.0 + ppm * 1e-6));
    double t_emu = 0.0, t_dev = 0.0, frac = 0.0;
    std::vector<s16> frame;
    std::vector<float> out(device_frames * 2);
    double min_q = 1e9, max_q = 0.0, sum_q = 0.0;
    int n_q = 0;
    double adjust = 0.0;
    while (t_dev < seconds) {
        if (t_emu <= t_dev) {
            // One emulated frame: 1/60 s worth of native samples.
            frac += in_rate / 60.0;
            const size_t n = static_cast<size_t>(frac);
            frac -= n;
            frame.assign(n * 2, 1000);
            st.push(frame.data(), n, in_rate);
            adjust = st.pacing_adjust();
            t_emu += (1.0 / 60.0) * (1.0 + adjust);
        } else {
            st.pull(out.data(), device_frames);
            t_dev += dev_period;
            if (t_dev > 20.0) {
                const double q = st.queued_seconds();
                min_q = std::min(min_q, q);
                max_q = std::max(max_q, q);
                sum_q += q;
                ++n_q;
            }
        }
    }
    const auto s = st.stats();
    const bool ok = s.underruns == 0 && s.dropped_frames == 0;
    std::printf("  %5u -> %5u Hz, buffer %4u, clock %+5.0f ppm: queue %.1f..%.1f ms (avg %.1f, target %.1f), "
                "trim %+.3f%%, underruns %llu  %s\n",
                in_rate, out_rate, device_frames, ppm, min_q * 1e3, max_q * 1e3, sum_q / n_q * 1e3, s.target_ms,
                -adjust * 100, static_cast<unsigned long long>(s.underruns), ok ? "ok" : "FAIL");
    return ok;
}

bool read_wav(const char* path, std::vector<s16>& data, u32& rate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char hdr[44];
    f.read(hdr, 44);
    std::memcpy(&rate, hdr + 24, 4);
    u32 bytes;
    std::memcpy(&bytes, hdr + 40, 4);
    data.resize(bytes / 2);
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return true;
}

void write_wav(const char* path, const std::vector<float>& data, u32 rate) {
    std::ofstream f(path, std::ios::binary);
    auto u32le = [&](u32 v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16le = [&](u16 v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    const u32 bytes = static_cast<u32>(data.size() * 2);
    f.write("RIFF", 4); u32le(36 + bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); u32le(16); u16le(1); u16le(2); u32le(rate); u32le(rate * 4); u16le(4); u16le(16);
    f.write("data", 4); u32le(bytes);
    for (float v : data) {
        const s16 s = static_cast<s16>(std::lround(std::fmax(-1.0f, std::fmin(1.0f, v)) * 32767.0f));
        f.write(reinterpret_cast<const char*>(&s), 2);
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc >= 3) {
        std::vector<s16> in;
        u32 in_rate = 0;
        if (!read_wav(argv[1], in, in_rate)) {
            std::fprintf(stderr, "cannot read %s\n", argv[1]);
            return 1;
        }
        const u32 out_rate = argc >= 4 ? static_cast<u32>(std::stoul(argv[3])) : 48000;
        AudioStream st;
        st.configure(out_rate, 1024);
        st.set_playing(true);
        std::vector<float> out, chunk(1024 * 2);
        const size_t frames = in.size() / 2, per = in_rate / 60;
        for (size_t pos = 0; pos < frames; pos += per) {
            st.push(in.data() + pos * 2, std::min(per, frames - pos), in_rate);
            while (st.queued_seconds() > st.target_seconds()) {
                st.pull(chunk.data(), 1024);
                out.insert(out.end(), chunk.begin(), chunk.end());
            }
        }
        write_wav(argv[2], out, out_rate);
        std::printf("%s: %u Hz -> %s: %u Hz, %zu frames\n", argv[1], in_rate, argv[2], out_rate, out.size() / 2);
        return 0;
    }

    bool ok = true;
    std::printf("Resampler (pure tones, 0.5 FS):\n");
    for (u32 in_rate : {22047u, 26807u, 32006u, 44100u, 48000u}) {
        for (u32 out_rate : {44100u, 48000u}) {
            const double nyq = std::min(in_rate, out_rate) / 2.0;
            ok &= check_tone(in_rate, out_rate, 441.0, 70.0);
            ok &= check_tone(in_rate, out_rate, 0.25 * nyq, 70.0);
            ok &= check_tone(in_rate, out_rate, 0.8 * nyq, 60.0, 0.3); // near the cutoff: small droop
        }
    }
    std::printf("Pacing (60 s, device clock off by +-ppm):\n");
    for (u32 buf : {256u, 1024u, 2048u}) {
        for (double ppm : {-500.0, 0.0, 500.0}) ok &= check_pacing(32006, 48000, buf, ppm, 60.0);
    }
    ok &= check_pacing(22047, 44100, 512, 300.0, 60.0);
    std::printf(ok ? "ALL OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}
