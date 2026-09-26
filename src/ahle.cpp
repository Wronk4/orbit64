#include "ahle.hpp"
#include <algorithm>
#include <cstring>

// Command semantics follow the RSP audio microcodes as documented by the
// long-standing reference HLE implementations (and the SDK's own command
// macros in abi.h / n_abi.h): the same DMEM layouts, rounding, saturation
// and state formats, so the output matches what the RSP would produce.

namespace {
    constexpr u8 A_INIT = 0x01;
    constexpr u8 A_LEFT = 0x02;
    constexpr u8 A_VOL  = 0x04;
    constexpr u8 A_AUX  = 0x08;

    // ABI 1 command buffers are DMEM offsets from here.
    constexpr u16 AUDIO_DMEM_BASE = 0x5C0;

    // n_audio works on fixed buffers of NAUDIO_COUNT bytes (184 samples).
    constexpr u16 NAUDIO_COUNT     = 0x170;
    constexpr u16 NAUDIO_MAIN      = 0x4F0;
    constexpr u16 NAUDIO_MAIN2     = 0x660;
    constexpr u16 NAUDIO_DRY_LEFT  = 0x9D0;
    constexpr u16 NAUDIO_DRY_RIGHT = 0xB40;
    constexpr u16 NAUDIO_WET_LEFT  = 0xCB0;
    constexpr u16 NAUDIO_WET_RIGHT = 0xE20;

    inline s16 clamp_s16(s32 v) {
        return static_cast<s16>(std::clamp<s32>(v, -32768, 32767));
    }
    inline u32 align_up(u32 v, u32 a) { return (v + a - 1) & ~(a - 1); }

    // sum_{k=0}^{n-1} x[k] * y[n-1-k]
    inline s32 rdot(size_t n, const s16* x, const s16* y) {
        s32 acc = 0;
        for (size_t k = 0; k < n; ++k) acc += x[k] * y[n - 1 - k];
        return acc;
    }

    // RSP VMULF of one lane: Q15 multiply with rounding.
    inline s32 vmulf(s16 a, s16 b) { return (static_cast<s32>(a) * b + 0x4000) >> 15; }

    // Volume ramp of the ABI 1 / n_audio envelope mixers (Q16.16).
    struct Ramp {
        s64 value{0}, step{0}, target{0};
        s16 tick() {
            value += step;
            const bool reached = (step <= 0) ? (value <= target) : (value >= target);
            if (reached) {
                value = target;
                step = 0;
            }
            return static_cast<s16>(value >> 16);
        }
    };

    // 4-tap resampling filter coefficients (64 phases x 4 taps), transcribed
    // from the RSP audio microcode's own data segment (audio.s, DMEM offset
    // 0xC0). RESAMPLE indexes this with the top 6 bits of the fractional
    // pitch accumulator to pick a phase.
    constexpr s16 RESAMPLE_LUT[64 * 4] = {
        s16(0x0c39), s16(0x66ad), s16(0x0d46), s16(0xffdf),
        s16(0x0b39), s16(0x6696), s16(0x0e5f), s16(0xffd8),
        s16(0x0a44), s16(0x6669), s16(0x0f83), s16(0xffd0),
        s16(0x095a), s16(0x6626), s16(0x10b4), s16(0xffc8),
        s16(0x087d), s16(0x65cd), s16(0x11f0), s16(0xffbf),
        s16(0x07ab), s16(0x655e), s16(0x1338), s16(0xffb6),
        s16(0x06e4), s16(0x64d9), s16(0x148c), s16(0xffac),
        s16(0x0628), s16(0x643f), s16(0x15eb), s16(0xffa1),
        s16(0x0577), s16(0x638f), s16(0x1756), s16(0xff96),
        s16(0x04d1), s16(0x62cb), s16(0x18cb), s16(0xff8a),
        s16(0x0435), s16(0x61f3), s16(0x1a4c), s16(0xff7e),
        s16(0x03a4), s16(0x6106), s16(0x1bd7), s16(0xff71),
        s16(0x031c), s16(0x6007), s16(0x1d6c), s16(0xff64),
        s16(0x029f), s16(0x5ef5), s16(0x1f0b), s16(0xff56),
        s16(0x022a), s16(0x5dd0), s16(0x20b3), s16(0xff48),
        s16(0x01be), s16(0x5c9a), s16(0x2264), s16(0xff3a),
        s16(0x015b), s16(0x5b53), s16(0x241e), s16(0xff2c),
        s16(0x0101), s16(0x59fc), s16(0x25e0), s16(0xff1e),
        s16(0x00ae), s16(0x5896), s16(0x27a9), s16(0xff10),
        s16(0x0063), s16(0x5720), s16(0x297a), s16(0xff02),
        s16(0x001f), s16(0x559d), s16(0x2b50), s16(0xfef4),
        s16(0xffe2), s16(0x540d), s16(0x2d2c), s16(0xfee8),
        s16(0xffac), s16(0x5270), s16(0x2f0d), s16(0xfedb),
        s16(0xff7c), s16(0x50c7), s16(0x30f3), s16(0xfed0),
        s16(0xff53), s16(0x4f14), s16(0x32dc), s16(0xfec6),
        s16(0xff2e), s16(0x4d57), s16(0x34c8), s16(0xfebd),
        s16(0xff0f), s16(0x4b91), s16(0x36b6), s16(0xfeb6),
        s16(0xfef5), s16(0x49c2), s16(0x38a5), s16(0xfeb0),
        s16(0xfedf), s16(0x47ed), s16(0x3a95), s16(0xfeac),
        s16(0xfece), s16(0x4611), s16(0x3c85), s16(0xfeab),
        s16(0xfec0), s16(0x4430), s16(0x3e74), s16(0xfeac),
        s16(0xfeb6), s16(0x424a), s16(0x4060), s16(0xfeaf),
        s16(0xfeaf), s16(0x4060), s16(0x424a), s16(0xfeb6),
        s16(0xfeac), s16(0x3e74), s16(0x4430), s16(0xfec0),
        s16(0xfeab), s16(0x3c85), s16(0x4611), s16(0xfece),
        s16(0xfeac), s16(0x3a95), s16(0x47ed), s16(0xfedf),
        s16(0xfeb0), s16(0x38a5), s16(0x49c2), s16(0xfef5),
        s16(0xfeb6), s16(0x36b6), s16(0x4b91), s16(0xff0f),
        s16(0xfebd), s16(0x34c8), s16(0x4d57), s16(0xff2e),
        s16(0xfec6), s16(0x32dc), s16(0x4f14), s16(0xff53),
        s16(0xfed0), s16(0x30f3), s16(0x50c7), s16(0xff7c),
        s16(0xfedb), s16(0x2f0d), s16(0x5270), s16(0xffac),
        s16(0xfee8), s16(0x2d2c), s16(0x540d), s16(0xffe2),
        s16(0xfef4), s16(0x2b50), s16(0x559d), s16(0x001f),
        s16(0xff02), s16(0x297a), s16(0x5720), s16(0x0063),
        s16(0xff10), s16(0x27a9), s16(0x5896), s16(0x00ae),
        s16(0xff1e), s16(0x25e0), s16(0x59fc), s16(0x0101),
        s16(0xff2c), s16(0x241e), s16(0x5b53), s16(0x015b),
        s16(0xff3a), s16(0x2264), s16(0x5c9a), s16(0x01be),
        s16(0xff48), s16(0x20b3), s16(0x5dd0), s16(0x022a),
        s16(0xff56), s16(0x1f0b), s16(0x5ef5), s16(0x029f),
        s16(0xff64), s16(0x1d6c), s16(0x6007), s16(0x031c),
        s16(0xff71), s16(0x1bd7), s16(0x6106), s16(0x03a4),
        s16(0xff7e), s16(0x1a4c), s16(0x61f3), s16(0x0435),
        s16(0xff8a), s16(0x18cb), s16(0x62cb), s16(0x04d1),
        s16(0xff96), s16(0x1756), s16(0x638f), s16(0x0577),
        s16(0xffa1), s16(0x15eb), s16(0x643f), s16(0x0628),
        s16(0xffac), s16(0x148c), s16(0x64d9), s16(0x06e4),
        s16(0xffb6), s16(0x1338), s16(0x655e), s16(0x07ab),
        s16(0xffbf), s16(0x11f0), s16(0x65cd), s16(0x087d),
        s16(0xffc8), s16(0x10b4), s16(0x6626), s16(0x095a),
        s16(0xffd0), s16(0x0f83), s16(0x6669), s16(0x0a44),
        s16(0xffd8), s16(0x0e5f), s16(0x6696), s16(0x0b39),
        s16(0xffdf), s16(0x0d46), s16(0x66ad), s16(0x0c39),
    };
}

void AudioHLE::reset() {
    scratch.fill(0);
    segments.fill(0);
    table.fill(0);
    in = out = count = 0;
    dry_right = wet_left = wet_right = 0;
    vol[0] = vol[1] = target[0] = target[1] = 0;
    rate[0] = rate[1] = 0;
    dry = wet = 0;
    loop = 0;
    env_values[0] = env_values[1] = env_values[2] = 0;
    env_steps[0] = env_steps[1] = env_steps[2] = 0;
    filter_count = 0;
    filter_lut[0] = filter_lut[1] = 0;
    abi = Abi::Audio;
}

int AudioHLE::get_abi_index() const {
    switch (abi) {
        case Abi::Audio:
        case Abi::AudioGE: return 0;
        case Abi::NAudio:
        case Abi::NAudioDK:
        case Abi::NAudioMP3: return 1;
        case Abi::NeadMK: return 2;
        case Abi::NeadSF:
        case Abi::NeadFZ: return 3;
        case Abi::NeadZelda: return 4;
        case Abi::MusyX:
        case Abi::MusyX2: return 5;
    }
    return 0;
}

AudioHLE::Abi AudioHLE::detect(const u8* rdram, size_t rdram_size, u32 p) {
    if (p == 0 || p + 0x40 > rdram_size) return Abi::Audio;
    auto word = [&](u32 off) {
        const u8* b = rdram + p + off;
        return (static_cast<u32>(b[0]) << 24) | (static_cast<u32>(b[1]) << 16) | (static_cast<u32>(b[2]) << 8) | b[3];
    };
    if (word(0) == 0x00000001) {
        if (word(0x30) == 0xF0000F00) {
            switch (word(0x28)) {
                case 0x1DC8138C: // GoldenEye 007
                case 0x1E3C1390: // Blast Corps, Diddy Kong Racing
                    return Abi::AudioGE;
                default: return Abi::Audio; // 0x1E24138C and the rest of ABI 1
            }
        }
        switch (word(0x10)) {
            case 0x11181350: return Abi::NeadMK; // Mario Kart 64, Wave Race 64 (E)
            case 0x111812E0: // Star Fox 64 (J)
            case 0x110412AC: // Wave Race 64 (J rev B)
            case 0x110412CC: // Star Fox 64 / Lylat Wars
                return Abi::NeadSF;
            case 0x1CD01250: // F-Zero X
            case 0x1F4C1230: // F-Zero X Expansion Kit
                return Abi::NeadFZ;
            case 0x00010010: return Abi::MusyX2; // MusyX v2
            default: return Abi::NeadZelda; // OoT/MM, Yoshi's Story, 1080, Animal Crossing, ...
        }
    }
    switch (word(0x10)) {
        case 0x00000001: return Abi::MusyX; // MusyX v1
        case 0x1C58126C: return Abi::NAudioDK; // Donkey Kong 64
        case 0x1AE8143C: // Banjo-Tooie, Jet Force Gemini, Perfect Dark, Mickey's Speedway USA
        case 0x1AB0140C: // Conker's Bad Fur Day
            return Abi::NAudioMP3;
        default: return Abi::NAudio; // 0x127C, Banjo-Kazooie's 0x1280, ...
    }
}

void AudioHLE::process(u8* rdram, size_t rdram_size, u32 addr, u32 size, u32 ucode_data_ptr) {
    if (!rdram || rdram_size == 0 || size == 0) return;
    task_count++;
    rdram_ptr = rdram;
    rdram_sz = rdram_size;
    abi = detect(rdram, rdram_size, ucode_data_ptr);
    if (abi == Abi::MusyX) {
        run_musyx_v1(addr, size);
        return;
    }
    if (abi == Abi::MusyX2) {
        run_musyx_v2(addr, size);
        return;
    }
    segments.fill(0);

    const u32 n_commands = size >> 3; // each command is 8 bytes: w1, w2
    for (u32 i = 0; i < n_commands; ++i) {
        const u32 w1 = dram_u32(addr + i * 8);
        const u32 w2 = dram_u32(addr + i * 8 + 4);
        dispatch((w1 >> 24) & 0x7F, w1, w2);
    }
}

void AudioHLE::dispatch(u32 acmd, u32 w1, u32 w2) {
    switch (abi) {
        case Abi::Audio:
        case Abi::AudioGE: run_audio(acmd, w1, w2); break;
        case Abi::NAudio:
        case Abi::NAudioDK:
        case Abi::NAudioMP3: run_naudio(acmd, w1, w2); break;
        case Abi::NeadMK:
        case Abi::NeadSF:
        case Abi::NeadFZ:
        case Abi::NeadZelda: run_nead(acmd, w1, w2); break;
        case Abi::MusyX:
        case Abi::MusyX2: break;
    }
}

// -----------------------------------------------------------------------------
// Shared building blocks
// -----------------------------------------------------------------------------

u32 AudioHLE::segment_address(u32 so) const {
    return segments[(so >> 24) & 0xFF] + (so & 0x00FFFFFF);
}

void AudioHLE::load_table(u32 address, u32 entries) {
    entries = std::min<u32>(entries, static_cast<u32>(table.size()));
    for (u32 i = 0; i < entries; ++i) table[i] = dram_s16(address + i * 2);
}

void AudioHLE::clear(u16 dmem, u16 n) {
    for (u32 i = 0; i < n; ++i) set_dmem_u8(dmem + i, 0);
}

void AudioHLE::load(u16 dmem, u32 address, u16 n) {
    // DMA alignment: DMEM on 4 bytes, RDRAM on 8, whole 8-byte transfers.
    dmem &= ~3u;
    address &= ~7u;
    const u32 bytes = align_up(n, 8);
    for (u32 i = 0; i < bytes; ++i) set_dmem_u8(dmem + i, dram_u8(address + i));
}

void AudioHLE::save(u16 dmem, u32 address, u16 n) {
    dmem &= ~3u;
    address &= ~7u;
    const u32 bytes = align_up(n, 8);
    for (u32 i = 0; i < bytes; ++i) set_dram_u8(address + i, dmem_u8(dmem + i));
}

void AudioHLE::move(u16 dmemo, u16 dmemi, u16 n) {
    for (u32 i = 0; i < n; ++i) set_dmem_u8(dmemo + i, dmem_u8(dmemi + i));
}

void AudioHLE::mix(u16 dmemo, u16 dmemi, u16 n, s16 gain) {
    for (u32 k = 0; k < n / 2u; ++k) {
        const s32 v = (static_cast<s32>(dmem_s16(dmemi + k * 2)) * gain) >> 15;
        set_dmem_s16(dmemo + k * 2, clamp_s16(dmem_s16(dmemo + k * 2) + v));
    }
}

void AudioHLE::add(u16 dmemo, u16 dmemi, u16 n) {
    for (u32 k = 0; k < n / 2u; ++k)
        set_dmem_s16(dmemo + k * 2, clamp_s16(dmem_s16(dmemo + k * 2) + dmem_s16(dmemi + k * 2)));
}

void AudioHLE::mult_q44(u16 dmem, u16 n, s8 gain) {
    for (u32 k = 0; k < n / 2u; ++k)
        set_dmem_s16(dmem + k * 2, clamp_s16((static_cast<s32>(dmem_s16(dmem + k * 2)) * gain) >> 4));
}

void AudioHLE::interleave(u16 dmemo, u16 left, u16 right, u16 n) {
    // Two frames per step, like the microcode (which matters when the output
    // overlaps an input).
    for (u32 k = 0; k < n / 4u; ++k) {
        const s16 l1 = dmem_s16(left + k * 4), l2 = dmem_s16(left + k * 4 + 2);
        const s16 r1 = dmem_s16(right + k * 4), r2 = dmem_s16(right + k * 4 + 2);
        set_dmem_s16(dmemo + k * 8 + 0, l1);
        set_dmem_s16(dmemo + k * 8 + 2, r1);
        set_dmem_s16(dmemo + k * 8 + 4, l2);
        set_dmem_s16(dmemo + k * 8 + 6, r2);
    }
}

void AudioHLE::copy_every_other_sample(u16 dmemo, u16 dmemi, u16 samples) {
    for (u32 k = 0; k < samples; ++k) set_dmem_s16(dmemo + k * 2, dmem_s16(dmemi + k * 4));
}

void AudioHLE::repeat64(u16 dmemo, u16 dmemi, u8 times) {
    u8 block[128];
    for (u32 i = 0; i < 128; ++i) block[i] = dmem_u8(dmemi + i);
    for (u32 t = 0; t < times; ++t)
        for (u32 i = 0; i < 128; ++i) set_dmem_u8(dmemo + t * 128 + i, block[i]);
}

void AudioHLE::copy_blocks(u16 dmemo, u16 dmemi, u16 block_size, u8 blocks) {
    u32 o = dmemo, i = dmemi;
    s32 left_blocks = blocks;
    do {
        s32 left_bytes = block_size;
        do {
            for (u32 b = 0; b < 0x20; ++b) set_dmem_u8(o + b, dmem_u8(i + b));
            left_bytes -= 0x20;
            i += 0x20;
            o += 0x20;
        } while (left_bytes > 0);
    } while (--left_blocks > 0);
}

void AudioHLE::adpcm(bool init, bool loop_state, bool two_bit, u16 dmemo, u16 dmemi, u16 n, u32 state) {
    s16 last[16];
    if (init) {
        std::memset(last, 0, sizeof(last));
    } else {
        const u32 src = loop_state ? loop : state;
        for (int i = 0; i < 16; ++i) last[i] = dram_s16(src + i * 2);
    }
    // The previous frame leads the output, so a following RESAMPLE has
    // history to interpolate from.
    for (int i = 0; i < 16; ++i, dmemo += 2) set_dmem_s16(dmemo, last[i]);

    for (u32 left = align_up(n, 32); left != 0; left -= 32) {
        const u8 code = dmem_u8(dmemi++);
        const unsigned scale = code >> 4;
        const s16* book1 = table.data() + ((code & 0xF) << 4);
        const s16* book2 = book1 + 8;

        // Residuals: sign-extended nibbles (or 2-bit crumbs), scaled.
        s16 frame[16];
        if (two_bit) {
            const unsigned rshift = scale < 14 ? 14 - scale : 0;
            for (int i = 0; i < 4; ++i) {
                const u8 b = dmem_u8(dmemi++);
                frame[i * 4 + 0] = static_cast<s16>(static_cast<s16>((b & 0xC0) << 8) >> rshift);
                frame[i * 4 + 1] = static_cast<s16>(static_cast<s16>((b & 0x30) << 10) >> rshift);
                frame[i * 4 + 2] = static_cast<s16>(static_cast<s16>((b & 0x0C) << 12) >> rshift);
                frame[i * 4 + 3] = static_cast<s16>(static_cast<s16>((b & 0x03) << 14) >> rshift);
            }
        } else {
            const unsigned rshift = scale < 12 ? 12 - scale : 0;
            for (int i = 0; i < 8; ++i) {
                const u8 b = dmem_u8(dmemi++);
                frame[i * 2 + 0] = static_cast<s16>(static_cast<s16>((b & 0xF0) << 8) >> rshift);
                frame[i * 2 + 1] = static_cast<s16>(static_cast<s16>((b & 0x0F) << 12) >> rshift);
            }
        }

        // Second-order prediction from the two previous samples, per half.
        for (int half = 0; half < 2; ++half) {
            const s16* res = frame + half * 8;
            const s16 l1 = half ? last[6] : last[14];
            const s16 l2 = half ? last[7] : last[15];
            s16 outv[8];
            for (int i = 0; i < 8; ++i) {
                s32 acc = static_cast<s32>(res[i]) << 11;
                acc += book1[i] * l1 + book2[i] * l2 + rdot(i, book2, res);
                outv[i] = clamp_s16(acc >> 11);
            }
            std::memcpy(last + half * 8, outv, sizeof(outv));
        }
        for (int i = 0; i < 16; ++i, dmemo += 2) set_dmem_s16(dmemo, last[i]);
    }

    for (int i = 0; i < 16; ++i) set_dram_s16(state + i * 2, last[i]);
}

void AudioHLE::resample(bool init, u16 dmemo, u16 dmemi, u16 n, u32 pitch, u32 state) {
    // The four history samples go right in front of the input, so every
    // output reads a window of four consecutive DMEM samples.
    u32 ipos = (dmemi >> 1) - 4;
    u32 opos = dmemo >> 1;
    u32 accu = 0;
    if (init) {
        for (u32 k = 0; k < 4; ++k) set_sample(ipos + k, 0);
    } else {
        for (u32 k = 0; k < 4; ++k) set_sample(ipos + k, dram_s16(state + k * 2));
        accu = static_cast<u16>(dram_s16(state + 8));
    }

    for (u32 left = n >> 1; left != 0; --left) {
        const s16* lut = RESAMPLE_LUT + ((accu & 0xFC00) >> 8);
        const s32 acc = sample(ipos) * lut[0] + sample(ipos + 1) * lut[1] + sample(ipos + 2) * lut[2] +
                        sample(ipos + 3) * lut[3];
        set_sample(opos++, clamp_s16(acc >> 15));
        accu += pitch;
        ipos += accu >> 16;
        accu &= 0xFFFF;
    }

    for (u32 k = 0; k < 4; ++k) set_dram_s16(state + k * 2, sample(ipos + k));
    set_dram_s16(state + 8, static_cast<s16>(accu));
}

void AudioHLE::resample_zoh(u16 dmemo, u16 dmemi, u16 n, u32 pitch, u32 pitch_accu) {
    u32 ipos = dmemi >> 1, opos = dmemo >> 1;
    for (u32 left = n >> 1; left != 0; --left) {
        set_sample(opos++, sample(ipos));
        pitch_accu += pitch;
        ipos += pitch_accu >> 16;
        pitch_accu &= 0xFFFF;
    }
}

void AudioHLE::polef(bool init, u16 dmemo, u16 dmemi, u16 n, s16 gain, u32 state) {
    // Two-pole IIR: coefficients in the table (h1 = first 8, h2 = next 8),
    // state = the last four outputs.
    s16* h1 = table.data();
    s16* h2 = table.data() + 8;
    s16 l1 = 0, l2 = 0;
    if (!init) {
        l1 = dram_s16(state + 4);
        l2 = dram_s16(state + 6);
    }
    s16 h2_before[8];
    for (int i = 0; i < 8; ++i) {
        h2_before[i] = h2[i];
        h2[i] = static_cast<s16>((static_cast<s32>(h2[i]) * gain) >> 14);
    }
    s16 last4[4] = {0, 0, 0, 0};
    u32 left = align_up(n, 16);
    do {
        s16 frame[8], outv[8];
        for (int i = 0; i < 8; ++i, dmemi += 2) frame[i] = dmem_s16(dmemi);
        for (int i = 0; i < 8; ++i) {
            s32 acc = static_cast<s32>(frame[i]) * gain;
            acc += h1[i] * l1 + h2_before[i] * l2 + rdot(i, h2, frame);
            outv[i] = clamp_s16(acc >> 14);
        }
        for (int i = 0; i < 8; ++i, dmemo += 2) set_dmem_s16(dmemo, outv[i]);
        l1 = outv[6];
        l2 = outv[7];
        std::memcpy(last4, outv + 4, sizeof(last4));
        left = left >= 16 ? left - 16 : 0;
    } while (left != 0);
    for (int i = 0; i < 4; ++i) set_dram_s16(state + i * 2, last4[i]);
}

void AudioHLE::iirf(bool init, u16 dmemo, u16 dmemi, u16 n, u32 state) {
    // Biquad with Q15 coefficients from the table: b0 = t[0], b1 = t[1],
    // b2 = t[0], feedback a1 = t[8], a2 = t[9] (both doubled).
    s16 y1 = 0, y2 = 0, x1 = 0, x2 = 0;
    if (!init) {
        y2 = dram_s16(state + 4);
        y1 = dram_s16(state + 6);
        x2 = dram_s16(state + 8);
        x1 = dram_s16(state + 10);
    }
    const s16* t = table.data();
    for (u32 left = align_up(n, 16) / 2; left != 0; --left, dmemi += 2, dmemo += 2) {
        const s16 x = dmem_s16(dmemi);
        const s32 acc = vmulf(t[0], x) + vmulf(t[1], x1) + vmulf(t[0], x2) + vmulf(t[8], y1) * 2 +
                        vmulf(t[9], y2) * 2;
        const s16 y = clamp_s16(acc);
        set_dmem_s16(dmemo, y);
        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;
    }
    set_dram_s16(state + 4, y2);
    set_dram_s16(state + 6, y1);
    set_dram_s16(state + 8, x2);
    set_dram_s16(state + 10, x1);
}

void AudioHLE::fir_filter(bool init, u16 dmem, u16 n, u32 state) {
    // 8-tap FIR over the buffer. The coefficients (set up by the previous
    // FILTER with flags > 1) are averaged with the last call's, kept at
    // state + 0x10, to soften filter changes; state holds the last 8 inputs.
    // A_INIT starts a voice: no history, nothing to average with.
    s16 c[8];
    for (int k = 0; k < 8; ++k) {
        const s16 cur = dram_s16(filter_lut[0] + k * 2);
        const s16 prev = init ? cur : dram_s16(filter_lut[1] + k * 2);
        const s16 v = static_cast<s16>((prev + cur) >> 1);
        set_dram_s16(filter_lut[0] + k * 2, v);
        set_dram_s16(filter_lut[1] + k * 2, v);
        c[k] = v;
    }
    s16 hist[16]; // [0..7] previous block, [8..15] current block
    for (int k = 0; k < 8; ++k) hist[k] = init ? 0 : dram_s16(state + k * 2);
    for (u32 x = 0; x < n; x += 16) {
        for (int k = 0; k < 8; ++k) hist[8 + k] = dmem_s16(dmem + x + k * 2);
        for (int i = 0; i < 8; ++i) {
            s32 acc = 0;
            for (int k = 0; k < 8; ++k) acc += c[k] * hist[8 + i - k];
            set_dmem_s16(dmem + x + i * 2, clamp_s16((acc + 0x4000) >> 15));
        }
        std::memcpy(hist, hist + 8, 8 * sizeof(s16));
    }
    for (int k = 0; k < 8; ++k) set_dram_s16(state + k * 2, hist[k]);
}

// State layout of the ABI 1 / n_audio envelope mixers (an 80-byte
// ENVMIX_STATE the game allocates): wet, dry, then 32-bit ramp fields.
namespace {
    enum : u32 { ENV_WET = 0, ENV_DRY = 2, ENV_TARGET = 4, ENV_RATE = 12, ENV_SEQ = 20, ENV_VALUE = 28 };
}

void AudioHLE::envmix_exp(bool init, bool aux, u32 state) {
    Ramp ramps[2];
    s32 exp_seq[2], exp_rate[2];
    s16 d = dry, w = wet;
    if (init) {
        for (int i = 0; i < 2; ++i) {
            ramps[i].value = static_cast<s64>(vol[i]) << 16;
            ramps[i].target = static_cast<s64>(target[i]) << 16;
            exp_rate[i] = rate[i];
            exp_seq[i] = static_cast<s32>(static_cast<s64>(vol[i]) * rate[i]);
        }
    } else {
        w = dram_s16(state + ENV_WET);
        d = dram_s16(state + ENV_DRY);
        for (int i = 0; i < 2; ++i) {
            ramps[i].target = dram_s32(state + ENV_TARGET + i * 4);
            exp_rate[i] = dram_s32(state + ENV_RATE + i * 4);
            exp_seq[i] = dram_s32(state + ENV_SEQ + i * 4);
            ramps[i].value = dram_s32(state + ENV_VALUE + i * 4);
        }
    }
    // step != 0 exactly while a ramp hasn't reached its target.
    for (auto& r : ramps) r.step = r.target - r.value;

    const u16 dl = out, dr = dry_right, wl = wet_left, wr = wet_right, src = in;
    u32 k = 0;
    for (u32 y = 0; y < count; y += 16) {
        for (int i = 0; i < 2; ++i) {
            if (ramps[i].step != 0) {
                exp_seq[i] = static_cast<s32>((static_cast<s64>(exp_seq[i]) * exp_rate[i]) >> 16);
                ramps[i].step = (exp_seq[i] - ramps[i].value) >> 3;
            }
        }
        for (int j = 0; j < 8; ++j, ++k) {
            const s16 lv = ramps[0].tick(), rv = ramps[1].tick();
            const s32 s = dmem_s16(src + k * 2);
            const s16 g[4] = {clamp_s16((lv * d + 0x4000) >> 15), clamp_s16((rv * d + 0x4000) >> 15),
                              clamp_s16((lv * w + 0x4000) >> 15), clamp_s16((rv * w + 0x4000) >> 15)};
            const u16 bufs[4] = {dl, dr, wl, wr};
            for (int b = 0; b < (aux ? 4 : 2); ++b) {
                const u32 a = bufs[b] + k * 2;
                set_dmem_s16(a, clamp_s16(dmem_s16(a) + ((s * g[b]) >> 15)));
            }
        }
    }

    set_dram_s16(state + ENV_WET, w);
    set_dram_s16(state + ENV_DRY, d);
    for (int i = 0; i < 2; ++i) {
        set_dram_s32(state + ENV_TARGET + i * 4, static_cast<s32>(ramps[i].target));
        set_dram_s32(state + ENV_RATE + i * 4, exp_rate[i]);
        set_dram_s32(state + ENV_SEQ + i * 4, exp_seq[i]);
        set_dram_s32(state + ENV_VALUE + i * 4, static_cast<s32>(ramps[i].value));
    }
}

void AudioHLE::envmix_ge(bool init, bool aux, u32 state) {
    Ramp ramps[2];
    s16 d = dry, w = wet;
    if (init) {
        for (int i = 0; i < 2; ++i) {
            ramps[i].value = static_cast<s64>(vol[i]) << 16;
            ramps[i].target = static_cast<s64>(target[i]) << 16;
            ramps[i].step = rate[i] / 8;
        }
    } else {
        w = dram_s16(state + ENV_WET);
        d = dram_s16(state + ENV_DRY);
        for (int i = 0; i < 2; ++i) {
            ramps[i].target = dram_s32(state + ENV_TARGET + i * 4);
            ramps[i].step = dram_s32(state + ENV_RATE + i * 4);
            ramps[i].value = dram_s32(state + ENV_VALUE + i * 4);
        }
    }
    const u16 bufs[4] = {out, dry_right, wet_left, wet_right};
    for (u32 k = 0; k < count / 2u; ++k) {
        const s16 lv = ramps[0].tick(), rv = ramps[1].tick();
        const s32 s = dmem_s16(in + k * 2);
        const s16 g[4] = {clamp_s16((lv * d + 0x4000) >> 15), clamp_s16((rv * d + 0x4000) >> 15),
                          clamp_s16((lv * w + 0x4000) >> 15), clamp_s16((rv * w + 0x4000) >> 15)};
        for (int b = 0; b < (aux ? 4 : 2); ++b) {
            const u32 a = bufs[b] + k * 2;
            set_dmem_s16(a, clamp_s16(dmem_s16(a) + ((s * g[b]) >> 15)));
        }
    }
    set_dram_s16(state + ENV_WET, w);
    set_dram_s16(state + ENV_DRY, d);
    for (int i = 0; i < 2; ++i) {
        set_dram_s32(state + ENV_TARGET + i * 4, static_cast<s32>(ramps[i].target));
        set_dram_s32(state + ENV_RATE + i * 4, static_cast<s32>(ramps[i].step));
        set_dram_s32(state + ENV_VALUE + i * 4, static_cast<s32>(ramps[i].value));
    }
}

void AudioHLE::envmix_lin(bool init, u16 dl, u16 dr, u16 wl, u16 wr, u16 dmemi, u16 n, u32 state) {
    Ramp ramps[2];
    s16 d = dry, w = wet;
    if (init) {
        for (int i = 0; i < 2; ++i) {
            ramps[i].step = rate[i] / 8;
            ramps[i].value = static_cast<s64>(vol[i]) << 16;
            ramps[i].target = static_cast<s64>(target[i]) << 16;
        }
    } else {
        w = dram_s16(state + ENV_WET);
        d = dram_s16(state + ENV_DRY);
        for (int i = 0; i < 2; ++i) {
            ramps[i].target = static_cast<s64>(dram_s16(state + ENV_TARGET + i * 4)) << 16;
            ramps[i].step = dram_s32(state + ENV_RATE + i * 4);
            ramps[i].value = dram_s32(state + ENV_VALUE + i * 4);
        }
    }
    const u16 bufs[4] = {dl, dr, wl, wr};
    for (u32 k = 0; k < n / 2u; ++k) {
        const s16 lv = ramps[0].tick(), rv = ramps[1].tick();
        const s32 s = dmem_s16(dmemi + k * 2);
        const s16 g[4] = {clamp_s16((lv * d + 0x4000) >> 15), clamp_s16((rv * d + 0x4000) >> 15),
                          clamp_s16((lv * w + 0x4000) >> 15), clamp_s16((rv * w + 0x4000) >> 15)};
        for (int b = 0; b < 4; ++b) {
            const u32 a = bufs[b] + k * 2;
            set_dmem_s16(a, clamp_s16(dmem_s16(a) + ((s * g[b]) >> 15)));
        }
    }
    set_dram_s16(state + ENV_WET, w);
    set_dram_s16(state + ENV_DRY, d);
    for (int i = 0; i < 2; ++i) {
        set_dram_s16(state + ENV_TARGET + i * 4, static_cast<s16>(ramps[i].target >> 16));
        set_dram_s32(state + ENV_RATE + i * 4, static_cast<s32>(ramps[i].step));
        set_dram_s32(state + ENV_VALUE + i * 4, static_cast<s32>(ramps[i].value));
    }
}

void AudioHLE::envmix_nead(bool swap_wet, u16 dl, u16 dr, u16 wl, u16 wr, u16 dmemi, u32 samples,
                           const s16 xors[4]) {
    if (swap_wet) std::swap(wl, wr);
    // Volumes are unsigned Q16; a set bit in `xors` inverts that output's
    // phase (the headset/surround effects).
    for (u32 k = 0; k < align_up(samples, 8); k += 8) {
        for (u32 i = k; i < k + 8; ++i) {
            const s32 s = dmem_s16(dmemi + i * 2);
            const s16 l = static_cast<s16>(static_cast<s16>((s * static_cast<s32>(env_values[0])) >> 16) ^ xors[0]);
            const s16 r = static_cast<s16>(static_cast<s16>((s * static_cast<s32>(env_values[1])) >> 16) ^ xors[1]);
            const s16 l2 = static_cast<s16>(static_cast<s16>((l * static_cast<s32>(env_values[2])) >> 16) ^ xors[2]);
            const s16 r2 = static_cast<s16>(static_cast<s16>((r * static_cast<s32>(env_values[2])) >> 16) ^ xors[3]);
            set_dmem_s16(dl + i * 2, clamp_s16(dmem_s16(dl + i * 2) + l));
            set_dmem_s16(dr + i * 2, clamp_s16(dmem_s16(dr + i * 2) + r));
            set_dmem_s16(wl + i * 2, clamp_s16(dmem_s16(wl + i * 2) + l2));
            set_dmem_s16(wr + i * 2, clamp_s16(dmem_s16(wr + i * 2) + r2));
        }
        for (int j = 0; j < 3; ++j) env_values[j] = static_cast<u16>(env_values[j] + env_steps[j]);
    }
}

// -----------------------------------------------------------------------------
// ABI 1 (libultra "audio" microcode)
// -----------------------------------------------------------------------------

void AudioHLE::run_audio(u32 acmd, u32 w1, u32 w2) {
    const u8 flags = static_cast<u8>(w1 >> 16);
    switch (acmd) {
        case 0x1: // ADPCM
            adpcm(flags & A_INIT, flags & 0x2, false, out, in, static_cast<u16>(align_up(count, 32)),
                  segment_address(w2));
            break;
        case 0x2: { // CLEARBUFF
            const u16 n = w2 & 0xFFF;
            if (n) clear(static_cast<u16>(w1 + AUDIO_DMEM_BASE), static_cast<u16>(align_up(n, 16)));
            break;
        }
        case 0x3: // ENVMIXER
            if (abi == Abi::AudioGE) envmix_ge(flags & A_INIT, flags & A_AUX, segment_address(w2));
            else envmix_exp(flags & A_INIT, flags & A_AUX, segment_address(w2));
            break;
        case 0x4: // LOADBUFF
            if (count) load(in, segment_address(w2), count);
            break;
        case 0x5: // RESAMPLE
            resample(flags & A_INIT, out, in, static_cast<u16>(align_up(count, 16)), (w1 & 0xFFFF) << 1,
                     segment_address(w2));
            break;
        case 0x6: // SAVEBUFF
            if (count) save(out, segment_address(w2), count);
            break;
        case 0x7: // SEGMENT
            segments[(w2 >> 24) & 0xFF] = w2 & 0x00FFFFFF;
            break;
        case 0x8: // SETBUFF
            if (flags & A_AUX) {
                dry_right = static_cast<u16>(w1 + AUDIO_DMEM_BASE);
                wet_left = static_cast<u16>((w2 >> 16) + AUDIO_DMEM_BASE);
                wet_right = static_cast<u16>(w2 + AUDIO_DMEM_BASE);
            } else {
                in = static_cast<u16>(w1 + AUDIO_DMEM_BASE);
                out = static_cast<u16>((w2 >> 16) + AUDIO_DMEM_BASE);
                count = static_cast<u16>(w2);
            }
            break;
        case 0x9: // SETVOL
            if (flags & A_AUX) {
                dry = static_cast<s16>(w1);
                wet = static_cast<s16>(w2);
            } else {
                const int lr = (flags & A_LEFT) ? 0 : 1;
                if (flags & A_VOL) {
                    vol[lr] = static_cast<s16>(w1);
                } else {
                    target[lr] = static_cast<s16>(w1);
                    rate[lr] = static_cast<s32>(w2);
                }
            }
            break;
        case 0xA: { // DMEMMOVE
            const u16 n = static_cast<u16>(w2);
            if (n)
                move(static_cast<u16>((w2 >> 16) + AUDIO_DMEM_BASE), static_cast<u16>(w1 + AUDIO_DMEM_BASE),
                     static_cast<u16>(align_up(n, 16)));
            break;
        }
        case 0xB: // LOADADPCM
            load_table(segment_address(w2), align_up(w1 & 0xFFFF, 8) >> 1);
            break;
        case 0xC: // MIXER
            if (count)
                mix(static_cast<u16>(w2 + AUDIO_DMEM_BASE), static_cast<u16>((w2 >> 16) + AUDIO_DMEM_BASE),
                    static_cast<u16>(align_up(count, 32)), static_cast<s16>(w1));
            break;
        case 0xD: // INTERLEAVE
            if (count)
                interleave(out, static_cast<u16>((w2 >> 16) + AUDIO_DMEM_BASE), static_cast<u16>(w2 + AUDIO_DMEM_BASE),
                           static_cast<u16>(align_up(count, 16)));
            break;
        case 0xE: // POLEF
            if (count)
                polef(flags & A_INIT, out, in, static_cast<u16>(align_up(count, 16)), static_cast<s16>(w1),
                      segment_address(w2));
            break;
        case 0xF: // SETLOOP
            loop = segment_address(w2);
            break;
        default: break; // SPNOOP
    }
}

// -----------------------------------------------------------------------------
// n_audio
// -----------------------------------------------------------------------------

void AudioHLE::run_naudio(u32 acmd, u32 w1, u32 w2) {
    const u8 flags = static_cast<u8>(w1 >> 16);
    const u32 address24 = w2 & 0x00FFFFFF;
    switch (acmd) {
        case 0x1: // ADPCM: state in w1, flags/count/input offset/output in w2
            adpcm((w2 >> 28) & 0x1, (w2 >> 28) & 0x2, false, static_cast<u16>((w2 & 0xFFF) + NAUDIO_MAIN),
                  static_cast<u16>(((w2 >> 12) & 0xF) + NAUDIO_MAIN),
                  static_cast<u16>(align_up((w2 >> 16) & 0xFFF, 32)), w1 & 0x00FFFFFF);
            break;
        case 0x2: // CLEARBUFF
            clear(static_cast<u16>(w1 + NAUDIO_MAIN), w2 & 0xFFF);
            break;
        case 0x3: // ENVMIXER
            vol[1] = static_cast<s16>(w1);
            envmix_lin(flags & A_INIT, NAUDIO_DRY_LEFT, NAUDIO_DRY_RIGHT, NAUDIO_WET_LEFT, NAUDIO_WET_RIGHT,
                       NAUDIO_MAIN, NAUDIO_COUNT, address24);
            break;
        case 0x4: // LOADBUFF
            load(static_cast<u16>((w1 & 0xFFF) + NAUDIO_MAIN), address24, (w1 >> 12) & 0xFFF);
            break;
        case 0x5: // RESAMPLE: state in w1, flags/pitch/input/output select in w2
            resample((w2 >> 30) & 0x1, (w2 & 0x3) ? NAUDIO_MAIN2 : NAUDIO_MAIN,
                     static_cast<u16>(((w2 >> 2) & 0xFFF) + NAUDIO_MAIN), NAUDIO_COUNT, ((w2 >> 14) & 0xFFFF) << 1,
                     w1 & 0x00FFFFFF);
            break;
        case 0x6: // SAVEBUFF
            save(static_cast<u16>((w1 & 0xFFF) + NAUDIO_MAIN), address24, (w1 >> 12) & 0xFFF);
            break;
        case 0x7: // MP3 (NAudioMP3) - not emulated; DK64 reuses the slot for MIXER
        case 0x8:
            if (abi == Abi::NAudioDK)
                mix(static_cast<u16>(w2 + NAUDIO_MAIN), static_cast<u16>((w2 >> 16) + NAUDIO_MAIN), NAUDIO_COUNT,
                    static_cast<s16>(w1));
            break;
        case 0x9: // SETVOL
            if (flags & A_VOL) {
                if (flags & A_LEFT) {
                    vol[0] = static_cast<s16>(w1);
                    dry = static_cast<s16>(w2 >> 16);
                    wet = static_cast<s16>(w2);
                } else {
                    target[1] = static_cast<s16>(w1);
                    rate[1] = static_cast<s32>(w2);
                }
            } else {
                target[0] = static_cast<s16>(w1);
                rate[0] = static_cast<s32>(w2);
            }
            break;
        case 0xA: // DMEMMOVE
            move(static_cast<u16>((w2 >> 16) + NAUDIO_MAIN), static_cast<u16>(w1 + NAUDIO_MAIN),
                 static_cast<u16>(align_up(w2 & 0xFFFF, 4)));
            break;
        case 0xB: // LOADADPCM
            load_table(address24, (w1 & 0xFFFF) >> 1);
            break;
        case 0xC: // MIXER
            mix(static_cast<u16>(w2 + NAUDIO_MAIN), static_cast<u16>((w2 >> 16) + NAUDIO_MAIN), NAUDIO_COUNT,
                static_cast<s16>(w1));
            break;
        case 0xD: // INTERLEAVE
            interleave(NAUDIO_MAIN, NAUDIO_DRY_LEFT, NAUDIO_DRY_RIGHT, NAUDIO_COUNT);
            break;
        case 0xE:
            if (abi == Abi::NAudioMP3) {
                // POLEF / IIR filter on one of the two main buffers.
                const u16 dmem = ((w2 >> 24) == 0) ? NAUDIO_MAIN : NAUDIO_MAIN2;
                if (table[0] == 0 && table[1] == 0)
                    polef(flags & A_INIT, dmem, dmem, NAUDIO_COUNT, static_cast<s16>(w1), address24);
                else
                    iirf(flags & A_INIT, dmem, dmem, NAUDIO_COUNT, address24);
            } else {
                // This microcode's jump table sends 0x0E into the middle of
                // SETVOL, which just replaces the low half of the right rate.
                rate[1] = static_cast<s32>((static_cast<u32>(rate[1]) & 0xFFFF0000u) | (w2 & 0xFFFF));
            }
            break;
        case 0xF: // SETLOOP
            loop = address24;
            break;
        default: break;
    }
}

// -----------------------------------------------------------------------------
// Nintendo EAD microcodes
// -----------------------------------------------------------------------------

void AudioHLE::run_nead(u32 acmd, u32 w1, u32 w2) {
    const u8 flags = static_cast<u8>(w1 >> 16);
    const u32 address = w2 & 0x00FFFFFF;
    const bool mk = abi == Abi::NeadMK;
    const bool zelda = abi == Abi::NeadZelda;
    switch (acmd) {
        case 0x01: // ADPCM
            adpcm(flags & 0x1, flags & 0x2, flags & 0x4, out, in, static_cast<u16>(align_up(count, 32)), address);
            break;
        case 0x02: { // CLEARBUFF
            const u16 n = w2 & 0xFFF;
            if (n) clear(static_cast<u16>(w1), n);
            break;
        }
        case 0x04: // ADDMIXER
            if (!mk) add(static_cast<u16>(w2), static_cast<u16>(w2 >> 16), (w1 >> 12) & 0xFF0);
            break;
        case 0x05: // RESAMPLE
            resample(flags & 0x1, out, in, static_cast<u16>(align_up(count, 16)), (w1 & 0xFFFF) << 1, address);
            break;
        case 0x06: // RESAMPLE_ZOH
            if (!mk) resample_zoh(out, in, count, (w1 & 0xFFFF) << 1, w2 & 0xFFFF);
            break;
        case 0x07: // FILTER (Zelda family); SEGMENT (a no-op) elsewhere
            if (zelda) {
                if (flags > 1) {
                    filter_count = static_cast<u16>(w1);
                    filter_lut[0] = address;
                } else {
                    filter_lut[1] = address + 0x10;
                    fir_filter(flags & A_INIT, static_cast<u16>(w1), filter_count, address);
                }
            }
            break;
        case 0x08: // SETBUFF
            in = static_cast<u16>(w1);
            out = static_cast<u16>(w2 >> 16);
            count = static_cast<u16>(w2);
            break;
        case 0x09: // DUPLICATE (Zelda family)
            if (zelda) repeat64(static_cast<u16>(w2 >> 16), static_cast<u16>(w1), flags);
            break;
        case 0x0A: { // DMEMMOVE
            const u16 n = static_cast<u16>(w2);
            if (n) move(static_cast<u16>(w2 >> 16), static_cast<u16>(w1), static_cast<u16>(align_up(n, 4)));
            break;
        }
        case 0x0B: // LOADADPCM
            load_table(address, (w1 & 0xFFFF) >> 1);
            break;
        case 0x0C: // MIXER
            mix(static_cast<u16>(w2), static_cast<u16>(w2 >> 16), (w1 >> 12) & 0xFF0, static_cast<s16>(w1));
            break;
        case 0x0D: // INTERLEAVE
            if (zelda) {
                interleave(static_cast<u16>(w1), static_cast<u16>(w2 >> 16), static_cast<u16>(w2), (w1 >> 12) & 0xFF0);
            } else if (count) {
                interleave(out, static_cast<u16>(w2 >> 16), static_cast<u16>(w2), count);
            }
            break;
        case 0x0E: // HILOGAIN (Zelda family); POLEF elsewhere
            if (zelda) {
                mult_q44(static_cast<u16>(w2 >> 16), w1 & 0xFFF, static_cast<s8>(flags));
            } else if (count) {
                polef(flags & A_INIT, out, in, count, static_cast<s16>(w1), address);
            }
            break;
        case 0x0F: // SETLOOP
            loop = address;
            break;
        case 0x10: // copy blocks
            copy_blocks(static_cast<u16>(w2 >> 16), static_cast<u16>(w1), static_cast<u16>(w2), flags);
            break;
        case 0x11: // INTERL: every other sample (halves the rate)
            copy_every_other_sample(static_cast<u16>(w2), static_cast<u16>(w2 >> 16), static_cast<u16>(w1));
            break;
        case 0x12: // ENVSETUP1
            env_values[2] = static_cast<u16>((w1 >> 8) & 0xFF00);
            env_steps[2] = mk ? 0 : static_cast<u16>(w1);
            env_steps[0] = static_cast<u16>(w2 >> 16);
            env_steps[1] = static_cast<u16>(w2);
            break;
        case 0x13: { // ENVMIXER
            const s16 xors[4] = {static_cast<s16>((w1 & 0x2) ? -1 : 0), static_cast<s16>((w1 & 0x1) ? -1 : 0),
                                 static_cast<s16>(!mk && (w1 & 0x8) ? -1 : 0),
                                 static_cast<s16>(!mk && (w1 & 0x4) ? -1 : 0)};
            envmix_nead(!mk && ((w1 >> 4) & 0x1), static_cast<u16>((w2 >> 20) & 0xFF0),
                        static_cast<u16>((w2 >> 12) & 0xFF0), static_cast<u16>((w2 >> 4) & 0xFF0),
                        static_cast<u16>((w2 << 4) & 0xFF0), static_cast<u16>((w1 >> 12) & 0xFF0), (w1 >> 8) & 0xFF,
                        xors);
            break;
        }
        case 0x14: // LOADBUFF
            load(static_cast<u16>(w1 & 0xFFF), address, (w1 >> 12) & 0xFFF);
            break;
        case 0x15: // SAVEBUFF
            save(static_cast<u16>(w1 & 0xFFF), address, (w1 >> 12) & 0xFFF);
            break;
        case 0x16: // ENVSETUP2
            env_values[0] = static_cast<u16>(w2 >> 16);
            env_values[1] = static_cast<u16>(w2);
            break;
        case 0x18: // HILOGAIN (Star Fox family)
            if (!mk && !zelda) mult_q44(static_cast<u16>(w2 >> 16), w1 & 0xFFF, static_cast<s8>(flags));
            break;
        case 0x1A: // DUPLICATE (Star Fox family)
            if (!mk && !zelda) repeat64(static_cast<u16>(w2 >> 16), static_cast<u16>(w1), flags);
            break;
        case 0x1B: // FILTER (Wave Race 64 J rev B)
            if (abi == Abi::NeadSF) {
                if (flags > 1) {
                    filter_count = static_cast<u16>(w1);
                    filter_lut[0] = address;
                } else {
                    filter_lut[1] = address + 0x10;
                    fir_filter(flags & A_INIT, static_cast<u16>(w1), filter_count, address);
                }
            }
            break;
        default: break;
    }
}

// -----------------------------------------------------------------------------
// Factor 5 MusyX synthesizer (v1 and v2)
// -----------------------------------------------------------------------------

namespace {
    constexpr size_t MUSYX_SUBFRAME_SIZE = 192;
    constexpr size_t MUSYX_MAX_VOICES = 32;
    constexpr size_t MUSYX_SAMPLE_BUFFER_SIZE = 0x200;

    // SFD offsets (shared across v1 and v2)
    constexpr u32 SFD_VOICE_COUNT     = 0x00;
    constexpr u32 SFD_SFX_INDEX       = 0x02;
    constexpr u32 SFD_VOICE_BITMASK   = 0x04;
    constexpr u32 SFD_STATE_PTR       = 0x08;
    constexpr u32 SFD_SFX_PTR         = 0x0C;
    constexpr u32 SFD_VOICES          = 0x10;

    // MusyX v2 SFD offsets
    constexpr u32 SFD2_10_PTR         = 0x10;
    constexpr u32 SFD2_14_BITMASK     = 0x14;
    constexpr u32 SFD2_15_BITMASK     = 0x15;
    constexpr u32 SFD2_16_BITMASK     = 0x16;
    constexpr u32 SFD2_18_PTR         = 0x18;
    constexpr u32 SFD2_1C_PTR         = 0x1C;
    constexpr u32 SFD2_20_PTR         = 0x20;
    constexpr u32 SFD2_24_PTR         = 0x24;
    constexpr u32 SFD2_VOICES         = 0x28;

    // Per-voice parameter block (80 bytes)
    constexpr u32 VOICE_ENV_BEGIN         = 0x00;
    constexpr u32 VOICE_ENV_STEP          = 0x10;
    constexpr u32 VOICE_PITCH_Q16         = 0x20;
    constexpr u32 VOICE_PITCH_SHIFT       = 0x22;
    constexpr u32 VOICE_CATSRC_0          = 0x24;
    constexpr u32 VOICE_CATSRC_1          = 0x30;
    constexpr u32 VOICE_ADPCM_FRAMES      = 0x3C;
    constexpr u32 VOICE_SKIP_SAMPLES      = 0x3E;
    constexpr u32 VOICE_U16_40            = 0x40;
    constexpr u32 VOICE_U16_42            = 0x42;
    constexpr u32 VOICE_ADPCM_TABLE_PTR   = 0x40;
    constexpr u32 VOICE_INTERLEAVED_PTR   = 0x44;
    constexpr u32 VOICE_END_POINT         = 0x48;
    constexpr u32 VOICE_RESTART_POINT     = 0x4A;
    constexpr u32 VOICE_U16_4C            = 0x4C;
    constexpr u32 VOICE_U16_4E            = 0x4E;
    constexpr u32 VOICE_SIZE              = 0x50;

    // Concatenated sample source descriptor
    constexpr u32 CATSRC_PTR1  = 0x00;
    constexpr u32 CATSRC_PTR2  = 0x04;
    constexpr u32 CATSRC_SIZE1 = 0x08;
    constexpr u32 CATSRC_SIZE2 = 0x0A;

    // State buffer layout
    constexpr u32 STATE_LAST_SAMPLE  = 0x000;
    constexpr u32 STATE_BASE_VOL     = 0x100;
    constexpr u32 STATE_CC0          = 0x110;
    constexpr u32 STATE_740_LAST4_V1 = 0x290;
    constexpr u32 STATE_740_LAST4_V2 = 0x110;

    // Sound effects delay/reverb processor block
    constexpr u32 SFX_CBUFFER_PTR    = 0x00;
    constexpr u32 SFX_CBUFFER_LENGTH = 0x04;
    constexpr u32 SFX_TAP_COUNT      = 0x08;
    constexpr u32 SFX_FIR4_HGAIN     = 0x0A;
    constexpr u32 SFX_TAP_DELAYS     = 0x0C;
    constexpr u32 SFX_TAP_GAINS      = 0x2C;
    constexpr u32 SFX_U16_3C         = 0x3C;
    constexpr u32 SFX_U16_3E         = 0x3E;
    constexpr u32 SFX_FIR4_HCOEFFS   = 0x40;

    inline s32 musyx_dot4(const s16* x, const s16* y) {
        s32 accu = 0;
        for (size_t i = 0; i < 4; ++i) {
            accu = clamp_s16(accu + (((s32)x[i] * (s32)y[i]) >> 15));
        }
        return accu;
    }

    inline s16 musyx_predict_sample(u8 byte, u8 mask, unsigned lshift, unsigned rshift) {
        s16 sample = static_cast<u16>(byte & mask) << lshift;
        sample >>= rshift;
        return sample;
    }

    inline void musyx_predict_frame(s16* dst, const u8* src, const u8* nibbles, unsigned int rshift) {
        *(dst++) = static_cast<s16>((src[0] << 8) | src[1]);
        *(dst++) = static_cast<s16>((src[2] << 8) | src[3]);
        for (unsigned int i = 1; i < 16; ++i) {
            const u8 byte = nibbles[i];
            *(dst++) = musyx_predict_sample(byte, 0xF0, 8, rshift);
            *(dst++) = musyx_predict_sample(byte, 0x0F, 12, rshift);
        }
    }

    inline void musyx_compute_residuals(s16* dst, const s16* src, const s16* cb_entry,
                                       const s16* last_samples, size_t count) {
        const s16* const book1 = cb_entry;
        const s16* const book2 = cb_entry + 8;
        const s16 l1 = last_samples[0];
        const s16 l2 = last_samples[1];
        for (size_t i = 0; i < count; ++i) {
            s32 accu = static_cast<s32>(src[i]) << 11;
            accu += static_cast<s32>(book1[i]) * l1 + static_cast<s32>(book2[i]) * l2 + rdot(i, book2, src);
            dst[i] = clamp_s16(accu >> 11);
        }
    }

    void musyx_decode_frames(s16* dst, const u8* src, const s16* table, u8 count, u8 skip_samples) {
        s16 frame[32];
        const u8* nibbles = src + 8;
        bool jump_gap = false;
        if (skip_samples >= 32) {
            jump_gap = true;
            nibbles += 16;
            src += 4;
        }
        for (unsigned int i = 0; i < count; ++i) {
            const u8 c2 = nibbles[0];
            const s16* book = (c2 & 0xF0) + table;
            const unsigned int rshift = (c2 & 0x0F);

            musyx_predict_frame(frame, src, nibbles, rshift);

            std::memcpy(dst, frame, 2 * sizeof(s16));
            musyx_compute_residuals(dst + 2, frame + 2, book, dst, 6);
            musyx_compute_residuals(dst + 8, frame + 8, book, dst + 6, 8);
            musyx_compute_residuals(dst + 16, frame + 16, book, dst + 14, 8);
            musyx_compute_residuals(dst + 24, frame + 24, book, dst + 22, 8);

            if (jump_gap) {
                nibbles += 8;
                src += 32;
            }
            jump_gap = !jump_gap;
            nibbles += 16;
            src += 4;
            dst += 32;
        }
    }

    inline void musyx_mix_samples(s16* y, s16 x, s16 hgain) {
        *y = clamp_s16(*y + ((static_cast<s32>(x) * hgain + 0x4000) >> 15));
    }

    inline void musyx_mix_subframes(s16* y, const s16* x, s16 hgain) {
        for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
            musyx_mix_samples(&y[i], x[i], hgain);
        }
    }

    inline void musyx_mix_fir4(s16* y, const s16* x, s16 hgain, const s16* hcoeffs) {
        s32 h[4];
        for (int k = 0; k < 4; ++k) {
            h[k] = (static_cast<s32>(hgain) * hcoeffs[k]) >> 15;
        }
        for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
            s32 v = (h[0] * x[i] + h[1] * x[i + 1] + h[2] * x[i + 2] + h[3] * x[i + 3]) >> 15;
            y[i] = clamp_s16(y[i] + v);
        }
    }
}

struct AudioHLE::MusyXState {
    s16 left[MUSYX_SUBFRAME_SIZE]{};
    s16 right[MUSYX_SUBFRAME_SIZE]{};
    s16 cc0[MUSYX_SUBFRAME_SIZE]{};
    s16 e50[MUSYX_SUBFRAME_SIZE]{};
    s32 base_vol[4]{0, 0, 0, 0};
    s16 subframe_740_last4[4]{0, 0, 0, 0};
};

void AudioHLE::musyx_dma_cat8(u8* dst, u32 catsrc_ptr) {
    const u32 ptr1  = dram_u32(catsrc_ptr + CATSRC_PTR1);
    const u32 ptr2  = dram_u32(catsrc_ptr + CATSRC_PTR2);
    const u16 size1 = dram_u16(catsrc_ptr + CATSRC_SIZE1);
    const u16 size2 = dram_u16(catsrc_ptr + CATSRC_SIZE2);

    for (size_t i = 0; i < size1; ++i) dst[i] = dram_u8(ptr1 + i);
    if (size2 != 0) {
        for (size_t i = 0; i < size2; ++i) dst[size1 + i] = dram_u8(ptr2 + i);
    }
}

void AudioHLE::musyx_dma_cat16(s16* dst, u32 catsrc_ptr) {
    const u32 ptr1  = dram_u32(catsrc_ptr + CATSRC_PTR1);
    const u32 ptr2  = dram_u32(catsrc_ptr + CATSRC_PTR2);
    const u16 size1 = dram_u16(catsrc_ptr + CATSRC_SIZE1);
    const u16 size2 = dram_u16(catsrc_ptr + CATSRC_SIZE2);

    const size_t count1 = size1 >> 1;
    const size_t count2 = size2 >> 1;

    for (size_t i = 0; i < count1; ++i) dst[i] = dram_s16(ptr1 + i * 2);
    if (size2 != 0) {
        for (size_t i = 0; i < count2; ++i) dst[count1 + i] = dram_s16(ptr2 + i * 2);
    }
}

void AudioHLE::musyx_load_samples_pcm16(u32 voice_ptr, s16* samples, unsigned& segbase, unsigned& offset) {
    const u8  u8_3e  = dram_u8(voice_ptr + VOICE_SKIP_SAMPLES);
    const u16 u16_40 = dram_u16(voice_ptr + VOICE_U16_40);
    const u16 u16_42 = dram_u16(voice_ptr + VOICE_U16_42);

    const unsigned count = align_up(u16_40 + u8_3e, 4);
    segbase = MUSYX_SAMPLE_BUFFER_SIZE - count;
    offset  = u8_3e;

    musyx_dma_cat16(samples + segbase, voice_ptr + VOICE_CATSRC_0);
    if (u16_42 != 0) {
        musyx_dma_cat16(samples, voice_ptr + VOICE_CATSRC_1);
    }
}

void AudioHLE::musyx_load_samples_adpcm(u32 voice_ptr, s16* samples, unsigned& segbase, unsigned& offset) {
    u8 buffer[MUSYX_SAMPLE_BUFFER_SIZE * 2 * 5 / 16];
    s16 adpcm_table[128];

    const u8  u8_3c = dram_u8(voice_ptr + VOICE_ADPCM_FRAMES);
    const u8  u8_3d = dram_u8(voice_ptr + VOICE_ADPCM_FRAMES + 1);
    const u8  u8_3e = dram_u8(voice_ptr + VOICE_SKIP_SAMPLES);
    const u8  u8_3f = dram_u8(voice_ptr + VOICE_SKIP_SAMPLES + 1);
    const u32 adpcm_table_ptr = dram_u32(voice_ptr + VOICE_ADPCM_TABLE_PTR);

    for (size_t i = 0; i < 128; ++i) {
        adpcm_table[i] = dram_s16(adpcm_table_ptr + i * 2);
    }

    const unsigned count = static_cast<unsigned>(u8_3c) << 5;
    segbase = MUSYX_SAMPLE_BUFFER_SIZE - count;
    offset  = u8_3e & 0x1F;

    musyx_dma_cat8(buffer, voice_ptr + VOICE_CATSRC_0);
    musyx_decode_frames(samples + segbase, buffer, adpcm_table, u8_3c, u8_3e);

    if (u8_3d != 0) {
        musyx_dma_cat8(buffer, voice_ptr + VOICE_CATSRC_1);
        musyx_decode_frames(samples, buffer, adpcm_table, u8_3d, u8_3f);
    }
}

void AudioHLE::musyx_mix_voice_samples(MusyXState& musyx, u32 voice_ptr, const s16* samples,
                                      unsigned segbase, unsigned offset, u32 last_sample_ptr) {
    const u16 pitch_q16   = dram_u16(voice_ptr + VOICE_PITCH_Q16);
    const u16 pitch_shift = dram_u16(voice_ptr + VOICE_PITCH_SHIFT);

    const u16 end_point     = dram_u16(voice_ptr + VOICE_END_POINT);
    const u16 restart_point = dram_u16(voice_ptr + VOICE_RESTART_POINT);

    const u16 u16_4e = dram_u16(voice_ptr + VOICE_U16_4E);

    const s16* sample = samples + segbase + offset + u16_4e;
    const s16* const sample_end = samples + segbase + end_point;
    const s16* const sample_restart = samples + (restart_point & 0x7FFF) +
                                      (((restart_point & 0x8000) != 0) ? 0 : segbase);

    u32 pitch_accu = pitch_q16;
    const u32 pitch_step = static_cast<u32>(pitch_shift) << 4;

    s32 v4_env[4];
    s32 v4_env_step[4];
    for (int k = 0; k < 4; ++k) {
        v4_env[k]      = dram_s32(voice_ptr + VOICE_ENV_BEGIN + k * 4);
        v4_env_step[k] = dram_s32(voice_ptr + VOICE_ENV_STEP  + k * 4);
    }

    s16* v4_dst[4] = { musyx.left, musyx.right, musyx.cc0, musyx.e50 };
    s16 v4[4]{0, 0, 0, 0};

    for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
        const s16* lut = RESAMPLE_LUT + ((pitch_accu & 0xFC00) >> 8);

        sample += (pitch_accu >> 16);
        pitch_accu &= 0xFFFF;
        pitch_accu += pitch_step;

        const std::ptrdiff_t dist = sample - sample_end;
        if (dist >= 0) {
            sample = sample_restart + dist;
        }

        const s16 v = clamp_s16(musyx_dot4(sample, lut));

        for (int k = 0; k < 4; ++k) {
            const s32 accu = (static_cast<s32>(v) * (v4_env[k] >> 16)) >> 15;
            v4[k] = clamp_s16(accu);
            *(v4_dst[k]) = clamp_s16(accu + *(v4_dst[k]));

            ++(v4_dst[k]);
            v4_env[k] += v4_env_step[k];
        }
    }

    for (int k = 0; k < 4; ++k) {
        set_dram_s16(last_sample_ptr + k * 2, v4[k]);
    }
}

u32 AudioHLE::musyx_voice_stage(MusyXState& musyx, u32 voice_ptr, u32 last_sample_ptr) {
    u32 output_ptr = 0;
    if (dram_u16(voice_ptr + VOICE_CATSRC_0 + CATSRC_SIZE1) == 0) {
        output_ptr = dram_u32(voice_ptr + VOICE_INTERLEAVED_PTR);
    } else {
        for (int i = 0; i < static_cast<int>(MUSYX_MAX_VOICES); ++i) {
            s16 samples[MUSYX_SAMPLE_BUFFER_SIZE];
            unsigned segbase = 0;
            unsigned offset = 0;

            if (dram_u8(voice_ptr + VOICE_ADPCM_FRAMES) == 0) {
                musyx_load_samples_pcm16(voice_ptr, samples, segbase, offset);
            } else {
                musyx_load_samples_adpcm(voice_ptr, samples, segbase, offset);
            }

            musyx_mix_voice_samples(musyx, voice_ptr, samples, segbase, offset, last_sample_ptr + i * 8);

            output_ptr = dram_u32(voice_ptr + VOICE_INTERLEAVED_PTR);
            if (output_ptr != 0) {
                break;
            }

            voice_ptr += VOICE_SIZE;
        }
    }
    return output_ptr;
}

void AudioHLE::musyx_sfx_stage(MusyXState& musyx, u32 sfx_ptr, u16 idx, bool is_v2) {
    if (sfx_ptr == 0) return;

    s16 buffer[MUSYX_SUBFRAME_SIZE + 4];
    s16* subframe = buffer + 4;

    const u32 pos = static_cast<u32>(idx) * MUSYX_SUBFRAME_SIZE;

    const u32 cbuffer_ptr    = dram_u32(sfx_ptr + SFX_CBUFFER_PTR);
    const u32 cbuffer_length = dram_u32(sfx_ptr + SFX_CBUFFER_LENGTH);
    if (cbuffer_ptr == 0 || cbuffer_length == 0) return;

    const u16 tap_count = dram_u16(sfx_ptr + SFX_TAP_COUNT);

    u32 tap_delays[8];
    s16 tap_gains[8];
    for (int k = 0; k < 8; ++k) {
        tap_delays[k] = dram_u32(sfx_ptr + SFX_TAP_DELAYS + k * 4);
        tap_gains[k]  = dram_s16(sfx_ptr + SFX_TAP_GAINS  + k * 2);
    }

    const s16 fir4_hgain = dram_s16(sfx_ptr + SFX_FIR4_HGAIN);
    s16 fir4_hcoeffs[4];
    for (int k = 0; k < 4; ++k) {
        fir4_hcoeffs[k] = dram_s16(sfx_ptr + SFX_FIR4_HCOEFFS + k * 2);
    }

    const u16 sfx_gains[2] = {
        dram_u16(sfx_ptr + SFX_U16_3C),
        dram_u16(sfx_ptr + SFX_U16_3E)
    };

    std::memset(subframe, 0, MUSYX_SUBFRAME_SIZE * sizeof(s16));
    s16 delayed[MUSYX_SUBFRAME_SIZE];

    for (size_t i = 0; i < tap_count && i < 8; ++i) {
        int dpos = static_cast<int>(pos) - static_cast<int>(tap_delays[i]);
        while (dpos <= 0) {
            dpos += static_cast<int>(cbuffer_length);
        }
        int dlength = static_cast<int>(MUSYX_SUBFRAME_SIZE);

        if (static_cast<u32>(dpos + MUSYX_SUBFRAME_SIZE) > cbuffer_length) {
            dlength = static_cast<int>(cbuffer_length) - dpos;
            for (int s = 0; s < static_cast<int>(MUSYX_SUBFRAME_SIZE) - dlength; ++s) {
                delayed[dlength + s] = dram_s16(cbuffer_ptr + s * 2);
            }
        }

        for (int s = 0; s < dlength; ++s) {
            delayed[s] = dram_s16(cbuffer_ptr + (dpos + s) * 2);
        }

        musyx_mix_subframes(subframe, delayed, tap_gains[i]);
    }

    if (!is_v2) {
        for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
            const s16 v = subframe[i];
            musyx.left[i]  = clamp_s16(musyx.left[i]  + v);
            musyx.right[i] = clamp_s16(musyx.right[i] + v);
        }
    } else {
        for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
            const s16 v = subframe[i];
            const s16 v1 = static_cast<s16>((static_cast<s32>(v) * sfx_gains[0]) >> 16);
            const s16 v2 = static_cast<s16>((static_cast<s32>(v) * sfx_gains[1]) >> 16);

            musyx.left[i]  = clamp_s16(musyx.left[i]  + v1);
            musyx.right[i] = clamp_s16(musyx.right[i] + v1);
            musyx.cc0[i]   = clamp_s16(musyx.cc0[i]   + v2);
        }
    }

    std::memcpy(buffer, musyx.subframe_740_last4, 4 * sizeof(s16));
    std::memcpy(musyx.subframe_740_last4, subframe + MUSYX_SUBFRAME_SIZE - 4, 4 * sizeof(s16));
    musyx_mix_fir4(musyx.e50, buffer + 1, fir4_hgain, fir4_hcoeffs);
    for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
        set_dram_s16(cbuffer_ptr + (pos + i) * 2, musyx.e50[i]);
    }
}

void AudioHLE::musyx_load_base_vol(s32* base_vol, u32 address) {
    base_vol[0] = (static_cast<s32>(dram_u16(address))     << 16) | dram_u16(address + 8);
    base_vol[1] = (static_cast<s32>(dram_u16(address + 2)) << 16) | dram_u16(address + 10);
    base_vol[2] = (static_cast<s32>(dram_u16(address + 4)) << 16) | dram_u16(address + 12);
    base_vol[3] = (static_cast<s32>(dram_u16(address + 6)) << 16) | dram_u16(address + 14);
}

void AudioHLE::musyx_save_base_vol(const s32* base_vol, u32 address) {
    for (int k = 0; k < 4; ++k) {
        set_dram_s16(address + k * 2, static_cast<s16>(base_vol[k] >> 16));
    }
    for (int k = 0; k < 4; ++k) {
        set_dram_s16(address + 8 + k * 2, static_cast<s16>(base_vol[k]));
    }
}

void AudioHLE::musyx_update_base_vol(s32* base_vol, u32 voice_mask, u32 last_sample_ptr, u8 mask_15, u32 ptr_24) {
    if (voice_mask != 0) {
        u32 mask = 1;
        for (size_t i = 0; i < MUSYX_MAX_VOICES; ++i, mask <<= 1, last_sample_ptr += 8) {
            if ((voice_mask & mask) == 0) continue;
            for (int k = 0; k < 4; ++k) {
                base_vol[k] += dram_s16(last_sample_ptr + k * 2);
            }
        }
    }
    if (mask_15 != 0) {
        u32 mask = 1;
        for (size_t i = 0; i < 4; ++i, mask <<= 1, ptr_24 += 8) {
            if ((mask_15 & mask) == 0) continue;
            for (int k = 0; k < 4; ++k) {
                base_vol[k] += dram_s16(ptr_24 + k * 2);
            }
        }
    }
    for (int k = 0; k < 4; ++k) {
        base_vol[k] = static_cast<s32>((static_cast<s64>(base_vol[k]) * 0x0000F850) >> 16);
    }
}

void AudioHLE::musyx_init_subframes_v1(MusyXState& musyx) {
    const s16 base_cc0 = clamp_s16(musyx.base_vol[2]);
    const s16 base_e50 = clamp_s16(musyx.base_vol[3]);

    for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
        musyx.e50[i]   = base_e50;
        musyx.left[i]  = clamp_s16(musyx.cc0[i] + base_cc0);
        musyx.right[i] = clamp_s16(-musyx.cc0[i] - base_cc0);
        musyx.cc0[i]   = 0;
    }
}

void AudioHLE::musyx_init_subframes_v2(MusyXState& musyx) {
    s16 values[4];
    for (int k = 0; k < 4; ++k) {
        values[k] = clamp_s16(musyx.base_vol[k]);
    }
    for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
        musyx.left[i]  = values[0];
        musyx.right[i] = values[1];
        musyx.cc0[i]   = values[2];
        musyx.e50[i]   = values[3];
    }
}

void AudioHLE::musyx_interleave_stage_v1(MusyXState& musyx, u32 output_ptr) {
    const s16 base_left  = clamp_s16(musyx.base_vol[0]);
    const s16 base_right = clamp_s16(musyx.base_vol[1]);

    for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
        const s16 l = clamp_s16(musyx.left[i]  + base_left);
        const s16 r = clamp_s16(musyx.right[i] + base_right);
        set_dram_s16(output_ptr + i * 4,     l);
        set_dram_s16(output_ptr + i * 4 + 2, r);
    }
}

void AudioHLE::musyx_interleave_stage_v2(MusyXState& musyx, u16 mask_16, u32 ptr_18, u32 ptr_1c, u32 output_ptr) {
    s16 subframe[MUSYX_SUBFRAME_SIZE];

    for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
        const s16 v = dram_s16(ptr_1c + i * 2);
        musyx.left[i]  = v;
        musyx.right[i] = clamp_s16(-v);
        subframe[i]    = 0;
    }

    u16 mask = 1;
    for (size_t k = 0; k < 8; ++k, mask <<= 1, ptr_18 += 8) {
        if ((mask_16 & mask) == 0) continue;

        u32 address = dram_u32(ptr_18);
        const s16 hgain = dram_s16(ptr_18 + 4);

        for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i, address += 2) {
            musyx_mix_samples(&musyx.left[i],  dram_s16(address), hgain);
            musyx_mix_samples(&musyx.right[i], dram_s16(address + 2 * MUSYX_SUBFRAME_SIZE), hgain);
            musyx_mix_samples(&subframe[i],    dram_s16(address + 4 * MUSYX_SUBFRAME_SIZE), hgain);
        }
    }

    for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
        set_dram_s16(output_ptr + i * 4,     musyx.left[i]);
        set_dram_s16(output_ptr + i * 4 + 2, musyx.right[i]);
    }

    for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
        set_dram_s16(ptr_1c + i * 2, subframe[i]);
    }
}

void AudioHLE::run_musyx_v1(u32 sfd_ptr, u32 sfd_count) {
    MusyXState musyx;
    u32 state_ptr = dram_u32(sfd_ptr + SFD_STATE_PTR);

    musyx_load_base_vol(musyx.base_vol, state_ptr + STATE_BASE_VOL);
    for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
        musyx.cc0[i] = dram_s16(state_ptr + STATE_CC0 + i * 2);
    }
    for (size_t i = 0; i < 4; ++i) {
        musyx.subframe_740_last4[i] = dram_s16(state_ptr + STATE_740_LAST4_V1 + i * 2);
    }

    while (sfd_count > 0) {
        const u16 sfx_index       = dram_u16(sfd_ptr + SFD_SFX_INDEX);
        const u32 voice_mask      = dram_u32(sfd_ptr + SFD_VOICE_BITMASK);
        const u32 sfx_ptr         = dram_u32(sfd_ptr + SFD_SFX_PTR);
        const u32 voice_ptr       = sfd_ptr + SFD_VOICES;
        const u32 last_sample_ptr = state_ptr + STATE_LAST_SAMPLE;

        musyx_update_base_vol(musyx.base_vol, voice_mask, last_sample_ptr, 0, 0);
        musyx_init_subframes_v1(musyx);

        const u32 output_ptr = musyx_voice_stage(musyx, voice_ptr, last_sample_ptr);

        musyx_sfx_stage(musyx, sfx_ptr, sfx_index, false);

        musyx_interleave_stage_v1(musyx, output_ptr);

        --sfd_count;
        if (sfd_count == 0) break;

        sfd_ptr += SFD_VOICES + MUSYX_MAX_VOICES * VOICE_SIZE;
        state_ptr = dram_u32(sfd_ptr + SFD_STATE_PTR);
    }

    musyx_save_base_vol(musyx.base_vol, state_ptr + STATE_BASE_VOL);
    for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
        set_dram_s16(state_ptr + STATE_CC0 + i * 2, musyx.cc0[i]);
    }
    for (size_t i = 0; i < 4; ++i) {
        set_dram_s16(state_ptr + STATE_740_LAST4_V1 + i * 2, musyx.subframe_740_last4[i]);
    }
}

void AudioHLE::run_musyx_v2(u32 sfd_ptr, u32 sfd_count) {
    MusyXState musyx;

    while (sfd_count > 0) {
        const u16 sfx_index  = dram_u16(sfd_ptr + SFD_SFX_INDEX);
        const u32 voice_mask = dram_u32(sfd_ptr + SFD_VOICE_BITMASK);
        const u32 state_ptr  = dram_u32(sfd_ptr + SFD_STATE_PTR);
        const u32 sfx_ptr    = dram_u32(sfd_ptr + SFD_SFX_PTR);
        const u32 voice_ptr  = sfd_ptr + SFD2_VOICES;

        const u8  mask_15    = dram_u8(sfd_ptr + SFD2_15_BITMASK);
        const u16 mask_16    = dram_u16(sfd_ptr + SFD2_16_BITMASK);
        const u32 ptr_18     = dram_u32(sfd_ptr + SFD2_18_PTR);
        const u32 ptr_1c     = dram_u32(sfd_ptr + SFD2_1C_PTR);
        const u32 ptr_20     = dram_u32(sfd_ptr + SFD2_20_PTR);
        const u32 ptr_24     = dram_u32(sfd_ptr + SFD2_24_PTR);

        const u32 last_sample_ptr = state_ptr + STATE_LAST_SAMPLE;

        musyx_load_base_vol(musyx.base_vol, state_ptr + STATE_BASE_VOL);
        for (size_t i = 0; i < 4; ++i) {
            musyx.subframe_740_last4[i] = dram_s16(state_ptr + STATE_740_LAST4_V2 + i * 2);
        }

        musyx_update_base_vol(musyx.base_vol, voice_mask, last_sample_ptr, mask_15, ptr_24);
        musyx_init_subframes_v2(musyx);

        const u32 output_ptr = musyx_voice_stage(musyx, voice_ptr, last_sample_ptr);

        musyx_sfx_stage(musyx, sfx_ptr, sfx_index, true);

        for (size_t i = 0; i < MUSYX_SUBFRAME_SIZE; ++i) {
            set_dram_s16(output_ptr + i * 2,                            musyx.left[i]);
            set_dram_s16(output_ptr + 2 * MUSYX_SUBFRAME_SIZE + i * 2, musyx.right[i]);
            set_dram_s16(output_ptr + 4 * MUSYX_SUBFRAME_SIZE + i * 2, musyx.cc0[i]);
        }

        musyx_save_base_vol(musyx.base_vol, state_ptr + STATE_BASE_VOL);
        for (size_t i = 0; i < 4; ++i) {
            set_dram_s16(state_ptr + STATE_740_LAST4_V2 + i * 2, musyx.subframe_740_last4[i]);
        }

        if (mask_16 != 0) {
            musyx_interleave_stage_v2(musyx, mask_16, ptr_18, ptr_1c, ptr_20);
        }

        --sfd_count;
        if (sfd_count == 0) break;

        sfd_ptr += SFD2_VOICES + MUSYX_MAX_VOICES * VOICE_SIZE;
    }
}
