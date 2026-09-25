#include "ahle.hpp"
#include <algorithm>
#include <cstring>
#include <vector>

namespace {
    constexpr u8 A_INIT = 0x01;
    constexpr u8 A_LOOP = 0x02;
    constexpr u8 A_LEFT = 0x02;
    constexpr u8 A_VOL  = 0x04;
    constexpr u8 A_AUX  = 0x08;

    inline s16 clamp_s16(s32 v) {
        if (v < -32768) return -32768;
        if (v > 32767) return 32767;
        return static_cast<s16>(v);
    }

    inline u16 align16(u16 v) { return (v + 15) & ~static_cast<u16>(15); }

    // Reconstructs a single ADPCM residual nibble the way the hardware's
    // predictor expects it: sign-extend within `lshift` bits, then rescale
    // by the per-frame quantization shift.
    inline s16 adpcm_nibble(u8 byte, u8 mask, unsigned lshift, unsigned rshift) {
        s16 v = static_cast<s16>(static_cast<u16>(byte & mask) << lshift);
        v = static_cast<s16>(v >> rshift);
        return v;
    }

    // sum_{k=0}^{n-1} x[k] * y[n-1-k]
    inline s32 reverse_dot(size_t n, const s16* x, const s16* y) {
        s32 acc = 0;
        for (size_t k = 0; k < n; ++k) acc += x[k] * y[n - 1 - k];
        return acc;
    }

    // 4-tap resampling filter coefficients (64 phases x 4 taps), transcribed
    // from the RSP audio microcode's own data segment (audio.s, DMEM offset
    // 0xC0). RESAMPLE indexes this with the top 6 bits of the fractional
    // pitch accumulator to pick a phase, matching real hardware's output
    // instead of a plain linear interpolation.
    constexpr s16 RESAMPLE_LUT[64 * 4] = {
        static_cast<s16>(0x0c39), static_cast<s16>(0x66ad), static_cast<s16>(0x0d46), static_cast<s16>(0xffdf),
        static_cast<s16>(0x0b39), static_cast<s16>(0x6696), static_cast<s16>(0x0e5f), static_cast<s16>(0xffd8),
        static_cast<s16>(0x0a44), static_cast<s16>(0x6669), static_cast<s16>(0x0f83), static_cast<s16>(0xffd0),
        static_cast<s16>(0x095a), static_cast<s16>(0x6626), static_cast<s16>(0x10b4), static_cast<s16>(0xffc8),
        static_cast<s16>(0x087d), static_cast<s16>(0x65cd), static_cast<s16>(0x11f0), static_cast<s16>(0xffbf),
        static_cast<s16>(0x07ab), static_cast<s16>(0x655e), static_cast<s16>(0x1338), static_cast<s16>(0xffb6),
        static_cast<s16>(0x06e4), static_cast<s16>(0x64d9), static_cast<s16>(0x148c), static_cast<s16>(0xffac),
        static_cast<s16>(0x0628), static_cast<s16>(0x643f), static_cast<s16>(0x15eb), static_cast<s16>(0xffa1),
        static_cast<s16>(0x0577), static_cast<s16>(0x638f), static_cast<s16>(0x1756), static_cast<s16>(0xff96),
        static_cast<s16>(0x04d1), static_cast<s16>(0x62cb), static_cast<s16>(0x18cb), static_cast<s16>(0xff8a),
        static_cast<s16>(0x0435), static_cast<s16>(0x61f3), static_cast<s16>(0x1a4c), static_cast<s16>(0xff7e),
        static_cast<s16>(0x03a4), static_cast<s16>(0x6106), static_cast<s16>(0x1bd7), static_cast<s16>(0xff71),
        static_cast<s16>(0x031c), static_cast<s16>(0x6007), static_cast<s16>(0x1d6c), static_cast<s16>(0xff64),
        static_cast<s16>(0x029f), static_cast<s16>(0x5ef5), static_cast<s16>(0x1f0b), static_cast<s16>(0xff56),
        static_cast<s16>(0x022a), static_cast<s16>(0x5dd0), static_cast<s16>(0x20b3), static_cast<s16>(0xff48),
        static_cast<s16>(0x01be), static_cast<s16>(0x5c9a), static_cast<s16>(0x2264), static_cast<s16>(0xff3a),
        static_cast<s16>(0x015b), static_cast<s16>(0x5b53), static_cast<s16>(0x241e), static_cast<s16>(0xff2c),
        static_cast<s16>(0x0101), static_cast<s16>(0x59fc), static_cast<s16>(0x25e0), static_cast<s16>(0xff1e),
        static_cast<s16>(0x00ae), static_cast<s16>(0x5896), static_cast<s16>(0x27a9), static_cast<s16>(0xff10),
        static_cast<s16>(0x0063), static_cast<s16>(0x5720), static_cast<s16>(0x297a), static_cast<s16>(0xff02),
        static_cast<s16>(0x001f), static_cast<s16>(0x559d), static_cast<s16>(0x2b50), static_cast<s16>(0xfef4),
        static_cast<s16>(0xffe2), static_cast<s16>(0x540d), static_cast<s16>(0x2d2c), static_cast<s16>(0xfee8),
        static_cast<s16>(0xffac), static_cast<s16>(0x5270), static_cast<s16>(0x2f0d), static_cast<s16>(0xfedb),
        static_cast<s16>(0xff7c), static_cast<s16>(0x50c7), static_cast<s16>(0x30f3), static_cast<s16>(0xfed0),
        static_cast<s16>(0xff53), static_cast<s16>(0x4f14), static_cast<s16>(0x32dc), static_cast<s16>(0xfec6),
        static_cast<s16>(0xff2e), static_cast<s16>(0x4d57), static_cast<s16>(0x34c8), static_cast<s16>(0xfebd),
        static_cast<s16>(0xff0f), static_cast<s16>(0x4b91), static_cast<s16>(0x36b6), static_cast<s16>(0xfeb6),
        static_cast<s16>(0xfef5), static_cast<s16>(0x49c2), static_cast<s16>(0x38a5), static_cast<s16>(0xfeb0),
        static_cast<s16>(0xfedf), static_cast<s16>(0x47ed), static_cast<s16>(0x3a95), static_cast<s16>(0xfeac),
        static_cast<s16>(0xfece), static_cast<s16>(0x4611), static_cast<s16>(0x3c85), static_cast<s16>(0xfeab),
        static_cast<s16>(0xfec0), static_cast<s16>(0x4430), static_cast<s16>(0x3e74), static_cast<s16>(0xfeac),
        static_cast<s16>(0xfeb6), static_cast<s16>(0x424a), static_cast<s16>(0x4060), static_cast<s16>(0xfeaf),
        static_cast<s16>(0xfeaf), static_cast<s16>(0x4060), static_cast<s16>(0x424a), static_cast<s16>(0xfeb6),
        static_cast<s16>(0xfeac), static_cast<s16>(0x3e74), static_cast<s16>(0x4430), static_cast<s16>(0xfec0),
        static_cast<s16>(0xfeab), static_cast<s16>(0x3c85), static_cast<s16>(0x4611), static_cast<s16>(0xfece),
        static_cast<s16>(0xfeac), static_cast<s16>(0x3a95), static_cast<s16>(0x47ed), static_cast<s16>(0xfedf),
        static_cast<s16>(0xfeb0), static_cast<s16>(0x38a5), static_cast<s16>(0x49c2), static_cast<s16>(0xfef5),
        static_cast<s16>(0xfeb6), static_cast<s16>(0x36b6), static_cast<s16>(0x4b91), static_cast<s16>(0xff0f),
        static_cast<s16>(0xfebd), static_cast<s16>(0x34c8), static_cast<s16>(0x4d57), static_cast<s16>(0xff2e),
        static_cast<s16>(0xfec6), static_cast<s16>(0x32dc), static_cast<s16>(0x4f14), static_cast<s16>(0xff53),
        static_cast<s16>(0xfed0), static_cast<s16>(0x30f3), static_cast<s16>(0x50c7), static_cast<s16>(0xff7c),
        static_cast<s16>(0xfedb), static_cast<s16>(0x2f0d), static_cast<s16>(0x5270), static_cast<s16>(0xffac),
        static_cast<s16>(0xfee8), static_cast<s16>(0x2d2c), static_cast<s16>(0x540d), static_cast<s16>(0xffe2),
        static_cast<s16>(0xfef4), static_cast<s16>(0x2b50), static_cast<s16>(0x559d), static_cast<s16>(0x001f),
        static_cast<s16>(0xff02), static_cast<s16>(0x297a), static_cast<s16>(0x5720), static_cast<s16>(0x0063),
        static_cast<s16>(0xff10), static_cast<s16>(0x27a9), static_cast<s16>(0x5896), static_cast<s16>(0x00ae),
        static_cast<s16>(0xff1e), static_cast<s16>(0x25e0), static_cast<s16>(0x59fc), static_cast<s16>(0x0101),
        static_cast<s16>(0xff2c), static_cast<s16>(0x241e), static_cast<s16>(0x5b53), static_cast<s16>(0x015b),
        static_cast<s16>(0xff3a), static_cast<s16>(0x2264), static_cast<s16>(0x5c9a), static_cast<s16>(0x01be),
        static_cast<s16>(0xff48), static_cast<s16>(0x20b3), static_cast<s16>(0x5dd0), static_cast<s16>(0x022a),
        static_cast<s16>(0xff56), static_cast<s16>(0x1f0b), static_cast<s16>(0x5ef5), static_cast<s16>(0x029f),
        static_cast<s16>(0xff64), static_cast<s16>(0x1d6c), static_cast<s16>(0x6007), static_cast<s16>(0x031c),
        static_cast<s16>(0xff71), static_cast<s16>(0x1bd7), static_cast<s16>(0x6106), static_cast<s16>(0x03a4),
        static_cast<s16>(0xff7e), static_cast<s16>(0x1a4c), static_cast<s16>(0x61f3), static_cast<s16>(0x0435),
        static_cast<s16>(0xff8a), static_cast<s16>(0x18cb), static_cast<s16>(0x62cb), static_cast<s16>(0x04d1),
        static_cast<s16>(0xff96), static_cast<s16>(0x1756), static_cast<s16>(0x638f), static_cast<s16>(0x0577),
        static_cast<s16>(0xffa1), static_cast<s16>(0x15eb), static_cast<s16>(0x643f), static_cast<s16>(0x0628),
        static_cast<s16>(0xffac), static_cast<s16>(0x148c), static_cast<s16>(0x64d9), static_cast<s16>(0x06e4),
        static_cast<s16>(0xffb6), static_cast<s16>(0x1338), static_cast<s16>(0x655e), static_cast<s16>(0x07ab),
        static_cast<s16>(0xffbf), static_cast<s16>(0x11f0), static_cast<s16>(0x65cd), static_cast<s16>(0x087d),
        static_cast<s16>(0xffc8), static_cast<s16>(0x10b4), static_cast<s16>(0x6626), static_cast<s16>(0x095a),
        static_cast<s16>(0xffd0), static_cast<s16>(0x0f83), static_cast<s16>(0x6669), static_cast<s16>(0x0a44),
        static_cast<s16>(0xffd8), static_cast<s16>(0x0e5f), static_cast<s16>(0x6696), static_cast<s16>(0x0b39),
        static_cast<s16>(0xffdf), static_cast<s16>(0x0d46), static_cast<s16>(0x66ad), static_cast<s16>(0x0c39),
    };
}

void AudioHLE::reset() {
    scratch.fill(0);
    segments.fill(0);
    codebook.fill(0);
    buf_in = buf_out = buf_count = 0;
    buf_dry_right = buf_wet_left = buf_wet_right = 0;
    vol[0] = vol[1] = target[0] = target[1] = 0;
    rate[0] = rate[1] = 0;
    dry = wet = 0;
    loop_addr = 0;
    adpcm_state.clear();
    envmix_state.clear();
    resample_state.clear();
    polef_state.clear();
    current_abi = AudioABI::ABI1;
    envsetup2 = {};
    nead_env_values[0] = nead_env_values[1] = nead_env_values[2] = 0;
    nead_env_steps[0] = nead_env_steps[1] = nead_env_steps[2] = 0;
    nead_filter_count = 0;
    nead_filter_lut[0] = nead_filter_lut[1] = 0;
}

u8 AudioHLE::dram_u8(u32 addr) const {
    if (!rdram_ptr || rdram_sz == 0) return 0;
    return rdram_ptr[addr & (rdram_sz - 1)];
}
void AudioHLE::set_dram_u8(u32 addr, u8 v) {
    if (!rdram_ptr || rdram_sz == 0) return;
    rdram_ptr[addr & (rdram_sz - 1)] = v;
}
s16 AudioHLE::dram_s16(u32 addr) const {
    return static_cast<s16>((dram_u8(addr) << 8) | dram_u8(addr + 1));
}
void AudioHLE::set_dram_s16(u32 addr, s16 v) {
    set_dram_u8(addr, static_cast<u8>((static_cast<u16>(v) >> 8) & 0xFF));
    set_dram_u8(addr + 1, static_cast<u8>(static_cast<u16>(v) & 0xFF));
}
u32 AudioHLE::dram_u32(u32 addr) const {
    return (static_cast<u32>(dram_u8(addr)) << 24) | (static_cast<u32>(dram_u8(addr + 1)) << 16) |
           (static_cast<u32>(dram_u8(addr + 2)) << 8) | static_cast<u32>(dram_u8(addr + 3));
}

u32 AudioHLE::resolve_address(u32 so) const {
    if (current_abi == AudioABI::ABI2) {
        return so & 0x00FFFFFF;
    }
    u8 seg = (so >> 24) & 0xFF;
    u32 off = so & 0x00FFFFFF;
    return (seg >= segments.size()) ? off : segments[seg] + off;
}

void AudioHLE::process(u8* rdram, size_t rdram_size, u32 addr, u32 size, u32 ucode_data_ptr) {
    if (!rdram || rdram_size == 0 || size == 0) return;
    task_count++;
    rdram_ptr = rdram;
    rdram_sz = rdram_size;
    segments.fill(0);

    u32 n_commands = size >> 3; // each command is 8 bytes: w1, w2

    current_abi = AudioABI::ABI1;
    if (ucode_data_ptr != 0 && ucode_data_ptr + 0x40 <= rdram_size) {
        u32 u0 = dram_u32(ucode_data_ptr);
        if (u0 == 0x00000001) {
            if (dram_u32(ucode_data_ptr + 0x30) == 0xf0000f00) {
                current_abi = AudioABI::ABI1; // Super Mario 64, Wave Race 64, GoldenEye, DKR
            } else {
                u32 v = dram_u32(ucode_data_ptr + 0x10);
                switch (v) {
                    case 0x11181350: current_abi = AudioABI::NEAD_MK; break; // Mario Kart 64, Wave Race PAL
                    case 0x111812e0: // Star Fox 64 (J)
                    case 0x110412ac: // Wave Race 64 (J RevB)
                    case 0x110412cc: current_abi = AudioABI::NEAD_SF; break; // Star Fox / Lylat Wars
                    case 0x1cd01250: current_abi = AudioABI::NEAD_SF; break; // F-Zero X
                    case 0x1f08122c: // Yoshi's Story
                    case 0x1f38122c: // 1080 Snowboarding
                    case 0x1f681230: current_abi = AudioABI::NEAD_OOT; break; // Zelda OoT
                    case 0x1f801250: // Zelda MM
                    case 0x109411f8: // Zelda MM E Beta
                    case 0x1eac11b8: current_abi = AudioABI::NEAD_OOT; break; // Animal Crossing
                    default:
                        current_abi = AudioABI::NEAD_MK;
                        break;
                }
            }
        } else {
            u32 v = dram_u32(ucode_data_ptr + 0x10);
            switch (v) {
                case 0x0000127c:
                case 0x00001280:
                case 0x1c58126c:
                case 0x1ae8143c:
                case 0x1ab0140c:
                    current_abi = AudioABI::ABI2; // N_AUDIO
                    break;
                default:
                    current_abi = AudioABI::ABI2;
                    break;
            }
        }
    } else {
        // Fallback heuristic: check command list for distinguishing opcodes
        for (u32 i = 0; i < n_commands; ++i) {
            u32 w1 = dram_u32(addr + i * 8);
            u32 acmd = (w1 >> 24) & 0xFF;
            if (acmd == 0x14 || acmd == 0x15) {
                current_abi = AudioABI::NEAD_MK;
                break;
            }
            if (acmd == 0x10 || acmd == 0x12 || acmd == 0x13 || acmd == 0x16) {
                current_abi = AudioABI::NEAD_MK;
                break;
            }
        }
    }

    u32 pos = addr;
    for (u32 i = 0; i < n_commands; ++i) {
        u32 w1 = dram_u32(pos);
        u32 w2 = dram_u32(pos + 4);
        pos += 8;
        u32 acmd = (w1 >> 24) & 0x7F;
        switch (current_abi) {
            case AudioABI::ABI1: dispatch_abi1(acmd, w1, w2); break;
            case AudioABI::ABI2: dispatch_abi2(acmd, w1, w2); break;
            case AudioABI::NEAD_MK: dispatch_nead_mk(acmd, w1, w2); break;
            case AudioABI::NEAD_SF: dispatch_nead_sf(acmd, w1, w2); break;
            case AudioABI::NEAD_OOT: dispatch_nead_oot(acmd, w1, w2); break;
        }
    }
}

void AudioHLE::dispatch_abi1(u32 acmd, u32 w1, u32 w2) {
    switch (acmd) {
        case 0x0: break; // SPNOOP
        case 0x1: cmd_adpcm(w1, w2); break;
        case 0x2: cmd_clearbuff(w1, w2); break;
        case 0x3: cmd_envmixer(w1, w2); break;
        case 0x4: cmd_loadbuff(w1, w2); break;
        case 0x5: cmd_resample(w1, w2); break;
        case 0x6: cmd_savebuff(w1, w2); break;
        case 0x7: cmd_segment(w1, w2); break;
        case 0x8: cmd_setbuff(w1, w2); break;
        case 0x9: cmd_setvol(w1, w2); break;
        case 0xA: cmd_dmemmove(w1, w2); break;
        case 0xB: cmd_loadadpcm(w1, w2); break;
        case 0xC: cmd_mixer(w1, w2); break;
        case 0xD: cmd_interleave(w1, w2); break;
        case 0xE: cmd_polef(w1, w2); break;
        case 0xF: cmd_setloop(w1, w2); break;
        default: break;
    }
}

void AudioHLE::dispatch_abi2(u32 acmd, u32 w1, u32 w2) {
    switch (acmd) {
        case 0x0: break; // SPNOOP
        case 0x1: cmd_adpcm_abi2(w1, w2); break;
        case 0x2: cmd_clearbuff(w1, w2); break;
        case 0x4: cmd_addmixer_abi2(w1, w2); break;
        case 0x5: cmd_resample_abi2(w1, w2); break;
        case 0x6: cmd_resample_zoh_abi2(w1, w2); break;
        case 0x8: cmd_setbuff(w1, w2); break;
        case 0xA: cmd_dmemmove(w1, w2); break;
        case 0xB: cmd_loadadpcm(w1, w2); break;
        case 0xC: cmd_mixer_abi2(w1, w2); break;
        case 0xD: cmd_interleave_abi2(w1, w2); break;
        case 0xF: cmd_setloop(w1, w2); break;
        case 0x10: cmd_dmemmove2_abi2(w1, w2); break;
        case 0x12: cmd_envsetup1_abi2(w1, w2); break;
        case 0x13: cmd_envmixer_abi2(w1, w2); break;
        case 0x14: cmd_loadbuff_abi2(w1, w2); break;
        case 0x15: cmd_savebuff_abi2(w1, w2); break;
        case 0x16: cmd_envsetup2_abi2(w1, w2); break;
        default: break;
    }
}

void AudioHLE::cmd_segment(u32 /*w1*/, u32 w2) {
    u8 seg = (w2 >> 24) & 0xFF;
    u32 off = w2 & 0x00FFFFFF;
    if (seg < segments.size()) segments[seg] = off;
}

void AudioHLE::cmd_setloop(u32 /*w1*/, u32 w2) {
    loop_addr = resolve_address(w2);
}

void AudioHLE::cmd_setbuff(u32 w1, u32 w2) {
    u8 flags = (w1 >> 16) & 0xFF;
    if (flags & A_AUX) {
        buf_dry_right = static_cast<u16>(w1 & 0xFFFF) + DMEM_BASE;
        buf_wet_left = static_cast<u16>((w2 >> 16) & 0xFFFF) + DMEM_BASE;
        buf_wet_right = static_cast<u16>(w2 & 0xFFFF) + DMEM_BASE;
    } else {
        buf_in = static_cast<u16>(w1 & 0xFFFF) + DMEM_BASE;
        buf_out = static_cast<u16>((w2 >> 16) & 0xFFFF) + DMEM_BASE;
        buf_count = static_cast<u16>(w2 & 0xFFFF);
    }
}

void AudioHLE::cmd_setvol(u32 w1, u32 w2) {
    u8 flags = (w1 >> 16) & 0xFF;
    if (flags & A_AUX) {
        dry = static_cast<s16>(w1 & 0xFFFF);
        wet = static_cast<s16>(w2 & 0xFFFF);
    } else {
        unsigned lr = (flags & A_LEFT) ? 0 : 1;
        if (flags & A_VOL) {
            vol[lr] = static_cast<s16>(w1 & 0xFFFF);
        } else {
            target[lr] = static_cast<s16>(w1 & 0xFFFF);
            rate[lr] = static_cast<s32>(w2);
        }
    }
}

void AudioHLE::cmd_clearbuff(u32 w1, u32 w2) {
    u16 dmem = static_cast<u16>(w1 & 0xFFFF) + DMEM_BASE;
    u16 count = align16(static_cast<u16>(w2 & 0xFFFF));
    for (u16 i = 0; i < count; ++i) set_scratch_u8(dmem + i, 0);
}

void AudioHLE::cmd_loadbuff(u32 /*w1*/, u32 w2) {
    if (buf_count == 0) return;
    u32 address = resolve_address(w2) & ~7u;
    u16 dmem = buf_in;
    u16 count = align16(buf_count);
    for (u16 i = 0; i < count; ++i) set_scratch_u8(dmem + i, dram_u8(address + i));
}

void AudioHLE::cmd_savebuff(u32 /*w1*/, u32 w2) {
    if (buf_count == 0) return;
    u32 address = resolve_address(w2) & ~7u;
    u16 dmem = buf_out;
    u16 count = align16(buf_count);
    for (u16 i = 0; i < count; ++i) set_dram_u8(address + i, scratch_u8(dmem + i));
}

void AudioHLE::cmd_dmemmove(u32 w1, u32 w2) {
    u16 dmemi = static_cast<u16>(w1 & 0xFFFF) + DMEM_BASE;
    u16 dmemo = static_cast<u16>((w2 >> 16) & 0xFFFF) + DMEM_BASE;
    u16 count = align16(static_cast<u16>(w2 & 0xFFFF));
    if (count == 0) return;
    std::vector<u8> tmp(count);
    for (u16 i = 0; i < count; ++i) tmp[i] = scratch_u8(dmemi + i);
    for (u16 i = 0; i < count; ++i) set_scratch_u8(dmemo + i, tmp[i]);
}

void AudioHLE::cmd_loadadpcm(u32 w1, u32 w2) {
    u16 count = static_cast<u16>(w1 & 0xFFFF);
    u32 address = resolve_address(w2);
    u16 n_entries = align16(count) >> 1;
    n_entries = std::min<u16>(n_entries, static_cast<u16>(codebook.size()));
    for (u16 i = 0; i < n_entries; ++i) codebook[i] = dram_s16(address + i * 2u);
}

void AudioHLE::cmd_mixer(u32 w1, u32 w2) {
    if (buf_count == 0) return;
    s16 gain = static_cast<s16>(w1 & 0xFFFF);
    u16 dmemi = static_cast<u16>((w2 >> 16) & 0xFFFF) + DMEM_BASE;
    u16 dmemo = static_cast<u16>(w2 & 0xFFFF) + DMEM_BASE;
    u16 n = align16(buf_count) >> 1;
    for (u16 k = 0; k < n; ++k) {
        s16 src = scratch_s16(dmemi + k * 2u);
        s16 dst = scratch_s16(dmemo + k * 2u);
        dst = clamp_s16(dst + ((static_cast<s32>(src) * gain) >> 15));
        set_scratch_s16(dmemo + k * 2u, dst);
    }
}

void AudioHLE::cmd_interleave(u32 /*w1*/, u32 w2) {
    if (buf_count == 0) return;
    u16 left = static_cast<u16>((w2 >> 16) & 0xFFFF) + DMEM_BASE;
    u16 right = static_cast<u16>(w2 & 0xFFFF) + DMEM_BASE;
    // buf_count is the per-channel byte count set via aSetBuffer(..., outCount << 1)
    // so the number of stereo sample pairs to interleave is buf_count / 2!
    u16 n = align16(buf_count) >> 1;
    for (u16 k = 0; k < n; ++k) {
        s16 l = scratch_s16(left + k * 2u);
        s16 r = scratch_s16(right + k * 2u);
        set_scratch_s16(buf_out + k * 4u, l);
        set_scratch_s16(buf_out + k * 4u + 2u, r);
    }
}

void AudioHLE::cmd_adpcm(u32 w1, u32 w2) {
    u8 flags = (w1 >> 16) & 0xFF;
    bool init = flags & A_INIT;
    bool loop = flags & A_LOOP;
    u32 address = resolve_address(w2);
    u16 count = align16(buf_count) & ~31u;
    if (count == 0 && buf_count != 0) count = 32; // at least one block

    s16 hist[16];
    if (init) {
        std::memset(hist, 0, sizeof(hist));
        // On init, lead-in 16 samples are zeroed
        for (int i = 0; i < 16; ++i) set_scratch_s16(buf_out + i * 2u, 0);
    } else {
        u32 src = loop ? loop_addr : address;
        // Read 16 samples directly from RDRAM continuation state / loop state
        bool found_rdram = false;
        if (src != 0 && src + 32 <= rdram_sz) {
            for (int i = 0; i < 16; ++i) {
                hist[i] = dram_s16(src + i * 2u);
                if (hist[i] != 0) found_rdram = true;
            }
        }
        if (!found_rdram && !loop) {
            auto it = adpcm_state.find(src);
            if (it != adpcm_state.end()) {
                std::memcpy(hist, it->second.data(), sizeof(hist));
            } else {
                std::memset(hist, 0, sizeof(hist));
            }
        }
        // Real hardware loads the 16 history samples into buf_out[0..15] as continuation lead-in!
        for (int i = 0; i < 16; ++i) set_scratch_s16(buf_out + i * 2u, hist[i]);
    }

    u16 dmemi = buf_in;
    u16 dmemo = buf_out + 32;
    u16 remaining = count;
    while (remaining != 0) {
        u8 control = scratch_u8(dmemi++);
        unsigned scale = (control >> 4) & 0xF;
        unsigned predictor = control & 0xF;
        const s16* cb = codebook.data() + (predictor << 4);
        if (predictor >= 16) cb = codebook.data();

        s16 residual[16];
        unsigned rshift = (scale < 12) ? (12 - scale) : 0;
        for (int i = 0; i < 8; ++i) {
            u8 byte = scratch_u8(dmemi++);
            residual[2 * i] = adpcm_nibble(byte, 0xF0, 8, rshift);
            residual[2 * i + 1] = adpcm_nibble(byte, 0x0F, 12, rshift);
        }

        const s16* book1 = cb;
        const s16* book2 = cb + 8;

        s16 out0[8];
        for (int i = 0; i < 8; ++i) {
            s32 acc = static_cast<s32>(residual[i]) << 11;
            acc += book1[i] * hist[14] + book2[i] * hist[15] + reverse_dot(i, book2, residual);
            out0[i] = clamp_s16(acc >> 11);
        }
        for (int i = 0; i < 8; ++i) hist[i] = out0[i];

        s16 out1[8];
        for (int i = 0; i < 8; ++i) {
            s32 acc = static_cast<s32>(residual[8 + i]) << 11;
            acc += book1[i] * hist[6] + book2[i] * hist[7] + reverse_dot(i, book2, residual + 8);
            out1[i] = clamp_s16(acc >> 11);
        }
        for (int i = 0; i < 8; ++i) hist[8 + i] = out1[i];

        for (int i = 0; i < 16; ++i) {
            set_scratch_s16(dmemo, hist[i]);
            dmemo += 2;
        }

        remaining -= 32;
    }

    // Save final 16 samples to local cache (do not write to RDRAM to prevent corrupting non-ABI1 games)
    std::array<s16, 16> saved;
    std::memcpy(saved.data(), hist, sizeof(hist));
    adpcm_state[address] = saved;
}

namespace {
    struct ExpRamp {
        s64 value{0};
        s64 step{0};
        s64 target{0};
        s32 exp_rate{0};
        s32 exp_seq{0};

        void maybe_recompute() {
            if (step == 0) return;
            if (exp_rate != 0) {
                exp_seq = static_cast<s32>((static_cast<s64>(exp_seq) * exp_rate) >> 16);
                step = (static_cast<s64>(exp_seq) - value) >> 3;
            }
        }

        s16 tick() {
            value += step;
            bool reached = (step <= 0) ? (value <= target) : (value >= target);
            if (reached) { value = target; step = 0; }
            return static_cast<s16>(value >> 16);
        }
    };
}

void AudioHLE::cmd_envmixer(u32 w1, u32 w2) {
    u8 flags = (w1 >> 16) & 0xFF;
    bool init = flags & A_INIT;
    bool aux = flags & A_AUX;
    u32 address = resolve_address(w2);

    ExpRamp rampL, rampR;
    s16 cur_dry = dry, cur_wet = wet;

    if (init) {
        rampL.value = static_cast<s64>(vol[0]) << 16;
        rampL.target = static_cast<s64>(target[0]) << 16;
        rampL.exp_rate = rate[0];
        rampL.exp_seq = (rate[0] != 0) ? static_cast<s32>((static_cast<s64>(vol[0]) * rate[0])) : static_cast<s32>(rampL.value);

        rampR.value = static_cast<s64>(vol[1]) << 16;
        rampR.target = static_cast<s64>(target[1]) << 16;
        rampR.exp_rate = rate[1];
        rampR.exp_seq = (rate[1] != 0) ? static_cast<s32>((static_cast<s64>(vol[1]) * rate[1])) : static_cast<s32>(rampR.value);
    } else {
        auto it = envmix_state.find(address);
        if (it != envmix_state.end()) {
            const EnvmixState& s = it->second;
            cur_wet = s.wet;
            cur_dry = s.dry;
            rampL.target = s.target[0];
            rampR.target = s.target[1];
            rampL.value = s.value[0];
            rampR.value = s.value[1];
            rampL.exp_rate = s.exp_rate[0];
            rampR.exp_rate = s.exp_rate[1];
            rampL.exp_seq = s.exp_seq[0];
            rampR.exp_seq = s.exp_seq[1];
        }
    }

    rampL.step = (rampL.target != rampL.value && rampL.exp_rate != 0) ? (rampL.target - rampL.value) : 0;
    rampR.step = (rampR.target != rampR.value && rampR.exp_rate != 0) ? (rampR.target - rampR.value) : 0;

    u16 n = align16(buf_count) >> 1;
    for (u16 k = 0; k < n; ++k) {
        if (k % 8 == 0) {
            rampL.maybe_recompute();
            rampR.maybe_recompute();
        }
        s16 lvol = rampL.tick();
        s16 rvol = rampR.tick();
        s16 gain_dl = clamp_s16((static_cast<s32>(lvol) * cur_dry + 0x4000) >> 15);
        s16 gain_dr = clamp_s16((static_cast<s32>(rvol) * cur_dry + 0x4000) >> 15);
        s16 src = scratch_s16(buf_in + k * 2u);

        s16 dl = scratch_s16(buf_out + k * 2u);
        set_scratch_s16(buf_out + k * 2u, clamp_s16(dl + ((static_cast<s32>(src) * gain_dl) >> 15)));
        s16 dr = scratch_s16(buf_dry_right + k * 2u);
        set_scratch_s16(buf_dry_right + k * 2u, clamp_s16(dr + ((static_cast<s32>(src) * gain_dr) >> 15)));

        if (aux) {
            s16 gain_wl = clamp_s16((static_cast<s32>(lvol) * cur_wet + 0x4000) >> 15);
            s16 gain_wr = clamp_s16((static_cast<s32>(rvol) * cur_wet + 0x4000) >> 15);
            s16 wl = scratch_s16(buf_wet_left + k * 2u);
            set_scratch_s16(buf_wet_left + k * 2u, clamp_s16(wl + ((static_cast<s32>(src) * gain_wl) >> 15)));
            s16 wr = scratch_s16(buf_wet_right + k * 2u);
            set_scratch_s16(buf_wet_right + k * 2u, clamp_s16(wr + ((static_cast<s32>(src) * gain_wr) >> 15)));
        }
    }

    EnvmixState saved;
    saved.wet = cur_wet;
    saved.dry = cur_dry;
    saved.target[0] = rampL.target; saved.target[1] = rampR.target;
    saved.value[0] = rampL.value; saved.value[1] = rampR.value;
    saved.exp_rate[0] = rampL.exp_rate; saved.exp_rate[1] = rampR.exp_rate;
    saved.exp_seq[0] = rampL.exp_seq; saved.exp_seq[1] = rampR.exp_seq;
    envmix_state[address] = saved;
}

void AudioHLE::cmd_resample(u32 w1, u32 w2) {
    u8 flags = (w1 >> 16) & 0xFF;
    bool init = flags & A_INIT;
    u32 pitch = (w1 & 0xFFFF) << 1;
    u32 address = resolve_address(w2);

    s16 hist[4] = {0, 0, 0, 0};
    u32 frac = 0;
    if (!init) {
        bool found_rdram = false;
        if (address != 0 && address + 10 <= rdram_sz) {
            for (int k = 0; k < 4; ++k) {
                hist[k] = dram_s16(address + k * 2u);
                if (hist[k] != 0) found_rdram = true;
            }
            frac = static_cast<u16>(dram_s16(address + 8));
        }
        if (!found_rdram) {
            auto it = resample_state.find(address);
            if (it != resample_state.end()) {
                std::memcpy(hist, it->second.hist, sizeof(hist));
                frac = it->second.frac;
            }
        }
    }

    auto sample_at = [&](s32 rel) -> s32 {
        if (rel < 0) return (rel >= -4) ? hist[4 + rel] : 0;
        return scratch_s16(buf_in + static_cast<u16>(rel) * 2u);
    };

    u16 n_samples = align16(buf_count) >> 1;
    s32 ipos = 0;
    for (u16 i = 0; i < n_samples; ++i) {
        // Top 6 bits of the fractional accumulator select one of 64 filter phases.
        // Peak interpolation coefficient is Tap 1 (sample ipos).
        // Tap 0 is sample ipos - 1 (lookback hist), Tap 2 is ipos + 1, Tap 3 is ipos + 2.
        const s16* lut = RESAMPLE_LUT + ((frac & 0xFC00) >> 8);
        s32 acc = sample_at(ipos - 1) * lut[0] +
                  sample_at(ipos)     * lut[1] +
                  sample_at(ipos + 1) * lut[2] +
                  sample_at(ipos + 2) * lut[3];
        set_scratch_s16(buf_out + i * 2u, clamp_s16(acc >> 15));

        frac += pitch;
        ipos += static_cast<s32>(frac >> 16);
        frac &= 0xFFFF;
    }

    ResampleState saved;
    for (int k = 0; k < 4; ++k) {
        saved.hist[k] = static_cast<s16>(sample_at(ipos - 4 + k));
    }
    saved.frac = frac;
    resample_state[address] = saved;
}

void AudioHLE::cmd_polef(u32 w1, u32 w2) {
    if (buf_count == 0) return;
    u8 flags = (w1 >> 16) & 0xFF;
    bool init = flags & A_INIT;
    s16 gain = static_cast<s16>(w1 & 0xFFFF);
    u32 address = resolve_address(w2);

    s16 hist[4] = {0, 0, 0, 0};
    if (!init) {
        if (address != 0 && address + 8 <= rdram_sz) {
            for (int i = 0; i < 4; ++i) hist[i] = dram_s16(address + i * 2u);
        } else {
            auto it = polef_state.find(address);
            if (it != polef_state.end()) {
                std::memcpy(hist, it->second.hist, sizeof(hist));
            }
        }
    }

    u16 n_samples = align16(buf_count) >> 1;
    const s16* coef = codebook.data();
    for (u16 k = 0; k < n_samples; ++k) {
        s16 in_sample = scratch_s16(buf_in + k * 2u);
        s32 acc = (static_cast<s32>(in_sample) * gain) >> 15;
        acc += (static_cast<s32>(hist[3]) * coef[0]) >> 15;
        acc += (static_cast<s32>(hist[2]) * coef[1]) >> 15;
        s16 out_sample = clamp_s16(acc);
        hist[0] = hist[1];
        hist[1] = hist[2];
        hist[2] = hist[3];
        hist[3] = out_sample;
        set_scratch_s16(buf_out + k * 2u, out_sample);
    }

    PolefState saved;
    std::memcpy(saved.hist, hist, sizeof(hist));
    polef_state[address] = saved;
}

// -----------------------------------------------------------------------------
// ABI 2 (N_AUDIO) Implementations
// -----------------------------------------------------------------------------

void AudioHLE::cmd_adpcm_abi2(u32 w1, u32 w2) {
    u32 address = resolve_address(w1 & 0x00FFFFFF);
    u8 flags = (w2 >> 28) & 0xF;
    bool init = flags & 0x1;
    bool loop = flags & 0x2;
    u16 count = align16((w2 >> 16) & 0xFFF);
    u16 dmem = (w2 & 0xFFF) + DMEM_BASE;

    buf_count = count;
    buf_out = dmem;

    s16 hist[16];
    if (init) {
        std::memset(hist, 0, sizeof(hist));
        for (int i = 0; i < 16; ++i) set_scratch_s16(buf_out + i * 2u, 0);
    } else {
        u32 src = loop ? loop_addr : address;
        if (src != 0 && src + 32 <= rdram_sz) {
            for (int i = 0; i < 16; ++i) hist[i] = dram_s16(src + i * 2u);
        } else {
            std::memset(hist, 0, sizeof(hist));
        }
        for (int i = 0; i < 16; ++i) set_scratch_s16(buf_out + i * 2u, hist[i]);
    }

    u16 dmemi = buf_in;
    u16 dmemo = buf_out + 32;
    u16 remaining = count;
    while (remaining != 0) {
        u8 control = scratch_u8(dmemi++);
        unsigned scale = (control >> 4) & 0xF;
        unsigned predictor = control & 0xF;
        const s16* cb = codebook.data() + (predictor << 4);

        s16 residual[16];
        unsigned rshift = (scale < 12) ? (12 - scale) : 0;
        for (int i = 0; i < 8; ++i) {
            u8 byte = scratch_u8(dmemi++);
            residual[2 * i] = adpcm_nibble(byte, 0xF0, 8, rshift);
            residual[2 * i + 1] = adpcm_nibble(byte, 0x0F, 12, rshift);
        }

        const s16* book1 = cb;
        const s16* book2 = cb + 8;

        s16 out0[8];
        for (int i = 0; i < 8; ++i) {
            s32 acc = static_cast<s32>(residual[i]) << 11;
            acc += book1[i] * hist[14] + book2[i] * hist[15] + reverse_dot(i, book2, residual);
            out0[i] = clamp_s16(acc >> 11);
        }
        for (int i = 0; i < 8; ++i) hist[i] = out0[i];

        s16 out1[8];
        for (int i = 0; i < 8; ++i) {
            s32 acc = static_cast<s32>(residual[8 + i]) << 11;
            acc += book1[i] * hist[6] + book2[i] * hist[7] + reverse_dot(i, book2, residual + 8);
            out1[i] = clamp_s16(acc >> 11);
        }
        for (int i = 0; i < 8; ++i) hist[8 + i] = out1[i];

        for (int i = 0; i < 16; ++i) {
            set_scratch_s16(dmemo, hist[i]);
            dmemo += 2;
        }

        remaining = (remaining > 32) ? (remaining - 32) : 0;
    }

    // End of cmd_adpcm_abi2
}

void AudioHLE::cmd_addmixer_abi2(u32 /*w1*/, u32 w2) {
    u16 dmemi = static_cast<u16>((w2 >> 16) & 0xFFFF) + DMEM_BASE;
    u16 dmemo = static_cast<u16>(w2 & 0xFFFF) + DMEM_BASE;
    u16 n = align16(buf_count) >> 1;
    for (u16 k = 0; k < n; ++k) {
        s16 src = scratch_s16(dmemi + k * 2u);
        s16 dst = scratch_s16(dmemo + k * 2u);
        set_scratch_s16(dmemo + k * 2u, clamp_s16(dst + src));
    }
}

void AudioHLE::cmd_resample_abi2(u32 w1, u32 w2) {
    u32 address = resolve_address(w1 & 0x00FFFFFF);
    bool init = (w2 >> 30) & 1;
    u32 pitch = ((w2 >> 14) & 0xFFFF) << 1;
    u16 dmemi = ((w2 >> 2) & 0xFFF) + DMEM_BASE;
    u16 dmemo = (w2 & 3) + DMEM_BASE;

    buf_in = dmemi;
    buf_out = dmemo;

    s16 hist[4] = {0, 0, 0, 0};
    u32 frac = 0;
    if (!init && address != 0 && address + 10 <= rdram_sz) {
        for (int k = 0; k < 4; ++k) hist[k] = dram_s16(address + k * 2u);
        frac = static_cast<u16>(dram_s16(address + 8));
    }

    auto sample_at = [&](s32 rel) -> s32 {
        if (rel < 0) return (rel >= -4) ? hist[4 + rel] : 0;
        return scratch_s16(buf_in + static_cast<u16>(rel) * 2u);
    };

    u16 n_samples = align16(buf_count) >> 1;
    s32 ipos = 0;
    for (u16 i = 0; i < n_samples; ++i) {
        const s16* lut = RESAMPLE_LUT + ((frac & 0xFC00) >> 8);
        s32 acc = sample_at(ipos - 1) * lut[0] +
                  sample_at(ipos)     * lut[1] +
                  sample_at(ipos + 1) * lut[2] +
                  sample_at(ipos + 2) * lut[3];
        set_scratch_s16(buf_out + i * 2u, clamp_s16(acc >> 15));

        frac += pitch;
        ipos += static_cast<s32>(frac >> 16);
        frac &= 0xFFFF;
    }

    // End of cmd_resample_abi2
}

void AudioHLE::cmd_resample_zoh_abi2(u32 /*w1*/, u32 w2) {
    u32 pitch = ((w2 >> 14) & 0xFFFF) << 1;
    u16 dmemi = ((w2 >> 2) & 0xFFF) + DMEM_BASE;
    u16 dmemo = (w2 & 3) + DMEM_BASE;
    u16 n_samples = align16(buf_count) >> 1;
    u32 frac = 0;
    s32 ipos = 0;
    for (u16 i = 0; i < n_samples; ++i) {
        s16 val = scratch_s16(dmemi + ipos * 2u);
        set_scratch_s16(dmemo + i * 2u, val);
        frac += pitch;
        ipos += static_cast<s32>(frac >> 16);
        frac &= 0xFFFF;
    }
}

void AudioHLE::cmd_dmemmove2_abi2(u32 w1, u32 w2) {
    u16 count = align16(static_cast<u16>((w1 >> 12) & 0xFF0));
    u16 dmemi = static_cast<u16>(w1 & 0xFFF) + DMEM_BASE;
    u16 dmemo = static_cast<u16>(w2 & 0xFFFF) + DMEM_BASE;
    if (count == 0) return;
    std::vector<u8> tmp(count);
    for (u16 i = 0; i < count; ++i) tmp[i] = scratch_u8(dmemi + i);
    for (u16 i = 0; i < count; ++i) set_scratch_u8(dmemo + i, tmp[i]);
}

void AudioHLE::cmd_envsetup1_abi2(u32 w1, u32 w2) {
    envsetup2.vol[0] = static_cast<s16>((w1 >> 8) & 0xFF00);
    envsetup2.vol[1] = static_cast<s16>(w1 & 0xFFFF);
    envsetup2.delta[0] = static_cast<s16>((w2 >> 16) & 0xFFFF);
    envsetup2.delta[1] = static_cast<s16>(w2 & 0xFFFF);
}

void AudioHLE::cmd_envsetup2_abi2(u32 /*w1*/, u32 w2) {
    envsetup2.wet_vol[0] = static_cast<s16>((w2 >> 16) & 0xFFFF);
    envsetup2.wet_vol[1] = static_cast<s16>(w2 & 0xFFFF);
}

void AudioHLE::cmd_envmixer_abi2(u32 w1, u32 w2) {
    u16 dmem_dl = ((w2 >> 20) & 0xFF0) + DMEM_BASE;
    u16 dmem_dr = ((w2 >> 12) & 0xFF0) + DMEM_BASE;
    u16 dmem_wl = ((w2 >> 4) & 0xFF0) + DMEM_BASE;
    u16 dmem_wr = ((w2 << 4) & 0xFF0) + DMEM_BASE;
    u16 n = align16((w1 >> 8) & 0xFF) >> 1;
    bool aux = (w1 & 0x4) != 0;

    s32 lvol = static_cast<s32>(envsetup2.vol[0]) << 16;
    s32 rvol = static_cast<s32>(envsetup2.vol[1]) << 16;
    s32 lstep = static_cast<s32>(envsetup2.delta[0]) << 13;
    s32 rstep = static_cast<s32>(envsetup2.delta[1]) << 13;

    for (u16 k = 0; k < n; ++k) {
        s16 v_l = static_cast<s16>(lvol >> 16);
        s16 v_r = static_cast<s16>(rvol >> 16);
        s16 src = scratch_s16(buf_in + k * 2u);

        s16 dl = scratch_s16(dmem_dl + k * 2u);
        set_scratch_s16(dmem_dl + k * 2u, clamp_s16(dl + ((static_cast<s32>(src) * v_l) >> 15)));
        s16 dr = scratch_s16(dmem_dr + k * 2u);
        set_scratch_s16(dmem_dr + k * 2u, clamp_s16(dr + ((static_cast<s32>(src) * v_r) >> 15)));

        if (aux) {
            s16 wl = scratch_s16(dmem_wl + k * 2u);
            set_scratch_s16(dmem_wl + k * 2u, clamp_s16(wl + ((static_cast<s32>(src) * envsetup2.wet_vol[0]) >> 15)));
            s16 wr = scratch_s16(dmem_wr + k * 2u);
            set_scratch_s16(dmem_wr + k * 2u, clamp_s16(wr + ((static_cast<s32>(src) * envsetup2.wet_vol[1]) >> 15)));
        }

        lvol += lstep;
        rvol += rstep;
    }
}

void AudioHLE::cmd_loadbuff_abi2(u32 w1, u32 w2) {
    u16 count = align16((w1 >> 12) & 0xFF0);
    u16 dmem = (w1 & 0xFFF) + DMEM_BASE;
    u32 address = resolve_address(w2 & 0x00FFFFFF);
    for (u16 i = 0; i < count; ++i) set_scratch_u8(dmem + i, dram_u8(address + i));
}

void AudioHLE::cmd_savebuff_abi2(u32 w1, u32 w2) {
    u16 count = align16((w1 >> 12) & 0xFF0);
    u16 dmem = (w1 & 0xFFF) + DMEM_BASE;
    u32 address = resolve_address(w2 & 0x00FFFFFF);
    for (u16 i = 0; i < count; ++i) set_dram_u8(address + i, scratch_u8(dmem + i));
}

void AudioHLE::cmd_mixer_abi2(u32 w1, u32 w2) {
    s16 gain = static_cast<s16>(w1 & 0xFFFF);
    u16 count = align16((w1 >> 12) & 0xFF0) >> 1;
    u16 dmemi = static_cast<u16>((w2 >> 16) & 0xFFFF) + DMEM_BASE;
    u16 dmemo = static_cast<u16>(w2 & 0xFFFF) + DMEM_BASE;
    for (u16 k = 0; k < count; ++k) {
        s16 src = scratch_s16(dmemi + k * 2u);
        s16 dst = scratch_s16(dmemo + k * 2u);
        dst = clamp_s16(dst + ((static_cast<s32>(src) * gain) >> 15));
        set_scratch_s16(dmemo + k * 2u, dst);
    }
}

void AudioHLE::cmd_interleave_abi2(u32 /*w1*/, u32 w2) {
    u16 left = static_cast<u16>((w2 >> 16) & 0xFFFF) + DMEM_BASE;
    u16 right = static_cast<u16>(w2 & 0xFFFF) + DMEM_BASE;
    u16 n = align16(buf_count) >> 1;
    for (u16 k = 0; k < n; ++k) {
        s16 l = scratch_s16(left + k * 2u);
        s16 r = scratch_s16(right + k * 2u);
        set_scratch_s16(buf_out + k * 4u, l);
        set_scratch_s16(buf_out + k * 4u + 2u, r);
    }
}

// -----------------------------------------------------------------------------
// NEAD (Nintendo EAD) Implementations
// -----------------------------------------------------------------------------

void AudioHLE::dispatch_nead_mk(u32 acmd, u32 w1, u32 w2) {
    switch (acmd) {
        case 0x00: break; // SPNOOP
        case 0x01: cmd_adpcm_nead(w1, w2); break;
        case 0x02: cmd_clearbuff_nead(w1, w2); break;
        case 0x05: cmd_resample_nead(w1, w2); break;
        case 0x07: break; // SEGMENT (noop in NEAD)
        case 0x08: cmd_setbuff_nead(w1, w2); break;
        case 0x0A: cmd_dmemmove_nead(w1, w2); break;
        case 0x0B: cmd_loadadpcm_nead(w1, w2); break;
        case 0x0C: cmd_mixer_nead(w1, w2); break;
        case 0x0D: cmd_interleave_nead_mk(w1, w2); break;
        case 0x0E: cmd_polef_nead(w1, w2); break;
        case 0x0F: cmd_setloop(w1, w2); break;
        case 0x10: cmd_nead16(w1, w2); break;
        case 0x11: cmd_interl_nead(w1, w2); break;
        case 0x12: cmd_envsetup1_nead_mk(w1, w2); break;
        case 0x13: cmd_envmixer_nead_mk(w1, w2); break;
        case 0x14: cmd_loadbuff_nead(w1, w2); break;
        case 0x15: cmd_savebuff_nead(w1, w2); break;
        case 0x16: cmd_envsetup2_nead(w1, w2); break;
        default: break;
    }
}

void AudioHLE::dispatch_nead_sf(u32 acmd, u32 w1, u32 w2) {
    switch (acmd) {
        case 0x00: break; // SPNOOP
        case 0x01: cmd_adpcm_nead(w1, w2); break;
        case 0x02: cmd_clearbuff_nead(w1, w2); break;
        case 0x04: cmd_addmixer_nead(w1, w2); break;
        case 0x05: cmd_resample_nead(w1, w2); break;
        case 0x06: cmd_resample_zoh_nead(w1, w2); break;
        case 0x08: cmd_setbuff_nead(w1, w2); break;
        case 0x0A: cmd_dmemmove_nead(w1, w2); break;
        case 0x0B: cmd_loadadpcm_nead(w1, w2); break;
        case 0x0C: cmd_mixer_nead(w1, w2); break;
        case 0x0D: cmd_interleave_nead_mk(w1, w2); break;
        case 0x0E: cmd_polef_nead(w1, w2); break;
        case 0x0F: cmd_setloop(w1, w2); break;
        case 0x10: cmd_nead16(w1, w2); break;
        case 0x11: cmd_interl_nead(w1, w2); break;
        case 0x12: cmd_envsetup1_nead(w1, w2); break;
        case 0x13: cmd_envmixer_nead(w1, w2); break;
        case 0x14: cmd_loadbuff_nead(w1, w2); break;
        case 0x15: cmd_savebuff_nead(w1, w2); break;
        case 0x16: cmd_envsetup2_nead(w1, w2); break;
        case 0x18: cmd_hilogain_nead(w1, w2); break;
        case 0x1A: cmd_duplicate_nead(w1, w2); break;
        default: break;
    }
}

void AudioHLE::dispatch_nead_oot(u32 acmd, u32 w1, u32 w2) {
    switch (acmd) {
        case 0x00: break; // SPNOOP
        case 0x01: cmd_adpcm_nead(w1, w2); break;
        case 0x02: cmd_clearbuff_nead(w1, w2); break;
        case 0x04: cmd_addmixer_nead(w1, w2); break;
        case 0x05: cmd_resample_nead(w1, w2); break;
        case 0x06: cmd_resample_zoh_nead(w1, w2); break;
        case 0x07: cmd_filter_nead(w1, w2); break;
        case 0x08: cmd_setbuff_nead(w1, w2); break;
        case 0x09: cmd_duplicate_nead(w1, w2); break;
        case 0x0A: cmd_dmemmove_nead(w1, w2); break;
        case 0x0B: cmd_loadadpcm_nead(w1, w2); break;
        case 0x0C: cmd_mixer_nead(w1, w2); break;
        case 0x0D: cmd_interleave_nead(w1, w2); break;
        case 0x0E: cmd_hilogain_nead(w1, w2); break;
        case 0x0F: cmd_setloop(w1, w2); break;
        case 0x10: cmd_nead16(w1, w2); break;
        case 0x11: cmd_interl_nead(w1, w2); break;
        case 0x12: cmd_envsetup1_nead(w1, w2); break;
        case 0x13: cmd_envmixer_nead(w1, w2); break;
        case 0x14: cmd_loadbuff_nead(w1, w2); break;
        case 0x15: cmd_savebuff_nead(w1, w2); break;
        case 0x16: cmd_envsetup2_nead(w1, w2); break;
        default: break;
    }
}

void AudioHLE::cmd_setbuff_nead(u32 w1, u32 w2) {
    buf_in = static_cast<u16>(w1 & 0xFFFF);
    buf_out = static_cast<u16>((w2 >> 16) & 0xFFFF);
    buf_count = static_cast<u16>(w2 & 0xFFFF);
}

void AudioHLE::cmd_loadbuff_nead(u32 w1, u32 w2) {
    u16 count = align16((w1 >> 12) & 0xFFF);
    u16 dmem = (w1 & 0xFFF) & ~3u;
    u32 address = (w2 & 0x00FFFFFF) & ~7u;
    for (u16 i = 0; i < count; ++i) {
        set_scratch_u8(dmem + i, dram_u8(address + i));
    }
}

void AudioHLE::cmd_savebuff_nead(u32 w1, u32 w2) {
    u16 count = align16((w1 >> 12) & 0xFFF);
    u16 dmem = (w1 & 0xFFF) & ~3u;
    u32 address = (w2 & 0x00FFFFFF) & ~7u;
    for (u16 i = 0; i < count; ++i) {
        set_dram_u8(address + i, scratch_u8(dmem + i));
    }
}

void AudioHLE::cmd_clearbuff_nead(u32 w1, u32 w2) {
    u16 dmem = static_cast<u16>(w1 & 0xFFFF);
    u16 count = align16(static_cast<u16>(w2 & 0xFFFF));
    for (u16 i = 0; i < count; ++i) {
        set_scratch_u8(dmem + i, 0);
    }
}

void AudioHLE::cmd_dmemmove_nead(u32 w1, u32 w2) {
    u16 dmemi = static_cast<u16>((w1 >> 16) & 0xFFFF);
    u16 dmemo = static_cast<u16>(w1 & 0xFFFF);
    u16 count = (static_cast<u16>(w2 & 0xFFFF) + 3u) & ~3u;
    if (count == 0) return;
    std::vector<u8> tmp(count);
    for (u16 i = 0; i < count; ++i) tmp[i] = scratch_u8(dmemi + i);
    for (u16 i = 0; i < count; ++i) set_scratch_u8(dmemo + i, tmp[i]);
}

void AudioHLE::cmd_loadadpcm_nead(u32 w1, u32 w2) {
    u16 count = static_cast<u16>(w1 & 0xFFFF);
    u32 address = w2 & 0x00FFFFFF;
    u16 n = count >> 1;
    for (u16 i = 0; i < n && i < codebook.size(); ++i) {
        codebook[i] = dram_s16(address + i * 2u);
    }
}

void AudioHLE::cmd_mixer_nead(u32 w1, u32 w2) {
    u16 count = align16((w1 >> 12) & 0xFF0) >> 1;
    s16 gain = static_cast<s16>(w1 & 0xFFFF);
    u16 dmemi = static_cast<u16>((w2 >> 16) & 0xFFFF);
    u16 dmemo = static_cast<u16>(w2 & 0xFFFF);
    for (u16 k = 0; k < count; ++k) {
        s16 src = scratch_s16(dmemi + k * 2u);
        s16 dst = scratch_s16(dmemo + k * 2u);
        dst = clamp_s16(dst + ((static_cast<s32>(src) * gain) >> 15));
        set_scratch_s16(dmemo + k * 2u, dst);
    }
}

void AudioHLE::cmd_addmixer_nead(u32 w1, u32 w2) {
    u16 count = align16((w1 >> 12) & 0xFF0) >> 1;
    u16 dmemi = static_cast<u16>((w2 >> 16) & 0xFFFF);
    u16 dmemo = static_cast<u16>(w2 & 0xFFFF);
    for (u16 k = 0; k < count; ++k) {
        s16 src = scratch_s16(dmemi + k * 2u);
        s16 dst = scratch_s16(dmemo + k * 2u);
        set_scratch_s16(dmemo + k * 2u, clamp_s16(dst + src));
    }
}

void AudioHLE::cmd_interleave_nead_mk(u32 /*w1*/, u32 w2) {
    u16 left = static_cast<u16>((w2 >> 16) & 0xFFFF);
    u16 right = static_cast<u16>(w2 & 0xFFFF);
    u16 dmemo = buf_out;
    u16 n_samples = buf_count >> 1;
    if (n_samples == 0) return;

    for (u16 i = 0; i < n_samples; ++i) {
        s16 l = scratch_s16(left + i * 2u);
        s16 r = scratch_s16(right + i * 2u);
        set_scratch_s16(dmemo + i * 4u, l);
        set_scratch_s16(dmemo + i * 4u + 2u, r);
    }
}

void AudioHLE::cmd_interleave_nead(u32 w1, u32 w2) {
    u16 count = align16((w1 >> 12) & 0xFF0);
    u16 dmemo = static_cast<u16>(w1 & 0xFFF);
    u16 left = static_cast<u16>((w2 >> 16) & 0xFFFF);
    u16 right = static_cast<u16>(w2 & 0xFFFF);
    u16 n_samples = count >> 1;
    if (n_samples == 0) return;

    for (u16 i = 0; i < n_samples; ++i) {
        s16 l = scratch_s16(left + i * 2u);
        s16 r = scratch_s16(right + i * 2u);
        set_scratch_s16(dmemo + i * 4u, l);
        set_scratch_s16(dmemo + i * 4u + 2u, r);
    }
}

void AudioHLE::cmd_polef_nead(u32 w1, u32 w2) {
    if (buf_count == 0) return;
    u8 flags = (w1 >> 16) & 0xFF;
    bool init = flags & A_INIT;
    s16 gain = static_cast<s16>(w1 & 0xFFFF);
    u32 address = w2 & 0x00FFFFFF;

    s16 hist[4] = {0, 0, 0, 0};
    if (!init) {
        auto it = polef_state.find(address);
        if (it != polef_state.end()) {
            std::memcpy(hist, it->second.hist, sizeof(hist));
        } else if (address != 0 && address + 8 <= rdram_sz) {
            for (int i = 0; i < 4; ++i) hist[i] = dram_s16(address + i * 2u);
        }
    }

    u16 n_samples = align16(buf_count) >> 1;
    const s16* coef = codebook.data();
    for (u16 k = 0; k < n_samples; ++k) {
        s16 in_sample = scratch_s16(buf_in + k * 2u);
        s32 acc = (static_cast<s32>(in_sample) * gain) >> 15;
        acc += (static_cast<s32>(hist[3]) * coef[0]) >> 15;
        acc += (static_cast<s32>(hist[2]) * coef[1]) >> 15;
        s16 out_sample = clamp_s16(acc);
        hist[0] = hist[1];
        hist[1] = hist[2];
        hist[2] = hist[3];
        hist[3] = out_sample;
        set_scratch_s16(buf_out + k * 2u, out_sample);
    }

    PolefState saved;
    std::memcpy(saved.hist, hist, sizeof(hist));
    polef_state[address] = saved;
}

void AudioHLE::cmd_nead16(u32 w1, u32 w2) {
    u8 count = static_cast<u8>((w1 >> 16) & 0xFF);
    u16 dmemi = static_cast<u16>(w1 & 0xFFFF);
    u16 dmemo = static_cast<u16>((w2 >> 16) & 0xFFFF);
    u16 block_size = static_cast<u16>(w2 & 0xFFFF);

    for (u8 b = 0; b < count; ++b) {
        for (u16 i = 0; i < block_size; i += 32) {
            for (u16 k = 0; k < 32; ++k) {
                set_scratch_u8(dmemo + k, scratch_u8(dmemi + k));
            }
            dmemi += 32;
            dmemo += 32;
        }
    }
}

void AudioHLE::cmd_interl_nead(u32 w1, u32 w2) {
    u16 count = static_cast<u16>(w1 & 0xFFFF);
    u16 dmemi = static_cast<u16>((w2 >> 16) & 0xFFFF);
    u16 dmemo = static_cast<u16>(w2 & 0xFFFF);

    for (u16 i = 0; i < count; ++i) {
        set_scratch_s16(dmemo + i * 2u, scratch_s16(dmemi + i * 4u));
    }
}

void AudioHLE::cmd_envsetup1_nead_mk(u32 w1, u32 w2) {
    nead_env_values[2] = static_cast<u16>((w1 >> 8) & 0xFF00);
    nead_env_steps[2] = 0;
    nead_env_steps[0] = static_cast<u16>((w2 >> 16) & 0xFFFF);
    nead_env_steps[1] = static_cast<u16>(w2 & 0xFFFF);
}

void AudioHLE::cmd_envsetup1_nead(u32 w1, u32 w2) {
    nead_env_values[2] = static_cast<u16>((w1 >> 8) & 0xFF00);
    nead_env_steps[2] = static_cast<u16>(w1 & 0xFFFF);
    nead_env_steps[0] = static_cast<u16>((w2 >> 16) & 0xFFFF);
    nead_env_steps[1] = static_cast<u16>(w2 & 0xFFFF);
}

void AudioHLE::cmd_envsetup2_nead(u32 /*w1*/, u32 w2) {
    nead_env_values[0] = static_cast<u16>((w2 >> 16) & 0xFFFF);
    nead_env_values[1] = static_cast<u16>(w2 & 0xFFFF);
}

void AudioHLE::cmd_envmixer_nead_mk(u32 w1, u32 w2) {
    u16 dmemi = static_cast<u16>((w1 >> 12) & 0xFF0);
    u8 count = static_cast<u8>((w1 >> 8) & 0xFF);
    u16 dmem_dl = static_cast<u16>((w2 >> 20) & 0xFF0);
    u16 dmem_dr = static_cast<u16>((w2 >> 12) & 0xFF0);
    u16 dmem_wl = static_cast<u16>((w2 >> 4) & 0xFF0);
    u16 dmem_wr = static_cast<u16>((w2 << 4) & 0xFF0);

    s16 xors[4] = {
        static_cast<s16>(0 - static_cast<s16>((w1 & 0x2) >> 1)),
        static_cast<s16>(0 - static_cast<s16>(w1 & 0x1)),
        0,
        0
    };

    u16 n_samples = (static_cast<u16>(count) + 7u) & ~7u;
    for (u16 s = 0; s < n_samples; s += 8) {
        for (u16 i = 0; i < 8; ++i) {
            s16 in = scratch_s16(dmemi + (s + i) * 2u);
            s16 l = static_cast<s16>(((static_cast<s32>(in) * static_cast<u32>(nead_env_values[0])) >> 16) ^ xors[0]);
            s16 r = static_cast<s16>(((static_cast<s32>(in) * static_cast<u32>(nead_env_values[1])) >> 16) ^ xors[1]);
            s16 l2 = static_cast<s16>(((static_cast<s32>(l) * static_cast<u32>(nead_env_values[2])) >> 16) ^ xors[2]);
            s16 r2 = static_cast<s16>(((static_cast<s32>(r) * static_cast<u32>(nead_env_values[2])) >> 16) ^ xors[3]);

            s16 dl = scratch_s16(dmem_dl + (s + i) * 2u);
            s16 dr = scratch_s16(dmem_dr + (s + i) * 2u);
            s16 wl = scratch_s16(dmem_wl + (s + i) * 2u);
            s16 wr = scratch_s16(dmem_wr + (s + i) * 2u);

            set_scratch_s16(dmem_dl + (s + i) * 2u, clamp_s16(dl + l));
            set_scratch_s16(dmem_dr + (s + i) * 2u, clamp_s16(dr + r));
            set_scratch_s16(dmem_wl + (s + i) * 2u, clamp_s16(wl + l2));
            set_scratch_s16(dmem_wr + (s + i) * 2u, clamp_s16(wr + r2));
        }

        nead_env_values[0] += nead_env_steps[0];
        nead_env_values[1] += nead_env_steps[1];
        nead_env_values[2] += nead_env_steps[2];
    }
}

void AudioHLE::cmd_envmixer_nead(u32 w1, u32 w2) {
    u16 dmemi = static_cast<u16>((w1 >> 12) & 0xFF0);
    u8 count = static_cast<u8>((w1 >> 8) & 0xFF);
    bool swap_wet_LR = (w1 >> 4) & 0x1;
    u16 dmem_dl = static_cast<u16>((w2 >> 20) & 0xFF0);
    u16 dmem_dr = static_cast<u16>((w2 >> 12) & 0xFF0);
    u16 dmem_wl = static_cast<u16>((w2 >> 4) & 0xFF0);
    u16 dmem_wr = static_cast<u16>((w2 << 4) & 0xFF0);

    if (swap_wet_LR) std::swap(dmem_wl, dmem_wr);

    s16 xors[4] = {
        static_cast<s16>(0 - static_cast<s16>((w1 & 0x2) >> 1)),
        static_cast<s16>(0 - static_cast<s16>(w1 & 0x1)),
        static_cast<s16>(0 - static_cast<s16>((w1 & 0x8) >> 1)),
        static_cast<s16>(0 - static_cast<s16>((w1 & 0x4) >> 1))
    };

    u16 n_samples = (static_cast<u16>(count) + 7u) & ~7u;
    for (u16 s = 0; s < n_samples; s += 8) {
        for (u16 i = 0; i < 8; ++i) {
            s16 in = scratch_s16(dmemi + (s + i) * 2u);
            s16 l = static_cast<s16>(((static_cast<s32>(in) * static_cast<u32>(nead_env_values[0])) >> 16) ^ xors[0]);
            s16 r = static_cast<s16>(((static_cast<s32>(in) * static_cast<u32>(nead_env_values[1])) >> 16) ^ xors[1]);
            s16 l2 = static_cast<s16>(((static_cast<s32>(l) * static_cast<u32>(nead_env_values[2])) >> 16) ^ xors[2]);
            s16 r2 = static_cast<s16>(((static_cast<s32>(r) * static_cast<u32>(nead_env_values[2])) >> 16) ^ xors[3]);

            s16 dl = scratch_s16(dmem_dl + (s + i) * 2u);
            s16 dr = scratch_s16(dmem_dr + (s + i) * 2u);
            s16 wl = scratch_s16(dmem_wl + (s + i) * 2u);
            s16 wr = scratch_s16(dmem_wr + (s + i) * 2u);

            set_scratch_s16(dmem_dl + (s + i) * 2u, clamp_s16(dl + l));
            set_scratch_s16(dmem_dr + (s + i) * 2u, clamp_s16(dr + r));
            set_scratch_s16(dmem_wl + (s + i) * 2u, clamp_s16(wl + l2));
            set_scratch_s16(dmem_wr + (s + i) * 2u, clamp_s16(wr + r2));
        }

        nead_env_values[0] += nead_env_steps[0];
        nead_env_values[1] += nead_env_steps[1];
        nead_env_values[2] += nead_env_steps[2];
    }
}

void AudioHLE::cmd_adpcm_nead(u32 w1, u32 w2) {
    u8 flags = (w1 >> 16) & 0xFF;
    bool init = flags & 0x1;
    bool loop = flags & 0x2;
    bool two_bit = flags & 0x4;
    u32 address = w2 & 0x00FFFFFF;

    u16 dmemo = buf_out;
    u16 dmemi = buf_in;
    u16 count = (buf_count + 0x1Fu) & ~0x1Fu;

    s16 hist[16];
    if (init) {
        std::memset(hist, 0, sizeof(hist));
    } else {
        u32 src = loop ? loop_addr : address;
        auto it = adpcm_state.find(src);
        if (it != adpcm_state.end()) {
            std::memcpy(hist, it->second.data(), sizeof(hist));
        } else if (src != 0 && src + 32 <= rdram_sz) {
            for (int i = 0; i < 16; ++i) hist[i] = dram_s16(src + i * 2u);
        } else {
            std::memset(hist, 0, sizeof(hist));
        }
    }

    for (int i = 0; i < 16; ++i) {
        set_scratch_s16(dmemo, hist[i]);
        dmemo += 2;
    }

    while (count != 0) {
        u8 control = scratch_u8(dmemi++);
        unsigned scale = (control >> 4) & 0xF;
        unsigned predictor = control & 0xF;
        const s16* cb = codebook.data() + (predictor << 4);
        if (predictor >= 16) cb = codebook.data();

        s16 residual[16];
        if (two_bit) {
            unsigned rshift = (scale < 14) ? (14 - scale) : 0;
            for (int i = 0; i < 4; ++i) {
                u8 byte = scratch_u8(dmemi++);
                residual[4 * i + 0] = adpcm_nibble(byte, 0xC0, 8, rshift);
                residual[4 * i + 1] = adpcm_nibble(byte, 0x30, 10, rshift);
                residual[4 * i + 2] = adpcm_nibble(byte, 0x0C, 12, rshift);
                residual[4 * i + 3] = adpcm_nibble(byte, 0x03, 14, rshift);
            }
        } else {
            unsigned rshift = (scale < 12) ? (12 - scale) : 0;
            for (int i = 0; i < 8; ++i) {
                u8 byte = scratch_u8(dmemi++);
                residual[2 * i + 0] = adpcm_nibble(byte, 0xF0, 8, rshift);
                residual[2 * i + 1] = adpcm_nibble(byte, 0x0F, 12, rshift);
            }
        }

        const s16* book1 = cb;
        const s16* book2 = cb + 8;

        s16 out0[8];
        for (int i = 0; i < 8; ++i) {
            s32 acc = static_cast<s32>(residual[i]) << 11;
            acc += book1[i] * hist[14] + book2[i] * hist[15] + reverse_dot(i, book2, residual);
            out0[i] = clamp_s16(acc >> 11);
        }
        for (int i = 0; i < 8; ++i) hist[i] = out0[i];

        s16 out1[8];
        for (int i = 0; i < 8; ++i) {
            s32 acc = static_cast<s32>(residual[8 + i]) << 11;
            acc += book1[i] * hist[6] + book2[i] * hist[7] + reverse_dot(i, book2, residual + 8);
            out1[i] = clamp_s16(acc >> 11);
        }
        for (int i = 0; i < 8; ++i) hist[8 + i] = out1[i];

        for (int i = 0; i < 16; ++i) {
            set_scratch_s16(dmemo, hist[i]);
            dmemo += 2;
        }

        count = (count > 32) ? (count - 32) : 0;
    }

    std::array<s16, 16> saved;
    std::memcpy(saved.data(), hist, sizeof(hist));
    adpcm_state[address] = saved;
}

void AudioHLE::cmd_resample_nead(u32 w1, u32 w2) {
    u8 flags = (w1 >> 16) & 0xFF;
    bool init = flags & 0x1;
    u32 pitch = (w1 & 0xFFFF) << 1;
    u32 address = w2 & 0x00FFFFFF;

    u16 dmemo = buf_out;
    u16 dmemi = buf_in;
    u16 count = (buf_count + 0xFu) & ~0xFu;
    u16 n_samples = count >> 1;

    s16 hist[4] = {0, 0, 0, 0};
    u32 frac = 0;
    if (!init) {
        auto it = resample_state.find(address);
        if (it != resample_state.end()) {
            std::memcpy(hist, it->second.hist, sizeof(hist));
            frac = it->second.frac;
        } else if (address != 0 && address + 10 <= rdram_sz) {
            for (int k = 0; k < 4; ++k) hist[k] = dram_s16(address + k * 2u);
            frac = static_cast<u16>(dram_s16(address + 8));
        }
    }

    auto sample_at = [&](s32 rel) -> s32 {
        if (rel < 0) return (rel >= -4) ? hist[4 + rel] : 0;
        return scratch_s16(dmemi + static_cast<u16>(rel) * 2u);
    };

    s32 ipos = 0;
    for (u16 i = 0; i < n_samples; ++i) {
        const s16* lut = RESAMPLE_LUT + ((frac & 0xFC00) >> 8);
        s32 acc = sample_at(ipos - 1) * lut[0] +
                  sample_at(ipos)     * lut[1] +
                  sample_at(ipos + 1) * lut[2] +
                  sample_at(ipos + 2) * lut[3];
        set_scratch_s16(dmemo + i * 2u, clamp_s16(acc >> 15));

        frac += pitch;
        ipos += static_cast<s32>(frac >> 16);
        frac &= 0xFFFF;
    }

    ResampleState saved;
    for (int k = 0; k < 4; ++k) {
        saved.hist[k] = static_cast<s16>(sample_at(ipos - 4 + k));
    }
    saved.frac = frac;
    resample_state[address] = saved;
}

void AudioHLE::cmd_resample_zoh_nead(u32 w1, u32 w2) {
    u32 pitch = (w1 & 0xFFFF) << 1;
    u32 frac = w2 & 0xFFFF;
    u16 dmemo = buf_out;
    u16 dmemi = buf_in;
    u16 count = buf_count >> 1;
    s32 ipos = 0;

    for (u16 i = 0; i < count; ++i) {
        s16 val = scratch_s16(dmemi + ipos * 2u);
        set_scratch_s16(dmemo + i * 2u, val);
        frac += pitch;
        ipos += static_cast<s32>(frac >> 16);
        frac &= 0xFFFF;
    }
}

void AudioHLE::cmd_hilogain_nead(u32 w1, u32 w2) {
    s8 gain = static_cast<s8>((w1 >> 16) & 0xFF);
    u16 count = (w1 & 0xFFF) >> 1;
    u16 dmem = static_cast<u16>((w2 >> 16) & 0xFFFF);
    for (u16 i = 0; i < count; ++i) {
        s16 val = scratch_s16(dmem + i * 2u);
        set_scratch_s16(dmem + i * 2u, clamp_s16((static_cast<s32>(val) * gain) >> 4));
    }
}

void AudioHLE::cmd_duplicate_nead(u32 w1, u32 w2) {
    u8 count = static_cast<u8>((w1 >> 16) & 0xFF);
    u16 dmemi = static_cast<u16>(w1 & 0xFFFF);
    u16 dmemo = static_cast<u16>((w2 >> 16) & 0xFFFF);

    std::array<u8, 128> buf;
    for (size_t i = 0; i < 128; ++i) buf[i] = scratch_u8(dmemi + i);
    while (count != 0) {
        for (size_t i = 0; i < 128; ++i) set_scratch_u8(dmemo + i, buf[i]);
        dmemo += 128;
        --count;
    }
}

void AudioHLE::cmd_filter_nead(u32 w1, u32 w2) {
    u8 flags = static_cast<u8>((w1 >> 16) & 0xFF);
    u32 address = w2 & 0x00FFFFFF;

    if (flags > 1) {
        nead_filter_count = static_cast<u16>(w1 & 0xFFFF);
        nead_filter_lut[0] = address;
    } else {
        u16 dmem = static_cast<u16>(w1 & 0xFFFF);
        nead_filter_lut[1] = address + 0x10;
        u16 count = nead_filter_count;
        if (count == 0) return;

        s16 lutt6[8], lutt5[8];
        for (int x = 0; x < 8; ++x) {
            lutt6[x] = dram_s16(nead_filter_lut[0] + x * 2u);
            lutt5[x] = dram_s16(nead_filter_lut[1] + x * 2u);
            s32 avg = (static_cast<s32>(lutt5[x]) + lutt6[x]) >> 1;
            lutt5[x] = lutt6[x] = static_cast<s16>(avg);
        }

        for (u16 x = 0; x < count; x += 16) {
            s16 in1[8], in2[8];
            for (int k = 0; k < 8; ++k) {
                in1[k] = dram_s16(address + (x + k) * 2u);
                in2[k] = scratch_s16(dmem + (x + k) * 2u);
            }
            s32 v[8];
            v[0] = in1[3] * lutt6[6] + in1[2] * lutt6[7] + in1[1] * lutt6[4] + in1[4] * lutt6[5] +
                   in1[3] * lutt6[2] + in1[6] * lutt6[3] + in1[5] * lutt6[0] + in2[0] * lutt6[1];
            v[1] = in1[0] * lutt6[6] + in1[3] * lutt6[7] + in1[2] * lutt6[4] + in1[5] * lutt6[5] +
                   in1[4] * lutt6[2] + in1[7] * lutt6[3] + in1[6] * lutt6[0] + in2[1] * lutt6[1];
            for (int k = 2; k < 8; ++k) {
                v[k] = in2[k - 2] * lutt6[6] + in2[k + 1] * lutt6[7] + in2[k + 0] * lutt6[4] +
                       in2[k + 3] * lutt6[5] + in2[k + 2] * lutt6[2] + in2[k + 5] * lutt6[3] +
                       in2[k + 4] * lutt6[0] + in2[k - 1] * lutt6[1];
            }
            for (int k = 0; k < 8; ++k) {
                set_scratch_s16(dmem + (x + k) * 2u, clamp_s16(v[k] >> 15));
            }
        }
    }
}

