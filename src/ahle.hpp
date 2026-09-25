#pragma once

#include "common.hpp"
#include <array>
#include <unordered_map>

// High-level emulation of the RSP audio microcode (the "audio command list"
// / ALIST interpreter). Games submit an audio task (M_AUDTASK) whose DMEM
// contains a list of (w1,w2) command pairs; real hardware runs those on the
// RSP to ADPCM-decode, envelope-mix, resample and interleave voices into a
// PCM buffer in RDRAM, which the CPU then hands to the AI for DMA playback.
// This class reimplements that command list against a private DMEM-sized
// scratch buffer, independent of the RSP's real DMEM/IMEM.
class AudioHLE {
public:
    void reset();

    // Processes `size` bytes of command-list data starting at RDRAM address
    // `addr` (as found in the audio OSTask's data_ptr/data_size fields).
    void process(u8* rdram, size_t rdram_size, u32 addr, u32 size, u32 ucode_data_ptr = 0);

    // Frontend status queries (read-only). ABI index follows AudioABI order:
    // 0=ABI1, 1=ABI2, 2=NEAD_MK, 3=NEAD_SF, 4=NEAD_OOT.
    int get_abi_index() const { return static_cast<int>(current_abi); }
    u64 get_task_count() const { return task_count; }

private:
    static constexpr u32 DMEM_BASE = 0x5C0;
    static constexpr size_t SCRATCH_SIZE = 0x1000;

    std::array<u8, SCRATCH_SIZE> scratch{};
    // SM64's real ucode uses the segment number as an unmasked 8-bit word
    // index into its segment table (see audio.s cmd_SEGMENT: `srl $2,$25,24`
    // with no further masking), so we size this to the full byte range.
    std::array<u32, 256> segments{};
    std::array<s16, 256> codebook{};

    // State shared across commands within (and across) a command list, set
    // up by SETBUFF/SETVOL/SEGMENT/SETLOOP and consumed by ADPCM/ENVMIXER/
    // RESAMPLE/MIXER/INTERLEAVE.
    u16 buf_in{0};
    u16 buf_out{0};
    u16 buf_count{0};
    u16 buf_dry_right{0};
    u16 buf_wet_left{0};
    u16 buf_wet_right{0};
    s16 vol[2]{0, 0};
    s16 target[2]{0, 0};
    s32 rate[2]{0, 0};
    s16 dry{0};
    s16 wet{0};
    u32 loop_addr{0};

    u8* rdram_ptr{nullptr};
    size_t rdram_sz{0};

    // State structures used both for local continuation caching and RDRAM
    // serialization matching real hardware RSP DMA.
    struct EnvmixState {
        s16 dry{0};
        s16 wet{0};
        s64 target[2]{0, 0};
        s64 value[2]{0, 0};
        s32 exp_rate[2]{0, 0};
        s32 exp_seq[2]{0, 0};
    };
    struct ResampleState {
        s16 hist[4]{0, 0, 0, 0};
        u32 frac{0};
    };
    struct PolefState {
        s16 hist[4]{0, 0, 0, 0};
    };

    std::unordered_map<u32, std::array<s16, 16>> adpcm_state;
    std::unordered_map<u32, EnvmixState> envmix_state;
    std::unordered_map<u32, ResampleState> resample_state;
    std::unordered_map<u32, PolefState> polef_state;

    // ABI selection
    enum class AudioABI {
        ABI1,       // Standard libultra audio (Super Mario 64, Wave Race 64, GoldenEye, DKR)
        ABI2,       // N_AUDIO (Banjo-Kazooie, Donkey Kong 64, Conker, Perfect Dark)
        NEAD_MK,    // Nintendo EAD Mario Kart 64
        NEAD_SF,    // Nintendo EAD Star Fox 64 / F-Zero X
        NEAD_OOT,   // Nintendo EAD Zelda Ocarina of Time / Majora's Mask
    };
    AudioABI current_abi{AudioABI::ABI1};
    u64 task_count{0};

    // ABI 2 specific state
    struct EnvSetupABI2 {
        s16 vol[2]{0, 0};
        s16 delta[2]{0, 0};
        s16 wet_vol[2]{0, 0};
        s16 wet_delta[2]{0, 0};
    } envsetup2{};

    // NEAD specific state
    uint16_t nead_env_values[3]{0, 0, 0};
    uint16_t nead_env_steps[3]{0, 0, 0};
    uint16_t nead_filter_count{0};
    uint32_t nead_filter_lut[2]{0, 0};

    // Scratch (DMEM-relative) accessors.
    inline u8 scratch_u8(u16 off) const {
        return scratch[off & (SCRATCH_SIZE - 1)];
    }
    inline void set_scratch_u8(u16 off, u8 v) {
        scratch[off & (SCRATCH_SIZE - 1)] = v;
    }
    inline s16 scratch_s16(u16 off) const {
        off &= (SCRATCH_SIZE - 1);
        return static_cast<s16>((scratch[off] << 8) | scratch[off + 1]);
    }
    inline void set_scratch_s16(u16 off, s16 v) {
        off &= (SCRATCH_SIZE - 1);
        scratch[off] = static_cast<u8>((static_cast<u16>(v) >> 8) & 0xFF);
        scratch[off + 1] = static_cast<u8>(static_cast<u16>(v) & 0xFF);
    }

    // RDRAM accessors (big-endian, wrapped to RDRAM size).
    u8 dram_u8(u32 addr) const;
    void set_dram_u8(u32 addr, u8 v);
    s16 dram_s16(u32 addr) const;
    void set_dram_s16(u32 addr, s16 v);
    u32 dram_u32(u32 addr) const;

    u32 resolve_address(u32 seg_offset) const;

    void dispatch_abi1(u32 acmd, u32 w1, u32 w2);
    void dispatch_abi2(u32 acmd, u32 w1, u32 w2);
    void dispatch_nead_mk(u32 acmd, u32 w1, u32 w2);
    void dispatch_nead_sf(u32 acmd, u32 w1, u32 w2);
    void dispatch_nead_oot(u32 acmd, u32 w1, u32 w2);

    // ABI 1 commands
    void cmd_adpcm(u32 w1, u32 w2);
    void cmd_clearbuff(u32 w1, u32 w2);
    void cmd_envmixer(u32 w1, u32 w2);
    void cmd_loadbuff(u32 w1, u32 w2);
    void cmd_resample(u32 w1, u32 w2);
    void cmd_savebuff(u32 w1, u32 w2);
    void cmd_segment(u32 w1, u32 w2);
    void cmd_setbuff(u32 w1, u32 w2);
    void cmd_setvol(u32 w1, u32 w2);
    void cmd_dmemmove(u32 w1, u32 w2);
    void cmd_loadadpcm(u32 w1, u32 w2);
    void cmd_mixer(u32 w1, u32 w2);
    void cmd_interleave(u32 w1, u32 w2);
    void cmd_polef(u32 w1, u32 w2);
    void cmd_setloop(u32 w1, u32 w2);

    // ABI 2 (N_AUDIO) commands
    void cmd_adpcm_abi2(u32 w1, u32 w2);
    void cmd_addmixer_abi2(u32 w1, u32 w2);
    void cmd_resample_abi2(u32 w1, u32 w2);
    void cmd_resample_zoh_abi2(u32 w1, u32 w2);
    void cmd_dmemmove2_abi2(u32 w1, u32 w2);
    void cmd_envsetup1_abi2(u32 w1, u32 w2);
    void cmd_envsetup2_abi2(u32 w1, u32 w2);
    void cmd_envmixer_abi2(u32 w1, u32 w2);
    void cmd_loadbuff_abi2(u32 w1, u32 w2);
    void cmd_savebuff_abi2(u32 w1, u32 w2);
    void cmd_mixer_abi2(u32 w1, u32 w2);
    void cmd_interleave_abi2(u32 w1, u32 w2);

    // NEAD commands
    void cmd_setbuff_nead(u32 w1, u32 w2);
    void cmd_loadbuff_nead(u32 w1, u32 w2);
    void cmd_savebuff_nead(u32 w1, u32 w2);
    void cmd_clearbuff_nead(u32 w1, u32 w2);
    void cmd_dmemmove_nead(u32 w1, u32 w2);
    void cmd_loadadpcm_nead(u32 w1, u32 w2);
    void cmd_mixer_nead(u32 w1, u32 w2);
    void cmd_addmixer_nead(u32 w1, u32 w2);
    void cmd_interleave_nead_mk(u32 w1, u32 w2);
    void cmd_interleave_nead(u32 w1, u32 w2);
    void cmd_polef_nead(u32 w1, u32 w2);
    void cmd_nead16(u32 w1, u32 w2);
    void cmd_interl_nead(u32 w1, u32 w2);
    void cmd_envsetup1_nead_mk(u32 w1, u32 w2);
    void cmd_envsetup1_nead(u32 w1, u32 w2);
    void cmd_envsetup2_nead(u32 w1, u32 w2);
    void cmd_envmixer_nead_mk(u32 w1, u32 w2);
    void cmd_envmixer_nead(u32 w1, u32 w2);
    void cmd_adpcm_nead(u32 w1, u32 w2);
    void cmd_resample_nead(u32 w1, u32 w2);
    void cmd_resample_zoh_nead(u32 w1, u32 w2);
    void cmd_hilogain_nead(u32 w1, u32 w2);
    void cmd_duplicate_nead(u32 w1, u32 w2);
    void cmd_filter_nead(u32 w1, u32 w2);
};
