#pragma once

#include "common.hpp"
#include <array>

// High-level emulation of the RSP audio microcode (the "audio command list"
// / ALIST interpreter). Games submit an audio task (M_AUDTASK) whose DMEM
// contains a list of (w1,w2) command pairs; real hardware runs those on the
// RSP to ADPCM-decode, envelope-mix, resample and interleave voices into a
// PCM buffer in RDRAM, which the CPU then hands to the AI for DMA playback.
// This class reimplements that command list against a private DMEM-sized
// scratch buffer, independent of the RSP's real DMEM/IMEM.
//
// Like the microcode, every command that carries state from one audio frame
// to the next (ADPCM predictor history, resampler history and phase, the
// envelope ramps, pole/FIR filter history) keeps it in RDRAM at the address
// the game passes: the game owns that memory, restarts a voice by passing
// A_INIT, and a save state captures it along with the rest of RDRAM.
//
// The microcode flavour is recognised from its data segment, the same
// signature words the reference HLE implementations key on.
class AudioHLE {
public:
    void reset();

    // Processes `size` bytes of command-list data starting at RDRAM address
    // `addr` (as found in the audio OSTask's data_ptr/data_size fields).
    void process(u8* rdram, size_t rdram_size, u32 addr, u32 size, u32 ucode_data_ptr = 0);

    // Frontend status queries (read-only). ABI index follows AudioABI order:
    // 0=ABI1, 1=n_audio, 2=NEAD (Mario Kart), 3=NEAD (Star Fox / F-Zero X),
    // 4=NEAD (Zelda / Yoshi / 1080), 5=MusyX (not emulated: silent).
    int get_abi_index() const;
    u64 get_task_count() const { return task_count; }

    // Save states (savestate.hpp): what one command list leaves behind for
    // the next (the per-voice state lives in RDRAM, see above).
    template <class S> void serialize(S& s) {
        s(scratch, segments, table, in, out, count, dry_right, wet_left, wet_right, vol, target, rate, dry, wet, loop,
          env_values, env_steps, filter_count, filter_lut, abi, task_count);
    }

    enum class Abi : u8 {
        Audio,      // libultra ABI 1 (Super Mario 64, Wave Race 64, Dr. Mario 64, ...)
        AudioGE,    // ABI 1 with the GoldenEye / Blast Corps / Diddy Kong Racing envelope mixer
        NAudio,     // n_audio (Rare and many third-party titles)
        NAudioDK,   // n_audio, Donkey Kong 64 command table
        NAudioMP3,  // n_audio with MP3 support (Banjo-Tooie, Perfect Dark, Jet Force Gemini, Conker)
        NeadMK,     // Nintendo EAD: Mario Kart 64, Wave Race 64 (E)
        NeadSF,     // Nintendo EAD: Star Fox 64, Wave Race 64 (J rev B)
        NeadFZ,     // Nintendo EAD: F-Zero X
        NeadZelda,  // Nintendo EAD: Zelda OoT/MM, Yoshi's Story, 1080, Animal Crossing, Pokemon Stadium 2
        MusyX,      // Factor 5 MusyX: a different synthesizer altogether, not emulated
    };

private:
    static constexpr size_t SCRATCH_SIZE = 0x1000;

    std::array<u8, SCRATCH_SIZE> scratch{};
    // SM64's real ucode uses the segment number as an unmasked 8-bit word
    // index into its segment table (see audio.s cmd_SEGMENT: `srl $2,$25,24`
    // with no further masking), so we size this to the full byte range.
    std::array<u32, 256> segments{};
    // ADPCM codebook, also the pole filter's coefficients (LOADADPCM).
    std::array<s16, 256> table{};

    // Registers set by SETBUFF/SETVOL/SETLOOP and consumed by the others.
    u16 in{0}, out{0}, count{0};
    u16 dry_right{0}, wet_left{0}, wet_right{0};
    s16 vol[2]{0, 0};
    s16 target[2]{0, 0};
    s32 rate[2]{0, 0};
    s16 dry{0}, wet{0};
    u32 loop{0};
    // Nintendo EAD envelope and filter setup.
    u16 env_values[3]{0, 0, 0};
    u16 env_steps[3]{0, 0, 0};
    u16 filter_count{0};
    u32 filter_lut[2]{0, 0};

    Abi abi{Abi::Audio};
    u64 task_count{0};

    u8* rdram_ptr{nullptr};
    size_t rdram_sz{0};

    static Abi detect(const u8* rdram, size_t rdram_size, u32 ucode_data_ptr);
    void dispatch(u32 acmd, u32 w1, u32 w2);

    // DMEM scratch accessors (big-endian, wrapped to the 4 KB DMEM).
    u8 dmem_u8(u32 off) const { return scratch[off & (SCRATCH_SIZE - 1)]; }
    void set_dmem_u8(u32 off, u8 v) { scratch[off & (SCRATCH_SIZE - 1)] = v; }
    s16 dmem_s16(u32 off) const {
        return static_cast<s16>((dmem_u8(off) << 8) | dmem_u8(off + 1));
    }
    void set_dmem_s16(u32 off, s16 v) {
        set_dmem_u8(off, static_cast<u8>(static_cast<u16>(v) >> 8));
        set_dmem_u8(off + 1, static_cast<u8>(v));
    }
    // One 16-bit sample slot of DMEM by sample index (the resamplers' view).
    s16 sample(u32 pos) const { return dmem_s16((pos & 0x7FF) * 2); }
    void set_sample(u32 pos, s16 v) { set_dmem_s16((pos & 0x7FF) * 2, v); }

    // RDRAM accessors (big-endian, wrapped to RDRAM size).
    u8 dram_u8(u32 addr) const { return rdram_ptr[addr & (rdram_sz - 1)]; }
    void set_dram_u8(u32 addr, u8 v) { rdram_ptr[addr & (rdram_sz - 1)] = v; }
    s16 dram_s16(u32 addr) const { return static_cast<s16>((dram_u8(addr) << 8) | dram_u8(addr + 1)); }
    void set_dram_s16(u32 addr, s16 v) {
        set_dram_u8(addr, static_cast<u8>(static_cast<u16>(v) >> 8));
        set_dram_u8(addr + 1, static_cast<u8>(v));
    }
    s32 dram_s32(u32 addr) const {
        return static_cast<s32>((static_cast<u32>(static_cast<u16>(dram_s16(addr))) << 16) |
                                static_cast<u16>(dram_s16(addr + 2)));
    }
    void set_dram_s32(u32 addr, s32 v) {
        set_dram_s16(addr, static_cast<s16>(static_cast<u32>(v) >> 16));
        set_dram_s16(addr + 2, static_cast<s16>(v));
    }
    u32 dram_u32(u32 addr) const { return static_cast<u32>(dram_s32(addr)); }

    u32 segment_address(u32 so) const;
    void load_table(u32 address, u32 entries);

    // Building blocks shared by the microcode flavours; counts are in bytes
    // unless named otherwise.
    void clear(u16 dmem, u16 n);
    void load(u16 dmem, u32 address, u16 n);
    void save(u16 dmem, u32 address, u16 n);
    void move(u16 dmemo, u16 dmemi, u16 n);
    void mix(u16 dmemo, u16 dmemi, u16 n, s16 gain);
    void add(u16 dmemo, u16 dmemi, u16 n);
    void mult_q44(u16 dmem, u16 n, s8 gain);
    void interleave(u16 dmemo, u16 left, u16 right, u16 n);
    void copy_every_other_sample(u16 dmemo, u16 dmemi, u16 samples);
    void repeat64(u16 dmemo, u16 dmemi, u8 times);
    void copy_blocks(u16 dmemo, u16 dmemi, u16 block_size, u8 blocks);
    void adpcm(bool init, bool loop_state, bool two_bit, u16 dmemo, u16 dmemi, u16 n, u32 state);
    void resample(bool init, u16 dmemo, u16 dmemi, u16 n, u32 pitch, u32 state);
    void resample_zoh(u16 dmemo, u16 dmemi, u16 n, u32 pitch, u32 pitch_accu);
    void polef(bool init, u16 dmemo, u16 dmemi, u16 n, s16 gain, u32 state);
    void iirf(bool init, u16 dmemo, u16 dmemi, u16 n, u32 state);
    void fir_filter(bool init, u16 dmem, u16 n, u32 state);
    void envmix_exp(bool init, bool aux, u32 state);
    void envmix_ge(bool init, bool aux, u32 state);
    void envmix_lin(bool init, u16 dl, u16 dr, u16 wl, u16 wr, u16 dmemi, u16 n, u32 state);
    void envmix_nead(bool swap_wet, u16 dl, u16 dr, u16 wl, u16 wr, u16 dmemi, u32 samples, const s16 xors[4]);

    void run_audio(u32 acmd, u32 w1, u32 w2);
    void run_naudio(u32 acmd, u32 w1, u32 w2);
    void run_nead(u32 acmd, u32 w1, u32 w2);
};
