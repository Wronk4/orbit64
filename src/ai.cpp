#include "ai.hpp"
#include "mi.hpp"
#include "audio_stream.hpp"
#include "jit/jit_invalidate.hpp"

AI::AI() {
    reset();
}

void AI::reset() {
    dram_addr_reg = 0;
    control_reg = 0;
    dacrate_reg = 0;
    bitrate_reg = 0;
    current_buffer = {};
    next_buffer = {};
    busy = false;
    full = false;
    cycle_byte_accum = 0.0;
}

u32 AI::native_sample_rate() const {
    return static_cast<u32>(AI_VIDEO_CLOCK_NTSC / (dacrate_reg + 1));
}

u32 AI::read_reg(u32 addr) const {
    u32 reg = (addr & 0x1F) >> 2;
    switch (reg) {
        case 0: return dram_addr_reg;
        case 1: return current_buffer.remaining;
        case 2: return control_reg;
        case 3: {
            u32 status = 0;
            if (full) status |= (1 << 31);
            if (busy) status |= (1 << 30);
            return status;
        }
        case 4: return dacrate_reg;
        case 5: return bitrate_reg;
        default: return 0;
    }
}

void AI::write_reg(u32 addr, u32 val, MI& mi, const u8* rdram, size_t rdram_size) {
    u32 reg = (addr & 0x1F) >> 2;
    switch (reg) {
        case 0: // AI_DRAM_ADDR_REG
            dram_addr_reg = val & 0x00FFFFF8; // 24-bit, 8-byte aligned
            break;
        case 1: { // AI_LEN_REG
            u32 length = (val & 0x3FFF8);
            if (length == 0) break;

            if (!busy) {
                current_buffer.dram_addr = dram_addr_reg;
                current_buffer.length = length;
                current_buffer.remaining = length;
                busy = true;
            } else if (!full) {
                next_buffer.dram_addr = dram_addr_reg;
                next_buffer.length = length;
                next_buffer.remaining = length;
                full = true;
            }

            // The samples play at the native AI_DACRATE rate; the host side
            // (AudioStream) resamples them to the device rate.
            if (dram_addr_reg + length <= rdram_size && (sink_ || capture)) {
                jit::notify_read(dram_addr_reg, length);
                const u32 frame_count = length / 4;
                frames_.resize(frame_count * 2);
                for (u32 i = 0; i < frame_count * 2; i++) {
                    const u32 idx = dram_addr_reg + i * 2;
                    frames_[i] = static_cast<s16>((rdram[idx] << 8) | rdram[idx + 1]);
                }
                if (sink_) sink_->push(frames_.data(), frame_count, dacrate_reg ? native_sample_rate() : 44100);
                if (capture) capture->insert(capture->end(), frames_.begin(), frames_.end());
            }
            break;
        }
        case 2: // AI_CONTROL_REG
            control_reg = val & 1;
            break;
        case 3: // AI_STATUS_REG
            // Writing any value clears AI interrupt in MI
            mi.clear_interrupt(MIInterrupt::AI);
            break;
        case 4: // AI_DACRATE_REG
            dacrate_reg = val & 0x3FFF;
            break;
        case 5: // AI_BITRATE_REG
            bitrate_reg = val & 0xF;
            break;
    }
}

void AI::step(u32 cycles, MI& mi, const u8* /*rdram*/, size_t /*rdram_size*/) {
    if (!busy) return;

    // Convert elapsed CPU cycles into bytes consumed at the DAC's actual
    // sample rate (4 bytes per stereo 16-bit frame), carrying the
    // fractional remainder across calls to avoid drift.
    u32 rate = native_sample_rate();
    if (rate == 0) rate = 44100;
    cycle_byte_accum += static_cast<double>(cycles) * static_cast<double>(rate) * 4.0 / static_cast<double>(CPU_CLOCK_RATE);
    u32 bytes_to_consume = static_cast<u32>(cycle_byte_accum);
    if (bytes_to_consume == 0) return;
    cycle_byte_accum -= bytes_to_consume;

    if (current_buffer.remaining <= bytes_to_consume) {
        current_buffer.remaining = 0;
        mi.raise_interrupt(MIInterrupt::AI);

        if (full) {
            current_buffer = next_buffer;
            next_buffer = {};
            full = false;
        } else {
            busy = false;
        }
    } else {
        current_buffer.remaining -= bytes_to_consume;
    }
}
