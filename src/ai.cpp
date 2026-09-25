#include "ai.hpp"
#include "mi.hpp"
#include <cmath>

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
    resample_pos = 0.0;
    prev_left = 0.0f;
    prev_right = 0.0f;
    cycle_byte_accum = 0.0;
    carry_l = 0.0f;
    carry_r = 0.0f;
    std::lock_guard<std::mutex> lock(audio_mutex);
    ring_read = ring_write = ring_count = 0;
}

u32 AI::native_sample_rate() const {
    return static_cast<u32>(AI_VIDEO_CLOCK_NTSC / (dacrate_reg + 1));
}

void AI::push_frame(float l, float r) {
    ring[ring_write * 2] = l;
    ring[ring_write * 2 + 1] = r;
    ring_write = (ring_write + 1) % FIFO_CAPACITY_FRAMES;
    if (ring_count < FIFO_CAPACITY_FRAMES) {
        ring_count++;
    } else {
        // Buffer is full; drop the oldest frame instead of growing.
        ring_read = (ring_read + 1) % FIFO_CAPACITY_FRAMES;
    }
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

            // Decode audio samples (native AI_DACRATE rate) and resample them
            // to the host output rate before queuing, since games rarely run
            // the DAC at exactly the host's fixed output rate.
            if (dram_addr_reg + length <= rdram_size) {
                u32 frame_count = length / 4;
                std::vector<float> new_left(frame_count);
                std::vector<float> new_right(frame_count);
                for (u32 f = 0; f < frame_count; f++) {
                    u32 idx = dram_addr_reg + f * 4;
                    s16 left = static_cast<s16>((rdram[idx] << 8) | rdram[idx + 1]);
                    s16 right = static_cast<s16>((rdram[idx + 2] << 8) | rdram[idx + 3]);
                    new_left[f] = left / 32768.0f;
                    new_right[f] = right / 32768.0f;
                }

                u32 native_rate = native_sample_rate();
                if (native_rate == 0) native_rate = output_sample_rate;
                double ratio = static_cast<double>(native_rate) / static_cast<double>(output_sample_rate);

                std::lock_guard<std::mutex> lock(audio_mutex);

                // Gentle adaptive drift compensation: keep ring buffer around ~2048-4096 frames
                if (ring_count < 1024) {
                    ratio *= 0.995; // buffer running low: generate slightly more samples
                } else if (ring_count > 8192) {
                    ratio *= 1.005; // buffer growing too large: generate slightly fewer samples
                }

                double pos = resample_pos;
                while (frame_count > 0 && pos < static_cast<double>(frame_count)) {
                    float l, r;
                    if (pos < 0.0) {
                        double frac = pos + 1.0;
                        l = prev_left + (new_left[0] - prev_left) * static_cast<float>(frac);
                        r = prev_right + (new_right[0] - prev_right) * static_cast<float>(frac);
                    } else {
                        size_t idx = static_cast<size_t>(pos);
                        double frac = pos - static_cast<double>(idx);
                        float l0 = new_left[idx];
                        float r0 = new_right[idx];
                        float l1 = (idx + 1 < frame_count) ? new_left[idx + 1] : l0;
                        float r1 = (idx + 1 < frame_count) ? new_right[idx + 1] : r0;
                        l = l0 + (l1 - l0) * static_cast<float>(frac);
                        r = r0 + (r1 - r0) * static_cast<float>(frac);
                    }

                    push_frame(l, r);
                    pos += ratio;
                }
                if (frame_count > 0) {
                    resample_pos = pos - static_cast<double>(frame_count);
                    prev_left = new_left[frame_count - 1];
                    prev_right = new_right[frame_count - 1];
                }
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

void AI::get_samples(float* out_stream, size_t count) {
    // `count` is a total float count (interleaved L,R); SDL always requests
    // whole stereo frames for a stereo stream, so this is always even.
    size_t frames_requested = count / 2;
    size_t available_frames;
    {
        std::lock_guard<std::mutex> lock(audio_mutex);
        available_frames = std::min(frames_requested, ring_count);
        for (size_t i = 0; i < available_frames; i++) {
            size_t idx = (ring_read + i) % FIFO_CAPACITY_FRAMES;
            out_stream[i * 2] = ring[idx * 2];
            out_stream[i * 2 + 1] = ring[idx * 2 + 1];
        }
        ring_read = (ring_read + available_frames) % FIFO_CAPACITY_FRAMES;
        ring_count -= available_frames;
    }

    // Fade in from the previous call's last output level, so a resume after
    // an underrun doesn't jump straight from silence to full volume.
    constexpr size_t FADE_FRAMES = 64;
    size_t fade_in = std::min(available_frames, FADE_FRAMES);
    for (size_t i = 0; i < fade_in; i++) {
        float t = static_cast<float>(i + 1) / static_cast<float>(fade_in);
        out_stream[i * 2] = carry_l + (out_stream[i * 2] - carry_l) * t;
        out_stream[i * 2 + 1] = carry_r + (out_stream[i * 2 + 1] - carry_r) * t;
    }
    if (available_frames > 0) {
        carry_l = out_stream[(available_frames - 1) * 2];
        carry_r = out_stream[(available_frames - 1) * 2 + 1];
    }

    // Underrun: fade the tail out to silence instead of hard-cutting to it.
    size_t missing_frames = frames_requested - available_frames;
    if (missing_frames > 0) {
        size_t fade_out = std::min(missing_frames, FADE_FRAMES);
        for (size_t i = 0; i < fade_out; i++) {
            float t = 1.0f - static_cast<float>(i + 1) / static_cast<float>(fade_out);
            out_stream[(available_frames + i) * 2] = carry_l * t;
            out_stream[(available_frames + i) * 2 + 1] = carry_r * t;
        }
        for (size_t i = fade_out; i < missing_frames; i++) {
            out_stream[(available_frames + i) * 2] = 0.0f;
            out_stream[(available_frames + i) * 2 + 1] = 0.0f;
        }
        carry_l = 0.0f;
        carry_r = 0.0f;
    }
}
