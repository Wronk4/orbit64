#include "vi.hpp"
#include "mi.hpp"
#include <algorithm>

VI::VI() {
    reset();
}

void VI::reset() {
    status = 0;
    origin = 0;
    width = 320;
    v_intr = 2;
    v_current = 0;
    burst = 0;
    v_sync = 525;
    h_sync = 0;
    leap = 0;
    h_start = 0;
    v_start = 0;
    v_burst = 0;
    x_scale = 0x400;
    y_scale = 0x400;
}

u32 VI::read_reg(u32 addr) const {
    u32 reg = (addr & 0x3F) >> 2;
    switch (reg) {
        case 0: return status;
        case 1: return origin;
        case 2: return width;
        case 3: return v_intr;
        case 4: return v_current;
        case 5: return burst;
        case 6: return v_sync;
        case 7: return h_sync;
        case 8: return leap;
        case 9: return h_start;
        case 10: return v_start;
        case 11: return v_burst;
        case 12: return x_scale;
        case 13: return y_scale;
        default: return 0;
    }
}

void VI::write_reg(u32 addr, u32 val, MI& mi) {
    u32 reg = (addr & 0x3F) >> 2;
    switch (reg) {
        case 0: status = val; break;
        case 1: origin = val & 0x00FFFFFF; break;
        case 2: width = val & 0x00000FFF; break;
        case 3: v_intr = val & 0x000003FF; break;
        case 4:
            // Writing to VI_CURRENT clears VI interrupt
            mi.clear_interrupt(MIInterrupt::VI);
            break;
        case 5: burst = val; break;
        case 6: v_sync = val & 0x000003FF; break;
        case 7: h_sync = val; break;
        case 8: leap = val; break;
        case 9: h_start = val; break;
        case 10: v_start = val; break;
        case 11: v_burst = val; break;
        case 12: x_scale = val; break;
        case 13: y_scale = val; break;
    }
}

bool VI::step_scanline(MI& mi) {
    v_current = (v_current + 1) % (v_sync ? v_sync : 525);

    if (v_current == v_intr) {
        mi.raise_interrupt(MIInterrupt::VI);
        return true;
    }
    return false;
}

VIScanout VI::scanout(size_t rdram_size) const {
    VIScanout so;
    so.width = (width > 0 && width <= 640) ? width : 320;

    u32 type = status & 0x3;
    if (type < 2 || origin == 0 || origin >= rdram_size) {
        return so; // Blank or unconfigured
    }

    // Active display window check: when osViBlack(TRUE) is active, h_start/v_start window has zero size
    u32 h_end = h_start & 0x3FF;
    u32 h_beg = (h_start >> 16) & 0x3FF;
    u32 v_end = v_start & 0x3FF;
    u32 v_beg = (v_start >> 16) & 0x3FF;
    if (h_end <= h_beg || v_end <= v_beg) {
        return so; // Blackout
    }

    so.bpp = (type == 2) ? 2 : 4;
    u32 line_bytes = so.width * so.bpp;
    so.fb_base = origin & 0x007FFFFE;

    // N64 libultra VI modes (LAN1/LAN2) add one scanline (width * bpp) to origin for vertical filtering.
    // When AA mode is active and origin is offset by line_bytes, adjust fb_base back to the actual framebuffer start.
    if ((status & 0x300) != 0 && so.fb_base >= line_bytes) {
        so.fb_base -= line_bytes;
    }
    so.blank = false;
    return so;
}

void VI::render_frame(const u8* rdram, size_t rdram_size, std::vector<u32>& out_pixels, int& out_w, int& out_h) const {
    const VIScanout so = scanout(rdram_size);
    out_w = static_cast<int>(so.width);
    out_h = 240;

    out_pixels.resize(out_w * out_h, 0xFF000000); // Default opaque black

    if (so.blank) {
        return;
    }

    u32 line_bytes = so.width * so.bpp;
    u32 fb_base = so.fb_base;

    if (so.bpp == 2) {
        // 16-bit RGBA 5-5-5-1
        for (int y = 0; y < out_h; ++y) {
            u32 line_offset = fb_base + y * line_bytes;
            if (line_offset + line_bytes > rdram_size) break;

            for (int x = 0; x < out_w; ++x) {
                u32 byte_idx = line_offset + x * 2;
                u16 p = (static_cast<u16>(rdram[byte_idx]) << 8) | static_cast<u16>(rdram[byte_idx + 1]);

                u8 r = ((p >> 11) & 0x1F) * 255 / 31;
                u8 g = ((p >> 6) & 0x1F) * 255 / 31;
                u8 b = ((p >> 1) & 0x1F) * 255 / 31;
                u8 a = 255;

                out_pixels[y * out_w + x] = (static_cast<u32>(a) << 24) |
                                            (static_cast<u32>(r) << 16) |
                                            (static_cast<u32>(g) << 8)  |
                                             static_cast<u32>(b);
            }
        }
    } else {
        // 32-bit RGBA 8-8-8-8
        for (int y = 0; y < out_h; ++y) {
            u32 line_offset = fb_base + y * line_bytes;
            if (line_offset + line_bytes > rdram_size) break;

            for (int x = 0; x < out_w; ++x) {
                u32 byte_idx = line_offset + x * 4;
                u8 r = rdram[byte_idx + 0];
                u8 g = rdram[byte_idx + 1];
                u8 b = rdram[byte_idx + 2];
                u8 a = 255;

                out_pixels[y * out_w + x] = (static_cast<u32>(a) << 24) |
                                            (static_cast<u32>(r) << 16) |
                                            (static_cast<u32>(g) << 8)  |
                                             static_cast<u32>(b);
            }
        }
    }
}
