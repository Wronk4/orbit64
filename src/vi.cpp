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

    // The window is in VI pixels (640 a line) and half-lines; the scale
    // registers (2.10 fixed point) say how many frame buffer pixels each covers.
    const u32 xs = x_scale & 0xFFF, ys = y_scale & 0xFFF;
    if (xs == 0 || ys == 0) {
        so.shown_w = so.canvas_w = so.width;
        return so;
    }
    auto scaled = [](u32 n, u32 s) { return (n * s + 512) >> 10; };
    so.canvas_w = std::max<u32>(1, scaled(640, xs));
    so.canvas_h = std::max<u32>(1, scaled(240, ys));
    so.shown_w = std::min(so.width, std::max<u32>(1, scaled(h_end - h_beg, xs)));
    so.lines = std::max<u32>(1, scaled((v_end - v_beg) / 2, ys));
    // libultra's standard window is 237 lines of a 240-line buffer: show it
    // all rather than a sliver of black.
    if (so.lines < so.canvas_h && so.canvas_h - so.lines <= scaled(4, ys)) so.lines = so.canvas_h;
    so.canvas_w = std::max(so.canvas_w, so.shown_w);
    so.canvas_h = std::max(so.canvas_h, so.lines);
    so.x0 = (so.canvas_w - so.shown_w) / 2;
    so.y0 = (so.canvas_h - so.lines) / 2;
    return so;
}

void VIScanout::place(const u32* src, u32 src_w, u32 S, std::vector<u32>& out, int& out_w, int& out_h) const {
    out_w = static_cast<int>(canvas_w * S);
    out_h = static_cast<int>(canvas_h * S);
    out.assign(static_cast<size_t>(out_w) * out_h, 0xFF000000u);
    const u32 w = std::min(shown_w * S, src_w);
    for (u32 y = 0; y < lines * S; ++y) {
        const u32* s = src + static_cast<size_t>(y) * src_w;
        std::copy(s, s + w, out.data() + (static_cast<size_t>(y0 * S + y) * out_w + x0 * S));
    }
}

void VI::render_frame(const u8* rdram, size_t rdram_size, std::vector<u32>& out_pixels, int& out_w, int& out_h) const {
    const VIScanout so = scanout(rdram_size);
    out_w = static_cast<int>(so.canvas_w);
    out_h = static_cast<int>(so.canvas_h);
    out_pixels.assign(static_cast<size_t>(out_w) * out_h, 0xFF000000u); // opaque black borders

    if (so.blank) {
        return;
    }

    const u32 line_bytes = so.width * so.bpp;
    for (u32 y = 0; y < so.lines; ++y) {
        const size_t line_offset = so.fb_base + static_cast<size_t>(y) * line_bytes;
        if (line_offset + line_bytes > rdram_size) break;
        const u8* p = rdram + line_offset;
        u32* dst = out_pixels.data() + static_cast<size_t>(so.y0 + y) * out_w + so.x0;
        for (u32 x = 0; x < so.shown_w; ++x) {
            u32 r, g, b;
            if (so.bpp == 2) { // RGBA 5-5-5-1
                const u16 px = static_cast<u16>((p[x * 2] << 8) | p[x * 2 + 1]);
                r = ((px >> 11) & 0x1F) * 255 / 31;
                g = ((px >> 6) & 0x1F) * 255 / 31;
                b = ((px >> 1) & 0x1F) * 255 / 31;
            } else { // RGBA 8-8-8-8
                r = p[x * 4 + 0];
                g = p[x * 4 + 1];
                b = p[x * 4 + 2];
            }
            dst[x] = 0xFF000000u | (r << 16) | (g << 8) | b;
        }
    }
}
