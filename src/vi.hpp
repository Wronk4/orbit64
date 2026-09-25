#pragma once

#include "common.hpp"

class MI;

// The frame buffer the VI scans out this frame.
struct VIScanout {
    bool blank = true;  // nothing displayed (blanked, or not set up yet)
    u32 fb_base = 0;    // physical address of the first displayed pixel
    u32 width = 320;    // pixels per line, which is also the line stride
    u32 bpp = 2;        // bytes per pixel: 2 = RGBA5551, 4 = RGBA8888
};

class VI {
public:
    VI();

    void reset();

    u32 read_reg(u32 addr) const;
    void write_reg(u32 addr, u32 val, MI& mi);

    // Step scanline, trigger interrupt when v_current == v_intr
    bool step_scanline(MI& mi);

    VIScanout scanout(size_t rdram_size) const;
    // Converts the scanned-out frame buffer to ARGB8888, width x 240.
    void render_frame(const u8* rdram, size_t rdram_size, std::vector<u32>& out_pixels, int& out_w, int& out_h) const;

    u32 get_origin() const { return origin; }
    u32 get_status() const { return status; }
    u32 get_width() const { return width; }
    u32 get_h_start() const { return h_start; }
    u32 get_v_start() const { return v_start; }
    u32 get_v_sync() const { return v_sync; }
    // Field rate derived from VI_V_SYNC: ~525 lines = NTSC/MPAL 60 Hz, ~625 = PAL 50 Hz.
    u32 get_refresh_hz() const { return (v_sync >= 600) ? 50 : 60; }

    // Save states (savestate.hpp).
    template <class S> void serialize(S& s) {
        s(status, origin, width, v_intr, v_current, burst, v_sync, h_sync, leap, h_start, v_start, v_burst, x_scale,
          y_scale);
    }

private:
    u32 status{0};
    u32 origin{0};
    u32 width{320};
    u32 v_intr{2};
    u32 v_current{0};
    u32 burst{0};
    u32 v_sync{525};
    u32 h_sync{0};
    u32 leap{0};
    u32 h_start{0};
    u32 v_start{0};
    u32 v_burst{0};
    u32 x_scale{0x400};
    u32 y_scale{0x400};
};
