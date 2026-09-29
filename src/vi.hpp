#pragma once

#include "common.hpp"

class MI;

// The frame buffer the VI scans out this frame.
struct VIScanout {
    bool blank = true;  // nothing displayed (blanked, or not set up yet)
    u32 fb_base = 0;    // physical address of the first displayed pixel
    u32 width = 320;    // pixels per line, which is also the line stride
    u32 bpp = 2;        // bytes per pixel: 2 = RGBA5551, 4 = RGBA8888
    // The part of the frame buffer on screen (VI_H/V_START scaled by
    // VI_X/Y_SCALE), and where it sits on the whole screen, which is
    // canvas_w x canvas_h frame buffer pixels: games with a smaller window
    // (Banjo-Kazooie) get black borders, 480-line games (Perfect Dark,
    // Rainbow Six) a canvas twice as tall.
    u32 lines = 240, shown_w = 320;
    u32 canvas_w = 320, canvas_h = 240;
    u32 x0 = 0, y0 = 0;

    // Copies an image of the frame buffer (src_w pixels a row, S times the
    // frame buffer's resolution) onto the screen canvas, S times as large.
    void place(const u32* src, u32 src_w, u32 S, std::vector<u32>& out, int& out_w, int& out_h) const;
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
    // Converts the part of the frame buffer on screen to ARGB8888, placed on
    // the whole screen (VIScanout::canvas_w x canvas_h).
    void render_frame(const u8* rdram, size_t rdram_size, std::vector<u32>& out_pixels, int& out_w, int& out_h) const;
    // The VI as the hardware scans out (vi_exact.cpp): coverage anti-aliasing
    // or the dither filter, the divot filter, bilinear scaling from
    // X/Y_SCALE, gamma and gamma dither, over the whole 640-wide picture
    // including its borders. `rdp` has the RDRAM's ninth bits (coverage);
    // without it they are what CPU writes leave. The picture comes out
    // 640 x 480 (576 PAL), each line of a progressive frame shown twice -
    // times the RDP's internal resolution when it has one: the VI then
    // scans the RDP's high-resolution copy of the frame buffer, with every
    // position scaled up (as parallel-rdp does; no fetch bug).
    void render_frame_exact(const u8* rdram, size_t rdram_size, class ExactRdp* rdp, std::vector<u32>& out_pixels,
                            int& out_w, int& out_h);

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
    // render_frame_exact()'s state (a separate, later section): the frame
    // counter and, after an interlaced frame, the picture the next field is
    // woven into - at native resolution (every scale-th pixel), so a state
    // doesn't depend on the internal resolution.
    template <class S> void serialize_exact(S& s) {
        s(exact_frames_);
        std::vector<u32> woven;
        if constexpr (!S::loading) woven = exact_native_picture();
        s(woven);
        if constexpr (S::loading) {
            exact_woven_ = std::move(woven);
            exact_serrate_ = false; // (until the next frame says)
        }
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

    // render_frame_exact(): the woven picture of interlaced modes, and a
    // frame counter (gamma dither noise, field parity).
    std::vector<u32> exact_picture_;
    u32 exact_frames_{0};
    int exact_scale_{1};        // of exact_picture_
    bool exact_serrate_{false}; // the last frame was interlaced
    std::vector<u32> exact_woven_; // from a loaded state, for the next frame
    std::vector<u32> exact_native_picture() const;
};
