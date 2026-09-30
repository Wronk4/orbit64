#pragma once
// Pixel pipeline shared by the native RDP pass (which writes RDRAM) and the
// high-resolution pass (hires.cpp): texture sampling, colour combiner,
// blender and the triangle / texture-rectangle scan loops.
//
// Everything a draw reads from the RDP is captured in a DrawState, so the
// high-resolution pass can replay draws on worker threads while the RDP
// moves on. Both passes run the same code; at scale 1 the scan loops visit
// exactly the pixels, and compute exactly the values, the RDP always has.

#include "common.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>

#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define ORBIT64_RASTER_SSE2 1
#endif

struct Vertex {
    f32 x{0}, y{0}, z{0}, w{1.0f};      // Clip space position
    f32 sx{0}, sy{0}, sz{0};            // Screen space position
    f32 u{0}, v{0};                     // Texture coordinates
    u8 r{255}, g{255}, b{255}, a{255};  // Color
    u32 clip_flags{0};                  // Frustum clip flags for G_CULLDL
};

struct Tile {
    u8 format{0};     // RGBA=0, YUV=1, CI=2, IA=3, I=4
    u8 size{0};       // 4b=0, 8b=1, 16b=2, 32b=3
    u16 line{0};      // Line size in 64-bit words
    u16 tmem{0};      // TMEM address (offset in 64-bit words)
    u8 palette{0};
    u8 clamp_s{0}, mirror_s{0}, mask_s{0}, shift_s{0};
    u8 clamp_t{0}, mirror_t{0}, mask_t{0}, shift_t{0};
    u16 sl{0}, tl{0}, sh{0}, th{0}; // Tile coordinates (10.2 fixed point)

    // Save states (savestate.hpp): field by field, the struct has padding.
    template <class S> void serialize(S& s) {
        s(format, size, line, tmem, palette, clamp_s, mirror_s, mask_s, shift_s, clamp_t, mirror_t, mask_t, shift_t,
          sl, tl, sh, th);
    }
};

// The high-resolution renderer's buffers are 240 lines tall; the software
// RDP draws up to 480 (interlaced high-resolution screens, like Perfect
// Dark's and Rainbow Six's), which the VI then shows at native resolution.
constexpr u32 kFbLines = 240;
constexpr u32 kMaxFbLines = 480;

namespace raster {

// i / 255.0f for every byte value. Both are the correctly rounded quotient,
// so a lookup gives the same value as dividing per pixel.
inline constexpr std::array<f32, 256> kByteToUnit = [] {
    std::array<f32, 256> t{};
    for (int i = 0; i < 256; ++i) t[i] = static_cast<f32>(i) / 255.0f;
    return t;
}();

// fetch_texel() of one tile for every wrapped coordinate, used by the
// high-resolution pass, which samples each texel many times. The first
// thread that needs it fills it in; others read TMEM until it is ready.
struct TexCache {
    std::atomic<int> state{0}; // 0 empty, 1 being filled, 2 ready
    u32 w = 0, h = 0;          // wrapped S in [0, w), T in [0, h)
    u32* texels = nullptr;     // w * h, row-major
};

// A tile with everything the texture unit derives from it precomputed.
struct TexUnit {
    Tile tile{};
    f32 shift_mul_s{1.0f}, shift_mul_t{1.0f}; // SHIFT_S/T as a power-of-two factor
    f32 origin_s{0.0f}, origin_t{0.0f};       // SL/TL in texels
    s32 extent_s{0}, extent_t{0};             // clamp extent in texels
    u32 tmem_base{0};                         // byte offset in TMEM
    u32 row_stride{0};                        // bytes per texel row
    TexCache* cache{nullptr};                 // decoded texels (high-resolution pass only)
    // The texture coordinate unit's view of the tile (sample_texture):
    // clamping (CLAMP, or no MASK), mirroring, masks limited to 10 bits.
    bool hw_clamp_s{false}, hw_clamp_t{false};
    u8 hw_mask_s{0}, hw_mask_t{0};

    void prepare(const Tile& t) {
        tile = t;
        cache = nullptr; // decoded for the previous tile settings, if any
        // SHIFT 1..10 divides by 2^shift, 11..15 multiplies by 2^(16-shift).
        // Scaling by a power of two is exact, so a multiplier gives the same
        // result as the division.
        auto mul = [](u8 shift) {
            if (shift == 0) return 1.0f;
            if (shift > 10) return static_cast<f32>(1 << (16 - shift));
            return 1.0f / static_cast<f32>(1 << shift);
        };
        shift_mul_s = mul(t.shift_s);
        shift_mul_t = mul(t.shift_t);
        // The tile's upper-left corner is the TMEM origin, so texture
        // coordinates are relative to SL/TL.
        origin_s = static_cast<f32>(t.sl) / 4.0f;
        origin_t = static_cast<f32>(t.tl) / 4.0f;
        // A tile that never got a G_SETTILESIZE has a zero extent; clamping to
        // it would collapse the whole texture to texel 0, so fall back to the
        // MASK extent (or the full 10-bit range).
        extent_s = static_cast<s32>(t.sh / 4) - static_cast<s32>(t.sl / 4);
        extent_t = static_cast<s32>(t.th / 4) - static_cast<s32>(t.tl / 4);
        if (extent_s <= 0) extent_s = t.mask_s ? ((1 << t.mask_s) - 1) : 1023;
        if (extent_t <= 0) extent_t = t.mask_t ? ((1 << t.mask_t) - 1) : 1023;
        hw_mask_s = t.mask_s > 10 ? 10 : t.mask_s;
        hw_mask_t = t.mask_t > 10 ? 10 : t.mask_t;
        hw_clamp_s = t.clamp_s || t.mask_s == 0;
        hw_clamp_t = t.clamp_t || t.mask_t == 0;
        tmem_base = t.tmem * 8u;
        u32 texels_per_row = static_cast<u32>(extent_s + 1);
        u32 fallback_stride;
        switch (t.size) {
            case 0:  fallback_stride = (texels_per_row + 1) / 2; break; // 4-bit
            case 1:  fallback_stride = texels_per_row; break;           // 8-bit
            default: fallback_stride = texels_per_row * 2; break;       // 16/32-bit
        }
        row_stride = t.line ? (t.line * 8u) : fallback_stride;
    }
};

// The colour combiner's inputs (rows of per-channel values), as the
// hardware's muxes pick them.
enum CcIn : u8 {
    kCcComb, kCcT0, kCcT1, kCcPrim, kCcShade, kCcEnv, kCcOne, kCcNoise, kCcZero, kCcKeyCenter, kCcK4, kCcKeyScale,
    kCcCombA, kCcT0A, kCcT1A, kCcPrimA, kCcShadeA, kCcEnvA, kCcLod, kCcPrimLod, kCcK5, kCcCount
};

// Everything one draw reads from the RDP, plus the mode decoding derived
// from it. Built when the RDP state changes, then shared by the draws that
// follow (and copied for the high-resolution pass).
struct DrawState {
    u32 other_mode_h{0}, other_mode_l{0};
    u32 combine_w0{0}, combine_w1{0};
    u32 prim_color{0}, env_color{0}, blend_color{0}, fog_color{0};
    // gDPSetKeyR/GB (chroma key centre and scale per channel) and
    // gDPSetConvert's K4 and K5: combiner inputs.
    u8 key_center[3]{}, key_scale[3]{};
    u16 k4{0}, k5{0};
    u32 noise_seed{0}; // varies the combiner's NOISE input between draws
    u32 scissor_ulx{0}, scissor_uly{0}, scissor_lrx{0}, scissor_lry{0};
    u32 fb_addr{0};   // colour image address (physical)
    u32 fb_w{320};    // colour image width in pixels
    u8 fb_size{2};    // colour image pixel size: 2 = RGBA5551, 3 = RGBA8888
    bool combine_set{false};
    bool texture_enabled{false};
    bool smooth_shading{false};
    u32 active_tile{0};
    const u8* tmem{nullptr};       // 4 KB of TMEM the draw samples
    const bool* tmem_dxt{nullptr}; // 512 per-word "loaded with DXT 0" flags
    TexUnit tex[8];

    // ---- Derived by finalize()
    bool copy_mode{false}, two_cycle{false}, fill_or_copy{false};
    u32 tlut_type{0};          // G_MDSFT_TEXTLUT: 0 = none, 2 = RGBA16, 3 = IA16
    bool point_sample{true};   // G_TF_POINT, or COPY mode (never filters)
    bool mid_texel{false};     // G_TF_AVERAGE: exactly between four texels, their average
    bool bilerp0{true}, bilerp1{true}; // G_MDSFT_TEXTCONV: TEXEL0 / TEXEL1 filtered (else converted)
    // Whether the combiner can read TEXEL0 / TEXEL1 at all (conservative).
    bool need_tex0{true}, need_tex1{true};
    u8 cc_a0{0}, cc_b0{0}, cc_c0{0}, cc_d0{0}, ac_a0{0}, ac_b0{0}, ac_c0{0}, ac_d0{0};
    u8 cc_a1{0}, cc_b1{0}, cc_c1{0}, cc_d1{0}, ac_a1{0}, ac_b1{0}, ac_c1{0}, ac_d1{0};
    f32 prim_r{0}, prim_g{0}, prim_b{0}, prim_a{0};
    f32 env_r{0}, env_g{0}, env_b{0}, env_a{0};
    // The combiner as the hardware runs it: per cycle, the row of cc_in each
    // of A, B, C, D reads, per channel (r, g, b, a). 1-cycle mode runs the
    // second cycle's settings.
    u8 cc_sel[2][4][4]{};
    s32 cc_const[kCcCount][4]{}; // the rows that don't change per pixel
    // 1-cycle mode: TEXEL1 is the next pixel's TEXEL0 (the texture unit is a
    // pixel ahead), which a combiner can still read.
    bool pipelined_tex1{false};
    bool uses_noise{false};
    u8 bl_p{0}, bl_a{0}, bl_m{0}, bl_b{0};     // blender cycle 1 (the only one in 1-cycle mode)
    u8 bl2_p{0}, bl2_a{0}, bl2_m{0}, bl2_b{0}; // blender cycle 2
    bool blend_pass_through{true}; // an unblended pixel comes out of the blender as it went in
    bool blend_enabled{false};  // FORCE_BL / ZMODE_XLU / AA with CVG_X_ALPHA: blend when alpha < 255
    bool force_blend{false};    // FORCE_BL: P*A + M*B as it is; otherwise divided by A + B
    u8 alpha_compare{0};        // G_AC_*: 0 none, 1 threshold, 3 dither
    u8 alpha_threshold{0};      // blend colour alpha
    bool alpha_zero_kill{false}; // CVG_X_ALPHA/ALPHA_CVG_SEL/FORCE_BL/ZMODE_XLU/IM_RD/AA_EN: drop alpha-0 pixels
    // ALPHA_CVG_SEL without CVG_X_ALPHA: the pixel's alpha is its coverage
    // (full inside a primitive), not what the combiner worked out.
    bool alpha_from_cvg{false};
    bool z_compare{false}, z_update{false};
    bool z_decal{false}; // ZMODE_DEC: passes when about as deep as what is there
    // The memory stage as the hardware has it (pixel_backend()).
    bool aa_en{false}, image_read{false}, color_on_cvg{false}, cvg_times_alpha{false}, alpha_cvg_select{false};
    bool alpha_test{false}, alpha_test_dither{false};
    u8 z_mode{0};   // ZMODE: 0 opaque, 1 interpenetrating, 2 translucent, 3 decal
    u8 cvg_mode{0}; // CVG_DST: 0 clamp, 1 wrap, 2 full, 3 save
    // Texture level of detail: G_TEXTURE's level count, gDPSetPrimColor's
    // minimum level and fraction; derived: whether LOD is worked out at all.
    u8 max_level{0}, min_level{0}, prim_lod_frac{0};
    bool tex_lod_en{false}, sharpen{false}, detail{false}, dolod{false};

    void finalize() {
        u32 cycle_type = (other_mode_h >> 20) & 0x3;
        copy_mode = cycle_type == 2;
        two_cycle = cycle_type == 1;
        fill_or_copy = cycle_type == 2 || cycle_type == 3;
        tlut_type = (other_mode_h >> 14) & 0x3;
        point_sample = ((other_mode_h >> 13) & 1) == 0 || copy_mode;
        mid_texel = (other_mode_h >> 12) & 1;
        bilerp0 = (other_mode_h >> 11) & 1;
        bilerp1 = (other_mode_h >> 10) & 1;

        cc_a0 = (combine_w0 >> 20) & 0xF;
        cc_c0 = (combine_w0 >> 15) & 0x1F;
        ac_a0 = (combine_w0 >> 12) & 0x7;
        ac_c0 = (combine_w0 >> 9) & 0x7;
        cc_b0 = (combine_w1 >> 28) & 0xF;
        cc_d0 = (combine_w1 >> 15) & 0x7;
        ac_b0 = (combine_w1 >> 12) & 0x7;
        ac_d0 = (combine_w1 >> 9) & 0x7;
        cc_a1 = (combine_w0 >> 5) & 0xF;
        cc_c1 = (combine_w0 >> 0) & 0x1F;
        ac_a1 = (combine_w1 >> 21) & 0x7;
        ac_c1 = (combine_w1 >> 18) & 0x7;
        cc_b1 = (combine_w1 >> 24) & 0xF;
        cc_d1 = (combine_w1 >> 6) & 0x7;
        ac_b1 = (combine_w1 >> 3) & 0x7;
        ac_d1 = (combine_w1 >> 0) & 0x7;

        prim_r = ((prim_color >> 24) & 0xFF) / 255.0f;
        prim_g = ((prim_color >> 16) & 0xFF) / 255.0f;
        prim_b = ((prim_color >> 8) & 0xFF) / 255.0f;
        prim_a = (prim_color & 0xFF) / 255.0f;
        env_r = ((env_color >> 24) & 0xFF) / 255.0f;
        env_g = ((env_color >> 16) & 0xFF) / 255.0f;
        env_b = ((env_color >> 8) & 0xFF) / 255.0f;
        env_a = (env_color & 0xFF) / 255.0f;

        // Which texels each cycle reads (the second cycle sees them swapped:
        // its TEXEL0 is TEXEL1).
        auto reads = [](u8 a, u8 b, u8 c, u8 d, u8 aa, u8 ab, u8 ac, u8 ad, u8 t) {
            return a == t || b == t || c == t || d == t || c == t + 7 || aa == t || ab == t || ac == t || ad == t;
        };
        const bool c0_t0 = reads(cc_a0, cc_b0, cc_c0, cc_d0, ac_a0, ac_b0, ac_c0, ac_d0, 1);
        const bool c0_t1 = reads(cc_a0, cc_b0, cc_c0, cc_d0, ac_a0, ac_b0, ac_c0, ac_d0, 2);
        const bool c1_t0 = reads(cc_a1, cc_b1, cc_c1, cc_d1, ac_a1, ac_b1, ac_c1, ac_d1, 1);
        const bool c1_t1 = reads(cc_a1, cc_b1, cc_c1, cc_d1, ac_a1, ac_b1, ac_c1, ac_d1, 2);
        if (two_cycle) {
            need_tex0 = c0_t0 || c1_t1;
            need_tex1 = c0_t1 || c1_t0;
            pipelined_tex1 = false;
        } else {
            // TEXEL1 comes from the draw's own tile too, one pixel on.
            need_tex0 = c1_t0 || c1_t1;
            need_tex1 = false;
            pipelined_tex1 = c1_t1;
        }
        {
            static constexpr u8 kRgbA[16] = {kCcComb, kCcT0, kCcT1, kCcPrim, kCcShade, kCcEnv, kCcOne, kCcNoise,
                                             kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero};
            static constexpr u8 kRgbB[16] = {kCcComb, kCcT0, kCcT1, kCcPrim, kCcShade, kCcEnv, kCcKeyCenter, kCcK4,
                                             kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero};
            static constexpr u8 kRgbC[32] = {kCcComb, kCcT0, kCcT1, kCcPrim, kCcShade, kCcEnv, kCcKeyScale, kCcCombA,
                                             kCcT0A, kCcT1A, kCcPrimA, kCcShadeA, kCcEnvA, kCcLod, kCcPrimLod, kCcK5,
                                             kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero,
                                             kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero, kCcZero};
            static constexpr u8 kRgbD[8] = {kCcComb, kCcT0, kCcT1, kCcPrim, kCcShade, kCcEnv, kCcOne, kCcZero};
            static constexpr u8 kAlphaAbd[8] = {kCcComb, kCcT0, kCcT1, kCcPrim, kCcShade, kCcEnv, kCcOne, kCcZero};
            static constexpr u8 kAlphaC[8] = {kCcLod, kCcT0, kCcT1, kCcPrim, kCcShade, kCcEnv, kCcPrimLod, kCcZero};
            const u8 m[2][8] = {{cc_a0, cc_b0, cc_c0, cc_d0, ac_a0, ac_b0, ac_c0, ac_d0},
                                {cc_a1, cc_b1, cc_c1, cc_d1, ac_a1, ac_b1, ac_c1, ac_d1}};
            uses_noise = false;
            for (int c = 0; c < 2; ++c) {
                for (int i = 0; i < 3; ++i) {
                    cc_sel[c][0][i] = kRgbA[m[c][0] & 15];
                    cc_sel[c][1][i] = kRgbB[m[c][1] & 15];
                    cc_sel[c][2][i] = kRgbC[m[c][2] & 31];
                    cc_sel[c][3][i] = kRgbD[m[c][3] & 7];
                }
                cc_sel[c][0][3] = kAlphaAbd[m[c][4] & 7];
                cc_sel[c][1][3] = kAlphaAbd[m[c][5] & 7];
                cc_sel[c][2][3] = kAlphaC[m[c][6] & 7];
                cc_sel[c][3][3] = kAlphaAbd[m[c][7] & 7];
                if ((c == 1 || two_cycle) && cc_sel[c][0][0] == kCcNoise) uses_noise = true;
            }
            for (int i = 0; i < 4; ++i) {
                cc_const[kCcPrim][i] = static_cast<s32>((prim_color >> (24 - 8 * i)) & 0xFF);
                cc_const[kCcEnv][i] = static_cast<s32>((env_color >> (24 - 8 * i)) & 0xFF);
                cc_const[kCcOne][i] = 0x100;
                cc_const[kCcZero][i] = 0;
                cc_const[kCcKeyCenter][i] = i < 3 ? key_center[i] : 0;
                cc_const[kCcK4][i] = k4 & 0x1FF;
                cc_const[kCcKeyScale][i] = i < 3 ? key_scale[i] : 0;
                cc_const[kCcPrimA][i] = static_cast<s32>(prim_color & 0xFF);
                cc_const[kCcEnvA][i] = static_cast<s32>(env_color & 0xFF);
                cc_const[kCcPrimLod][i] = prim_lod_frac;
                cc_const[kCcK5][i] = k5 & 0x1FF;
            }
        }

        // G_MDSFT_BLENDER=16: the first cycle's P/A/M/B at bits 30/26/22/18,
        // the second cycle's (2-cycle mode) at 28/24/20/16.
        bl_p = (other_mode_l >> 30) & 0x3;
        bl_a = (other_mode_l >> 26) & 0x3;
        bl_m = (other_mode_l >> 22) & 0x3;
        bl_b = (other_mode_l >> 18) & 0x3;
        bl2_p = (other_mode_l >> 28) & 0x3;
        bl2_a = (other_mode_l >> 24) & 0x3;
        bl2_m = (other_mode_l >> 20) & 0x3;
        bl2_b = (other_mode_l >> 16) & 0x3;
        // In 2-cycle mode the first cycle always blends (that is where fog
        // goes in); IN*0 + IN*1 and IN*A + IN*(1-A) change nothing.
        const bool first_identity = bl_p == 0 && bl_m == 0 && ((bl_a == 3 && bl_b == 2) || bl_b == 0);
        blend_pass_through = two_cycle ? (first_identity && bl2_p == 0) : bl_p == 0;
        // FORCE_BL / ZMODE_XLU, or anti-aliasing with coverage times alpha:
        // the texture's alpha is the coverage of edge pixels, which the
        // blender mixes with what is behind them (billboards' soft edges).
        force_blend = (other_mode_l & 0x4000) != 0;
        blend_enabled = (other_mode_l & 0x4800) != 0 || (other_mode_l & 0x1008) == 0x1008;
        alpha_compare = other_mode_l & 0x3;
        // COPY mode ignores the blend colour: a texel passes the alpha test unless its alpha
        // is 0 (Hydro Thunder's menu sprites rely on it; RGBA16 palette entries have 0 or 255).
        alpha_threshold = copy_mode ? 1 : (blend_color & 0xFF);
        alpha_zero_kill = (other_mode_l & 0x7848) != 0;
        alpha_from_cvg = (other_mode_l & 0x3000) == 0x2000;
        aa_en = (other_mode_l >> 3) & 1;
        image_read = (other_mode_l >> 6) & 1;
        color_on_cvg = (other_mode_l >> 7) & 1;
        cvg_mode = (other_mode_l >> 8) & 3;
        z_mode = (other_mode_l >> 10) & 3;
        cvg_times_alpha = (other_mode_l >> 12) & 1;
        alpha_cvg_select = (other_mode_l >> 13) & 1;
        alpha_test = other_mode_l & 1;
        alpha_test_dither = (other_mode_l >> 1) & 1;
        z_compare = (other_mode_l & 0x10) != 0;
        z_update = (other_mode_l & 0x20) != 0;
        z_decal = ((other_mode_l >> 10) & 3) == 3;

        tex_lod_en = (other_mode_h >> 16) & 1;
        sharpen = (other_mode_h >> 17) & 1;
        detail = (other_mode_h >> 18) & 1;
        // The combiner's LOD_FRACTION: colour C 13, alpha C 0.
        const bool uses_lf = cc_c0 == 13 || ac_c0 == 0 || (two_cycle && (cc_c1 == 13 || ac_c1 == 0));
        dolod = !fill_or_copy && (tex_lod_en || uses_lf);
    }
};

// The texture unit's level of detail for a pixel whose texture coordinates
// change by at most `delta` texels to the next pixel and the next row: which
// tiles TEXEL0 and TEXEL1 come from (mip-mapping, from the draw's tile on)
// and the LOD fraction the combiner sees (0..255, +256 while sharpening).
inline void lod_tiles(const DrawState& st, f32 delta, u32 prim_tile, u32& t0, u32& t1, s32& lf) {
    // In 1/32 texels (the coordinates' 10.5 fixed point); 0x4000 and up is
    // too far away to have a level.
    const s32 lod = delta >= 512.0f ? 0x4000 : static_cast<s32>(delta * 32.0f);
    bool magnify, distant;
    u32 l_tile = 0;
    if (lod & 0x4000) {
        magnify = false;
        distant = true;
        lf = 0xFF;
    } else if (lod < 32 || lod < st.min_level) { // magnified: less than a texel per pixel
        magnify = true;
        distant = st.max_level == 0;
        if (!st.sharpen && !st.detail) {
            lf = distant ? 0xFF : 0;
        } else {
            lf = (lod < st.min_level ? st.min_level : lod) << 3;
            if (st.sharpen) lf |= 0x100;
        }
    } else {
        magnify = false;
        const u32 texels = static_cast<u32>(lod >> 5) & 0xFF;
        l_tile = texels ? 31 - static_cast<u32>(__builtin_clz(texels)) : 0; // floor(log2)
        distant = st.max_level == 0 || (lod & 0x6000) || l_tile >= st.max_level;
        lf = (!st.sharpen && !st.detail && distant) ? 0xFF : ((lod << 3) >> l_tile) & 0xFF;
    }
    if (!st.tex_lod_en) {
        t0 = prim_tile & 7;
        t1 = (prim_tile + 1) & 7;
        return;
    }
    if (distant) l_tile = st.max_level;
    if (!st.detail) {
        t0 = (prim_tile + l_tile) & 7;
        t1 = (distant || (!st.sharpen && magnify)) ? t0 : ((t0 + 1) & 7);
    } else {
        t0 = (prim_tile + l_tile + (magnify ? 0 : 1)) & 7;
        t1 = (prim_tile + l_tile + ((!distant && !magnify) ? 2 : 1)) & 7;
    }
}

// The depth test. Decals (ZMODE_DEC) are drawn on a surface at about its
// depth: they pass within a small distance either way.
constexpr f32 kDecalDepth = 2.0f / 1023.0f;
inline bool depth_fails(const DrawState& st, f32 z, f32 stored) {
    if (!st.z_compare) return false;
    if (st.z_decal) return std::fabs(z - stored) > kDecalDepth;
    return z > stored;
}

// 5-bit texel channel -> 8 bits, as the texture unit expands it: the upper
// bits repeated below.
inline constexpr std::array<u8, 32> kFiveToEight = [] {
    std::array<u8, 32> t{};
    for (int c = 0; c < 32; ++c) t[c] = static_cast<u8>((c << 3) | (c >> 2));
    return t;
}();

// RGBA5551 -> ARGB8888. The single alpha bit is the texel's coverage flag.
inline u32 rgba16_to_rgba32(u16 p) {
    u8 r = kFiveToEight[(p >> 11) & 0x1F];
    u8 g = kFiveToEight[(p >> 6) & 0x1F];
    u8 b = kFiveToEight[(p >> 1) & 0x1F];
    u8 a = (p & 1) ? 255 : 0;
    return (static_cast<u32>(a) << 24) | (static_cast<u32>(r) << 16) | (static_cast<u32>(g) << 8) | b;
}

// Palette fetch for CI4/CI8. The TLUT lives in the upper half of TMEM (0x800).
// tlut_type is the G_MDSFT_TEXTLUT field: G_TT_NONE=0, G_TT_RGBA16=2, G_TT_IA16=3.
// IA16 palettes are what give CI textures a real 8-bit alpha ramp; decoding them
// as RGBA5551 collapses alpha to the low colour bit and destroys the gradient.
inline u32 lookup_tlut(const u8* tmem, u32 index, u32 tlut_type) {
    u32 off = 0x800 + index * 2;
    if (off + 1 >= 4096) return 0;
    u16 p = (static_cast<u16>(tmem[off]) << 8) | static_cast<u16>(tmem[off + 1]);
    if (tlut_type == 3) { // G_TT_IA16
        u8 i = (p >> 8) & 0xFF;
        u8 a = p & 0xFF;
        return (static_cast<u32>(a) << 24) | (static_cast<u32>(i) << 16) | (static_cast<u32>(i) << 8) | i;
    }
    return rgba16_to_rgba32(p);
}

// One axis of the texture coordinate unit: the hardware wraps/mirrors with
// MASK and clamps to the tile extent. With MASK==0 there is no power-of-two
// wrap at all, so the coordinate is always clamped. (For a power-of-two
// period, masking is the non-negative remainder.)
inline s32 wrap_texel_coord(s32 c, u8 mask, bool clamp_en, bool mirror_en, s32 extent) {
    if (mask == 0) {
        return std::clamp(c, 0, extent > 0 ? extent : 0);
    }
    if (clamp_en) {
        c = std::clamp(c, 0, extent > 0 ? extent : 0);
    }
    s32 period = 1 << mask;
    if (mirror_en) {
        s32 span = period * 2;
        c &= span - 1;
        if (c >= period) c = span - 1 - c;
        return c;
    }
    return c & (period - 1);
}

inline s32 wrap_s(const TexUnit& tu, s32 c) {
    return wrap_texel_coord(c, tu.tile.mask_s, tu.tile.clamp_s != 0, tu.tile.mirror_s != 0, tu.extent_s);
}
inline s32 wrap_t(const TexUnit& tu, s32 c) {
    return wrap_texel_coord(c, tu.tile.mask_t, tu.tile.clamp_t != 0, tu.tile.mirror_t != 0, tu.extent_t);
}

// Texel at an already wrapped coordinate.
inline u32 fetch_wrapped(const TexUnit& tu, const u8* tmem, const bool* tmem_dxt, u32 tlut_type, s32 is, s32 it) {
    const Tile& tile = tu.tile;
    const u32 tmem_base = tu.tmem_base;
    const u32 row_stride = tu.row_stride;
    constexpr u32 kTmemSize = 4096;

    if (tile.size == 3) { // 32-bit RGBA (bank 0 = RG, bank 1 = BA)
        u32 offset = tmem_base + (it * row_stride + is * 2);
        u32 word_addr = offset / 8;
        if ((it & 1) && word_addr < 512 && tmem_dxt[word_addr]) {
            offset ^= 4;
        }
        if (offset + 0x801 < kTmemSize) {
            u8 r = tmem[offset + 0];
            u8 g = tmem[offset + 1];
            u8 b = tmem[offset + 0x800 + 0];
            u8 a = tmem[offset + 0x800 + 1];
            return (static_cast<u32>(a) << 24) | (static_cast<u32>(r) << 16) | (static_cast<u32>(g) << 8) | b;
        }
    } else if (tile.size == 2) { // 16-bit
        u32 offset = tmem_base + (it * row_stride + is * 2);
        u32 word_addr = offset / 8;
        if ((it & 1) && word_addr < 512 && tmem_dxt[word_addr]) {
            offset ^= 4;
        }
        if (offset + 1 < kTmemSize) {
            u16 p = (static_cast<u16>(tmem[offset]) << 8) | static_cast<u16>(tmem[offset + 1]);
            if (tile.format == 3) { // IA16: 8-bit I + 8-bit A
                u8 i = (p >> 8) & 0xFF;
                u8 a = p & 0xFF;
                return (static_cast<u32>(a) << 24) | (static_cast<u32>(i) << 16) | (static_cast<u32>(i) << 8) | i;
            }
            return rgba16_to_rgba32(p);
        }
    } else if (tile.size == 1) { // 8-bit
        u32 offset = tmem_base + (it * row_stride + is);
        u32 word_addr = offset / 8;
        if ((it & 1) && word_addr < 512 && tmem_dxt[word_addr]) {
            offset ^= 4;
        }
        if (offset < kTmemSize) {
            u8 val = tmem[offset];
            // With TLUT enabled the hardware looks every 4/8-bit texel up in the palette,
            // whatever the tile's format field says (Madden 2000's grass is RGBA with TLUT on).
            if (tile.format == 2 || tlut_type != 0) { // CI8: full 256-entry TLUT
                if (tlut_type == 0) return 0xFF000000 | (val << 16) | (val << 8) | val;
                return lookup_tlut(tmem, val, tlut_type);
            } else if (tile.format == 3) { // IA8: 4 bits intensity, 4 bits alpha
                u8 i = static_cast<u8>(((val >> 4) & 0xF) * 17);
                u8 a = static_cast<u8>((val & 0xF) * 17);
                return (static_cast<u32>(a) << 24) | (static_cast<u32>(i) << 16) | (static_cast<u32>(i) << 8) | i;
            }
            // I8: the texture unit replicates intensity into alpha.
            return (static_cast<u32>(val) << 24) | (val << 16) | (val << 8) | val;
        }
    } else if (tile.size == 0) { // 4-bit
        u32 offset = tmem_base + (it * row_stride + (is / 2));
        u32 word_addr = offset / 8;
        if ((it & 1) && word_addr < 512 && tmem_dxt[word_addr]) {
            offset ^= 4;
        }
        if (offset < kTmemSize) {
            u8 byte_val = tmem[offset];
            u8 val = (is & 1) ? (byte_val & 0xF) : ((byte_val >> 4) & 0xF);
            if (tile.format == 2 || tlut_type != 0) { // CI4: 16-entry sub-palette
                if (tlut_type == 0) {
                    u8 i = static_cast<u8>(val * 17);
                    return (static_cast<u32>(i) << 24) | (i << 16) | (i << 8) | i;
                }
                return lookup_tlut(tmem, tile.palette * 16 + val, tlut_type);
            } else if (tile.format == 3) { // IA4: 3 bits intensity, 1 bit alpha
                u8 i = static_cast<u8>(((val >> 1) & 0x7) * 255 / 7);
                u8 a = (val & 1) ? 255 : 0;
                return (static_cast<u32>(a) << 24) | (static_cast<u32>(i) << 16) | (static_cast<u32>(i) << 8) | i;
            }
            // I4: intensity also drives alpha.
            u8 i = static_cast<u8>(val * 17);
            return (static_cast<u32>(i) << 24) | (i << 16) | (i << 8) | i;
        }
    }
    return 0;
}

inline u32 fetch_texel(const TexUnit& tu, const u8* tmem, const bool* tmem_dxt, u32 tlut_type, s32 is, s32 it) {
    return fetch_wrapped(tu, tmem, tmem_dxt, tlut_type, wrap_s(tu, is), wrap_t(tu, it));
}

// Size of the table a TexCache needs for a tile: the texel coordinates
// sample_texture() reads without masking (the clamped range plus the next
// texel), or the mask's period. Anything else is fetched from TMEM.
inline void tex_cache_dims(const TexUnit& tu, u32& w, u32& h) {
    w = tu.hw_mask_s ? (1u << tu.hw_mask_s) : static_cast<u32>(((tu.tile.sh >> 2) - (tu.tile.sl >> 2)) & 0x3FF) + 2;
    h = tu.hw_mask_t ? (1u << tu.hw_mask_t) : static_cast<u32>(((tu.tile.th >> 2) - (tu.tile.tl >> 2)) & 0x3FF) + 2;
}

// Identifies what a tile's TexCache holds, together with the TMEM contents
// (whoever keys caches adds a TMEM generation): every TexUnit/DrawState field
// fetch_wrapped() and wrap_s/wrap_t read.
inline void tex_cache_key(const TexUnit& tu, u32 tlut_type, u64& a, u64& b) {
    const Tile& t = tu.tile;
    a = static_cast<u64>(t.format) | static_cast<u64>(t.size) << 3 | static_cast<u64>(t.palette) << 5 |
        static_cast<u64>(t.mask_s) << 9 | static_cast<u64>(t.mask_t) << 13 | static_cast<u64>(t.clamp_s) << 17 |
        static_cast<u64>(t.clamp_t) << 18 | static_cast<u64>(t.mirror_s) << 19 | static_cast<u64>(t.mirror_t) << 20 |
        static_cast<u64>(tlut_type) << 21 | static_cast<u64>(tu.row_stride) << 23 | static_cast<u64>(tu.tmem_base) << 40;
    b = static_cast<u64>(static_cast<u32>(tu.extent_s)) | static_cast<u64>(static_cast<u32>(tu.extent_t)) << 32;
}

// The tile's decoded texels, decoding them now if no thread has started to;
// nullptr while another thread is still at it (or for tiles without a cache).
inline const u32* tex_table(const DrawState& st, const TexUnit& tu) {
    TexCache* c = tu.cache;
    if (!c) return nullptr;
    int state = c->state.load(std::memory_order_acquire);
    if (state == 2) return c->texels;
    if (state != 0 || !c->state.compare_exchange_strong(state, 1, std::memory_order_acquire)) return nullptr;
    for (u32 t = 0; t < c->h; ++t)
        for (u32 s = 0; s < c->w; ++s)
            c->texels[t * c->w + s] = fetch_wrapped(tu, st.tmem, st.tmem_dxt, st.tlut_type, static_cast<s32>(s), static_cast<s32>(t));
    c->state.store(2, std::memory_order_release);
    return c->texels;
}

// static_cast<s32>(std::floor(v)), without the libm-style sequence baseline
// x86-64 (no SSE4.1 ROUNDSS) needs for floor: truncate, then step down for
// negative non-integers. Identical for every input, including out-of-range
// ones and NaN, which both forms turn into INT32_MIN.
inline s32 floor_to_s32(f32 v) {
    const s32 i = static_cast<s32>(v);
    return (i != INT32_MIN && v < static_cast<f32>(i)) ? i - 1 : i;
}

// The texture coordinate unit, one axis: a coordinate in texels becomes
// the RDP's s10.5 (saturated to 16 bits), is shifted (SHIFT_S/T), clamped to
// the tile (CLAMP, or no MASK) and made relative to its upper-left corner.
// Returns the coordinate in 1/32 texels.
inline s32 tex_coord(f32 c, u8 shift, bool clamp, u16 lo, u16 hi) {
    const f32 v = c * 32.0f;
    s32 x = !(v > -32768.0f) ? -0x8000 : v >= 32767.0f ? 0x7FFF : floor_to_s32(v);
    if (shift < 11) x >>= shift;
    else x = static_cast<s32>(static_cast<u32>(x) << (32 - shift)) >> 16;
    const s32 l = lo, h = hi;
    if (clamp) {
        if ((x >> 3) >= h) return (((h >> 2) - (l >> 2)) & 0x3FF) << 5;
        return std::max(x - (l << 3), 0);
    }
    return x - (l << 3);
}
// MASK_S/T (with MIRROR) on a texel coordinate.
inline s32 tex_mask(s32 c, u8 mask, bool mirror) {
    if (mask == 0) return c;
    const s32 m = 1 << mask;
    if (mirror) c ^= std::max((c & m) - 1, 0);
    return c & (m - 1);
}

// The texel at masked coordinates (s, t): from the tile's decoded table
// when it has one and covers them, else from TMEM.
inline u32 texel_at(const DrawState& st, const TexUnit& tu, const u32* table, s32 s, s32 t) {
    if (table && static_cast<u32>(s) < tu.cache->w && static_cast<u32>(t) < tu.cache->h)
        return table[static_cast<u32>(t) * tu.cache->w + static_cast<u32>(s)];
    return fetch_wrapped(tu, st.tmem, st.tmem_dxt, st.tlut_type, s, t);
}

// The texture unit: TEXEL0 (or TEXEL1: `second`) of tile `tile_idx` at
// (s, t) in texels, as the RDP filters it: nearest, the 3-texel triangle
// filter (BILERP; it blends the texel with its right and lower
// neighbours, or - past the diagonal - the lower-right one with those two)
// or, in AVERAGE mode, a box filter exactly between four texels.
inline u32 sample_texture(const DrawState& st, u32 tile_idx, f32 s, f32 t, bool second = false) {
    const TexUnit& tu = st.tex[tile_idx & 0x7];
    const Tile& tile = tu.tile;
    const s32 cs = tex_coord(s, tile.shift_s, tu.hw_clamp_s, tile.sl, tile.sh);
    const s32 ct = tex_coord(t, tile.shift_t, tu.hw_clamp_t, tile.tl, tile.th);
    const u32* table = tex_table(st, tu);
    const s32 is = cs >> 5, it = ct >> 5;
    const s32 s0 = tex_mask(is, tu.hw_mask_s, tile.mirror_s), t0r = tex_mask(it, tu.hw_mask_t, tile.mirror_t);
    if (st.point_sample) return texel_at(st, tu, table, s0, t0r & 0xFF);

    const s32 fs = cs & 31, ft = ct & 31;
    const s32 s1 = tex_mask(is + 1, tu.hw_mask_s, tile.mirror_s);
    const s32 t1r = tex_mask(it + 1, tu.hw_mask_t, tile.mirror_t);
    // Rows are 8 bits wide past the masks; the next row follows the first.
    const s32 t0 = t0r & 0xFF, t1 = t0 + std::max(t1r - t0r, -255);
    const bool bilerp = second ? st.bilerp1 : st.bilerp0;
    const bool upper = fs + ft >= 32;
    const u32 c10 = texel_at(st, tu, table, s1, t0);
    const u32 c01 = texel_at(st, tu, table, s0, t1);
    if (!bilerp) return upper ? texel_at(st, tu, table, s1, t1) : texel_at(st, tu, table, s0, t0);
    if (st.mid_texel && fs == 16 && ft == 16) {
        const u32 c00 = texel_at(st, tu, table, s0, t0), c11 = texel_at(st, tu, table, s1, t1);
        u32 out = 0;
        for (int sh = 0; sh < 32; sh += 8) {
            const u32 sum = ((c00 >> sh) & 0xFF) + ((c10 >> sh) & 0xFF) + ((c01 >> sh) & 0xFF) + ((c11 >> sh) & 0xFF);
            out |= ((sum + 2) >> 2) << sh;
        }
        return out;
    }
    const u32 base = upper ? texel_at(st, tu, table, s1, t1) : texel_at(st, tu, table, s0, t0);
    const s32 f0 = upper ? 32 - ft : fs, f1 = upper ? 32 - fs : ft;
    u32 out = 0;
    for (int sh = 0; sh < 32; sh += 8) {
        const s32 b = static_cast<s32>((base >> sh) & 0xFF);
        const s32 v = (((static_cast<s32>((c10 >> sh) & 0xFF) - b) * f0 + (static_cast<s32>((c01 >> sh) & 0xFF) - b) * f1 + 0x10) >> 5) + b;
        out |= static_cast<u32>(v & 0xFF) << sh;
    }
    return out;
}

// The combiner's arithmetic: 9-bit signed inputs (A, B and D wrap around
// 0x80..0x17F the way the hardware's special expansion does), the product
// rounded, not clamped until the last cycle.
inline s32 cc_sext9(s32 v) { return static_cast<s32>(static_cast<u32>(v) << 23) >> 23; }
inline s32 cc_expand(s32 v) { return cc_sext9(v - 0x80) + 0x80; }
inline s32 cc_equation(s32 a, s32 b, s32 c, s32 d) {
    const s32 p = static_cast<s32>(static_cast<u32>(cc_expand(a) - cc_expand(b)) * static_cast<u32>(cc_sext9(c)) + 0x80u);
    return (p >> 8) + cc_expand(d);
}
inline s32 cc_clamp(s32 v) { return std::clamp(cc_sext9(v - 0x80) + 0x80, 0, 0xFF); }

// The combiner's NOISE input for a pixel (the hardware's comes from a free
// running generator; any per-pixel pseudo-random value looks the same).
inline u32 pixel_noise(const DrawState& st, u32 x, u32 y) {
    u32 h = x * 0x9E3779B1u ^ (y + st.noise_seed) * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h;
}
inline s32 cc_noise(const DrawState& st, u32 x, u32 y) { return static_cast<s32>(((pixel_noise(st, x, y) & 7) << 6) | 0x20); }

// Colour combiner: (A - B) * C + D per channel, once or twice (2-cycle), in
// the hardware's integer arithmetic. tex0/tex1 are the pixel's TEXEL0 and
// TEXEL1 (in 1-cycle mode TEXEL1 is the next pixel's TEXEL0, see
// DrawState::pipelined_tex1). Returns the clamped result (ARGB); `alpha0`
// gets the first cycle's clamped alpha in 2-cycle mode (the alpha compare's).
inline u32 combine(const DrawState& st, u32 tex0, u32 tex1, u8 sr, u8 sg, u8 sb, u8 sa, s32 lod_frac = 0,
                   s32 noise = 0, u8* alpha0 = nullptr) {
    // The rows that change per pixel; the others point into the draw state.
    s32 comb[4] = {0, 0, 0, 0}, comb_a[4] = {0, 0, 0, 0};
    s32 t0[4], t1[4], t0a[4], t1a[4];
    const s32 shade[4] = {sr, sg, sb, sa}, shade_a[4] = {sa, sa, sa, sa};
    const s32 nz[4] = {noise, noise, noise, noise}, lod[4] = {lod_frac, lod_frac, lod_frac, lod_frac};
    const s32* row[kCcCount];
    for (int i = 0; i < kCcCount; ++i) row[i] = st.cc_const[i];
    row[kCcComb] = comb;
    row[kCcCombA] = comb_a;
    row[kCcT0] = t0;
    row[kCcT1] = t1;
    row[kCcT0A] = t0a;
    row[kCcT1A] = t1a;
    row[kCcShade] = shade;
    row[kCcShadeA] = shade_a;
    row[kCcNoise] = nz;
    row[kCcLod] = lod;
    auto set_texel = [](s32* c, s32* a, u32 t) {
        c[0] = (t >> 16) & 0xFF;
        c[1] = (t >> 8) & 0xFF;
        c[2] = t & 0xFF;
        c[3] = static_cast<s32>(t >> 24);
        a[0] = a[1] = a[2] = a[3] = c[3];
    };
    auto cycle = [&](int c, s32 out[4]) {
        const u8 (*sel)[4] = st.cc_sel[c];
        for (int i = 0; i < 4; ++i)
            out[i] = cc_equation(row[sel[0][i]][i], row[sel[1][i]][i], row[sel[2][i]][i], row[sel[3][i]][i]);
    };
    set_texel(t0, t0a, tex0);
    set_texel(t1, t1a, tex1);
    s32 out[4];
    if (st.two_cycle) {
        s32 c0[4];
        cycle(0, c0);
        if (alpha0) *alpha0 = static_cast<u8>(cc_clamp(c0[3]));
        // The second cycle sees the texels swapped (the texture unit's pipelining).
        row[kCcT0] = t1;
        row[kCcT1] = t0;
        row[kCcT0A] = t1a;
        row[kCcT1A] = t0a;
        for (int i = 0; i < 4; ++i) {
            comb[i] = c0[i];
            comb_a[i] = c0[3];
        }
        cycle(1, out);
    } else {
        cycle(1, out);
    }
    return (static_cast<u32>(cc_clamp(out[3])) << 24) | (static_cast<u32>(cc_clamp(out[0])) << 16) |
           (static_cast<u32>(cc_clamp(out[1])) << 8) | static_cast<u32>(cc_clamp(out[2]));
}

// What a pixel brings to the memory stage besides its colour.
struct PixelAux {
    u8 shade_a{0xFF}; // the blender's A_SHADE (shade alpha)
    u8 cvg{8};        // how many of its 8 coverage samples the primitive covers
    s16 alpha0{-1};   // 2-cycle mode: the first cycle's alpha, which the alpha compare sees
    u8 noise{0};      // the dithered alpha compare's threshold
};
// The frame buffer pixel as the blender reads it: 8-bit channels (the upper
// 5 bits of a 16-bit pixel) and its coverage (0..7).
struct MemPixel {
    s32 r{0}, g{0}, b{0};
    s32 cvg{7};
};
struct PixelResult {
    u8 r{0}, g{0}, b{0};
    u8 cvg{7};         // coverage to store with the colour
    bool z_write{false};
};

// The blender's divider (4-bit denominator, 11-bit numerator): a
// non-restoring division as the hardware carries it out.
inline const u8* blender_divider() {
    static const std::array<u8, 0x8000> lut = [] {
        std::array<u8, 0x8000> t{};
        for (int i = 0; i < 0x8000; ++i) {
            const int d = (i >> 11) & 0xF, n = i & 0x7FF, invd = ~d & 0xF;
            int res = 0;
            int partial = (invd + (n >> 8) + 1) & 7;
            for (int k = 0; k < 8; ++k) {
                const int nbit = (n >> (7 - k)) & 1;
                const int sum = (res & (0x100 >> k)) ? invd + (partial << 1) + nbit + 1 : d + (partial << 1) + nbit;
                partial = sum & 7;
                if (sum & 0x10) res |= 1 << (7 - k);
            }
            t[i] = static_cast<u8>(res);
        }
        return t;
    }();
    return lut.data();
}

// Depth slack of the tests that allow for the primitives' depth slopes
// (nearer/farther), in the units of Vertex::sz (1023 = the viewport's range).
constexpr f32 kDepthSlack = 1.0f / (1023.0f * 32.0f);
// What the depth buffer holds where nothing has been drawn since it was cleared.
constexpr f32 kDepthCleared = 1e30f;

// The memory stage of a 1- or 2-cycle pixel (not FILL; COPY writes the
// texel as it is): coverage and alpha, the alpha compare, the depth test,
// the blender (both cycles) and the coverage stored with the colour, in the
// hardware's integer arithmetic. `color` is the combiner's output (ARGB),
// `mem_z` the depth buffer's value. False when the pixel is not written.
inline bool pixel_backend(const DrawState& st, u32 color, const PixelAux& aux, f32 z, f32 mem_z, const MemPixel& mem,
                          PixelResult& out) {
    const s32 in_r = (color >> 16) & 0xFF, in_g = (color >> 8) & 0xFF, in_b = color & 0xFF, in_a = color >> 24;
    if (st.copy_mode) {
        if (st.alpha_test && in_a == 0) return false;
        out.r = static_cast<u8>(in_r);
        out.g = static_cast<u8>(in_g);
        out.b = static_cast<u8>(in_b);
        out.cvg = in_a ? 7 : 0;
        out.z_write = false;
        return true;
    }
    // Coverage and alpha
    s32 cvg = aux.cvg;
    const s32 expanded = in_a + ((in_a + 1) >> 8);
    s32 modulated;
    if (st.cvg_times_alpha) {
        modulated = (expanded * cvg + 4) >> 3;
        cvg = modulated >> 5;
    } else {
        modulated = cvg << 5;
    }
    const s32 alpha = std::clamp(st.alpha_cvg_select ? modulated : expanded, 0, 0xFF);
    if (st.aa_en && cvg == 0) return false;
    if (st.alpha_test) {
        s32 ref = alpha;
        if (st.two_cycle && aux.alpha0 >= 0) {
            s32 ea = aux.alpha0 + ((aux.alpha0 + 1) >> 8);
            if (st.alpha_cvg_select) ea = st.cvg_times_alpha ? (ea * aux.cvg + 4) >> 3 : aux.cvg << 5;
            ref = std::clamp(ea, 0, 0xFF);
        }
        const s32 threshold = st.alpha_test_dither ? aux.noise : static_cast<s32>(st.blend_color & 0xFF);
        if (ref < threshold) return false;
    }

    // Depth
    const s32 mem_cvg = st.image_read ? mem.cvg : 7;
    const bool overflow = cvg + mem_cvg >= 8;
    bool blend_en;
    if (st.z_compare) {
        const bool max_z = mem_z >= kDepthCleared;
        const bool front = z < mem_z;
        const f32 slack = st.z_mode == 3 ? kDecalDepth : kDepthSlack;
        const bool farther = z + slack >= mem_z, nearer = z - slack <= mem_z;
        blend_en = st.force_blend || (!overflow && st.aa_en && farther);
        bool pass;
        switch (st.z_mode) {
            case 0: case 1: pass = max_z || (overflow ? front : nearer); break;
            case 2: pass = front || max_z; break;
            default: pass = farther && nearer && !max_z; break;
        }
        if (!pass) return false;
    } else {
        blend_en = st.force_blend || (!overflow && st.aa_en);
    }

    // Blender
    s32 px[4] = {in_r, in_g, in_b, alpha};
    const s32 memc[4] = {mem.r, mem.g, mem.b, mem_cvg << 5};
    const s32 fog[4] = {static_cast<s32>(st.fog_color >> 24), static_cast<s32>((st.fog_color >> 16) & 0xFF),
                        static_cast<s32>((st.fog_color >> 8) & 0xFF), static_cast<s32>(st.fog_color & 0xFF)};
    const s32 blc[4] = {static_cast<s32>(st.blend_color >> 24), static_cast<s32>((st.blend_color >> 16) & 0xFF),
                        static_cast<s32>((st.blend_color >> 8) & 0xFF), static_cast<s32>(st.blend_color & 0xFF)};
    auto pick = [&](u8 sel) -> const s32* { return sel == 0 ? px : sel == 1 ? memc : sel == 2 ? blc : fog; };
    auto cycle = [&](u8 p, u8 a, u8 m, u8 b, bool final_cycle, s32 res[3]) {
        const s32* src1 = pick(m);
        if (final_cycle && st.color_on_cvg && !overflow) {
            for (int i = 0; i < 3; ++i) res[i] = src1[i];
            return;
        }
        const s32* src0 = pick(p);
        if (final_cycle && (!blend_en || (a == 0 && b == 0 && px[3] == 0xFF))) {
            for (int i = 0; i < 3; ++i) res[i] = src0[i];
            return;
        }
        s32 a0 = a == 0 ? px[3] : a == 1 ? fog[3] : a == 2 ? aux.shade_a : 0;
        s32 a1 = b == 0 ? (~a0 & 0xFF) : b == 1 ? memc[3] : b == 2 ? 0xFF : 0;
        a0 >>= 3;
        a1 >>= 3;
        if (b == 1) {
            a0 &= 0x3C;
            a1 |= 3;
        }
        if (!final_cycle || st.force_blend) {
            for (int i = 0; i < 3; ++i) res[i] = ((src0[i] * a0 + src1[i] * (a1 + 1)) >> 5) & 0xFF;
        } else {
            const u8* div = blender_divider();
            const s32 sum = (a0 >> 2) + (a1 >> 2) + 1;
            for (int i = 0; i < 3; ++i) res[i] = div[((sum << 11) | (((src0[i] * a0 + src1[i] * (a1 + 1)) >> 2) & 0x7FF)) & 0x7FFF];
        }
    };
    s32 rgb[3];
    if (st.two_cycle) {
        cycle(st.bl_p, st.bl_a, st.bl_m, st.bl_b, false, rgb);
        px[0] = rgb[0];
        px[1] = rgb[1];
        px[2] = rgb[2];
        cycle(st.bl2_p, st.bl2_a, st.bl2_m, st.bl2_b, true, rgb);
    } else {
        cycle(st.bl_p, st.bl_a, st.bl_m, st.bl_b, true, rgb);
    }
    out.r = static_cast<u8>(rgb[0]);
    out.g = static_cast<u8>(rgb[1]);
    out.b = static_cast<u8>(rgb[2]);
    switch (st.cvg_mode) {
        case 0: out.cvg = static_cast<u8>(blend_en ? std::min(7, mem_cvg + cvg) : (cvg - 1) & 7); break;
        case 1: out.cvg = static_cast<u8>((cvg + mem_cvg) & 7); break;
        case 2: out.cvg = 7; break;
        default: out.cvg = static_cast<u8>(mem_cvg); break;
    }
    out.z_write = st.z_update;
    return true;
}

// The RDP walks each row of a triangle from its major edge (top vertex to
// bottom vertex) towards the other two: rightwards (1) when the major edge
// is on the left, else leftwards (-1).
template <class V>
inline f32 walk_direction(const V& v0, const V& v1, const V& v2) {
    const V* t = &v0;
    const V* m = &v1;
    const V* b = &v2;
    if (m->sy < t->sy) std::swap(t, m);
    if (b->sy < m->sy) std::swap(m, b);
    if (m->sy < t->sy) std::swap(t, m);
    const f32 dy = b->sy - t->sy;
    const f32 major_x = dy != 0.0f ? t->sx + (b->sx - t->sx) * (m->sy - t->sy) / dy : t->sx;
    return m->sx >= major_x ? 1.0f : -1.0f;
}

// Signed screen-space area (twice the triangle's area); its sign is the winding.
template <class V>
inline f32 triangle_area(const V& v0, const V& v1, const V& v2) {
    return (v1.sx - v0.sx) * (v2.sy - v0.sy) - (v2.sx - v0.sx) * (v1.sy - v0.sy);
}

// Rasterizes one (already culled) triangle at `scale` times the frame buffer
// resolution, visiting only output rows [row_begin, row_end). Pixels whose
// centre lies inside the triangle are shaded and handed to
// sink.write(x, y, colour, z, shade alpha). V is any vertex type with the screen-space
// fields of Vertex. Returns false when the scissor leaves nothing to draw.
// The scissored bounding box triangle() scans at `scale`; false if it is
// empty (nothing is drawn). Rows int(min_y) .. int(max_y) are the only ones
// it can touch, which the native pass relies on to skip whole row bands.
template <class V>
inline bool triangle_bounds(const DrawState& st, const V& v0, const V& v1, const V& v2, u32 scale,
                            f32& min_x, f32& max_x, f32& min_y, f32& max_y) {
    const u32 fb_w = st.fb_w;
    const u32 eff_lrx = (st.scissor_lrx > st.scissor_ulx) ? std::min(st.scissor_lrx, fb_w) : fb_w;
    const u32 eff_lry = (st.scissor_lry > st.scissor_uly) ? std::min(st.scissor_lry, kMaxFbLines) : kMaxFbLines;
    const f32 fs = static_cast<f32>(scale);
    min_x = std::clamp(std::min({v0.sx, v1.sx, v2.sx}) * fs, static_cast<f32>(st.scissor_ulx * scale), static_cast<f32>(eff_lrx > 0 ? eff_lrx * scale - 1 : 0));
    max_x = std::clamp(std::max({v0.sx, v1.sx, v2.sx}) * fs, static_cast<f32>(st.scissor_ulx * scale), static_cast<f32>(eff_lrx > 0 ? eff_lrx * scale - 1 : 0));
    min_y = std::clamp(std::min({v0.sy, v1.sy, v2.sy}) * fs, static_cast<f32>(st.scissor_uly * scale), static_cast<f32>(eff_lry > 0 ? eff_lry * scale - 1 : 0));
    max_y = std::clamp(std::max({v0.sy, v1.sy, v2.sy}) * fs, static_cast<f32>(st.scissor_uly * scale), static_cast<f32>(eff_lry > 0 ? eff_lry * scale - 1 : 0));
    return !(min_x > max_x || min_y > max_y);
}

template <class V, class Sink>
inline bool triangle(const DrawState& st, const V& v0, const V& v1, const V& v2, f32 area,
                     u32 scale, s32 row_begin, s32 row_end, Sink& sink) {
    const f32 fs = static_cast<f32>(scale);
    f32 min_x, max_x, min_y, max_y;
    if (!triangle_bounds(st, v0, v1, v2, scale, min_x, max_x, min_y, max_y)) return false;

    const int y_first = std::max(static_cast<int>(min_y), static_cast<int>(row_begin));
    const int y_last = std::min(static_cast<int>(max_y), static_cast<int>(row_end) - 1);
    if (y_first > y_last) return true;

    const f32 inv_area = 1.0f / area;
    const f32 inv_scale = 1.0f / fs; // pixel centres in frame buffer units

    const f32 inv_w0 = 1.0f / (v0.w != 0.0f ? v0.w : 1.0f);
    const f32 inv_w1 = 1.0f / (v1.w != 0.0f ? v1.w : 1.0f);
    const f32 inv_w2 = 1.0f / (v2.w != 0.0f ? v2.w : 1.0f);

    const f32 u_over_w0 = v0.u * inv_w0;
    const f32 v_over_w0 = v0.v * inv_w0;
    const f32 u_over_w1 = v1.u * inv_w1;
    const f32 v_over_w1 = v1.v * inv_w1;
    const f32 u_over_w2 = v2.u * inv_w2;
    const f32 v_over_w2 = v2.v * inv_w2;

    const bool smooth_shading = st.smooth_shading;
    const bool textured = st.texture_enabled;
    const bool combined = st.combine_set;
    u32 tile0 = st.active_tile;
    u32 tile1 = (st.active_tile + 1) & 0x7;
    s32 lod_frac = 0;

    // How the barycentric weights change one frame buffer pixel to the right
    // and down, for the texture coordinates of the neighbouring pixels the
    // level of detail compares with.
    const bool dolod = textured && st.dolod;
    const f32 dw0x = (v1.sy - v2.sy) * inv_area, dw1x = (v2.sy - v0.sy) * inv_area, dw2x = -dw0x - dw1x;
    const f32 dw0y = (v2.sx - v1.sx) * inv_area, dw1y = (v0.sx - v2.sx) * inv_area, dw2y = -dw0y - dw1y;
    auto ddx = [&](f32 a0, f32 a1, f32 a2) { return dw0x * a0 + dw1x * a1 + dw2x * a2; };
    auto ddy = [&](f32 a0, f32 a1, f32 a2) { return dw0y * a0 + dw1y * a1 + dw2y * a2; };
    const f32 duw_dx = ddx(u_over_w0, u_over_w1, u_over_w2), duw_dy = ddy(u_over_w0, u_over_w1, u_over_w2);
    const f32 dvw_dx = ddx(v_over_w0, v_over_w1, v_over_w2), dvw_dy = ddy(v_over_w0, v_over_w1, v_over_w2);
    const f32 diw_dx = ddx(inv_w0, inv_w1, inv_w2), diw_dy = ddy(inv_w0, inv_w1, inv_w2);
    const f32 walk_dir = st.pipelined_tex1 ? walk_direction(v0, v1, v2) : 1.0f;

    // The RDP samples a pixel at its upper-left corner: that point decides
    // whether the pixel is drawn, and the attributes are taken there. A
    // corner exactly on an edge belongs to the triangle on the edge's right
    // (or, for a horizontal edge, below it): left and top edges are
    // inclusive, right and bottom ones exclusive.
    const f32 sgn = area > 0.0f ? 1.0f : -1.0f;
    auto tie = [&](f32 a, f32 b) { return a * sgn > 0.0f || (a == 0.0f && b * sgn > 0.0f); };
    const bool tie0 = tie(v1.sy - v2.sy, v2.sx - v1.sx);
    const bool tie1 = tie(v2.sy - v0.sy, v0.sx - v2.sx);
    const bool tie2 = tie(v0.sy - v1.sy, v1.sx - v0.sx);
    auto inside = [&](f32 e, bool t) { return e * sgn > 0.0f || (e == 0.0f && t); };

    for (int y = y_first; y <= y_last; ++y) {
        for (int x = static_cast<int>(min_x); x <= static_cast<int>(max_x); ++x) {
            f32 px = x * inv_scale;
            f32 py = y * inv_scale;

            // Edge functions (twice the areas of the sub-triangles), then barycentric coordinates.
            const f32 e0 = (v1.sx - px) * (v2.sy - py) - (v2.sx - px) * (v1.sy - py);
            const f32 e1 = (v2.sx - px) * (v0.sy - py) - (v0.sx - px) * (v2.sy - py);
            const f32 e2 = area - e0 - e1;
            f32 w0 = e0 * inv_area;
            f32 w1 = e1 * inv_area;
            f32 w2 = 1.0f - w0 - w1;

            if (inside(e0, tie0) && inside(e1, tie1) && inside(e2, tie2)) {
                f32 z = w0 * v0.sz + w1 * v1.sz + w2 * v2.sz;

                u8 r, g, b, a;
                if (smooth_shading) {
                    r = static_cast<u8>(std::clamp(w0 * v0.r + w1 * v1.r + w2 * v2.r, 0.0f, 255.0f));
                    g = static_cast<u8>(std::clamp(w0 * v0.g + w1 * v1.g + w2 * v2.g, 0.0f, 255.0f));
                    b = static_cast<u8>(std::clamp(w0 * v0.b + w1 * v1.b + w2 * v2.b, 0.0f, 255.0f));
                    a = static_cast<u8>(std::clamp(w0 * v0.a + w1 * v1.a + w2 * v2.a, 0.0f, 255.0f));
                } else {
                    r = v0.r;
                    g = v0.g;
                    b = v0.b;
                    a = v0.a;
                }
                PixelAux aux;
                aux.shade_a = a; // the blender's A_SHADE

                u32 color;
                if (textured) {
                    f32 inv_w_interp = w0 * inv_w0 + w1 * inv_w1 + w2 * inv_w2;
                    f32 w_interp = (inv_w_interp != 0.0f) ? (1.0f / inv_w_interp) : 1.0f;
                    const f32 uw = w0 * u_over_w0 + w1 * u_over_w1 + w2 * u_over_w2;
                    const f32 vw = w0 * v_over_w0 + w1 * v_over_w1 + w2 * v_over_w2;
                    f32 u = uw * w_interp;
                    f32 v = vw * w_interp;
                    if (dolod) {
                        const f32 iwx = inv_w_interp + diw_dx, iwy = inv_w_interp + diw_dy;
                        const f32 rx = iwx != 0.0f ? 1.0f / iwx : 1.0f, ry = iwy != 0.0f ? 1.0f / iwy : 1.0f;
                        const f32 delta = std::max(std::max(std::fabs((uw + duw_dx) * rx - u), std::fabs((vw + dvw_dx) * rx - v)),
                                                   std::max(std::fabs((uw + duw_dy) * ry - u), std::fabs((vw + dvw_dy) * ry - v)));
                        lod_tiles(st, delta, st.active_tile, tile0, tile1, lod_frac);
                    }

                    if (combined) {
                        // A texel the combiner never reads doesn't change its output.
                        u32 tex = st.need_tex0 ? sample_texture(st, tile0, u, v) : 0;
                        u32 tex1 = 0;
                        if (st.need_tex1) {
                            tex1 = sample_texture(st, tile1, u, v, true);
                        } else if (st.pipelined_tex1) {
                            // The next pixel the RDP walks (away from the major edge).
                            const f32 step = walk_dir * inv_scale;
                            const f32 iwn = inv_w_interp + diw_dx * step;
                            const f32 rn = iwn != 0.0f ? 1.0f / iwn : 1.0f;
                            tex1 = sample_texture(st, tile0, (uw + duw_dx * step) * rn, (vw + dvw_dx * step) * rn);
                        }
                        const s32 noise = st.uses_noise ? cc_noise(st, static_cast<u32>(x), static_cast<u32>(y)) : 0;
                        u8 a0 = 0;
                        color = combine(st, tex, tex1, r, g, b, a, lod_frac, noise, &a0);
                        if (st.two_cycle) aux.alpha0 = a0;
                    } else {
                        // Modulate texture with vertex color
                        u32 tex = sample_texture(st, tile0, u, v);
                        u8 tr = (tex >> 16) & 0xFF;
                        u8 tg = (tex >> 8) & 0xFF;
                        u8 tb = tex & 0xFF;
                        u8 ta = (tex >> 24) & 0xFF;

                        r = (r * tr) / 255;
                        g = (g * tg) / 255;
                        b = (b * tb) / 255;
                        a = (a * ta) / 255;
                        color = (static_cast<u32>(a) << 24) | (static_cast<u32>(r) << 16) | (static_cast<u32>(g) << 8) | b;
                    }
                } else {
                    if (combined) {
                        const s32 noise = st.uses_noise ? cc_noise(st, static_cast<u32>(x), static_cast<u32>(y)) : 0;
                        u8 a0 = 0;
                        color = combine(st, 0, 0, r, g, b, a, 0, noise, &a0);
                        if (st.two_cycle) aux.alpha0 = a0;
                    } else {
                        color = (static_cast<u32>(a) << 24) | (static_cast<u32>(r) << 16) | (static_cast<u32>(g) << 8) | b;
                    }
                }

                if (st.alpha_test_dither) aux.noise = static_cast<u8>(pixel_noise(st, static_cast<u32>(x), static_cast<u32>(y)) >> 8);
                sink.write(static_cast<u32>(x), static_cast<u32>(y), color, z, aux);
            }
        }
    }
    return true;
}

// The S/T range a native texture rectangle w x h pixels samples.
struct TexRectClamp {
    f32 s_min, s_max, t_min, t_max;
};

inline TexRectClamp tex_rect_clamp(f32 s, f32 t, f32 dsdx, f32 dtdy, u32 w, u32 h) {
    const f32 s_end = s + static_cast<f32>(w - 1) * dsdx;
    const f32 t_end = t + static_cast<f32>(h - 1) * dtdy;
    return {std::min(s, s_end), std::max(s, s_end), std::min(t, t_end), std::max(t, t_end)};
}

// Texture rectangle (G_TEXRECT / G_TEXRECTFLIP / S2DEX objects) at `scale`
// times the frame buffer resolution, rows [row_begin, row_end) only. The
// rectangle and texture coordinates are in frame buffer units; each output
// pixel samples the texture at its upper-left corner, as the RDP does.
template <class Sink>
inline void tex_rect(const DrawState& st, u32 ulx, u32 uly, u32 lrx, u32 lry, u32 tile_idx, f32 s, f32 t,
                     f32 dsdx, f32 dtdy, bool flip, u32 scale, s32 row_begin, s32 row_end, Sink& sink) {
    const u32 fb_w = st.fb_w;
    u32 max_x = std::min(lrx, fb_w);
    u32 max_y = std::min(lry, kMaxFbLines);

    // In COPY (and FILL) cycle type the lower-right rectangle edge is inclusive,
    // and COPY steps four texels per cycle, so the S increment is quartered.
    // Without this 2D sprites lose their last row/column and are stretched 4x.
    if (st.fill_or_copy) {
        max_x = std::min(lrx + 1, fb_w);
        max_y = std::min(lry + 1, kMaxFbLines);
    }
    if (st.copy_mode) {
        dsdx *= 0.25f;
    }
    const f32 dsdx_px = dsdx / static_cast<f32>(scale); // S from one (output) pixel to the next

    const bool combined = st.combine_set && !st.copy_mode;
    // The texture coordinates step by DSDX / DTDY per pixel: one level of
    // detail for the whole rectangle.
    u32 t0 = tile_idx & 7, t1 = (tile_idx + 1) & 7;
    s32 lod_frac = 0;
    if (st.dolod) lod_tiles(st, std::max(std::fabs(dsdx), std::fabs(dtdy)), tile_idx, t0, t1, lod_frac);
    // Rectangles have no shade: the combiner's SHADE input is 0.
    auto shade = [&](f32 cur_s, f32 cur_t, u32 x, u32 y, PixelAux& aux) -> u32 {
        aux.shade_a = 0;
        if (st.alpha_test_dither) aux.noise = static_cast<u8>(pixel_noise(st, x, y) >> 8);
        f32 sample_s = flip ? cur_t : cur_s;
        f32 sample_t = flip ? cur_s : cur_t;
        if (!combined) return sample_texture(st, t0, sample_s, sample_t);
        u32 tex = st.need_tex0 ? sample_texture(st, t0, sample_s, sample_t) : 0;
        u32 tex1 = 0;
        if (st.need_tex1) tex1 = sample_texture(st, t1, sample_s, sample_t, true);
        else if (st.pipelined_tex1) // the next pixel's, rightwards
            tex1 = flip ? sample_texture(st, t0, cur_t, cur_s + dsdx_px) : sample_texture(st, t0, cur_s + dsdx_px, cur_t);
        const s32 noise = st.uses_noise ? cc_noise(st, x, y) : 0;
        u8 a0 = 0;
        const u32 c = combine(st, tex, tex1, 0, 0, 0, 0, lod_frac, noise, &a0);
        if (st.two_cycle) aux.alpha0 = a0;
        return c;
    };

    if (scale == 1) {
        // Stepping S/T per pixel, like the RDP.
        f32 cur_t = t;
        for (u32 y = uly; y < max_y; ++y) {
            if (static_cast<s32>(y) >= row_begin && static_cast<s32>(y) < row_end) {
                f32 cur_s = s;
                for (u32 x = ulx; x < max_x; ++x) {
                    PixelAux aux;
                    const u32 c = shade(cur_s, cur_t, x, y, aux);
                    sink.write(x, y, c, 0.0f, aux);
                    cur_s += dsdx;
                }
            }
            cur_t += dtdy;
        }
        return;
    }

    if (ulx >= max_x || uly >= max_y) return;
    const f32 dsdx_hr = dsdx / static_cast<f32>(scale);
    const f32 dtdy_hr = dtdy / static_cast<f32>(scale);
    const s32 x0 = static_cast<s32>(ulx * scale), x1 = static_cast<s32>(max_x * scale);
    const s32 y0 = static_cast<s32>(uly * scale);
    const s32 y_first = std::max(y0, row_begin);
    const s32 y_last = std::min(static_cast<s32>(max_y * scale), row_end);
    // Keep S/T within the texels the native rectangle samples. Past its last
    // pixel centre the hi-res pixels would otherwise filter in texels beyond
    // the loaded tile, which draws seams between rectangles drawn in strips.
    const TexRectClamp cl = tex_rect_clamp(s, t, dsdx, dtdy, max_x - ulx, max_y - uly);
    for (s32 y = y_first; y < y_last; ++y) {
        const f32 cur_t = std::clamp(t + static_cast<f32>(y - y0) * dtdy_hr, cl.t_min, cl.t_max);
        for (s32 x = x0; x < x1; ++x) {
            const f32 cur_s = std::clamp(s + static_cast<f32>(x - x0) * dsdx_hr, cl.s_min, cl.s_max);
            PixelAux aux;
            const u32 c = shade(cur_s, cur_t, static_cast<u32>(x), static_cast<u32>(y), aux);
            sink.write(static_cast<u32>(x), static_cast<u32>(y), c, 0.0f, aux);
        }
    }
}

} // namespace raster

using raster::DrawState;
using raster::TexCache;
