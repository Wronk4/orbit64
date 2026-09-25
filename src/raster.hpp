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
};

// The software RDP draws into, and the VI shows, 240 lines.
constexpr u32 kFbLines = 240;

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

// Everything one draw reads from the RDP, plus the mode decoding derived
// from it. Built when the RDP state changes, then shared by the draws that
// follow (and copied for the high-resolution pass).
struct DrawState {
    u32 other_mode_h{0}, other_mode_l{0};
    u32 combine_w0{0}, combine_w1{0};
    u32 prim_color{0}, env_color{0}, blend_color{0}, fog_color{0};
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
    // Whether the combiner can read TEXEL0 / TEXEL1 at all (conservative).
    bool need_tex0{true}, need_tex1{true};
    u8 cc_a0{0}, cc_b0{0}, cc_c0{0}, cc_d0{0}, ac_a0{0}, ac_b0{0}, ac_c0{0}, ac_d0{0};
    u8 cc_a1{0}, cc_b1{0}, cc_c1{0}, cc_d1{0}, ac_a1{0}, ac_b1{0}, ac_c1{0}, ac_d1{0};
    f32 prim_r{0}, prim_g{0}, prim_b{0}, prim_a{0};
    f32 env_r{0}, env_g{0}, env_b{0}, env_a{0};
    u8 bl_p{0}, bl_a{0}, bl_m{0}, bl_b{0};
    bool blend_enabled{false};  // FORCE_BL / ZMODE_XLU: blend when alpha < 255
    u8 alpha_compare{0};        // G_AC_*: 0 none, 1 threshold, 3 dither
    u8 alpha_threshold{0};      // blend colour alpha
    bool alpha_zero_kill{false}; // CVG_X_ALPHA/ALPHA_CVG_SEL/FORCE_BL/ZMODE_XLU/IM_RD/AA_EN: drop alpha-0 pixels
    bool z_compare{false}, z_update{false};

    void finalize() {
        u32 cycle_type = (other_mode_h >> 20) & 0x3;
        copy_mode = cycle_type == 2;
        two_cycle = cycle_type == 1;
        fill_or_copy = cycle_type == 2 || cycle_type == 3;
        tlut_type = (other_mode_h >> 14) & 0x3;
        point_sample = ((other_mode_h >> 12) & 0x3) == 0 || copy_mode;

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

        // Texel inputs of the first cycle. Its COMBINED inputs (colour C
        // COMBINED_ALPHA and alpha COMBINED) read TEXEL0's alpha, see combine().
        bool c1_t0 = cc_a0 == 1 || cc_b0 == 1 || cc_d0 == 1 || cc_c0 == 1 || cc_c0 == 7 || cc_c0 == 8 ||
                     ac_a0 <= 1 || ac_b0 <= 1 || ac_c0 <= 1 || ac_d0 <= 1;
        bool c1_t1 = cc_a0 == 2 || cc_b0 == 2 || cc_d0 == 2 || cc_c0 == 2 || cc_c0 == 9 ||
                     ac_a0 == 2 || ac_b0 == 2 || ac_c0 == 2 || ac_d0 == 2;
        // The second cycle sees the texels rotated: its TEXEL0 is TEXEL1.
        bool c2_t1 = cc_a1 == 1 || cc_b1 == 1 || cc_d1 == 1 || cc_c1 == 1 || cc_c1 == 8 ||
                     ac_a1 == 1 || ac_b1 == 1 || ac_c1 == 1 || ac_d1 == 1;
        bool c2_t0 = cc_a1 == 2 || cc_b1 == 2 || cc_d1 == 2 || cc_c1 == 2 || cc_c1 == 9 ||
                     ac_a1 == 2 || ac_b1 == 2 || ac_c1 == 2 || ac_d1 == 2;
        need_tex0 = c1_t0 || (two_cycle && c2_t0);
        need_tex1 = c1_t1 || (two_cycle && c2_t1);

        // G_MDSFT_BLENDER=16: 1-cycle (and 2-cycle first pass) P/A/M/B at bits
        // 30/26/22/18, the 2-cycle output pass at 28/24/20/16.
        bl_p = (other_mode_l >> (two_cycle ? 28 : 30)) & 0x3;
        bl_a = (other_mode_l >> (two_cycle ? 24 : 26)) & 0x3;
        bl_m = (other_mode_l >> (two_cycle ? 20 : 22)) & 0x3;
        bl_b = (other_mode_l >> (two_cycle ? 16 : 18)) & 0x3;
        blend_enabled = (other_mode_l & 0x4800) != 0;
        alpha_compare = other_mode_l & 0x3;
        alpha_threshold = blend_color & 0xFF;
        alpha_zero_kill = (other_mode_l & 0x7848) != 0;
        z_compare = (other_mode_l & 0x10) != 0;
        z_update = (other_mode_l & 0x20) != 0;
    }
};

// 5-bit colour channel -> 8 bits, rounded: (c * 255 + 15) / 31.
inline constexpr std::array<u8, 32> kFiveToEight = [] {
    std::array<u8, 32> t{};
    for (int c = 0; c < 32; ++c) t[c] = static_cast<u8>((c * 255 + 15) / 31);
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
            if (tile.format == 2) { // CI8: full 256-entry TLUT
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
            if (tile.format == 2) { // CI4: 16-entry sub-palette
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

// Size of the table a TexCache needs for a tile: every value wrap_s/wrap_t can return.
inline void tex_cache_dims(const TexUnit& tu, u32& w, u32& h) {
    w = tu.tile.mask_s ? (1u << tu.tile.mask_s) : static_cast<u32>(tu.extent_s) + 1;
    h = tu.tile.mask_t ? (1u << tu.tile.mask_t) : static_cast<u32>(tu.extent_t) + 1;
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

inline u8 lerp_u8(u8 a, u8 b, f32 t) {
    return static_cast<u8>(std::clamp(a + t * (static_cast<f32>(b) - static_cast<f32>(a)), 0.0f, 255.0f));
}

// Bilinear blend of four packed ARGB texels: lerp_u8 across S on both rows,
// then across T, per byte. The SSE2 version does the four channels in one
// register with exactly the scalar operations (sub, mul, add, clamp, truncate
// to an integer between the two stages), so the result is bit-identical.
inline u32 bilerp_argb(u32 c00, u32 c10, u32 c01, u32 c11, f32 frac_s, f32 frac_t) {
#if defined(ORBIT64_RASTER_SSE2)
    const __m128i zero = _mm_setzero_si128();
    auto unpack = [&](u32 c) {
        __m128i v = _mm_cvtsi32_si128(static_cast<int>(c));
        v = _mm_unpacklo_epi16(_mm_unpacklo_epi8(v, zero), zero);
        return _mm_cvtepi32_ps(v);
    };
    const __m128 lo = _mm_setzero_ps(), hi = _mm_set1_ps(255.0f);
    auto lerp = [&](__m128 a, __m128 b, __m128 t) {
        __m128 v = _mm_add_ps(a, _mm_mul_ps(t, _mm_sub_ps(b, a)));
        return _mm_cvttps_epi32(_mm_min_ps(_mm_max_ps(v, lo), hi));
    };
    const __m128 ts = _mm_set1_ps(frac_s);
    const __m128 top = _mm_cvtepi32_ps(lerp(unpack(c00), unpack(c10), ts));
    const __m128 bot = _mm_cvtepi32_ps(lerp(unpack(c01), unpack(c11), ts));
    __m128i out = lerp(top, bot, _mm_set1_ps(frac_t));
    out = _mm_packus_epi16(_mm_packs_epi32(out, out), zero);
    return static_cast<u32>(_mm_cvtsi128_si32(out));
#else
    u32 out = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        auto comp = [shift](u32 c) { return static_cast<u8>((c >> shift) & 0xFF); };
        u8 top = lerp_u8(comp(c00), comp(c10), frac_s);
        u8 bot = lerp_u8(comp(c01), comp(c11), frac_s);
        out |= static_cast<u32>(lerp_u8(top, bot, frac_t)) << shift;
    }
    return out;
#endif
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

inline u32 sample_texture(const DrawState& st, u32 tile_idx, f32 s, f32 t) {
    const TexUnit& tu = st.tex[tile_idx & 0x7];
    f32 shifted_s = s * tu.shift_mul_s;
    f32 shifted_t = t * tu.shift_mul_t;
    shifted_s -= tu.origin_s;
    shifted_t -= tu.origin_t;
    const u32* table = tex_table(st, tu);

    if (st.point_sample) {
        s32 is = floor_to_s32(shifted_s);
        s32 it = floor_to_s32(shifted_t);
        if (table) return table[static_cast<u32>(wrap_t(tu, it)) * tu.cache->w + static_cast<u32>(wrap_s(tu, is))];
        return fetch_texel(tu, st.tmem, st.tmem_dxt, st.tlut_type, is, it);
    }

    // Bilinear (G_TF_BILERP / G_TF_AVERAGE): the RDP splits the coordinate into
    // its integer and fractional parts directly -- no half-texel bias -- and
    // blends the 2x2 neighbourhood from there.
    s32 is0 = floor_to_s32(shifted_s);
    s32 it0 = floor_to_s32(shifted_t);
    f32 frac_s = shifted_s - static_cast<f32>(is0);
    f32 frac_t = shifted_t - static_cast<f32>(it0);

    u32 c00, c10, c01, c11;
    if (table) {
        const u32 w = tu.cache->w;
        const u32 s0 = static_cast<u32>(wrap_s(tu, is0)), s1 = static_cast<u32>(wrap_s(tu, is0 + 1));
        const u32* row0 = table + static_cast<u32>(wrap_t(tu, it0)) * w;
        const u32* row1 = table + static_cast<u32>(wrap_t(tu, it0 + 1)) * w;
        c00 = row0[s0];
        c10 = row0[s1];
        c01 = row1[s0];
        c11 = row1[s1];
    } else {
        c00 = fetch_texel(tu, st.tmem, st.tmem_dxt, st.tlut_type, is0,     it0);
        c10 = fetch_texel(tu, st.tmem, st.tmem_dxt, st.tlut_type, is0 + 1, it0);
        c01 = fetch_texel(tu, st.tmem, st.tmem_dxt, st.tlut_type, is0,     it0 + 1);
        c11 = fetch_texel(tu, st.tmem, st.tmem_dxt, st.tlut_type, is0 + 1, it0 + 1);
    }

    return bilerp_argb(c00, c10, c01, c11, frac_s, frac_t);
}

// Colour combiner: (A - B) * C + D per channel, once or twice (2-cycle).
inline u32 combine(const DrawState& st, u32 tex0, u32 tex1, u8 sr, u8 sg, u8 sb, u8 sa) {
    // Cycle-local TEXEL0/TEXEL1: on real hardware the texture unit is pipelined, so in
    // 2-cycle mode the second combiner stage sees the texels rotated by one slot -
    // its "TEXEL0" is the tile that was "TEXEL1" in the first stage (and vice versa).
    // Games rely on this to combine two textures (e.g. a color texture + a separate
    // alpha/intensity mask loaded into the next tile) across the two cycles.
    const auto& U = kByteToUnit;
    f32 t0r = U[(tex0 >> 16) & 0xFF], t0g = U[(tex0 >> 8) & 0xFF], t0b = U[tex0 & 0xFF], t0a = U[(tex0 >> 24) & 0xFF];
    f32 t1r = U[(tex1 >> 16) & 0xFF], t1g = U[(tex1 >> 8) & 0xFF], t1b = U[tex1 & 0xFF], t1a = U[(tex1 >> 24) & 0xFF];

    const f32 v_sr = U[sr], v_sg = U[sg], v_sb = U[sb], v_sa = U[sa];
    const f32 pr = st.prim_r, pg = st.prim_g, pb = st.prim_b, pa = st.prim_a;
    const f32 er = st.env_r, eg = st.env_g, eb = st.env_b, ea = st.env_a;

    // A, B and D inputs (B's 6 is CENTER, D's 7 is ZERO; neither is modelled apart from 1.0/0.0).
    auto get_color_abd = [&](u32 src, f32 comb_r, f32 comb_g, f32 comb_b, f32& out_r, f32& out_g, f32& out_b) {
        switch (src) {
            case 0: out_r = comb_r; out_g = comb_g; out_b = comb_b; break; // COMBINED
            case 1: out_r = t0r; out_g = t0g; out_b = t0b; break; // TEXEL0
            case 2: out_r = t1r; out_g = t1g; out_b = t1b; break; // TEXEL1
            case 3: out_r = pr; out_g = pg; out_b = pb; break; // PRIMITIVE
            case 4: out_r = v_sr; out_g = v_sg; out_b = v_sb; break; // SHADE
            case 5: out_r = er; out_g = eg; out_b = eb; break; // ENVIRONMENT
            case 6: out_r = out_g = out_b = 1.0f; break; // 1.0
            default: out_r = out_g = out_b = 0.0f; break; // 0.0
        }
    };

    auto get_color_c = [&](u32 src, f32 comb_r, f32 comb_g, f32 comb_b, f32 comb_a, f32& out_r, f32& out_g, f32& out_b) {
        switch (src) {
            case 0: out_r = comb_r; out_g = comb_g; out_b = comb_b; break; // COMBINED
            case 1: out_r = t0r; out_g = t0g; out_b = t0b; break; // TEXEL0
            case 2: out_r = t1r; out_g = t1g; out_b = t1b; break; // TEXEL1
            case 3: out_r = pr; out_g = pg; out_b = pb; break; // PRIMITIVE
            case 4: out_r = v_sr; out_g = v_sg; out_b = v_sb; break; // SHADE
            case 5: out_r = er; out_g = eg; out_b = eb; break; // ENVIRONMENT
            case 6: out_r = out_g = out_b = 1.0f; break; // SCALE
            case 7: out_r = out_g = out_b = comb_a; break; // COMBINED_ALPHA
            case 8: out_r = out_g = out_b = t0a; break; // TEXEL0_ALPHA
            case 9: out_r = out_g = out_b = t1a; break; // TEXEL1_ALPHA
            case 10: out_r = out_g = out_b = pa; break; // PRIMITIVE_ALPHA
            case 11: out_r = out_g = out_b = v_sa; break; // SHADE_ALPHA
            case 12: out_r = out_g = out_b = ea; break; // ENV_ALPHA
            default: out_r = out_g = out_b = 0.0f; break; // 0.0 (including 31)
        }
    };

    auto get_alpha_abd = [&](u32 src, f32 comb_a, bool is_c_slot = false) -> f32 {
        switch (src) {
            case 0: return comb_a; // COMBINED
            case 1: return t0a;  // TEXEL0
            case 2: return t1a;  // TEXEL1
            case 3: return pa;   // PRIMITIVE
            case 4: return v_sa; // SHADE
            case 5: return ea;   // ENVIRONMENT
            case 6: return is_c_slot ? 0.0f : 1.0f; // C slot: PRIM_LOD_FRAC (LOD not modeled); A/B/D: 1.0
            case 7: // 7 is ZERO (G_ACMUX_0 = 7)
            default: return 0.0f;
        }
    };

    // The first cycle has no COMBINED input yet: its COMBINED colour is 0 and
    // its COMBINED alpha is TEXEL0's.
    f32 A_r, A_g, A_b; get_color_abd(st.cc_a0, 0.0f, 0.0f, 0.0f, A_r, A_g, A_b);
    f32 B_r, B_g, B_b; get_color_abd(st.cc_b0, 0.0f, 0.0f, 0.0f, B_r, B_g, B_b);
    f32 C_r, C_g, C_b; get_color_c(st.cc_c0, 0.0f, 0.0f, 0.0f, t0a, C_r, C_g, C_b);
    f32 D_r, D_g, D_b; get_color_abd(st.cc_d0, 0.0f, 0.0f, 0.0f, D_r, D_g, D_b);

    f32 Aa = get_alpha_abd(st.ac_a0, t0a);
    f32 Ab = get_alpha_abd(st.ac_b0, t0a);
    f32 Ac = get_alpha_abd(st.ac_c0, t0a, true);
    f32 Ad = get_alpha_abd(st.ac_d0, t0a);

    f32 res_r = std::clamp((A_r - B_r) * C_r + D_r, 0.0f, 1.0f);
    f32 res_g = std::clamp((A_g - B_g) * C_g + D_g, 0.0f, 1.0f);
    f32 res_b = std::clamp((A_b - B_b) * C_b + D_b, 0.0f, 1.0f);
    f32 res_a = std::clamp((Aa - Ab) * Ac + Ad, 0.0f, 1.0f);

    if (st.two_cycle) {
        f32 c0_r = res_r, c0_g = res_g, c0_b = res_b, c0_a = res_a;

        // Rotate texel slots for the second cycle (see comment above).
        std::swap(t0r, t1r); std::swap(t0g, t1g); std::swap(t0b, t1b); std::swap(t0a, t1a);

        f32 A1_r, A1_g, A1_b; get_color_abd(st.cc_a1, c0_r, c0_g, c0_b, A1_r, A1_g, A1_b);
        f32 B1_r, B1_g, B1_b; get_color_abd(st.cc_b1, c0_r, c0_g, c0_b, B1_r, B1_g, B1_b);
        f32 C1_r, C1_g, C1_b; get_color_c(st.cc_c1, c0_r, c0_g, c0_b, c0_a, C1_r, C1_g, C1_b);
        f32 D1_r, D1_g, D1_b; get_color_abd(st.cc_d1, c0_r, c0_g, c0_b, D1_r, D1_g, D1_b);

        f32 a_Aa1 = get_alpha_abd(st.ac_a1, c0_a);
        f32 a_Ab1 = get_alpha_abd(st.ac_b1, c0_a);
        f32 a_Ac1 = get_alpha_abd(st.ac_c1, c0_a, true);
        f32 a_Ad1 = get_alpha_abd(st.ac_d1, c0_a);

        res_r = std::clamp((A1_r - B1_r) * C1_r + D1_r, 0.0f, 1.0f);
        res_g = std::clamp((A1_g - B1_g) * C1_g + D1_g, 0.0f, 1.0f);
        res_b = std::clamp((A1_b - B1_b) * C1_b + D1_b, 0.0f, 1.0f);
        res_a = std::clamp((a_Aa1 - a_Ab1) * a_Ac1 + a_Ad1, 0.0f, 1.0f);
    }

    u8 out_r = static_cast<u8>(res_r * 255.0f);
    u8 out_g = static_cast<u8>(res_g * 255.0f);
    u8 out_b = static_cast<u8>(res_b * 255.0f);
    u8 out_a = static_cast<u8>(res_a * 255.0f);

    return (static_cast<u32>(out_a) << 24) |
           (static_cast<u32>(out_r) << 16) |
           (static_cast<u32>(out_g) << 8)  |
            static_cast<u32>(out_b);
}

// RDP blender (G_BL_* P/A/M/B multiplexer): blends the combiner's output colour
// against the frame buffer using the active render mode.
inline void blend_rgb(const DrawState& st, u32 src_color, u8 dest_r, u8 dest_g, u8 dest_b, u8 dest_a,
                      u8& out_r, u8& out_g, u8& out_b) {
    u8 sr = (src_color >> 16) & 0xFF, sg = (src_color >> 8) & 0xFF, sb = src_color & 0xFF, sa = (src_color >> 24) & 0xFF;

    auto pick_color = [&](u32 sel, f32& r, f32& g, f32& b) {
        switch (sel) {
            case 0: r = sr; g = sg; b = sb; break; // G_BL_CLR_IN
            case 1: r = dest_r; g = dest_g; b = dest_b; break; // G_BL_CLR_MEM
            case 2: // G_BL_CLR_BL (blend_color register; true 2-cycle intermediate not modeled)
                r = (st.blend_color >> 24) & 0xFF; g = (st.blend_color >> 16) & 0xFF; b = (st.blend_color >> 8) & 0xFF;
                break;
            default: // G_BL_CLR_FOG
                r = (st.fog_color >> 24) & 0xFF; g = (st.fog_color >> 16) & 0xFF; b = (st.fog_color >> 8) & 0xFF;
                break;
        }
    };
    // A_IN/A_FOG/A_SHADE all approximated as the fragment's final alpha: this pipeline
    // doesn't track a separate depth-based fog factor or pre-combine shade alpha here.
    f32 A = (st.bl_a == 3) ? 0.0f : static_cast<f32>(sa);
    f32 B;
    switch (st.bl_b) {
        case 0: B = 255.0f - A; break;  // G_BL_1MA
        case 1: B = dest_a; break;      // G_BL_A_MEM
        case 2: B = 255.0f; break;      // G_BL_1
        default: B = 0.0f; break;       // G_BL_0
    }

    f32 pr, pg, pb, mr, mg, mb;
    pick_color(st.bl_p, pr, pg, pb);
    pick_color(st.bl_m, mr, mg, mb);

    out_r = static_cast<u8>(std::clamp((pr * A + mr * B) / 255.0f, 0.0f, 255.0f));
    out_g = static_cast<u8>(std::clamp((pg * A + mg * B) / 255.0f, 0.0f, 255.0f));
    out_b = static_cast<u8>(std::clamp((pb * A + mb * B) / 255.0f, 0.0f, 255.0f));
}

// Signed screen-space area (twice the triangle's area); its sign is the winding.
template <class V>
inline f32 triangle_area(const V& v0, const V& v1, const V& v2) {
    return (v1.sx - v0.sx) * (v2.sy - v0.sy) - (v2.sx - v0.sx) * (v1.sy - v0.sy);
}

// Rasterizes one (already culled) triangle at `scale` times the frame buffer
// resolution, visiting only output rows [row_begin, row_end). Pixels whose
// centre lies inside the triangle are shaded and handed to
// sink.write(x, y, colour, z). V is any vertex type with the screen-space
// fields of Vertex. Returns false when the scissor leaves nothing to draw.
// The scissored bounding box triangle() scans at `scale`; false if it is
// empty (nothing is drawn). Rows int(min_y) .. int(max_y) are the only ones
// it can touch, which the native pass relies on to skip whole row bands.
template <class V>
inline bool triangle_bounds(const DrawState& st, const V& v0, const V& v1, const V& v2, u32 scale,
                            f32& min_x, f32& max_x, f32& min_y, f32& max_y) {
    const u32 fb_w = st.fb_w;
    const u32 eff_lrx = (st.scissor_lrx > st.scissor_ulx) ? std::min(st.scissor_lrx, fb_w) : fb_w;
    const u32 eff_lry = (st.scissor_lry > st.scissor_uly) ? std::min(st.scissor_lry, kFbLines) : kFbLines;
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
    const u32 tile0 = st.active_tile;
    const u32 tile1 = (st.active_tile + 1) & 0x7;

    for (int y = y_first; y <= y_last; ++y) {
        for (int x = static_cast<int>(min_x); x <= static_cast<int>(max_x); ++x) {
            f32 px = (x + 0.5f) * inv_scale;
            f32 py = (y + 0.5f) * inv_scale;

            // Barycentric coordinates
            f32 w0 = ((v1.sx - px) * (v2.sy - py) - (v2.sx - px) * (v1.sy - py)) * inv_area;
            f32 w1 = ((v2.sx - px) * (v0.sy - py) - (v0.sx - px) * (v2.sy - py)) * inv_area;
            f32 w2 = 1.0f - w0 - w1;

            if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
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

                u32 color;
                if (textured) {
                    f32 inv_w_interp = w0 * inv_w0 + w1 * inv_w1 + w2 * inv_w2;
                    f32 w_interp = (inv_w_interp != 0.0f) ? (1.0f / inv_w_interp) : 1.0f;
                    f32 u = (w0 * u_over_w0 + w1 * u_over_w1 + w2 * u_over_w2) * w_interp;
                    f32 v = (w0 * v_over_w0 + w1 * v_over_w1 + w2 * v_over_w2) * w_interp;

                    if (combined) {
                        // A texel the combiner never reads doesn't change its output.
                        u32 tex = st.need_tex0 ? sample_texture(st, tile0, u, v) : 0;
                        u32 tex1 = st.need_tex1 ? sample_texture(st, tile1, u, v) : 0;
                        color = combine(st, tex, tex1, r, g, b, a);
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
                        color = combine(st, 0xFFFFFFFF, 0xFFFFFFFF, r, g, b, a);
                    } else {
                        color = (static_cast<u32>(a) << 24) | (static_cast<u32>(r) << 16) | (static_cast<u32>(g) << 8) | b;
                    }
                }

                sink.write(static_cast<u32>(x), static_cast<u32>(y), color, z);
            }
        }
    }
    return true;
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
    u32 max_y = std::min(lry, kFbLines);

    // In COPY (and FILL) cycle type the lower-right rectangle edge is inclusive,
    // and COPY steps four texels per cycle, so the S increment is quartered.
    // Without this 2D sprites lose their last row/column and are stretched 4x.
    if (st.fill_or_copy) {
        max_x = std::min(lrx + 1, fb_w);
        max_y = std::min(lry + 1, kFbLines);
    }
    if (st.copy_mode) {
        dsdx *= 0.25f;
    }

    const bool combined = st.combine_set && !st.copy_mode;
    auto shade = [&](f32 cur_s, f32 cur_t) -> u32 {
        f32 sample_s = flip ? cur_t : cur_s;
        f32 sample_t = flip ? cur_s : cur_t;
        if (!combined) return sample_texture(st, tile_idx, sample_s, sample_t);
        u32 tex = st.need_tex0 ? sample_texture(st, tile_idx, sample_s, sample_t) : 0;
        u32 tex1 = st.need_tex1 ? sample_texture(st, (tile_idx + 1) & 0x7, sample_s, sample_t) : 0;
        return combine(st, tex, tex1, 255, 255, 255, 255);
    };

    if (scale == 1) {
        // Stepping S/T per pixel, like the RDP.
        f32 cur_t = t;
        for (u32 y = uly; y < max_y; ++y) {
            if (static_cast<s32>(y) >= row_begin && static_cast<s32>(y) < row_end) {
                f32 cur_s = s;
                for (u32 x = ulx; x < max_x; ++x) {
                    sink.write(x, y, shade(cur_s, cur_t), 0.0f);
                    cur_s += dsdx;
                }
            }
            cur_t += dtdy;
        }
        return;
    }

    const f32 dsdx_hr = dsdx / static_cast<f32>(scale);
    const f32 dtdy_hr = dtdy / static_cast<f32>(scale);
    const s32 x0 = static_cast<s32>(ulx * scale), x1 = static_cast<s32>(max_x * scale);
    const s32 y0 = static_cast<s32>(uly * scale);
    const s32 y_first = std::max(y0, row_begin);
    const s32 y_last = std::min(static_cast<s32>(max_y * scale), row_end);
    for (s32 y = y_first; y < y_last; ++y) {
        const f32 cur_t = t + static_cast<f32>(y - y0) * dtdy_hr;
        for (s32 x = x0; x < x1; ++x) {
            const f32 cur_s = s + static_cast<f32>(x - x0) * dsdx_hr;
            sink.write(static_cast<u32>(x), static_cast<u32>(y), shade(cur_s, cur_t), 0.0f);
        }
    }
}

} // namespace raster

using raster::DrawState;
using raster::TexCache;
