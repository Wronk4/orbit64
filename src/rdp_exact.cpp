// Bit-exact low-level RDP (see rdp_exact.hpp).
//
// The rasterization, texturing, combiner, blender, depth and memory stages
// are a CPU port of parallel-rdp's shaders:
//
//   Copyright (c) 2020 Themaister
//
//   Permission is hereby granted, free of charge, to any person obtaining
//   a copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to
//   permit persons to whom the Software is furnished to do so, subject to
//   the following conditions:
//
//   The above copyright notice and this permission notice shall be
//   included in all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Integer semantics are those of the shaders: 32-bit wrap-around (computed
// through 64 bits and truncated), arithmetic right shifts.

#include "rdp_exact.hpp"
#include "jit/jit_invalidate.hpp"
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

// Framebuffer formats.
enum FbFmt : u8 { FB_I4 = 0, FB_I8 = 1, FB_RGBA5551 = 2, FB_IA88 = 3, FB_RGBA8888 = 4 };

// Setup flags.
constexpr u32 SETUP_FLIP = 1u << 0;
constexpr u32 SETUP_DO_OFFSET = 1u << 1;
constexpr u32 SETUP_SKIP_XFRAC = 1u << 2;
constexpr u32 SETUP_INTERLACE_FIELD = 1u << 3;
constexpr u32 SETUP_INTERLACE_KEEP_ODD = 1u << 4;

// Tile flags.
constexpr u8 TILE_CLAMP_S = 1u << 0;
constexpr u8 TILE_MIRROR_S = 1u << 1;
constexpr u8 TILE_CLAMP_T = 1u << 2;
constexpr u8 TILE_MIRROR_T = 1u << 3;

// Texture formats.
constexpr int TF_RGBA = 0, TF_YUV = 1, TF_CI = 2, TF_IA = 3, TF_I = 4;

inline s32 w32(s64 v) { return static_cast<s32>(v); }
// Division and remainder rounding towards minus infinity (the arithmetic
// shifts they replace for scales that are powers of two).
inline s32 fdiv(s32 a, s32 b) { return a >= 0 ? a / b : -((-static_cast<s64>(a) + b - 1) / b); }
inline s32 fmod_(s32 a, s32 b) { return a - fdiv(a, b) * b; }
inline s64 fdiv64(s64 a, s64 b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

// Zeroed memory whose pages the OS only backs once they are touched.
void* lazy_alloc(size_t n) {
#ifdef _WIN32
    return VirtualAlloc(nullptr, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}
void lazy_free(void* p, size_t n) {
    if (!p) return;
#ifdef _WIN32
    (void)n;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, n);
#endif
}
inline s32 sext(s32 v, int bits) { return static_cast<s32>(static_cast<u32>(v) << (32 - bits)) >> (32 - bits); }
inline int find_msb(s32 v) { return v <= 0 ? (v == 0 ? -1 : 31) : 31 - __builtin_clz(static_cast<u32>(v)); }
inline int find_lsb(u32 v) { return v ? __builtin_ctz(v) : -1; }

struct V4 {
    s32 v[4]{};
    s32& operator[](int i) { return v[i]; }
    s32 operator[](int i) const { return v[i]; }
};

// ---- The blender's divider (a 4-bit denominator, 11-bit numerator):
// a non-restoring division of n by d + 1 as the hardware carries it out.
struct BlenderDivider {
    u8 lut[0x8000];
    BlenderDivider() {
        for (int i = 0; i < 0x8000; ++i) {
            const int d = (i >> 11) & 0xf, n = i & 0x7ff, invd = ~d & 0xf;
            int res = 0;
            int partial = (invd + (n >> 8) + 1) & 7;
            for (int k = 0; k < 8; ++k) {
                const int nbit = (n >> (7 - k)) & 1;
                const int sum = (res & (0x100 >> k)) ? invd + (partial << 1) + nbit + 1 : d + (partial << 1) + nbit;
                partial = sum & 7;
                if (sum & 0x10) res |= 1 << (7 - k);
            }
            lut[i] = static_cast<u8>(res);
        }
    }
};
const BlenderDivider& blender_divider() {
    static const BlenderDivider d;
    return d;
}

// ---- Perspective division: s16 by s1.15 through the reciprocal table.
const s16 kPerspTable[64][2] = {
    {0x4000, -252 * 4}, {0x3f04, -244 * 4}, {0x3e10, -238 * 4}, {0x3d22, -230 * 4}, {0x3c3c, -223 * 4}, {0x3b5d, -218 * 4},
    {0x3a83, -210 * 4}, {0x39b1, -205 * 4}, {0x38e4, -200 * 4}, {0x381c, -194 * 4}, {0x375a, -189 * 4}, {0x369d, -184 * 4},
    {0x35e5, -179 * 4}, {0x3532, -175 * 4}, {0x3483, -170 * 4}, {0x33d9, -166 * 4}, {0x3333, -162 * 4}, {0x3291, -157 * 4},
    {0x31f4, -155 * 4}, {0x3159, -150 * 4}, {0x30c3, -147 * 4}, {0x3030, -143 * 4}, {0x2fa1, -140 * 4}, {0x2f15, -137 * 4},
    {0x2e8c, -134 * 4}, {0x2e06, -131 * 4}, {0x2d83, -128 * 4}, {0x2d03, -125 * 4}, {0x2c86, -123 * 4}, {0x2c0b, -120 * 4},
    {0x2b93, -117 * 4}, {0x2b1e, -115 * 4}, {0x2aab, -113 * 4}, {0x2a3a, -110 * 4}, {0x29cc, -108 * 4}, {0x2960, -106 * 4},
    {0x28f6, -104 * 4}, {0x288e, -102 * 4}, {0x2828, -100 * 4}, {0x27c4, -98 * 4},  {0x2762, -96 * 4},  {0x2702, -94 * 4},
    {0x26a4, -92 * 4},  {0x2648, -91 * 4},  {0x25ed, -89 * 4},  {0x2594, -87 * 4},  {0x253d, -86 * 4},  {0x24e7, -85 * 4},
    {0x2492, -83 * 4},  {0x243f, -81 * 4},  {0x23ee, -80 * 4},  {0x239e, -79 * 4},  {0x234f, -77 * 4},  {0x2302, -76 * 4},
    {0x22b6, -74 * 4},  {0x226c, -74 * 4},  {0x2222, -72 * 4},  {0x21da, -71 * 4},  {0x2193, -70 * 4},  {0x214d, -69 * 4},
    {0x2108, -67 * 4},  {0x20c5, -67 * 4},  {0x2082, -65 * 4},  {0x2041, -65 * 4},
};

// `copy`: the copy pipe has no clamp stage - a quotient that fits 17 bits
// but not 16 keeps its low 16 bits (hardware verified, diagnostic cartridge
// case 9:17); only the divider's own saturation applies.
void perspective_divide(s32 s, s32 t, s32 w, s32& out_s, s32& out_t, bool& overflow, bool copy = false) {
    const bool w_carry = w <= 0;
    w &= 0x7fff;
    const int shift = std::min(14 - find_msb(w), 14);
    const s32 normout = w32(static_cast<s64>(w) << shift) & 0x3fff;
    const s32 wnorm = normout & 0xff;
    const s32 rcp = ((kPerspTable[normout >> 8][1] * wnorm) >> 10) + kPerspTable[normout >> 8][0];
    s32 prod[2] = {w32(static_cast<s64>(s) * rcp), w32(static_cast<s64>(t) * rcp)};
    const s32 temp_mask = ((1 << 30) - 1) & -((1 << 29) >> shift);
    const s32 oob[2] = {prod[0] & temp_mask, prod[1] & temp_mask};
    s32 temp[2];
    for (int i = 0; i < 2; ++i) {
        if (shift != 14) temp[i] = prod[i] = prod[i] >> (13 - shift);
        else temp[i] = w32(static_cast<s64>(prod[i]) << 1);
    }
    bool sat[2] = {false, false};
    if (oob[0] != 0 || oob[1] != 0) {
        for (int i = 0; i < 2; ++i) {
            if (oob[i] != temp_mask && oob[i] != 0) {
                temp[i] = (prod[i] & (1 << 29)) == 0 ? 0x7fff : -0x8000;
                overflow = true;
                sat[i] = true;
            }
        }
    }
    if (w_carry) {
        temp[0] = temp[1] = 0x7fff;
        overflow = true;
        sat[0] = sat[1] = true;
    }
    if (copy) {
        out_s = sat[0] ? temp[0] : static_cast<s16>(temp[0] & 0xffff);
        out_t = sat[1] ? temp[1] : static_cast<s16>(temp[1] & 0xffff);
        return;
    }
    out_s = std::clamp(temp[0], -0x10000, 0xffff);
    out_t = std::clamp(temp[1], -0x10000, 0xffff);
}

// ---- Clamping
inline s32 clamp_9bit(s32 c) { return std::clamp(sext(c - 0x80, 9) + 0x80, 0, 0xff); }
inline s32 clamp_z(s32 z) {
    z -= 1 << 17;
    z = sext(z, 19);
    z += 1 << 17;
    return std::clamp(z, 0, 0x3ffff);
}

// ---- Depth compression: 14 bits, inverted floating point, to 18 bits.
inline s32 z_decompress(u32 z) {
    const s32 exponent = static_cast<s32>(z >> 11), mantissa = static_cast<s32>(z & 0x7ff);
    const s32 shift = std::max(6 - exponent, 0);
    const s32 base = 0x40000 - (0x40000 >> exponent);
    return (mantissa << shift) + base;
}
inline u16 z_compress(s32 z) {
    const s32 inv_z = std::max(0x3ffff - z, 1);
    const s32 exponent = std::clamp(17 - find_msb(inv_z), 0, 7);
    const s32 shift = std::max(6 - exponent, 0);
    const s32 mantissa = (z >> shift) & 0x7ff;
    return static_cast<u16>((exponent << 11) + mantissa);
}
inline s32 dz_decompress(s32 dz) { return 1 << dz; }
inline s32 dz_compress(s32 dz) { return std::max(find_msb(dz), 0); }
inline s32 combine_dz(s32 dz) { return dz != 0 ? 1 << find_msb(dz) : 0; }

s32 normalize_dzpix(s32 dz) {
    if (dz >= 0x8000) return 0x8000;
    if (dz == 0) return 1;
    return 1 << (find_msb(dz) + 1);
}
u16 dz_compress_prim(s32 dz) {
    int val = 0;
    if (dz & 0xff00) val |= 8;
    if (dz & 0xf0f0) val |= 4;
    if (dz & 0xcccc) val |= 2;
    if (dz & 0xaaaa) val |= 1;
    return static_cast<u16>(val);
}

// ---- Dithering
const u8 kDither[2][16] = {
    {0, 6, 1, 7, 4, 2, 5, 3, 3, 5, 2, 4, 7, 1, 6, 0}, // magic square
    {0, 4, 1, 5, 4, 0, 5, 1, 3, 7, 2, 6, 7, 3, 6, 2}, // Bayer
};

// ---- Noise (the hardware's is a free-running LFSR nothing can observe
// exactly; a per-pixel hash).
struct Noise {
    u32 value = 0;
    void reseed(u32 x, u32 y, u32 prim) {
        constexpr u32 kPrime = 1103515245u;
        u32 s[3] = {x, y, prim};
        for (int r = 0; r < 3; ++r) {
            const u32 n0 = ((s[0] >> 8) ^ s[1]) * kPrime, n1 = ((s[1] >> 8) ^ s[2]) * kPrime,
                      n2 = ((s[2] >> 8) ^ s[0]) * kPrime;
            s[0] = n0;
            s[1] = n1;
            s[2] = n2;
        }
        value = (s[0] >> 16) & 0xffff;
    }
    s32 combiner() const { return static_cast<s32>(((value & 7) << 6) | 0x20); }
    s32 dither_alpha() const { return static_cast<s32>(value & 7); }
    s32 dither_color() const { return static_cast<s32>(value & 0x1ff); }
    s32 blend_threshold() const { return static_cast<s32>(value & 0xff); }
};

void dither_coefficients(int x, int y, int mode_rgb, int mode_alpha, const Noise& noise, s32& rgb_dith, s32& alpha_dith) {
    constexpr s32 kSplat = (1 << 0) | (1 << 3) | (1 << 6);
    if (mode_rgb < 2) rgb_dith = kDither[mode_rgb][(y & 3) * 4 + (x & 3)] * kSplat;
    else if (mode_rgb == 2) rgb_dith = noise.dither_color();
    else rgb_dith = 0;
    if (mode_alpha == 3) {
        alpha_dith = 0;
    } else if (mode_alpha == 2) {
        alpha_dith = noise.dither_alpha();
    } else {
        alpha_dith = mode_rgb >= 2 ? kDither[mode_rgb & 1][(y & 3) * 4 + (x & 3)] : (rgb_dith & 7);
        if (mode_alpha == 1) alpha_dith = ~alpha_dith & 7;
    }
}

void rgb_dither(s32 rgb[3], s32 dith) {
    for (int i = 0; i < 3; ++i) {
        const s32 d = (dith >> (3 * i)) & 7;
        const s32 orig = rgb[i];
        const s32 up = orig > 247 ? 255 : (orig & 0xf8) + 8;
        const s32 replace = (d - (orig & 7)) >> 31;
        rgb[i] = (orig + ((up - orig) & replace)) & 0xff;
    }
}

// ---- Coverage: 8 samples, two per sub-scanline:
//   0x01 . 0x02 .
//   . 0x04 . 0x08
//   0x10 . 0x20 .
//   . 0x40 . 0x80
u32 compute_coverage(const s32 xl[4], const s32 xr[4], int x) {
    static const int kOffset[8] = {0, 4, 2, 6, 0, 4, 2, 6};
    static const int kSub[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    const s32 base = (x << 3) & 0xffff;
    u32 cvg = 0;
    for (int i = 0; i < 8; ++i) {
        const s32 xs = (base + kOffset[i]) & 0xffff;
        if (xs >= xl[kSub[i]] && xs < xr[kSub[i]]) cvg |= 1u << i;
    }
    return cvg;
}

s32 blend_coverage(s32 coverage, s32 memory_coverage, bool blend_en, int mode) {
    switch (mode) {
        case 0: return blend_en ? std::min(7, memory_coverage + coverage) : (coverage - 1) & 7; // clamp
        case 1: return (coverage + memory_coverage) & 7;                                         // wrap
        case 2: return 7;                                                                        // zap
        default: return memory_coverage;                                                         // save
    }
}

// ---- Combiner
inline s32 special_expand(s32 v) { return sext(v - 0x80, 9) + 0x80; }
inline s32 combiner_equation(s32 a, s32 b, s32 c, s32 d) {
    c = sext(c, 9);
    a = special_expand(a);
    b = special_expand(b);
    d = special_expand(d);
    s32 color = w32(static_cast<s64>(a - b) * c);
    color += 0x80;
    return (color >> 8) + d;
}

} // namespace

u8 ExactRdp::hidden_at(size_t h, u16 word) const {
    // What the RDP didn't write - or what was written over since - has the
    // ninth bits a CPU write gives each byte: its least significant bit.
    const HiddenStore& st = *hs_;
    if (h >= st.bits.size() || (st.bits[h] & 4) || st.word[h] != word)
        return static_cast<u8>((((word >> 8) & 1) << 1) | (word & 1));
    return st.bits[h];
}

void ExactRdp::cpu_wrote(u32 paddr, u32 len) {
    HiddenStore& st = *hs_;
    if (len == 0 || st.watched.empty()) return;
    const u32 last = paddr + len - 1;
    for (u32 pg = paddr >> 6; pg <= (last >> 6) && pg < st.watched.size(); ++pg) {
        if (!st.watched[pg]) continue;
        const u32 lo = std::max(paddr, pg << 6), hi = std::min(last, (pg << 6) + 63);
        for (u32 h = lo >> 1; h <= (hi >> 1); ++h) st.bits[h] = 4;
    }
}

void ExactRdp::rebuild_watched() {
    HiddenStore& st = *hs_;
    st.watched.assign(st.bits.size() / 32, 0);
    for (size_t h = 0; h < st.bits.size(); ++h) {
        if (st.bits[h] & 4) continue;
        u8& w = st.watched[h >> 5];
        if (!w) {
            w = 1;
            jit::watch_rdram(static_cast<u32>(h >> 5) << 6, 64);
        }
    }
}

UpStore::UpStore(u32 scale_, size_t rdram_size) : scale(scale_), size(rdram_size) {
    const size_t slices = static_cast<size_t>(scale) * scale;
    color_bytes_ = slices * size;
    hidden_bytes_ = slices * (size / 2);
    ref_bytes_ = size;
    color_ = static_cast<u8*>(lazy_alloc(color_bytes_));
    hidden_ = static_cast<u8*>(lazy_alloc(hidden_bytes_));
    ref = static_cast<u8*>(lazy_alloc(ref_bytes_));
    if (!color_ || !hidden_ || !ref) {
        std::fprintf(stderr, "ExactRdp: can't reserve %zu MB for %ux internal resolution\n",
                     (color_bytes_ + hidden_bytes_ + ref_bytes_) >> 20, scale);
        std::abort();
    }
}

UpStore::~UpStore() {
    lazy_free(color_, color_bytes_);
    lazy_free(hidden_, hidden_bytes_);
    lazy_free(ref, ref_bytes_);
}

UpStore* ExactRdp::upscaled(size_t rdram_size) {
    if (scale_ <= 1) return nullptr;
    if (!up_ || up_->size != rdram_size) up_ = std::make_shared<UpStore>(scale_, rdram_size);
    return up_.get();
}

void ExactRdp::up_sync_before(u64 lo, u64 hi, const u8* rdram, size_t rdram_size) {
    UpStore* up = upscaled(rdram_size);
    if (!up) return;
    lo &= ~static_cast<u64>(63);
    hi = std::min<u64>((hi + 63) & ~static_cast<u64>(63), rdram_size);
    const u32 slices = up->scale * up->scale;
    for (u64 a = lo; a < hi; a += 64) {
        if (std::memcmp(rdram + a, up->ref + a, 64) == 0) continue;
        for (u64 b = a; b < a + 64; b += 2) {
            if (rdram[b] == up->ref[b] && rdram[b + 1] == up->ref[b + 1]) continue;
            const size_t i = static_cast<size_t>(b >> 1);
            const u16 w = static_cast<u16>((rdram[b] << 8) | rdram[b + 1]);
            const u8 bits = hidden_at(i, w);
            for (u32 sl = 0; sl < slices; ++sl) {
                u8* c = up->color(sl) + b;
                c[0] = rdram[b];
                c[1] = rdram[b + 1];
                up->hidden(sl)[i] = bits;
            }
        }
        std::memcpy(up->ref + a, rdram + a, 64);
    }
}

void ExactRdp::up_sync_after(u64 lo, u64 hi, const u8* rdram, size_t rdram_size) {
    UpStore* up = upscaled(rdram_size);
    if (!up) return;
    hi = std::min<u64>(hi, rdram_size);
    if (lo < hi) std::memcpy(up->ref + lo, rdram + lo, static_cast<size_t>(hi - lo));
}

ExactRdp::ExactRdp() {
    reset();
}

void ExactRdp::reset() {
    tiles_ = {};
    tmem_.fill(0);
    hs_->bits.clear();
    hs_->word.clear();
    hs_->watched.clear();
    hs_->tail_serial = 0;
    std::fill(std::begin(hs_->tail_mem), std::end(hs_->tail_mem), 0);
    prim_count_ = 0;
    other_h_ = other_l_ = 0;
    combine_w0_ = combine_w1_ = 0;
    fill_color_ = fog_color_ = blend_color_ = prim_color_ = env_color_ = 0;
    prim_min_level_ = prim_lod_frac_ = 0;
    prim_depth_ = 0;
    prim_dz_ = 0;
    convert_.fill(0);
    key_center_.fill(0);
    key_scale_.fill(0);
    key_width_.fill(0);
    scissor_xlo_ = scissor_ylo_ = scissor_xhi_ = scissor_yhi_ = 0;
    scissor_field_ = scissor_odd_ = false;
    ti_addr_ = 0;
    ti_width_ = 1;
    ti_size_ = ti_fmt_ = 0;
    ci_addr_ = 0;
    ci_width_ = 1;
    ci_fmt_ = FB_RGBA5551;
    zi_addr_ = 0;
    std::fill(std::begin(prev_mem_), std::end(prev_mem_), 0);
    std::fill(std::begin(prev_mem_up_), std::end(prev_mem_up_), 0);
    mirror_ = false;
    // (The high-resolution copy stays: it is brought up to date with RDRAM
    // before anything is drawn over it.)
}

u32 ExactRdp::command_words(u32 op) {
    if (op >= 0x08 && op <= 0x0F) return 4 + ((op & 4) ? 8 : 0) + ((op & 2) ? 8 : 0) + ((op & 1) ? 2 : 0);
    if (op == 0x24 || op == 0x25) return 2;
    return 1;
}

namespace {

// Everything a primitive's pixels read, decoded once per primitive.
struct Prim {
    // Other modes
    int cycle = 0; // 0 1-cycle, 1 2-cycle, 2 copy, 3 fill
    bool persp = false, detail = false, sharpen = false, tex_lod = false, tlut = false, tlut_type = false;
    bool sample_quad = false, mid_texel = false, bilerp0 = false, bilerp1 = false, convert_one = false;
    bool force_blend = false, alpha_cvg_select = false, cvg_times_alpha = false, color_on_cvg = false;
    bool image_read = false, z_update = false, z_compare = false, aa = false;
    bool alpha_test_dither = false, alpha_test = false, dither_en = false, key_en = false;
    int dither = 0, coverage_mode = 0, z_mode = 0;
    bool interlace = false;
    u8 blend[2][4]{}; // per blender cycle: 1a, 1b, 2a, 2b
    u8 rgb[2][4]{};   // per combiner cycle: muladd, mulsub, mul, add
    u8 alpha[2][4]{};
    bool uses_texel0 = false, uses_texel1 = false, uses_pipelined_texel1 = false, uses_lod = false;
    bool need_noise = false, need_noise_dual = false;

    // Derived
    s32 dz = 0, dz_compressed = 0;
    s32 min_lod = 0;
    s32 factors[4]{};
    u32 prim_color = 0, env_color = 0, fog_color = 0, blend_color = 0, fill_color = 0;
    u8 prim_lod_frac = 0;
    u8 key_center[3]{}, key_scale[3]{};
    s32 key_width[3]{};
    s32 k4 = 0, k5 = 0;
    u32 prim_serial = 0;

    // Framebuffer
    u8 fb_fmt = FB_RGBA5551;
    u32 fb_index = 0; // colour image address in pixels
    u32 fb_width = 1;
    u32 z_index = 0;  // depth image address in halfwords
    bool alias = false;
};

bool accesses_texel0(const u8 rgb[4], const u8 a[4]) {
    return rgb[0] == 1 || rgb[1] == 1 || rgb[2] == 1 || rgb[3] == 1 || rgb[2] == 8 || a[0] == 1 || a[1] == 1 ||
           a[2] == 1 || a[3] == 1;
}
bool accesses_texel1(const u8 rgb[4], const u8 a[4]) {
    return rgb[0] == 2 || rgb[1] == 2 || rgb[2] == 2 || rgb[3] == 2 || rgb[2] == 9 || a[0] == 2 || a[1] == 2 ||
           a[2] == 2 || a[3] == 2;
}
bool accesses_lod_frac(const u8 rgb[4], const u8 a[4]) { return rgb[2] == 13 || a[2] == 0; }

struct Span {
    s32 rgba[4]{}, stzw[4]{};
    s32 xleft[4]{}, xright[4]{};
    s32 base_x = 0, start_x = 0, end_x = 0, lodlength = 0;
    bool valid = false;
};

} // namespace

// ============================================================================
// Pixel pipeline
// ============================================================================

namespace {

struct Tex {
    const std::array<u16, 2048>& tmem;
    u8 byte(u32 b) const {
        const u16 h = tmem[(b >> 1) & 0x7ff];
        return static_cast<u8>((b & 1) ? h & 0xff : h >> 8);
    }
    u16 half(u32 h) const { return tmem[h & 0x7ff]; }
};

struct T4 {
    s32 v[4]{};
};

T4 convert_rgba16(u32 word) {
    T4 r;
    for (int i = 0; i < 3; ++i) {
        u32 c = (word >> (11 - 5 * i)) & 31;
        r.v[i] = static_cast<s32>((c << 3) | (c >> 2));
    }
    r.v[3] = (word & 1) ? 0xff : 0;
    return r;
}
T4 convert_ia16(u32 word) {
    const s32 i = static_cast<s32>(word >> 8);
    return {{i, i, i, static_cast<s32>(word & 0xff)}};
}
T4 splat(s32 v) { return {{v, v, v, v}}; }

template <class TileT>
u32 texel_byte_offset(const TileT& tile, s32 s, s32 t, u32 x_bytes, u32 wrap) {
    u32 off = tile.offset + tile.stride * static_cast<u32>(t);
    off += x_bytes;
    off &= wrap;
    return off ^ ((static_cast<u32>(t) & 1) << 2);
    (void)s;
}

template <class TileT>
T4 texel_rgba4(const Tex& tx, const TileT& tile, s32 s, s32 t) {
    const u32 b = texel_byte_offset(tile, s, t, static_cast<u32>(s) >> 1, 0xfff);
    u32 w = (tx.byte(b) >> ((~s & 1) * 4)) & 0xf;
    w |= w << 4;
    return splat(static_cast<s32>(w));
}
template <class TileT>
T4 texel_ia4(const Tex& tx, const TileT& tile, s32 s, s32 t) {
    const u32 b = texel_byte_offset(tile, s, t, static_cast<u32>(s) >> 1, 0xfff);
    const u32 w = (tx.byte(b) >> ((~s & 1) * 4)) & 0xf;
    u32 i = w & 0xe;
    i = (i << 4) | (i << 1) | (i >> 2);
    return {{static_cast<s32>(i), static_cast<s32>(i), static_cast<s32>(i), static_cast<s32>((w & 1) * 0xff)}};
}
template <class TileT>
T4 texel_ci4(const Tex& tx, const TileT& tile, s32 s, s32 t, u32 pal) {
    const u32 b = texel_byte_offset(tile, s, t, static_cast<u32>(s) >> 1, 0xfff);
    u32 w = (tx.byte(b) >> ((~s & 1) * 4)) & 0xf;
    w |= pal << 4;
    return splat(static_cast<s32>(w));
}
// TLUT entries sit in the upper half, each repeated in the four banks;
// `lut_offset` picks the bank, the high texel of a quad reads them mirrored.
u32 tlut_entry(const Tex& tx, u32 entry, u32 lut_offset, bool upper) {
    const u32 idx = ((entry << 2) + lut_offset) ^ (upper ? 3u : 0u);
    return tx.half(0x400 | (idx & 0x3ff));
}
template <class TileT>
T4 texel_ci4_tlut(const Tex& tx, const TileT& tile, s32 s, s32 t, u32 pal, u32 lut_offset, bool upper, bool ia) {
    const u32 b = texel_byte_offset(tile, s, t, static_cast<u32>(s) >> 1, 0x7ff);
    u32 w = (tx.byte(b) >> ((~s & 1) * 4)) & 0xf;
    w |= pal << 4;
    const u32 word = tlut_entry(tx, w, lut_offset, upper);
    return ia ? convert_ia16(word) : convert_rgba16(word);
}
template <class TileT>
T4 texel_ci8_tlut(const Tex& tx, const TileT& tile, s32 s, s32 t, u32 lut_offset, bool upper, bool ia) {
    const u32 b = texel_byte_offset(tile, s, t, static_cast<u32>(s), 0x7ff);
    const u32 word = tlut_entry(tx, tx.byte(b), lut_offset, upper);
    return ia ? convert_ia16(word) : convert_rgba16(word);
}
template <class TileT>
u32 texel_half(const Tex& tx, const TileT& tile, s32 s, s32 t, u32 wrap) {
    u32 off = tile.offset + tile.stride * static_cast<u32>(t);
    off += static_cast<u32>(s) * 2;
    off &= wrap;
    return tx.half((off >> 1) ^ ((static_cast<u32>(t) & 1) << 1));
}
template <class TileT>
T4 texel_ci32(const Tex& tx, const TileT& tile, s32 s, s32 t) {
    const u32 w = texel_half(tx, tile, s, t, 0xfff);
    const s32 hi = static_cast<s32>(w >> 8), lo = static_cast<s32>(w & 0xff);
    return {{hi, lo, hi, lo}};
}
template <class TileT>
T4 texel_ci32_tlut(const Tex& tx, const TileT& tile, s32 s, s32 t, u32 lut_offset, bool upper, bool ia) {
    const u32 w = texel_half(tx, tile, s, t, 0x7ff);
    const u32 word = tlut_entry(tx, w >> 8, lut_offset, upper);
    return ia ? convert_ia16(word) : convert_rgba16(word);
}
template <class TileT>
T4 texel_rgba8(const Tex& tx, const TileT& tile, s32 s, s32 t) {
    const u32 b = texel_byte_offset(tile, s, t, static_cast<u32>(s), 0xfff);
    return splat(tx.byte(b));
}
template <class TileT>
T4 texel_ia8(const Tex& tx, const TileT& tile, s32 s, s32 t) {
    const u32 b = texel_byte_offset(tile, s, t, static_cast<u32>(s), 0xfff);
    const u32 w = tx.byte(b);
    u32 i = w >> 4, a = w & 0xf;
    a |= a << 4;
    i |= i << 4;
    return {{static_cast<s32>(i), static_cast<s32>(i), static_cast<s32>(i), static_cast<s32>(a)}};
}
template <class TileT>
T4 texel_yuv16(const Tex& tx, const TileT& tile, s32 s, s32 t, u32 chroma_x) {
    const u32 base = tile.offset + tile.stride * static_cast<u32>(t);
    const u32 luma_b = ((base + static_cast<u32>(s)) & 0x7ff) ^ ((static_cast<u32>(t) & 1) << 2);
    const u32 chroma_b = (base + chroma_x * 2) & 0x7ff;
    const u32 chroma_h = (chroma_b >> 1) ^ ((static_cast<u32>(t) & 1) << 1);
    const s32 luma = tx.byte(luma_b | 0x800);
    const u32 chroma = tx.half(chroma_h);
    return {{static_cast<s32>((chroma >> 8) & 0xff) - 0x80, static_cast<s32>(chroma & 0xff) - 0x80, luma, luma}};
}
template <class TileT>
T4 texel_rgba16(const Tex& tx, const TileT& tile, s32 s, s32 t) {
    return convert_rgba16(texel_half(tx, tile, s, t, 0xfff));
}
template <class TileT>
T4 texel_ia16(const Tex& tx, const TileT& tile, s32 s, s32 t) {
    return convert_ia16(texel_half(tx, tile, s, t, 0xfff));
}
template <class TileT>
T4 texel_rgba32(const Tex& tx, const TileT& tile, s32 s, s32 t) {
    u32 off = tile.offset + tile.stride * static_cast<u32>(t);
    off += static_cast<u32>(s) * 2;
    off &= 0x7ff;
    const u32 h = (off >> 1) ^ ((static_cast<u32>(t) & 1) << 1);
    const u32 lo = tx.half(h), hi = tx.half(h | 0x400);
    return {{static_cast<s32>(lo >> 8), static_cast<s32>(lo & 0xff), static_cast<s32>(hi >> 8), static_cast<s32>(hi & 0xff)}};
}

template <class TileT>
s32 mask_s(const TileT& tile, s32 s) {
    if (tile.mask_s != 0) {
        const s32 mask = 1 << tile.mask_s;
        if (tile.flags & TILE_MIRROR_S) s ^= std::max((s & mask) - 1, 0);
        s &= mask - 1;
    }
    return s;
}
template <class TileT>
s32 mask_t(const TileT& tile, s32 t) {
    if (tile.mask_t != 0) {
        const s32 mask = 1 << tile.mask_t;
        if (tile.flags & TILE_MIRROR_T) t ^= std::max((t & mask) - 1, 0);
        t &= mask - 1;
    }
    return t;
}

s32 clamp_and_shift(bool clamp_bit, s32 coord, s32 lo, s32 hi, int shift) {
    coord = std::clamp(coord, -0x8000, 0x7fff);
    if (shift < 11) coord >>= shift;
    else coord = w32(static_cast<s64>(coord) << (32 - shift)) >> 16;
    if (clamp_bit) {
        if ((coord >> 3) >= hi) coord = (((hi >> 2) - (lo >> 2)) & 0x3ff) << 5;
        else coord = std::max(coord - (lo << 3), 0);
    } else {
        coord -= lo << 3;
    }
    return coord;
}
s32 shift_coord(s32 coord, s32 lo, int shift) {
    coord = std::clamp(coord, -0x8000, 0x7fff);
    if (shift < 11) coord >>= shift;
    else coord = w32(static_cast<s64>(coord) << (32 - shift)) >> 16;
    return coord - (lo << 3);
}

T4 texture_convert_factors(const T4& in, const s32 f[4]) {
    const s32 r0 = sext(in.v[0], 9), g0 = sext(in.v[1], 9), b0 = sext(in.v[2], 9);
    const s32 r = b0 + ((f[0] * g0 + 0x80) >> 8);
    const s32 g = b0 + ((f[1] * r0 + f[2] * g0 + 0x80) >> 8);
    const s32 b = b0 + ((f[3] * r0 + 0x80) >> 8);
    return {{r, g, b, b0}};
}

s32 bilinear_3tap1(s32 t00, s32 t10, s32 t01, s32 t11, s32 fs, s32 ft) {
    const s32 sum = fs + ft;
    const s32 base = sum >= 32 ? t11 : t00;
    const s32 f0 = sum >= 32 ? 32 - ft : fs;
    const s32 f1 = sum >= 32 ? 32 - fs : ft;
    s32 acc = (t10 - base) * f0 + (t01 - base) * f1;
    acc += 0x10;
    acc >>= 5;
    return acc + base;
}

template <class TileT>
T4 sample_texture(const Tex& tx, const TileT& tile, s32 s_in, s32 t_in, bool tlut, bool tlut_type, bool sample_quad,
                  bool mid_texel_state, bool convert_one, bool bilerp, const s32 factors[4], const T4& prev) {
    s32 s = clamp_and_shift((tile.flags & TILE_CLAMP_S) != 0, s_in, static_cast<s32>(tile.slo), static_cast<s32>(tile.shi), tile.shift_s);
    s32 t = clamp_and_shift((tile.flags & TILE_CLAMP_T) != 0, t_in, static_cast<s32>(tile.tlo), static_cast<s32>(tile.thi), tile.shift_t);

    s32 fs = 0, ft = 0;
    if (sample_quad || tlut) {
        fs = s & 31;
        ft = t & 31;
    }
    s32 sum_frac = fs + ft;
    s >>= 5;
    t >>= 5;

    s32 s0 = mask_s(tile, s), t0 = mask_t(tile, t);
    s32 s1 = mask_s(tile, s + 1), t1 = mask_t(tile, t + 1);
    const s32 tdiff = std::max(t1 - t0, -255);
    t1 = (t0 & 0xff) + tdiff;
    t0 &= 0xff;

    T4 tb, t10, t01, t11;
    const bool mid_texel = mid_texel_state && bilerp && fs == 0x10 && ft == 0x10;
    const bool upper_lut = sum_frac >= 0x20;
    if (mid_texel) sum_frac = 0;

    const bool yuv = tile.fmt == TF_YUV;
    s32 bs = sum_frac >= 0x20 ? s1 : s0, bt = sum_frac >= 0x20 ? t1 : t0;
    const s32 chroma_frac = ((s0 & 1) << 4) | (fs >> 1);

    if (tlut) {
        if (!sample_quad) {
            bs = s0;
            bt = t0;
            s1 = s0;
            t1 = t0;
        }
        if (tile.fmt == TF_RGBA || tile.fmt == TF_CI || tile.fmt == TF_IA || tile.fmt == TF_I) {
            const bool upper = sum_frac >= 0x20;
            const u32 pal = tile.palette;
            auto fetch = [&](s32 ss, s32 tt, u32 off) {
                switch (tile.size) {
                    case 0: return texel_ci4_tlut(tx, tile, ss, tt, pal, off, upper_lut, tlut_type);
                    case 1: return texel_ci8_tlut(tx, tile, ss, tt, off, upper_lut, tlut_type);
                    default: return texel_ci32_tlut(tx, tile, ss, tt, off, upper_lut, tlut_type);
                }
            };
            tb = fetch(bs, bt, upper ? 3 : 0);
            if (bilerp) {
                t10 = fetch(s1, t0, 1);
                t01 = fetch(s0, t1, 2);
            }
            if (mid_texel) t11 = fetch(s1, t1, 3);
        }
    } else {
        auto fetch = [&](s32 ss, s32 tt) -> T4 {
            switch (tile.fmt) {
                case TF_RGBA:
                    switch (tile.size) {
                        case 0: return texel_rgba4(tx, tile, ss, tt);
                        case 1: return texel_rgba8(tx, tile, ss, tt);
                        case 2: return texel_rgba16(tx, tile, ss, tt);
                        default: return texel_rgba32(tx, tile, ss, tt);
                    }
                case TF_CI:
                    switch (tile.size) {
                        case 0: return texel_ci4(tx, tile, ss, tt, tile.palette);
                        case 1: return texel_rgba8(tx, tile, ss, tt);
                        default: return texel_ci32(tx, tile, ss, tt);
                    }
                case TF_IA:
                    switch (tile.size) {
                        case 0: return texel_ia4(tx, tile, ss, tt);
                        case 1: return texel_ia8(tx, tile, ss, tt);
                        case 2: return texel_ia16(tx, tile, ss, tt);
                        default: return texel_ci32(tx, tile, ss, tt);
                    }
                case TF_I:
                    switch (tile.size) {
                        case 0: return texel_rgba4(tx, tile, ss, tt);
                        case 1: return texel_rgba8(tx, tile, ss, tt);
                        default: return texel_ci32(tx, tile, ss, tt);
                    }
                default: return T4{}; // formats 5-7 read nothing
            }
        };
        if (yuv) {
            const u32 cx0 = static_cast<u32>(s0 >> 1);
            const u32 cx1 = static_cast<u32>((s1 + (s1 - s0)) >> 1);
            tb = texel_yuv16(tx, tile, s0, t0, cx0);
            if (sample_quad) {
                t10 = texel_yuv16(tx, tile, s1, t0, cx1);
                t01 = texel_yuv16(tx, tile, s0, t1, cx0);
                t11 = texel_yuv16(tx, tile, s1, t1, cx1);
            }
        } else {
            tb = fetch(bs, bt);
            if (sample_quad) {
                t10 = fetch(s1, t0);
                t01 = fetch(s0, t1);
            }
            if (mid_texel) t11 = fetch(s1, t1);
        }
    }

    T4 acc;
    if (convert_one) {
        s32 p[4];
        for (int i = 0; i < 4; ++i) p[i] = sext(prev.v[i], 9);
        if (sample_quad) {
            const bool mid_rg = yuv ? (mid_texel_state && chroma_frac == 0x10 && ft == 0x10) : mid_texel;
            const bool mid_ba = mid_texel;
            const bool upper_ba = sum_frac >= 32;
            const bool upper_rg = yuv ? ((chroma_frac + ft) >= 32 && !mid_rg) : upper_ba;
            const s32 frg0 = upper_rg ? p[1] : p[0], frg1 = upper_rg ? p[0] : p[1];
            const s32 fba0 = upper_ba ? p[1] : p[0], fba1 = upper_ba ? p[0] : p[1];
            s32 conv[4];
            for (int i = 0; i < 2; ++i) {
                if (mid_rg) {
                    conv[i] = frg0 * (t01.v[i] - t11.v[i]) + frg1 * (t10.v[i] - t11.v[i]) + ((tb.v[i] - t11.v[i]) << 6) + 0x80;
                } else {
                    const s32 base = upper_rg && yuv ? t11.v[i] : tb.v[i];
                    conv[i] = frg0 * (t10.v[i] - base) + frg1 * (t01.v[i] - base) + 0x80;
                }
            }
            for (int i = 2; i < 4; ++i) {
                if (mid_ba) {
                    conv[i] = fba0 * (t01.v[i] - t11.v[i]) + fba1 * (t10.v[i] - t11.v[i]) + ((tb.v[i] - t11.v[i]) << 6) + 0x80;
                } else {
                    const s32 base = upper_ba && yuv ? t11.v[i] : tb.v[i];
                    conv[i] = fba0 * (t10.v[i] - base) + fba1 * (t01.v[i] - base) + 0x80;
                }
            }
            for (int i = 0; i < 4; ++i) acc.v[i] = (conv[i] >> 8) + p[2];
        } else {
            acc = splat(p[2]);
        }
    } else if (yuv) {
        if (sample_quad) {
            if (bilerp) {
                const bool mid_chroma = mid_texel_state && chroma_frac == 0x10 && ft == 0x10;
                for (int i = 0; i < 2; ++i) {
                    acc.v[i] = mid_chroma ? (tb.v[i] + t10.v[i] + t11.v[i] + t01.v[i] + 2) >> 2
                                          : bilinear_3tap1(tb.v[i], t10.v[i], t01.v[i], t11.v[i], chroma_frac, ft);
                }
                for (int i = 2; i < 4; ++i) {
                    acc.v[i] = mid_texel ? (tb.v[i] + t10.v[i] + t11.v[i] + t01.v[i] + 2) >> 2
                                         : bilinear_3tap1(tb.v[i], t10.v[i], t01.v[i], t11.v[i], fs, ft);
                }
            } else {
                for (int i = 2; i < 4; ++i) acc.v[i] = fs + ft >= 32 ? t11.v[i] : tb.v[i];
                for (int i = 0; i < 2; ++i) acc.v[i] = chroma_frac + ft >= 32 ? t11.v[i] : tb.v[i];
            }
        } else {
            acc = tb;
        }
    } else if (mid_texel) {
        for (int i = 0; i < 4; ++i) acc.v[i] = (tb.v[i] + t01.v[i] + t10.v[i] + t11.v[i] + 2) >> 2;
    } else if (bilerp && (sample_quad || tlut)) {
        const s32 f0 = sum_frac >= 32 ? 32 - ft : fs;
        const s32 f1 = sum_frac >= 32 ? 32 - fs : ft;
        for (int i = 0; i < 4; ++i) {
            s32 a = (t10.v[i] - tb.v[i]) * f0 + (t01.v[i] - tb.v[i]) * f1;
            a += 0x10;
            a >>= 5;
            acc.v[i] = a + tb.v[i];
        }
    } else {
        acc = tb;
    }

    if (!bilerp && !convert_one) acc = texture_convert_factors(acc, factors);
    return acc;
}

void compute_lod(u32& tile0, u32& tile1, s32& lod_frac, u32 max_level, s32 min_lod, s32 s, s32 t, s32 sdx, s32 tdx,
                 s32 sdy, s32 tdy, bool persp_overflow, bool tex_lod_en, bool sharpen, bool detail) {
    bool magnify = false, distant = false;
    u32 tile_offset = 0;
    if (persp_overflow) {
        distant = true;
        lod_frac = 0xff;
    } else {
        s32 dx0 = sdx - s, dx1 = tdx - t, dy0 = sdy - s, dy1 = tdy - t;
        dx0 ^= dx0 >> 31;
        dx1 ^= dx1 >> 31;
        dy0 ^= dy0 >> 31;
        dy1 ^= dy1 >> 31;
        const s32 max_d = std::max(std::max(dx0, dy0), std::max(dx1, dy1));
        if (max_d >= 0x4000) {
            distant = true;
            lod_frac = 0xff;
            tile_offset = max_level;
        } else if (max_d < 32) {
            distant = max_level == 0;
            magnify = true;
            if (!sharpen && !detail) lod_frac = distant ? 0xff : 0;
            else lod_frac = (std::max(min_lod, max_d) << 3) + (sharpen ? -0x100 : 0);
        } else {
            const s32 mip_base = std::max(find_msb(max_d >> 5), 0);
            distant = mip_base >= static_cast<s32>(max_level);
            if (distant && !sharpen && !detail) {
                lod_frac = 0xff;
            } else {
                lod_frac = ((max_d << 3) >> mip_base) & 0xff;
                tile_offset = static_cast<u32>(mip_base);
            }
        }
    }
    if (tex_lod_en) {
        if (distant) tile_offset = max_level;
        if (!detail) {
            tile0 = (tile0 + tile_offset) & 7;
            tile1 = (distant || (!sharpen && magnify)) ? tile0 : ((tile0 + 1) & 7);
        } else {
            tile1 = (tile0 + tile_offset + ((distant || magnify) ? 1 : 2)) & 7;
            tile0 = (tile0 + tile_offset + (magnify ? 0 : 1)) & 7;
        }
    }
}

} // namespace

// ============================================================================
// Commands
// ============================================================================

bool ExactRdp::command(const u32* w, u8* rdram, size_t rdram_size) {
    if (hs_->bits.size() != rdram_size / 2) {
        // (Lanes all see the same size; lane 0 allocates before the others run.)
        hs_->bits.assign(rdram_size / 2, 4);
        hs_->word.assign(rdram_size / 2, 0);
        hs_->watched.assign(rdram_size / 64, 0);
    }
    const u32 op = (w[0] >> 24) & 0x3f;
    switch (op) {
        case 0x08: case 0x09: case 0x0a: case 0x0b: case 0x0c: case 0x0d: case 0x0e: case 0x0f: {
            const bool copy = ((other_h_ >> 20) & 3) == 2;
            Setup s;
            const bool flip = (w[0] & 0x800000u) != 0;
            const bool sign_dxhdy = (w[5] & 0x80000000u) != 0;
            s.flags |= flip ? SETUP_FLIP : 0;
            s.flags |= flip == sign_dxhdy ? SETUP_DO_OFFSET : 0;
            s.flags |= copy ? SETUP_SKIP_XFRAC : 0;
            s.tile = (w[0] >> 16) & 63;
            s.yl = sext(static_cast<s32>(w[0]), 14);
            s.ym = sext(static_cast<s32>(w[1] >> 16), 14);
            s.yh = sext(static_cast<s32>(w[1]), 14);
            s.xl = sext(static_cast<s32>(w[2]), 28) >> 1;
            s.xh = sext(static_cast<s32>(w[4]), 28) >> 1;
            s.xm = sext(static_cast<s32>(w[6]), 28) >> 1;
            s.dxldy = sext(static_cast<s32>(w[3] >> 2), 28) >> 1;
            s.dxhdy = sext(static_cast<s32>(w[5] >> 2), 28) >> 1;
            s.dxmdy = sext(static_cast<s32>(w[7] >> 2), 28) >> 1;
            Attr a;
            const u32* p = w + 8;
            // A block the command doesn't carry isn't zero: the loader latches
            // the header doubleword into all of its slots (hardware verified
            // on the diagnostic cartridge, cases 9:21 and "undefined shade").
            u32 header_block[16];
            for (int i = 0; i < 16; i += 2) {
                header_block[i] = w[0];
                header_block[i + 1] = w[1];
            }
            {
                const u32* q = (op & 4) ? p : header_block;
                for (int i = 0; i < 4; ++i) {
                    const int hw = i >> 1;
                    auto get = [&](int wi) {
                        return (i & 1) ? static_cast<s32>((q[wi + hw] << 16) | (q[wi + 4 + hw] & 0xffff))
                                       : static_cast<s32>((q[wi + hw] & 0xffff0000u) | ((q[wi + 4 + hw] >> 16) & 0xffff));
                    };
                    a.rgba[i] = get(0);
                    a.drgba_dx[i] = get(2);
                    a.drgba_de[i] = get(8);
                    a.drgba_dy[i] = get(10);
                }
                if (op & 4) p += 16;
            }
            if (op & 2) {
                // s, t, w (z is the depth block's)
                static const int kComp[3] = {0, 1, 3};
                for (int k = 0; k < 3; ++k) {
                    const int i = k; // s = 0, t = 1, w = 2 in the command's order
                    const int hw = i >> 1;
                    auto get = [&](int wi) {
                        return (i & 1) ? static_cast<s32>((p[wi + hw] << 16) | (p[wi + 4 + hw] & 0xffff))
                                       : static_cast<s32>((p[wi + hw] & 0xffff0000u) | ((p[wi + 4 + hw] >> 16) & 0xffff));
                    };
                    const int c = kComp[k];
                    a.stzw[c] = get(0);
                    a.dstzw_dx[c] = get(2);
                    a.dstzw_de[c] = get(8);
                    a.dstzw_dy[c] = get(10);
                }
                p += 16;
            }
            if (op & 1) {
                a.stzw[2] = static_cast<s32>(p[0]);
                a.dstzw_dx[2] = static_cast<s32>(p[1]);
                a.dstzw_de[2] = static_cast<s32>(p[2]);
                a.dstzw_dy[2] = static_cast<s32>(p[3]);
            } else {
                a.stzw[2] = static_cast<s32>(w[0]);
                a.dstzw_dx[2] = static_cast<s32>(w[1]);
                a.dstzw_de[2] = static_cast<s32>(w[0]);
                a.dstzw_dy[2] = static_cast<s32>(w[1]);
            }
            stale_.valid = false;
            draw(s, a, rdram, rdram_size);
            return false;
        }
        case 0x24: stale_.valid = false; texture_rectangle(w, false, rdram, rdram_size); return false;
        case 0x25: stale_.valid = false; texture_rectangle(w, true, rdram, rdram_size); return false;
        case 0x29: return true; // Sync Full
        case 0x2a: { // Set Key GB
            key_width_[1] = (w[0] >> 12) & 0xfff;
            key_width_[2] = w[0] & 0xfff;
            key_center_[1] = (w[1] >> 24) & 0xff;
            key_scale_[1] = (w[1] >> 16) & 0xff;
            key_center_[2] = (w[1] >> 8) & 0xff;
            key_scale_[2] = w[1] & 0xff;
            return false;
        }
        case 0x2b: // Set Key R
            key_width_[0] = (w[1] >> 16) & 0xfff;
            key_center_[0] = (w[1] >> 8) & 0xff;
            key_scale_[0] = w[1] & 0xff;
            return false;
        case 0x2c: { // Set Convert
            const u64 m = (static_cast<u64>(w[0]) << 32) | w[1];
            for (int i = 0; i < 4; ++i) convert_[i] = 2 * sext(static_cast<s32>((m >> (45 - 9 * i)) & 0x1ff), 9) + 1;
            convert_[4] = static_cast<s32>((m >> 9) & 0x1ff);
            convert_[5] = static_cast<s32>(m & 0x1ff);
            return false;
        }
        case 0x2d: // Set Scissor
            scissor_xlo_ = (w[0] >> 12) & 0xfff;
            scissor_ylo_ = w[0] & 0xfff;
            scissor_xhi_ = (w[1] >> 12) & 0xfff;
            scissor_yhi_ = w[1] & 0xfff;
            scissor_field_ = (w[1] >> 25) & 1;
            scissor_odd_ = (w[1] >> 24) & 1;
            return false;
        case 0x2e: // Set Prim Depth
            prim_depth_ = static_cast<s32>(((w[1] >> 16) & 0x7fff) << 16);
            prim_dz_ = static_cast<u16>(w[1] & 0xffff);
            return false;
        case 0x2f: // Set Other Modes
            other_h_ = w[0] & 0x00ffffff;
            other_l_ = w[1];
            stale_.valid = false;
            return false;
        case 0x30: load(w, LoadMode::Tlut, rdram, rdram_size); return false;
        case 0x32: { // Set Tile Size
            Tile& t = tiles_[(w[1] >> 24) & 7];
            t.slo = (w[0] >> 12) & 0xfff;
            t.tlo = w[0] & 0xfff;
            t.shi = (w[1] >> 12) & 0xfff;
            t.thi = w[1] & 0xfff;
            return false;
        }
        case 0x33: load(w, LoadMode::Block, rdram, rdram_size); return false;
        case 0x34: load(w, LoadMode::Tile, rdram, rdram_size); return false;
        case 0x35: { // Set Tile
            Tile& t = tiles_[(w[1] >> 24) & 7];
            t.offset = (w[0] & 511) << 3;
            t.stride = ((w[0] >> 9) & 511) << 3;
            t.size = (w[0] >> 19) & 3;
            t.fmt = (w[0] >> 21) & 7;
            t.palette = (w[1] >> 20) & 15;
            t.shift_s = w[1] & 15;
            t.raw_mask_s = (w[1] >> 4) & 15;
            t.shift_t = (w[1] >> 10) & 15;
            t.raw_mask_t = (w[1] >> 14) & 15;
            t.flags = 0;
            if (w[1] & (1u << 8)) t.flags |= TILE_MIRROR_S;
            if (w[1] & (1u << 9)) t.flags |= TILE_CLAMP_S;
            if (w[1] & (1u << 18)) t.flags |= TILE_MIRROR_T;
            if (w[1] & (1u << 19)) t.flags |= TILE_CLAMP_T;
            t.mask_s = t.raw_mask_s > 10 ? 10 : t.raw_mask_s;
            if (t.raw_mask_s == 0) t.flags |= TILE_CLAMP_S;
            t.mask_t = t.raw_mask_t > 10 ? 10 : t.raw_mask_t;
            if (t.raw_mask_t == 0) t.flags |= TILE_CLAMP_T;
            return false;
        }
        case 0x36: fill_rectangle(w, rdram, rdram_size); return false;
        case 0x37: fill_color_ = w[1]; return false;
        case 0x38: fog_color_ = w[1]; return false;
        case 0x39: blend_color_ = w[1]; return false;
        case 0x3a:
            prim_min_level_ = (w[0] >> 8) & 31;
            prim_lod_frac_ = w[0] & 0xff;
            prim_color_ = w[1];
            return false;
        case 0x3b: env_color_ = w[1]; return false;
        case 0x3c:
            combine_w0_ = w[0] & 0x00ffffff;
            combine_w1_ = w[1];
            return false;
        case 0x3d: // Set Texture Image
            ti_fmt_ = (w[0] >> 21) & 7;
            ti_size_ = (w[0] >> 19) & 3;
            ti_width_ = (w[0] & 0x3ff) + 1;
            ti_addr_ = w[1] & 0x00ffffff;
            return false;
        case 0x3e: zi_addr_ = w[1] & 0x00ffffff; return false; // Set Mask (depth) Image
        case 0x3f: { // Set Color Image
            const u32 fmt = (w[0] >> 21) & 7, size = (w[0] >> 19) & 3;
            ci_width_ = (w[0] & 1023) + 1;
            ci_addr_ = w[1] & 0x00ffffff;
            switch (size) {
                case 0: ci_fmt_ = FB_I4; break;
                case 1: ci_fmt_ = FB_I8; break;
                case 2: ci_fmt_ = fmt ? FB_IA88 : FB_RGBA5551; break;
                default: ci_fmt_ = FB_RGBA8888; break;
            }
            return false;
        }
        default: // no-ops, syncs, invalid commands
            return false;
    }
}

void ExactRdp::fill_rectangle(const u32* w, u8* rdram, size_t rdram_size) {
    const u32 xl = (w[0] >> 12) & 0xfff;
    u32 yl = w[0] & 0xfff;
    const u32 xh = (w[1] >> 12) & 0xfff, yh = w[1] & 0xfff;
    const u32 cycle = (other_h_ >> 20) & 3;

    // Stale reads. A span costs L clocks and the command processor leads the
    // pixel pipeline by D; when D > L a rectangle's framebuffer reads come
    // before its predecessor's writes to the same pixels land, so an
    // identical rectangle right behind it blends over the memory image from
    // before it. Modelled for what the diagnostic cartridge covers (cases
    // 12:15 and 12:16, after cen64): identical single-row 1-/2-cycle
    // rectangles up to 32 pixels wide, image read on, 16-bit colour image.
    const auto rd16 = [&](u32 h) -> u16 {
        const size_t a = (static_cast<size_t>(h) * 2) & (rdram_size - 1);
        return static_cast<u16>((rdram[a] << 8) | rdram[a + 1]);
    };
    const auto wr16 = [&](u32 h, u16 v) {
        const size_t a = (static_cast<size_t>(h) * 2) & (rdram_size - 1);
        rdram[a] = static_cast<u8>(v >> 8);
        rdram[a + 1] = static_cast<u8>(v);
    };
    u16 stale_cur[32]{};
    bool stale = false;
    {
        const u32 W = (xl >> 2) - (xh >> 2), H = (yl >> 2) - (yh >> 2);
        const u32 cycn = cycle == 1 ? 2 : 1;
        u32 L = cycn * W + cycn - 1;
        if (L < 4) L = 4;
        const u32 D = std::min<u32>(3 * L - 2, 25);
        const bool atomic_prim = (other_h_ >> 23) & 1;
        const bool image_read = (other_l_ >> 6) & 1;
        const bool sixteen = ci_fmt_ == FB_RGBA5551 || ci_fmt_ == FB_IA88;
        const bool ok = cycle < 2 && image_read && !atomic_prim && xl >= xh && yl >= yh && H == 1 && W >= 1 &&
                        W <= 32 && D > L && sixteen;
        // Only the lane that draws the row touches memory.
        const bool owner = lanes_ == 1 || (yh >> 2) % lanes_ == lane_;
        if (!ok) {
            stale_.valid = false;
        } else if (stale_.valid && stale_.n && stale_.w0 == w[0] && stale_.w1 == w[1]) {
            for (u32 i = 0; owner && i < stale_.n; ++i) {
                stale_cur[i] = rd16(stale_.idx[i]);
                wr16(stale_.idx[i], stale_.pre[i]);
            }
            stale = true;
        } else {
            stale_.n = static_cast<u8>(W);
            const u32 y0 = yh >> 2;
            for (u32 i = 0; i < W; ++i) {
                stale_.idx[i] = (ci_addr_ >> 1) + y0 * ci_width_ + (xh >> 2) + i;
                stale_.pre[i] = owner ? rd16(stale_.idx[i]) : 0;
            }
            stale_.w0 = w[0];
            stale_.w1 = w[1];
            stale_.valid = true;
        }
    }

    if (cycle >= 2) yl |= 3;
    Setup s;
    s.xh = static_cast<s32>(xh << 13);
    s.xl = static_cast<s32>(xl << 13);
    s.xm = static_cast<s32>(xl << 13);
    s.ym = static_cast<s32>(yl);
    s.yl = static_cast<s32>(yl);
    s.yh = static_cast<s32>(yh);
    s.flags = SETUP_FLIP;
    // At internal resolution, drawn natively and repeated (as parallel-rdp does).
    mirror_ = true;
    draw(s, Attr{}, rdram, rdram_size);
    mirror_ = false;
    // The next stale reader sees this rectangle's pre-image: the
    // predecessor's output saved above.
    if (stale)
        for (u32 k = 0; k < stale_.n; ++k) stale_.pre[k] = stale_cur[k];
}

void ExactRdp::texture_rectangle(const u32* w, bool flip, u8* rdram, size_t rdram_size) {
    const u32 xl = (w[0] >> 12) & 0xfff;
    u32 yl = w[0] & 0xfff;
    const u32 xh = (w[1] >> 12) & 0xfff, yh = w[1] & 0xfff;
    const u32 tile = (w[1] >> 24) & 7;
    const s32 s = static_cast<s32>((w[2] >> 16) & 0xffff), t = static_cast<s32>(w[2] & 0xffff);
    const s32 dsdx = sext(static_cast<s32>((w[3] >> 16) & 0xffff), 16);
    const s32 dtdy = sext(static_cast<s32>(w[3] & 0xffff), 16);
    const u32 cycle = (other_h_ >> 20) & 3;
    if (cycle >= 2) yl |= 3;
    Setup st;
    st.xh = static_cast<s32>(xh << 13);
    st.xl = static_cast<s32>(xl << 13);
    st.xm = static_cast<s32>(xl << 13);
    st.ym = static_cast<s32>(yl);
    st.yl = static_cast<s32>(yl);
    st.yh = static_cast<s32>(yh);
    st.flags = SETUP_FLIP | (cycle == 2 ? SETUP_SKIP_XFRAC : 0);
    st.tile = tile;
    Attr a;
    a.stzw[0] = w32(static_cast<s64>(s) << 16);
    a.stzw[1] = w32(static_cast<s64>(t) << 16);
    if (!flip) {
        a.dstzw_dx[0] = w32(static_cast<s64>(dsdx) << 11);
        a.dstzw_de[1] = w32(static_cast<s64>(dtdy) << 11);
        a.dstzw_dy[1] = w32(static_cast<s64>(dtdy) << 11);
    } else {
        a.dstzw_dx[1] = w32(static_cast<s64>(dtdy) << 11);
        a.dstzw_de[0] = w32(static_cast<s64>(dsdx) << 11);
        a.dstzw_dy[0] = w32(static_cast<s64>(dsdx) << 11);
    }
    draw(st, a, rdram, rdram_size);
}

// ============================================================================
// Rasterization
// ============================================================================

namespace {

// The scanline's span: the attributes where the major edge enters it, and
// where each of its four sub-scanlines starts and ends (x in 1/8 pixels).
// At a scale S > 1 (internal resolution, after parallel-rdp) y and the
// result are in S times finer lines and pixels; attributes are stepped by
// whole derivatives at native pixels and by 1/S of them in between.
void span_setup(s32 S, int y, s32 xh0, s32 xm0, s32 xl0, s32 yh, s32 ym, s32 yl, s32 dxhdy,
                s32 dxmdy, s32 dxldy, u32 flags, const s32 rgba0[4], const s32 drgba_dx[4], const s32 drgba_de[4],
                const s32 drgba_dy[4], const s32 stzw0[4], const s32 dstzw_dx[4], const s32 dstzw_de[4],
                const s32 dstzw_dy[4], s32 sc_xlo, s32 sc_ylo, s32 sc_xhi, s32 sc_yhi, Span& sp) {
    const bool flip = flags & SETUP_FLIP;
    const bool do_offset = flags & SETUP_DO_OFFSET;
    const bool skip_xfrac = flags & SETUP_SKIP_XFRAC;
    // dy steps of `d`: whole ones per native line, 1/S of one in between.
    auto snapped = [S](s32 d, s32 dy) -> s64 {
        if (S == 1) return static_cast<s64>(dy) * d;
        return static_cast<s64>(fdiv(dy, S)) * d + static_cast<s64>(fmod_(dy, S)) * fdiv(d, S);
    };
    {
        const s32 ybase = (yh >> 2) * S;
        const s32 dy = y - ybase;
        s32 xh = w32(static_cast<s64>(xh0) * S + static_cast<s64>(dy) * w32(static_cast<s64>(dxhdy) << 2));
        s32 drgba_diff[4]{}, dstzw_diff[4]{};
        if (do_offset) {
            xh = w32(static_cast<s64>(xh) + 3ll * S * dxhdy);
            for (int i = 0; i < 4; ++i) {
                const s32 deh = drgba_de[i] & ~0x1ff, dyh = drgba_dy[i] & ~0x1ff;
                drgba_diff[i] = w32(static_cast<s64>(deh) - (deh >> 2) - dyh + (dyh >> 2));
                const s32 tdeh = dstzw_de[i] & ~0x1ff, tdyh = dstzw_dy[i] & ~0x1ff;
                dstzw_diff[i] = w32(static_cast<s64>(tdeh) - (tdeh >> 2) - tdyh + (tdyh >> 2));
            }
        }
        sp.base_x = xh >> 15;
        const s32 xfrac = skip_xfrac ? 0 : ((xh >> 7) & 0xff);
        for (int i = 0; i < 4; ++i) {
            s32 c = w32(static_cast<s64>(rgba0[i]) + snapped(drgba_de[i], dy));
            c = w32(static_cast<s64>(c & ~0x1ff) + drgba_diff[i] - snapped((drgba_dx[i] >> 8) & ~1, xfrac)) & ~0x3ff;
            sp.rgba[i] = c;
            s32 t = w32(static_cast<s64>(stzw0[i]) + snapped(dstzw_de[i], dy));
            t = w32(static_cast<s64>(t & ~0x1ff) + dstzw_diff[i] - snapped((dstzw_dx[i] >> 8) & ~1, xfrac)) & ~0x3ff;
            sp.stzw[i] = t;
        }
    }

    const s32 yh_base = (yh & ~3) * S, ym_base = ym * S;
    const s32 ylo = std::max(yh, sc_ylo) * S, yhi = std::min(yl, sc_yhi) * S;
    const s32 lo_sc = S * (sc_xlo << 1), hi_sc = S * (sc_xhi << 1);
    const int xbits = 27 + (S > 1 ? 32 - __builtin_clz(static_cast<u32>(S - 1)) : 0);
    bool invalid[4];
    s32 xleft[4], xright[4];
    bool all_over = true, all_under = true;
    for (int i = 0; i < 4; ++i) {
        const s32 ys = y * 4 + i;
        const bool clip_y = ys < ylo || ys >= yhi;
        s32 xh = w32(static_cast<s64>(xh0) * S + static_cast<s64>(ys - yh_base) * dxhdy);
        const s32 xm = w32(static_cast<s64>(xm0) * S + static_cast<s64>(ys - yh_base) * dxmdy);
        s32 xl = w32(static_cast<s64>(xl0) * S + static_cast<s64>(ys - ym_base) * dxldy);
        if (ys < ym * S) xl = xm;
        xl = sext(xl, xbits);
        xh = sext(xh, xbits);
        auto quantize = [](s32 x) { return (x >> 12) | ((x & 0xfff) ? 1 : 0); };
        const s32 xhs = quantize(xh), xls = quantize(xl);
        s32 l = flip ? xhs : xls, r = flip ? xls : xhs;
        invalid[i] = (l >> 1) > (r >> 1);
        if (!(std::min(l, r) >= hi_sc)) all_over = false;
        if (!(std::max(l, r) < lo_sc)) all_under = false;
        l = std::min(std::max(l, lo_sc), hi_sc);
        r = std::min(std::max(r, lo_sc), hi_sc);
        invalid[i] = invalid[i] || clip_y;
        xleft[i] = invalid[i] ? 0xffff : l;
        xright[i] = invalid[i] ? 0 : r;
    }
    s32 mn = xleft[0], mx = xright[0];
    for (int i = 1; i < 4; ++i) {
        mn = std::min(mn, xleft[i]);
        mx = std::max(mx, xright[i]);
    }
    for (int i = 0; i < 4; ++i) {
        sp.xleft[i] = xleft[i];
        sp.xright[i] = xright[i];
    }
    sp.start_x = mn >> 3;
    sp.end_x = mx >> 3;
    sp.valid = !(invalid[0] && invalid[1] && invalid[2] && invalid[3]) && !all_over && !all_under;
    if (flags & SETUP_INTERLACE_FIELD)
        if ((fdiv(y, S) & 1) != ((flags & SETUP_INTERLACE_KEEP_ODD) ? 1 : 0)) sp.valid = false;
    sp.lodlength = flip ? sp.end_x - sp.base_x : sp.base_x - sp.start_x;
}

} // namespace

void ExactRdp::draw(Setup& setup, Attr attr, u8* rdram, size_t rdram_size) {
    // ---- Setup fix-ups
    if (setup.ym < (setup.yh & ~3)) setup.ym = 0x7fff; // YM above the first sub-scanline: never reached
    if (scissor_field_) {
        setup.flags |= SETUP_INTERLACE_FIELD;
        if (scissor_odd_) setup.flags |= SETUP_INTERLACE_KEEP_ODD;
    }

    // ---- Modes
    Prim p;
    const u32 H = other_h_, L = other_l_;
    p.cycle = static_cast<int>((H >> 20) & 3);
    p.persp = (H >> 19) & 1;
    p.detail = (H >> 18) & 1;
    p.sharpen = (H >> 17) & 1;
    p.tex_lod = (H >> 16) & 1;
    p.tlut = (H >> 15) & 1;
    p.tlut_type = (H >> 14) & 1;
    p.sample_quad = (H >> 13) & 1;
    p.mid_texel = (H >> 12) & 1;
    p.bilerp0 = (H >> 11) & 1;
    p.bilerp1 = (H >> 10) & 1;
    p.convert_one = (H >> 9) & 1;
    p.dither = static_cast<int>((H >> 4) & 0xf);
    p.key_en = (H >> 8) & 1;
    p.force_blend = (L >> 14) & 1;
    p.alpha_cvg_select = (L >> 13) & 1;
    p.cvg_times_alpha = (L >> 12) & 1;
    p.z_mode = static_cast<int>((L >> 10) & 3);
    p.coverage_mode = static_cast<int>((L >> 8) & 3);
    p.color_on_cvg = (L >> 7) & 1;
    p.image_read = (L >> 6) & 1;
    p.z_update = (L >> 5) & 1;
    p.z_compare = (L >> 4) & 1;
    p.aa = (L >> 3) & 1;
    const bool prim_z = (L >> 2) & 1;
    p.alpha_test_dither = (L >> 1) & 1;
    p.alpha_test = L & 1;
    p.dither_en = ((p.dither >> 2) & 3) != 3;
    p.interlace = scissor_field_;
    for (int c = 0; c < 2; ++c) {
        p.blend[c][0] = static_cast<u8>((L >> (30 - 2 * c)) & 3);
        p.blend[c][1] = static_cast<u8>((L >> (26 - 2 * c)) & 3);
        p.blend[c][2] = static_cast<u8>((L >> (22 - 2 * c)) & 3);
        p.blend[c][3] = static_cast<u8>((L >> (18 - 2 * c)) & 3);
    }
    const u32 C0 = combine_w0_, C1 = combine_w1_;
    p.rgb[0][0] = (C0 >> 20) & 0xf;
    p.rgb[0][2] = (C0 >> 15) & 0x1f;
    p.rgb[0][1] = (C1 >> 28) & 0xf;
    p.rgb[0][3] = (C1 >> 15) & 0x7;
    p.alpha[0][0] = (C0 >> 12) & 0x7;
    p.alpha[0][1] = (C1 >> 12) & 0x7;
    p.alpha[0][2] = (C0 >> 9) & 0x7;
    p.alpha[0][3] = (C1 >> 9) & 0x7;
    p.rgb[1][0] = (C0 >> 5) & 0xf;
    p.rgb[1][2] = C0 & 0x1f;
    p.rgb[1][1] = (C1 >> 24) & 0xf;
    p.rgb[1][3] = (C1 >> 6) & 0x7;
    p.alpha[1][0] = (C1 >> 21) & 0x7;
    p.alpha[1][1] = (C1 >> 3) & 0x7;
    p.alpha[1][2] = (C1 >> 18) & 0x7;
    p.alpha[1][3] = C1 & 0x7;

    const bool multi = p.cycle == 1;
    if (multi) {
        p.uses_texel0 = accesses_texel0(p.rgb[0], p.alpha[0]) || accesses_texel1(p.rgb[1], p.alpha[1]);
        p.uses_texel1 = accesses_texel1(p.rgb[0], p.alpha[0]) || accesses_texel0(p.rgb[1], p.alpha[1]);
        p.uses_pipelined_texel1 = false;
        p.uses_lod = accesses_lod_frac(p.rgb[0], p.alpha[0]) || accesses_lod_frac(p.rgb[1], p.alpha[1]) || p.tex_lod;
    } else {
        p.uses_texel0 = accesses_texel0(p.rgb[1], p.alpha[1]);
        p.uses_texel1 = false;
        p.uses_pipelined_texel1 = accesses_texel1(p.rgb[1], p.alpha[1]);
        p.uses_lod = p.tex_lod;
    }
    if (p.uses_texel1 && p.convert_one) p.uses_texel0 = true;
    if (!multi && !p.uses_pipelined_texel1) {
        p.bilerp1 = false;
        p.convert_one = false;
    }
    // Noise
    if ((p.dither & 3) == 2 || ((p.dither >> 2) & 3) == 2) p.need_noise = true;
    if (p.cycle < 2) {
        if (multi && p.rgb[0][0] == 7) p.need_noise = true;
        if (p.rgb[1][0] == 7) p.need_noise = true;
        if (multi && p.rgb[0][0] == 7 && p.rgb[1][0] == 7) p.need_noise_dual = true;
        if (p.alpha_test && p.alpha_test_dither) p.need_noise = true;
    }

    // Derived
    if (prim_z) {
        p.dz = prim_dz_;
        p.dz_compressed = dz_compress_prim(p.dz);
    } else {
        const s32 dzdx = attr.dstzw_dx[2] >> 16, dzdy = attr.dstzw_dy[2] >> 16;
        s32 dzpix = (dzdx < 0 ? (~dzdx & 0x7fff) : dzdx) + (dzdy < 0 ? (~dzdy & 0x7fff) : dzdy);
        dzpix = normalize_dzpix(dzpix);
        p.dz = dzpix;
        p.dz_compressed = dz_compress_prim(dzpix);
    }
    if (prim_z) {
        attr.stzw[2] = prim_depth_;
        attr.dstzw_dx[2] = attr.dstzw_de[2] = attr.dstzw_dy[2] = 0;
    }
    p.min_lod = prim_min_level_;
    for (int i = 0; i < 4; ++i) p.factors[i] = static_cast<s16>(convert_[i]);
    p.prim_color = prim_color_;
    p.env_color = env_color_;
    p.fog_color = fog_color_;
    p.blend_color = blend_color_;
    p.fill_color = fill_color_;
    p.prim_lod_frac = prim_lod_frac_;
    for (int i = 0; i < 3; ++i) {
        p.key_center[i] = key_center_[i];
        p.key_scale[i] = key_scale_[i];
        p.key_width[i] = key_width_[i];
    }
    p.k4 = convert_[4] & 0x1ff;
    p.k5 = convert_[5] & 0x1ff;

    // Framebuffer
    p.fb_fmt = ci_fmt_;
    p.fb_width = ci_width_;
    switch (ci_fmt_) {
        case FB_I4: case FB_I8: p.fb_index = ci_addr_; break;
        case FB_RGBA8888: p.fb_index = ci_addr_ >> 2; break;
        default: p.fb_index = ci_addr_ >> 1; break;
    }
    p.z_index = zi_addr_ >> 1;
    p.alias = ci_addr_ == zi_addr_;
    p.prim_serial = ++prim_count_;

    // ---- Rows
    const s32 sc_ylo = static_cast<s32>(scissor_ylo_), sc_yhi = static_cast<s32>(scissor_yhi_);
    const s32 min_line = std::max(setup.yh, sc_ylo) >> 2;
    const s32 max_line = std::min(setup.yl - 1, sc_yhi - 1) >> 2;
    if (max_line < min_line) return;

    // Internal resolution: after the native pass (psc = 1, RDRAM), a second
    // one at psc = scale_ into the high-resolution copy - or, for primitives
    // that are repeated rather than upscaled, their native writes copied to
    // every sample (`mirror`).
    UpStore* const up = scale_ > 1 ? upscaled(rdram_size) : nullptr;
    const bool mirror = up && (mirror_ || p.cycle >= 2);
    const int passes = up && !mirror ? 2 : 1;
    s32 psc = 1;
    bool hires = false;     // in the high-resolution pass
    u8* fbm = rdram;        // the memory drawn into: RDRAM, or the current pixel's sample
    u8* hbits = nullptr;    // and that sample's ninth bits

    auto make_span = [&](int y, Span& sp) {
        span_setup(psc, y, setup.xh, setup.xm, setup.xl, setup.yh, setup.ym, setup.yl, setup.dxhdy, setup.dxmdy,
                   setup.dxldy, setup.flags, attr.rgba, attr.drgba_dx, attr.drgba_de, attr.drgba_dy, attr.stzw,
                   attr.dstzw_dx, attr.dstzw_de, attr.dstzw_dy, static_cast<s32>(scissor_xlo_), sc_ylo,
                   static_cast<s32>(scissor_xhi_), sc_yhi, sp);
    };

    const Tex tx{tmem_};
    const size_t mask8 = rdram_size - 1, mask16 = mask8 >> 1, mask32 = mask8 >> 2;
    const bool flip = setup.flags & SETUP_FLIP;
    const u32 setup_tile = setup.tile;
    const u32 max_level = setup_tile >> 3;
    const BlenderDivider& bdiv = blender_divider();

    auto hidden_get = [&](size_t hidx, u16 word) -> u8 { return hires ? hbits[hidx] : hidden_at(hidx, word); };
    // Every store ends with this, once per halfword it wrote.
    auto hidden_set = [&](size_t hidx, u8 bits) {
        if (hires) {
            hbits[hidx] = bits;
            return;
        }
        hs_->bits[hidx] = bits;
        hs_->word[hidx] = static_cast<u16>((rdram[hidx * 2] << 8) | rdram[hidx * 2 + 1]);
        u8& w = hs_->watched[hidx >> 5];
        if (!w) {
            w = 1;
            jit::watch_rdram(static_cast<u32>(hidx >> 5) << 6, 64);
        }
        if (mirror) {
            const u32 slices = up->scale * up->scale;
            for (u32 sl = 0; sl < slices; ++sl) {
                u8* c = up->color(sl) + hidx * 2;
                c[0] = rdram[hidx * 2];
                c[1] = rdram[hidx * 2 + 1];
                up->hidden(sl)[hidx] = bits;
            }
        }
    };
    auto rd16 = [&](size_t hidx) -> u16 { return static_cast<u16>((fbm[hidx * 2] << 8) | fbm[hidx * 2 + 1]); };
    auto wr16 = [&](size_t hidx, u16 v) {
        fbm[hidx * 2] = static_cast<u8>(v >> 8);
        fbm[hidx * 2 + 1] = static_cast<u8>(v);
    };

    // A pixel's colour as read from the colour image (col) and as the
    // blender sees it (mem; alpha = memory coverage << 5).
    auto read_colour = [&](s32 x, s32 y, s32 col[4]) -> size_t {
        size_t cidx = p.fb_index + static_cast<size_t>(p.fb_width) * static_cast<u32>(y) + static_cast<u32>(x);
        switch (p.fb_fmt) {
            case FB_I4: case FB_I8: {
                cidx &= mask8;
                col[0] = col[1] = col[2] = fbm[cidx];
                col[3] = hidden_get(cidx >> 1, rd16(cidx >> 1));
                break;
            }
            case FB_RGBA5551: {
                cidx &= mask16;
                const u16 wv = rd16(cidx);
                col[0] = (wv >> 8) & 0xf8;
                col[1] = (wv >> 3) & 0xf8;
                col[2] = (wv << 2) & 0xf8;
                col[3] = (hidden_get(cidx, wv) << 5) | ((wv & 1) << 7);
                break;
            }
            case FB_IA88: {
                cidx &= mask16;
                const u16 wv = rd16(cidx);
                col[0] = col[1] = col[2] = wv >> 8;
                col[3] = wv & 0xff;
                break;
            }
            default: {
                cidx &= mask32;
                const u8* q = fbm + cidx * 4;
                col[0] = q[0];
                col[1] = q[1];
                col[2] = q[2];
                col[3] = q[3];
                break;
            }
        }
        return cidx;
    };
    auto decode_memory = [&](const s32 col[4], s32 mem[4]) {
        s32 cvg_bits = p.image_read ? (col[3] & 0xe0) : 0xe0;
        switch (p.fb_fmt) {
            case FB_I4: mem[0] = mem[1] = mem[2] = 0; cvg_bits = 0xe0; break;
            case FB_I8: mem[0] = mem[1] = mem[2] = col[0]; cvg_bits = 0xe0; break;
            case FB_RGBA5551: mem[0] = col[0] & 0xf8; mem[1] = col[1] & 0xf8; mem[2] = col[2] & 0xf8; break;
            case FB_IA88: mem[0] = mem[1] = mem[2] = col[0]; break;
            default: mem[0] = col[0]; mem[1] = col[1]; mem[2] = col[2]; break;
        }
        mem[3] = cvg_bits;
    };
    auto memory_colour = [&](s32 x, s32 y, s32 mem[4]) {
        s32 col[4];
        read_colour(x, y, col);
        decode_memory(col, mem);
    };

    // The combiner's inputs as a table: each cycle's A, B, C and D pick a
    // row per channel (csel), resolved once per primitive; the rows that
    // change per pixel are filled in by run_cycle().
    enum : u8 {
        S_COMB, S_T0, S_T1, S_PRIM, S_SHADE, S_ENV, S_ONE, S_NOISE, S_ZERO, S_KEYC, S_K4, S_KEYS,
        S_COMBA, S_T0A, S_T1A, S_PRIMA, S_SHADEA, S_ENVA, S_LOD, S_PLOD, S_K5, S_COUNT
    };
    s32 csrc[S_COUNT][4]{};
    for (int i = 0; i < 4; ++i) {
        csrc[S_PRIM][i] = static_cast<s32>((p.prim_color >> (24 - 8 * i)) & 0xff);
        csrc[S_ENV][i] = static_cast<s32>((p.env_color >> (24 - 8 * i)) & 0xff);
        csrc[S_ONE][i] = 0x100;
        csrc[S_ZERO][i] = 0;
        csrc[S_KEYC][i] = i < 3 ? p.key_center[i] : 0;
        csrc[S_K4][i] = p.k4;
        csrc[S_KEYS][i] = i < 3 ? p.key_scale[i] : 0;
        csrc[S_PRIMA][i] = static_cast<s32>(p.prim_color & 0xff);
        csrc[S_ENVA][i] = static_cast<s32>(p.env_color & 0xff);
        csrc[S_PLOD][i] = p.prim_lod_frac;
        csrc[S_K5][i] = p.k5;
    }
    u8 csel[2][4][4];
    for (int c = 0; c < 2; ++c) {
        static const u8 kRgbA[16] = {S_COMB, S_T0, S_T1, S_PRIM, S_SHADE, S_ENV, S_ONE, S_NOISE,
                                     S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO};
        static const u8 kRgbB[16] = {S_COMB, S_T0, S_T1, S_PRIM, S_SHADE, S_ENV, S_KEYC, S_K4,
                                     S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO};
        static const u8 kRgbC[32] = {S_COMB, S_T0, S_T1, S_PRIM, S_SHADE, S_ENV, S_KEYS, S_COMBA,
                                     S_T0A, S_T1A, S_PRIMA, S_SHADEA, S_ENVA, S_LOD, S_PLOD, S_K5,
                                     S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO,
                                     S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO, S_ZERO};
        static const u8 kRgbD[8] = {S_COMB, S_T0, S_T1, S_PRIM, S_SHADE, S_ENV, S_ONE, S_ZERO};
        static const u8 kAlphaABD[8] = {S_COMB, S_T0, S_T1, S_PRIM, S_SHADE, S_ENV, S_ONE, S_ZERO};
        static const u8 kAlphaC[8] = {S_LOD, S_T0, S_T1, S_PRIM, S_SHADE, S_ENV, S_PLOD, S_ZERO};
        for (int i = 0; i < 3; ++i) {
            csel[c][0][i] = kRgbA[p.rgb[c][0] & 15];
            csel[c][1][i] = kRgbB[p.rgb[c][1] & 15];
            csel[c][2][i] = kRgbC[p.rgb[c][2] & 31];
            csel[c][3][i] = kRgbD[p.rgb[c][3] & 7];
        }
        csel[c][0][3] = kAlphaABD[p.alpha[c][0] & 7];
        csel[c][1][3] = kAlphaABD[p.alpha[c][1] & 7];
        csel[c][2][3] = kAlphaC[p.alpha[c][2] & 7];
        csel[c][3][3] = kAlphaABD[p.alpha[c][3] & 7];
    }

    // Lanes: which rows this copy draws, and - when primitives that walk
    // pixels carry the delayed memory colour from row to row - whether it
    // draws the primitive's last walked row, whose state it hands on.
    if (lanes_ > 1 && serial_ && lane_ != 0) return;
    const bool all_rows = lanes_ == 1 || serial_;
    if (lanes_ > 1 && serial_) {
        std::lock_guard<std::mutex> lk(hs_->tail_mutex);
        std::copy(std::begin(hs_->tail_mem), std::end(hs_->tail_mem), prev_mem_);
    }
    s32 last_row = -1;
    if (lanes_ > 1 && p.cycle < 2) {
        Span probe;
        for (s32 y = max_line; y >= min_line; --y) {
            make_span(y, probe);
            if (probe.valid) {
                last_row = y;
                break;
            }
        }
    }

    for (int pass = 0; pass < passes; ++pass) {
        hires = pass == 1;
        psc = hires ? static_cast<s32>(scale_) : 1;
        s32* const pm = hires ? prev_mem_up_ : prev_mem_; // this pass's delayed memory colour
        fbm = rdram;
        Span cur, next;
        for (s32 y = min_line * psc; y <= max_line * psc + psc - 1; ++y) {
            if (!all_rows && static_cast<u32>(fdiv(y, psc)) % lanes_ != lane_) continue;
            make_span(y, cur);
            make_span(y + 1, next);
            if (cur.valid) {
                const s32 x0 = cur.start_x, x1 = cur.end_x;
                // The inputs of the pixel at x: shade, texels, LOD fraction, depth.
                struct PixIn {
                    s32 shade[4];
                    T4 texel0, texel1;
                    s32 lod_frac;
                    s32 z;
                };
                const s32 idir = flip ? 1 : -1;
                auto pixel_inputs = [&](s32 x, u32 coverage, PixIn& in) {
                        const s32 dx = x - cur.base_x;
                        // The first covered sample; a pixel with none (past the end
                        // of the span) is taken at its corner.
                        const int first = coverage ? find_lsb(coverage) : 0;
                        const s32 yoff = first >> 1, xoff = coverage ? ((first & 1) << 1) + (yoff & 1) : 0;
                        s32* shade = in.shade;

                        // Shade
                        for (int i = 0; i < 4; ++i) {
                            const s32 c = w32(static_cast<s64>(cur.rgba[i]) + static_cast<s64>(fdiv(attr.drgba_dx[i] & ~0x1f, psc)) * dx);
                            s32 sn = c >> 14;
                            sn *= 4 * psc;
                            sn += xoff * (attr.drgba_dx[i] >> 14) + yoff * (attr.drgba_dy[i] >> 14);
                            sn = fdiv(sn, 16 * psc);
                            shade[i] = clamp_9bit(sn);
                        }

                        // S, T, Z
                        s32 stw[3], stw_dx[3], stw_dy[3];
                        static const int kC[3] = {0, 1, 3};
                        for (int k = 0; k < 3; ++k) {
                            const int c = kC[k];
                            stw[k] = w32(static_cast<s64>(cur.stzw[c]) + static_cast<s64>(fdiv(attr.dstzw_dx[c] & ~0x1f, psc)) * dx);
                            if (p.uses_lod) {
                                stw_dx[k] = w32(static_cast<s64>(stw[k]) + static_cast<s64>(idir) * (attr.dstzw_dx[c] & ~0x1f));
                                stw_dy[k] = w32(static_cast<s64>(stw[k]) + (attr.dstzw_dy[c] & ~0x7fff));
                            }
                        }
                        s32 st_s, st_t, sdx = 0, tdx = 0, sdy = 0, tdy = 0;
                        bool povf = false;
                        if (p.persp) {
                            perspective_divide(stw[0] >> 16, stw[1] >> 16, stw[2] >> 16, st_s, st_t, povf);
                            if (p.uses_lod) {
                                perspective_divide(stw_dx[0] >> 16, stw_dx[1] >> 16, stw_dx[2] >> 16, sdx, tdx, povf);
                                perspective_divide(stw_dy[0] >> 16, stw_dy[1] >> 16, stw_dy[2] >> 16, sdy, tdy, povf);
                            }
                        } else {
                            st_s = stw[0] >> 16;
                            st_t = stw[1] >> 16;
                            if (p.uses_lod) {
                                sdx = stw_dx[0] >> 16;
                                tdx = stw_dx[1] >> 16;
                                sdy = stw_dy[0] >> 16;
                                tdy = stw_dy[1] >> 16;
                            }
                        }
                        {
                            const s32 zz = w32(static_cast<s64>(cur.stzw[2]) + static_cast<s64>(attr.dstzw_dx[2]) * fdiv(dx, psc) +
                                               static_cast<s64>(fdiv(attr.dstzw_dx[2], psc)) * fmod_(dx, psc));
                            s32 sz = zz >> 10;
                            sz = w32(static_cast<s64>(sz) * 4 * psc);
                            sz = w32(static_cast<s64>(sz) + static_cast<s64>(xoff) * (attr.dstzw_dx[2] >> 10) +
                                     static_cast<s64>(yoff) * (attr.dstzw_dy[2] >> 10));
                            sz = w32(fdiv64(sz, 32 * psc));
                            in.z = clamp_z(sz);
                        }

                        // Textures
                        u32 tile0 = setup_tile & 7, tile1 = (tile0 + 1) & 7;
                        s32& lod_frac = in.lod_frac;
                        lod_frac = 0;
                        if (p.uses_lod && !multi) {
                            // In 1-cycle mode the LOD unit measures the pipelined pair
                            // along the span - the next pixel and the one after - or,
                            // at the second-to-last pixel of a long span whose four
                            // sublines are all valid, the centred pair (P - 1, P + 1).
                            // No sample from the scanline below. (Hardware verified,
                            // diagnostic cartridge cases 11:23 and 11:46.)
                            const s32 span_last_x = flip ? cur.end_x : cur.start_x;
                            const bool all_valid = cur.xleft[0] != 0xffff && cur.xleft[1] != 0xffff &&
                                                   cur.xleft[2] != 0xffff && cur.xleft[3] != 0xffff;
                            const bool centred = x + idir * psc == span_last_x && cur.lodlength >= 8 * psc && all_valid;
                            s32 ln[3], lf[3];
                            for (int k = 0; k < 3; ++k) {
                                const s64 d = attr.dstzw_dx[kC[k]] & ~0x1f;
                                ln[k] = w32(static_cast<s64>(stw[k]) + d * idir);
                                lf[k] = w32(static_cast<s64>(stw[k]) + d * (centred ? -idir : 2 * idir));
                            }
                            s32 ns, nt, fs, ft;
                            bool lov = false;
                            if (p.persp) {
                                perspective_divide(ln[0] >> 16, ln[1] >> 16, ln[2] >> 16, ns, nt, lov);
                                perspective_divide(lf[0] >> 16, lf[1] >> 16, lf[2] >> 16, fs, ft, lov);
                            } else {
                                ns = ln[0] >> 16;
                                nt = ln[1] >> 16;
                                fs = lf[0] >> 16;
                                ft = lf[1] >> 16;
                            }
                            compute_lod(tile0, tile1, lod_frac, max_level, p.min_lod, ns, nt, fs, ft, ns, nt, lov,
                                        p.tex_lod, p.sharpen, p.detail);
                        } else if (p.uses_lod) {
                            compute_lod(tile0, tile1, lod_frac, max_level, p.min_lod, st_s, st_t, sdx, tdx, sdy, tdy, povf,
                                        p.tex_lod, p.sharpen, p.detail);
                        }
                        T4& texel0 = in.texel0;
                        T4& texel1 = in.texel1;
                        texel0 = T4{};
                        texel1 = T4{};
                        if (p.uses_texel0)
                            texel0 = sample_texture(tx, tiles_[tile0], st_s, st_t, p.tlut, p.tlut_type, p.sample_quad,
                                                    p.mid_texel, false, p.bilerp0, p.factors, T4{});
                        bool uses_texel1 = p.uses_texel1;
                        s32 s1 = st_s, t1 = st_t;
                        if (p.uses_pipelined_texel1) {
                            const bool long_span = cur.lodlength >= 8 * psc;
                            const bool end_span = x == (flip ? cur.end_x : cur.start_x);
                            if (end_span && long_span && next.valid) {
                                const s32 ns = next.stzw[0] >> 16, nt = next.stzw[1] >> 16, nw = next.stzw[3] >> 16;
                                if (p.persp) {
                                    bool ov = false;
                                    perspective_divide(ns, nt, nw, s1, t1, ov);
                                } else {
                                    s1 = ns;
                                    t1 = nt;
                                }
                            } else {
                                const s32 d2 = dx + idir * psc; // the next (native) pixel
                                const s32 a0 = w32(static_cast<s64>(cur.stzw[0]) + static_cast<s64>(fdiv(attr.dstzw_dx[0] & ~0x1f, psc)) * d2) >> 16;
                                const s32 a1 = w32(static_cast<s64>(cur.stzw[1]) + static_cast<s64>(fdiv(attr.dstzw_dx[1] & ~0x1f, psc)) * d2) >> 16;
                                const s32 a3 = w32(static_cast<s64>(cur.stzw[3]) + static_cast<s64>(fdiv(attr.dstzw_dx[3] & ~0x1f, psc)) * d2) >> 16;
                                if (p.persp) {
                                    bool ov = false;
                                    perspective_divide(a0, a1, a3, s1, t1, ov);
                                } else {
                                    s1 = a0;
                                    t1 = a1;
                                }
                            }
                            tile1 = tile0;
                            uses_texel1 = true;
                        }
                        if (uses_texel1) {
                            if (p.convert_one && !p.bilerp1) texel1 = texture_convert_factors(texel0, p.factors);
                            else
                                texel1 = sample_texture(tx, tiles_[tile1], s1, t1, p.tlut, p.tlut_type, p.sample_quad,
                                                        p.mid_texel, p.convert_one, p.bilerp1, p.factors, texel0);
                        }

                };
                // Pixels are walked from the major edge towards the minor one.
                for (s32 x = flip ? x0 : x1; flip ? x <= x1 : x >= x0; x += idir) {
                    // The frame buffer pixel (and, at internal resolution, its sample).
                    s32 nx = x, ny = y;
                    if (hires) {
                        nx = fdiv(x, psc);
                        ny = fdiv(y, psc);
                        const u32 slice = static_cast<u32>((y - ny * psc) * psc + (x - nx * psc));
                        fbm = up->color(slice);
                        hbits = up->hidden(slice);
                    }
                    // 2-cycle mode: the first blender cycle sees the memory colour
                    // of the pixel walked before this one (pipelining; the state
                    // carries over between spans and primitives), the second
                    // cycle this pixel's.
                    s32 delayed_mem[4] = {pm[0], pm[1], pm[2], pm[3]};
                    if (p.cycle < 2) memory_colour(nx, ny, pm); // 1-cycle pixels read it too
                    // ---- Shade the pixel
                    Noise noise;
                    if (p.need_noise) noise.reseed(static_cast<u32>(x), static_cast<u32>(y), p.prim_serial);

                    enum { K_NONE, K_FILL, K_COPY, K_NORMAL } kind = K_NONE;
                    u32 copy_word = 0;
                    s32 comb[4]{}, z = 0, dith = 0, cov_count = 0, shade_a = 0;

                    if (p.cycle == 2) { // copy
                        if (x < cur.start_x || x > cur.end_x) continue;
                        s32 dx = flip ? (x - cur.start_x) : (cur.end_x - x);
                        int dx_shift = 0;
                        s32 dx_mask = 0;
                        int fb_size = 0;
                        switch (p.fb_fmt) {
                            case FB_I4: fb_size = 0; dx_mask = 0; dx_shift = 0; break;
                            case FB_I8: fb_size = 1; dx_mask = ~7; dx_shift = 3; break;
                            case FB_RGBA5551: case FB_IA88: fb_size = 2; dx_mask = ~3; dx_shift = 2; break;
                            default: fb_size = 4; dx_mask = 0; dx_shift = 1; break;
                        }
                        const s32 snapped = dx & dx_mask;
                        const s32 s_offset = dx - snapped;
                        const s32 lerp_dx = (dx >> dx_shift) * (flip ? 1 : -1);
                        s32 ss = w32(static_cast<s64>(cur.stzw[0]) + static_cast<s64>(attr.dstzw_dx[0] & ~0x1f) * lerp_dx) >> 16;
                        s32 tt = w32(static_cast<s64>(cur.stzw[1]) + static_cast<s64>(attr.dstzw_dx[1] & ~0x1f) * lerp_dx) >> 16;
                        const s32 ww = w32(static_cast<s64>(cur.stzw[3]) + static_cast<s64>(attr.dstzw_dx[3] & ~0x1f) * lerp_dx) >> 16;
                        if (p.persp) {
                            bool ov = false;
                            perspective_divide(ss, tt, ww, ss, tt, ov, true);
                        }
                        const Tile& tile = tiles_[setup_tile & 7];
                        s32 cs = shift_coord(ss, static_cast<s32>(tile.slo), tile.shift_s) >> 5;
                        s32 ct = shift_coord(tt, static_cast<s32>(tile.tlo), tile.shift_t) >> 5;
                        auto copy_word_at = [&](s32 so) -> s32 {
                            const bool high_word = so < 2;
                            const bool replicate = high_word && tile.size != 2 && !p.tlut;
                            const int s_shamt = std::min<int>(tile.size, 2);
                            const u32 idx_mask = (tile.size == 3 || p.tlut) ? 0x3ff : 0x7ff;
                            s32 samp;
                            if (replicate) {
                                const s32 sx = cs + 2 * so;
                                const s32 sA = mask_s(tile, sx), sB = mask_s(tile, sx + 1);
                                const s32 t = mask_t(tile, ct);
                                const u32 tbase = tile.offset + tile.stride * static_cast<u32>(t);
                                u32 noA = (tbase * 2 + (static_cast<u32>(sA) << s_shamt)) & 0x1fff;
                                u32 noB = (tbase * 2 + (static_cast<u32>(sB) << s_shamt)) & 0x1fff;
                                noA ^= (static_cast<u32>(t) & 1) * 8;
                                noB ^= (static_cast<u32>(t) & 1) * 8;
                                s32 a0 = tx.half((noA >> 2) & idx_mask), a1 = tx.half((noB >> 2) & idx_mask);
                                if (tile.size == 1) {
                                    a0 = (a0 >> (8 - 4 * static_cast<int>(noA & 2))) & 0xff;
                                    a1 = (a1 >> (8 - 4 * static_cast<int>(noB & 2))) & 0xff;
                                } else if (tile.size == 0) {
                                    a0 = ((a0 >> (12 - 4 * static_cast<int>(noA & 3))) & 0xf) * 0x11;
                                    a1 = ((a1 >> (12 - 4 * static_cast<int>(noB & 3))) & 0xf) * 0x11;
                                } else {
                                    a0 >>= 8;
                                    a1 >>= 8;
                                }
                                samp = (a0 << 8) | a1;
                            } else {
                                const s32 sx = mask_s(tile, cs + so);
                                const s32 t = mask_t(tile, ct);
                                const u32 tbase = tile.offset + tile.stride * static_cast<u32>(t);
                                u32 no = (tbase * 2 + (static_cast<u32>(sx) << s_shamt)) & 0x1fff;
                                no ^= (static_cast<u32>(t) & 1) * 8;
                                samp = tx.half((no >> 2) & idx_mask);
                                if (p.tlut) {
                                    if (tile.size == 0) {
                                        samp >>= 12 - 4 * static_cast<int>(no & 3);
                                        samp &= 0xf;
                                        samp |= tile.palette << 4;
                                    } else {
                                        samp >>= 8 - 4 * static_cast<int>(no & 2);
                                        samp &= 0xff;
                                    }
                                    samp <<= 2;
                                    samp += so;
                                    samp = tx.half(static_cast<u32>(samp | 0x400) & 0x7ff);
                                }
                            }
                            return samp;
                        };
                        s32 samp;
                        if (fb_size == 0) samp = 0;
                        else if (fb_size == 1) {
                            samp = copy_word_at(s_offset >> 1);
                            samp >>= 8 - 8 * (s_offset & 1);
                            samp &= 0xff;
                        } else {
                            samp = copy_word_at(s_offset);
                        }
                        if (p.alpha_test && fb_size == 2 && (samp & 1) == 0) continue;
                        copy_word = static_cast<u32>(samp);
                        kind = K_COPY;
                    } else if (p.cycle == 3) { // fill
                        if (x < cur.start_x || x > cur.end_x) continue;
                        kind = K_FILL;
                    } else {
                        const u32 coverage = compute_coverage(cur.xleft, cur.xright, x);
                        if (coverage == 0) continue;
                        s32 coverage_count = __builtin_popcount(coverage);
                        if (!p.aa && (coverage & 1) == 0) continue;

                        PixIn in{};
                        pixel_inputs(x, coverage, in);
                        const s32* shade = in.shade;
                        const T4& texel0 = in.texel0;
                        const T4& texel1 = in.texel1;
                        const s32 lod_frac = in.lod_frac;
                        z = in.z;

                        s32 rgb_dith, alpha_dith;
                        dither_coefficients(x, y >> (p.interlace ? 1 : 0), p.dither >> 2, p.dither & 3, noise, rgb_dith,
                                            alpha_dith);

                        // Combiner
                        auto byte_of = [](u32 c, int i) { return static_cast<s32>((c >> (24 - 8 * i)) & 0xff); };
                        // Chroma key: the RGB inputs A and the unrounded sums, per channel.
                        s32 key_a[3]{}, key_sum[3]{};
                        auto run_cycle = [&](int cyc, const PixIn& pin, const s32 combined[4], const T4& t0, const T4& t1, s32 noise_v,
                                             s32 out[4]) {
                            for (int i = 0; i < 4; ++i) {
                                csrc[S_COMB][i] = combined[i];
                                csrc[S_T0][i] = t0.v[i];
                                csrc[S_T1][i] = t1.v[i];
                                csrc[S_SHADE][i] = pin.shade[i];
                                csrc[S_NOISE][i] = noise_v;
                                csrc[S_COMBA][i] = combined[3];
                                csrc[S_T0A][i] = t0.v[3];
                                csrc[S_T1A][i] = t1.v[3];
                                csrc[S_SHADEA][i] = pin.shade[3];
                                csrc[S_LOD][i] = pin.lod_frac;
                            }
                            const u8 (*sel)[4] = csel[cyc];
                            for (int i = 0; i < 4; ++i) {
                                const s32 va = csrc[sel[0][i]][i], vb = csrc[sel[1][i]][i];
                                const s32 vc = csrc[sel[2][i]][i], vd = csrc[sel[3][i]][i];
                                out[i] = combiner_equation(va, vb, vc, vd);
                                if (i < 3 && p.key_en) {
                                    key_a[i] = va;
                                    key_sum[i] = w32(static_cast<s64>(special_expand(va) - special_expand(vb)) * sext(vc, 9) +
                                                     (static_cast<s64>(special_expand(vd)) << 8) + 0x80) & 0x1ffff;
                                }
                            }
                        };

                        s32 alpha_reference = 0;
                        const s32 noise_v = noise.combiner();
                        auto finish_cycle1 = [&](s32 out[4]) {
                            for (int i = 0; i < 4; ++i) out[i] = clamp_9bit(out[i]);
                            s32 key_alpha = 0;
                            if (p.key_en) {
                                // The colour bypasses the combiner (input A); how close its
                                // result is to the key centre becomes the alpha.
                                key_alpha = 0x7fffffff;
                                for (int i = 0; i < 3; ++i) {
                                    s32 k = sext(key_sum[i], 17);
                                    if (k > 0) k = (k & 0xf) == 8 ? -k + 0x10 : -k;
                                    k += p.key_width[i] << 4;
                                    key_alpha = std::min(key_alpha, k);
                                    out[i] = clamp_9bit(key_a[i]);
                                }
                                key_alpha = std::clamp(key_alpha, 0, 0xff);
                            }
                            const s32 expanded = out[3] + ((out[3] + 1) >> 8);
                            s32 modulated;
                            if (p.cvg_times_alpha) {
                                modulated = (expanded * coverage_count + 4) >> 3;
                                coverage_count = modulated >> 5;
                            } else {
                                modulated = coverage_count << 5;
                            }
                            const s32 e = p.alpha_cvg_select ? modulated : p.key_en ? key_alpha : expanded + alpha_dith;
                            out[3] = std::clamp(e, 0, 0xff);
                        };
                        if (multi) {
                            const s32 zero[4] = {0, 0, 0, 0};
                            s32 c0[4];
                            run_cycle(0, in, zero, texel0, texel1, noise_v, c0);
                            if (p.alpha_test) {
                                // Pipelining: a pixel's alpha compare sees the first
                                // cycle of the pixel after it (in the direction the
                                // span is walked; past its end, a pixel with no
                                // coverage), with this pixel's alpha dither.
                                const s32 xn = x + idir;
                                const bool last = x == (flip ? cur.end_x : cur.start_x);
                                const u32 cov_n = last ? 0u : compute_coverage(cur.xleft, cur.xright, xn);
                                PixIn nin{};
                                pixel_inputs(xn, cov_n, nin);
                                s32 n0[4];
                                run_cycle(0, nin, zero, nin.texel0, nin.texel1, noise_v, n0);
                                const s32 cnt_n = __builtin_popcount(cov_n);
                                const s32 ca = clamp_9bit(n0[3]);
                                s32 ea = ca + ((ca + 1) >> 8);
                                if (p.alpha_cvg_select) ea = p.cvg_times_alpha ? (ea * cnt_n + 4) >> 3 : cnt_n << 5;
                                else ea += alpha_dith;
                                alpha_reference = std::clamp(ea, 0, 0xff);
                            }
                            s32 nv = noise_v;
                            if (p.need_noise_dual) {
                                Noise n2;
                                n2.reseed(static_cast<u32>(x + 1023), static_cast<u32>(y + 7), p.prim_serial + 11);
                                nv = n2.combiner();
                            }
                            // The second cycle sees the texels swapped (pipelining).
                            run_cycle(1, in, c0, texel1, texel0, nv, comb);
                            finish_cycle1(comb);
                        } else {
                            const s32 zero[4] = {0, 0, 0, 0};
                            run_cycle(1, in, zero, texel0, texel1, noise_v, comb);
                            finish_cycle1(comb);
                            alpha_reference = comb[3];
                        }

                        if (p.aa && coverage_count == 0) continue;
                        if (p.alpha_test) {
                            const s32 threshold = p.alpha_test_dither ? noise.blend_threshold() : static_cast<s32>(p.blend_color & 0xff);
                            if (alpha_reference < threshold) continue;
                        }
                        dith = rgb_dith;
                        cov_count = coverage_count;
                        shade_a = std::min(shade[3] + alpha_dith, 0xff);
                        kind = K_NORMAL;
                    }
                    if (kind == K_NONE) continue;
                    static const long dbg_addr = [] {
                        const char* e = std::getenv("ORBIT64_EXACT_PIX");
                        return e ? std::strtol(e, nullptr, 16) : -1L;
                    }();

                    // ---- Memory: load the pixel's colour and depth
                    s32 col[4]{};
                    const size_t cidx = read_colour(nx, ny, col);
                    u16 cur_depth = 0;
                    u8 cur_dz = 0;
                    bool color_dirty = false, depth_dirty = false;
                    size_t zidx = (p.z_index + static_cast<size_t>(p.fb_width) * static_cast<u32>(ny) + static_cast<u32>(nx)) & mask16;
                    {
                        const u16 wv = rd16(zidx);
                        cur_depth = wv >> 2;
                        cur_dz = static_cast<u8>(hidden_get(zidx, wv) | ((wv & 3) << 2));
                    }

                    auto write_color = [&](s32 r, s32 g, s32 b, s32 a) {
                        if (p.fb_fmt == FB_I4) {
                            col[0] = r;
                            col[1] = g;
                            col[2] = b;
                        } else {
                            col[0] = r;
                            col[1] = g;
                            col[2] = b;
                            col[3] = a;
                        }
                        color_dirty = true;
                    };
                    auto alias_color_to_depth = [&]() {
                        if (p.fb_fmt == FB_RGBA5551) {
                            cur_dz = static_cast<u8>(((col[3] & 0xff) >> 3) | (col[2] & 8));
                            u32 wv = (static_cast<u32>(col[0]) & 0xf8) << 6;
                            wv |= (static_cast<u32>(col[1]) & 0xf8) << 1;
                            wv |= (static_cast<u32>(col[2]) & 0xf8) >> 4;
                            cur_depth = static_cast<u16>(wv);
                        } else if (p.fb_fmt == FB_IA88) {
                            const u32 wv = (static_cast<u32>(col[0]) << 8) | static_cast<u32>(col[3]);
                            cur_depth = static_cast<u16>(wv >> 2);
                            cur_dz = static_cast<u8>(((wv & 3) << 2) | ((wv & 1) * 3));
                        }
                    };
                    auto alias_depth_to_color = [&]() {
                        const u32 wv = (static_cast<u32>(cur_depth) << 4) | cur_dz;
                        if (p.fb_fmt == FB_RGBA5551) {
                            col[0] = (wv >> 10) & 0xf8;
                            col[1] = (wv >> 5) & 0xf8;
                            col[2] = wv & 0xf8;
                            col[3] = (wv & 7) << 5;
                        } else if (p.fb_fmt == FB_IA88) {
                            col[0] = (wv >> 10) & 0xff;
                            col[3] = (wv >> 2) & 0xff;
                        }
                        color_dirty = true;
                    };

                    if (kind == K_FILL) {
                        u32 c = p.fill_color;
                        switch (p.fb_fmt) {
                            case FB_RGBA8888: write_color((c >> 24) & 0xff, (c >> 16) & 0xff, (c >> 8) & 0xff, c & 0xff); break;
                            case FB_RGBA5551:
                                c >>= ((cidx & 1) ^ 1) * 16;
                                write_color((c >> 8) & 0xf8, (c >> 3) & 0xf8, (c << 2) & 0xf8, (c & 1) * 0xe0);
                                break;
                            case FB_IA88:
                                c >>= ((cidx & 1) ^ 1) * 16;
                                c &= 0xffff;
                                write_color((c >> 8) & 0xff, (c >> 8) & 0xff, (c >> 8) & 0xff, c & 0xff);
                                break;
                            case FB_I8:
                                c >>= ((cidx & 3) ^ 3) * 8;
                                c &= 0xff;
                                write_color(c, c, c, c);
                                break;
                            default: break;
                        }
                        if (p.alias) alias_color_to_depth();
                    } else if (kind == K_COPY) {
                        const u32 wv = copy_word;
                        switch (p.fb_fmt) {
                            case FB_I4:
                                col[0] = col[1] = col[2] = col[3] = 0;
                                color_dirty = true;
                                break;
                            case FB_I8: write_color(wv & 0xff, wv & 0xff, wv & 0xff, wv & 0xff); break;
                            case FB_RGBA5551:
                                write_color((wv >> 8) & 0xf8, (wv >> 3) & 0xf8, (wv << 2) & 0xf8, (wv & 1) * 0xe0);
                                break;
                            default: break;
                        }
                        if (p.alias) alias_color_to_depth();
                    } else {
                        // ---- Depth test
                        // Memory colour as the blender sees it.
                        s32 mem[4];
                        decode_memory(col, mem);
                        const s32 memory_coverage = mem[3] >> 5;

                        bool blend_en, coverage_wrap, z_pass;
                        s32 shift_a, shift_b;
                        if (p.z_compare) {
                            const s32 memory_z = z_decompress(cur_depth);
                            s32 memory_dz = dz_decompress(cur_dz);
                            const s32 precision = (cur_depth >> 11) & 0xf;
                            bool coplanar = false;
                            shift_a = std::clamp(p.dz_compressed - static_cast<s32>(cur_dz), 0, 4);
                            shift_b = std::clamp(static_cast<s32>(cur_dz) - p.dz_compressed, 0, 4);
                            if (precision < 3) {
                                if (memory_dz != 0x8000) {
                                    memory_dz = std::max(memory_dz << 1, 16 >> precision);
                                } else {
                                    coplanar = true;
                                    memory_dz = 0xffff;
                                }
                            }
                            s32 combined_dz = combine_dz(p.dz | memory_dz);
                            const s32 combined_dz_ip = combined_dz;
                            combined_dz <<= 3;
                            const bool farther = coplanar || (z + combined_dz) >= memory_z;
                            const bool overflow = (cov_count + memory_coverage) >= 8;
                            blend_en = p.force_blend || (!overflow && p.aa && farther);
                            coverage_wrap = overflow;
                            const bool max_z = memory_z == 0x3ffff;
                            const bool front = z < memory_z;
                            const bool nearer = coplanar || (z - combined_dz) <= memory_z;
                            switch (p.z_mode) {
                                case 0: z_pass = max_z || (overflow ? front : nearer); break;
                                case 1:
                                    if (!front || !farther || !overflow) {
                                        z_pass = max_z || (overflow ? front : nearer);
                                    } else {
                                        const s32 cdz = dz_compress(combined_dz_ip & 0xffff);
                                        const s32 coeff = ((memory_z >> cdz) - (z >> cdz)) & 0xf;
                                        cov_count = std::min((coeff * cov_count) >> 3, 8);
                                        z_pass = true;
                                    }
                                    break;
                                case 2: z_pass = front || max_z; break;
                                default: z_pass = farther && nearer && !max_z; break;
                            }
                        } else {
                            shift_a = 0;
                            shift_b = std::min(0xf - p.dz_compressed, 4);
                            const bool overflow = (cov_count + memory_coverage) >= 8;
                            blend_en = p.force_blend || (!overflow && p.aa);
                            coverage_wrap = overflow;
                            z_pass = true;
                        }

                        if (z_pass && (!p.aa || cov_count != 0)) {
                            s32 pixel[4] = {comb[0], comb[1], comb[2], comb[3]};
                            const s32 fog[4] = {static_cast<s32>(p.fog_color >> 24), static_cast<s32>((p.fog_color >> 16) & 0xff),
                                                static_cast<s32>((p.fog_color >> 8) & 0xff), static_cast<s32>(p.fog_color & 0xff)};
                            const s32 bcol[4] = {static_cast<s32>(p.blend_color >> 24), static_cast<s32>((p.blend_color >> 16) & 0xff),
                                                 static_cast<s32>((p.blend_color >> 8) & 0xff), static_cast<s32>(p.blend_color & 0xff)};
                            auto blender = [&](const u8 m[4], bool final_cycle, s32 out[3]) {
                                const s32* memc = (multi && !final_cycle) ? delayed_mem : mem;
                                const s32* src1 = m[2] == 0 ? pixel : m[2] == 1 ? memc : m[2] == 2 ? bcol : fog;
                                if (final_cycle && p.color_on_cvg && !coverage_wrap) {
                                    for (int i = 0; i < 3; ++i) out[i] = src1[i];
                                    return;
                                }
                                const s32* src0 = m[0] == 0 ? pixel : m[0] == 1 ? memc : m[0] == 2 ? bcol : fog;
                                if (final_cycle && (!blend_en || (m[1] == 0 && m[3] == 0 && pixel[3] == 0xff))) {
                                    for (int i = 0; i < 3; ++i) out[i] = src0[i];
                                    return;
                                }
                                s32 a0, a1;
                                switch (m[1]) {
                                    case 0: a0 = pixel[3]; break;
                                    case 1: a0 = fog[3]; break;
                                    case 2: a0 = shade_a; break;
                                    default: a0 = 0; break;
                                }
                                switch (m[3]) {
                                    case 0: a1 = ~a0 & 0xff; break;
                                    case 1: a1 = memc[3]; break;
                                    case 2: a1 = 0xff; break;
                                    default: a1 = 0; break;
                                }
                                a0 >>= 3;
                                a1 >>= 3;
                                if (m[3] == 1) {
                                    a0 = (a0 >> shift_a) & 0x3c;
                                    a1 = (a1 >> shift_b) | 3;
                                }
                                s32 bl[3];
                                for (int i = 0; i < 3; ++i) bl[i] = src0[i] * a0 + src1[i] * (a1 + 1);
                                if (!final_cycle || p.force_blend) {
                                    for (int i = 0; i < 3; ++i) out[i] = (bl[i] >> 5) & 0xff;
                                } else {
                                    const s32 sum = (a0 >> 2) + (a1 >> 2) + 1;
                                    for (int i = 0; i < 3; ++i) out[i] = bdiv.lut[((sum << 11) | ((bl[i] >> 2) & 0x7ff)) & 0x7fff];
                                }
                            };
                            s32 rgb[3];
                            if (multi) {
                                s32 first_out[3];
                                blender(p.blend[0], false, first_out);
                                pixel[0] = first_out[0];
                                pixel[1] = first_out[1];
                                pixel[2] = first_out[2];
                                blender(p.blend[1], true, rgb);
                            } else {
                                blender(p.blend[0], true, rgb);
                            }
                            if (p.dither_en) rgb_dither(rgb, dith);
                            const s32 new_cov = blend_coverage(cov_count, memory_coverage, blend_en, p.coverage_mode);
                            write_color(rgb[0], rgb[1], rgb[2], new_cov << 5);
                            if (p.z_update) {
                                cur_depth = z_compress(z);
                                cur_dz = static_cast<u8>(p.dz_compressed);
                                depth_dirty = true;
                                if (p.alias) alias_depth_to_color();
                            } else if (p.alias) {
                                alias_color_to_depth();
                            }
                        }
                    }

                    if (dbg_addr >= 0 && !hires) {
                        const size_t bpp = p.fb_fmt == FB_RGBA8888 ? 4 : p.fb_fmt <= FB_I8 ? 1 : 2;
                        if (static_cast<long>(cidx * bpp) == (dbg_addr & ~static_cast<long>(bpp - 1)))
                            std::fprintf(stderr, "PIX x=%d y=%d kind=%d comb=%d,%d,%d,%d cvg=%d dith=%03x z=%05x -> col=%d,%d,%d,%d\n", x,
                                         y, kind, comb[0], comb[1], comb[2], comb[3], cov_count, dith, z, col[0], col[1],
                                         col[2], col[3]);
                    }
                    // ---- Store
                    if (color_dirty) {
                        switch (p.fb_fmt) {
                            case FB_I4: case FB_I8: {
                                // The ninth bits go with the odd byte; writing the even
                                // one leaves them as they were.
                                const u8 before = hidden_get(cidx >> 1, rd16(cidx >> 1));
                                const s32 c = p.fb_fmt == FB_I4 ? 0 : (cidx & 1) ? col[1] : col[0];
                                fbm[cidx] = static_cast<u8>(c);
                                const u8 bits = p.fb_fmt == FB_I4 ? static_cast<u8>(col[3] & 3) : static_cast<u8>((c & 1) * 3);
                                hidden_set(cidx >> 1, (cidx & 1) ? bits : before);
                                break;
                            }
                            case FB_RGBA5551: {
                                const u32 r = static_cast<u32>(col[0]) & 0xf8, g = static_cast<u32>(col[1]) & 0xf8,
                                          b = static_cast<u32>(col[2]) & 0xf8;
                                const u32 cov = (static_cast<u32>(col[3]) & 0xff) >> 5;
                                wr16(cidx, static_cast<u16>((r << 8) | (g << 3) | (b >> 2) | (cov >> 2)));
                                hidden_set(cidx, static_cast<u8>(cov & 3));
                                break;
                            }
                            case FB_IA88: {
                                const u32 wv = ((static_cast<u32>(col[0]) & 0xff) << 8) | (static_cast<u32>(col[3]) & 0xff);
                                wr16(cidx, static_cast<u16>(wv));
                                hidden_set(cidx, static_cast<u8>((col[3] & 1) * 3));
                                break;
                            }
                            default: {
                                u8* q = fbm + cidx * 4;
                                q[0] = static_cast<u8>(col[0]);
                                q[1] = static_cast<u8>(col[1]);
                                q[2] = static_cast<u8>(col[2]);
                                q[3] = static_cast<u8>(col[3]);
                                hidden_set(2 * cidx, static_cast<u8>((col[1] & 1) * 3));
                                hidden_set(2 * cidx + 1, static_cast<u8>((col[3] & 1) * 3));
                                break;
                            }
                        }
                    }
                    if (!p.alias && depth_dirty) {
                        wr16(zidx, static_cast<u16>((cur_depth << 2) | (cur_dz >> 2)));
                        hidden_set(zidx, static_cast<u8>(cur_dz & 3));
                    }
                }
            }
            if (!hires && y == last_row) {
                std::lock_guard<std::mutex> lk(hs_->tail_mutex);
                if (static_cast<s32>(p.prim_serial - hs_->tail_serial) > 0) {
                    hs_->tail_serial = p.prim_serial;
                    std::copy(prev_mem_, prev_mem_ + 4, hs_->tail_mem);
                }
            }
        }
    }
}

bool ExactRdp::draw_needs_serial() const {
    if (((other_h_ >> 20) & 3) != 1) return false;
    // The first blender cycle's 1a, 2a (memory colour) and 2b (memory alpha).
    return ((other_l_ >> 30) & 3) == 1 || ((other_l_ >> 22) & 3) == 1 || ((other_l_ >> 18) & 3) == 1;
}

// ============================================================================
// TMEM loads
// ============================================================================
//
// Loads run through the RDP like a rectangle: every row of the image it
// covers is fetched from RDRAM 64 bits at a time, the S and T of each
// 64-bit word turned into a TMEM address by the tile descriptor, and the four
// 16-bit pieces written to the four TMEM banks - swapped on odd rows,
// split between the halves for 32-bit and YUV tiles, and replicated four
// times for TLUT entries.

void ExactRdp::load(const u32* w, LoadMode mode, const u8* rdram, size_t rdram_size) {
    const u32 tilenum = (w[1] >> 24) & 7;
    Tile& tile = tiles_[tilenum];
    const u32 sl = (w[0] >> 12) & 0xfff, tl = w[0] & 0xfff;
    const u32 sh = (w[1] >> 12) & 0xfff, th = w[1] & 0xfff; // th = DxT for Load Block
    tile.slo = sl;
    tile.tlo = tl;
    tile.shi = sh;
    tile.thi = th;

    const bool ltlut = mode == LoadMode::Tlut;
    const bool coord_quad = mode != LoadMode::Tile;
    if (ti_size_ == 0) return; // 4-bit images hang the RDP

    int formatting;
    if (tile.fmt == TF_YUV) formatting = 0;
    else if (tile.fmt == TF_RGBA && tile.size == 3) formatting = 1;
    else formatting = 2;

    u32 ti_adv = 8, span_adv = 8;
    switch (ti_size_) {
        case 1: ti_adv = 8; span_adv = 8; break;
        case 2: ti_adv = ltlut ? 2 : 8; span_adv = ltlut ? 1 : 4; break;
        default: ti_adv = 8; span_adv = 2; break;
    }

    // The load as the rasterizer walks it: rows, S range, S/T and their steps
    // (S and T in s10.5 << 16).
    s32 row_first, row_last, x_first, x_last;
    s32 s, t, ds, dt, dt_row;
    if (mode == LoadMode::Block) {
        const u32 tlc = tl & 0x3ff;
        row_first = row_last = static_cast<s32>(tlc);
        x_first = static_cast<s32>(sl);
        x_last = static_cast<s32>(sh);
        s = static_cast<s32>(((sl << 3) & 0xffff) << 16);
        t = static_cast<s32>(((tl << 3) & 0xffff) << 16);
        ds = static_cast<s32>((0x80u >> ti_size_) << 16);
        dt = static_cast<s32>(th << 8);
        dt_row = 0x20 << 16;
    } else {
        row_first = static_cast<s32>(tl >> 2);
        row_last = static_cast<s32>(th >> 2);
        x_first = static_cast<s32>(sl >> 2);
        x_last = static_cast<s32>(sh >> 2);
        s = static_cast<s32>(((sl << 3) & 0xffff) << 16);
        t = static_cast<s32>(((tl << 3) & 0xffff) << 16);
        ds = static_cast<s32>((0x200u >> ti_size_) << 16);
        dt = 0;
        dt_row = 0x20 << 16;
    }
    ds &= ~0x1f;
    dt &= ~0x1f;
    if (row_last > row_first && ltlut) return; // multi-row TLUT loads hang the RDP
    const s32 length = (x_last - x_first + 1) & 0xfff;

    auto read32 = [&](u32 idx) -> u32 {
        const size_t a = static_cast<size_t>(idx) * 4;
        if (a + 3 >= rdram_size) return 0;
        return (static_cast<u32>(rdram[a]) << 24) | (static_cast<u32>(rdram[a + 1]) << 16) |
               (static_cast<u32>(rdram[a + 2]) << 8) | rdram[a + 3];
    };

    s32 row_t = t;
    for (s32 row = row_first; row <= row_last; ++row) {
        s32 cs = s & ~0x3ff, ct = row_t & ~0x3ff;
        row_t += dt_row;
        if (row & ~0xfff) continue; // outside the rasterizer's range
        const u32 ti_index = ti_width_ * static_cast<u32>(row) + static_cast<u32>(x_first);
        u32 tiptr = ti_addr_ + ((ti_index << ti_size_) >> 1);

        for (s32 j = 0; j < length; j += static_cast<s32>(span_adv)) {
            // Texture coordinates relative to the tile, in texels (quarter
            // texels for Block and TLUT).
            s32 ss = static_cast<s16>((cs >> 16) & 0xffff), st = static_cast<s16>((ct >> 16) & 0xffff);
            ss -= static_cast<s32>(tile.slo << 3);
            st -= static_cast<s32>(tile.tlo << 3);
            ss >>= coord_quad ? 3 : 5;
            st >>= coord_quad ? 3 : 5;
            const bool dswap = st & 1;

            // TMEM address of the 64-bit word: its four 16-bit indices by bank.
            u32 tbase = (static_cast<u32>(tile.stride >> 3) * static_cast<u32>(st)) & 0x1ff;
            tbase += tile.offset >> 3;
            u32 sshorts;
            if (tile.size == 1 || tile.fmt == TF_YUV) sshorts = static_cast<u32>(ss) >> 1;
            else if (tile.size >= 2) sshorts = static_cast<u32>(ss);
            else sshorts = static_cast<u32>(ss) >> 2;
            sshorts &= 0x7ff;
            const bool bit3fl = ((sshorts & 2) != 0) ^ ((st & 1) != 0);
            u32 ia = ((tbase << 2) + sshorts) & 0x7fd;
            u32 ib = (ia + 1) & 0x7ff, ic = (ia + 2) & 0x7ff, id = (ia + 3) & 0x7ff;
            const bool hibit = (ia & 0x400) != 0;
            if (st & 1) {
                ia ^= 2;
                ib ^= 2;
                ic ^= 2;
                id ^= 2;
            }
            u32 bank[4];
            for (u32 k = 0; k < 4; ++k) {
                if ((ia & 3) == k) bank[k] = ia & 0x3ff;
                else if ((ib & 3) == k) bank[k] = ib & 0x3ff;
                else if ((ic & 3) == k) bank[k] = ic & 0x3ff;
                else if ((id & 3) == k) bank[k] = id & 0x3ff;
                else bank[k] = 0;
            }

            // 64 bits of RDRAM at tiptr (any alignment); 16-bit TLUT loads
            // take one texel and repeat it.
            const u32 ri = (tiptr >> 2) & ~1u;
            const u32 r0 = read32(ri), r1 = read32(ri + 1), r2 = read32(ri + 2), r3 = read32(ri + 3);
            u64 q = 0;
            auto rep = [](u32 v) { const u64 x = v & 0xffff; return (x << 48) | (x << 32) | (x << 16) | x; };
            switch (tiptr & 7) {
                case 0: q = ltlut ? rep(r0 >> 16) : (static_cast<u64>(r0) << 32) | r1; break;
                case 1: q = (static_cast<u64>(r0) << 40) | (static_cast<u64>(r1) << 8) | (r2 >> 24); break;
                case 2: q = ltlut ? rep(r0) : (static_cast<u64>(r0) << 48) | (static_cast<u64>(r1) << 16) | (r2 >> 16); break;
                case 3: q = (static_cast<u64>(r0) << 56) | (static_cast<u64>(r1) << 24) | (r2 >> 8); break;
                case 4: q = ltlut ? rep(r1 >> 16) : (static_cast<u64>(r1) << 32) | r2; break;
                case 5: q = (static_cast<u64>(r1) << 40) | (static_cast<u64>(r2) << 8) | (r3 >> 24); break;
                case 6: q = ltlut ? rep(r1) : (static_cast<u64>(r1) << 48) | (static_cast<u64>(r2) << 16) | (r3 >> 16); break;
                default: q = (static_cast<u64>(r1) << 56) | (static_cast<u64>(r2) << 24) | (r3 >> 8); break;
            }

            switch (formatting) {
                case 0: case 1: {
                    // YUV: chroma (U, V) to the lower half, luma to the upper;
                    // 32-bit: R, G to the lower half, B, A to the upper.
                    u32 lo, hi;
                    if (formatting == 0) {
                        lo = static_cast<u32>((((q >> 56) & 0xff) << 24) | (((q >> 40) & 0xff) << 16) | (((q >> 24) & 0xff) << 8) | ((q >> 8) & 0xff));
                        hi = static_cast<u32>((((q >> 48) & 0xff) << 24) | (((q >> 32) & 0xff) << 16) | (((q >> 16) & 0xff) << 8) | (q & 0xff));
                    } else {
                        lo = static_cast<u32>(((q >> 48) << 16) | ((q >> 16) & 0xffff));
                        hi = static_cast<u32>((((q >> 32) & 0xffff) << 16) | (q & 0xffff));
                    }
                    const u32 b0 = bit3fl ? bank[2] : bank[0], b1 = bit3fl ? bank[3] : bank[1];
                    tmem_[b0] = static_cast<u16>(lo >> 16);
                    tmem_[b1] = static_cast<u16>(lo & 0xffff);
                    tmem_[b0 | 0x400] = static_cast<u16>(hi >> 16);
                    tmem_[b1 | 0x400] = static_cast<u16>(hi & 0xffff);
                    break;
                }
                default: {
                    const u32 half = hibit ? 0x400 : 0;
                    const u16 p0 = static_cast<u16>(q >> 48), p1 = static_cast<u16>(q >> 32), p2 = static_cast<u16>(q >> 16),
                              p3 = static_cast<u16>(q);
                    if (!dswap) {
                        tmem_[bank[0] | half] = p0;
                        tmem_[bank[1] | half] = p1;
                        tmem_[bank[2] | half] = p2;
                        tmem_[bank[3] | half] = p3;
                    } else {
                        tmem_[bank[0] | half] = p2;
                        tmem_[bank[1] | half] = p3;
                        tmem_[bank[2] | half] = p0;
                        tmem_[bank[3] | half] = p1;
                    }
                    break;
                }
            }

            cs = (cs + ds) & ~0x1f;
            ct = (ct + dt) & ~0x1f;
            tiptr += ti_adv;
        }
    }
}

// ============================================================================
// Lanes
// ============================================================================

#include "raster_pool.hpp"
#include <thread>

ExactRdpLanes::ExactRdpLanes() {
    const unsigned hw = std::thread::hardware_concurrency();
    u32 n = std::clamp<u32>(hw > 1 ? hw - 1 : 1, 1, 8);
    if (const char* e = std::getenv("ORBIT64_RDP_LANES")) n = std::clamp(std::atoi(e), 1, 16); // debugging
    for (u32 i = 0; i < n; ++i) {
        lanes_.push_back(std::make_unique<ExactRdp>());
        if (i) lanes_[i]->share_hidden_with(*lanes_[0]);
        lanes_[i]->set_lane(i, n);
    }
    if (n > 1) pool_ = std::make_unique<RasterPool>(n - 1);
}

ExactRdpLanes::~ExactRdpLanes() = default;

void ExactRdpLanes::reset() {
    buf_.clear();
    starts_.clear();
    for (auto& l : lanes_) l->reset();
    other_h_ = other_l_ = 0;
    ti_addr_ = 0;
    ti_width_ = 1;
    ti_size_ = 0;
    ci_addr_ = zi_addr_ = 0;
    ci_width_ = 1;
    ci_size_ = 2;
        scissor_yhi_ = scissor_xhi_ = 0;
        written_lo_ = ~0ull;
        written_hi_ = 0;
        sync_ranges_.clear();
    }

    void ExactRdpLanes::set_scale(u32 scale) {
        flush();
        for (auto& l : lanes_) l->set_scale(scale);
    }

    void ExactRdpLanes::add_sync_range(u64 lo, u64 hi) {
        if (scale() <= 1) return;
        for (auto& r : sync_ranges_)
            if (lo <= r.second && hi >= r.first) {
                r.first = std::min(r.first, lo);
                r.second = std::max(r.second, hi);
                return;
            }
        sync_ranges_.emplace_back(lo, hi);
    }

    // What a draw under the current state can write: the colour and the depth
    // image down to the scissor's last line.
    void ExactRdpLanes::draw_ranges(u64& lo, u64& hi, u64& zlo, u64& zhi) const {
        auto bytes_pp = [](u32 size) { return size == 0 ? 1u : 1u << (size - 1); };
        const u64 rows = (static_cast<u64>(scissor_yhi_) >> 2) + 1;
        lo = ci_addr_;
        hi = ci_addr_ + (rows + 1) * ci_width_ * bytes_pp(ci_size_) + 8;
        zlo = zi_addr_;
        zhi = zi_addr_ + (rows + 1) * ci_width_ * 2 + 8;
    }

void ExactRdpLanes::flush() {
    if (starts_.empty()) return;
    // Internal resolution: every lane draws into lane 0's high-resolution
    // copy, which first takes what the CPU changed in the frame buffers.
    if (scale() > 1) {
        const UpStore* up = lanes_[0]->upscaled(rdram_size_);
        for (size_t i = 1; i < lanes_.size(); ++i)
            if (lanes_[i]->upscaled() != up) lanes_[i]->share_upscaled_with(*lanes_[0]);
        for (const auto& r : sync_ranges_) lanes_[0]->up_sync_before(r.first, r.second, rdram_, rdram_size_);
    }
    auto run_lane = [&](u32 k) {
        for (u32 st : starts_) lanes_[k]->command(&buf_[st], rdram_, rdram_size_);
    };
    static const bool no_pool = std::getenv("ORBIT64_RDP_NOPOOL") != nullptr; // debugging
    if (pool_ && !no_pool) pool_->run(static_cast<u32>(lanes_.size()), run_lane);
    else
        for (u32 k = 0; k < lanes_.size(); ++k) run_lane(k);
    for (const auto& r : sync_ranges_) lanes_[0]->up_sync_after(r.first, r.second, rdram_, rdram_size_);
    sync_ranges_.clear();
    buf_.clear();
    starts_.clear();
    written_lo_ = ~0ull;
    written_hi_ = 0;
}

void ExactRdpLanes::cpu_wrote(u32 paddr, u32 len) {
    const auto& watched = lanes_[0]->watched_pages();
    if (watched.empty()) return;
    bool any = false;
    for (u32 pg = paddr >> 6; pg <= ((paddr + len - 1) >> 6) && pg < watched.size(); ++pg) any |= watched[pg] != 0;
    if (!any) return;
    flush();
    lanes_[0]->cpu_wrote(paddr, len);
}

void ExactRdpLanes::run_serial(const u32* w, u32 nwords) {
    (void)nwords;
    u64 lo, hi, zlo, zhi;
    draw_ranges(lo, hi, zlo, zhi);
    add_sync_range(lo, hi);
    add_sync_range(zlo, zhi);
    if (scale() > 1) {
        const UpStore* up = lanes_[0]->upscaled(rdram_size_);
        for (size_t i = 1; i < lanes_.size(); ++i)
            if (lanes_[i]->upscaled() != up) lanes_[i]->share_upscaled_with(*lanes_[0]);
        for (const auto& r : sync_ranges_) lanes_[0]->up_sync_before(r.first, r.second, rdram_, rdram_size_);
    }
    for (auto& l : lanes_) {
        l->set_serial(true);
        l->command(w, rdram_, rdram_size_);
        l->set_serial(false);
    }
    for (const auto& r : sync_ranges_) lanes_[0]->up_sync_after(r.first, r.second, rdram_, rdram_size_);
    sync_ranges_.clear();
}

bool ExactRdpLanes::command(const u32* w, u8* rdram, size_t rdram_size) {
    const u32 op = (w[0] >> 24) & 0x3f;
    const u32 nwords = ExactRdp::command_words(op) * 2;
    if (rdram_ != rdram || rdram_size_ != rdram_size) flush();
    rdram_ = rdram;
    rdram_size_ = rdram_size;
    if (lanes_.size() == 1 && scale() <= 1) return lanes_[0]->command(w, rdram, rdram_size);
    // The ninth bits are allocated before any lane runs.
    if (lanes_[0]->hidden().size() != rdram_size / 2) {
        flush();
        const u32 nop[2] = {0, 0};
        lanes_[0]->command(nop, rdram, rdram_size);
    }

    const bool pending_draws = written_hi_ > written_lo_;
    auto bytes_pp = [](u32 size) { return size == 0 ? 1u : 1u << (size - 1); };
    bool draw = false;
    switch (op) {
        case 0x29: // Sync Full
            flush();
            return true;
        case 0x2d:
            scissor_yhi_ = w[1] & 0xfff;
            scissor_xhi_ = (w[1] >> 12) & 0xfff;
            break;
        case 0x2f:
            other_h_ = w[0] & 0x00ffffff;
            other_l_ = w[1];
            break;
        case 0x3d:
            ti_size_ = (w[0] >> 19) & 3;
            ti_width_ = (w[0] & 0x3ff) + 1;
            ti_addr_ = w[1] & 0x00ffffff;
            break;
        case 0x3e: case 0x3f:
            if (pending_draws) flush();
            if (op == 0x3e) {
                zi_addr_ = w[1] & 0x00ffffff;
            } else {
                ci_size_ = (w[0] >> 19) & 3;
                ci_width_ = (w[0] & 1023) + 1;
                ci_addr_ = w[1] & 0x00ffffff;
            }
            break;
        case 0x30: case 0x33: case 0x34: {
            // A load from memory the queued draws may be writing waits for them.
            if (pending_draws) {
                const u32 sl = (w[0] >> 12) & 0xfff, tl = w[0] & 0xfff, sh = (w[1] >> 12) & 0xfff, th = w[1] & 0xfff;
                const u64 bpp = bytes_pp(ti_size_);
                u64 lo, hi;
                if (op == 0x33) {
                    lo = ti_addr_ + (static_cast<u64>(tl & 0x3ff) * ti_width_ + sl) * bpp;
                    hi = lo + (static_cast<u64>(sh) + 1) * bpp + 16;
                } else {
                    lo = ti_addr_ + static_cast<u64>(tl >> 2) * ti_width_ * bpp;
                    hi = ti_addr_ + (static_cast<u64>(th >> 2) + 1) * ti_width_ * bpp + 16;
                }
                if (lo < written_hi_ && hi > written_lo_) flush();
            }
            break;
        }
        case 0x08: case 0x09: case 0x0a: case 0x0b: case 0x0c: case 0x0d: case 0x0e: case 0x0f:
        case 0x24: case 0x25: case 0x36:
            draw = true;
            break;
        default: break;
    }

    if (draw) {
        const u32 cycle = (other_h_ >> 20) & 3;
        // (Pixels past the end of a row land in the next one, another lane's.
        // Copy and fill modes draw the pixel at the scissor's right edge too.)
        const bool serial = (cycle == 1 && (((other_l_ >> 30) & 3) == 1 || ((other_l_ >> 22) & 3) == 1 ||
                                            ((other_l_ >> 18) & 3) == 1)) ||
                            (scissor_xhi_ >> 2) > ci_width_ || (cycle >= 2 && (scissor_xhi_ >> 2) >= ci_width_);
        if (serial) {
            flush();
            run_serial(w, nwords);
            return false;
        }
        u64 lo, hi, zlo, zhi;
        draw_ranges(lo, hi, zlo, zhi);
        written_lo_ = std::min({written_lo_, lo, zlo});
        written_hi_ = std::max({written_hi_, hi, zhi});
        add_sync_range(lo, hi);
        add_sync_range(zlo, zhi);
    }

    starts_.push_back(static_cast<u32>(buf_.size()));
    buf_.insert(buf_.end(), w, w + nwords);
    if (starts_.size() >= 4096) flush();
    return false;
}
