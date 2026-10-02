// The bit-exact RDP's pixel pipeline (src/rdp_exact.cpp, ExactRdp::draw), line
// for line, in two parts as parallel-rdp has it: shade_pixel() - rasterization,
// texturing, combiner, alpha test, everything that doesn't read memory - which
// exact_shade.comp runs for every (tile, primitive) pair at once, and
// memory_pixel() - depth test, blender, coverage, the store - which
// exact_memory.comp runs per pixel for its tile's primitives in drawing order,
// reading and writing RDRAM (R) and its ninth bits (H) as the CPU does. What
// the CPU front end recorded is in D (exact_layout.h). Integer arithmetic
// throughout - 32-bit wrap-around and arithmetic right shifts, as on the CPU.
//
// Derived from parallel-rdp by Themaister (MIT licence; see rdp_exact.cpp).
//
// Before including: the Data buffer D; with EXACT_MEMORY_PASS also R, H, TAIL
// and the uniform block P (mask8, debug_pixel, debug_at).

// ============================================================================
// Helpers

int sext(int v, int bits) { return (v << (32 - bits)) >> (32 - bits); }
int find_msb(int v) { return v < 0 ? 31 : findMSB(v); }
ivec4 sext4(ivec4 v, int bits) { return (v << (32 - bits)) >> (32 - bits); }
// Division rounding towards minus infinity, by b > 0 (internal resolution).
int fdiv(int a, int b) { return b == 1 ? a : a >= 0 ? a / b : -1 - (-(a + 1)) / b; }
int fmod_(int a, int b) { return a - fdiv(a, b) * b; }
ivec3 fdiv3(ivec3 a, int b) { return ivec3(fdiv(a.x, b), fdiv(a.y, b), fdiv(a.z, b)); }
ivec4 fdiv4(ivec4 a, int b) { return ivec4(fdiv(a.x, b), fdiv(a.y, b), fdiv(a.z, b), fdiv(a.w, b)); }
ivec4 special_expand4(ivec4 v) { return sext4(v - 0x80, 9) + 0x80; }

// Perspective division: s16 by s1.15 through the reciprocal table.
const int kPersp[128] = int[](
    0x4000, -252 * 4, 0x3f04, -244 * 4, 0x3e10, -238 * 4, 0x3d22, -230 * 4, 0x3c3c, -223 * 4, 0x3b5d, -218 * 4,
    0x3a83, -210 * 4, 0x39b1, -205 * 4, 0x38e4, -200 * 4, 0x381c, -194 * 4, 0x375a, -189 * 4, 0x369d, -184 * 4,
    0x35e5, -179 * 4, 0x3532, -175 * 4, 0x3483, -170 * 4, 0x33d9, -166 * 4, 0x3333, -162 * 4, 0x3291, -157 * 4,
    0x31f4, -155 * 4, 0x3159, -150 * 4, 0x30c3, -147 * 4, 0x3030, -143 * 4, 0x2fa1, -140 * 4, 0x2f15, -137 * 4,
    0x2e8c, -134 * 4, 0x2e06, -131 * 4, 0x2d83, -128 * 4, 0x2d03, -125 * 4, 0x2c86, -123 * 4, 0x2c0b, -120 * 4,
    0x2b93, -117 * 4, 0x2b1e, -115 * 4, 0x2aab, -113 * 4, 0x2a3a, -110 * 4, 0x29cc, -108 * 4, 0x2960, -106 * 4,
    0x28f6, -104 * 4, 0x288e, -102 * 4, 0x2828, -100 * 4, 0x27c4, -98 * 4, 0x2762, -96 * 4, 0x2702, -94 * 4,
    0x26a4, -92 * 4, 0x2648, -91 * 4, 0x25ed, -89 * 4, 0x2594, -87 * 4, 0x253d, -86 * 4, 0x24e7, -85 * 4,
    0x2492, -83 * 4, 0x243f, -81 * 4, 0x23ee, -80 * 4, 0x239e, -79 * 4, 0x234f, -77 * 4, 0x2302, -76 * 4,
    0x22b6, -74 * 4, 0x226c, -74 * 4, 0x2222, -72 * 4, 0x21da, -71 * 4, 0x2193, -70 * 4, 0x214d, -69 * 4,
    0x2108, -67 * 4, 0x20c5, -67 * 4, 0x2082, -65 * 4, 0x2041, -65 * 4);

// `copy`: the copy pipe has no clamp stage (see rdp_exact.cpp).
void perspective_divide(int s, int t, int w, out int out_s, out int out_t, inout bool overflow, bool copy) {
    const bool w_carry = w <= 0;
    w &= 0x7fff;
    const int shift = min(14 - find_msb(w), 14);
    const int normout = (w << shift) & 0x3fff;
    const int wnorm = normout & 0xff;
    const int ti = normout >> 8;
    const int rcp = ((kPersp[ti * 2 + 1] * wnorm) >> 10) + kPersp[ti * 2];
    int prod0 = s * rcp, prod1 = t * rcp;
    const int temp_mask = ((1 << 30) - 1) & -((1 << 29) >> shift);
    const int oob0 = prod0 & temp_mask, oob1 = prod1 & temp_mask;
    int temp0, temp1;
    if (shift != 14) {
        prod0 = prod0 >> (13 - shift);
        prod1 = prod1 >> (13 - shift);
        temp0 = prod0;
        temp1 = prod1;
    } else {
        temp0 = prod0 << 1;
        temp1 = prod1 << 1;
    }
    bool sat0 = false, sat1 = false;
    if (oob0 != 0 || oob1 != 0) {
        if (oob0 != temp_mask && oob0 != 0) {
            temp0 = (prod0 & (1 << 29)) == 0 ? 0x7fff : -0x8000;
            overflow = true;
            sat0 = true;
        }
        if (oob1 != temp_mask && oob1 != 0) {
            temp1 = (prod1 & (1 << 29)) == 0 ? 0x7fff : -0x8000;
            overflow = true;
            sat1 = true;
        }
    }
    if (w_carry) {
        temp0 = temp1 = 0x7fff;
        overflow = true;
        sat0 = sat1 = true;
    }
    if (copy) {
        out_s = sat0 ? temp0 : sext(temp0 & 0xffff, 16);
        out_t = sat1 ? temp1 : sext(temp1 & 0xffff, 16);
        return;
    }
    out_s = clamp(temp0, -0x10000, 0xffff);
    out_t = clamp(temp1, -0x10000, 0xffff);
}

int clamp_9bit(int c) { return clamp(sext(c - 0x80, 9) + 0x80, 0, 0xff); }
int clamp_z(int z) {
    z -= 1 << 17;
    z = sext(z, 19);
    z += 1 << 17;
    return clamp(z, 0, 0x3ffff);
}

int z_decompress(uint z) {
    const int exponent = int(z >> 11), mantissa = int(z & 0x7ffu);
    const int shift = max(6 - exponent, 0);
    const int base = 0x40000 - (0x40000 >> exponent);
    return (mantissa << shift) + base;
}
uint z_compress(int z) {
    const int inv_z = max(0x3ffff - z, 1);
    const int exponent = clamp(17 - find_msb(inv_z), 0, 7);
    const int shift = max(6 - exponent, 0);
    const int mantissa = (z >> shift) & 0x7ff;
    return uint((exponent << 11) + mantissa);
}
int dz_decompress(int dz) { return 1 << dz; }
int dz_compress(int dz) { return max(find_msb(dz), 0); }
int combine_dz(int dz) { return dz != 0 ? 1 << find_msb(dz) : 0; }

const int kDither[32] = int[](0, 6, 1, 7, 4, 2, 5, 3, 3, 5, 2, 4, 7, 1, 6, 0, // magic square
                              0, 4, 1, 5, 4, 0, 5, 1, 3, 7, 2, 6, 7, 3, 6, 2); // Bayer

uint noise_reseed(uint x, uint y, uint prim) {
    const uint kPrime = 1103515245u;
    uint s0 = x, s1 = y, s2 = prim;
    for (int r = 0; r < 3; ++r) {
        const uint n0 = ((s0 >> 8) ^ s1) * kPrime, n1 = ((s1 >> 8) ^ s2) * kPrime, n2 = ((s2 >> 8) ^ s0) * kPrime;
        s0 = n0;
        s1 = n1;
        s2 = n2;
    }
    return (s0 >> 16) & 0xffffu;
}
int noise_combiner(uint v) { return int(((v & 7u) << 6) | 0x20u); }

void dither_coefficients(int x, int y, int mode_rgb, int mode_alpha, uint noise, out int rgb_dith, out int alpha_dith) {
    const int kSplat = (1 << 0) | (1 << 3) | (1 << 6);
    if (mode_rgb < 2) rgb_dith = kDither[mode_rgb * 16 + (y & 3) * 4 + (x & 3)] * kSplat;
    else if (mode_rgb == 2) rgb_dith = int(noise & 0x1ffu);
    else rgb_dith = 0;
    if (mode_alpha == 3) {
        alpha_dith = 0;
    } else if (mode_alpha == 2) {
        alpha_dith = int(noise & 7u);
    } else {
        alpha_dith = mode_rgb >= 2 ? kDither[(mode_rgb & 1) * 16 + (y & 3) * 4 + (x & 3)] : (rgb_dith & 7);
        if (mode_alpha == 1) alpha_dith = ~alpha_dith & 7;
    }
}

ivec3 rgb_dither(ivec3 rgb, int dith) {
    const ivec3 d = (ivec3(dith) >> ivec3(0, 3, 6)) & 7;
    const ivec3 up = mix((rgb & 0xf8) + 8, ivec3(255), greaterThan(rgb, ivec3(247)));
    const ivec3 replace = (d - (rgb & 7)) >> 31;
    return (rgb + ((up - rgb) & replace)) & 0xff;
}

int blend_coverage(int coverage, int memory_coverage, bool blend_en, int mode) {
    if (mode == 0) return blend_en ? min(7, memory_coverage + coverage) : (coverage - 1) & 7;
    if (mode == 1) return (coverage + memory_coverage) & 7;
    if (mode == 2) return 7;
    return memory_coverage;
}

int special_expand(int v) { return sext(v - 0x80, 9) + 0x80; }
int combiner_equation(int a, int b, int c, int d) {
    c = sext(c, 9);
    a = special_expand(a);
    b = special_expand(b);
    d = special_expand(d);
    int color = (a - b) * c;
    color += 0x80;
    return (color >> 8) + d;
}

int blender_divide(int sum, int n) {
    // A non-restoring division of n (11 bits) by sum (4 bits) + 1.
    const int d = sum & 0xf, invd = ~d & 0xf;
    int res = 0;
    int partial = (invd + (n >> 8) + 1) & 7;
    for (int k = 0; k < 8; ++k) {
        const int nbit = (n >> (7 - k)) & 1;
        const int s = (res & (0x100 >> k)) != 0 ? invd + (partial << 1) + nbit + 1 : d + (partial << 1) + nbit;
        partial = s & 7;
        if ((s & 0x10) != 0) res |= 1 << (7 - k);
    }
    return res;
}

// ============================================================================
// The primitive being drawn

int g_S = 1;        // internal resolution of the pass (1: native)
// fdiv(a, g_S << shift): a shift at native resolution.
int fdiv_s(int a, int shift) { return g_S == 1 ? a >> shift : fdiv(a, g_S << shift); }
uint g_po, g_so;    // word offsets of the primitive and its state
uint g_f0, g_f1;    // state flags
uint g_tmem;        // word offset of the TMEM snapshot
uint g_span, g_next; // word offsets of this scanline's span and the next one's
bool g_flip;

#ifdef EXACT_SHARED
// exact_shade.comp: a workgroup's (tile, primitive) pair - the primitive,
// its state and TMEM, and the spans of the tile's rows and the one below -
// copied once into shared memory; g_span and g_next index s_span.
shared uint s_prim[EXACT_PRIM_WORDS];
shared uint s_state[EXACT_STATE_WORDS];
shared uint s_span[(EXACT_TILE_SIZE + 1u) * EXACT_SPAN_WORDS];
shared uint s_tmem[EXACT_TMEM_WORDS];
uint SW(uint o) { return s_state[o]; }
uint PW(uint o) { return s_prim[o]; }
uint SPW(uint o) { return s_span[g_span + o]; }
uint NXW(uint o) { return s_span[g_next + o]; }
uint TMW(uint i) { return s_tmem[i]; }
#else
uint SW(uint o) { return D[g_so + o]; }
uint PW(uint o) { return D[g_po + o]; }
uint SPW(uint o) { return D[g_span + o]; }
uint NXW(uint o) { return D[g_next + o]; }
uint TMW(uint i) { return D[g_tmem + i]; }
#endif

// Shader variants (EXACT_VARIANT, see exact_shade.comp): the state bits and
// cycle types a variant handles; anything else is compiled out.
#ifndef EXACT_VARIANT
#define EXACT_VARIANT 0
#endif
#if EXACT_VARIANT == 1 // 1-cycle, no textures
#define VARIANT_FLAGS (~(EF_USES_TEXEL0 | EF_USES_TEXEL1 | EF_USES_PIPELINED_TEXEL1 | EF_USES_LOD | EF_NEED_NOISE_DUAL))
#define VARIANT_CYCLE 0u
#elif EXACT_VARIANT == 2 // 1-cycle
#define VARIANT_FLAGS (~(EF_USES_TEXEL1 | EF_NEED_NOISE_DUAL))
#define VARIANT_CYCLE 0u
#elif EXACT_VARIANT == 3 // 2-cycle
#define VARIANT_FLAGS (~EF_USES_PIPELINED_TEXEL1)
#define VARIANT_CYCLE 1u
#elif EXACT_VARIANT == 4 // copy
#define VARIANT_FLAGS (EF_PERSP | EF_TLUT | EF_ALPHA_TEST)
#define VARIANT_CYCLE 2u
#elif EXACT_VARIANT == 5 // fill
#define VARIANT_FLAGS 0u
#define VARIANT_CYCLE 3u
#else
#define VARIANT_FLAGS 0xffffffffu
#endif
#ifdef VARIANT_CYCLE
uint cycle_type() { return VARIANT_CYCLE; }
#else
uint cycle_type() { return g_f0 & 3u; }
#endif
bool F(uint b) { return (g_f0 & b & VARIANT_FLAGS) != 0u; }
int SI(uint o) { return int(SW(o)); }
int PI(uint o) { return int(PW(o)); }
int SPAN(uint o) { return int(SPW(o)); }
int NEXT(uint o) { return int(NXW(o)); }

uint fb_fmt() { return (g_f1 >> 24) & 7u; }
#define FB_I4 0u
#define FB_I8 1u
#define FB_RGBA5551 2u
#define FB_IA88 3u
#define FB_RGBA8888 4u

#ifdef EXACT_MEMORY_PASS
// ---- Memory
// Addresses are RDRAM's; at internal resolution (g_S > 1) R and H are the
// high-resolution copy, where the pixel's sample g_sample of an RDRAM byte
// (or halfword's ninth bits) is (exact_layout.h).

uint g_sample = 0u;

uint mem_byte(uint a) {
    if (g_S == 1) return a;
    const uint slot = PT[a >> EXACT_PAGE_SHIFT];
    const uint page = 1u << EXACT_PAGE_SHIFT;
    return (slot * uint(g_S * g_S) + g_sample) * page + (a & (page - 1u));
}
uint hidden_index(uint h) {
    if (g_S == 1) return h;
    const uint slot = PT[(h << 1) >> EXACT_PAGE_SHIFT];
    const uint page = 1u << (EXACT_PAGE_SHIFT - 1u);
    return (slot * uint(g_S * g_S) + g_sample) * page + (h & (page - 1u));
}

uint rd8(uint a) {
    a = mem_byte(a & P.mask8);
    return (R[a >> 2] >> ((a & 3u) * 8u)) & 0xffu;
}
void wr8(uint a, uint v) {
    a = mem_byte(a & P.mask8);
    const uint sh = (a & 3u) * 8u;
    atomicAnd(R[a >> 2], ~(0xffu << sh));
    atomicOr(R[a >> 2], (v & 0xffu) << sh);
}
uint rd16(uint h) { // halfword index (already masked)
    const uint b = mem_byte(h << 1);
    const uint w = (R[b >> 2] >> (((b >> 1) & 1u) * 16u)) & 0xffffu;
    return ((w & 0xffu) << 8) | (w >> 8);
}
void wr16(uint h, uint v) {
    const uint b = mem_byte(h << 1);
    const uint sh = ((b >> 1) & 1u) * 16u;
    const uint sw = ((v & 0xffu) << 8) | ((v >> 8) & 0xffu);
    atomicAnd(R[b >> 2], ~(0xffffu << sh));
    atomicOr(R[b >> 2], sw << sh);
}
uint rd32(uint i) { return R[mem_byte(i << 2) >> 2]; } // word index (already masked)
void wr32(uint i, uint v) { R[mem_byte(i << 2) >> 2] = v; }
uint hidden_get(uint h) {
    h = hidden_index(h);
    return (H[h >> 2] >> ((h & 3u) * 8u)) & 0xffu;
}
void hidden_set(uint h, uint bits) {
    h = hidden_index(h);
    const uint sh = (h & 3u) * 8u;
    atomicAnd(H[h >> 2], ~(0xffu << sh));
    atomicOr(H[h >> 2], (bits & 0xffu) << sh);
}


// A pixel's colour as read from the colour image; returns its index.
uint read_colour(int x, int y, out ivec4 col) {
    uint cidx = SW(ES_FB_INDEX) + SW(ES_FB_WIDTH) * uint(y) + uint(x);
    const uint fmt = fb_fmt();
    if (fmt == FB_I4 || fmt == FB_I8) {
        cidx &= P.mask8;
        const int c = int(rd8(cidx));
        col = ivec4(c, c, c, int(hidden_get(cidx >> 1)));
    } else if (fmt == FB_RGBA5551) {
        cidx &= P.mask8 >> 1;
        const uint wv = rd16(cidx);
        col = ivec4(int((wv >> 8) & 0xf8u), int((wv >> 3) & 0xf8u), int((wv << 2) & 0xf8u),
                    int((hidden_get(cidx) << 5) | ((wv & 1u) << 7)));
    } else if (fmt == FB_IA88) {
        cidx &= P.mask8 >> 1;
        const uint wv = rd16(cidx);
        col = ivec4(int(wv >> 8), int(wv >> 8), int(wv >> 8), int(wv & 0xffu));
    } else {
        cidx &= P.mask8 >> 2;
        const uint w = rd32(cidx);
        col = ivec4(int(w & 0xffu), int((w >> 8) & 0xffu), int((w >> 16) & 0xffu), int(w >> 24));
    }
    return cidx;
}
// The colour as the blender sees it (alpha = memory coverage << 5).
ivec4 decode_memory(ivec4 col) {
    int cvg_bits = F(EF_IMAGE_READ) ? (col.a & 0xe0) : 0xe0;
    ivec4 mem;
    const uint fmt = fb_fmt();
    if (fmt == FB_I4) {
        mem.rgb = ivec3(0);
        cvg_bits = 0xe0;
    } else if (fmt == FB_I8) {
        mem.rgb = ivec3(col.r);
        cvg_bits = 0xe0;
    } else if (fmt == FB_RGBA5551) {
        mem.rgb = col.rgb & 0xf8;
    } else if (fmt == FB_IA88) {
        mem.rgb = ivec3(col.r);
    } else {
        mem.rgb = col.rgb;
    }
    mem.a = cvg_bits;
    return mem;
}

#endif

// ---- Coverage: 8 samples, two per sub-scanline (see rdp_exact.cpp).
uint compute_coverage(int x) {
    // sample 2k at offset (0, 2, 0, 2)[k], sample 2k + 1 at (4, 6, 4, 6)[k] of sub-scanline k
    const ivec4 xl = ivec4(SPAN(EX_XLEFT), SPAN(EX_XLEFT + 1u), SPAN(EX_XLEFT + 2u), SPAN(EX_XLEFT + 3u));
    const ivec4 xr = ivec4(SPAN(EX_XRIGHT), SPAN(EX_XRIGHT + 1u), SPAN(EX_XRIGHT + 2u), SPAN(EX_XRIGHT + 3u));
    const int base = (x << 3) & 0xffff;
    const ivec4 xe = (base + ivec4(0, 2, 0, 2)) & 0xffff, xo = (base + ivec4(4, 6, 4, 6)) & 0xffff;
    const uvec4 even = uvec4(greaterThanEqual(xe, xl)) & uvec4(lessThan(xe, xr));
    const uvec4 odd = uvec4(greaterThanEqual(xo, xl)) & uvec4(lessThan(xo, xr));
    const uvec4 bits = (even << uvec4(0u, 2u, 4u, 6u)) | (odd << uvec4(1u, 3u, 5u, 7u));
    return bits.x | bits.y | bits.z | bits.w;
}

// ============================================================================
// Texture unit

struct Tile {
    int slo, shi, tlo, thi;
    uint offset, stride;
    uint fmt, size, palette, mask_s, shift_s, mask_t, shift_t, flags;
};
#define TILE_CLAMP_S 1u
#define TILE_MIRROR_S 2u
#define TILE_CLAMP_T 4u
#define TILE_MIRROR_T 8u
#define TF_RGBA 0u
#define TF_YUV 1u
#define TF_CI 2u
#define TF_IA 3u
#define TF_I 4u

Tile load_tile(uint t) {
    const uint o = ES_TILES + EXACT_TILE_WORDS * (t & 7u);
    Tile r;
    r.slo = int(SW(o) & 0xffffu);
    r.shi = int(SW(o) >> 16);
    r.tlo = int(SW(o + 1u) & 0xffffu);
    r.thi = int(SW(o + 1u) >> 16);
    r.offset = SW(o + 2u) & 0xffffu;
    r.stride = SW(o + 2u) >> 16;
    const uint q = SW(o + 3u);
    r.fmt = q & 7u;
    r.size = (q >> 3) & 3u;
    r.palette = (q >> 5) & 15u;
    r.mask_s = (q >> 9) & 15u;
    r.shift_s = (q >> 13) & 15u;
    r.mask_t = (q >> 17) & 15u;
    r.shift_t = (q >> 21) & 15u;
    r.flags = (q >> 25) & 15u;
    return r;
}

uint tmem_half(uint h) {
    h &= 0x7ffu;
    return (TMW(h >> 1) >> ((h & 1u) * 16u)) & 0xffffu;
}
uint tmem_byte(uint b) {
    const uint h = tmem_half(b >> 1);
    return (b & 1u) != 0u ? h & 0xffu : h >> 8;
}

ivec4 convert_rgba16(uint word) {
    const uvec3 c = (uvec3(word) >> uvec3(11u, 6u, 1u)) & 31u;
    return ivec4(ivec3((c << 3) | (c >> 2)), (word & 1u) != 0u ? 0xff : 0);
}
ivec4 convert_ia16(uint word) {
    const int i = int(word >> 8);
    return ivec4(i, i, i, int(word & 0xffu));
}

uint texel_byte_offset(Tile tile, int t, uint x_bytes, uint wrap) {
    uint off = tile.offset + tile.stride * uint(t);
    off += x_bytes;
    off &= wrap;
    return off ^ ((uint(t) & 1u) << 2);
}
ivec4 texel_rgba4(Tile tile, int s, int t) {
    const uint b = texel_byte_offset(tile, t, uint(s) >> 1, 0xfffu);
    uint w = (tmem_byte(b) >> ((uint(~s) & 1u) * 4u)) & 0xfu;
    w |= w << 4;
    return ivec4(int(w));
}
ivec4 texel_ia4(Tile tile, int s, int t) {
    const uint b = texel_byte_offset(tile, t, uint(s) >> 1, 0xfffu);
    const uint w = (tmem_byte(b) >> ((uint(~s) & 1u) * 4u)) & 0xfu;
    uint i = w & 0xeu;
    i = (i << 4) | (i << 1) | (i >> 2);
    return ivec4(int(i), int(i), int(i), int((w & 1u) * 0xffu));
}
ivec4 texel_ci4(Tile tile, int s, int t, uint pal) {
    const uint b = texel_byte_offset(tile, t, uint(s) >> 1, 0xfffu);
    uint w = (tmem_byte(b) >> ((uint(~s) & 1u) * 4u)) & 0xfu;
    w |= pal << 4;
    return ivec4(int(w));
}
uint tlut_entry(uint entry, uint lut_offset, bool upper) {
    const uint idx = ((entry << 2) + lut_offset) ^ (upper ? 3u : 0u);
    return tmem_half(0x400u | (idx & 0x3ffu));
}
ivec4 texel_ci4_tlut(Tile tile, int s, int t, uint pal, uint lut_offset, bool upper, bool ia) {
    const uint b = texel_byte_offset(tile, t, uint(s) >> 1, 0x7ffu);
    uint w = (tmem_byte(b) >> ((uint(~s) & 1u) * 4u)) & 0xfu;
    w |= pal << 4;
    const uint word = tlut_entry(w, lut_offset, upper);
    return ia ? convert_ia16(word) : convert_rgba16(word);
}
ivec4 texel_ci8_tlut(Tile tile, int s, int t, uint lut_offset, bool upper, bool ia) {
    const uint b = texel_byte_offset(tile, t, uint(s), 0x7ffu);
    const uint word = tlut_entry(tmem_byte(b), lut_offset, upper);
    return ia ? convert_ia16(word) : convert_rgba16(word);
}
uint texel_half(Tile tile, int s, int t, uint wrap) {
    uint off = tile.offset + tile.stride * uint(t);
    off += uint(s) * 2u;
    off &= wrap;
    return tmem_half((off >> 1) ^ ((uint(t) & 1u) << 1));
}
ivec4 texel_ci32(Tile tile, int s, int t) {
    const uint w = texel_half(tile, s, t, 0xfffu);
    const int hi = int(w >> 8), lo = int(w & 0xffu);
    return ivec4(hi, lo, hi, lo);
}
ivec4 texel_ci32_tlut(Tile tile, int s, int t, uint lut_offset, bool upper, bool ia) {
    const uint w = texel_half(tile, s, t, 0x7ffu);
    const uint word = tlut_entry(w >> 8, lut_offset, upper);
    return ia ? convert_ia16(word) : convert_rgba16(word);
}
ivec4 texel_rgba8(Tile tile, int s, int t) {
    const uint b = texel_byte_offset(tile, t, uint(s), 0xfffu);
    return ivec4(int(tmem_byte(b)));
}
ivec4 texel_ia8(Tile tile, int s, int t) {
    const uint b = texel_byte_offset(tile, t, uint(s), 0xfffu);
    const uint w = tmem_byte(b);
    uint i = w >> 4, a = w & 0xfu;
    a |= a << 4;
    i |= i << 4;
    return ivec4(int(i), int(i), int(i), int(a));
}
ivec4 texel_yuv16(Tile tile, int s, int t, uint chroma_x) {
    const uint base = tile.offset + tile.stride * uint(t);
    const uint luma_b = ((base + uint(s)) & 0x7ffu) ^ ((uint(t) & 1u) << 2);
    const uint chroma_b = (base + chroma_x * 2u) & 0x7ffu;
    const uint chroma_h = (chroma_b >> 1) ^ ((uint(t) & 1u) << 1);
    const int luma = int(tmem_byte(luma_b | 0x800u));
    const uint chroma = tmem_half(chroma_h);
    return ivec4(int((chroma >> 8) & 0xffu) - 0x80, int(chroma & 0xffu) - 0x80, luma, luma);
}
ivec4 texel_rgba32(Tile tile, int s, int t) {
    uint off = tile.offset + tile.stride * uint(t);
    off += uint(s) * 2u;
    off &= 0x7ffu;
    const uint h = (off >> 1) ^ ((uint(t) & 1u) << 1);
    const uint lo = tmem_half(h), hi = tmem_half(h | 0x400u);
    return ivec4(int(lo >> 8), int(lo & 0xffu), int(hi >> 8), int(hi & 0xffu));
}

int mask_s(Tile tile, int s) {
    if (tile.mask_s != 0u) {
        const int mask = 1 << tile.mask_s;
        if ((tile.flags & TILE_MIRROR_S) != 0u) s ^= max((s & mask) - 1, 0);
        s &= mask - 1;
    }
    return s;
}
int mask_t(Tile tile, int t) {
    if (tile.mask_t != 0u) {
        const int mask = 1 << tile.mask_t;
        if ((tile.flags & TILE_MIRROR_T) != 0u) t ^= max((t & mask) - 1, 0);
        t &= mask - 1;
    }
    return t;
}

int clamp_and_shift(bool clamp_bit, int coord, int lo, int hi, uint shift) {
    coord = clamp(coord, -0x8000, 0x7fff);
    if (shift < 11u) coord >>= int(shift);
    else coord = (coord << (32u - shift)) >> 16;
    if (clamp_bit) {
        if ((coord >> 3) >= hi) coord = (((hi >> 2) - (lo >> 2)) & 0x3ff) << 5;
        else coord = max(coord - (lo << 3), 0);
    } else {
        coord -= lo << 3;
    }
    return coord;
}
int shift_coord(int coord, int lo, uint shift) {
    coord = clamp(coord, -0x8000, 0x7fff);
    if (shift < 11u) coord >>= int(shift);
    else coord = (coord << (32u - shift)) >> 16;
    return coord - (lo << 3);
}

ivec4 texture_convert_factors(ivec4 inp) {
    const int f0 = SI(ES_FACTORS), f1 = SI(ES_FACTORS + 1u), f2 = SI(ES_FACTORS + 2u), f3 = SI(ES_FACTORS + 3u);
    const int r0 = sext(inp.x, 9), g0 = sext(inp.y, 9), b0 = sext(inp.z, 9);
    const int r = b0 + ((f0 * g0 + 0x80) >> 8);
    const int g = b0 + ((f1 * r0 + f2 * g0 + 0x80) >> 8);
    const int b = b0 + ((f3 * r0 + 0x80) >> 8);
    return ivec4(r, g, b, b0);
}

int bilinear_3tap1(int t00, int t10, int t01, int t11, int fs, int ft) {
    const int sum = fs + ft;
    const int base = sum >= 32 ? t11 : t00;
    const int f0 = sum >= 32 ? 32 - ft : fs;
    const int f1 = sum >= 32 ? 32 - fs : ft;
    int acc = (t10 - base) * f0 + (t01 - base) * f1;
    acc += 0x10;
    acc >>= 5;
    return acc + base;
}

ivec4 fetch_tlut(Tile tile, int ss, int tt, uint off, bool upper_lut, bool tlut_type) {
    if (tile.size == 0u) return texel_ci4_tlut(tile, ss, tt, tile.palette, off, upper_lut, tlut_type);
    if (tile.size == 1u) return texel_ci8_tlut(tile, ss, tt, off, upper_lut, tlut_type);
    return texel_ci32_tlut(tile, ss, tt, off, upper_lut, tlut_type);
}
ivec4 fetch_texel(Tile tile, int ss, int tt) {
    switch (tile.fmt) {
        case TF_RGBA:
            if (tile.size == 0u) return texel_rgba4(tile, ss, tt);
            if (tile.size == 1u) return texel_rgba8(tile, ss, tt);
            if (tile.size == 2u) return convert_rgba16(texel_half(tile, ss, tt, 0xfffu));
            return texel_rgba32(tile, ss, tt);
        case TF_CI:
            if (tile.size == 0u) return texel_ci4(tile, ss, tt, tile.palette);
            if (tile.size == 1u) return texel_rgba8(tile, ss, tt);
            return texel_ci32(tile, ss, tt);
        case TF_IA:
            if (tile.size == 0u) return texel_ia4(tile, ss, tt);
            if (tile.size == 1u) return texel_ia8(tile, ss, tt);
            if (tile.size == 2u) return convert_ia16(texel_half(tile, ss, tt, 0xfffu));
            return texel_ci32(tile, ss, tt);
        case TF_I:
            if (tile.size == 0u) return texel_rgba4(tile, ss, tt);
            if (tile.size == 1u) return texel_rgba8(tile, ss, tt);
            return texel_ci32(tile, ss, tt);
        default: return ivec4(0); // formats 5-7 read nothing
    }
}

ivec4 sample_texture(uint tile_index, int s_in, int t_in, bool convert_one, bool bilerp, ivec4 prev) {
    const Tile tile = load_tile(tile_index);
    const bool tlut = F(EF_TLUT), tlut_type = F(EF_TLUT_TYPE), sample_quad = F(EF_SAMPLE_QUAD);
    const bool mid_texel_state = F(EF_MID_TEXEL);
    int s = clamp_and_shift((tile.flags & TILE_CLAMP_S) != 0u, s_in, tile.slo, tile.shi, tile.shift_s);
    int t = clamp_and_shift((tile.flags & TILE_CLAMP_T) != 0u, t_in, tile.tlo, tile.thi, tile.shift_t);

    int fs = 0, ft = 0;
    if (sample_quad || tlut) {
        fs = s & 31;
        ft = t & 31;
    }
    int sum_frac = fs + ft;
    s >>= 5;
    t >>= 5;

    int s0 = mask_s(tile, s), t0 = mask_t(tile, t);
    int s1 = mask_s(tile, s + 1), t1 = mask_t(tile, t + 1);
    const int tdiff = max(t1 - t0, -255);
    t1 = (t0 & 0xff) + tdiff;
    t0 &= 0xff;

    ivec4 tb = ivec4(0), t10 = ivec4(0), t01 = ivec4(0), t11 = ivec4(0);
    const bool mid_texel = mid_texel_state && bilerp && fs == 0x10 && ft == 0x10;
    const bool upper_lut = sum_frac >= 0x20;
    if (mid_texel) sum_frac = 0;

    const bool yuv = tile.fmt == TF_YUV;
    int bs = sum_frac >= 0x20 ? s1 : s0, bt = sum_frac >= 0x20 ? t1 : t0;
    const int chroma_frac = ((s0 & 1) << 4) | (fs >> 1);

    // The texels read - the base one, (s + 1, t), (s, t + 1), (s + 1, t + 1) -
    // fetched in a loop with one call site each, which keeps the shader small.
    int ntaps = 0;
    int mode = 0; // 0 none, 1 TLUT, 2 YUV, 3 any other format
    uint lut0 = 0u, cx0 = 0u, cx1 = 0u;
    if (tlut) {
        if (!sample_quad) {
            bs = s0;
            bt = t0;
            s1 = s0;
            t1 = t0;
        }
        if (tile.fmt == TF_RGBA || tile.fmt == TF_CI || tile.fmt == TF_IA || tile.fmt == TF_I) {
            mode = 1;
            ntaps = mid_texel ? 4 : bilerp ? 3 : 1;
        }
        lut0 = sum_frac >= 0x20 ? 3u : 0u;
    } else if (yuv) {
        mode = 2;
        ntaps = sample_quad ? 4 : 1;
        cx0 = uint(s0 >> 1);
        cx1 = uint((s1 + (s1 - s0)) >> 1);
        bs = s0;
        bt = t0;
    } else {
        mode = 3;
        ntaps = mid_texel ? 4 : sample_quad ? 3 : 1;
    }
    for (int i = 0; i < ntaps; ++i) {
        const int ts = i == 0 ? bs : (i & 1) != 0 ? s1 : s0;
        const int tt = i == 0 ? bt : i == 1 ? t0 : t1;
        ivec4 v;
        if (mode == 1) v = fetch_tlut(tile, ts, tt, i == 0 ? lut0 : uint(i), upper_lut, tlut_type);
        else if (mode == 2) v = texel_yuv16(tile, ts, tt, (i & 1) != 0 ? cx1 : cx0);
        else v = fetch_texel(tile, ts, tt);
        if (i == 0) tb = v;
        else if (i == 1) t10 = v;
        else if (i == 2) t01 = v;
        else t11 = v;
    }

    ivec4 acc;
    if (convert_one) {
        const ivec4 p = sext4(prev, 9);
        if (sample_quad) {
            const bool mid_rg = yuv ? (mid_texel_state && chroma_frac == 0x10 && ft == 0x10) : mid_texel;
            const bool mid_ba = mid_texel;
            const bool upper_ba = sum_frac >= 32;
            const bool upper_rg = yuv ? ((chroma_frac + ft) >= 32 && !mid_rg) : upper_ba;
            // weights per channel: R, G use the RG ones, B, A the BA ones
            const ivec4 f0 = ivec4(upper_rg ? p.y : p.x, upper_rg ? p.y : p.x, upper_ba ? p.y : p.x, upper_ba ? p.y : p.x);
            const ivec4 f1 = ivec4(upper_rg ? p.x : p.y, upper_rg ? p.x : p.y, upper_ba ? p.x : p.y, upper_ba ? p.x : p.y);
            const bvec4 mid = bvec4(mid_rg, mid_rg, mid_ba, mid_ba);
            const bvec4 from11 = bvec4(upper_rg && yuv, upper_rg && yuv, upper_ba && yuv, upper_ba && yuv);
            const ivec4 base = mix(tb, t11, from11);
            const ivec4 conv_mid = f0 * (t01 - t11) + f1 * (t10 - t11) + ((tb - t11) << 6) + 0x80;
            const ivec4 conv_lin = f0 * (t10 - base) + f1 * (t01 - base) + 0x80;
            acc = (mix(conv_lin, conv_mid, mid) >> 8) + p.z;
        } else {
            acc = ivec4(p.z);
        }
    } else if (yuv) {
        if (sample_quad) {
            const ivec4 fsv = ivec4(chroma_frac, chroma_frac, fs, fs);
            if (bilerp) {
                const bool mid_chroma = mid_texel_state && chroma_frac == 0x10 && ft == 0x10;
                const bvec4 mid = bvec4(mid_chroma, mid_chroma, mid_texel, mid_texel);
                // bilinear_3tap1 per channel
                const bvec4 up = greaterThanEqual(fsv + ft, ivec4(32));
                const ivec4 base = mix(tb, t11, up);
                const ivec4 w0 = mix(fsv, 32 - ivec4(ft), up);
                const ivec4 w1 = mix(ivec4(ft), 32 - fsv, up);
                const ivec4 lin = ((((t10 - base) * w0 + (t01 - base) * w1) + 0x10) >> 5) + base;
                const ivec4 avg = (tb + t10 + t11 + t01 + 2) >> 2;
                acc = mix(lin, avg, mid);
            } else {
                acc = mix(tb, t11, greaterThanEqual(fsv + ft, ivec4(32)));
            }
        } else {
            acc = tb;
        }
    } else if (mid_texel) {
        acc = (tb + t01 + t10 + t11 + 2) >> 2;
    } else if (bilerp && (sample_quad || tlut)) {
        const int f0 = sum_frac >= 32 ? 32 - ft : fs;
        const int f1 = sum_frac >= 32 ? 32 - fs : ft;
        acc = ((((t10 - tb) * f0 + (t01 - tb) * f1) + 0x10) >> 5) + tb;
    } else {
        acc = tb;
    }

    if (!bilerp && !convert_one) acc = texture_convert_factors(acc);
    return acc;
}

void compute_lod(inout uint tile0, inout uint tile1, inout int lod_frac, uint max_level, int min_lod, int s, int t,
                 int sdx, int tdx, int sdy, int tdy, bool persp_overflow) {
    const bool tex_lod_en = F(EF_TEX_LOD), sharpen = F(EF_SHARPEN), detail = F(EF_DETAIL);
    bool magnify = false, distant = false;
    uint tile_offset = 0u;
    if (persp_overflow) {
        distant = true;
        lod_frac = 0xff;
    } else {
        int dx0 = sdx - s, dx1 = tdx - t, dy0 = sdy - s, dy1 = tdy - t;
        dx0 ^= dx0 >> 31;
        dx1 ^= dx1 >> 31;
        dy0 ^= dy0 >> 31;
        dy1 ^= dy1 >> 31;
        const int max_d = max(max(dx0, dy0), max(dx1, dy1));
        if (max_d >= 0x4000) {
            distant = true;
            lod_frac = 0xff;
            tile_offset = max_level;
        } else if (max_d < 32) {
            distant = max_level == 0u;
            magnify = true;
            if (!sharpen && !detail) lod_frac = distant ? 0xff : 0;
            else lod_frac = (max(min_lod, max_d) << 3) + (sharpen ? -0x100 : 0);
        } else {
            const int mip_base = max(find_msb(max_d >> 5), 0);
            distant = mip_base >= int(max_level);
            if (distant && !sharpen && !detail) {
                lod_frac = 0xff;
            } else {
                lod_frac = ((max_d << 3) >> mip_base) & 0xff;
                tile_offset = uint(mip_base);
            }
        }
    }
    if (tex_lod_en) {
        if (distant) tile_offset = max_level;
        if (!detail) {
            tile0 = (tile0 + tile_offset) & 7u;
            tile1 = (distant || (!sharpen && magnify)) ? tile0 : ((tile0 + 1u) & 7u);
        } else {
            tile1 = (tile0 + tile_offset + ((distant || magnify) ? 1u : 2u)) & 7u;
            tile0 = (tile0 + tile_offset + (magnify ? 0u : 1u)) & 7u;
        }
    }
}

// ============================================================================
// A pixel's inputs: shade, texels, LOD fraction, depth (pixel_inputs in rdp_exact.cpp)

struct PixIn {
    ivec4 shade, texel0, texel1;
    int lod_frac, z;
};

const uint kC[3] = uint[](0u, 1u, 3u); // s, t, w in stzw

PixIn pixel_inputs(int x, uint coverage) {
    PixIn pin;
    const bool multi = cycle_type() == 1u;
    const bool persp = F(EF_PERSP), uses_lod = F(EF_USES_LOD);
    const int base_x = SPAN(EX_BASE_X);
    const int dx = x - base_x;
    const int idir = g_flip ? 1 : -1;
    const int first = coverage != 0u ? findLSB(coverage) : 0;
    const int yoff = first >> 1, xoff = coverage != 0u ? ((first & 1) << 1) + (yoff & 1) : 0;

    {
        const ivec4 ddx = ivec4(PI(EP_DRGBA_DX), PI(EP_DRGBA_DX + 1u), PI(EP_DRGBA_DX + 2u), PI(EP_DRGBA_DX + 3u));
        const ivec4 ddy = ivec4(PI(EP_DRGBA_DY), PI(EP_DRGBA_DY + 1u), PI(EP_DRGBA_DY + 2u), PI(EP_DRGBA_DY + 3u));
        // (internal resolution: steps of 1/S of the derivative between native pixels)
        const ivec4 c = ivec4(SPAN(EX_RGBA), SPAN(EX_RGBA + 1u), SPAN(EX_RGBA + 2u), SPAN(EX_RGBA + 3u)) + fdiv4(ddx & ~0x1f, g_S) * dx;
        const ivec4 sv = ((c >> 14) * (4 * g_S)) + xoff * (ddx >> 14) + yoff * (ddy >> 14);
        const ivec4 sn = ivec4(fdiv_s(sv.x, 4), fdiv_s(sv.y, 4), fdiv_s(sv.z, 4), fdiv_s(sv.w, 4));
        pin.shade = clamp(sext4(sn - 0x80, 9) + 0x80, 0, 0xff); // clamp_9bit
    }

    {
        const int dzdx = PI(EP_DSTZW_DX + 2u), dzdy = PI(EP_DSTZW_DY + 2u);
        const int zz = SPAN(EX_STZW + 2u) + dzdx * fdiv(dx, g_S) + fdiv(dzdx, g_S) * fmod_(dx, g_S);
        int sz = zz >> 10;
        sz = sz * (4 * g_S);
        sz = sz + xoff * (dzdx >> 10) + yoff * (dzdy >> 10);
        sz = fdiv_s(sz, 5);
        pin.z = clamp_z(sz);
    }

    // S, T, W at the pixel and where the LOD and a pipelined TEXEL1 look:
    //   0  the pixel
    //   1  2-cycle: a pixel across; 1-cycle: the next pixel along the span
    //   2  2-cycle: a line down; 1-cycle: the one after (or before, centred)
    //   3  pipelined TEXEL1: the next pixel, or the next line's start
    // each through the perspective division (one call site).
    const ivec3 dstw = ivec3(PI(EP_DSTZW_DX) & ~0x1f, PI(EP_DSTZW_DX + 1u) & ~0x1f, PI(EP_DSTZW_DX + 3u) & ~0x1f);
    const ivec3 stw = ivec3(SPAN(EX_STZW), SPAN(EX_STZW + 1u), SPAN(EX_STZW + 3u)) + fdiv3(dstw, g_S) * dx;
    bool centred = false;
    if (uses_lod && !multi) {
        // 1-cycle mode: the pipelined pair along the span (see rdp_exact.cpp).
        const int span_last_x = g_flip ? SPAN(EX_END_X) : SPAN(EX_START_X);
        const bool all_valid = SPAN(EX_XLEFT) != 0xffff && SPAN(EX_XLEFT + 1u) != 0xffff &&
                               SPAN(EX_XLEFT + 2u) != 0xffff && SPAN(EX_XLEFT + 3u) != 0xffff;
        centred = x + idir * g_S == span_last_x && SPAN(EX_LODLENGTH) >= 8 * g_S && all_valid;
    }
    const bool pipelined = F(EF_USES_PIPELINED_TEXEL1);
    bool next_line = false;
    if (pipelined) {
        const bool long_span = SPAN(EX_LODLENGTH) >= 8 * g_S;
        const bool end_span = x == (g_flip ? SPAN(EX_END_X) : SPAN(EX_START_X));
        next_line = end_span && long_span && NEXT(EX_VALID) != 0;
    }
    ivec2 c0 = ivec2(0), c1 = ivec2(0), c2 = ivec2(0), c3 = ivec2(0);
    bool ov0 = false, ov12 = false;
    for (int r = 0; r < 4; ++r) {
        if ((r == 1 || r == 2) && !uses_lod) continue;
        if (r == 3 && !pipelined) break;
        ivec3 v;
        if (r == 0) {
            v = stw;
        } else if (r == 1) {
            v = multi ? stw + idir * dstw : stw + dstw * idir;
        } else if (r == 2) {
            v = multi ? stw + ivec3(PI(EP_DSTZW_DY) & ~0x7fff, PI(EP_DSTZW_DY + 1u) & ~0x7fff, PI(EP_DSTZW_DY + 3u) & ~0x7fff)
                      : stw + dstw * (centred ? -idir : 2 * idir);
        } else if (next_line) {
            v = ivec3(NEXT(EX_STZW), NEXT(EX_STZW + 1u), NEXT(EX_STZW + 3u));
        } else {
            v = ivec3(SPAN(EX_STZW), SPAN(EX_STZW + 1u), SPAN(EX_STZW + 3u)) + fdiv3(dstw, g_S) * (dx + idir * g_S);
        }
        v >>= 16;
        ivec2 o = v.xy;
        bool ov = false;
        if (persp) perspective_divide(v.x, v.y, v.z, o.x, o.y, ov, false);
        if (r == 0) {
            c0 = o;
            ov0 = ov;
        } else if (r == 1) {
            c1 = o;
            ov12 = ov12 || ov;
        } else if (r == 2) {
            c2 = o;
            ov12 = ov12 || ov;
        } else {
            c3 = o;
        }
    }
    const int st_s = c0.x, st_t = c0.y;

    const uint pflags = PW(EP_FLAGS);
    const uint setup_tile = (pflags >> 1) & 7u, max_level = (pflags >> 4) & 7u;
    uint tile0 = setup_tile, tile1 = (tile0 + 1u) & 7u;
    pin.lod_frac = 0;
    if (uses_lod && !multi) {
        compute_lod(tile0, tile1, pin.lod_frac, max_level, SI(ES_MIN_LOD), c1.x, c1.y, c2.x, c2.y, c1.x, c1.y, ov12);
    } else if (uses_lod) {
        compute_lod(tile0, tile1, pin.lod_frac, max_level, SI(ES_MIN_LOD), st_s, st_t, c1.x, c1.y, c2.x, c2.y,
                    ov0 || ov12);
    }
    pin.texel0 = ivec4(0);
    pin.texel1 = ivec4(0);
    bool uses_texel1 = F(EF_USES_TEXEL1);
    int s1 = st_s, t1 = st_t;
    if (pipelined) {
        s1 = c3.x;
        t1 = c3.y;
        tile1 = tile0;
        uses_texel1 = true;
    }
    // TEXEL0, then TEXEL1 (one call site of sample_texture).
    const bool convert1 = uses_texel1 && F(EF_CONVERT_ONE) && !F(EF_BILERP1);
    const int k0 = F(EF_USES_TEXEL0) ? 0 : 1, k1 = uses_texel1 && !convert1 ? 2 : 1;
    for (int k = k0; k < k1; ++k) {
        const bool second = k == 1;
        const ivec4 v = sample_texture(second ? tile1 : tile0, second ? s1 : st_s, second ? t1 : st_t,
                                       second && F(EF_CONVERT_ONE), second ? F(EF_BILERP1) : F(EF_BILERP0), pin.texel0);
        if (k == 0) pin.texel0 = v;
        else pin.texel1 = v;
    }
    if (convert1) pin.texel1 = texture_convert_factors(pin.texel0);
    return pin;
}

// ============================================================================
// Combiner

ivec4 unpack_color(uint c) { return ivec4(int(c >> 24), int((c >> 16) & 0xffu), int((c >> 8) & 0xffu), int(c & 0xffu)); }

uint csel(uint cyc, uint k, uint ch) {
    const uint idx = 4u * (4u * cyc + k) + ch;
    return (SW(ES_CSEL + idx / 4u) >> (8u * (idx % 4u))) & 0xffu;
}

ivec4 crow(uint sel, ivec4 comb, ivec4 t0, ivec4 t1, ivec4 shade, int noise_v, int lod) {
    switch (sel) {
        case 0u: return comb;
        case 1u: return t0;
        case 2u: return t1;
        case 3u: return unpack_color(SW(ES_PRIM_COLOR));
        case 4u: return shade;
        case 5u: return unpack_color(SW(ES_ENV_COLOR));
        case 6u: return ivec4(0x100);
        case 7u: return ivec4(noise_v);
        case 8u: return ivec4(0);
        case 9u: { const uint k = SW(ES_KEY_CENTER); return ivec4(int(k & 0xffu), int((k >> 8) & 0xffu), int((k >> 16) & 0xffu), 0); }
        case 10u: return ivec4(SI(ES_K4));
        case 11u: { const uint k = SW(ES_KEY_SCALE); return ivec4(int(k & 0xffu), int((k >> 8) & 0xffu), int((k >> 16) & 0xffu), 0); }
        case 12u: return ivec4(comb.a);
        case 13u: return ivec4(t0.a);
        case 14u: return ivec4(t1.a);
        case 15u: return ivec4(int(SW(ES_PRIM_COLOR) & 0xffu));
        case 16u: return ivec4(shade.a);
        case 17u: return ivec4(int(SW(ES_ENV_COLOR) & 0xffu));
        case 18u: return ivec4(lod);
        case 19u: return ivec4(SI(ES_PRIM_LOD_FRAC));
        default: return ivec4(SI(ES_K5));
    }
}

ivec3 g_key_a, g_key_sum; // chroma key: the RGB inputs A and the unrounded sums

ivec4 run_cycle(uint cyc, PixIn pin, ivec4 combined, ivec4 t0, ivec4 t1, int noise_v) {
    // A, B, C, D: the RGB inputs' rows, alpha from the alpha inputs' rows.
    const ivec4 va = ivec4(crow(csel(cyc, 0u, 0u), combined, t0, t1, pin.shade, noise_v, pin.lod_frac).xyz,
                           crow(csel(cyc, 0u, 3u), combined, t0, t1, pin.shade, noise_v, pin.lod_frac).w);
    const ivec4 vb = ivec4(crow(csel(cyc, 1u, 0u), combined, t0, t1, pin.shade, noise_v, pin.lod_frac).xyz,
                           crow(csel(cyc, 1u, 3u), combined, t0, t1, pin.shade, noise_v, pin.lod_frac).w);
    const ivec4 vc = ivec4(crow(csel(cyc, 2u, 0u), combined, t0, t1, pin.shade, noise_v, pin.lod_frac).xyz,
                           crow(csel(cyc, 2u, 3u), combined, t0, t1, pin.shade, noise_v, pin.lod_frac).w);
    const ivec4 vd = ivec4(crow(csel(cyc, 3u, 0u), combined, t0, t1, pin.shade, noise_v, pin.lod_frac).xyz,
                           crow(csel(cyc, 3u, 3u), combined, t0, t1, pin.shade, noise_v, pin.lod_frac).w);
    // combiner_equation, per channel
    const ivec4 ea = special_expand4(va), eb = special_expand4(vb), ec = sext4(vc, 9), ed = special_expand4(vd);
    const ivec4 prod = (ea - eb) * ec;
    if (F(EF_KEY_EN)) {
        g_key_a = va.xyz;
        g_key_sum = (prod.xyz + (ed.xyz << 8) + 0x80) & 0x1ffff;
    }
    return ((prod + 0x80) >> 8) + ed;
}

// The pixel's shading - everything that doesn't read memory - packed:
//   x: the combined colour r | g << 8 | b << 16 | a << 24 (copy mode: the word copied)
//   y: z (18 bits) | coverage count << 18 | kind << 22 (0 not drawn, 1 fill, 2 copy, 3 shaded)
//   z: rgb dither | shade alpha + alpha dither << 9
uvec3 shade_pixel(int x, int y) {
    const uint cycle = cycle_type();
    const bool multi = cycle == 1u;
    const int idir = g_flip ? 1 : -1;
    const int start_x = SPAN(EX_START_X), end_x = SPAN(EX_END_X);
    const uint serial = PW(EP_SERIAL);

    uint noise = 0u;
    if (F(EF_NEED_NOISE)) noise = noise_reseed(uint(x), uint(y), serial);

    // 0 none, 1 fill, 2 copy, 3 normal
    int kind = 0;
    uint copy_word = 0u;
    ivec4 comb = ivec4(0);
    int z = 0, dith = 0, cov_count = 0, shade_a = 0;

    if (cycle == 2u) { // copy
        int dx = g_flip ? (x - start_x) : (end_x - x);
        int dx_shift = 0, dx_mask = 0, fb_size = 0;
        const uint fmt = fb_fmt();
        if (fmt == FB_I4) { fb_size = 0; dx_mask = 0; dx_shift = 0; }
        else if (fmt == FB_I8) { fb_size = 1; dx_mask = ~7; dx_shift = 3; }
        else if (fmt == FB_RGBA5551 || fmt == FB_IA88) { fb_size = 2; dx_mask = ~3; dx_shift = 2; }
        else { fb_size = 4; dx_mask = 0; dx_shift = 1; }
        const int snapped = dx & dx_mask;
        const int s_offset = dx - snapped;
        const int lerp_dx = (dx >> dx_shift) * (g_flip ? 1 : -1);
        int ss = (SPAN(EX_STZW) + (PI(EP_DSTZW_DX) & ~0x1f) * lerp_dx) >> 16;
        int tt = (SPAN(EX_STZW + 1u) + (PI(EP_DSTZW_DX + 1u) & ~0x1f) * lerp_dx) >> 16;
        const int ww = (SPAN(EX_STZW + 3u) + (PI(EP_DSTZW_DX + 3u) & ~0x1f) * lerp_dx) >> 16;
        if (F(EF_PERSP)) {
            bool ov = false;
            perspective_divide(ss, tt, ww, ss, tt, ov, true);
        }
        const Tile tile = load_tile((PW(EP_FLAGS) >> 1) & 7u);
        const bool tlut = F(EF_TLUT);
        const int cs = shift_coord(ss, tile.slo, tile.shift_s) >> 5;
        const int ct = shift_coord(tt, tile.tlo, tile.shift_t) >> 5;
        int samp = 0;
        if (fb_size != 0) {
            const int so = fb_size == 1 ? (s_offset >> 1) : s_offset;
            const bool high_word = so < 2;
            const bool replicate = high_word && tile.size != 2u && !tlut;
            const uint s_shamt = min(tile.size, 2u);
            const uint idx_mask = (tile.size == 3u || tlut) ? 0x3ffu : 0x7ffu;
            if (replicate) {
                const int sx = cs + 2 * so;
                const int sA = mask_s(tile, sx), sB = mask_s(tile, sx + 1);
                const int t = mask_t(tile, ct);
                const uint tbase = tile.offset + tile.stride * uint(t);
                uint noA = (tbase * 2u + (uint(sA) << s_shamt)) & 0x1fffu;
                uint noB = (tbase * 2u + (uint(sB) << s_shamt)) & 0x1fffu;
                noA ^= (uint(t) & 1u) * 8u;
                noB ^= (uint(t) & 1u) * 8u;
                int a0 = int(tmem_half((noA >> 2) & idx_mask)), a1 = int(tmem_half((noB >> 2) & idx_mask));
                if (tile.size == 1u) {
                    a0 = (a0 >> (8 - 4 * int(noA & 2u))) & 0xff;
                    a1 = (a1 >> (8 - 4 * int(noB & 2u))) & 0xff;
                } else if (tile.size == 0u) {
                    a0 = ((a0 >> (12 - 4 * int(noA & 3u))) & 0xf) * 0x11;
                    a1 = ((a1 >> (12 - 4 * int(noB & 3u))) & 0xf) * 0x11;
                } else {
                    a0 >>= 8;
                    a1 >>= 8;
                }
                samp = (a0 << 8) | a1;
            } else {
                const int sx = mask_s(tile, cs + so);
                const int t = mask_t(tile, ct);
                const uint tbase = tile.offset + tile.stride * uint(t);
                uint no = (tbase * 2u + (uint(sx) << s_shamt)) & 0x1fffu;
                no ^= (uint(t) & 1u) * 8u;
                samp = int(tmem_half((no >> 2) & idx_mask));
                if (tlut) {
                    if (tile.size == 0u) {
                        samp >>= 12 - 4 * int(no & 3u);
                        samp &= 0xf;
                        samp |= int(tile.palette) << 4;
                    } else {
                        samp >>= 8 - 4 * int(no & 2u);
                        samp &= 0xff;
                    }
                    samp <<= 2;
                    samp += so;
                    samp = int(tmem_half(uint(samp | 0x400) & 0x7ffu));
                }
            }
            if (fb_size == 1) {
                samp >>= 8 - 8 * (s_offset & 1);
                samp &= 0xff;
            }
        }
        if (F(EF_ALPHA_TEST) && fb_size == 2 && (samp & 1) == 0) return uvec3(0u);
        copy_word = uint(samp);
        kind = 2;
    } else if (cycle == 3u) { // fill
        kind = 1;
    } else {
        const uint coverage = compute_coverage(x);
        if (coverage == 0u) return uvec3(0u);
        int coverage_count = bitCount(coverage);
        if (!F(EF_AA) && (coverage & 1u) == 0u) return uvec3(0u);

        // The pixel's inputs, and in 2-cycle mode with the alpha test the next
        // walked pixel's (one call site of pixel_inputs).
        const int xn = x + idir;
        const bool last = x == (g_flip ? end_x : start_x);
        const bool lookahead = multi && F(EF_ALPHA_TEST);
        const uint cov_n = lookahead && !last ? compute_coverage(xn) : 0u;
        PixIn pin, nin;
        for (int q = 0; q < (lookahead ? 2 : 1); ++q) {
            const PixIn r = pixel_inputs(q == 0 ? x : xn, q == 0 ? coverage : cov_n);
            if (q == 0) pin = r;
            else nin = r;
        }
        z = pin.z;

        int rgb_dith, alpha_dith;
        dither_coefficients(x, y >> (F(EF_INTERLACE) ? 1 : 0), int((g_f1 >> 2) & 3u), int(g_f1 & 3u), noise, rgb_dith,
                            alpha_dith);

        int alpha_reference = 0;
        const int noise_v = noise_combiner(noise);
        // The combiner (one call site of run_cycle): in 2-cycle mode the first
        // cycle, the next pixel's first cycle (alpha test), then the second
        // cycle, which sees the texels swapped (pipelining).
        int nv_dual = noise_v;
        if (multi && F(EF_NEED_NOISE_DUAL)) nv_dual = noise_combiner(noise_reseed(uint(x + 1023), uint(y + 7), serial + 11u));
        const int steps = multi ? (lookahead ? 3 : 2) : 1;
        ivec4 c0 = ivec4(0), n0 = ivec4(0);
        for (int st = 0; st < steps; ++st) {
            const bool final_step = st == steps - 1, next_step = st == 1 && !final_step;
            const PixIn pp = next_step ? nin : pin;
            const bool swap = final_step && multi;
            const ivec4 r = run_cycle(final_step ? 1u : 0u, pp, swap ? c0 : ivec4(0), swap ? pp.texel1 : pp.texel0,
                                      swap ? pp.texel0 : pp.texel1, swap ? nv_dual : noise_v);
            if (final_step) comb = r;
            else if (next_step) n0 = r;
            else c0 = r;
        }
        if (lookahead) {
            const int cnt_n = bitCount(cov_n);
            const int ca = clamp_9bit(n0.a);
            int ea = ca + ((ca + 1) >> 8);
            if (F(EF_ALPHA_CVG_SELECT)) ea = F(EF_CVG_TIMES_ALPHA) ? (ea * cnt_n + 4) >> 3 : cnt_n << 5;
            else ea += alpha_dith;
            alpha_reference = clamp(ea, 0, 0xff);
        }
        // finish_cycle1
        for (int i = 0; i < 4; ++i) comb[i] = clamp_9bit(comb[i]);
        int key_alpha = 0;
        if (F(EF_KEY_EN)) {
            key_alpha = 0x7fffffff;
            for (int i = 0; i < 3; ++i) {
                int k = sext(g_key_sum[i], 17);
                if (k > 0) k = (k & 0xf) == 8 ? -k + 0x10 : -k;
                k += SI(ES_KEY_WIDTH + uint(i)) << 4;
                key_alpha = min(key_alpha, k);
                comb[i] = clamp_9bit(g_key_a[i]);
            }
            key_alpha = clamp(key_alpha, 0, 0xff);
        }
        const int expanded = comb.a + ((comb.a + 1) >> 8);
        int modulated;
        if (F(EF_CVG_TIMES_ALPHA)) {
            modulated = (expanded * coverage_count + 4) >> 3;
            coverage_count = modulated >> 5;
        } else {
            modulated = coverage_count << 5;
        }
        const int e = F(EF_ALPHA_CVG_SELECT) ? modulated : F(EF_KEY_EN) ? key_alpha : expanded + alpha_dith;
        comb.a = clamp(e, 0, 0xff);
        if (!multi) alpha_reference = comb.a;

        if (F(EF_AA) && coverage_count == 0) return uvec3(0u);
        if (F(EF_ALPHA_TEST)) {
            const int threshold = F(EF_ALPHA_TEST_DITHER) ? int(noise & 0xffu) : int(SW(ES_BLEND_COLOR) & 0xffu);
            if (alpha_reference < threshold) return uvec3(0u);
        }
        dith = rgb_dith;
        cov_count = coverage_count;
        shade_a = min(pin.shade.a + alpha_dith, 0xff);
        kind = 3;
    }

    if (kind == 2) return uvec3(copy_word, 2u << 22, 0u);
    return uvec3(uint(comb.r) | uint(comb.g) << 8 | uint(comb.b) << 16 | uint(comb.a) << 24,
                 uint(z) | uint(cov_count) << 18 | uint(kind) << 22, uint(dith) | uint(shade_a) << 9);
}

#ifdef EXACT_MEMORY_PASS
uint g_debug_xy = 0xffffffffu; // the pixel (at the pass's scale) debug_log() is about
void debug_log(int x, int y, int a, int b, int c, int d, int e, int f, int g, int h) {
    if (P.debug_pixel != g_debug_xy) return;
    const uint n = uint(TAIL[P.debug_at]);
    TAIL[P.debug_at] = int(n + 1u);
    if (n >= 64u) return;
    const uint o = P.debug_at + 1u + n * 16u;
    TAIL[o] = int(PW(EP_SERIAL));
    TAIL[o + 1u] = a; TAIL[o + 2u] = b; TAIL[o + 3u] = c; TAIL[o + 4u] = d;
    TAIL[o + 5u] = e; TAIL[o + 6u] = f; TAIL[o + 7u] = g; TAIL[o + 8u] = h;
}

// The memory stage of a shaded pixel (see shade_pixel): depth test,
// blender, coverage, and the store.
void memory_pixel(int x, int y, uvec3 sh) {
    const bool multi = cycle_type() == 1u;
    const int kind = int((sh.y >> 22) & 3u);
    const uint copy_word = sh.x;
    const ivec4 comb = ivec4(int(sh.x & 0xffu), int((sh.x >> 8) & 0xffu), int((sh.x >> 16) & 0xffu), int(sh.x >> 24));
    const int z = int(sh.y & 0x3ffffu);
    int cov_count = int((sh.y >> 18) & 15u);
    const int dith = int(sh.z & 0x1ffu), shade_a = int((sh.z >> 9) & 0xffu);
    // ---- Memory: load the pixel's colour and depth
    ivec4 col;
    const uint cidx = read_colour(x, y, col);
    const uint fmt = fb_fmt();
    const bool alias = ((g_f1 >> 27) & 1u) != 0u;
    uint cur_depth = 0u, cur_dz = 0u;
    bool color_dirty = false, depth_dirty = false;
    const uint zidx = (SW(ES_Z_INDEX) + SW(ES_FB_WIDTH) * uint(y) + uint(x)) & (P.mask8 >> 1);
    {
        const uint wv = rd16(zidx);
        cur_depth = wv >> 2;
        cur_dz = hidden_get(zidx) | ((wv & 3u) << 2);
    }

    if (kind == 1) {
        uint c = SW(ES_FILL_COLOR);
        if (fmt == FB_RGBA8888) {
            col = ivec4(int(c >> 24), int((c >> 16) & 0xffu), int((c >> 8) & 0xffu), int(c & 0xffu));
            color_dirty = true;
        } else if (fmt == FB_RGBA5551) {
            c >>= ((cidx & 1u) ^ 1u) * 16u;
            col = ivec4(int((c >> 8) & 0xf8u), int((c >> 3) & 0xf8u), int((c << 2) & 0xf8u), int((c & 1u) * 0xe0u));
            color_dirty = true;
        } else if (fmt == FB_IA88) {
            c >>= ((cidx & 1u) ^ 1u) * 16u;
            c &= 0xffffu;
            col = ivec4(int((c >> 8) & 0xffu), int((c >> 8) & 0xffu), int((c >> 8) & 0xffu), int(c & 0xffu));
            color_dirty = true;
        } else if (fmt == FB_I8) {
            c >>= ((cidx & 3u) ^ 3u) * 8u;
            c &= 0xffu;
            col = ivec4(int(c));
            color_dirty = true;
        }
    } else if (kind == 2) {
        const uint wv = copy_word;
        if (fmt == FB_I4) {
            col = ivec4(0);
            color_dirty = true;
        } else if (fmt == FB_I8) {
            col = ivec4(int(wv & 0xffu));
            color_dirty = true;
        } else if (fmt == FB_RGBA5551) {
            col = ivec4(int((wv >> 8) & 0xf8u), int((wv >> 3) & 0xf8u), int((wv << 2) & 0xf8u), int((wv & 1u) * 0xe0u));
            color_dirty = true;
        }
    }
    if (kind == 1 || kind == 2) {
        if (alias && color_dirty) {
            // alias_color_to_depth
            if (fmt == FB_RGBA5551) {
                cur_dz = uint(((col.a & 0xff) >> 3) | (col.b & 8));
                uint wv = (uint(col.r) & 0xf8u) << 6;
                wv |= (uint(col.g) & 0xf8u) << 1;
                wv |= (uint(col.b) & 0xf8u) >> 4;
                cur_depth = wv & 0xffffu;
            } else if (fmt == FB_IA88) {
                const uint wv = (uint(col.r) << 8) | uint(col.a);
                cur_depth = (wv >> 2) & 0xffffu;
                cur_dz = ((wv & 3u) << 2) | ((wv & 1u) * 3u);
            }
        } else if (alias) {
            // (the C++ aliases whatever col holds; nothing was written)
            if (fmt == FB_RGBA5551) {
                cur_dz = uint(((col.a & 0xff) >> 3) | (col.b & 8));
                uint wv = (uint(col.r) & 0xf8u) << 6;
                wv |= (uint(col.g) & 0xf8u) << 1;
                wv |= (uint(col.b) & 0xf8u) >> 4;
                cur_depth = wv & 0xffffu;
            } else if (fmt == FB_IA88) {
                const uint wv = (uint(col.r) << 8) | uint(col.a);
                cur_depth = (wv >> 2) & 0xffffu;
                cur_dz = ((wv & 3u) << 2) | ((wv & 1u) * 3u);
            }
        }
    } else if (kind == 3) {
        const ivec4 mem = decode_memory(col);
        const int memory_coverage = mem.a >> 5;
        const int pdz = PI(EP_DZ), pdzc = PI(EP_DZ_COMPRESSED);
        const bool aa = F(EF_AA), force_blend = F(EF_FORCE_BLEND);

        bool blend_en, coverage_wrap, z_pass;
        int shift_a, shift_b;
        if (F(EF_Z_COMPARE)) {
            const int memory_z = z_decompress(cur_depth);
            int memory_dz = dz_decompress(int(cur_dz));
            const int prec = int((cur_depth >> 11) & 0xfu);
            bool coplanar = false;
            shift_a = clamp(pdzc - int(cur_dz), 0, 4);
            shift_b = clamp(int(cur_dz) - pdzc, 0, 4);
            if (prec < 3) {
                if (memory_dz != 0x8000) {
                    memory_dz = max(memory_dz << 1, 16 >> prec);
                } else {
                    coplanar = true;
                    memory_dz = 0xffff;
                }
            }
            int combined_dz = combine_dz(pdz | memory_dz);
            const int combined_dz_ip = combined_dz;
            combined_dz <<= 3;
            const bool farther = coplanar || (z + combined_dz) >= memory_z;
            const bool overflow = (cov_count + memory_coverage) >= 8;
            blend_en = force_blend || (!overflow && aa && farther);
            coverage_wrap = overflow;
            const bool max_z = memory_z == 0x3ffff;
            const bool front = z < memory_z;
            const bool nearer = coplanar || (z - combined_dz) <= memory_z;
            const int z_mode = int((g_f1 >> 6) & 3u);
            if (z_mode == 0) {
                z_pass = max_z || (overflow ? front : nearer);
            } else if (z_mode == 1) {
                if (!front || !farther || !overflow) {
                    z_pass = max_z || (overflow ? front : nearer);
                } else {
                    const int cdz = dz_compress(combined_dz_ip & 0xffff);
                    const int coeff = ((memory_z >> cdz) - (z >> cdz)) & 0xf;
                    cov_count = min((coeff * cov_count) >> 3, 8);
                    z_pass = true;
                }
            } else if (z_mode == 2) {
                z_pass = front || max_z;
            } else {
                z_pass = farther && nearer && !max_z;
            }
        } else {
            shift_a = 0;
            shift_b = min(0xf - pdzc, 4);
            const bool overflow = (cov_count + memory_coverage) >= 8;
            blend_en = force_blend || (!overflow && aa);
            coverage_wrap = overflow;
            z_pass = true;
        }

        if (z_pass && (!aa || cov_count != 0)) {
            ivec4 pixel = comb;
            const ivec4 fog = unpack_color(SW(ES_FOG_COLOR));
            const ivec4 bcol = unpack_color(SW(ES_BLEND_COLOR));
            ivec3 rgb = ivec3(0);
            // Blender: in 2-cycle mode the first cycle (memc = the delayed memory
            // colour) never reads memory here - those primitives are drawn on the CPU.
            const int ncyc = multi ? 2 : 1;
            for (int bc = 0; bc < ncyc; ++bc) {
                const bool final_cycle = bc == ncyc - 1;
                const uint mc = multi ? uint(bc) : 0u;
                const uint m0 = (g_f1 >> (8u + 2u * (4u * mc + 0u))) & 3u;
                const uint m1 = (g_f1 >> (8u + 2u * (4u * mc + 1u))) & 3u;
                const uint m2 = (g_f1 >> (8u + 2u * (4u * mc + 2u))) & 3u;
                const uint m3 = (g_f1 >> (8u + 2u * (4u * mc + 3u))) & 3u;
                const ivec4 memc = mem;
                const ivec4 src1 = m2 == 0u ? pixel : m2 == 1u ? memc : m2 == 2u ? bcol : fog;
                ivec3 outc;
                bool done = false;
                if (final_cycle && F(EF_COLOR_ON_CVG) && !coverage_wrap) {
                    outc = src1.rgb;
                    done = true;
                }
                const ivec4 src0 = m0 == 0u ? pixel : m0 == 1u ? memc : m0 == 2u ? bcol : fog;
                if (!done && final_cycle && (!blend_en || (m1 == 0u && m3 == 0u && pixel.a == 0xff))) {
                    outc = src0.rgb;
                    done = true;
                }
                if (!done) {
                    int a0, a1;
                    if (m1 == 0u) a0 = pixel.a;
                    else if (m1 == 1u) a0 = fog.a;
                    else if (m1 == 2u) a0 = shade_a;
                    else a0 = 0;
                    if (m3 == 0u) a1 = ~a0 & 0xff;
                    else if (m3 == 1u) a1 = memc.a;
                    else if (m3 == 2u) a1 = 0xff;
                    else a1 = 0;
                    a0 >>= 3;
                    a1 >>= 3;
                    if (m3 == 1u) {
                        a0 = (a0 >> shift_a) & 0x3c;
                        a1 = (a1 >> shift_b) | 3;
                    }
                    const ivec3 bl = src0.rgb * a0 + src1.rgb * (a1 + 1);
                    if (!final_cycle || force_blend) {
                        outc = (bl >> 5) & 0xff;
                    } else {
                        const int sum = (a0 >> 2) + (a1 >> 2) + 1;
                        for (int i = 0; i < 3; ++i) outc[i] = blender_divide(sum & 0xf, (bl[i] >> 2) & 0x7ff);
                    }
                }
                if (final_cycle) rgb = outc;
                else pixel.rgb = outc;
            }
            if (F(EF_DITHER_EN)) rgb = rgb_dither(rgb, dith);
            const int new_cov = blend_coverage(cov_count, memory_coverage, blend_en, int((g_f1 >> 4) & 3u));
            if (fmt == FB_I4) col.rgb = rgb;
            else col = ivec4(rgb, new_cov << 5);
            color_dirty = true;
            if (F(EF_Z_UPDATE)) {
                cur_depth = z_compress(z);
                cur_dz = uint(pdzc);
                depth_dirty = true;
                if (alias) {
                    // alias_depth_to_color
                    const uint wv = (cur_depth << 4) | cur_dz;
                    if (fmt == FB_RGBA5551) {
                        col = ivec4(int((wv >> 10) & 0xf8u), int((wv >> 5) & 0xf8u), int(wv & 0xf8u), int((wv & 7u) << 5));
                    } else if (fmt == FB_IA88) {
                        col.r = int((wv >> 10) & 0xffu);
                        col.a = int((wv >> 2) & 0xffu);
                    }
                }
            } else if (alias) {
                if (fmt == FB_RGBA5551) {
                    cur_dz = uint(((col.a & 0xff) >> 3) | (col.b & 8));
                    uint wv = (uint(col.r) & 0xf8u) << 6;
                    wv |= (uint(col.g) & 0xf8u) << 1;
                    wv |= (uint(col.b) & 0xf8u) >> 4;
                    cur_depth = wv & 0xffffu;
                } else if (fmt == FB_IA88) {
                    const uint wv = (uint(col.r) << 8) | uint(col.a);
                    cur_depth = (wv >> 2) & 0xffffu;
                    cur_dz = ((wv & 3u) << 2) | ((wv & 1u) * 3u);
                }
            }
        }
    }

    debug_log(x, y, kind, cov_count, z, int(cur_depth), int(cur_dz), (comb.r << 24) | (comb.g << 16) | (comb.b << 8) | comb.a,
              (col.r << 24) | (col.g << 16) | (col.b << 8) | (col.a & 0xff), int(depth_dirty) | int(color_dirty) << 1);
    // ---- Store
    if (color_dirty) {
        if (fmt == FB_I4 || fmt == FB_I8) {
            // The ninth bits go with the odd byte; writing the even one leaves them.
            const int c = fmt == FB_I4 ? 0 : (cidx & 1u) != 0u ? col.g : col.r;
            wr8(cidx, uint(c) & 0xffu);
            if ((cidx & 1u) != 0u) {
                const uint bits = fmt == FB_I4 ? uint(col.a & 3) : uint((c & 1) * 3);
                hidden_set(cidx >> 1, bits);
            }
        } else if (fmt == FB_RGBA5551) {
            const uint r = uint(col.r) & 0xf8u, g = uint(col.g) & 0xf8u, b = uint(col.b) & 0xf8u;
            const uint cov = (uint(col.a) & 0xffu) >> 5;
            wr16(cidx, (r << 8) | (g << 3) | (b >> 2) | (cov >> 2));
            hidden_set(cidx, cov & 3u);
        } else if (fmt == FB_IA88) {
            const uint wv = ((uint(col.r) & 0xffu) << 8) | (uint(col.a) & 0xffu);
            wr16(cidx, wv);
            hidden_set(cidx, uint(col.a & 1) * 3u);
        } else {
            wr32(cidx, (uint(col.r) & 0xffu) | (uint(col.g) & 0xffu) << 8 | (uint(col.b) & 0xffu) << 16 |
                           (uint(col.a) & 0xffu) << 24);
            hidden_set(2u * cidx, uint(col.g & 1) * 3u);
            hidden_set(2u * cidx + 1u, uint(col.a & 1) * 3u);
        }
    }
    if (!alias && depth_dirty) {
        wr16(zidx, ((cur_depth << 2) | (cur_dz >> 2)) & 0xffffu);
        hidden_set(zidx, cur_dz & 3u);
    }
}
#endif
