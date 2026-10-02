// Layout of what the bit-exact RDP (src/rdp_exact.cpp) records for its GPU
// back end (src/gpu/rdp_exact_gpu.cpp, shaders/exact.comp). Included by the
// C++ side and by the GLSL shader, so it holds nothing but #defines.
//
// The front end (ExactRdp on the CPU) runs every command as usual - state,
// tiles, TMEM loads - and for each primitive records its decoded draw state,
// its own setup and its spans, one per scanline. The GPU draws the pixels.
//
// One read-only storage buffer of 32-bit words holds a flush's data:
//   states   EXACT_STATE_WORDS per recorded state
//   prims    EXACT_PRIM_WORDS per primitive, in drawing order
//   spans    EXACT_SPAN_WORDS per scanline of a primitive
//   TMEM     EXACT_TMEM_WORDS per snapshot (halfword h in bits 16*(h&1) of word h/2)
//   bins     per pass: the primitive indices of every 8x8-pixel tile, and
//            the work list - per tile with primitives: x | y << 16, first, count

#ifndef ORBIT64_EXACT_LAYOUT_H
#define ORBIT64_EXACT_LAYOUT_H

#define EXACT_TILE_SIZE 8u
#define EXACT_TMEM_WORDS 1024u

// ---- Draw state ----------------------------------------------------------
#define EXACT_STATE_WORDS 64u
#define ES_FLAGS0 0u
// bits of ES_FLAGS0
#define EF_PERSP (1u << 2)
#define EF_DETAIL (1u << 3)
#define EF_SHARPEN (1u << 4)
#define EF_TEX_LOD (1u << 5)
#define EF_TLUT (1u << 6)
#define EF_TLUT_TYPE (1u << 7)
#define EF_SAMPLE_QUAD (1u << 8)
#define EF_MID_TEXEL (1u << 9)
#define EF_BILERP0 (1u << 10)
#define EF_BILERP1 (1u << 11)
#define EF_CONVERT_ONE (1u << 12)
#define EF_FORCE_BLEND (1u << 13)
#define EF_ALPHA_CVG_SELECT (1u << 14)
#define EF_CVG_TIMES_ALPHA (1u << 15)
#define EF_COLOR_ON_CVG (1u << 16)
#define EF_IMAGE_READ (1u << 17)
#define EF_Z_UPDATE (1u << 18)
#define EF_Z_COMPARE (1u << 19)
#define EF_AA (1u << 20)
#define EF_ALPHA_TEST_DITHER (1u << 21)
#define EF_ALPHA_TEST (1u << 22)
#define EF_DITHER_EN (1u << 23)
#define EF_KEY_EN (1u << 24)
#define EF_INTERLACE (1u << 25)
#define EF_USES_TEXEL0 (1u << 26)
#define EF_USES_TEXEL1 (1u << 27)
#define EF_USES_PIPELINED_TEXEL1 (1u << 28)
#define EF_USES_LOD (1u << 29)
#define EF_NEED_NOISE (1u << 30)
#define EF_NEED_NOISE_DUAL (1u << 31)
// bits 0-1: cycle type (0 1-cycle, 1 2-cycle, 2 copy, 3 fill)
#define ES_FLAGS1 1u // dither (4 bits) | coverage mode << 4 | z mode << 6 | blender muxes << 8
                     // (cycle c, input k at 8 + 2 * (4c + k)) | frame buffer format << 24 | alias << 27
#define ES_CSEL 2u   // 8 words: the combiner's input rows, byte (4 * (4c + k) + channel)
#define ES_MIN_LOD 10u
#define ES_FACTORS 11u // 4 words (convert K0-K3)
#define ES_PRIM_COLOR 15u
#define ES_ENV_COLOR 16u
#define ES_FOG_COLOR 17u
#define ES_BLEND_COLOR 18u
#define ES_FILL_COLOR 19u
#define ES_PRIM_LOD_FRAC 20u
#define ES_KEY_CENTER 21u // r | g << 8 | b << 16
#define ES_KEY_SCALE 22u
#define ES_KEY_WIDTH 23u  // 3 words
#define ES_K4 26u
#define ES_K5 27u
#define ES_FB_INDEX 28u   // colour image address in pixels
#define ES_FB_WIDTH 29u
#define ES_Z_INDEX 30u    // depth image address in halfwords
#define ES_TMEM 31u       // word offset of the TMEM snapshot (set by the back end)
#define ES_TILES 32u      // 8 tiles of 4 words:
// slo | shi << 16, tlo | thi << 16, offset | stride << 16,
// fmt | size << 3 | palette << 5 | mask_s << 9 | shift_s << 13 | mask_t << 17 | shift_t << 21 | flags << 25
#define EXACT_TILE_WORDS 4u

// ---- Primitive -----------------------------------------------------------
#define EXACT_PRIM_WORDS 32u
#define EP_STATE 0u     // word offset of its state
#define EP_FLAGS 1u     // flip | setup tile (0-7) << 1 | max LOD level << 4 | walks pixels << 8 | EPF_WRAP
// Copy/fill mode: only the pixels at x == the frame buffer's width, each drawn
// where it lands - at x = 0 a row down (the shaders run them at that pixel).
#define EPF_WRAP (1u << 9)
// Internal resolution: the primitive isn't upscaled - every sample of a pixel
// gets what the native pixel gets (copy and fill modes).
#define EPF_MIRROR (1u << 10)
#define EP_SERIAL 2u    // the primitive counter the noise is seeded from
#define EP_DZ 3u
#define EP_DZ_COMPRESSED 4u
#define EP_DRGBA_DX 5u  // 4 words each
#define EP_DRGBA_DY 9u
#define EP_DSTZW_DX 13u
#define EP_DSTZW_DY 17u
#define EP_SPANS 21u    // word offset of its first span
#define EP_Y0 22u       // the scanline of the first span
#define EP_ROWS 23u     // spans recorded: every drawn scanline and the one after
#define EP_TAIL_X 24u   // the last pixel walked (its memory colour is the delayed one of
#define EP_TAIL_Y 25u   // the next 2-cycle primitive), -1 if none
#define EP_TAIL_SLOT 26u
// Internal resolution (scale S > 1): the spans at S times the resolution
// (scanlines y0 * S on), as EP_SPANS, EP_Y0, EP_ROWS are for the native ones.
#define EP_HSPANS 27u
#define EP_HY0 28u
#define EP_HROWS 29u

// ---- The high-resolution copy of memory (internal resolution S > 1) -----
// Per 4 KB page of RDRAM that frame buffers use, a slot: S * S copies of the
// page's bytes (sample (x % S, y % S) of a pixel is copy (y % S) * S + x % S)
// in the colour buffer, and of its ninth bits (a byte per halfword) in the
// hidden one. A page table maps RDRAM pages to slots (EXACT_NO_SLOT: none).
#define EXACT_PAGE_SHIFT 12u
#define EXACT_NO_SLOT 0xffffffffu

// ---- Span ----------------------------------------------------------------
#define EXACT_SPAN_WORDS 21u
#define EX_RGBA 0u
#define EX_STZW 4u
#define EX_XLEFT 8u
#define EX_XRIGHT 12u
#define EX_BASE_X 16u
#define EX_START_X 17u
#define EX_END_X 18u
#define EX_LODLENGTH 19u
#define EX_VALID 20u

#endif
