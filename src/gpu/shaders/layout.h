// Layout of the data the GPU RDP renderer (src/gpu/rdp_gpu.cpp) hands to its
// compute shaders. Included by the C++ side and by the GLSL shaders, so it
// holds nothing but #defines.
//
// Everything a frame draws lives in one read-only storage buffer of 32-bit
// words ("D" in the shaders):
//
//   [0, 256)          i / 255.0f for every byte value (float bits), so the
//                     shaders use exactly the values the CPU pipeline does
//   states            GPU_STATE_WORDS per recorded draw state
//   TMEM snapshots    GPU_TMEM_WORDS each: 4 KB of TMEM (byte k is bits
//                     8*(k&3).. of word k/4), then the 512 per-word "loaded
//                     with DXT 0" flags as a bit set
//   blit / upload     pixel data some primitives point at
//   prims             GPU_PRIM_WORDS per primitive, in drawing order
//   bins              per pass: a (first, count) pair per 8x8-pixel tile of
//                     the frame buffer, the primitive indices those point
//                     into, and the list of tiles that have any

#ifndef ORBIT64_GPU_LAYOUT_H
#define ORBIT64_GPU_LAYOUT_H

#define GPU_LUT_WORDS 256u

// Tile of the frame buffer (native pixels) primitives are binned by. The
// shaders run 8x8 threads per workgroup, so one tile at scale S is S*S
// workgroups.
#define GPU_BIN_SIZE 8u

// ---- Primitives ------------------------------------------------------------
#define GPU_PRIM_WORDS 32u
// word 0: type
// word 1: x0 | x1 << 16   the pixels it can touch, inclusive, at scale S
// word 2: y0 | y1 << 16
// word 31: word offset of its draw state (FILL and UPLOAD have none)
#define GPU_PRIM_STATE 31u
#define GPU_PRIM_TRI 1u    // 3..23: per vertex sx, sy, sz, 1/w, u/w, v/w, rgba (r in bits 0-7); 24: 1/area,
                           // 25: the direction rows are walked in (1.0 or -1.0),
                           // 26: depth slope (raster::normalize_dzpix) | compressed << 16
#define GPU_PRIM_RECT 2u   // 3: tile, 4: s, 5: t, 6: ds per pixel, 7: dt per row (both at scale S), 8: flip,
                           // 9..12: S min/max, T min/max (the native rectangle's range),
                           // 13: TEXEL0 tile | TEXEL1 tile << 4 | LOD fraction << 8 (raster::lod_tiles)
#define GPU_PRIM_FILL 3u   // 3: ARGB (written as it is)
#define GPU_PRIM_BLOCK 4u  // 3: ARGB, 4: z (Vertex::sz) (one native pixel through the pixel pipeline)
#define GPU_PRIM_BLIT 5u   // 3: data offset, 4: width, 5: x0, 6: y0 (native): ARGB per native pixel, depth 0
#define GPU_PRIM_UPLOAD 6u // 3: data offset, 4: width, 5: x0, 6: y0 (native): (valid, ARGB) per native pixel

// ---- Draw states -------------------------------------------------------------
#define GPU_STATE_WORDS 103u
// word 0: flags
#define GPU_ST_TWO_CYCLE (1u << 0)
#define GPU_ST_COPY (1u << 1)
#define GPU_ST_FILL_OR_COPY (1u << 2)
#define GPU_ST_POINT (1u << 3)
#define GPU_ST_TEXTURED (1u << 4)
#define GPU_ST_SMOOTH (1u << 5)
#define GPU_ST_COMBINED (1u << 6)
#define GPU_ST_NEED_TEX0 (1u << 7)
#define GPU_ST_NEED_TEX1 (1u << 8)
#define GPU_ST_PIPE_TEX1 (1u << 9)     // DrawState::pipelined_tex1
#define GPU_ST_MID_TEXEL (1u << 10)    // G_TF_AVERAGE
#define GPU_ST_Z_COMPARE (1u << 11)
#define GPU_ST_Z_UPDATE (1u << 12)
#define GPU_ST_16BIT (1u << 13)
#define GPU_ST_BILERP0 (1u << 14)
#define GPU_ST_BILERP1 (1u << 15)
#define GPU_ST_FORCE_BLEND (1u << 16)  // DrawState::force_blend
#define GPU_ST_AA (1u << 17)
#define GPU_ST_IMAGE_READ (1u << 18)
#define GPU_ST_COLOR_ON_CVG (1u << 19)
#define GPU_ST_CVG_X_ALPHA (1u << 20)
#define GPU_ST_ALPHA_CVG_SEL (1u << 21)
#define GPU_ST_ALPHA_TEST (1u << 22)
#define GPU_ST_ALPHA_DITHER (1u << 23)
#define GPU_ST_NOISE (1u << 24)        // DrawState::uses_noise
#define GPU_ST_Z_PRIM (1u << 25)       // DrawState::z_source_prim
// word 1: tlut type | active tile << 4 | z mode << 8 | coverage mode << 10 | alpha threshold << 16
// words 2-5: combiner inputs (raster::CcIn) of cycle 1: colour A, B, C, D bytes, then alpha;
//            then cycle 2's (1-cycle mode runs cycle 2's)
// words 6-9: primitive, environment, blend and fog colour (RDP order: r in bits 24-31)
// word 10: blender cycle 1 p | a << 8 | m << 16 | b << 24 (word 97: cycle 2)
// words 11-14: scissor x0, x1, y0, y1 at scale S, half-open
// word 15: word offset of the TMEM snapshot
// words 16..95: 8 tiles of GPU_TILE_WORDS
// word 96: level of detail: flags | max level << 8 | min level << 16 | PRIM_LOD_FRAC << 24
// word 98: key centre r, g, b (bytes 0-2); word 99: key scale r, g, b; word 100: K4 | K5 << 16
// word 101: the seed of the combiner's noise (DrawState::noise_seed)
// word 102: primitive depth | its slope << 16 (gDPSetPrimDepth)
#define GPU_ST_LOD 96u
#define GPU_ST_BLEND2 97u
#define GPU_ST_KEY_CENTER 98u
#define GPU_ST_KEY_SCALE 99u
#define GPU_ST_K45 100u
#define GPU_ST_NOISE_SEED 101u
#define GPU_ST_PRIM_Z 102u
#define GPU_LOD_TEX_EN (1u << 0)
#define GPU_LOD_SHARPEN (1u << 1)
#define GPU_LOD_DETAIL (1u << 2)
#define GPU_LOD_DOLOD (1u << 3)
#define GPU_ST_TILES 16u
#define GPU_TILE_WORDS 10u
// tile word 0: format | size << 4 | palette << 8 | mask_s << 12 | mask_t << 16 (limited to 10) |
//              clamp_s << 20 | clamp_t << 21 (CLAMP, or no mask) | mirror_s << 22 | mirror_t << 23
// 1: TMEM byte offset, 2: row stride, 3: SL | SH << 16, 4: TL | TH << 16 (10.2),
// 5: shift s | shift t << 8, 6-8: unused,
// 9: offset of the tile's decoded texels in the texel buffer, or GPU_NO_TABLE
#define GPU_NO_TABLE 0xFFFFFFFFu

#define GPU_TMEM_WORDS (1024u + 16u)

// ---- Decoded textures ----------------------------------------------------------
// decode.comp fills the texel buffer ("T") before the draws: for each job,
// fetch_wrapped() of one tile for every wrapped coordinate, w x h ARGB8888
// (the CPU renderer's TexCache). Jobs are GPU_JOB_WORDS each:
// 0: word offset of the tile (in its state), 1: TMEM snapshot, 2: TLUT type,
// 3: w, 4: h, 5: offset in the texel buffer, 6: first block (GPU_DECODE_BLOCK texels)
#define GPU_JOB_WORDS 8u
#define GPU_DECODE_BLOCK 256u

#endif
