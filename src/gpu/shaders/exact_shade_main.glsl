// The bit-exact RDP's shading (exact.glsl, shade_pixel): one workgroup per
// (8x8-pixel tile, primitive) pair - nothing here reads memory. Included by
// exact_shade*.comp, which pick the variant (EXACT_VARIANT, exact.glsl): the
// pairs are dispatched by the variant their primitive's state needs.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define NUM_UNIFORM 1
#define NUM_RO_BUF 1
#define NUM_RW_TEX 0
#include "common.glsl"
#include "exact_layout.h"

layout(std430, RO_BUFFER(0)) readonly buffer Data { uint D[]; };
layout(std430, RW_BUFFER(0)) buffer Shaded { uint S[]; }; // 3 words per pixel of every pair

layout(std140, UNIFORM(0)) uniform Pass {
    uint entries;  // word offset of the pairs: primitive, tile x | y << 16
    uint first;    // the first pair of the chunk (S starts with it)
    uint count;    // pairs this dispatch shades
    uint groups_x; // workgroups per row of the dispatch
    uint list;     // word offset of the indices of those pairs
    uint scale;    // internal resolution of the pass (1: native)
} P;

#define EXACT_SHARED
#include "exact.glsl"

void main() {
    const uint k = gl_WorkGroupID.y * P.groups_x + gl_WorkGroupID.x;
    if (k >= P.count) return; // (the whole workgroup)
    const uint e = D[P.list + k];
    const uint li = gl_LocalInvocationID.y * EXACT_TILE_SIZE + gl_LocalInvocationID.x;
    const uint out_at = ((e - P.first) * EXACT_TILE_SIZE * EXACT_TILE_SIZE + li) * 3u;
    const uint tw = D[P.entries + e * 2u + 1u];
    const uint ty = tw >> 16;
    // The pair into shared memory: primitive, state, the spans of the tile's
    // rows and the next, TMEM.
    const uint po = D[P.entries + e * 2u] * EXACT_PRIM_WORDS + D[P.entries - 1u];
    const uint so = D[po + EP_STATE];
    if (li < EXACT_PRIM_WORDS) s_prim[li] = D[po + li];
    s_state[li] = D[so + li];
    // Internal resolution S > 1: the primitive's high-resolution spans and
    // pixels - or, for one that isn't upscaled (EPF_MIRROR), the native pixel
    // each sample belongs to.
    const uint pflags = D[po + EP_FLAGS];
    const int scale = int(P.scale);
    const bool up = scale > 1 && (pflags & EPF_MIRROR) == 0u, mirror = scale > 1 && !up;
    const int y0 = int(D[po + (up ? EP_HY0 : EP_Y0)]), rows = int(D[po + (up ? EP_HROWS : EP_ROWS)]);
    const uint spans = D[po + (up ? EP_HSPANS : EP_SPANS)];
    // (EPF_WRAP: the pixels at x == width of the row above)
    const bool wrap = (pflags & EPF_WRAP) != 0u;
    const int row0 = (mirror ? int(ty * EXACT_TILE_SIZE) / scale : int(ty * EXACT_TILE_SIZE)) - y0 - (wrap ? 1 : 0);
    for (uint w = li; w < (EXACT_TILE_SIZE + 1u) * EXACT_SPAN_WORDS; w += 64u) {
        const int r = row0 + int(w / EXACT_SPAN_WORDS);
        s_span[w] = r >= 0 && r < rows ? D[spans + uint(r) * EXACT_SPAN_WORDS + w % EXACT_SPAN_WORDS] : 0u;
    }
    const uint f0 = D[so + ES_FLAGS0];
    if ((f0 & 3u) == 2u || (f0 & (EF_USES_TEXEL0 | EF_USES_TEXEL1 | EF_USES_PIPELINED_TEXEL1)) != 0u) {
        const uint tm = D[so + ES_TMEM];
        for (uint w = li; w < EXACT_TMEM_WORDS; w += 64u) s_tmem[w] = D[tm + w];
    }
    barrier();

    const int x = int((tw & 0xffffu) * EXACT_TILE_SIZE + gl_LocalInvocationID.x);
    const int y = int(ty * EXACT_TILE_SIZE + gl_LocalInvocationID.y);
    const int px = mirror ? x / scale : x, py = mirror ? y / scale : y; // the pixel shaded
    g_S = up ? scale : 1;
    g_f0 = s_state[ES_FLAGS0];
    g_f1 = s_state[ES_FLAGS1];
    g_flip = (pflags & 1u) != 0u;
    const int row = py - y0 - (wrap ? 1 : 0);
    g_span = uint(row - row0) * EXACT_SPAN_WORDS;
    g_next = g_span + EXACT_SPAN_WORDS;
    uvec3 r = uvec3(0u);
    // The pixel shaded: this one, or for EPF_WRAP (at x = 0) the one at x == width a row up.
    const int width = int(s_state[ES_FB_WIDTH]) * g_S;
    const int sx = wrap ? width : px, sy = wrap ? py - 1 : py;
    if (row >= 0 && row < rows - 1 && (wrap ? px == 0 : px < width) && s_span[g_span + EX_VALID] != 0u &&
        sx >= int(s_span[g_span + EX_START_X]) && sx <= int(s_span[g_span + EX_END_X]))
        r = shade_pixel(sx, sy);
    S[out_at] = r.x;
    S[out_at + 1u] = r.y;
    S[out_at + 2u] = r.z;
}
