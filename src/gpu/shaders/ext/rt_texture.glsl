// Texture alpha for ray tracing: whether a ray goes through a cut-out
// texture (foliage, fences, billboards) where it hits it. The texture unit
// of raster.comp (raster::sample_texture), nearest texel only. The including
// shader declares D (frame data, with the recorded draw states) and T (the
// decoded textures), and includes ../layout.h.

#include "../tmem.glsl"

int rt_floor_to_int(float v) {
    int i = int(v);
    return v < float(i) ? i - 1 : i;
}

// raster::tex_coord(): texels -> s10.5, shifted, clamped, relative to the tile.
int rt_tex_coord(float c, uint shift, bool clmp, uint lo, uint hi) {
    float v = c * 32.0;
    int x = !(v > -32768.0) ? -0x8000 : v >= 32767.0 ? 0x7FFF : rt_floor_to_int(v);
    if (shift < 11u) x >>= int(shift);
    else x = int(uint(x) << (32u - shift)) >> 16;
    int l = int(lo), h = int(hi);
    if (clmp) {
        if ((x >> 3) >= h) return (((h >> 2) - (l >> 2)) & 0x3FF) << 5;
        return max(x - (l << 3), 0);
    }
    return x - (l << 3);
}

int rt_tex_mask(int c, uint mask, bool mirror) {
    if (mask == 0u) return c;
    int m = 1 << int(mask);
    if (mirror) c ^= max((c & m) - 1, 0);
    return c & (m - 1);
}

uint rt_texel_at(uint sb, uint tb, int s, int t) {
    uint table = D[tb + 9u];
    if (table != GPU_NO_TABLE) {
        uint d = D[tb];
        uint ms = (d >> 12u) & 15u, mt = (d >> 16u) & 15u;
        uint ws = D[tb + 3u], wt = D[tb + 4u];
        uint w = ms != 0u ? (1u << ms) : ((((ws >> 16u) >> 2u) - ((ws & 0xFFFFu) >> 2u)) & 0x3FFu) + 2u;
        uint h = mt != 0u ? (1u << mt) : ((((wt >> 16u) >> 2u) - ((wt & 0xFFFFu) >> 2u)) & 0x3FFu) + 2u;
        if (uint(s) < w && uint(t) < h) return T[table + uint(t) * w + uint(s)];
    }
    return fetch_wrapped(tb, D[sb + 15u], D[sb + 1u] & 15u, s, t);
}

// The alpha (0..255) of draw state sb's active tile at texture coordinates s, t.
uint rt_texel_alpha(uint sb, float s, float t) {
    uint tile = (D[sb + 1u] >> 4u) & 7u;
    uint tb = sb + GPU_ST_TILES + tile * GPU_TILE_WORDS;
    uint d = D[tb], sh = D[tb + 5u];
    int cs = rt_tex_coord(s, sh & 0xFFu, ((d >> 20u) & 1u) != 0u, D[tb + 3u] & 0xFFFFu, D[tb + 3u] >> 16u);
    int ct = rt_tex_coord(t, (sh >> 8u) & 0xFFu, ((d >> 21u) & 1u) != 0u, D[tb + 4u] & 0xFFFFu, D[tb + 4u] >> 16u);
    int s0 = rt_tex_mask(cs >> 5, (d >> 12u) & 15u, ((d >> 22u) & 1u) != 0u);
    int t0 = rt_tex_mask(ct >> 5, (d >> 16u) & 15u, ((d >> 23u) & 1u) != 0u) & 0xFF;
    return rt_texel_at(sb, tb, s0, t0) >> 24u;
}
