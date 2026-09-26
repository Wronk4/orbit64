// TMEM texel decoding: fetch_wrapped() of src/raster.hpp. The including
// shader declares the frame data buffer D (shaders/layout.h).

uint tmem_byte(uint tm, uint off) { return (D[tm + (off >> 2u)] >> ((off & 3u) << 3u)) & 0xFFu; }
bool tmem_dxt(uint tm, uint word) { return ((D[tm + 1024u + (word >> 5u)] >> (word & 31u)) & 1u) != 0u; }

uint rgba16_to_argb(uint p) {
    uint r = (((p >> 11u) & 31u) * 255u + 15u) / 31u;
    uint g = (((p >> 6u) & 31u) * 255u + 15u) / 31u;
    uint b = (((p >> 1u) & 31u) * 255u + 15u) / 31u;
    uint a = (p & 1u) != 0u ? 255u : 0u;
    return (a << 24u) | (r << 16u) | (g << 8u) | b;
}

uint ia_argb(uint i, uint a) { return (a << 24u) | (i << 16u) | (i << 8u) | i; }

uint lookup_tlut(uint tm, uint index, uint tlut_type) {
    uint off = 0x800u + index * 2u;
    if (off + 1u >= 4096u) return 0u;
    uint p = (tmem_byte(tm, off) << 8u) | tmem_byte(tm, off + 1u);
    if (tlut_type == 3u) return ia_argb((p >> 8u) & 0xFFu, p & 0xFFu);
    return rgba16_to_argb(p);
}

int wrap_coord(int c, uint mask, bool clamp_en, bool mirror_en, int extent) {
    int ext = extent > 0 ? extent : 0;
    if (mask == 0u) return clamp(c, 0, ext);
    if (clamp_en) c = clamp(c, 0, ext);
    int period = 1 << int(mask);
    if (mirror_en) {
        int span = period * 2;
        c &= span - 1;
        if (c >= period) c = span - 1 - c;
        return c;
    }
    return c & (period - 1);
}

int wrap_s(uint tb, int c) {
    uint d = D[tb];
    return wrap_coord(c, (d >> 12u) & 15u, ((d >> 20u) & 1u) != 0u, ((d >> 22u) & 1u) != 0u, int(D[tb + 3u]));
}
int wrap_t(uint tb, int c) {
    uint d = D[tb];
    return wrap_coord(c, (d >> 16u) & 15u, ((d >> 21u) & 1u) != 0u, ((d >> 23u) & 1u) != 0u, int(D[tb + 4u]));
}

// fetch_wrapped() of raster.hpp.
uint fetch_wrapped(uint tb, uint tm, uint tlut_type, int is, int it) {
    uint d = D[tb];
    uint format = d & 15u, size = (d >> 4u) & 15u, palette = (d >> 8u) & 15u;
    uint base = D[tb + 1u], stride = D[tb + 2u];
    uint uis = uint(is), uit = uint(it);
    if (size == 3u) {
        uint off = base + (uit * stride + uis * 2u);
        uint wa = off / 8u;
        if ((uit & 1u) != 0u && wa < 512u && tmem_dxt(tm, wa)) off ^= 4u;
        if (off + 0x801u < 4096u) {
            uint r = tmem_byte(tm, off), g = tmem_byte(tm, off + 1u);
            uint b = tmem_byte(tm, off + 0x800u), a = tmem_byte(tm, off + 0x801u);
            return (a << 24u) | (r << 16u) | (g << 8u) | b;
        }
    } else if (size == 2u) {
        uint off = base + (uit * stride + uis * 2u);
        uint wa = off / 8u;
        if ((uit & 1u) != 0u && wa < 512u && tmem_dxt(tm, wa)) off ^= 4u;
        if (off + 1u < 4096u) {
            uint p = (tmem_byte(tm, off) << 8u) | tmem_byte(tm, off + 1u);
            if (format == 3u) return ia_argb((p >> 8u) & 0xFFu, p & 0xFFu);
            return rgba16_to_argb(p);
        }
    } else if (size == 1u) {
        uint off = base + (uit * stride + uis);
        uint wa = off / 8u;
        if ((uit & 1u) != 0u && wa < 512u && tmem_dxt(tm, wa)) off ^= 4u;
        if (off < 4096u) {
            uint val = tmem_byte(tm, off);
            if (format == 2u) {
                if (tlut_type == 0u) return 0xFF000000u | (val << 16u) | (val << 8u) | val;
                return lookup_tlut(tm, val, tlut_type);
            } else if (format == 3u) {
                return ia_argb(((val >> 4u) & 0xFu) * 17u, (val & 0xFu) * 17u);
            }
            return ia_argb(val, val);
        }
    } else if (size == 0u) {
        uint off = base + (uit * stride + (uis / 2u));
        uint wa = off / 8u;
        if ((uit & 1u) != 0u && wa < 512u && tmem_dxt(tm, wa)) off ^= 4u;
        if (off < 4096u) {
            uint byte_val = tmem_byte(tm, off);
            uint val = (uis & 1u) != 0u ? (byte_val & 0xFu) : ((byte_val >> 4u) & 0xFu);
            if (format == 2u) {
                if (tlut_type == 0u) {
                    uint i = val * 17u;
                    return ia_argb(i, i);
                }
                return lookup_tlut(tm, palette * 16u + val, tlut_type);
            } else if (format == 3u) {
                return ia_argb(((val >> 1u) & 7u) * 255u / 7u, (val & 1u) != 0u ? 255u : 0u);
            }
            uint i = val * 17u;
            return ia_argb(i, i);
        }
    }
    return 0u;
}

uint fetch_texel(uint tb, uint tm, uint tlut_type, int is, int it) {
    return fetch_wrapped(tb, tm, tlut_type, wrap_s(tb, is), wrap_t(tb, it));
}

