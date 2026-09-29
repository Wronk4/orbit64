// The VI's scan-out as the hardware does it (VI::render_frame_exact).
//
// A CPU port of parallel-rdp's VI stages (extract_vram, vi_fetch, vi_divot,
// vi_scale; Copyright (c) 2020 Themaister, MIT licence - the full notice is
// in rdp_exact.cpp), which reproduce Angrylion's reference VI:
//
//  1. fetch: each frame buffer pixel with its 3-bit coverage (RGBA5551: the
//     alpha bit and the two hidden bits; RGBA8888: alpha bits 5-7);
//  2. anti-aliasing: a pixel with partial coverage is pulled towards the
//     second-lowest/second-highest of its fully covered neighbours; a
//     fully covered one goes through the dither filter, which rebuilds 8
//     bits from 5 using its 3x3 neighbourhood;
//  3. divot: where coverage is partial, the median of three neighbours;
//  4. scaling: X/Y_START + X/Y_ADD * position in 2.10 fixed point, with
//     bilinear filtering in the resampling AA modes;
//  5. gamma (integer square root) and gamma dither.

#include "vi.hpp"
#include "rdp_exact.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cstring>

namespace {

constexpr u32 kCtrlTypeMask = 3;
constexpr u32 kCtrlGammaDither = 1u << 2;
constexpr u32 kCtrlGamma = 1u << 3;
constexpr u32 kCtrlDivot = 1u << 4;
constexpr u32 kCtrlSerrate = 1u << 6;
constexpr u32 kCtrlDitherFilter = 1u << 16;

constexpr int kVSyncNtsc = 525;
constexpr int kHOffsetNtsc = 108, kHOffsetPal = 128;
constexpr int kVOffsetNtsc = 34, kVOffsetPal = 44;
constexpr int kVResNtsc = 480, kVResPal = 576;
constexpr int kScanoutWidth = 640;
constexpr int kVEndNtsc = (kVOffsetNtsc + kVResNtsc) | 1;
constexpr int kVEndPal = (kVOffsetPal + kVResPal) | 1;
constexpr int kMaxLines = kVResPal >> 1;

u32 isqrt(u32 a) {
    u32 op = a, res = 0, one = 1u << 30;
    while (one > op) one >>= 2;
    while (one != 0) {
        if (op >= res + one) {
            op -= res + one;
            res += one << 1;
        }
        res >>= 1;
        one >>= 2;
    }
    return res;
}

struct GammaTables {
    std::array<u8, 256> plain{};
    std::array<u8, 0x4000> dither{};
    GammaTables() {
        for (u32 i = 0; i < 256; ++i) plain[i] = static_cast<u8>(isqrt(i << 6) << 1);
        for (u32 i = 0; i < 0x4000; ++i) dither[i] = static_cast<u8>(isqrt(i) << 1);
    }
};
const GammaTables& gamma_tables() {
    static const GammaTables t;
    return t;
}

// Gamma-dither noise (the hardware's is an unobservable free-running one).
u32 noise(u32 x, u32 y, u32 frame) {
    constexpr u32 kPrime = 1103515245u;
    u32 s[3] = {x, y, frame};
    for (int r = 0; r < 3; ++r) {
        const u32 n0 = ((s[0] >> 8) ^ s[1]) * kPrime, n1 = ((s[1] >> 8) ^ s[2]) * kPrime, n2 = ((s[2] >> 8) ^ s[0]) * kPrime;
        s[0] = n0;
        s[1] = n1;
        s[2] = n2;
    }
    return (s[0] >> 16) & 0xffff;
}

struct Px {
    s32 r, g, b, a; // a: coverage 0..7
};

u32 median3(u32 l, u32 c, u32 r) {
    if (l < c) std::swap(l, c);
    if (c < r) std::swap(c, r);
    if (l < c) std::swap(l, c);
    return c;
}

} // namespace

void VI::render_frame_exact(const u8* rdram, size_t rdram_size, const ExactRdp* rdp, std::vector<u32>& out,
                            int& out_w, int& out_h) {
    const u32 frame = exact_frames_++;
    u32 st = status;
    const int vi_width = static_cast<int>(width & 0xfff);
    const u32 vi_offset = origin & 0xffffff;
    const int v_sync_r = static_cast<int>(v_sync & 0x3ff);
    const bool is_pal = v_sync_r > kVSyncNtsc + 25;
    const int out_lines = is_pal ? kVResPal : kVResNtsc;
    const bool serrate = (st & kCtrlSerrate) != 0;
    const u32 field = frame & 1; // the field being scanned out (interlaced modes)

    out_w = kScanoutWidth;
    out_h = out_lines;
    if (exact_picture_.size() != static_cast<size_t>(out_w) * out_h) exact_picture_.assign(static_cast<size_t>(out_w) * out_h, 0xFF000000u);
    auto finish = [&]() { out = exact_picture_; };
    auto blank = [&]() {
        std::fill(exact_picture_.begin(), exact_picture_.end(), 0xFF000000u);
        finish();
    };

    const bool is_blank = (st & 2) == 0;
    if (vi_offset == 0 || is_blank) return blank();

    // ---- Registers
    int v_start_r = static_cast<int>((v_start >> 16) & 0x3ff);
    int v_end = static_cast<int>(v_start & 0x3ff);
    int y_start = static_cast<int>((y_scale >> 16) & 0xfff);
    const int y_add = static_cast<int>(y_scale & 0xfff);
    const int v_end_max = is_pal ? kVEndPal : kVEndNtsc;
    if (v_end > v_end_max) v_end = v_end_max;
    if (v_start_r > v_end_max) v_start_r = v_end_max;
    const int v_off = is_pal ? kVOffsetPal : kVOffsetNtsc;
    int v_res = (v_end - v_start_r) >> 1;
    int vstart = (v_start_r - v_off) / 2;
    if (vstart < 0) {
        y_start -= y_add * vstart;
        vstart = 0;
    }
    v_res = std::min(v_res, kMaxLines - vstart);

    int x_start = static_cast<int>((x_scale >> 16) & 0xfff);
    const int x_add = static_cast<int>(x_scale & 0xfff);
    int h_start = static_cast<int>((this->h_start >> 16) & 0x3ff);
    int h_end = static_cast<int>(this->h_start & 0x3ff);
    const int h_off = is_pal ? kHOffsetPal : kHOffsetNtsc;
    h_start -= h_off;
    h_end -= h_off;
    bool left_clamp = false, right_clamp = false;
    if (h_start < 0) {
        x_start -= x_add * h_start;
        h_start = 0;
        left_clamp = true;
    }
    if (h_end > kScanoutWidth) {
        h_end = kScanoutWidth;
        right_clamp = true;
    }
    const int h_start_clamp = h_start + (left_clamp ? 0 : 8);
    const int h_end_clamp = h_end - (right_clamp ? 0 : 7);
    const int h_res = h_end - h_start;
    if (h_res <= 0 || h_start >= kScanoutWidth || v_res <= 0) return blank();
    const int max_x = (x_start + h_res * x_add) >> 10;
    const int max_y = (y_start + v_res * y_add) >> 10;

    const u32 aa_mode = (st >> 8) & 3;
    const bool fetch_aa = aa_mode < 2;   // coverage anti-aliasing
    const bool scale_aa = aa_mode < 3;   // bilinear resampling
    const bool divot = (st & kCtrlDivot) != 0;
    const bool dither_filter = (st & kCtrlDitherFilter) != 0;
    const bool gamma = (st & kCtrlGamma) != 0, gamma_dither = (st & kCtrlGammaDither) != 0;
    const bool rgba8888 = (st & kCtrlTypeMask) == 3;
    const bool fetch_bug = y_add < 1024;

    // ---- 1. Fetch (frame buffer x in [-3, max_x + 5], y in [-2, max_y + 3])
    const int fx0 = divot ? -3 : -2, fy0 = -2;
    const int fw = max_x + 2 + 4 + (divot ? 2 : 0), fh = max_y + 1 + 4;
    static thread_local std::vector<Px> vram;
    vram.resize(static_cast<size_t>(fw) * fh);
    const size_t mask16 = (rdram_size >> 1) - 1, mask32 = (rdram_size >> 2) - 1;
    for (int y = 0; y < fh; ++y) {
        for (int x = 0; x < fw; ++x) {
            const s64 lin = static_cast<s64>(y + fy0) * vi_width + (x + fx0);
            Px p{};
            if (rgba8888) {
                const size_t i = static_cast<size_t>(lin + (vi_offset >> 2)) & mask32;
                const u8* q = rdram + i * 4;
                p = {q[0], q[1], q[2], (q[3] >> 5) & 7};
            } else {
                const size_t i = static_cast<size_t>(lin + (vi_offset >> 1)) & mask16;
                const u32 w = (static_cast<u32>(rdram[i * 2]) << 8) | rdram[i * 2 + 1];
                const u32 h = rdp ? rdp->hidden_at(i, static_cast<u16>(w)) : ((((w >> 8) & 1) << 1) | (w & 1));
                p = {static_cast<s32>((w >> 8) & 0xf8), static_cast<s32>((w >> 3) & 0xf8), static_cast<s32>((w << 2) & 0xf8),
                     static_cast<s32>(((w & 1) << 2) | h)};
            }
            if (!fetch_aa) p.a = 7;
            vram[static_cast<size_t>(y) * fw + x] = p;
        }
    }
    auto V = [&](int x, int y) -> const Px& { return vram[static_cast<size_t>(y) * fw + x]; };

    // ---- 2. Anti-aliasing / dither filter; aa[y][x] is frame buffer
    // (x + fx0 + 2, y); `bug` the fetch-bug variant.
    const int aw = max_x + 3 + (divot ? 2 : 0), ah = max_y + 2;
    static thread_local std::vector<Px> aa, aa_bug;
    aa.resize(static_cast<size_t>(aw) * ah);
    aa_bug.resize(fetch_bug ? static_cast<size_t>(aw) * ah : 0);
    for (int y = 0; y < ah; ++y) {
        for (int x = 0; x < aw; ++x) {
            const int px = x + 2, py = y + 2; // in vram coordinates
            const Px& mid = V(px, py);
            Px out_p = mid, out_b = mid;
            if (mid.a != 7) {
                s32 lo[3] = {mid.r, mid.g, mid.b}, hi[3] = {mid.r, mid.g, mid.b};
                s32 slo[3] = {mid.r, mid.g, mid.b}, shi[3] = {mid.r, mid.g, mid.b};
                auto check = [](const Px& c, s32* lo_, s32* hi_, s32* slo_, s32* shi_) {
                    if (c.a != 7) return;
                    const s32 v[3] = {c.r, c.g, c.b};
                    for (int k = 0; k < 3; ++k) {
                        slo_[k] = std::min(slo_[k], std::max(v[k], lo_[k]));
                        shi_[k] = std::max(shi_[k], std::min(v[k], hi_[k]));
                        lo_[k] = std::min(v[k], lo_[k]);
                        hi_[k] = std::max(v[k], hi_[k]);
                    }
                };
                const Px &lu = V(px - 1, py - 1), &ru = V(px + 1, py - 1), &tl = V(px - 2, py), &tr = V(px + 2, py);
                const Px &ld = V(px - 1, py + 1), &rd = V(px + 1, py + 1);
                check(lu, lo, hi, slo, shi);
                check(ru, lo, hi, slo, shi);
                check(tl, lo, hi, slo, shi);
                check(tr, lo, hi, slo, shi);
                s32 blo[3], bhi[3], bslo[3], bshi[3];
                std::memcpy(blo, lo, sizeof lo);
                std::memcpy(bhi, hi, sizeof hi);
                std::memcpy(bslo, slo, sizeof slo);
                std::memcpy(bshi, shi, sizeof shi);
                check(ld, lo, hi, slo, shi);
                check(rd, lo, hi, slo, shi);
                const s32 m[3] = {mid.r, mid.g, mid.b};
                const s32 coeff = 7 - mid.a;
                s32 res[3], resb[3];
                if (fetch_bug) {
                    // In the fetch-bug state the lower neighbours aren't read:
                    // the sides are, twice.
                    check(tl, blo, bhi, bslo, bshi);
                    check(tr, blo, bhi, bslo, bshi);
                    for (int k = 0; k < 3; ++k) {
                        if (m[k] == lo[k]) slo[k] = lo[k];
                        if (m[k] == hi[k]) shi[k] = hi[k];
                        if (m[k] == blo[k]) bslo[k] = blo[k];
                        if (m[k] == bhi[k]) bshi[k] = bhi[k];
                    }
                }
                for (int k = 0; k < 3; ++k) {
                    const u32 off = static_cast<u32>(slo[k] + shi[k] - (m[k] << 1));
                    res[k] = static_cast<s32>((static_cast<u32>(m[k]) + ((off * static_cast<u32>(coeff) + 4u) >> 3)) & 0xff);
                    const u32 offb = static_cast<u32>(bslo[k] + bshi[k] - (m[k] << 1));
                    resb[k] = static_cast<s32>((static_cast<u32>(m[k]) + ((offb * static_cast<u32>(coeff) + 4u) >> 3)) & 0xff);
                }
                out_p = {res[0], res[1], res[2], mid.a};
                out_b = {resb[0], resb[1], resb[2], mid.a};
            } else if (dither_filter) {
                const s32 t[3] = {mid.r >> 3, mid.g >> 3, mid.b >> 3};
                s32 acc[3] = {0, 0, 0};
                auto add = [&](const Px& c, s32* a_) {
                    const s32 v[3] = {c.r >> 3, c.g >> 3, c.b >> 3};
                    for (int k = 0; k < 3; ++k) a_[k] += std::clamp(v[k] - t[k], -1, 1);
                };
                for (int dy = -1; dy <= 0; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) add(V(px + dx, py + dy), acc);
                s32 accb[3] = {acc[0], acc[1], acc[2]};
                add(V(px - 1, py + 1), acc);
                add(V(px + 1, py + 1), acc);
                add(V(px, py + 1), acc);
                add(V(px - 1, py), accb);
                add(V(px + 1, py), accb);
                out_p = {(mid.r & 0xf8) + acc[0], (mid.g & 0xf8) + acc[1], (mid.b & 0xf8) + acc[2], mid.a};
                out_b = {(mid.r & 0xf8) + accb[0], (mid.g & 0xf8) + accb[1], (mid.b & 0xf8) + accb[2], mid.a};
            }
            aa[static_cast<size_t>(y) * aw + x] = out_p;
            if (fetch_bug) aa_bug[static_cast<size_t>(y) * aw + x] = out_b;
        }
    }

    // ---- 3. Divot: output (x, y) is frame buffer (x, y).
    const int dw = max_x + 2, dh = max_y + 2;
    static thread_local std::vector<Px> dv, dv_bug;
    const std::vector<Px>* img = &aa;
    const std::vector<Px>* img_bug = &aa_bug;
    int iw = aw;
    if (divot) {
        dv.resize(static_cast<size_t>(dw) * dh);
        dv_bug.resize(fetch_bug ? dv.size() : 0);
        auto run = [&](const std::vector<Px>& src, std::vector<Px>& dst) {
            for (int y = 0; y < dh; ++y)
                for (int x = 0; x < dw; ++x) {
                    const Px& l = src[static_cast<size_t>(y) * aw + x];
                    const Px& m = src[static_cast<size_t>(y) * aw + x + 1];
                    const Px& r = src[static_cast<size_t>(y) * aw + x + 2];
                    Px o = m;
                    if ((l.a & m.a & r.a) != 7) {
                        o.r = static_cast<s32>(median3(static_cast<u32>(l.r), static_cast<u32>(m.r), static_cast<u32>(r.r)));
                        o.g = static_cast<s32>(median3(static_cast<u32>(l.g), static_cast<u32>(m.g), static_cast<u32>(r.g)));
                        o.b = static_cast<s32>(median3(static_cast<u32>(l.b), static_cast<u32>(m.b), static_cast<u32>(r.b)));
                    }
                    dst[static_cast<size_t>(y) * dw + x] = o;
                }
        };
        run(aa, dv);
        if (fetch_bug) run(aa_bug, dv_bug);
        img = &dv;
        img_bug = &dv_bug;
        iw = dw;
    }
    const int ih = divot ? dh : ah;
    auto D = [&](const std::vector<Px>& v, int x, int y) -> const Px& {
        x = std::clamp(x, 0, iw - 1);
        y = std::clamp(y, 0, ih - 1);
        return v[static_cast<size_t>(y) * iw + x];
    };

    // ---- 4/5. Scale, gamma
    const GammaTables& gt = gamma_tables();
    const int vs = serrate ? vstart * 2 : vstart;
    const int vr = serrate ? v_res * 2 : v_res;
    for (int Y = 0; Y < out_lines; ++Y) {
        // Progressive frames fill both lines of each pair; interlaced ones
        // only the current field's, keeping the other field's.
        int line;
        if (serrate) {
            if (static_cast<u32>(Y & 1) != field) continue;
            line = Y;
        } else {
            line = Y >> 1;
        }
        u32* dst = exact_picture_.data() + static_cast<size_t>(Y) * out_w;
        if (!serrate && (Y & 1)) { // the second showing of a progressive line
            std::memcpy(dst, dst - out_w, sizeof(u32) * static_cast<size_t>(out_w));
            continue;
        }
        const int cy = line - vs;
        if (cy < 0 || cy >= vr) {
            std::fill(dst, dst + out_w, 0xFF000000u);
            continue;
        }
        const int py = serrate ? cy >> 1 : cy;
        for (int X = 0; X < out_w; ++X) {
            if (X < h_start_clamp || X >= h_end_clamp) {
                dst[X] = 0xFF000000u;
                continue;
            }
            const int cx = X - h_start;
            const int x = cx * x_add + x_start;
            const int y = py * y_add + y_start;
            const int bx = x >> 10, by = y >> 10;
            const Px& p00 = D(*img, bx, by);
            s32 c[3] = {p00.r, p00.g, p00.b};
            if (scale_aa) {
                int bug = 0;
                if (fetch_bug) {
                    const int prev_y = (y - y_add) >> 10, next_y = (y + y_add) >> 10;
                    if (py != 0 && by == prev_y && by != next_y) bug = 1;
                }
                const std::vector<Px>& lower = bug ? *img_bug : *img;
                const Px& p10 = D(*img, bx + 1, by);
                const Px& p01 = D(lower, bx, by + 1);
                const Px& p11 = D(lower, bx + 1, by + 1);
                const u32 xf = static_cast<u32>((x >> 5) & 31), yf = static_cast<u32>((y >> 5) & 31);
                auto lerp = [](s32 a, s32 b, u32 l) {
                    return static_cast<s32>((static_cast<u32>(a) + (((static_cast<u32>(b) - static_cast<u32>(a)) * l + 16u) >> 5)) & 0xff);
                };
                const s32 a0[3] = {p00.r, p00.g, p00.b}, a1[3] = {p10.r, p10.g, p10.b};
                const s32 b0[3] = {p01.r, p01.g, p01.b}, b1[3] = {p11.r, p11.g, p11.b};
                for (int k = 0; k < 3; ++k) {
                    const s32 l0 = lerp(a0[k], b0[k], yf), l1 = lerp(a1[k], b1[k], yf);
                    c[k] = lerp(l0, l1, xf);
                }
            }
            if (gamma) {
                if (gamma_dither) {
                    const u32 n = noise(static_cast<u32>(cx), static_cast<u32>(py), frame);
                    const u32 d[3] = {n & 0x3f, (n >> 6) & 0x3f, ((n >> 9) & 0x38) | (n & 7)};
                    for (int k = 0; k < 3; ++k) c[k] = gt.dither[((static_cast<u32>(c[k]) & 0xff) << 6) + d[k]];
                } else {
                    for (int k = 0; k < 3; ++k) c[k] = gt.plain[static_cast<u32>(c[k]) & 0xff];
                }
            } else if (gamma_dither) {
                const u32 n = noise(static_cast<u32>(cx), static_cast<u32>(py), frame);
                for (int k = 0; k < 3; ++k) c[k] = std::min<s32>(c[k] + static_cast<s32>((n >> k) & 1), 0xff);
            }
            dst[X] = 0xFF000000u | (static_cast<u32>(c[0] & 0xff) << 16) | (static_cast<u32>(c[1] & 0xff) << 8) |
                     static_cast<u32>(c[2] & 0xff);
        }
    }
    finish();
}
