#include "hires.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {

// write_pixel() of the RDP, for a high-resolution buffer: scissor, alpha
// compare, depth test and blending, then an 8-bit-per-channel store. Pixels
// of 16-bit colour images keep only a coverage bit of alpha, as in RDRAM.
struct HiResSink {
    const DrawState& st;
    u32* color;
    f32* depth;
    u32 width, height;        // pixels (row stride = width)
    u32 sx0, sx1, sy0, sy1;   // scissor [sx0, sx1) x [sy0, sy1)
    bool is16;

    HiResSink(const DrawState& s, HiResTarget& t, HiResDepth* d, u32 scale)
        : st(s), color(t.color.data()), depth(d ? d->z.data() : nullptr) {
        width = s.fb_w * scale;
        height = kFbLines * scale;
        u32 eff_lrx = (s.scissor_lrx > s.scissor_ulx) ? s.scissor_lrx : s.fb_w;
        u32 eff_lry = (s.scissor_lry > s.scissor_uly) ? s.scissor_lry : kFbLines;
        sx0 = s.scissor_ulx * scale;
        sx1 = eff_lrx * scale;
        sy0 = s.scissor_uly * scale;
        sy1 = eff_lry * scale;
        is16 = s.fb_size == 2;
    }

    void write(u32 x, u32 y, u32 c, f32 z, u8 shade_a = 255) {
        if (x >= width || y >= height) return;
        if (x < sx0 || x >= sx1 || y < sy0 || y >= sy1) return;
        if (st.alpha_from_cvg) c |= 0xFF000000u;
        u8 a = (c >> 24) & 0xFF;
        if (st.alpha_compare == 1) {
            if (a < st.alpha_threshold) return;
        } else if (st.alpha_compare == 3) {
            if (a == 0) return;
        }
        if (a == 0 && st.alpha_zero_kill) return;
        const size_t idx = static_cast<size_t>(y) * width + x;
        if (st.z_compare && raster::depth_fails(st, z, depth[idx])) return;
        if (st.z_update) depth[idx] = z;
        u8 r = (c >> 16) & 0xFF, g = (c >> 8) & 0xFF, b = c & 0xFF;
        const bool blend = st.blend_enabled && a < 255;
        if (blend || !st.blend_pass_through) {
            u32 d = color[idx];
            raster::blend_pixel(st, c, shade_a, (d >> 16) & 0xFF, (d >> 8) & 0xFF, d & 0xFF, (d >> 24) & 0xFF, blend, r, g, b);
        }
#ifdef HIRES_EXACT_TEST
        // Test builds: keep exactly what RDRAM holds, so that at scale 1 the
        // composed image must match the native one bit for bit.
        if (is16) {
            u32 p = (((r * 31 / 255) & 0x1F) << 11) | (((g * 31 / 255) & 0x1F) << 6) | (((b * 31 / 255) & 0x1F) << 1) | (a > 0 ? 1 : 0);
            color[idx] = fb_pixel_to_argb(p, 2);
            return;
        }
#endif
        u32 out_a = is16 ? (a > 0 ? 255u : 0u) : a;
        color[idx] = (out_a << 24) | (static_cast<u32>(r) << 16) | (static_cast<u32>(g) << 8) | b;
    }
};

// Fills pixel columns [x0, x1) of one row.
inline void fill_span(u32* row, u32 x0, u32 x1, u32 v) {
    std::fill(row + x0, row + x1, v);
}

} // namespace

// ---------------------------------------------------------------------------
// Arena

void* CpuHiResRenderer::Arena::alloc(size_t bytes) {
    bytes = (bytes + 15) & ~size_t(15);
    used_ += bytes;
    for (;;) {
        if (cur_ < blocks_.size()) {
            if (off_ + bytes <= sizes_[cur_]) {
                void* p = blocks_[cur_].get() + off_;
                off_ += bytes;
                return p;
            }
            if (cur_ + 1 < blocks_.size()) {
                ++cur_;
                off_ = 0;
                continue;
            }
        }
        blocks_.emplace_back(new u8[std::max(kBlock, bytes)]);
        sizes_.push_back(std::max(kBlock, bytes));
        cur_ = blocks_.size() - 1;
        off_ = 0;
    }
}

void CpuHiResRenderer::Arena::reset() {
    // Keep a few standard blocks for the next use; oversized ones go.
    std::vector<std::unique_ptr<u8[]>> keep;
    for (size_t i = 0; i < blocks_.size(); ++i)
        if (sizes_[i] == kBlock && keep.size() < 8) keep.push_back(std::move(blocks_[i]));
    blocks_ = std::move(keep);
    sizes_.assign(blocks_.size(), kBlock);
    cur_ = 0;
    off_ = 0;
    used_ = 0;
}

// ---------------------------------------------------------------------------
// Setup / teardown

CpuHiResRenderer::CpuHiResRenderer(u32 scale) : HiResRenderer(scale) {
    for (Segment& s : segs_) s.chunks = std::make_unique<std::unique_ptr<Cmd[]>[]>(kMaxChunks);
    segs_[0].end.store(~0ull, std::memory_order_relaxed);

    // One worker per core except the emulation thread's. The workers sleep
    // whenever there is nothing to draw.
    const unsigned hw = std::thread::hardware_concurrency();
    nworkers_ = std::clamp(hw > 1 ? static_cast<int>(hw) - 1 : 1, 1, 12);

    // Several bands per thread keep the load balanced when one part of the
    // screen is much more expensive than the rest (and when some cores are
    // slower than others).
    const s32 rows = static_cast<s32>(kFbLines * scale_);
    nbands_ = std::clamp((nworkers_ + 1) * 6, 1, rows / 4);
    bands_ = std::make_unique<Band[]>(nbands_);
    for (int b = 0; b < nbands_; ++b) {
        bands_[b].y0 = rows * b / nbands_;
        bands_[b].y1 = rows * (b + 1) / nbands_;
    }
    for (int i = 0; i < nworkers_; ++i) workers_.emplace_back(&CpuHiResRenderer::worker_main, this, i);
}

CpuHiResRenderer::~CpuHiResRenderer() {
    quit_.store(true, std::memory_order_release);
    { std::lock_guard<std::mutex> lk(mutex_); }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
}

// ---------------------------------------------------------------------------
// Command storage

const CpuHiResRenderer::Cmd& CpuHiResRenderer::cmd_at(u64 i, int& hint) const {
    for (int k = 0; k < kSegments; ++k) {
        const int si = (hint + k) % kSegments;
        const Segment& s = segs_[si];
        // End before start: rotate() stores a reused segment's new start
        // before its new end. A segment's old range is never looked up once
        // it may be reused, since every band has executed it by then.
        const u64 end = s.end.load(std::memory_order_acquire);
        const u64 start = s.start.load(std::memory_order_acquire);
        if (i >= start && i < end) {
            hint = si;
            const u64 local = i - start;
            return s.chunks[local >> kChunkBits][local & (kChunkSize - 1)];
        }
    }
    std::abort(); // every published command belongs to a segment
}

void CpuHiResRenderer::reserve() {
    const Segment& s = segs_[cur_];
    if (count_ + 1 - s.start.load(std::memory_order_relaxed) >= static_cast<u64>(kMaxChunks) * kChunkSize ||
        s.arena.used() > kMaxArenaBytes) {
        rotate();
    }
    Segment& r = segs_[cur_];
    const u64 chunk = (count_ - r.start.load(std::memory_order_relaxed)) >> kChunkBits;
    if (!r.chunks[chunk]) r.chunks[chunk].reset(new Cmd[kChunkSize]);
}

CpuHiResRenderer::Cmd& CpuHiResRenderer::append(CmdType type, HiResTarget* t, const DrawState* st, HiResDepth* d) {
    const Segment& s = segs_[cur_];
    const u64 local = count_ - s.start.load(std::memory_order_relaxed);
    Cmd& c = s.chunks[local >> kChunkBits][local & (kChunkSize - 1)];
    c.type = type;
    c.target = t;
    c.st = st;
    c.depth = d;
    if (t) t->last_cmd = count_ + 1;
    if (d) d->last_cmd = count_ + 1;
    return c;
}

void CpuHiResRenderer::publish() {
    ++count_;
    published_.store(count_, std::memory_order_release);
    // Waking the workers per command would cost more than drawing small
    // triangles; they pick up batches instead.
    if (count_ - notified_ >= 256) flush();
}

void CpuHiResRenderer::flush() {
    if (notified_ == count_) return;
    notified_ = count_;
    { std::lock_guard<std::mutex> lk(mutex_); }
    cv_.notify_all();
}

void CpuHiResRenderer::rotate() {
    flush();
    const int next = (cur_ + 1) % kSegments;
    Segment& n = segs_[next];
    // Everything the segment held last time round must have been executed.
    wait_executed(n.end.load(std::memory_order_relaxed));
    n.arena.reset();
    n.start.store(count_, std::memory_order_release);
    n.end.store(~0ull, std::memory_order_release);
    segs_[cur_].end.store(count_, std::memory_order_release);
    cur_ = next;
    rec_state_ = nullptr;
    rec_serial_ = rec_tmem_gen_ = ~0ull;
    tmem_copy_ = nullptr;
    tmem_copy_gen_ = ~0ull;
    tex_caches_.clear(); // they live in the previous segment's arena
}

// ---------------------------------------------------------------------------
// Recording

HiResTarget* CpuHiResRenderer::find(u32 addr, u32 width, u8 size) const {
    for (const auto& t : targets_)
        if (t->addr == addr && t->width == width && t->size == size) return t.get();
    return nullptr;
}

void hires::scan_changes(HiResTarget& t, const u8* rdram, size_t rdram_size, std::vector<ChangedPixel>& changed) {
    changed.clear();
    const u32 bpp = t.size == 2 ? 2 : 4;
    for (u32 y = 0; y < kFbLines; ++y) {
        const size_t row = static_cast<size_t>(t.addr) + static_cast<size_t>(y) * t.width * bpp;
        if (row + static_cast<size_t>(t.width) * bpp > rdram_size) break;
        u32* sh = t.shadow.data() + static_cast<size_t>(y) * t.width;
        for (u32 x = 0; x < t.width; ++x) {
            const u32 raw = fb_read_pixel(rdram + row + x * bpp, bpp);
            if (raw != sh[x]) {
                sh[x] = raw;
                changed.push_back({static_cast<u16>(x), static_cast<u16>(y), fb_pixel_to_argb(raw, bpp)});
            }
        }
    }
}

HiResTarget* CpuHiResRenderer::bind(u32 addr, u32 width, u8 size, const u8* rdram, size_t rdram_size) {
    if (bound_ && bound_->addr == addr && bound_->width == width && bound_->size == size) return bound_;
    if (width == 0 || width > kMaxWidth || (size != 2 && size != 3)) return nullptr;

    const u32 S = scale_;
    const u32 bpp = size == 2 ? 2 : 4;
    const u32 hr_w = width * S;
    HiResTarget* t = find(addr, width, size);
    if (!t) {
        // New buffer: no command refers to it yet, so fill it in directly from RDRAM.
        auto nt = std::make_unique<HiResTarget>();
        nt->addr = addr;
        nt->width = width;
        nt->size = size;
        nt->color.assign(static_cast<size_t>(hr_w) * kFbLines * S, 0);
        nt->shadow.assign(static_cast<size_t>(width) * kFbLines, 0);
        for (u32 y = 0; y < kFbLines; ++y) {
            const size_t row = static_cast<size_t>(addr) + static_cast<size_t>(y) * width * bpp;
            if (row + static_cast<size_t>(width) * bpp > rdram_size) break;
            for (u32 x = 0; x < width; ++x) {
                u32 raw = fb_read_pixel(rdram + row + x * bpp, bpp);
                nt->shadow[static_cast<size_t>(y) * width + x] = raw;
                u32 argb = fb_pixel_to_argb(raw, bpp);
                for (u32 j = 0; j < S; ++j)
                    fill_span(nt->color.data() + (static_cast<size_t>(y) * S + j) * hr_w, x * S, x * S + S, argb);
            }
        }
        t = nt.get();
        targets_.push_back(std::move(nt));
    } else {
        // Copy in the pixels something other than the RDP changed since it last drew here.
        std::vector<UploadPx>& changed = changed_;
        hires::scan_changes(*t, rdram, rdram_size, changed);
        if (!changed.empty()) {
            reserve();
            auto* items = static_cast<UploadPx*>(arena().alloc(changed.size() * sizeof(UploadPx)));
            std::memcpy(items, changed.data(), changed.size() * sizeof(UploadPx));
            Cmd& c = append(CmdType::Upload, t, nullptr, nullptr);
            c.list = {items, static_cast<u32>(changed.size())};
            publish();
        }
    }
    t->last_used = frame_;
    bound_ = t;
    return t;
}

DrawState* CpuHiResRenderer::recorded_state(const DrawState& st, u64 serial, u64 tmem_gen, bool needs_tmem) {
    if (rec_state_ && rec_serial_ == serial && (!needs_tmem || rec_tmem_gen_ == tmem_gen)) return rec_state_;
    DrawState* s = arena().make(st);
    // The native pass's decoded textures (RDP::attach_native_tex_cache) are
    // only valid until its next TMEM load; workers get their own.
    for (raster::TexUnit& tu : s->tex) tu.cache = nullptr;
    if (needs_tmem) {
        if (!tmem_copy_ || tmem_copy_gen_ != tmem_gen) {
            auto* copy = static_cast<TmemCopy*>(arena().alloc(sizeof(TmemCopy)));
            std::memcpy(copy->data.data(), st.tmem, 4096);
            std::memcpy(copy->dxt.data(), st.tmem_dxt, 512 * sizeof(bool));
            tmem_copy_ = copy;
            tmem_copy_gen_ = tmem_gen;
        }
        s->tmem = tmem_copy_->data.data();
        s->tmem_dxt = tmem_copy_->dxt.data();
        rec_tmem_gen_ = tmem_gen;
    } else {
        s->tmem = nullptr;
        s->tmem_dxt = nullptr;
        rec_tmem_gen_ = ~0ull;
    }
    rec_state_ = s;
    rec_serial_ = serial;
    return s;
}

// Gives a recorded state's tile a decoded-texel table, shared by every state
// of this segment that samples the same TMEM contents through the same tile.
// Workers only read tex[tile].cache for draws that use the tile, and those
// are published after this.
void CpuHiResRenderer::attach_tex_cache(DrawState* rs, u32 tile, u64 tmem_gen) {
    raster::TexUnit& tu = rs->tex[tile & 7];
    if (tu.cache || !rs->tmem) return;
    u32 w = 0, h = 0;
    raster::tex_cache_dims(tu, w, h);
    if (static_cast<u64>(w) * h > 256 * 256) return; // too big to be worth decoding
    TexKey key{tmem_gen, 0, 0};
    raster::tex_cache_key(tu, rs->tlut_type, key.a, key.b);
    auto it = tex_caches_.find(key);
    if (it == tex_caches_.end()) {
        auto* c = new (arena().alloc(sizeof(TexCache))) TexCache();
        c->w = w;
        c->h = h;
        c->texels = static_cast<u32*>(arena().alloc(static_cast<size_t>(w) * h * sizeof(u32)));
        it = tex_caches_.emplace(key, c).first;
    }
    tu.cache = it->second;
}

HiResDepth* CpuHiResRenderer::depth_for(const DrawState& st) {
    if (!st.z_compare && !st.z_update) return nullptr;
    for (auto& d : depths_) {
        if (d->width == st.fb_w) {
            d->last_used = frame_;
            return d.get();
        }
    }
    // A new plane isn't referenced by any recorded command yet.
    auto d = std::make_unique<HiResDepth>();
    d->width = st.fb_w;
    d->z.assign(static_cast<size_t>(st.fb_w) * scale_ * kFbLines * scale_, 1e30f);
    d->last_used = frame_;
#ifdef HIRES_EXACT_TEST
    std::fprintf(stderr, "[hires] frame %llu new depth plane w=%u\n", (unsigned long long)frame_, st.fb_w);
#endif
    depths_.push_back(std::move(d));
    return depths_.back().get();
}

void CpuHiResRenderer::triangle(HiResTarget* t, const DrawState& st, u64 serial, u64 tmem_gen,
                             const Vertex& v0, const Vertex& v1, const Vertex& v2, f32 area) {
    reserve();
    DrawState* rs = recorded_state(st, serial, tmem_gen, st.texture_enabled);
    if (st.texture_enabled) {
        if (!st.combine_set || st.need_tex0) attach_tex_cache(rs, st.active_tile, tmem_gen);
        if (st.combine_set && st.need_tex1) attach_tex_cache(rs, st.active_tile + 1, tmem_gen);
    }
    Cmd& c = append(CmdType::Triangle, t, rs, depth_for(st));
    const Vertex* src[3] = {&v0, &v1, &v2};
    for (int i = 0; i < 3; ++i) {
        const Vertex& v = *src[i];
        c.tri.v[i] = {v.sx, v.sy, v.sz, v.w, v.u, v.v, v.r, v.g, v.b, v.a};
    }
    c.tri.area = area;
    publish();
}

void CpuHiResRenderer::tex_rect(HiResTarget* t, const DrawState& st, u64 serial, u64 tmem_gen, u32 ulx, u32 uly,
                             u32 lrx, u32 lry, u32 tile, f32 s, f32 tc, f32 dsdx, f32 dtdy, bool flip) {
    reserve();
    DrawState* rs = recorded_state(st, serial, tmem_gen, true);
    const bool combined = st.combine_set && !st.copy_mode;
    if (!combined || st.need_tex0) attach_tex_cache(rs, tile, tmem_gen);
    if (combined && st.need_tex1) attach_tex_cache(rs, tile + 1, tmem_gen);
    Cmd& c = append(CmdType::TexRect, t, rs, depth_for(st));
    c.rect = {ulx, uly, lrx, lry, tile, s, tc, dsdx, dtdy, flip};
    publish();
}

void CpuHiResRenderer::fill_rect(HiResTarget* t, u32 x0, u32 y0, u32 x1, u32 y1, u32 argb) {
    reserve();
    Cmd& c = append(CmdType::Fill, t, nullptr, nullptr);
    c.fill = {x0, y0, x1, y1, argb};
    publish();
}

void CpuHiResRenderer::pixels(HiResTarget* t, const DrawState& st, u64 serial, const std::vector<Pixel>& px) {
    if (px.empty()) return;
    reserve();
    const DrawState* rs = recorded_state(st, serial, 0, false);
    auto* items = static_cast<Pixel*>(arena().alloc(px.size() * sizeof(Pixel)));
    std::memcpy(items, px.data(), px.size() * sizeof(Pixel));
    Cmd& c = append(CmdType::Pixels, t, rs, depth_for(st));
    c.list = {items, static_cast<u32>(px.size())};
    publish();
}

void CpuHiResRenderer::blit(HiResTarget* t, const DrawState& st, u64 serial, u32 x0, u32 y0, u32 w, u32 h,
                         const u32* colors) {
    if (w == 0 || h == 0) return;
    reserve();
    const DrawState* rs = recorded_state(st, serial, 0, false);
    auto* copy = static_cast<u32*>(arena().alloc(static_cast<size_t>(w) * h * sizeof(u32)));
    std::memcpy(copy, colors, static_cast<size_t>(w) * h * sizeof(u32));
    Cmd& c = append(CmdType::Blit, t, rs, depth_for(st));
    c.blit = {copy, x0, y0, w, h};
    publish();
}

void CpuHiResRenderer::clear_depth() {
    if (depths_.empty()) return;
    reserve();
    auto** planes = static_cast<HiResDepth**>(arena().alloc(depths_.size() * sizeof(HiResDepth*)));
    for (size_t i = 0; i < depths_.size(); ++i) {
        planes[i] = depths_[i].get();
        planes[i]->last_cmd = count_ + 1;
    }
    Cmd& c = append(CmdType::DepthClear, nullptr, nullptr, nullptr);
    c.clear = {planes, static_cast<u32>(depths_.size())};
    publish();
}

// ---------------------------------------------------------------------------
// Execution

bool CpuHiResRenderer::run_bands(int first, u64 limit) {
    bool worked = false;
    const u64 end = std::min(limit, published_.load(std::memory_order_acquire));
    for (int k = 0; k < nbands_; ++k) {
        if (quit_.load(std::memory_order_relaxed)) break;
        Band& b = bands_[(first + k) % nbands_];
        if (b.next.load(std::memory_order_acquire) >= end) continue;
        bool expected = false;
        if (!b.busy.compare_exchange_strong(expected, true, std::memory_order_acquire)) continue;
        u64 i = b.next.load(std::memory_order_relaxed);
        for (; i < end; ++i) execute(cmd_at(i, b.seg), b.y0, b.y1);
        b.next.store(i, std::memory_order_release);
        b.busy.store(false, std::memory_order_release);
        worked = true;
    }
    return worked;
}

void CpuHiResRenderer::worker_main(int index) {
    const int first = index * nbands_ / nworkers_;
    while (!quit_.load(std::memory_order_acquire)) {
        const u64 seen = published_.load(std::memory_order_acquire);
        if (run_bands(first, ~0ull)) continue;
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait(lk, [&] {
            return quit_.load(std::memory_order_acquire) || published_.load(std::memory_order_acquire) != seen;
        });
    }
}

u64 CpuHiResRenderer::executed() const {
    u64 n = ~0ull;
    for (int b = 0; b < nbands_; ++b) n = std::min(n, bands_[b].next.load(std::memory_order_acquire));
    return n;
}

void CpuHiResRenderer::wait_executed(u64 n) {
    if (executed() >= n) return;
    flush();
    while (executed() < n) {
        if (!run_bands(0, n)) std::this_thread::yield();
    }
}

void CpuHiResRenderer::execute(const Cmd& c, s32 y0, s32 y1) const {
    const u32 S = scale_;
    switch (c.type) {
        case CmdType::Triangle: {
            HiResSink sink(*c.st, *c.target, c.depth, S);
            raster::triangle(*c.st, c.tri.v[0], c.tri.v[1], c.tri.v[2], c.tri.area, S, y0, y1, sink);
            break;
        }
        case CmdType::TexRect: {
            HiResSink sink(*c.st, *c.target, c.depth, S);
            const RectData& r = c.rect;
            raster::tex_rect(*c.st, r.ulx, r.uly, r.lrx, r.lry, r.tile, r.s, r.t, r.dsdx, r.dtdy, r.flip, S, y0, y1, sink);
            break;
        }
        case CmdType::Fill: {
            const FillData& f = c.fill;
            const u32 w = c.target->width * S;
            const s32 ya = std::max(static_cast<s32>(f.y0 * S), y0), yb = std::min(static_cast<s32>(f.y1 * S), y1);
            for (s32 y = ya; y < yb; ++y) fill_span(c.target->color.data() + static_cast<size_t>(y) * w, f.x0 * S, f.x1 * S, f.argb);
            break;
        }
        case CmdType::Pixels: {
            HiResSink sink(*c.st, *c.target, c.depth, S);
            const auto* px = static_cast<const Pixel*>(c.list.items);
            for (u32 i = 0; i < c.list.count; ++i) {
                const s32 ya = std::max(static_cast<s32>(px[i].y * S), y0);
                const s32 yb = std::min(static_cast<s32>(px[i].y * S + S), y1);
                for (s32 y = ya; y < yb; ++y)
                    for (u32 x = px[i].x * S; x < px[i].x * S + S; ++x) sink.write(x, static_cast<u32>(y), px[i].color, px[i].z);
            }
            break;
        }
        case CmdType::Blit: {
            HiResSink sink(*c.st, *c.target, c.depth, S);
            const BlitData& b = c.blit;
            const s32 ya = std::max(static_cast<s32>(b.y0 * S), y0);
            const s32 yb = std::min(static_cast<s32>((b.y0 + b.h) * S), y1);
            for (s32 y = ya; y < yb; ++y) {
                const u32* src = b.colors + static_cast<size_t>(y / S - b.y0) * b.w;
                for (u32 x = 0; x < b.w * S; ++x) sink.write(b.x0 * S + x, static_cast<u32>(y), src[x / S], 0.0f);
            }
            break;
        }
        case CmdType::Upload: {
            const auto* px = static_cast<const UploadPx*>(c.list.items);
            const u32 w = c.target->width * S;
            for (u32 i = 0; i < c.list.count; ++i) {
                const s32 ya = std::max(static_cast<s32>(px[i].y * S), y0);
                const s32 yb = std::min(static_cast<s32>(px[i].y * S + S), y1);
                for (s32 y = ya; y < yb; ++y)
                    fill_span(c.target->color.data() + static_cast<size_t>(y) * w, px[i].x * S, px[i].x * S + S, px[i].argb);
            }
            break;
        }
        case CmdType::DepthClear: {
            const s32 yb = std::min(y1, static_cast<s32>(kFbLines * S));
            for (u32 i = 0; i < c.clear.count; ++i) {
                HiResDepth& d = *c.clear.planes[i];
                const size_t w = static_cast<size_t>(d.width) * S;
                if (y0 < yb) std::fill(d.z.data() + y0 * w, d.z.data() + yb * w, 1e30f);
            }
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Output

bool CpuHiResRenderer::compose(const VIScanout& so, const u8* rdram, size_t rdram_size, std::vector<u32>& out,
                            int& out_w, int& out_h) {
    const u32 S = scale_;
    out_w = static_cast<int>(so.width * S);
    out_h = static_cast<int>(kFbLines * S);
    if (so.blank) {
        out.assign(static_cast<size_t>(out_w) * out_h, 0xFF000000u);
        return true;
    }
    u32 first_line = 0;
    HiResTarget* t = hires::find_scanout(targets_, so, first_line);
    if (!t) return false;
    t->last_used = frame_;
    // Only the draws into the displayed buffer have to be finished.
    wait_executed(t->last_cmd);
    out.resize(static_cast<size_t>(out_w) * out_h);

    // Output line y shows target line y + first_line. A pixel comes from the
    // high-resolution buffer while RDRAM still holds what the RDP wrote there.
    const u32 W = t->width, ow = W * S, bpp = so.bpp;
    for (u32 ny = 0; ny < kFbLines; ++ny) {
        u32* out_row = out.data() + static_cast<size_t>(ny) * S * ow;
        const size_t row = static_cast<size_t>(so.fb_base) + static_cast<size_t>(ny) * W * bpp;
        if (row + static_cast<size_t>(W) * bpp > rdram_size) {
            std::fill(out_row, out_row + static_cast<size_t>(S) * ow, 0xFF000000u);
            continue;
        }
        const u8* p = rdram + row;
        const u32 ty = ny + first_line;
        const u32* sh = ty < kFbLines ? t->shadow.data() + static_cast<size_t>(ty) * W : nullptr;
        for (u32 x = 0; x < W; ++x) {
            const u32 raw = fb_read_pixel(p + x * bpp, bpp);
            if (sh && raw == sh[x]) {
                for (u32 j = 0; j < S; ++j) {
                    const u32* src = t->color.data() + (static_cast<size_t>(ty) * S + j) * ow + x * S;
                    u32* dst = out_row + static_cast<size_t>(j) * ow + x * S;
                    for (u32 k = 0; k < S; ++k) dst[k] = src[k] | 0xFF000000u;
                }
            } else {
                const u32 argb = fb_pixel_to_argb(raw, bpp) | 0xFF000000u;
                for (u32 j = 0; j < S; ++j) fill_span(out_row + static_cast<size_t>(j) * ow, x * S, x * S + S, argb);
            }
        }
    }

#ifdef HIRES_EXACT_TEST
    if (S == 1) {
        // Every pixel whose shadow matches RDRAM must hold exactly what RDRAM shows.
        u32 bad = 0;
        for (u32 y = 0; y + first_line < kFbLines; ++y) {
            for (u32 x = 0; x < W; ++x) {
                const size_t a = static_cast<size_t>(so.fb_base) + (static_cast<size_t>(y) * W + x) * bpp;
                if (a + bpp > rdram_size) continue;
                const u32 raw = fb_read_pixel(rdram + a, bpp);
                if (raw != t->shadow[(y + first_line) * W + x]) continue;
                const u32 hv = t->color[(y + first_line) * W + x];
                if ((hv & 0x00FFFFFF) != (fb_pixel_to_argb(raw, bpp) & 0x00FFFFFF)) ++bad;
            }
        }
        if (bad) std::fprintf(stderr, "[hires] frame %llu: %u shadow-matching pixels differ\n", (unsigned long long)frame_, bad);
    }
#endif
    return true;
}

bool CpuHiResRenderer::present(const VIScanout& so, const u8* rdram, size_t rdram_size, VideoFrame& out) {
    int cw = 0, ch = 0;
    if (!compose(so, rdram, rdram_size, composed_, cw, ch)) return false;
    out.gpu.reset();
    so.place(composed_.data(), static_cast<u32>(cw), scale_, out.pixels, out.w, out.h);
    out.scale = static_cast<int>(scale_);
    return true;
}

void CpuHiResRenderer::end_frame() {
    rotate();
    // Colour images and depth planes the game hasn't drawn to or shown for
    // ten seconds are gone (or were one-off render targets). Buffers queued
    // commands still use stay.
    constexpr u64 kIdleFrames = 600;
    constexpr size_t kMaxTargets = 16;
    const u64 done = executed();
    auto removable = [&](u64 last_used, u64 last_cmd) { return frame_ - last_used > kIdleFrames && done >= last_cmd; };
    targets_.erase(std::remove_if(targets_.begin(), targets_.end(),
                                  [&](const auto& t) { return removable(t->last_used, t->last_cmd); }),
                   targets_.end());
    while (targets_.size() > kMaxTargets) {
        auto oldest = targets_.end();
        for (auto it = targets_.begin(); it != targets_.end(); ++it)
            if (done >= (*it)->last_cmd && (oldest == targets_.end() || (*it)->last_used < (*oldest)->last_used)) oldest = it;
        if (oldest == targets_.end()) break;
        targets_.erase(oldest);
    }
    depths_.erase(std::remove_if(depths_.begin(), depths_.end(),
                                 [&](const auto& d) { return removable(d->last_used, d->last_cmd); }),
                  depths_.end());
    bound_ = nullptr;
    ++frame_;
}
