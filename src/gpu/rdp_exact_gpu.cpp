#include "rdp_exact_gpu.hpp"
#include "../jit/jit_invalidate.hpp"

#include <SDL3/SDL.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gpu {

namespace {

constexpr u32 kBlock = 64;  // bytes of RDRAM the copy is kept in step by
constexpr u32 kPage = 4096; // compared at once first

// Pairs exact_shade.comp shades in one dispatch at most (768 bytes each).
constexpr u32 kShadeEntries = 1u << 15;
// Walking primitives per submission at most (ExactRdpLanes submits sooner).
constexpr u32 kTailSlots = 1u << 16;
// ORBIT64_EXACT_GPU_PIX=x,y: a log of what the primitives do at that pixel
// (debugging; at internal resolution also at that high-resolution pixel),
// after the tails. ORBIT64_EXACT_PIX / ORBIT64_EXACT_HPIX are the CPU's.
constexpr u32 kDebugWords = 1 + 64 * 16;

// The shaders' uniform blocks (std140: 4-byte scalars).
struct ShadeParams {
    u32 entries, first, count, groups_x, list, scale, pad[2];
};

constexpr int kVariants = Device::kExactVariants;

// The shading variant a primitive's state needs (exact.glsl, EXACT_VARIANT).
int shade_variant(u32 f0) {
    switch (f0 & 3) {
        case 0:
            if (!(f0 & (EF_USES_TEXEL0 | EF_USES_TEXEL1 | EF_USES_PIPELINED_TEXEL1 | EF_USES_LOD))) return 1;
            return (f0 & EF_USES_TEXEL1) ? 0 : 2;
        case 1: return (f0 & EF_USES_PIPELINED_TEXEL1) ? 0 : 3;
        case 2: return 4;
        default: return 5;
    }
}
struct MemoryParams {
    u32 work, work_count, entries, first, mask8, width, groups_x, debug_pixel, debug_at, scale, pad[2];
};
struct ApplyParams {
    u32 blocks, count, bytes, hidden, scale, pad[3];
};
struct InitParams {
    u32 pages, count, scale, pad;
};

constexpr u32 kSlotPage = 1u << EXACT_PAGE_SHIFT; // RDRAM bytes per slot sample

u32 bytes_per_pixel(u32 fmt) {
    switch (fmt) {
        case 0: case 1: return 1; // I4, I8
        case 4: return 4;         // RGBA8888
        default: return 2;
    }
}

u32 debug_pixel() {
    static const u32 p = [] {
        int x = -1, y = -1;
        if (const char* e = std::getenv("ORBIT64_EXACT_GPU_PIX")) std::sscanf(e, "%d,%d", &x, &y);
        return x < 0 ? ~0u : static_cast<u32>(x) | static_cast<u32>(y) << 16;
    }();
    return p;
}

// Sorts and merges [first, second) ranges.
void merge_ranges(std::vector<std::pair<u64, u64>>& r) {
    std::sort(r.begin(), r.end());
    size_t n = 0;
    for (const auto& x : r) {
        if (x.first >= x.second) continue;
        if (n && x.first <= r[n - 1].second) r[n - 1].second = std::max(r[n - 1].second, x.second);
        else r[n++] = x;
    }
    r.resize(n);
}

} // namespace

ExactRdpGpu::ExactRdpGpu(std::shared_ptr<Device> device) : dev_(std::move(device)), gpu_(dev_->get()) {
    stats_.on = std::getenv("ORBIT64_GPU_STATS") != nullptr;
    stats_.last = SDL_GetTicksNS();
    tails_buf_ = make_buffer((kTailSlots * 4 + kDebugWords) * 4, "orbit64 exact tails");
    SDL_GPUTransferBufferCreateInfo ti{};
    ti.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
    ti.size = 16 + kDebugWords * 4;
    tail_down_ = SDL_CreateGPUTransferBuffer(gpu_, &ti);
}

ExactRdpGpu::~ExactRdpGpu() {
    // (What is in flight finishes before the device goes away: SDL releases
    // resources once the GPU is done with them.)
    for (SDL_GPUBuffer* b : {rdram_buf_, hidden_buf_, data_buf_, tails_buf_, shaded_buf_, up_color_, up_hidden_, pages_buf_})
        if (b) SDL_ReleaseGPUBuffer(gpu_, b);
    for (SDL_GPUTransferBuffer* t : {up_, down_, tail_down_, fetch_down_})
        if (t) SDL_ReleaseGPUTransferBuffer(gpu_, t);
}

SDL_GPUBuffer* ExactRdpGpu::make_buffer(u32 bytes, const char* name) {
    SDL_GPUBufferCreateInfo ci{};
    ci.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE;
    ci.size = bytes;
    ci.props = SDL_CreateProperties();
    SDL_SetStringProperty(ci.props, SDL_PROP_GPU_BUFFER_CREATE_NAME_STRING, name);
    SDL_GPUBuffer* b = SDL_CreateGPUBuffer(gpu_, &ci);
    SDL_DestroyProperties(ci.props);
    return b;
}

bool ExactRdpGpu::ensure_memory(size_t rdram_size) {
    if (size_ == rdram_size && rdram_buf_ && hidden_buf_) return true;
    if (rdram_buf_) SDL_ReleaseGPUBuffer(gpu_, rdram_buf_);
    if (hidden_buf_) SDL_ReleaseGPUBuffer(gpu_, hidden_buf_);
    rdram_buf_ = make_buffer(static_cast<u32>(rdram_size), "orbit64 exact rdram");
    hidden_buf_ = make_buffer(static_cast<u32>(rdram_size / 2), "orbit64 exact ninth bits");
    size_ = rdram_size;
    ref_.assign(rdram_size, 0);
    ref_hidden_.assign(rdram_size / 2, 0);
    valid_.assign(rdram_size / kBlock, 0);
    gpu_dirty_.assign(rdram_size / kBlock, 0);
    pre_.assign(rdram_size, 0);
    snap_.assign(rdram_size / kBlock, 0);
    cycle_runs_.clear();
    flag_runs_.clear();
    if (pages_buf_) SDL_ReleaseGPUBuffer(gpu_, pages_buf_);
    pages_buf_ = make_buffer(static_cast<u32>(std::max<size_t>(rdram_size >> EXACT_PAGE_SHIFT, 1)) * 4, "orbit64 exact pages");
    page_slot_.assign(rdram_size >> EXACT_PAGE_SHIFT, EXACT_NO_SLOT);
    drop_slots();
    return rdram_buf_ && hidden_buf_ && tails_buf_ && tail_down_ && pages_buf_;
}

void ExactRdpGpu::drop_slots() {
    std::fill(page_slot_.begin(), page_slot_.end(), EXACT_NO_SLOT);
    slots_used_ = 0;
    new_pages_.clear();
    pages_dirty_ = true;
}

bool ExactRdpGpu::grow_slots(SDL_GPUCopyPass* copy) {
    if (slots_used_ <= slots_cap_ && up_color_ && up_hidden_) return true;
    u32 cap = std::max<u32>(slots_cap_ * 2, 32);
    while (cap < slots_used_) cap *= 2;
    SDL_GPUBuffer* color = make_buffer(cap * slot_bytes(), "orbit64 exact high-resolution colour");
    SDL_GPUBuffer* hidden = make_buffer(cap * slot_bytes() / 2, "orbit64 exact high-resolution ninth bits");
    if (!color || !hidden) {
        if (color) SDL_ReleaseGPUBuffer(gpu_, color);
        if (hidden) SDL_ReleaseGPUBuffer(gpu_, hidden);
        return false;
    }
    if (up_color_ && slots_cap_) {
        SDL_GPUBufferLocation cs{up_color_, 0}, cd{color, 0};
        SDL_CopyGPUBufferToBuffer(copy, &cs, &cd, slots_cap_ * slot_bytes(), false);
        SDL_GPUBufferLocation hs{up_hidden_, 0}, hd{hidden, 0};
        SDL_CopyGPUBufferToBuffer(copy, &hs, &hd, slots_cap_ * slot_bytes() / 2, false);
    }
    if (up_color_) SDL_ReleaseGPUBuffer(gpu_, up_color_);
    if (up_hidden_) SDL_ReleaseGPUBuffer(gpu_, up_hidden_);
    up_color_ = color;
    up_hidden_ = hidden;
    slots_cap_ = cap;
    return true;
}

void ExactRdpGpu::set_scale(u32 scale) {
    scale = std::max<u32>(scale, 1);
    if (scale == scale_) return;
    scale_ = scale;
    for (SDL_GPUBuffer** b : {&up_color_, &up_hidden_})
        if (*b) {
            SDL_ReleaseGPUBuffer(gpu_, *b);
            *b = nullptr;
        }
    slots_cap_ = 0;
    drop_slots();
}

void ExactRdpGpu::protect(u64 lo, u64 hi, const u8* rdram) {
    lo = lo / kBlock * kBlock;
    hi = std::min<u64>((hi + kBlock - 1) / kBlock * kBlock, size_);
    if (lo >= hi) return;
    if (!cycle_runs_.empty() && lo >= cycle_runs_.back().first && hi <= cycle_runs_.back().second) return;
    u64 run = ~0ull;
    auto end_run = [&](u64 a) {
        if (run == ~0ull) return;
        // Writes there are reported (so they come after these primitives),
        // and so are reads (they see them).
        jit::watch_rdram(static_cast<u32>(run), static_cast<u32>(a - run));
        jit::watch_reads(static_cast<u32>(run), static_cast<u32>(a - run), true);
        if (!cycle_runs_.empty() && cycle_runs_.back().second == run) cycle_runs_.back().second = a;
        else cycle_runs_.emplace_back(run, a);
        run = ~0ull;
    };
    for (u64 a = lo; a < hi; a += kBlock) {
        u8& s = snap_[a / kBlock];
        if (s) {
            end_run(a);
            continue;
        }
        std::memcpy(pre_.data() + a, rdram + a, kBlock);
        s = 1;
        if (run == ~0ull) run = a;
    }
    end_run(hi);
}

bool ExactRdpGpu::touches(u64 paddr, u64 len) const {
    if (len == 0 || snap_.empty()) return false;
    const u64 last = std::min<u64>((paddr + len - 1) / kBlock, snap_.size() - 1);
    for (u64 b = paddr / kBlock; b <= last; ++b)
        if (snap_[b] || gpu_dirty_[b]) return true;
    return false;
}

void ExactRdpGpu::invalidate(u64 paddr, u64 len) {
    if (valid_.empty() || len == 0) return;
    const u64 first = paddr / kBlock, last = (paddr + len - 1) / kBlock;
    for (u64 b = first; b <= last && b < valid_.size(); ++b) valid_[b] = 0;
}

void ExactRdpGpu::invalidate_all() {
    std::fill(valid_.begin(), valid_.end(), 0);
    // (the high-resolution copy starts over from RDRAM)
    drop_slots();
}

void ExactRdpGpu::reset_recording() {
    states_.clear();
    prims_.clear();
    spans_.clear();
    tmem_.clear();
    passes_.clear();
    last_state_ = ~0u;
    last_tmem_gen_ = ~0ull;
    last_tmem_ = ~0u;
    tail_slots_ = 0;
    last_tail_slot_ = -1;
}

// ---------------------------------------------------------------------------
// Recording

void ExactRdpGpu::record(const ExactRecord& r) {
    u32 st[EXACT_STATE_WORDS];
    std::memcpy(st, r.state, sizeof st);
    st[ES_TMEM] = 0;
    if (r.tmem) {
        if (r.tmem_gen != last_tmem_gen_ || last_tmem_ == ~0u) {
            last_tmem_ = static_cast<u32>(tmem_.size());
            last_tmem_gen_ = r.tmem_gen;
            tmem_.resize(tmem_.size() + EXACT_TMEM_WORDS);
            std::memcpy(tmem_.data() + last_tmem_, r.tmem, EXACT_TMEM_WORDS * 4);
        }
        st[ES_TMEM] = last_tmem_;
    }
    if (last_state_ == ~0u || std::memcmp(states_.data() + last_state_, st, sizeof st) != 0) {
        last_state_ = static_cast<u32>(states_.size());
        states_.insert(states_.end(), st, st + EXACT_STATE_WORDS);
    }

    // A new pass whenever the colour or the depth image changes.
    const u32 fb_index = st[ES_FB_INDEX], fb_width = st[ES_FB_WIDTH], z_index = st[ES_Z_INDEX];
    const u32 fb_fmt = (st[ES_FLAGS1] >> 24) & 7;
    const bool uses_z = (st[ES_FLAGS0] & (EF_Z_COMPARE | EF_Z_UPDATE)) != 0;
    if (passes_.empty() || passes_.back().fb_index != fb_index || passes_.back().fb_width != fb_width ||
        passes_.back().fb_fmt != fb_fmt || passes_.back().z_index != z_index) {
        Pass p;
        p.first = pending();
        p.fb_index = fb_index;
        p.fb_width = fb_width;
        p.fb_fmt = fb_fmt;
        p.z_index = z_index;
        passes_.push_back(p);
    }
    Pass& pass = passes_.back();
    ++pass.count;
    pass.uses_z |= uses_z;

    u32 pr[EXACT_PRIM_WORDS];
    std::memcpy(pr, r.prim, sizeof pr);
    const u32 rows = pr[EP_ROWS];
    pr[EP_STATE] = last_state_;
    pr[EP_SPANS] = static_cast<u32>(spans_.size());
    if (pr[EP_FLAGS] & (1u << 8)) {
        pr[EP_TAIL_SLOT] = tail_slots_;
        last_tail_slot_ = static_cast<s32>(tail_slots_++);
    }
    // The rows it can touch (EPF_WRAP: a row down; the last walked pixel's
    // may be the row after the last).
    const bool wrap = (pr[EP_FLAGS] & EPF_WRAP) != 0;
    const s32 y_first = static_cast<s32>(pr[EP_Y0]) + (wrap ? 1 : 0);
    s32 y_last = static_cast<s32>(pr[EP_Y0] + rows) - (wrap ? 1 : 2);
    if (pr[EP_FLAGS] & (1u << 8)) y_last = std::max(y_last, static_cast<s32>(pr[EP_TAIL_Y]));
    pass.min_row = std::min(pass.min_row, y_first);
    pass.max_row = std::max(pass.max_row, y_last);
    if (ensure_memory(r.rdram_size) && y_last >= y_first) {
        const u64 bpp = bytes_per_pixel(fb_fmt);
        protect((fb_index + static_cast<u64>(y_first) * fb_width) * bpp,
                (fb_index + static_cast<u64>(y_last + 1) * fb_width) * bpp, r.rdram);
        if (uses_z)
            protect((z_index + static_cast<u64>(y_first) * fb_width) * 2, (z_index + static_cast<u64>(y_last + 1) * fb_width) * 2,
                    r.rdram);
    }
    // (internal resolution: the high-resolution spans follow the native ones)
    const u32 hrows = scale_ > 1 && !(pr[EP_FLAGS] & EPF_MIRROR) ? pr[EP_HROWS] : 0;
    pr[EP_HROWS] = hrows;
    if (hrows) pr[EP_HSPANS] += pr[EP_SPANS];
    prims_.insert(prims_.end(), pr, pr + EXACT_PRIM_WORDS);
    const u32* sp = reinterpret_cast<const u32*>(r.spans);
    spans_.insert(spans_.end(), sp, sp + static_cast<size_t>(rows + hrows) * EXACT_SPAN_WORDS);
}

// ---------------------------------------------------------------------------
// Bins

// The (tile, primitive) pairs of every pass at one scale, by tile and in
// drawing order within a tile - "entries" (in data_, preceded by the word
// offset of primitive 0), per pass a work list (x | y << 16, first entry,
// count per tile with primitives), and the chunks shaded at once.
struct ExactRdpGpu::Binned {
    struct PassWork {
        u32 work_at = 0, work_count = 0, width = 0;
        std::vector<u32> tile_entries; // per work item, for chunking
    };
    struct Segment {
        u32 pass, wi0, wi1;
    };
    // A run of entries exact_shade*.comp shades at once into the shading
    // buffer, and the work items (tiles) exact_memory.comp then finishes -
    // all of a tile's entries in one chunk.
    struct Chunk {
        u32 e0 = 0, e1 = 0;
        std::vector<Segment> segs;
        u32 list_at[kVariants] = {}, list_count[kVariants] = {}; // the pairs by shading variant
    };
    std::vector<PassWork> work;
    std::vector<Chunk> chunks;
    u32 entries_at = 0, total = 0, max_chunk = 1;
};

void ExactRdpGpu::bin(u32 S, u32 prims_at, const std::vector<u8>& variant, Binned& out) {
    out.work.assign(passes_.size(), {});
    u32 total = 0;
    entries_.clear();
    for (size_t pi = 0; pi < passes_.size(); ++pi) {
        const Pass& p = passes_[pi];
        Binned::PassWork& pw = out.work[pi];
        pw.width = p.fb_width * S;
        if (p.max_row < 0 || p.fb_width == 0) continue;
        const s32 width = static_cast<s32>(p.fb_width);
        const u32 tiles_x = (p.fb_width * S + EXACT_TILE_SIZE - 1) / EXACT_TILE_SIZE;
        const u32 tiles_y = ((static_cast<u32>(p.max_row) + 1) * S + EXACT_TILE_SIZE - 1) / EXACT_TILE_SIZE;
        const size_t ntiles = static_cast<size_t>(tiles_x) * tiles_y;
        tile_count_.assign(ntiles, 0);
        pairs_.clear();
        for (u32 k = 0; k < p.count; ++k) {
            const u32 prim = p.first + k;
            const u32* pr = prims_.data() + static_cast<size_t>(prim) * EXACT_PRIM_WORDS;
            const bool walks = S == 1 && (pr[EP_FLAGS] & (1u << 8)) != 0;
            const s32 tail_tx = walks ? static_cast<s32>(pr[EP_TAIL_X]) / static_cast<s32>(EXACT_TILE_SIZE) : -1;
            const s32 tail_ty = walks ? static_cast<s32>(pr[EP_TAIL_Y]) / static_cast<s32>(EXACT_TILE_SIZE) : -1;
            bool tail_done = !walks;
            // Rows in increasing order, each with the pixels it covers (at this
            // scale); a band of tiles at a time.
            s32 band = -1, bx0 = 0, bx1 = -1;
            auto emit_band = [&]() {
                if (band < 0 || bx1 < bx0 || static_cast<u32>(band) >= tiles_y) return;
                for (s32 tx = bx0; tx <= bx1; ++tx) {
                    const u32 t = static_cast<u32>(band) * tiles_x + static_cast<u32>(tx);
                    pairs_.push_back(t);
                    pairs_.push_back(prim);
                    ++tile_count_[t];
                }
                if (band == tail_ty && tail_tx >= bx0 && tail_tx <= bx1) tail_done = true;
            };
            auto add_row = [&](s32 y, s32 x0, s32 x1) {
                const s32 by = y / static_cast<s32>(EXACT_TILE_SIZE);
                if (by != band) {
                    emit_band();
                    band = by;
                    bx0 = 0;
                    bx1 = -1;
                }
                const s32 tx0 = x0 / static_cast<s32>(EXACT_TILE_SIZE), tx1 = x1 / static_cast<s32>(EXACT_TILE_SIZE);
                if (bx1 < bx0) {
                    bx0 = tx0;
                    bx1 = tx1;
                } else {
                    bx0 = std::min(bx0, tx0);
                    bx1 = std::max(bx1, tx1);
                }
            };
            const bool high = S > 1 && !(pr[EP_FLAGS] & EPF_MIRROR);
            if (!high) {
                // Native spans (at a scale, every sample of the pixels they cover).
                const u32* sp = spans_.data() + pr[EP_SPANS];
                const s32 y0 = static_cast<s32>(pr[EP_Y0]), drawn = static_cast<s32>(pr[EP_ROWS]) - 1;
                const bool wrap = (pr[EP_FLAGS] & EPF_WRAP) != 0;
                for (s32 r = 0; r < drawn; ++r) {
                    const u32* sr = sp + static_cast<size_t>(r) * EXACT_SPAN_WORDS;
                    if (!sr[EX_VALID]) continue;
                    s32 x0, x1, y;
                    if (wrap) {
                        // the pixel at x == width, at x = 0 a row down
                        if (static_cast<s32>(sr[EX_START_X]) > width || static_cast<s32>(sr[EX_END_X]) < width) continue;
                        x0 = x1 = 0;
                        y = y0 + r + 1;
                    } else {
                        x0 = std::max(static_cast<s32>(sr[EX_START_X]), 0);
                        x1 = std::min(static_cast<s32>(sr[EX_END_X]), width - 1);
                        y = y0 + r;
                        if (x1 < x0) continue;
                    }
                    for (s32 sub = 0; sub < static_cast<s32>(S); ++sub)
                        add_row(y * static_cast<s32>(S) + sub, x0 * static_cast<s32>(S), x1 * static_cast<s32>(S) + static_cast<s32>(S) - 1);
                }
            } else {
                const u32* sp = spans_.data() + pr[EP_HSPANS];
                const s32 y0 = static_cast<s32>(pr[EP_HY0]), drawn = static_cast<s32>(pr[EP_HROWS]) - 1;
                for (s32 r = 0; r < drawn; ++r) {
                    const u32* sr = sp + static_cast<size_t>(r) * EXACT_SPAN_WORDS;
                    if (!sr[EX_VALID]) continue;
                    const s32 x0 = std::max(static_cast<s32>(sr[EX_START_X]), 0);
                    const s32 x1 = std::min(static_cast<s32>(sr[EX_END_X]), width * static_cast<s32>(S) - 1);
                    if (x1 >= x0) add_row(y0 + r, x0, x1);
                }
            }
            emit_band();
            if (!tail_done && tail_ty >= 0 && static_cast<u32>(tail_ty) < tiles_y && static_cast<u32>(tail_tx) < tiles_x) {
                const u32 t = static_cast<u32>(tail_ty) * tiles_x + static_cast<u32>(tail_tx);
                pairs_.push_back(t);
                pairs_.push_back(prim);
                ++tile_count_[t];
            }
        }
        // Counting sort by tile (stable: each tile's primitives stay in order).
        tile_first_.assign(ntiles, 0);
        u32 at = total;
        for (size_t t = 0; t < ntiles; ++t) {
            tile_first_[t] = at;
            at += tile_count_[t];
        }
        entries_.resize(static_cast<size_t>(at) * 2);
        std::vector<u32> fill(tile_first_);
        for (size_t e = 0; e < pairs_.size(); e += 2) {
            const u32 t = pairs_[e], slot = fill[t]++;
            entries_[static_cast<size_t>(slot) * 2] = pairs_[e + 1];
            entries_[static_cast<size_t>(slot) * 2 + 1] = static_cast<u32>(t % tiles_x) | static_cast<u32>(t / tiles_x) << 16;
        }
        total = at;
        pw.work_at = static_cast<u32>(data_.size());
        for (size_t t = 0; t < ntiles; ++t) {
            if (!tile_count_[t]) continue;
            data_.push_back(static_cast<u32>(t % tiles_x) | static_cast<u32>(t / tiles_x) << 16);
            data_.push_back(tile_first_[t]);
            data_.push_back(tile_count_[t]);
            pw.tile_entries.push_back(tile_count_[t]);
            ++pw.work_count;
        }
    }
    data_.push_back(prims_at);
    out.entries_at = static_cast<u32>(data_.size());
    out.total = total;
    data_.insert(data_.end(), entries_.begin(), entries_.begin() + static_cast<size_t>(total) * 2);

    out.chunks.clear();
    Binned::Chunk cur;
    for (u32 pi = 0; pi < passes_.size(); ++pi) {
        const Binned::PassWork& pw = out.work[pi];
        for (u32 wi = 0; wi < pw.work_count; ++wi) {
            const u32 n = pw.tile_entries[wi];
            const u32 first = data_[pw.work_at + wi * 3 + 1];
            if (cur.e1 > cur.e0 && cur.e1 - cur.e0 + n > kShadeEntries) {
                out.chunks.push_back(std::move(cur));
                cur = Binned::Chunk{};
            }
            if (cur.e1 == cur.e0) cur.e0 = cur.e1 = first;
            if (cur.segs.empty() || cur.segs.back().pass != pi) cur.segs.push_back({pi, wi, wi});
            cur.segs.back().wi1 = wi + 1;
            cur.e1 = first + n;
        }
    }
    if (cur.e1 > cur.e0) out.chunks.push_back(std::move(cur));
    out.max_chunk = 1;
    for (Binned::Chunk& c : out.chunks) {
        out.max_chunk = std::max(out.max_chunk, c.e1 - c.e0);
        for (int v = 0; v < kVariants; ++v) {
            c.list_at[v] = static_cast<u32>(data_.size());
            for (u32 e = c.e0; e < c.e1; ++e)
                if (variant[entries_[static_cast<size_t>(e) * 2]] == v) data_.push_back(e);
            c.list_count[v] = static_cast<u32>(data_.size()) - c.list_at[v];
        }
    }
}

// ---------------------------------------------------------------------------
// Drawing

void ExactRdpGpu::submit(ExactRdp& front, u8* rdram, size_t rdram_size) {
    if (prims_.empty()) return;
    const u64 t0 = SDL_GetTicksNS();
    if (!dev_->exact_ok() || !ensure_memory(rdram_size)) {
        reset_recording();
        return;
    }
    if (front.hidden_epoch() != epoch_) {
        // The ninth bits changed under the copy: start it over (from RDRAM as
        // it is once what is drawing is in).
        sync(front, rdram);
        epoch_ = front.hidden_epoch();
        invalidate_all();
    }

    // ---- What the passes draw into: colour and depth images down to the
    // last row any of their primitives reaches (64-byte blocks).
    ranges_.clear();
    for (const Pass& p : passes_) {
        if (p.max_row < p.min_row) continue;
        const u64 first = static_cast<u64>(p.min_row) * p.fb_width, end = (static_cast<u64>(p.max_row) + 1) * p.fb_width;
        const u64 bpp = bytes_per_pixel(p.fb_fmt);
        ranges_.emplace_back((p.fb_index + first) * bpp, (p.fb_index + end) * bpp);
        if (p.uses_z) ranges_.emplace_back((p.z_index + first) * 2, (p.z_index + end) * 2);
    }
    for (auto& r : ranges_) {
        r.first = std::min<u64>(r.first / kBlock * kBlock, rdram_size);
        r.second = std::min<u64>((r.second + kBlock - 1) / kBlock * kBlock, rdram_size);
    }
    merge_ranges(ranges_);

    // ---- Blocks whose RDRAM (or ninth bits) the GPU doesn't have: runs of
    // (offset, bytes) to upload - of RDRAM as the primitives see it (pre_
    // where protect() kept it). A block drawn into since the last read-back
    // is the GPU's - unless the CPU wrote it since, which needs the GPU's
    // picture first.
    auto source = [&](u64 a) -> const u8* { return snap_[a / kBlock] ? pre_.data() + a : rdram + a; };
    for (int attempt = 0; attempt < 2; ++attempt) {
        runs_.clear();
        bool conflict = false;
        for (const auto& r : ranges_) {
            for (u64 page = r.first; page < r.second && !conflict;) {
                const u64 end = std::min<u64>((page / kPage + 1) * kPage, r.second);
                bool plain = true; // no snapshot in the page: RDRAM compared at once
                for (u64 a = page; a < end && plain; a += kBlock) plain = !snap_[a / kBlock];
                const bool same = plain && std::memcmp(rdram + page, ref_.data() + page, end - page) == 0;
                for (u64 a = page; a < end; a += kBlock) {
                    const size_t b = static_cast<size_t>(a / kBlock);
                    const bool changed = !same && std::memcmp(source(a), ref_.data() + a, kBlock) != 0;
                    if (gpu_dirty_[b]) {
                        if (changed) {
                            conflict = true;
                            break;
                        }
                        continue;
                    }
                    if (valid_[b] && !changed) continue;
                    if (!runs_.empty() && runs_.back().first + runs_.back().second == a) runs_.back().second += kBlock;
                    else runs_.emplace_back(static_cast<u32>(a), kBlock);
                }
                page = end;
            }
            if (conflict) break;
        }
        if (!conflict) break;
        sync(front, rdram);
    }
    u64 up_rdram = 0;
    for (const auto& [a, n] : runs_) {
        for (u64 b = a; b < a + n; b += kBlock) {
            const u8* src = source(b);
            std::memcpy(ref_.data() + b, src, kBlock);
            // (hidden_range() takes all of RDRAM: the block's words where it is)
            front.hidden_range(b / 2, kBlock / 2, src - b, ref_hidden_.data() + b / 2);
        }
        std::fill(valid_.begin() + a / kBlock, valid_.begin() + (a + n) / kBlock, 1);
        up_rdram += n;
    }
    // The snapshots are in; what is recorded from now on sees RDRAM anew.
    // (Reads stay reported until the read-back.)
    for (const auto& r : cycle_runs_) std::fill(snap_.begin() + r.first / kBlock, snap_.begin() + r.second / kBlock, 0);
    flag_runs_.insert(flag_runs_.end(), cycle_runs_.begin(), cycle_runs_.end());
    cycle_runs_.clear();

    // ---- Data: states, primitives, spans, TMEM, then the bins.
    data_.clear();
    const u32 states_at = 0;
    data_.insert(data_.end(), states_.begin(), states_.end());
    const u32 prims_at = static_cast<u32>(data_.size());
    data_.insert(data_.end(), prims_.begin(), prims_.end());
    const u32 spans_at = static_cast<u32>(data_.size());
    data_.insert(data_.end(), spans_.begin(), spans_.end());
    const u32 tmem_at = static_cast<u32>(data_.size());
    data_.insert(data_.end(), tmem_.begin(), tmem_.end());
    for (size_t i = 0; i < states_.size(); i += EXACT_STATE_WORDS) data_[states_at + i + ES_TMEM] += tmem_at;
    for (size_t i = 0; i < prims_.size(); i += EXACT_PRIM_WORDS) {
        data_[prims_at + i + EP_STATE] += states_at;
        data_[prims_at + i + EP_SPANS] += spans_at;
        data_[prims_at + i + EP_HSPANS] += spans_at;
    }

    // ---- Internal resolution: every page drawn into has a slot in the
    // high-resolution copy (new ones start from RDRAM, exact_init.comp).
    const bool up = scale_ > 1;
    if (up) {
        for (const auto& r : ranges_)
            for (u64 page = r.first >> EXACT_PAGE_SHIFT; page <= (r.second - 1) >> EXACT_PAGE_SHIFT; ++page)
                if (page_slot_[page] == EXACT_NO_SLOT) {
                    page_slot_[page] = slots_used_++;
                    new_pages_.push_back(static_cast<u32>(page));
                    pages_dirty_ = true;
                }
    }

    // ---- Bins, natively and (internal resolution) at S times the resolution.
    std::vector<u8> variant(pending());
    for (u32 i = 0; i < pending(); ++i) {
        const u32 so = prims_[static_cast<size_t>(i) * EXACT_PRIM_WORDS + EP_STATE];
        variant[i] = static_cast<u8>(shade_variant(states_[so + ES_FLAGS0]));
    }
    Binned native, high;
    bin(1, prims_at, variant, native);
    if (up) bin(scale_, prims_at, variant, high);
    for (const auto& c : native.chunks)
        for (int v = 0; v < kVariants; ++v) stats_.variant[v] += c.list_count[v];

    // Internal resolution: the changed blocks go through exact_apply.comp -
    // their addresses, bytes and ninth bits - and the new pages through
    // exact_init.comp.
    u32 apply_at = 0, apply_count = 0, apply_bytes_at = 0, apply_hidden_at = 0, init_at = 0;
    if (up) {
        apply_at = static_cast<u32>(data_.size());
        for (const auto& [a, n] : runs_)
            for (u32 o = 0; o < n; o += kBlock) data_.push_back(a + o);
        apply_count = static_cast<u32>(data_.size()) - apply_at;
        apply_bytes_at = static_cast<u32>(data_.size());
        data_.resize(data_.size() + static_cast<size_t>(apply_count) * (kBlock / 4));
        apply_hidden_at = static_cast<u32>(data_.size());
        data_.resize(data_.size() + static_cast<size_t>(apply_count) * (kBlock / 8));
        u32 k = 0;
        for (const auto& [a, n] : runs_)
            for (u32 o = 0; o < n; o += kBlock, ++k) {
                std::memcpy(data_.data() + apply_bytes_at + k * (kBlock / 4), ref_.data() + a + o, kBlock);
                std::memcpy(data_.data() + apply_hidden_at + k * (kBlock / 8), ref_hidden_.data() + (a + o) / 2, kBlock / 2);
            }
        init_at = static_cast<u32>(data_.size());
        data_.insert(data_.end(), new_pages_.begin(), new_pages_.end());
    }

    // ---- Buffers (the ones the GPU may still be using are replaced, SDL
    // releases them when it is done)
    const u32 shaded_bytes = std::max(native.max_chunk, high.max_chunk) * EXACT_TILE_SIZE * EXACT_TILE_SIZE * 12;
    if (shaded_bytes > shaded_bytes_) {
        if (shaded_buf_) SDL_ReleaseGPUBuffer(gpu_, shaded_buf_);
        shaded_bytes_ = std::max<u32>(shaded_bytes + shaded_bytes / 2, 1u << 20);
        shaded_buf_ = make_buffer(shaded_bytes_, "orbit64 exact shading");
    }
    const u32 data_bytes = static_cast<u32>(data_.size() * 4);
    if (data_bytes > data_bytes_) {
        if (data_buf_) SDL_ReleaseGPUBuffer(gpu_, data_buf_);
        data_bytes_ = std::max<u32>(data_bytes + data_bytes / 2, 1u << 20);
        data_buf_ = make_buffer(data_bytes_, "orbit64 exact data");
    }
    const u32 pages_bytes = up && pages_dirty_ ? static_cast<u32>(page_slot_.size() * 4) : 0;
    const u64 up_bytes = (up ? 0 : up_rdram + up_rdram / 2) + data_bytes + 16 + pages_bytes;
    if (up_bytes > up_bytes_) {
        if (up_) SDL_ReleaseGPUTransferBuffer(gpu_, up_);
        up_bytes_ = static_cast<u32>(std::max<u64>(up_bytes + up_bytes / 2, 1u << 20));
        SDL_GPUTransferBufferCreateInfo ti{};
        ti.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        ti.size = up_bytes_;
        up_ = SDL_CreateGPUTransferBuffer(gpu_, &ti);
    }
    SDL_GPUCommandBuffer* cmd = data_buf_ && shaded_buf_ && up_ ? SDL_AcquireGPUCommandBuffer(gpu_) : nullptr;
    u8* mapped = cmd ? static_cast<u8*>(SDL_MapGPUTransferBuffer(gpu_, up_, true)) : nullptr;
    if (!mapped) {
        if (cmd) SDL_CancelGPUCommandBuffer(cmd);
        SDL_Log("GPU bit-exact RDP: submission failed: %s", SDL_GetError());
        invalidate_all();
        reset_recording();
        return;
    }
    // Upload layout: RDRAM runs and their ninth bits (natively), the data, a
    // zero word, the page table.
    u64 off = 0;
    if (!up) {
        for (const auto& [a, n] : runs_) {
            std::memcpy(mapped + off, ref_.data() + a, n);
            off += n;
        }
    }
    const u64 hidden_at = off;
    if (!up) {
        for (const auto& [a, n] : runs_) {
            std::memcpy(mapped + off, ref_hidden_.data() + a / 2, n / 2);
            off += n / 2;
        }
    }
    const u64 data_at = off;
    std::memcpy(mapped + off, data_.data(), data_bytes);
    std::memset(mapped + off + data_bytes, 0, 16);
    const u64 pages_at = off + data_bytes + 16;
    if (pages_bytes) std::memcpy(mapped + pages_at, page_slot_.data(), pages_bytes);
    SDL_UnmapGPUTransferBuffer(gpu_, up_);

    const u32 dbg = debug_pixel();
    const u32 debug_at = kTailSlots * 4;
    {
        SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
        u64 o = 0, ho = hidden_at;
        if (!up)
            for (const auto& [a, n] : runs_) {
                SDL_GPUTransferBufferLocation src{up_, static_cast<u32>(o)};
                SDL_GPUBufferRegion dst{rdram_buf_, a, n};
                SDL_UploadToGPUBuffer(copy, &src, &dst, false);
                SDL_GPUTransferBufferLocation hsrc{up_, static_cast<u32>(ho)};
                SDL_GPUBufferRegion hdst{hidden_buf_, a / 2, n / 2};
                SDL_UploadToGPUBuffer(copy, &hsrc, &hdst, false);
                o += n;
                ho += n / 2;
            }
        if (dbg != ~0u) { // the log's count starts at zero
            SDL_GPUTransferBufferLocation zsrc{up_, static_cast<u32>(data_at + data_bytes)};
            SDL_GPUBufferRegion zdst{tails_buf_, debug_at * 4, 4};
            SDL_UploadToGPUBuffer(copy, &zsrc, &zdst, false);
        }
        SDL_GPUTransferBufferLocation dsrc{up_, static_cast<u32>(data_at)};
        SDL_GPUBufferRegion ddst{data_buf_, 0, data_bytes};
        SDL_UploadToGPUBuffer(copy, &dsrc, &ddst, true);
        if (pages_bytes) {
            SDL_GPUTransferBufferLocation psrc{up_, static_cast<u32>(pages_at)};
            SDL_GPUBufferRegion pdst{pages_buf_, 0, pages_bytes};
            SDL_UploadToGPUBuffer(copy, &psrc, &pdst, false);
            pages_dirty_ = false;
        }
        if (up && !grow_slots(copy)) SDL_Log("GPU bit-exact RDP: no memory for the high-resolution copy");
        SDL_EndGPUCopyPass(copy);
    }

    // Internal resolution: new pages from RDRAM (as the GPU had it), then
    // what changed in RDRAM into it and every sample.
    if (up && (!new_pages_.empty() || apply_count)) {
        SDL_GPUStorageBufferReadWriteBinding rw[4]{};
        rw[0].buffer = rdram_buf_;
        rw[1].buffer = hidden_buf_;
        rw[2].buffer = up_color_;
        rw[3].buffer = up_hidden_;
        SDL_GPUBuffer* ro[2] = {data_buf_, pages_buf_};
        if (!new_pages_.empty()) {
            SDL_GPUComputePass* cp = SDL_BeginGPUComputePass(cmd, nullptr, 0, rw, 4);
            SDL_BindGPUComputePipeline(cp, dev_->exact_init());
            SDL_BindGPUComputeStorageBuffers(cp, 0, ro, 2);
            InitParams ip{init_at, static_cast<u32>(new_pages_.size()), scale_, 0};
            SDL_PushGPUComputeUniformData(cmd, 0, &ip, sizeof ip);
            SDL_DispatchGPUCompute(cp, (ip.count * (kSlotPage / 4) + 63) / 64, 1, 1);
            SDL_EndGPUComputePass(cp);
        }
        if (apply_count) {
            SDL_GPUComputePass* cp = SDL_BeginGPUComputePass(cmd, nullptr, 0, rw, 4);
            SDL_BindGPUComputePipeline(cp, dev_->exact_apply());
            SDL_BindGPUComputeStorageBuffers(cp, 0, ro, 2);
            ApplyParams ap{apply_at, apply_count, apply_bytes_at, apply_hidden_at, scale_, {}};
            SDL_PushGPUComputeUniformData(cmd, 0, &ap, sizeof ap);
            SDL_DispatchGPUCompute(cp, (apply_count * 8 + 63) / 64, 1, 1);
            SDL_EndGPUComputePass(cp);
        }
    }
    new_pages_.clear();

    // ---- Drawing: natively into RDRAM, then the high-resolution copy.
    auto draw = [&](const Binned& bins, u32 scale) {
        SDL_GPUBuffer* mem = scale > 1 ? up_color_ : rdram_buf_;
        SDL_GPUBuffer* hid = scale > 1 ? up_hidden_ : hidden_buf_;
        for (const Binned::Chunk& c : bins.chunks) {
            {
                SDL_GPUStorageBufferReadWriteBinding rw[1]{};
                rw[0].buffer = shaded_buf_;
                rw[0].cycle = true; // (written whole)
                SDL_GPUComputePass* cp = SDL_BeginGPUComputePass(cmd, nullptr, 0, rw, 1);
                SDL_BindGPUComputeStorageBuffers(cp, 0, &data_buf_, 1);
                for (int v = 0; v < kVariants; ++v) {
                    if (!c.list_count[v]) continue;
                    SDL_BindGPUComputePipeline(cp, dev_->exact_shade(v));
                    ShadeParams sp{};
                    sp.entries = bins.entries_at;
                    sp.first = c.e0;
                    sp.count = c.list_count[v];
                    sp.groups_x = std::min<u32>(sp.count, 32768);
                    sp.list = c.list_at[v];
                    sp.scale = scale;
                    SDL_PushGPUComputeUniformData(cmd, 0, &sp, sizeof sp);
                    SDL_DispatchGPUCompute(cp, sp.groups_x, (sp.count + sp.groups_x - 1) / sp.groups_x, 1);
                }
                SDL_EndGPUComputePass(cp);
            }
            for (const Binned::Segment& sg : c.segs) {
                const Binned::PassWork& pw = bins.work[sg.pass];
                SDL_GPUStorageBufferReadWriteBinding rw[3]{};
                rw[0].buffer = mem;
                rw[1].buffer = hid;
                rw[2].buffer = tails_buf_;
                SDL_GPUComputePass* cp = SDL_BeginGPUComputePass(cmd, nullptr, 0, rw, 3);
                SDL_BindGPUComputePipeline(cp, dev_->exact_memory());
                SDL_GPUBuffer* ro[3] = {data_buf_, shaded_buf_, pages_buf_};
                SDL_BindGPUComputeStorageBuffers(cp, 0, ro, 3);
                MemoryParams mp{};
                mp.work = pw.work_at + sg.wi0 * 3;
                mp.work_count = sg.wi1 - sg.wi0;
                mp.entries = bins.entries_at;
                mp.first = c.e0;
                mp.mask8 = static_cast<u32>(rdram_size - 1);
                mp.width = pw.width;
                mp.groups_x = std::min<u32>(mp.work_count, 32768);
                mp.debug_pixel = dbg; // (at the pass's scale)
                mp.debug_at = debug_at;
                mp.scale = scale;
                SDL_PushGPUComputeUniformData(cmd, 0, &mp, sizeof mp);
                SDL_DispatchGPUCompute(cp, mp.groups_x, (mp.work_count + mp.groups_x - 1) / mp.groups_x, 1);
                SDL_EndGPUComputePass(cp);
            }
        }
    };
    draw(native, 1);
    if (up) draw(high, scale_);
    const u32 total_entries = native.total + high.total;
    const size_t nchunks = native.chunks.size() + high.chunks.size();
    // The last walked pixel's memory colour (and the debug log) on its way.
    if (last_tail_slot_ >= 0 || dbg != ~0u) {
        SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
        if (last_tail_slot_ >= 0) {
            SDL_GPUBufferRegion src{tails_buf_, static_cast<u32>(last_tail_slot_) * 16, 16};
            SDL_GPUTransferBufferLocation dst{tail_down_, 0};
            SDL_DownloadFromGPUBuffer(copy, &src, &dst);
            tail_queued_ = true;
        }
        if (dbg != ~0u) {
            SDL_GPUBufferRegion src{tails_buf_, debug_at * 4, kDebugWords * 4};
            SDL_GPUTransferBufferLocation dst{tail_down_, 16};
            SDL_DownloadFromGPUBuffer(copy, &src, &dst);
        }
        SDL_EndGPUCopyPass(copy);
    }
    SDL_SubmitGPUCommandBuffer(cmd);

    for (const auto& r : ranges_) {
        std::fill(gpu_dirty_.begin() + r.first / kBlock, gpu_dirty_.begin() + r.second / kBlock, 1);
        dirty_ranges_.push_back(r);
    }
    unsynced_ = true;

    if (stats_.on) {
        ++stats_.submits;
        stats_.prims += pending();
        stats_.passes += passes_.size();
        stats_.chunks += nchunks;
        stats_.entries += total_entries;
        stats_.up_bytes += up_bytes;
        stats_.cpu_ms += (SDL_GetTicksNS() - t0) / 1e6;
    }
    reset_recording();
}

void ExactRdpGpu::sync(ExactRdp& front, u8* rdram) {
    if (!unsynced_) return;
    const u64 t0 = SDL_GetTicksNS();
    merge_ranges(dirty_ranges_);
    u64 down_total = 0;
    for (const auto& r : dirty_ranges_) down_total += (r.second - r.first) + (r.second - r.first) / 2;
    if (down_total > down_bytes_) {
        if (down_) SDL_ReleaseGPUTransferBuffer(gpu_, down_);
        down_bytes_ = static_cast<u32>(std::max<u64>(down_total + down_total / 2, 1u << 20));
        SDL_GPUTransferBufferCreateInfo ti{};
        ti.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
        ti.size = down_bytes_;
        down_ = SDL_CreateGPUTransferBuffer(gpu_, &ti);
    }
    SDL_GPUCommandBuffer* cmd = down_ ? SDL_AcquireGPUCommandBuffer(gpu_) : nullptr;
    SDL_GPUFence* fence = nullptr;
    if (cmd) {
        SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
        u64 o = 0;
        for (const auto& r : dirty_ranges_) {
            SDL_GPUBufferRegion src{rdram_buf_, static_cast<u32>(r.first), static_cast<u32>(r.second - r.first)};
            SDL_GPUTransferBufferLocation dst{down_, static_cast<u32>(o)};
            SDL_DownloadFromGPUBuffer(copy, &src, &dst);
            o += r.second - r.first;
        }
        for (const auto& r : dirty_ranges_) {
            SDL_GPUBufferRegion src{hidden_buf_, static_cast<u32>(r.first / 2), static_cast<u32>((r.second - r.first) / 2)};
            SDL_GPUTransferBufferLocation dst{down_, static_cast<u32>(o)};
            SDL_DownloadFromGPUBuffer(copy, &src, &dst);
            o += (r.second - r.first) / 2;
        }
        SDL_EndGPUCopyPass(copy);
        fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    }
    const u64 t1 = SDL_GetTicksNS();
    if (fence) {
        SDL_WaitForGPUFences(gpu_, true, &fence, 1);
        SDL_ReleaseGPUFence(gpu_, fence);
    }
    const u64 t2 = SDL_GetTicksNS();
    const u8* got = fence ? static_cast<const u8*>(SDL_MapGPUTransferBuffer(gpu_, down_, false)) : nullptr;
    if (!got) {
        SDL_Log("GPU bit-exact RDP: read-back failed: %s", SDL_GetError());
        invalidate_all();
    } else {
        // What the GPU changed goes to RDRAM - all but bytes the CPU wrote
        // since they were uploaded (RDRAM differs from the copy there), which
        // came later.
        u64 o = 0, ho = 0; // RDRAM bytes, then the ninth bits
        for (const auto& r : dirty_ranges_) ho += r.second - r.first;
        for (const auto& r : dirty_ranges_) {
            for (u64 a = r.first; a < r.second; a += kBlock, o += kBlock, ho += kBlock / 2) {
                const u8* nb = got + o;
                const u8* nh = got + ho;
                u8* ref = ref_.data() + a;
                if (std::memcmp(nb, ref, kBlock) == 0 && std::memcmp(nh, ref_hidden_.data() + a / 2, kBlock / 2) == 0)
                    continue;
                if (std::memcmp(rdram + a, ref, kBlock) == 0) {
                    std::memcpy(rdram + a, nb, kBlock);
                    front.gpu_wrote_hidden(a / 2, kBlock / 2, nh, rdram);
                } else {
                    for (u32 h = 0; h < kBlock / 2; ++h) {
                        if (rdram[a + h * 2] != ref[h * 2] || rdram[a + h * 2 + 1] != ref[h * 2 + 1]) continue;
                        rdram[a + h * 2] = nb[h * 2];
                        rdram[a + h * 2 + 1] = nb[h * 2 + 1];
                        front.gpu_wrote_hidden(a / 2 + h, 1, nh + h, rdram);
                    }
                }
                // A snapshot of the block (protect()) is of a time after these
                // primitives: it gets what they drew the same way.
                if (snap_[a / kBlock]) {
                    u8* p = pre_.data() + a;
                    for (u32 k = 0; k < kBlock; k += 2)
                        if (p[k] == ref[k] && p[k + 1] == ref[k + 1]) {
                            p[k] = nb[k];
                            p[k + 1] = nb[k + 1];
                        }
                }
                std::memcpy(ref, nb, kBlock);
                std::memcpy(ref_hidden_.data() + a / 2, nh, kBlock / 2);
            }
            std::fill(gpu_dirty_.begin() + r.first / kBlock, gpu_dirty_.begin() + r.second / kBlock, 0);
        }
        SDL_UnmapGPUTransferBuffer(gpu_, down_);
    }
    if (tail_queued_ || debug_pixel() != ~0u) {
        if (const auto* t = static_cast<const u8*>(SDL_MapGPUTransferBuffer(gpu_, tail_down_, false))) {
            if (tail_queued_) {
                s32 m[4];
                std::memcpy(m, t, 16);
                front.set_prev_mem(m);
            }
            if (debug_pixel() != ~0u) {
                s32 log[kDebugWords];
                std::memcpy(log, t + 16, sizeof log);
                for (s32 i = 0; i < std::min(log[0], 64); ++i) {
                    const s32* e = log + 1 + i * 16;
                    std::fprintf(stderr,
                                 "GPU PIX x=%u y=%u prim=%d kind=%d cvg=%d z=%05x depth=%04x dz=%x comb=%08x col=%08x dirty=%d\n",
                                 debug_pixel() & 0xffff, debug_pixel() >> 16, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7],
                                 e[8]);
                }
            }
            SDL_UnmapGPUTransferBuffer(gpu_, tail_down_);
        }
    }
    tail_queued_ = false;
    dirty_ranges_.clear();
    unsynced_ = false;
    // RDRAM is all there is again: reads go on unreported - but for what is
    // recorded and not submitted yet.
    for (const auto& r : flag_runs_) jit::watch_reads(static_cast<u32>(r.first), static_cast<u32>(r.second - r.first), false);
    flag_runs_.clear();
    for (const auto& r : cycle_runs_) jit::watch_reads(static_cast<u32>(r.first), static_cast<u32>(r.second - r.first), true);
    if (stats_.on) {
        ++stats_.syncs;
        stats_.down_bytes += down_total;
        stats_.cpu_ms += (t1 - t0) / 1e6 + (SDL_GetTicksNS() - t2) / 1e6;
        stats_.wait_ms += (t2 - t1) / 1e6;
        report_stats();
    }
}

void ExactRdpGpu::fetch_upscaled(u64 lo, u64 hi, UpStore& up) {
    if (scale_ <= 1 || up.scale != scale_ || up.size != size_ || !up_color_ || page_slot_.empty() || hi <= lo) return;
    hi = std::min<u64>(hi, size_);
    std::vector<u32> pages;
    for (u64 page = lo >> EXACT_PAGE_SHIFT; page <= (hi - 1) >> EXACT_PAGE_SHIFT; ++page)
        if (page_slot_[page] != EXACT_NO_SLOT) pages.push_back(static_cast<u32>(page));
    if (pages.empty()) return;
    const u32 sb = slot_bytes();
    const u32 bytes = static_cast<u32>(pages.size()) * (sb + sb / 2);
    if (bytes > fetch_bytes_) {
        if (fetch_down_) SDL_ReleaseGPUTransferBuffer(gpu_, fetch_down_);
        fetch_bytes_ = bytes + bytes / 2;
        SDL_GPUTransferBufferCreateInfo ti{};
        ti.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
        ti.size = fetch_bytes_;
        fetch_down_ = SDL_CreateGPUTransferBuffer(gpu_, &ti);
    }
    SDL_GPUCommandBuffer* cmd = fetch_down_ ? SDL_AcquireGPUCommandBuffer(gpu_) : nullptr;
    if (!cmd) return;
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    for (size_t i = 0; i < pages.size(); ++i) {
        const u32 slot = page_slot_[pages[i]];
        SDL_GPUBufferRegion cs{up_color_, slot * sb, sb};
        SDL_GPUTransferBufferLocation cd{fetch_down_, static_cast<u32>(i) * (sb + sb / 2)};
        SDL_DownloadFromGPUBuffer(copy, &cs, &cd);
        SDL_GPUBufferRegion hs{up_hidden_, slot * (sb / 2), sb / 2};
        SDL_GPUTransferBufferLocation hd{fetch_down_, static_cast<u32>(i) * (sb + sb / 2) + sb};
        SDL_DownloadFromGPUBuffer(copy, &hs, &hd);
    }
    SDL_EndGPUCopyPass(copy);
    SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (!fence) return;
    SDL_WaitForGPUFences(gpu_, true, &fence, 1);
    SDL_ReleaseGPUFence(gpu_, fence);
    const auto* got = static_cast<const u8*>(SDL_MapGPUTransferBuffer(gpu_, fetch_down_, false));
    if (!got) return;
    const u32 samples = scale_ * scale_;
    for (size_t i = 0; i < pages.size(); ++i) {
        const u64 a = static_cast<u64>(pages[i]) << EXACT_PAGE_SHIFT;
        const u8* src = got + i * (sb + sb / 2);
        for (u32 smp = 0; smp < samples; ++smp) {
            std::memcpy(up.color(smp) + a, src + smp * kSlotPage, kSlotPage);
            std::memcpy(up.hidden(smp) + a / 2, src + sb + smp * (kSlotPage / 2), kSlotPage / 2);
        }
        // What RDRAM held as the copy was drawn over it: where the CPU changed
        // it since, the VI's up_sync_before() takes RDRAM.
        std::memcpy(up.ref + a, ref_.data() + a, kSlotPage);
    }
    SDL_UnmapGPUTransferBuffer(gpu_, fetch_down_);
}

void ExactRdpGpu::store_upscaled(ExactRdp& front, u64 lo, u64 hi, const UpStore& up, const u8* rdram) {
    if (scale_ <= 1 || up.scale != scale_ || up.size != size_ || page_slot_.empty() || hi <= lo) return;
    hi = std::min<u64>(hi, size_);
    std::vector<u32> pages;
    for (u64 page = lo >> EXACT_PAGE_SHIFT; page <= (hi - 1) >> EXACT_PAGE_SHIFT; ++page) {
        if (page_slot_[page] == EXACT_NO_SLOT) {
            // (filled whole from `up` below, nothing to start it from)
            page_slot_[page] = slots_used_++;
            pages_dirty_ = true;
        }
        pages.push_back(static_cast<u32>(page));
    }
    // The pages' RDRAM is the GPU's copy's as it is: nothing to repeat into
    // the samples at the next submission.
    for (u32 page : pages) {
        const u64 a = static_cast<u64>(page) << EXACT_PAGE_SHIFT;
        std::memcpy(ref_.data() + a, rdram + a, kSlotPage);
        front.hidden_range(a / 2, kSlotPage / 2, rdram, ref_hidden_.data() + a / 2);
        std::fill(valid_.begin() + a / kBlock, valid_.begin() + (a + kSlotPage) / kBlock, 1);
    }
    const u32 sb = slot_bytes();
    const u64 per_page = static_cast<u64>(sb) + sb / 2 + kSlotPage + kSlotPage / 2;
    const u32 pages_bytes = pages_dirty_ ? static_cast<u32>(page_slot_.size() * 4) : 0;
    const u64 bytes = per_page * pages.size() + pages_bytes;
    if (bytes > up_bytes_) {
        if (up_) SDL_ReleaseGPUTransferBuffer(gpu_, up_);
        up_bytes_ = static_cast<u32>(bytes + bytes / 2);
        SDL_GPUTransferBufferCreateInfo ti{};
        ti.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        ti.size = up_bytes_;
        up_ = SDL_CreateGPUTransferBuffer(gpu_, &ti);
    }
    SDL_GPUCommandBuffer* cmd = up_ ? SDL_AcquireGPUCommandBuffer(gpu_) : nullptr;
    u8* mapped = cmd ? static_cast<u8*>(SDL_MapGPUTransferBuffer(gpu_, up_, true)) : nullptr;
    if (!mapped) {
        if (cmd) SDL_CancelGPUCommandBuffer(cmd);
        invalidate_all();
        return;
    }
    const u32 samples = scale_ * scale_;
    for (size_t i = 0; i < pages.size(); ++i) {
        const u64 a = static_cast<u64>(pages[i]) << EXACT_PAGE_SHIFT;
        u8* dst = mapped + i * per_page;
        for (u32 smp = 0; smp < samples; ++smp) {
            std::memcpy(dst + smp * kSlotPage, up.color(smp) + a, kSlotPage);
            std::memcpy(dst + sb + smp * (kSlotPage / 2), up.hidden(smp) + a / 2, kSlotPage / 2);
        }
        std::memcpy(dst + sb + sb / 2, ref_.data() + a, kSlotPage);
        std::memcpy(dst + sb + sb / 2 + kSlotPage, ref_hidden_.data() + a / 2, kSlotPage / 2);
    }
    if (pages_bytes) std::memcpy(mapped + per_page * pages.size(), page_slot_.data(), pages_bytes);
    SDL_UnmapGPUTransferBuffer(gpu_, up_);
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    if (!grow_slots(copy)) SDL_Log("GPU bit-exact RDP: no memory for the high-resolution copy");
    for (size_t i = 0; i < pages.size(); ++i) {
        const u32 slot = page_slot_[pages[i]];
        const u32 o = static_cast<u32>(i * per_page);
        const u32 a = pages[i] << EXACT_PAGE_SHIFT;
        SDL_GPUTransferBufferLocation src{up_, o};
        SDL_GPUBufferRegion dst{up_color_, slot * sb, sb};
        SDL_UploadToGPUBuffer(copy, &src, &dst, false);
        SDL_GPUTransferBufferLocation hsrc{up_, o + sb};
        SDL_GPUBufferRegion hdst{up_hidden_, slot * (sb / 2), sb / 2};
        SDL_UploadToGPUBuffer(copy, &hsrc, &hdst, false);
        SDL_GPUTransferBufferLocation nsrc{up_, o + sb + sb / 2};
        SDL_GPUBufferRegion ndst{rdram_buf_, a, kSlotPage};
        SDL_UploadToGPUBuffer(copy, &nsrc, &ndst, false);
        SDL_GPUTransferBufferLocation nhsrc{up_, o + sb + sb / 2 + kSlotPage};
        SDL_GPUBufferRegion nhdst{hidden_buf_, a / 2, kSlotPage / 2};
        SDL_UploadToGPUBuffer(copy, &nhsrc, &nhdst, false);
    }
    if (pages_bytes) {
        SDL_GPUTransferBufferLocation psrc{up_, static_cast<u32>(per_page * pages.size())};
        SDL_GPUBufferRegion pdst{pages_buf_, 0, pages_bytes};
        SDL_UploadToGPUBuffer(copy, &psrc, &pdst, false);
        pages_dirty_ = false;
    }
    SDL_EndGPUCopyPass(copy);
    SDL_SubmitGPUCommandBuffer(cmd);
}

void ExactRdpGpu::flush(ExactRdp& front, u8* rdram, size_t rdram_size) {
    submit(front, rdram, rdram_size);
    sync(front, rdram);
}

void ExactRdpGpu::report_stats() {
    const u64 now = SDL_GetTicksNS();
    if (now - stats_.last < 1000000000ull) return;
    std::fprintf(stderr,
                 "exact gpu: %llu submits, %llu syncs, %llu prims, %llu passes, %llu chunks, %llu tile entries, up %.1f MB, "
                 "down %.1f MB, cpu %.1f ms, wait %.1f ms, pairs by variant %llu/%llu/%llu/%llu/%llu/%llu\n",
                 static_cast<unsigned long long>(stats_.submits), static_cast<unsigned long long>(stats_.syncs),
                 static_cast<unsigned long long>(stats_.prims), static_cast<unsigned long long>(stats_.passes),
                 static_cast<unsigned long long>(stats_.chunks), static_cast<unsigned long long>(stats_.entries),
                 stats_.up_bytes / 1048576.0, stats_.down_bytes / 1048576.0, stats_.cpu_ms, stats_.wait_ms,
                 static_cast<unsigned long long>(stats_.variant[0]), static_cast<unsigned long long>(stats_.variant[1]),
                 static_cast<unsigned long long>(stats_.variant[2]), static_cast<unsigned long long>(stats_.variant[3]),
                 static_cast<unsigned long long>(stats_.variant[4]), static_cast<unsigned long long>(stats_.variant[5]));
    stats_ = Stats{};
    stats_.on = true;
    stats_.last = now;
}

ExactAccelFactory make_exact_accel_factory(std::shared_ptr<Device> device) {
    if (!device || !device->exact_ok()) return nullptr;
    return [device]() -> std::unique_ptr<ExactAccel> { return std::make_unique<ExactRdpGpu>(device); };
}

} // namespace gpu
