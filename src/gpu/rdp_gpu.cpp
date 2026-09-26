#include "rdp_gpu.hpp"
#include "shaders/layout.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gpu {

namespace {

u32 fbits(f32 v) {
    u32 u;
    std::memcpy(&u, &v, 4);
    return u;
}

const u32 kFarDepth = fbits(1e30f); // what the CPU renderer clears depth to

// Frames the emulation thread may run ahead of the GPU.
constexpr size_t kMaxInFlight = 3;

// Uniform blocks of the shaders (std140: 4-byte scalars, no padding).
struct RasterParams {
    u32 scale, width, height, tiles_x, tile_table, work_list, prims, has_depth;
    f32 inv_scale;
};
struct FillParams {
    u32 count, value, stride;
};
struct DecodeParams {
    u32 jobs, job_count, blocks, groups_x;
};

} // namespace

struct RdpRenderer::Compose {
    Target* target;
    std::shared_ptr<Image> image;
    u32 params[10]; // compose.comp's uniform block
};

// A composed frame in video memory. Read back only for screenshots and
// thumbnails.
class Image final : public GpuImage {
public:
    Image(SDL_GPUDevice* dev, u32 width, u32 height) : dev_(dev) {
        SDL_GPUTextureCreateInfo ci{};
        ci.type = SDL_GPU_TEXTURETYPE_2D;
        ci.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        ci.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
        ci.width = width;
        ci.height = height;
        ci.layer_count_or_depth = 1;
        ci.num_levels = 1;
        tex_ = SDL_CreateGPUTexture(dev, &ci);
        w = static_cast<int>(width);
        h = static_cast<int>(height);
    }
    ~Image() override {
        if (tex_) SDL_ReleaseGPUTexture(dev_, tex_);
    }
    bool ok() const { return tex_ != nullptr; }
    SDL_GPUTexture* tex() const { return tex_; }
    void* texture() const override { return tex_; }

    bool read(std::vector<u32>& argb) const override {
        const u32 bytes = static_cast<u32>(w) * static_cast<u32>(h) * 4;
        SDL_GPUTransferBufferCreateInfo ti{};
        ti.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
        ti.size = bytes;
        SDL_GPUTransferBuffer* tb = SDL_CreateGPUTransferBuffer(dev_, &ti);
        if (!tb) return false;
        bool ok = false;
        if (SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(dev_)) {
            SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
            SDL_GPUTextureRegion src{};
            src.texture = tex_;
            src.w = static_cast<u32>(w);
            src.h = static_cast<u32>(h);
            src.d = 1;
            SDL_GPUTextureTransferInfo dst{};
            dst.transfer_buffer = tb;
            dst.pixels_per_row = static_cast<u32>(w);
            dst.rows_per_layer = static_cast<u32>(h);
            SDL_DownloadFromGPUTexture(copy, &src, &dst);
            SDL_EndGPUCopyPass(copy);
            if (SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd)) {
                SDL_WaitForGPUFences(dev_, true, &fence, 1);
                SDL_ReleaseGPUFence(dev_, fence);
                if (const auto* p = static_cast<const u8*>(SDL_MapGPUTransferBuffer(dev_, tb, false))) {
                    argb.resize(static_cast<size_t>(w) * h);
                    for (size_t i = 0; i < argb.size(); ++i)
                        argb[i] = 0xFF000000u | static_cast<u32>(p[i * 4]) << 16 | static_cast<u32>(p[i * 4 + 1]) << 8 | p[i * 4 + 2];
                    SDL_UnmapGPUTransferBuffer(dev_, tb);
                    ok = true;
                }
            }
        }
        SDL_ReleaseGPUTransferBuffer(dev_, tb);
        return ok;
    }

private:
    SDL_GPUDevice* dev_;
    SDL_GPUTexture* tex_ = nullptr;
};

// ---------------------------------------------------------------------------
// Setup / teardown

RdpRenderer::RdpRenderer(std::shared_ptr<Device> device, u32 scale)
    : HiResRenderer(scale), dev_(std::move(device)), gpu_(dev_->get()) {
    no_depth_ = make_buffer(4, "orbit64 no depth");
    texel_buf_words_ = 1u << 16;
    texels_ = make_buffer(texel_buf_words_, "orbit64 texels");
    reset_batch();
    stats_.on = std::getenv("ORBIT64_GPU_STATS") != nullptr;
    stats_.sync = std::getenv("ORBIT64_GPU_SYNC") != nullptr;
    stats_.last = SDL_GetTicksNS();
}

RdpRenderer::~RdpRenderer() {
    // The GPU may still be reading what this renderer owns.
    for (SDL_GPUFence* f : in_flight_) {
        SDL_WaitForGPUFences(gpu_, true, &f, 1);
        SDL_ReleaseGPUFence(gpu_, f);
    }
    for (auto& t : targets_) SDL_ReleaseGPUBuffer(gpu_, t->buffer);
    for (auto& d : depths_) SDL_ReleaseGPUBuffer(gpu_, d->buffer);
    if (no_depth_) SDL_ReleaseGPUBuffer(gpu_, no_depth_);
    if (texels_) SDL_ReleaseGPUBuffer(gpu_, texels_);
    if (data_buf_) SDL_ReleaseGPUBuffer(gpu_, data_buf_);
    if (upload_) SDL_ReleaseGPUTransferBuffer(gpu_, upload_);
}

SDL_GPUBuffer* RdpRenderer::make_buffer(u32 words, const char* name) {
    SDL_GPUBufferCreateInfo ci{};
    ci.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE;
    ci.size = words * 4;
    ci.props = SDL_CreateProperties();
    SDL_SetStringProperty(ci.props, SDL_PROP_GPU_BUFFER_CREATE_NAME_STRING, name);
    SDL_GPUBuffer* b = SDL_CreateGPUBuffer(gpu_, &ci);
    SDL_DestroyProperties(ci.props);
    return b;
}

void RdpRenderer::reset_batch() {
    data_.resize(GPU_LUT_WORDS);
    for (u32 i = 0; i < GPU_LUT_WORDS; ++i) data_[i] = fbits(raster::kByteToUnit[i]);
    prims_.clear();
    info_.clear();
    fills_.clear();
    rec_state_ = kNone;
    rec_serial_ = rec_tmem_gen_ = ~0ull;
    tmem_copy_ = kNone;
    tmem_copy_gen_ = ~0ull;
    tables_.clear();
    jobs_.clear();
    texel_words_ = 0;
    decode_blocks_ = 0;
}

// ---------------------------------------------------------------------------
// Recording

RdpRenderer::Target* RdpRenderer::find(u32 addr, u32 width, u8 size) const {
    for (const auto& t : targets_)
        if (t->addr == addr && t->width == width && t->size == size) return t.get();
    return nullptr;
}

HiResTarget* RdpRenderer::bind(u32 addr, u32 width, u8 size, const u8* rdram, size_t rdram_size) {
    if (bound_ && bound_->addr == addr && bound_->width == width && bound_->size == size) return bound_;
    if (width == 0 || width > kMaxWidth || (size != 2 && size != 3)) return nullptr;

    const u32 S = scale_;
    Target* t = find(addr, width, size);
    if (!t) {
        auto nt = std::make_unique<Target>();
        nt->addr = addr;
        nt->width = width;
        nt->size = size;
        const u32 words = width * S * kFbLines * S;
        nt->buffer = make_buffer(words, "orbit64 colour image");
        if (!nt->buffer) return nullptr;
        // The buffer starts out cleared and the shadow as zeros, so every
        // pixel RDRAM holds counts as changed and is uploaded. (A zero pixel
        // is zero in the buffer too.)
        nt->shadow.assign(static_cast<size_t>(width) * kFbLines, 0);
        fills_.push_back({prims_.size() / GPU_PRIM_WORDS, nt->buffer, words, 0});
        t = nt.get();
        targets_.push_back(std::move(nt));
        hires::scan_changes(*t, rdram, rdram_size, changed_);
    } else {
        // Copy in the pixels something other than the RDP changed since it last drew here.
        hires::scan_changes(*t, rdram, rdram_size, changed_);
    }
    if (!changed_.empty()) add_upload(t, changed_);
    t->last_used = frame_;
    bound_ = t;
    return t;
}

void RdpRenderer::add_upload(Target* t, const std::vector<hires::ChangedPixel>& px) {
    u32 x0 = ~0u, y0 = ~0u, x1 = 0, y1 = 0;
    for (const auto& p : px) {
        x0 = std::min<u32>(x0, p.x);
        x1 = std::max<u32>(x1, p.x);
        y0 = std::min<u32>(y0, p.y);
        y1 = std::max<u32>(y1, p.y);
    }
    const u32 w = x1 - x0 + 1, h = y1 - y0 + 1;
    const u32 off = static_cast<u32>(data_.size());
    data_.resize(data_.size() + static_cast<size_t>(w) * h * 2, 0);
    for (const auto& p : px) {
        u32* e = data_.data() + off + (static_cast<size_t>(p.y - y0) * w + (p.x - x0)) * 2;
        e[0] = 1;
        e[1] = p.argb;
    }
    const s32 S = static_cast<s32>(scale_);
    u32* q = add_prim(GPU_PRIM_UPLOAD, t, nullptr, 0, static_cast<s32>(x0) * S, static_cast<s32>(y0) * S,
                      static_cast<s32>(x1 + 1) * S - 1, static_cast<s32>(y1 + 1) * S - 1);
    if (!q) return;
    q[3] = off;
    q[4] = w;
    q[5] = x0;
    q[6] = y0;
}

RdpRenderer::Depth* RdpRenderer::depth_for(const DrawState& st) {
    if (!st.z_compare && !st.z_update) return nullptr;
    for (auto& d : depths_) {
        if (d->width == st.fb_w) {
            d->last_used = frame_;
            return d.get();
        }
    }
    auto d = std::make_unique<Depth>();
    d->width = st.fb_w;
    const u32 words = st.fb_w * scale_ * kFbLines * scale_;
    d->buffer = make_buffer(words, "orbit64 depth");
    if (!d->buffer) return nullptr;
    d->last_used = frame_;
    fills_.push_back({prims_.size() / GPU_PRIM_WORDS, d->buffer, words, kFarDepth});
    depths_.push_back(std::move(d));
    return depths_.back().get();
}

u32 RdpRenderer::recorded_state(const DrawState& st, u64 serial, u64 tmem_gen, bool needs_tmem) {
    if (rec_state_ != kNone && rec_serial_ == serial && (!needs_tmem || rec_tmem_gen_ == tmem_gen)) return rec_state_;
    u32 tmem = kNone;
    if (needs_tmem) {
        if (tmem_copy_ == kNone || tmem_copy_gen_ != tmem_gen) {
            tmem_copy_ = static_cast<u32>(data_.size());
            data_.resize(data_.size() + GPU_TMEM_WORDS, 0);
            u32* w = data_.data() + tmem_copy_;
            std::memcpy(w, st.tmem, 4096); // little-endian hosts: byte k is bits 8*(k&3) of word k/4
            for (u32 i = 0; i < 512; ++i)
                if (st.tmem_dxt[i]) w[1024 + (i >> 5)] |= 1u << (i & 31);
            tmem_copy_gen_ = tmem_gen;
        }
        tmem = tmem_copy_;
        rec_tmem_gen_ = tmem_gen;
    } else {
        rec_tmem_gen_ = ~0ull;
    }

    const u32 off = static_cast<u32>(data_.size());
    data_.resize(data_.size() + GPU_STATE_WORDS, 0);
    u32* w = data_.data() + off;
    auto flag = [](bool b, u32 f) { return b ? f : 0u; };
    w[0] = flag(st.two_cycle, GPU_ST_TWO_CYCLE) | flag(st.copy_mode, GPU_ST_COPY) |
           flag(st.fill_or_copy, GPU_ST_FILL_OR_COPY) | flag(st.point_sample, GPU_ST_POINT) |
           flag(st.texture_enabled, GPU_ST_TEXTURED) | flag(st.smooth_shading, GPU_ST_SMOOTH) |
           flag(st.combine_set, GPU_ST_COMBINED) | flag(st.need_tex0, GPU_ST_NEED_TEX0) |
           flag(st.need_tex1, GPU_ST_NEED_TEX1) | flag(st.blend_enabled, GPU_ST_BLEND) |
           flag(st.alpha_zero_kill, GPU_ST_ZERO_KILL) | flag(st.z_compare, GPU_ST_Z_COMPARE) |
           flag(st.z_update, GPU_ST_Z_UPDATE) | flag(st.fb_size == 2, GPU_ST_16BIT);
    w[1] = (st.tlut_type & 15) | (st.active_tile & 7) << 4 | static_cast<u32>(st.alpha_compare & 3) << 8 |
           static_cast<u32>(st.alpha_threshold) << 16;
    auto mux = [](u8 a, u8 b, u8 c, u8 d) {
        return static_cast<u32>(a) | static_cast<u32>(b) << 8 | static_cast<u32>(c) << 16 | static_cast<u32>(d) << 24;
    };
    w[2] = mux(st.cc_a0, st.cc_b0, st.cc_c0, st.cc_d0);
    w[3] = mux(st.ac_a0, st.ac_b0, st.ac_c0, st.ac_d0);
    w[4] = mux(st.cc_a1, st.cc_b1, st.cc_c1, st.cc_d1);
    w[5] = mux(st.ac_a1, st.ac_b1, st.ac_c1, st.ac_d1);
    w[6] = st.prim_color;
    w[7] = st.env_color;
    w[8] = st.blend_color;
    w[9] = st.fog_color;
    w[10] = mux(st.bl_p, st.bl_a, st.bl_m, st.bl_b);
    // The scissor as HiResSink applies it.
    const u32 S = scale_;
    const u32 eff_lrx = (st.scissor_lrx > st.scissor_ulx) ? st.scissor_lrx : st.fb_w;
    const u32 eff_lry = (st.scissor_lry > st.scissor_uly) ? st.scissor_lry : kFbLines;
    w[11] = st.scissor_ulx * S;
    w[12] = eff_lrx * S;
    w[13] = st.scissor_uly * S;
    w[14] = eff_lry * S;
    w[15] = tmem;
    for (u32 i = 0; i < 8; ++i) {
        const raster::TexUnit& tu = st.tex[i];
        const Tile& tl = tu.tile;
        u32* tw = w + GPU_ST_TILES + i * GPU_TILE_WORDS;
        tw[0] = (tl.format & 15u) | (tl.size & 15u) << 4 | (tl.palette & 15u) << 8 | (tl.mask_s & 15u) << 12 |
                (tl.mask_t & 15u) << 16 | (tl.clamp_s ? 1u : 0u) << 20 | (tl.clamp_t ? 1u : 0u) << 21 |
                (tl.mirror_s ? 1u : 0u) << 22 | (tl.mirror_t ? 1u : 0u) << 23;
        tw[1] = tu.tmem_base;
        tw[2] = tu.row_stride;
        tw[3] = static_cast<u32>(tu.extent_s);
        tw[4] = static_cast<u32>(tu.extent_t);
        tw[5] = fbits(tu.shift_mul_s);
        tw[6] = fbits(tu.shift_mul_t);
        tw[7] = fbits(tu.origin_s);
        tw[8] = fbits(tu.origin_t);
        tw[9] = GPU_NO_TABLE;
    }
    rec_state_ = off;
    rec_serial_ = serial;
    return off;
}

void RdpRenderer::attach_table(u32 state, const DrawState& st, u32 tile, u64 tmem_gen) {
    const u32 tile_off = state + GPU_ST_TILES + (tile & 7) * GPU_TILE_WORDS;
    const u32 tmem = data_[state + 15];
    if (data_[tile_off + 9] != GPU_NO_TABLE || tmem == kNone) return;
    const raster::TexUnit& tu = st.tex[tile & 7];
    u32 w = 0, h = 0;
    raster::tex_cache_dims(tu, w, h);
    if (static_cast<u64>(w) * h > 256 * 256) return; // too big to be worth decoding
    TableKey key{tmem_gen, 0, 0};
    raster::tex_cache_key(tu, st.tlut_type, key.a, key.b);
    auto it = tables_.find(key);
    if (it == tables_.end()) {
        const u32 job[GPU_JOB_WORDS] = {tile_off, tmem, st.tlut_type, w, h, texel_words_, decode_blocks_, 0};
        jobs_.insert(jobs_.end(), job, job + GPU_JOB_WORDS);
        it = tables_.emplace(key, texel_words_).first;
        texel_words_ += w * h;
        decode_blocks_ += (w * h + GPU_DECODE_BLOCK - 1) / GPU_DECODE_BLOCK;
    }
    data_[tile_off + 9] = it->second;
}

u32* RdpRenderer::add_prim(u32 type, Target* t, Depth* d, u32 state, s32 x0, s32 y0, s32 x1, s32 y1) {
    // Clip to the colour image; nothing outside it is ever drawn.
    const s32 w = static_cast<s32>(t->width * scale_), h = static_cast<s32>(kFbLines * scale_);
    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);
    x1 = std::min(x1, w - 1);
    y1 = std::min(y1, h - 1);
    if (x0 > x1 || y0 > y1) return nullptr;
    info_.push_back({t, d, static_cast<u16>(x0), static_cast<u16>(x1), static_cast<u16>(y0), static_cast<u16>(y1)});
    const size_t at = prims_.size();
    prims_.resize(at + GPU_PRIM_WORDS, 0);
    u32* q = prims_.data() + at;
    q[0] = type;
    q[1] = static_cast<u32>(x0) | static_cast<u32>(x1) << 16;
    q[2] = static_cast<u32>(y0) | static_cast<u32>(y1) << 16;
    q[GPU_PRIM_STATE] = state;
    return q;
}

void RdpRenderer::triangle(HiResTarget* ht, const DrawState& st, u64 serial, u64 tmem_gen,
                           const Vertex& v0, const Vertex& v1, const Vertex& v2, f32 area) {
    f32 min_x, max_x, min_y, max_y;
    if (!raster::triangle_bounds(st, v0, v1, v2, scale_, min_x, max_x, min_y, max_y)) return;
    Target* t = static_cast<Target*>(ht);
    const u32 state = recorded_state(st, serial, tmem_gen, st.texture_enabled);
    if (st.texture_enabled) {
        if (!st.combine_set || st.need_tex0) attach_table(state, st, st.active_tile, tmem_gen);
        if (st.combine_set && st.need_tex1) attach_table(state, st, st.active_tile + 1, tmem_gen);
    }
    u32* q = add_prim(GPU_PRIM_TRI, t, depth_for(st), state, static_cast<s32>(min_x), static_cast<s32>(min_y),
                      static_cast<s32>(max_x), static_cast<s32>(max_y));
    if (!q) return;
    // What raster::triangle() derives per triangle, so both pipelines start
    // from the same values.
    const Vertex* v[3] = {&v0, &v1, &v2};
    for (int i = 0; i < 3; ++i) {
        u32* p = q + 3 + i * 7;
        const f32 inv_w = 1.0f / (v[i]->w != 0.0f ? v[i]->w : 1.0f);
        p[0] = fbits(v[i]->sx);
        p[1] = fbits(v[i]->sy);
        p[2] = fbits(v[i]->sz);
        p[3] = fbits(inv_w);
        p[4] = fbits(v[i]->u * inv_w);
        p[5] = fbits(v[i]->v * inv_w);
        p[6] = static_cast<u32>(v[i]->r) | static_cast<u32>(v[i]->g) << 8 | static_cast<u32>(v[i]->b) << 16 |
               static_cast<u32>(v[i]->a) << 24;
    }
    q[24] = fbits(1.0f / area);
}

void RdpRenderer::tex_rect(HiResTarget* ht, const DrawState& st, u64 serial, u64 tmem_gen, u32 ulx, u32 uly,
                           u32 lrx, u32 lry, u32 tile, f32 s, f32 tc, f32 dsdx, f32 dtdy, bool flip) {
    // The bounds and steps of raster::tex_rect().
    u32 max_x = std::min(lrx, st.fb_w);
    u32 max_y = std::min(lry, kMaxFbLines);
    if (st.fill_or_copy) {
        max_x = std::min(lrx + 1, st.fb_w);
        max_y = std::min(lry + 1, kMaxFbLines);
    }
    if (st.copy_mode) dsdx *= 0.25f;
    const u32 S = scale_;
    const s32 x0 = static_cast<s32>(ulx * S), x1 = static_cast<s32>(max_x * S);
    const s32 y0 = static_cast<s32>(uly * S), y1 = static_cast<s32>(max_y * S);
    if (x0 >= x1 || y0 >= y1) return;
    Target* t = static_cast<Target*>(ht);
    // The primitive's upper-left corner is where S/T start, so it can't be clipped.
    if (x0 >= static_cast<s32>(t->width * S) || y0 >= static_cast<s32>(kFbLines * S)) return;
    const u32 state = recorded_state(st, serial, tmem_gen, true);
    const bool combined = st.combine_set && !st.copy_mode;
    if (!combined || st.need_tex0) attach_table(state, st, tile, tmem_gen);
    if (combined && st.need_tex1) attach_table(state, st, tile + 1, tmem_gen);
    u32* q = add_prim(GPU_PRIM_RECT, t, depth_for(st), state, x0, y0, x1 - 1, y1 - 1);
    if (!q) return;
    q[3] = tile;
    q[4] = fbits(s);
    q[5] = fbits(tc);
    q[6] = fbits(dsdx / static_cast<f32>(S));
    q[7] = fbits(dtdy / static_cast<f32>(S));
    q[8] = flip ? 1u : 0u;
}

void RdpRenderer::fill_rect(HiResTarget* ht, u32 x0, u32 y0, u32 x1, u32 y1, u32 argb) {
    const s32 S = static_cast<s32>(scale_);
    u32* q = add_prim(GPU_PRIM_FILL, static_cast<Target*>(ht), nullptr, 0, static_cast<s32>(x0) * S,
                      static_cast<s32>(y0) * S, static_cast<s32>(x1) * S - 1, static_cast<s32>(y1) * S - 1);
    if (q) q[3] = argb;
}

void RdpRenderer::pixels(HiResTarget* ht, const DrawState& st, u64 serial, const std::vector<Pixel>& px) {
    if (px.empty()) return;
    Target* t = static_cast<Target*>(ht);
    const u32 state = recorded_state(st, serial, 0, false);
    Depth* d = depth_for(st);
    const s32 S = static_cast<s32>(scale_);
    for (const Pixel& p : px) {
        u32* q = add_prim(GPU_PRIM_BLOCK, t, d, state, p.x * S, p.y * S, p.x * S + S - 1, p.y * S + S - 1);
        if (!q) continue;
        q[3] = p.color;
        q[4] = fbits(p.z);
    }
}

void RdpRenderer::blit(HiResTarget* ht, const DrawState& st, u64 serial, u32 x0, u32 y0, u32 w, u32 h,
                       const u32* colors) {
    if (w == 0 || h == 0) return;
    const u32 state = recorded_state(st, serial, 0, false);
    const u32 off = static_cast<u32>(data_.size());
    data_.insert(data_.end(), colors, colors + static_cast<size_t>(w) * h);
    const s32 S = static_cast<s32>(scale_);
    u32* q = add_prim(GPU_PRIM_BLIT, static_cast<Target*>(ht), depth_for(st), state, static_cast<s32>(x0) * S,
                      static_cast<s32>(y0) * S, static_cast<s32>(x0 + w) * S - 1, static_cast<s32>(y0 + h) * S - 1);
    if (!q) return;
    q[3] = off;
    q[4] = w;
    q[5] = x0;
    q[6] = y0;
}

void RdpRenderer::clear_depth() {
    for (auto& d : depths_)
        fills_.push_back({prims_.size() / GPU_PRIM_WORDS, d->buffer, d->width * scale_ * kFbLines * scale_, kFarDepth});
}

void RdpRenderer::flush() {}

// ---------------------------------------------------------------------------
// Output

std::shared_ptr<Image> RdpRenderer::acquire_image(u32 w, u32 h) {
    // Images of another size are dropped from the pool; one the frontend
    // still shows lives on until it lets go.
    images_.erase(std::remove_if(images_.begin(), images_.end(),
                                 [&](const auto& im) { return im->w != static_cast<int>(w) || im->h != static_cast<int>(h); }),
                  images_.end());
    for (auto& im : images_)
        if (im.use_count() == 1) return im;
    auto im = std::make_shared<Image>(gpu_, w, h);
    if (!im->ok()) return nullptr;
    if (images_.size() < 4) images_.push_back(im);
    return im;
}

bool RdpRenderer::present(const VIScanout& so, const u8* rdram, size_t rdram_size, VideoFrame& out) {
    u32 first_line = 0;
    auto* t = static_cast<Target*>(hires::find_scanout(targets_, so, first_line));
    if (!t) return false;
    t->last_used = frame_;
    const u32 S = scale_, W = t->width, bpp = so.bpp;
    Compose c;
    c.target = t;
    c.image = acquire_image(so.canvas_w * S, so.canvas_h * S);
    if (!c.image) return false;

    // Per displayed pixel: 0 while RDRAM still holds what the RDP wrote there
    // (the high-resolution pixel is shown), else the colour RDRAM holds.
    const u32 lines = std::min(so.lines, kFbLines);
    const u32 ovr = static_cast<u32>(data_.size());
    data_.resize(data_.size() + static_cast<size_t>(W) * lines);
    for (u32 ny = 0; ny < lines; ++ny) {
        u32* o = data_.data() + ovr + static_cast<size_t>(ny) * W;
        const size_t row = static_cast<size_t>(so.fb_base) + static_cast<size_t>(ny) * W * bpp;
        if (row + static_cast<size_t>(W) * bpp > rdram_size) {
            std::fill(o, o + W, 0xFF000000u);
            continue;
        }
        const u8* p = rdram + row;
        const u32 ty = ny + first_line;
        const u32* sh = ty < kFbLines ? t->shadow.data() + static_cast<size_t>(ty) * W : nullptr;
        for (u32 x = 0; x < W; ++x) {
            const u32 raw = fb_read_pixel(p + x * bpp, bpp);
            o[x] = (sh && raw == sh[x]) ? 0u : (fb_pixel_to_argb(raw, bpp) | 0xFF000000u);
        }
    }
    const u32 params[10] = {S, W, first_line, lines, so.shown_w, so.x0, so.y0, so.canvas_w * S, so.canvas_h * S, ovr};
    std::memcpy(c.params, params, sizeof params);
    submit(&c);

    out.pixels.clear();
    out.gpu = c.image;
    out.w = c.image->w;
    out.h = c.image->h;
    out.scale = static_cast<int>(S);
    return true;
}

void RdpRenderer::end_frame() {
    if (!prims_.empty() || !fills_.empty()) submit(nullptr);
    // Colour images and depth buffers the game hasn't drawn to or shown for
    // ten seconds are gone (or were one-off render targets). The GPU lets go
    // of released buffers once it is done with them.
    constexpr u64 kIdleFrames = 600;
    constexpr size_t kMaxTargets = 16;
    auto idle = [&](u64 last_used) { return frame_ - last_used > kIdleFrames; };
    for (auto it = targets_.begin(); it != targets_.end();) {
        if (idle((*it)->last_used)) {
            SDL_ReleaseGPUBuffer(gpu_, (*it)->buffer);
            it = targets_.erase(it);
        } else {
            ++it;
        }
    }
    while (targets_.size() > kMaxTargets) {
        auto oldest = std::min_element(targets_.begin(), targets_.end(),
                                       [](const auto& a, const auto& b) { return a->last_used < b->last_used; });
        SDL_ReleaseGPUBuffer(gpu_, (*oldest)->buffer);
        targets_.erase(oldest);
    }
    for (auto it = depths_.begin(); it != depths_.end();) {
        if (idle((*it)->last_used)) {
            SDL_ReleaseGPUBuffer(gpu_, (*it)->buffer);
            it = depths_.erase(it);
        } else {
            ++it;
        }
    }
    bound_ = nullptr;
    ++frame_;
}

// ---------------------------------------------------------------------------
// Submission

void RdpRenderer::submit(const Compose* c) {
    const u64 t_start = SDL_GetTicksNS();
    const u32 S = scale_;
    const size_t nprims = info_.size();

    // Passes: runs of primitives drawn into one colour image, cut wherever a
    // buffer is filled. Each gets its tile bins appended to the data.
    struct Pass {
        size_t begin, end;
        Target* target;
        Depth* depth;
        u32 tiles_x, table, work, work_count;
    };
    std::vector<Pass> passes;
    size_t next_fill = 0;
    for (size_t i = 0; i < nprims;) {
        while (next_fill < fills_.size() && fills_[next_fill].at <= i) ++next_fill;
        const size_t cut = next_fill < fills_.size() ? fills_[next_fill].at : nprims;
        Pass p{i, i + 1, info_[i].target, info_[i].depth, 0, 0, 0, 0};
        while (p.end < nprims && p.end < cut && info_[p.end].target == p.target) {
            if (!p.depth) p.depth = info_[p.end].depth;
            ++p.end;
        }
        passes.push_back(p);
        i = p.end;
    }

    const u32 jobs_off = static_cast<u32>(data_.size());
    data_.insert(data_.end(), jobs_.begin(), jobs_.end());
    const u32 prims_off = static_cast<u32>(data_.size());
    data_.insert(data_.end(), prims_.begin(), prims_.end());
    const u32 tile_px = GPU_BIN_SIZE * S;
    for (Pass& p : passes) {
        const u32 tiles_x = (p.target->width + GPU_BIN_SIZE - 1) / GPU_BIN_SIZE;
        const u32 tiles_y = (kFbLines + GPU_BIN_SIZE - 1) / GPU_BIN_SIZE;
        const u32 ntiles = tiles_x * tiles_y;
        bin_count_.assign(ntiles, 0);
        for (size_t i = p.begin; i < p.end; ++i) {
            const PrimInfo& in = info_[i];
            for (u32 ty = in.y0 / tile_px; ty <= in.y1 / tile_px; ++ty)
                for (u32 tx = in.x0 / tile_px; tx <= in.x1 / tile_px; ++tx) ++bin_count_[ty * tiles_x + tx];
        }
        // Table of (first, count), then the lists, then the tiles with work.
        p.tiles_x = tiles_x;
        p.table = static_cast<u32>(data_.size());
        data_.resize(data_.size() + ntiles * 2);
        u32 list = static_cast<u32>(data_.size());
        bin_first_.resize(ntiles);
        u32 work_count = 0;
        for (u32 k = 0; k < ntiles; ++k) {
            data_[p.table + k * 2] = list;
            data_[p.table + k * 2 + 1] = bin_count_[k];
            bin_first_[k] = list;
            list += bin_count_[k];
            if (bin_count_[k]) ++work_count;
        }
        data_.resize(list);
        for (size_t i = p.begin; i < p.end; ++i) {
            const PrimInfo& in = info_[i];
            for (u32 ty = in.y0 / tile_px; ty <= in.y1 / tile_px; ++ty)
                for (u32 tx = in.x0 / tile_px; tx <= in.x1 / tile_px; ++tx) data_[bin_first_[ty * tiles_x + tx]++] = static_cast<u32>(i);
        }
        p.work = static_cast<u32>(data_.size());
        p.work_count = work_count;
        for (u32 k = 0; k < ntiles; ++k)
            if (bin_count_[k]) data_.push_back(k);
    }

    // Upload the data.
    const u32 words = static_cast<u32>(data_.size());
    if (words > data_buf_words_) {
        if (data_buf_) SDL_ReleaseGPUBuffer(gpu_, data_buf_);
        data_buf_words_ = std::max<u32>(words + words / 2, 1u << 18);
        data_buf_ = make_buffer(data_buf_words_, "orbit64 frame data");
    }
    if (texel_words_ > texel_buf_words_) {
        if (texels_) SDL_ReleaseGPUBuffer(gpu_, texels_);
        texel_buf_words_ = texel_words_ + texel_words_ / 2;
        texels_ = make_buffer(texel_buf_words_, "orbit64 texels");
    }
    if (words > upload_words_) {
        if (upload_) SDL_ReleaseGPUTransferBuffer(gpu_, upload_);
        upload_words_ = std::max<u32>(words + words / 2, 1u << 18);
        SDL_GPUTransferBufferCreateInfo ti{};
        ti.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        ti.size = upload_words_ * 4;
        upload_ = SDL_CreateGPUTransferBuffer(gpu_, &ti);
    }
    SDL_GPUCommandBuffer* cmd = data_buf_ && upload_ && texels_ ? SDL_AcquireGPUCommandBuffer(gpu_) : nullptr;
    void* mapped = cmd ? SDL_MapGPUTransferBuffer(gpu_, upload_, true) : nullptr;
    if (!mapped) {
        if (cmd) SDL_CancelGPUCommandBuffer(cmd);
        SDL_Log("GPU renderer: submission failed: %s", SDL_GetError());
        reset_batch();
        return;
    }
    std::memcpy(mapped, data_.data(), static_cast<size_t>(words) * 4);
    SDL_UnmapGPUTransferBuffer(gpu_, upload_);
    {
        SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
        SDL_GPUTransferBufferLocation src{upload_, 0};
        SDL_GPUBufferRegion dst{data_buf_, 0, words * 4};
        SDL_UploadToGPUBuffer(copy, &src, &dst, true);
        SDL_EndGPUCopyPass(copy);
    }

    auto run_fill = [&](const BufferFill& f) {
        SDL_GPUStorageBufferReadWriteBinding rw{};
        rw.buffer = f.buffer;
        SDL_GPUComputePass* cp = SDL_BeginGPUComputePass(cmd, nullptr, 0, &rw, 1);
        SDL_BindGPUComputePipeline(cp, dev_->fill());
        const u32 groups = std::min<u32>((f.words + 63) / 64, 65535);
        const FillParams fp{f.words, f.value, groups * 64};
        SDL_PushGPUComputeUniformData(cmd, 0, &fp, sizeof fp);
        SDL_DispatchGPUCompute(cp, groups, 1, 1);
        SDL_EndGPUComputePass(cp);
    };

    if (decode_blocks_) {
        // Every table the batch samples, in a fresh copy of the texel buffer
        // (the GPU may still be drawing the previous batch from it).
        SDL_GPUStorageBufferReadWriteBinding rw{};
        rw.buffer = texels_;
        rw.cycle = true;
        SDL_GPUComputePass* cp = SDL_BeginGPUComputePass(cmd, nullptr, 0, &rw, 1);
        SDL_BindGPUComputePipeline(cp, dev_->decode());
        SDL_BindGPUComputeStorageBuffers(cp, 0, &data_buf_, 1);
        const u32 gx = std::min<u32>(decode_blocks_, 65535);
        const DecodeParams dp{jobs_off, static_cast<u32>(jobs_.size() / GPU_JOB_WORDS), decode_blocks_, gx};
        SDL_PushGPUComputeUniformData(cmd, 0, &dp, sizeof dp);
        SDL_DispatchGPUCompute(cp, gx, (decode_blocks_ + gx - 1) / gx, 1);
        SDL_EndGPUComputePass(cp);
    }

    size_t fi = 0;
    for (const Pass& p : passes) {
        while (fi < fills_.size() && fills_[fi].at <= p.begin) run_fill(fills_[fi++]);
        if (p.work_count == 0) continue;
        SDL_GPUStorageBufferReadWriteBinding rw[2]{};
        rw[0].buffer = p.target->buffer;
        rw[1].buffer = p.depth ? p.depth->buffer : no_depth_;
        SDL_GPUComputePass* cp = SDL_BeginGPUComputePass(cmd, nullptr, 0, rw, 2);
        SDL_BindGPUComputePipeline(cp, dev_->raster());
        SDL_GPUBuffer* ro[2] = {data_buf_, texels_};
        SDL_BindGPUComputeStorageBuffers(cp, 0, ro, 2);
        RasterParams rp{};
        rp.scale = S;
        rp.width = p.target->width * S;
        rp.height = kFbLines * S;
        rp.tiles_x = p.tiles_x;
        rp.tile_table = p.table;
        rp.work_list = p.work;
        rp.prims = prims_off;
        rp.has_depth = p.depth ? 1 : 0;
        rp.inv_scale = 1.0f / static_cast<f32>(S);
        SDL_PushGPUComputeUniformData(cmd, 0, &rp, sizeof rp);
        // raster.comp: 16x16-pixel workgroups, (8 * S / 16)^2 of them per tile.
        SDL_DispatchGPUCompute(cp, S * S, p.work_count, 1);
        SDL_EndGPUComputePass(cp);
    }
    while (fi < fills_.size()) run_fill(fills_[fi++]);

    if (c) {
        SDL_GPUStorageTextureReadWriteBinding tw{};
        tw.texture = c->image->tex();
        tw.cycle = false;
        SDL_GPUComputePass* cp = SDL_BeginGPUComputePass(cmd, &tw, 1, nullptr, 0);
        SDL_BindGPUComputePipeline(cp, dev_->compose());
        SDL_GPUBuffer* ro[2] = {data_buf_, c->target->buffer};
        SDL_BindGPUComputeStorageBuffers(cp, 0, ro, 2);
        SDL_PushGPUComputeUniformData(cmd, 0, c->params, sizeof c->params);
        SDL_DispatchGPUCompute(cp, (c->params[7] + 7) / 8, (c->params[8] + 7) / 8, 1);
        SDL_EndGPUComputePass(cp);
    }

    if (stats_.on) {
        stats_.prims += nprims;
        stats_.passes += passes.size();
        stats_.data_words += words;
        for (const Pass& p : passes) stats_.work_tiles += p.work_count;
        stats_.bin_entries += words - prims_off - static_cast<u32>(prims_.size());
    }
    if (SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd)) in_flight_.push_back(fence);
    const u64 t_submitted = SDL_GetTicksNS();
    if (stats_.sync) {
        for (SDL_GPUFence* f : in_flight_) {
            SDL_WaitForGPUFences(gpu_, true, &f, 1);
            SDL_ReleaseGPUFence(gpu_, f);
        }
        in_flight_.clear();
    }
    // Stay at most a few frames ahead of the GPU.
    while (in_flight_.size() > kMaxInFlight) {
        SDL_GPUFence* f = in_flight_.front();
        in_flight_.pop_front();
        SDL_WaitForGPUFences(gpu_, true, &f, 1);
        SDL_ReleaseGPUFence(gpu_, f);
    }
    while (!in_flight_.empty() && SDL_QueryGPUFence(gpu_, in_flight_.front())) {
        SDL_ReleaseGPUFence(gpu_, in_flight_.front());
        in_flight_.pop_front();
    }
    reset_batch();
    if (stats_.on) {
        const u64 now = SDL_GetTicksNS();
        stats_.submit_ms += (t_submitted - t_start) / 1e6;
        stats_.wait_ms += (now - t_submitted) / 1e6;
        if (c) ++stats_.frames;
        if (now - stats_.last >= 1000000000ull && stats_.frames) {
            const double n = static_cast<double>(stats_.frames);
            std::fprintf(stderr,
                         "[gpu %ux] %llu frames: %.0f prims, %.1f passes, %.0f work tiles, %.0f bin words, %.0f KB data; "
                         "record+submit %.2f ms, %s %.2f ms per frame\n",
                         scale_, (unsigned long long)stats_.frames, stats_.prims / n, stats_.passes / n,
                         stats_.work_tiles / n, stats_.bin_entries / n, stats_.data_words * 4 / 1024.0 / n,
                         stats_.submit_ms / n, stats_.sync ? "gpu" : "wait", stats_.wait_ms / n);
            const bool on = stats_.on, sync = stats_.sync;
            stats_ = Stats{};
            stats_.on = on;
            stats_.sync = sync;
            stats_.last = now;
        }
    }
}

HiResFactory make_hires_factory(std::shared_ptr<Device> device) {
    return [device](u32 scale) -> std::unique_ptr<HiResRenderer> {
        if (!device || !device->ok()) return nullptr;
        return std::make_unique<RdpRenderer>(device, scale);
    };
}

} // namespace gpu
