#include "postfx.hpp"
#include "device.hpp"
#include "shaders_ext_gen.hpp"

#include <SDL3/SDL.h>
#include <algorithm>
#include <cstring>

namespace gpu {

namespace {

// Uniform blocks of shaders/ext/pfx_*.comp (std140: vec4s).
struct BrightParams {
    int size[4];
    float knee[4];
};
struct BlurParams {
    int size[4];
};
struct FinalParams {
    int size[4];
    float a[4], b[4], c[4];
};

SDL_GPUTexture* make_texture(SDL_GPUDevice* dev, SDL_GPUTextureFormat fmt, SDL_GPUTextureUsageFlags usage, int w,
                             int h, const char* name) {
    SDL_GPUTextureCreateInfo ci{};
    ci.type = SDL_GPU_TEXTURETYPE_2D;
    ci.format = fmt;
    ci.usage = usage;
    ci.width = static_cast<Uint32>(w);
    ci.height = static_cast<Uint32>(h);
    ci.layer_count_or_depth = 1;
    ci.num_levels = 1;
    ci.props = SDL_CreateProperties();
    SDL_SetStringProperty(ci.props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING, name);
    SDL_GPUTexture* t = SDL_CreateGPUTexture(dev, &ci);
    SDL_DestroyProperties(ci.props);
    return t;
}

} // namespace

PostFx::PostFx(SDL_GPUDevice* device) : dev_(device) {
    ShaderResources one_in;
    one_in.ro_textures = 1;
    one_in.rw_textures = 1;
    ShaderResources two_in;
    two_in.ro_textures = 2;
    two_in.rw_textures = 1;
    bright_ = make_spirv_pipeline(dev_, shaders::pfx_bright, one_in, "orbit64 postfx bright");
    blur_ = make_spirv_pipeline(dev_, shaders::pfx_blur, one_in, "orbit64 postfx blur");
    final_ = make_spirv_pipeline(dev_, shaders::pfx_final, two_in, "orbit64 postfx final");
}

PostFx::~PostFx() {
    release();
    if (upload_) SDL_ReleaseGPUTransferBuffer(dev_, upload_);
    if (upload_tex_) SDL_ReleaseGPUTexture(dev_, upload_tex_);
    if (bright_) SDL_ReleaseGPUComputePipeline(dev_, bright_);
    if (blur_) SDL_ReleaseGPUComputePipeline(dev_, blur_);
    if (final_) SDL_ReleaseGPUComputePipeline(dev_, final_);
}

void PostFx::release() {
    if (out_) SDL_ReleaseGPUTexture(dev_, out_);
    for (SDL_GPUTexture*& t : bloom_) {
        if (t) SDL_ReleaseGPUTexture(dev_, t);
        t = nullptr;
    }
    out_ = nullptr;
    w_ = h_ = 0;
}

bool PostFx::resize(int w, int h) {
    if (w == w_ && h == h_ && out_) return true;
    release();
    const SDL_GPUTextureUsageFlags rw = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
    out_ = make_texture(dev_, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
                        SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE, w, h,
                        "orbit64 postfx output");
    bw_ = std::max(1, (w + 3) / 4);
    bh_ = std::max(1, (h + 3) / 4);
    for (SDL_GPUTexture*& t : bloom_)
        t = make_texture(dev_, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, rw, bw_, bh_, "orbit64 postfx bloom");
    if (!out_ || !bloom_[0] || !bloom_[1]) {
        SDL_Log("Post-processing: %s", SDL_GetError());
        release();
        return false;
    }
    w_ = w;
    h_ = h;
    return true;
}

SDL_GPUTexture* PostFx::run(SDL_GPUTexture* src, int w, int h, const Params& p) {
    return process(src, w, h, false, p);
}

SDL_GPUTexture* PostFx::run(const std::uint32_t* argb, int w, int h, const Params& p) {
    if (!ok() || w <= 0 || h <= 0) return nullptr;
    if (!upload_tex_ || up_w_ != w || up_h_ != h) {
        if (upload_tex_) SDL_ReleaseGPUTexture(dev_, upload_tex_);
        if (upload_) SDL_ReleaseGPUTransferBuffer(dev_, upload_);
        upload_tex_ = make_texture(dev_, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ,
                                   w, h, "orbit64 postfx input");
        SDL_GPUTransferBufferCreateInfo ti{};
        ti.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        ti.size = static_cast<Uint32>(w * h * 4);
        upload_ = SDL_CreateGPUTransferBuffer(dev_, &ti);
        up_w_ = w;
        up_h_ = h;
        if (!upload_tex_ || !upload_) return nullptr;
    }
    void* mapped = SDL_MapGPUTransferBuffer(dev_, upload_, true);
    if (!mapped) return nullptr;
    std::memcpy(mapped, argb, static_cast<size_t>(w) * h * 4);
    SDL_UnmapGPUTransferBuffer(dev_, upload_);
    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(dev_);
    if (!cmd) return nullptr;
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTextureTransferInfo src{upload_, 0, static_cast<Uint32>(w), static_cast<Uint32>(h)};
    SDL_GPUTextureRegion dst{};
    dst.texture = upload_tex_;
    dst.w = static_cast<Uint32>(w);
    dst.h = static_cast<Uint32>(h);
    dst.d = 1;
    SDL_UploadToGPUTexture(copy, &src, &dst, true);
    SDL_EndGPUCopyPass(copy);
    SDL_SubmitGPUCommandBuffer(cmd);
    // ARGB8888 words are B, G, R, A bytes: red and blue come out swapped.
    return process(upload_tex_, w, h, true, p);
}

SDL_GPUTexture* PostFx::process(SDL_GPUTexture* src, int w, int h, bool swap_rb, const Params& p) {
    if (!ok() || !src || w <= 0 || h <= 0 || !resize(w, h)) return nullptr;
    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(dev_);
    if (!cmd) return nullptr;
    auto groups = [](int n) { return static_cast<Uint32>((n + 7) / 8); };

    // Bloom: bright pass, then two rounds of blurring, the second wider.
    {
        SDL_GPUStorageTextureReadWriteBinding rw{};
        rw.texture = bloom_[0];
        rw.cycle = true;
        SDL_GPUComputePass* cp = SDL_BeginGPUComputePass(cmd, &rw, 1, nullptr, 0);
        SDL_BindGPUComputePipeline(cp, bright_);
        SDL_BindGPUComputeStorageTextures(cp, 0, &src, 1);
        BrightParams bp{{w, h, bw_, bh_}, {0.8f, 0.2f, swap_rb ? 1.0f : 0.0f, 0.0f}};
        SDL_PushGPUComputeUniformData(cmd, 0, &bp, sizeof bp);
        SDL_DispatchGPUCompute(cp, groups(bw_), groups(bh_), 1);
        SDL_EndGPUComputePass(cp);
    }
    const int steps[4][2] = {{1, 0}, {0, 1}, {3, 0}, {0, 3}};
    for (int i = 0; i < 4; ++i) {
        SDL_GPUStorageTextureReadWriteBinding rw{};
        rw.texture = bloom_[(i + 1) & 1];
        rw.cycle = true;
        SDL_GPUComputePass* cp = SDL_BeginGPUComputePass(cmd, &rw, 1, nullptr, 0);
        SDL_BindGPUComputePipeline(cp, blur_);
        SDL_BindGPUComputeStorageTextures(cp, 0, &bloom_[i & 1], 1);
        BlurParams bp{{bw_, bh_, steps[i][0], steps[i][1]}};
        SDL_PushGPUComputeUniformData(cmd, 0, &bp, sizeof bp);
        SDL_DispatchGPUCompute(cp, groups(bw_), groups(bh_), 1);
        SDL_EndGPUComputePass(cp);
    }
    {
        SDL_GPUStorageTextureReadWriteBinding rw{};
        rw.texture = out_;
        rw.cycle = true;
        SDL_GPUComputePass* cp = SDL_BeginGPUComputePass(cmd, &rw, 1, nullptr, 0);
        SDL_BindGPUComputePipeline(cp, final_);
        SDL_GPUTexture* in[2] = {src, bloom_[0]};
        SDL_BindGPUComputeStorageTextures(cp, 0, in, 2);
        FinalParams fp{{w, h, bw_, bh_},
                       {p.bloom, p.sharpen, p.vibrance, p.vignette},
                       {p.exposure, p.tonemap ? 1.0f : 0.0f, p.fxaa ? 1.0f : 0.0f, swap_rb ? 1.0f : 0.0f},
                       {p.contrast, static_cast<float>(frame_++ & 63), 0.0f, 0.0f}};
        SDL_PushGPUComputeUniformData(cmd, 0, &fp, sizeof fp);
        SDL_DispatchGPUCompute(cp, groups(w), groups(h), 1);
        SDL_EndGPUComputePass(cp);
    }
    SDL_SubmitGPUCommandBuffer(cmd);
    return out_;
}

bool PostFx::read(std::vector<std::uint32_t>& argb) {
    if (!out_) return false;
    SDL_GPUTransferBufferCreateInfo ti{};
    ti.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
    ti.size = static_cast<Uint32>(w_ * h_ * 4);
    SDL_GPUTransferBuffer* tb = SDL_CreateGPUTransferBuffer(dev_, &ti);
    SDL_GPUCommandBuffer* cmd = tb ? SDL_AcquireGPUCommandBuffer(dev_) : nullptr;
    if (!cmd) {
        if (tb) SDL_ReleaseGPUTransferBuffer(dev_, tb);
        return false;
    }
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTextureRegion src{};
    src.texture = out_;
    src.w = static_cast<Uint32>(w_);
    src.h = static_cast<Uint32>(h_);
    src.d = 1;
    SDL_GPUTextureTransferInfo dst{tb, 0, static_cast<Uint32>(w_), static_cast<Uint32>(h_)};
    SDL_DownloadFromGPUTexture(copy, &src, &dst);
    SDL_EndGPUCopyPass(copy);
    SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    bool ok = false;
    if (fence) {
        SDL_WaitForGPUFences(dev_, true, &fence, 1);
        SDL_ReleaseGPUFence(dev_, fence);
        if (const auto* px = static_cast<const std::uint8_t*>(SDL_MapGPUTransferBuffer(dev_, tb, false))) {
            argb.resize(static_cast<size_t>(w_) * h_);
            for (size_t i = 0; i < argb.size(); ++i)
                argb[i] = 0xFF000000u | (std::uint32_t(px[i * 4]) << 16) | (std::uint32_t(px[i * 4 + 1]) << 8) | px[i * 4 + 2];
            SDL_UnmapGPUTransferBuffer(dev_, tb);
            ok = true;
        }
    }
    SDL_ReleaseGPUTransferBuffer(dev_, tb);
    return ok;
}

} // namespace gpu
