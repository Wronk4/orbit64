#pragma once
// Post-processing of the displayed frame on the GPU (shaders/ext/pfx_*.comp):
// bloom, anti-aliasing, contrast-adaptive sharpening, a filmic tone curve,
// vibrance and a vignette. Runs on the frontend's device, on frames the GPU
// renderer left in video memory or on frames uploaded from memory. Vulkan
// only (the shaders are SPIR-V); ok() is false elsewhere.

#include <SDL3/SDL_gpu.h>
#include <cstdint>
#include <vector>

namespace gpu {

class PostFx {
public:
    struct Params {
        float bloom = 0.6f;     // 0..1
        float sharpen = 0.5f;   // 0..1
        float vibrance = 0.35f;  // 0..1
        float vignette = 0.15f; // 0..1
        float exposure = 1.0f;
        float contrast = 1.1f;
        bool tonemap = true;
        bool fxaa = true;
    };

    explicit PostFx(SDL_GPUDevice* device);
    ~PostFx();
    PostFx(const PostFx&) = delete;
    PostFx& operator=(const PostFx&) = delete;

    bool ok() const { return bright_ && blur_ && final_; }

    // Processes `src` (R8G8B8A8, w x h, with COMPUTE_STORAGE_READ usage) into
    // the output texture, which it returns (nullptr on failure). The texture
    // stays the same while the size does.
    SDL_GPUTexture* run(SDL_GPUTexture* src, int w, int h, const Params& p);
    // The same for a frame in memory (ARGB8888).
    SDL_GPUTexture* run(const std::uint32_t* argb, int w, int h, const Params& p);
    // The last output as ARGB8888 (waits for the GPU): screenshots.
    bool read(std::vector<std::uint32_t>& argb);

private:
    SDL_GPUTexture* process(SDL_GPUTexture* src, int w, int h, bool swap_rb, const Params& p);
    bool resize(int w, int h);
    void release();

    SDL_GPUDevice* dev_;
    SDL_GPUComputePipeline* bright_ = nullptr;
    SDL_GPUComputePipeline* blur_ = nullptr;
    SDL_GPUComputePipeline* final_ = nullptr;
    int w_ = 0, h_ = 0, bw_ = 0, bh_ = 0;
    SDL_GPUTexture* out_ = nullptr;
    SDL_GPUTexture* bloom_[2] = {};
    SDL_GPUTexture* upload_tex_ = nullptr; // frames from memory
    SDL_GPUTransferBuffer* upload_ = nullptr;
    int up_w_ = 0, up_h_ = 0;
    std::uint32_t frame_ = 0;
};

} // namespace gpu
