#pragma once
// The SDL_GPU device the frontend draws with, and the compute pipelines of the
// GPU RDP renderer (rdp_gpu.hpp) built for it: Metal on macOS, Vulkan on
// Linux and Windows, Direct3D 12 on Windows machines without Vulkan.

#include <SDL3/SDL_gpu.h>
#include <memory>
#include <string>

namespace gpu {

namespace shaders {
struct SpirvBlob;
}

// What a compute shader binds (SDL_GPUComputePipelineCreateInfo).
struct ShaderResources {
    Uint32 ro_textures = 0, ro_buffers = 0, rw_textures = 0, rw_buffers = 0, uniforms = 1;
    Uint32 tx = 8, ty = 8;
};
// A compute pipeline from a SPIR-V-only shader (shaders_ext_gen.hpp), or
// nullptr on a driver that doesn't take SPIR-V (Metal, Direct3D 12).
SDL_GPUComputePipeline* make_spirv_pipeline(SDL_GPUDevice* dev, const shaders::SpirvBlob& blob,
                                            const ShaderResources& r, const char* name);

// A device for the current platform, preferring the drivers the renderer's
// shaders are verified on. ORBIT64_GPU_DRIVER (vulkan, metal, direct3d12)
// picks one; ORBIT64_GPU_DEBUG turns on the driver's validation.
SDL_GPUDevice* create_device();

class Device {
public:
    // Builds the pipelines on `device` (which stays the caller's). ok() says
    // whether it worked; the RDP then keeps rendering high resolutions on the
    // CPU.
    explicit Device(SDL_GPUDevice* device);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    bool ok() const { return raster_ && compose_ && fill_ && decode_; }
    const std::string& error() const { return error_; }
    SDL_GPUDevice* get() const { return device_; }
    const char* driver() const { return SDL_GetGPUDeviceDriver(device_); }

    SDL_GPUComputePipeline* raster() const { return raster_; }
    SDL_GPUComputePipeline* compose() const { return compose_; }
    SDL_GPUComputePipeline* fill() const { return fill_; }
    SDL_GPUComputePipeline* decode() const { return decode_; }
    // The bit-exact RDP's pixel pipeline (rdp_exact_gpu.hpp): shading and
    // the memory stage. False if they didn't build, which leaves the rest working.
    static constexpr int kExactVariants = 6;
    bool exact_ok() const {
        for (SDL_GPUComputePipeline* p : exact_shade_)
            if (!p) return false;
        return exact_memory_ && exact_apply_ && exact_init_;
    }
    // Shading variants (shaders/exact.glsl, EXACT_VARIANT): 0 any state,
    // 1 1-cycle without textures, 2 1-cycle, 3 2-cycle, 4 copy, 5 fill.
    SDL_GPUComputePipeline* exact_shade(int variant) const { return exact_shade_[variant]; }
    // Internal resolution: CPU changes into RDRAM and its high-resolution
    // copy; new pages of the copy from RDRAM.
    SDL_GPUComputePipeline* exact_apply() const { return exact_apply_; }
    SDL_GPUComputePipeline* exact_init() const { return exact_init_; }
    SDL_GPUComputePipeline* exact_memory() const { return exact_memory_; }
    // Per-pixel ray tracing (shaders/ext/rt_*.comp), Vulkan only.
    bool rt_ok() const { return rt_trace_ && rt_apply_; }
    SDL_GPUComputePipeline* rt_trace() const { return rt_trace_; }
    SDL_GPUComputePipeline* rt_apply() const { return rt_apply_; }

private:
    SDL_GPUDevice* device_;
    SDL_GPUComputePipeline* raster_ = nullptr;
    SDL_GPUComputePipeline* compose_ = nullptr;
    SDL_GPUComputePipeline* fill_ = nullptr;
    SDL_GPUComputePipeline* decode_ = nullptr;
    SDL_GPUComputePipeline* exact_shade_[kExactVariants] = {};
    SDL_GPUComputePipeline* exact_memory_ = nullptr;
    SDL_GPUComputePipeline* exact_apply_ = nullptr;
    SDL_GPUComputePipeline* exact_init_ = nullptr;
    SDL_GPUComputePipeline* rt_trace_ = nullptr;
    SDL_GPUComputePipeline* rt_apply_ = nullptr;
    std::string error_;
};

} // namespace gpu
