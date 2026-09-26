#pragma once
// The SDL_GPU device the frontend draws with, and the compute pipelines of the
// GPU RDP renderer (rdp_gpu.hpp) built for it: Metal on macOS, Vulkan on
// Linux and Windows, Direct3D 12 on Windows machines without Vulkan.

#include <SDL3/SDL_gpu.h>
#include <memory>
#include <string>

namespace gpu {

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

private:
    SDL_GPUDevice* device_;
    SDL_GPUComputePipeline* raster_ = nullptr;
    SDL_GPUComputePipeline* compose_ = nullptr;
    SDL_GPUComputePipeline* fill_ = nullptr;
    SDL_GPUComputePipeline* decode_ = nullptr;
    std::string error_;
};

} // namespace gpu
