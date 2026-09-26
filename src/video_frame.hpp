#pragma once
// A finished video frame, handed from the emulator to whoever shows it: an
// ARGB8888 image in memory, or one the GPU renderer left in video memory
// (src/gpu/), which the frontend draws without copying it back.

#include "common.hpp"
#include <memory>
#include <vector>

// An image in video memory. Holding a reference keeps the renderer from
// drawing the next frame into it.
class GpuImage {
public:
    virtual ~GpuImage() = default;
    virtual void* texture() const = 0; // SDL_GPUTexture*, R8G8B8A8
    // Copies the image into memory as ARGB8888 (waits for the GPU).
    virtual bool read(std::vector<u32>& argb) const = 0;
    int w = 0, h = 0;
};

struct VideoFrame {
    std::vector<u32> pixels;        // ARGB8888, w x h; empty when `gpu` holds the image
    std::shared_ptr<GpuImage> gpu;
    int w = 0, h = 0;
    int scale = 1;                  // internal resolution: the game's frame buffer is w/scale x h/scale

    bool empty() const { return w <= 0 || h <= 0 || (pixels.empty() && !gpu); }
    // The image in memory, reading it back from the GPU when it lives there.
    bool to_pixels(std::vector<u32>& out) const {
        if (gpu) return gpu->read(out);
        out = pixels;
        return !out.empty();
    }
};
