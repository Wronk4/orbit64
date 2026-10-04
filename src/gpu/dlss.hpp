#pragma once
// NVIDIA DLSS Super Resolution (NGX, the DLSS SDK in third_party/dlss) for
// the shown frame. Experimental and deliberately simple: SDL_GPU doesn't
// expose its device, so DLSS runs on a Direct3D 12 device of its own on the
// NVIDIA GPU, and frames go through memory both ways. The emulator has no
// motion vectors or camera jitter to give it, so they are zero and by
// default the history is reset every frame (no ghosting; DLSS then works as
// a single-frame upscaler and anti-aliaser).
//
// Built when CMake finds the SDK (ORBIT64_DLSS); otherwise available() is
// false and error() says why.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu {

class Dlss {
public:
    enum Mode { Off = 0, DLAA, Quality, Balanced, Performance, UltraPerformance, kModes };
    static const char* mode_name(int mode);
    // Output size for an input of w x h in `mode`.
    static void output_size(int mode, int w, int h, int& ow, int& oh);

    Dlss();
    ~Dlss();
    Dlss(const Dlss&) = delete;
    Dlss& operator=(const Dlss&) = delete;

    // Sets up Direct3D 12 and NGX on the NVIDIA GPU (once; later calls
    // return the first result).
    bool init();
    bool available() const { return available_; }
    const std::string& error() const { return error_; }
    const std::string& adapter() const { return adapter_; }

    // Upscales an ARGB8888 frame of w x h into `out` (ARGB8888, ow x oh).
    // `temporal` keeps DLSS's history between frames.
    bool evaluate(const std::uint32_t* argb, int w, int h, int mode, bool temporal, std::vector<std::uint32_t>& out,
                  int& ow, int& oh);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool tried_ = false, available_ = false;
    std::string error_, adapter_;
};

} // namespace gpu
