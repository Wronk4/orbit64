#pragma once
// Internal-resolution rendering on the GPU.
//
// The same recording interface as the CPU renderer (hires.hpp), replayed by
// compute shaders (shaders/raster.comp) instead of worker threads. Every
// displayed frame the emulation thread turns what it recorded into one
// buffer of primitives, draw states and TMEM snapshots (shaders/layout.h),
// bins the primitives by 8x8-pixel tile of their frame buffer and submits
// it: per run of draws into one colour image, one dispatch with a thread per
// output pixel that executes that pixel's primitives in order - so blending,
// depth and every other read of the frame buffer behave as on the CPU. The
// displayed frame is composed on the GPU too, into a texture the frontend
// draws directly (VideoFrame::gpu).
//
// Colour and depth buffers live in video memory only. RDRAM stays
// authoritative exactly as with the CPU renderer: each colour image keeps a
// shadow of what the RDP wrote, and pixels something else changed are
// uploaded before the RDP draws there again, or shown from RDRAM.

#include "../hires.hpp"
#include "device.hpp"

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

namespace gpu {

class Image;

class RdpRenderer final : public HiResRenderer {
public:
    RdpRenderer(std::shared_ptr<Device> device, u32 scale);
    ~RdpRenderer() override;

    HiResTarget* bind(u32 addr, u32 width, u8 size, const u8* rdram, size_t rdram_size) override;
    void triangle(HiResTarget* t, const DrawState& st, u64 serial, u64 tmem_gen,
                  const Vertex& v0, const Vertex& v1, const Vertex& v2, f32 area) override;
    void tex_rect(HiResTarget* t, const DrawState& st, u64 serial, u64 tmem_gen, u32 ulx, u32 uly, u32 lrx,
                  u32 lry, u32 tile, f32 s, f32 tc, f32 dsdx, f32 dtdy, bool flip) override;
    void fill_rect(HiResTarget* t, u32 x0, u32 y0, u32 x1, u32 y1, u32 argb) override;
    void pixels(HiResTarget* t, const DrawState& st, u64 serial, const std::vector<Pixel>& px) override;
    void blit(HiResTarget* t, const DrawState& st, u64 serial, u32 x0, u32 y0, u32 w, u32 h, const u32* colors) override;
    void clear_depth() override;
    bool supports_ray_tracing() const override { return dev_->rt_ok(); }
    void ray_trace(HiResTarget* t, const RtPass& pass) override;
    void flush() override;
    bool present(const VIScanout& so, const u8* rdram, size_t rdram_size, VideoFrame& out) override;
    void end_frame() override;

private:
    struct Depth;
    struct Target : HiResTarget {
        SDL_GPUBuffer* buffer = nullptr; // (width * S) x (240 * S) ARGB8888
        Depth* last_depth = nullptr; // what its last triangle tested against
    };
    struct Depth {
        u32 width = 0;
        SDL_GPUBuffer* buffer = nullptr; // (width * S) x (240 * S) floats
        u64 last_used = 0;
    };
    // A primitive's frame buffer and the pixels it can touch (at scale S, inclusive).
    struct PrimInfo {
        Target* target;
        Depth* depth;
        u16 x0, x1, y0, y1;
    };
    // Sets a whole buffer to one value before primitive `at` is drawn.
    struct BufferFill {
        size_t at;
        SDL_GPUBuffer* buffer;
        u32 words, value;
        int rt = -1; // or a ray tracing pass instead: index in rt_jobs_
    };
    struct RtJob {
        Target* target;
        Depth* depth;
        u32 params[44]; // RtParams
    };
    std::vector<RtJob> rt_jobs_;
    SDL_GPUBuffer* occ_ = nullptr; // rt_trace.comp's output
    u32 occ_words_ = 0;
    struct Compose;

    Target* find(u32 addr, u32 width, u8 size) const;
    Depth* depth_for(const DrawState& st);
    SDL_GPUBuffer* make_buffer(u32 words, const char* name);
    // Word offset of the recorded copy of `st` (see CpuHiResRenderer::recorded_state).
    u32 recorded_state(const DrawState& st, u64 serial, u64 tmem_gen, bool needs_tmem);
    // Gives tile `tile` of a recorded state decoded texels (the CPU renderer's
    // TexCache), shared by every state of the batch that samples the same
    // TMEM contents through the same tile.
    void attach_table(u32 state, const DrawState& st, u32 tile, u64 tmem_gen);
    // Appends a primitive of `type` and returns its words; x1/y1 inclusive.
    u32* add_prim(u32 type, Target* t, Depth* d, u32 state, s32 x0, s32 y0, s32 x1, s32 y1);
    void add_upload(Target* t, const std::vector<hires::ChangedPixel>& px);
    // Draws everything recorded (and composes `c`, if given) on the GPU.
    void submit(const Compose* c);
    void reset_batch();
    std::shared_ptr<Image> acquire_image(u32 w, u32 h);

    std::shared_ptr<Device> dev_;
    SDL_GPUDevice* gpu_;

    std::vector<std::unique_ptr<Target>> targets_;
    std::vector<std::unique_ptr<Depth>> depths_;
    std::vector<hires::ChangedPixel> changed_;
    u64 frame_ = 0;

    // ---- The batch being recorded (see shaders/layout.h)
    std::vector<u32> data_;      // lookup table, states, TMEM snapshots, pixel data
    std::vector<u32> prims_;     // GPU_PRIM_WORDS each
    std::vector<PrimInfo> info_; // one per primitive
    std::vector<BufferFill> fills_;
    static constexpr u32 kNone = ~0u;
    u32 rec_state_ = kNone;
    u64 rec_serial_ = ~0ull, rec_tmem_gen_ = ~0ull;
    u32 tmem_copy_ = kNone;
    u64 tmem_copy_gen_ = ~0ull;
    struct TableKey {
        u64 tmem_gen, a, b;
        bool operator==(const TableKey& o) const { return tmem_gen == o.tmem_gen && a == o.a && b == o.b; }
    };
    struct TableKeyHash {
        size_t operator()(const TableKey& k) const {
            return static_cast<size_t>((k.tmem_gen * 0x9E3779B97F4A7C15ull) ^ (k.a * 0xC2B2AE3D27D4EB4Full) ^ k.b);
        }
    };
    std::unordered_map<TableKey, u32, TableKeyHash> tables_; // -> offset in the texel buffer
    std::vector<u32> jobs_;  // GPU_JOB_WORDS per table to decode
    u32 texel_words_ = 0;    // texel buffer words the batch uses
    u32 decode_blocks_ = 0;

    // ---- GPU resources
    SDL_GPUBuffer* data_buf_ = nullptr;
    u32 data_buf_words_ = 0;
    SDL_GPUTransferBuffer* upload_ = nullptr;
    u32 upload_words_ = 0;
    SDL_GPUBuffer* no_depth_ = nullptr; // bound when a pass has no depth buffer
    SDL_GPUBuffer* texels_ = nullptr;   // decoded textures
    u32 texel_buf_words_ = 0;
    std::deque<SDL_GPUFence*> in_flight_;
    std::vector<std::shared_ptr<Image>> images_;
    // Scratch for submit().
    std::vector<u32> bin_count_, bin_first_;

    // ORBIT64_GPU_STATS=1: per-second figures on stderr. ORBIT64_GPU_SYNC=1
    // also waits for every submission, so "gpu" is the GPU's own time.
    struct Stats {
        bool on = false, sync = false;
        u64 frames = 0, prims = 0, passes = 0, bin_entries = 0, data_words = 0, work_tiles = 0;
        double submit_ms = 0, wait_ms = 0;
        u64 last = 0; // SDL_GetTicksNS of the last report
    } stats_;
};

// Makes GPU renderers while `device` has working pipelines.
HiResFactory make_hires_factory(std::shared_ptr<Device> device);

} // namespace gpu
