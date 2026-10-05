#pragma once
// Internal-resolution rendering.
//
// The RDP (rdp.cpp) draws every primitive into RDRAM at the game's frame
// buffer size, as the console does, and also records it here. A renderer
// replays the recorded draws at `scale` times that size into a
// high-resolution copy of each colour image, and the displayed frame is
// composed from those copies. CpuHiResRenderer (below) replays them on
// worker threads; the GPU renderer (src/gpu/rdp_gpu.hpp) in compute shaders.
//
// RDRAM stays authoritative. Each high-resolution buffer keeps a shadow of
// the value the RDP last wrote to every RDRAM pixel. Pixels the CPU or a DMA
// changed since then come from RDRAM instead, so CPU-drawn screens, overlays
// and frame buffer effects still appear. Such changes are also copied into
// the high-resolution buffer before the RDP draws into it again.
//
#include "raster.hpp"
#include "video_frame.hpp"
#include "vi.hpp"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// Frame buffer pixel as stored in RDRAM (big-endian), widened to 32 bits.
inline u32 fb_read_pixel(const u8* p, u32 bpp) {
    if (bpp == 2) return (static_cast<u32>(p[0]) << 8) | p[1];
    return (static_cast<u32>(p[0]) << 24) | (static_cast<u32>(p[1]) << 16) | (static_cast<u32>(p[2]) << 8) | p[3];
}

// RDRAM pixel -> ARGB8888 the way the VI shows it. For 16-bit pixels the
// coverage bit becomes alpha 0/255, which the blender reads as memory alpha.
inline u32 fb_pixel_to_argb(u32 raw, u32 bpp) {
    if (bpp == 2) {
        u32 r = ((raw >> 11) & 0x1F) * 255 / 31;
        u32 g = ((raw >> 6) & 0x1F) * 255 / 31;
        u32 b = ((raw >> 1) & 0x1F) * 255 / 31;
        u32 a = (raw & 1) ? 0xE0 : 0; // full coverage, or none
        return (a << 24) | (r << 16) | (g << 8) | b;
    }
    return ((raw & 0xFF) << 24) | (raw >> 8);
}

struct HiResTarget {
    u32 addr = 0;            // colour image address (physical)
    u32 width = 0;           // colour image width in pixels
    u8 size = 2;             // 2 = RGBA5551, 3 = RGBA8888
    std::vector<u32> color;  // (width * scale) x (240 * scale), 0xAARRGGBB
    std::vector<u32> shadow; // width x 240: what the RDP last wrote to each RDRAM pixel
    u64 last_used = 0;       // frame counter
    u64 last_cmd = 0;        // 1 + index of the last command that uses it (0: none)
};

// High-resolution depth buffer. There is one per frame buffer width (the
// RDP's own depth buffer is shared by every colour image the same way).
struct HiResDepth {
    u32 width = 0;
    std::vector<f32> z; // (width * scale) x (240 * scale)
    u64 last_used = 0;
    u64 last_cmd = 0;
};

// The part of the RDP's frame buffer handling both renderers share.
namespace hires {

// The colour image the VI is showing among `targets` (a range of pointers to
// HiResTarget). libultra's VI modes often point the origin one or more lines
// into the frame buffer, so a target also matches when the scan-out starts a
// whole number of lines past its address.
template <class Range>
HiResTarget* find_scanout(const Range& targets, const VIScanout& so, u32& first_line) {
    const u8 size = so.bpp == 2 ? 2 : 3;
    const u32 line_bytes = so.width * so.bpp;
    HiResTarget* best = nullptr;
    for (const auto& t : targets) {
        if (t->width != so.width || t->size != size || so.fb_base < t->addr) continue;
        const u32 off = so.fb_base - t->addr;
        if (off % line_bytes != 0 || off / line_bytes >= kFbLines) continue;
        if (!best || t->addr > best->addr) { // the closest start wins
            best = &*t;
            first_line = off / line_bytes;
        }
    }
    return best;
}

// A pixel RDRAM holds that the RDP didn't draw (native coordinates).
struct ChangedPixel { u16 x, y; u32 argb; };

// Brings a target's shadow up to date with RDRAM and lists the pixels that
// differed: what the CPU or a DMA wrote since the RDP last drew there.
void scan_changes(HiResTarget& t, const u8* rdram, size_t rdram_size, std::vector<ChangedPixel>& changed);

} // namespace hires

// What the RDP records into; see the top of this file.
// Per-pixel ray tracing of a finished 3D scene (RDP::rt_pixel_flush): the
// scene's BVH (RayTracingScene::serialize) and how to get from a pixel of the
// target and its depth back into the space the BVH is in.
struct RtPass {
    std::vector<u32> bvh;      // nodes (8 words each), then triangles (12 words each)
    u32 tri_offset = 0;        // word offset of the triangles in bvh
    f32 proj[16] = {};         // modelview space -> clip (row vectors: clip = v * proj)
    f32 inv_proj[16] = {};     // clip -> modelview space
    f32 vp[6] = {};            // viewport scale x, y, z and translation x, y, z (RDP::compute_screen_coords)
    f32 sun[3] = {0, 1, 0};    // toward the main light
    f32 eye[3] = {};           // the camera
    f32 scene_scale = 1.0f;    // RayTracingScene::scale()
    f32 shadow_strength = 0.55f, ao_strength = 1.0f;
    f32 specular = 0.35f;      // per-pixel highlights
    bool pixel_lighting = true; // relight lit triangles per pixel (RtPass::bvh's light sets)
    u32 lights_offset = 0;     // word offset of the light sets in bvh
    f32 water_reflect = 0.0f, metal_reflect = 0.0f; // reflection strengths (0: none)
    f32 gi = 0.0f;             // global illumination strength (0: ambient occlusion only)
    // 0 fast (shadow and bounce rays once per native pixel), 1 balanced
    // (twice as dense each way), 2 ultra (every pixel, more rays).
    u32 quality = 0;
    f32 sun_cone = 0.06f;            // shadow softness (radians)
    f32 sun_color[3] = {1, 1, 1};    // times its strength
    f32 shafts = 0.0f, haze = 0.0f, flare = 0.0f, cool_shade = 0.0f;
    f32 sun_screen[3] = {};          // native screen position, 1 when in front of the eye
    u64 tex_batch = 0;         // the batch the cut-outs' texture states are valid in
    u32 frame = 0;
};

class HiResRenderer {
public:
    static constexpr u32 kMaxScale = 8;
    static constexpr u32 kMaxWidth = 1024; // wider colour images are left at native resolution

    virtual ~HiResRenderer() = default;
    HiResRenderer(const HiResRenderer&) = delete;
    HiResRenderer& operator=(const HiResRenderer&) = delete;

    u32 scale() const { return scale_; }

    // ---- Recording (emulation thread) -------------------------------------
    // The high-resolution buffer of a colour image, created from RDRAM on first
    // use. After unbind(), the next bind() first copies in whatever changed in
    // RDRAM without the RDP drawing it. Returns nullptr for unsupported images.
    // The RDP keeps the target's shadow up to date as it draws.
    virtual HiResTarget* bind(u32 addr, u32 width, u8 size, const u8* rdram, size_t rdram_size) = 0;
    void unbind() { bound_ = nullptr; }

    // `serial` changes whenever the RDP state `st` was built from changes, and
    // `tmem_gen` whenever TMEM does, so consecutive draws share one recorded
    // copy of both.
    virtual void triangle(HiResTarget* t, const DrawState& st, u64 serial, u64 tmem_gen,
                          const Vertex& v0, const Vertex& v1, const Vertex& v2, f32 area) = 0;
    // Edges in quarter pixels (raster::tex_rect()).
    virtual void tex_rect(HiResTarget* t, const DrawState& st, u64 serial, u64 tmem_gen, u32 ulx, u32 uly, u32 lrx,
                          u32 lry, u32 tile, f32 s, f32 tc, f32 dsdx, f32 dtdy, bool flip) = 0;
    // FILL-mode rectangle, native pixel bounds [x0, x1) x [y0, y1).
    virtual void fill_rect(HiResTarget* t, u32 x0, u32 y0, u32 x1, u32 y1, u32 argb) = 0;
    // Pixels shaded at native resolution (lines), each drawn as a scale x scale block.
    struct Pixel { u16 x, y; u32 color; f32 z; };
    virtual void pixels(HiResTarget* t, const DrawState& st, u64 serial, const std::vector<Pixel>& px) = 0;
    // A block of colours shaded at native resolution and drawn at depth 0 (S2DEX backgrounds).
    virtual void blit(HiResTarget* t, const DrawState& st, u64 serial, u32 x0, u32 y0, u32 w, u32 h,
                      const u32* colors) = 0;
    virtual void clear_depth() = 0;
    // Ray traces shadows and ambient occlusion into what was drawn into `t`
    // so far (renderers that can; the RDP lights vertices itself otherwise).
    virtual bool supports_ray_tracing() const { return false; }
    virtual void ray_trace(HiResTarget* t, const RtPass& pass) { (void)t; (void)pass; }
    // Records `st` (with its texture) for ray_trace() to test texture alpha
    // with: its offset, or ~0u. Valid while rt_batch() stays the same.
    virtual u32 rt_texture_state(const DrawState& st, u64 serial, u64 tmem_gen) {
        (void)st; (void)serial; (void)tmem_gen;
        return ~0u;
    }
    virtual u64 rt_batch() const { return 0; }
    // Starts drawing everything recorded so far.
    virtual void flush() = 0;

    // ---- Output (emulation thread) ----------------------------------------
    // The displayed frame: the scanned-out part of the frame buffer placed on
    // the VI's canvas, both `scale` times their size (VIScanout::place).
    // Draws into the buffer the game works on next may still be going on.
    // Returns false when no high-resolution buffer covers the scanned-out
    // frame buffer.
    virtual bool present(const VIScanout& so, const u8* rdram, size_t rdram_size, VideoFrame& out) = 0;
    // Once per displayed frame, after present(): frees buffers the game
    // stopped using.
    virtual void end_frame() = 0;

protected:
    explicit HiResRenderer(u32 scale) : scale_(std::clamp<u32>(scale, 1, kMaxScale)) {}

    const u32 scale_;
    HiResTarget* bound_ = nullptr;
};

// Makes the renderer for a scale; the RDP uses CpuHiResRenderer without one.
using HiResFactory = std::function<std::unique_ptr<HiResRenderer>(u32 scale)>;

// Replays the recorded draws on worker threads. Each worker takes whole
// horizontal bands of output rows and executes every recorded command for its
// band in order. Bands never share pixels, so the workers need no locking
// beyond claiming a band.
class CpuHiResRenderer final : public HiResRenderer {
public:
    explicit CpuHiResRenderer(u32 scale);
    ~CpuHiResRenderer() override;

    HiResTarget* bind(u32 addr, u32 width, u8 size, const u8* rdram, size_t rdram_size) override;
    void triangle(HiResTarget* t, const DrawState& st, u64 serial, u64 tmem_gen,
                  const Vertex& v0, const Vertex& v1, const Vertex& v2, f32 area) override;
    void tex_rect(HiResTarget* t, const DrawState& st, u64 serial, u64 tmem_gen, u32 ulx, u32 uly, u32 lrx,
                  u32 lry, u32 tile, f32 s, f32 tc, f32 dsdx, f32 dtdy, bool flip) override;
    void fill_rect(HiResTarget* t, u32 x0, u32 y0, u32 x1, u32 y1, u32 argb) override;
    void pixels(HiResTarget* t, const DrawState& st, u64 serial, const std::vector<Pixel>& px) override;
    void blit(HiResTarget* t, const DrawState& st, u64 serial, u32 x0, u32 y0, u32 w, u32 h, const u32* colors) override;
    void clear_depth() override;
    void flush() override;
    bool present(const VIScanout& so, const u8* rdram, size_t rdram_size, VideoFrame& out) override;
    // Starts a new command segment and frees buffers the game stopped using.
    void end_frame() override;

    // Builds the frame buffer part of the displayed image, (width * scale) x
    // (240 * scale) ARGB8888, once the draws into the displayed buffer are
    // done. (Draws into the buffer the game is working on next carry on in
    // the background.)
    bool compose(const VIScanout& so, const u8* rdram, size_t rdram_size, std::vector<u32>& out, int& out_w, int& out_h);

private:
    enum class CmdType : u8 { Triangle, TexRect, Fill, Pixels, Blit, Upload, DepthClear };
    struct RVert { f32 sx, sy, sz, w, u, v; u8 r, g, b, a; };
    using UploadPx = hires::ChangedPixel;
    struct TriData { RVert v[3]; f32 area; };
    struct RectData { u32 ulx, uly, lrx, lry, tile; f32 s, t, dsdx, dtdy; bool flip; };
    struct FillData { u32 x0, y0, x1, y1, argb; };
    struct ListData { const void* items; u32 count; };
    struct BlitData { const u32* colors; u32 x0, y0, w, h; };
    struct ClearData { HiResDepth* const* planes; u32 count; };
    struct Cmd {
        CmdType type;
        HiResTarget* target;
        HiResDepth* depth;
        const DrawState* st;
        union {
            TriData tri;
            RectData rect;
            FillData fill;
            ListData list;
            BlitData blit;
            ClearData clear;
        };
    };
    struct TmemCopy { std::array<u8, 4096> data; std::array<bool, 512> dxt; };

    // Bump allocator for data commands point to. Blocks never move, so
    // workers can read them while more is recorded.
    class Arena {
    public:
        void* alloc(size_t bytes);
        template <class T> T* make(const T& v) { return new (alloc(sizeof(T))) T(v); }
        void reset();
        size_t used() const { return used_; }
    private:
        static constexpr size_t kBlock = 1 << 20;
        std::vector<std::unique_ptr<u8[]>> blocks_;
        std::vector<size_t> sizes_;
        size_t cur_ = 0, off_ = 0, used_ = 0;
    };

    // Commands are numbered from 0 up and stored in a ring of segments, one
    // per displayed frame (more if a frame records a lot). A segment is only
    // reused once every band has executed all of it, so the workers can still
    // be drawing one frame while the next is recorded.
    struct Segment {
        std::atomic<u64> start{0};
        std::atomic<u64> end{0}; // ~0 while it is the segment being recorded
        std::unique_ptr<std::unique_ptr<Cmd[]>[]> chunks;
        Arena arena;
    };
    static constexpr int kSegments = 4;
    static constexpr u32 kChunkBits = 12;
    static constexpr u32 kChunkSize = 1u << kChunkBits;
    static constexpr u32 kMaxChunks = 64;              // 256K commands per segment
    static constexpr size_t kMaxArenaBytes = 64u << 20; // per segment

    struct alignas(64) Band {
        std::atomic<bool> busy{false};
        std::atomic<u64> next{0}; // next command index to execute
        int seg = 0;              // segment of the last command executed (lookup hint)
        s32 y0 = 0, y1 = 0;       // output rows [y0, y1)
    };

    const Cmd& cmd_at(u64 i, int& seg_hint) const;
    Arena& arena() { return segs_[cur_].arena; }
    void reserve();              // makes room for one more command
    Cmd& append(CmdType type, HiResTarget* t, const DrawState* st, HiResDepth* d);
    void publish();
    void rotate();               // closes the segment being recorded and starts the next
    DrawState* recorded_state(const DrawState& st, u64 serial, u64 tmem_gen, bool needs_tmem);
    void attach_tex_cache(DrawState* rs, u32 tile, u64 tmem_gen);
    HiResDepth* depth_for(const DrawState& st);
    HiResTarget* find(u32 addr, u32 width, u8 size) const;

    u64 executed() const;        // number of commands every band has executed
    void wait_executed(u64 n);   // until executed() >= n; the calling thread helps
    bool run_bands(int first, u64 limit);
    void worker_main(int index);
    void execute(const Cmd& c, s32 y0, s32 y1) const;

    Segment segs_[kSegments];
    int cur_ = 0;                // segment being recorded
    u64 count_ = 0;              // commands recorded
    u64 notified_ = 0;           // count_ when the workers were last woken
    std::atomic<u64> published_{0};

    DrawState* rec_state_ = nullptr;
    u64 rec_serial_ = ~0ull, rec_tmem_gen_ = ~0ull;
    const TmemCopy* tmem_copy_ = nullptr;
    u64 tmem_copy_gen_ = ~0ull;
    // Decoded textures of the segment being recorded, by TMEM contents + tile.
    struct TexKey {
        u64 tmem_gen, a, b;
        bool operator==(const TexKey& o) const { return tmem_gen == o.tmem_gen && a == o.a && b == o.b; }
    };
    struct TexKeyHash {
        size_t operator()(const TexKey& k) const { return static_cast<size_t>((k.tmem_gen * 0x9E3779B97F4A7C15ull) ^ (k.a * 0xC2B2AE3D27D4EB4Full) ^ k.b); }
    };
    std::unordered_map<TexKey, TexCache*, TexKeyHash> tex_caches_;

    std::vector<std::unique_ptr<HiResTarget>> targets_;
    std::vector<std::unique_ptr<HiResDepth>> depths_;
    std::vector<UploadPx> changed_; // scratch for bind()
    std::vector<u32> composed_;     // scratch for present()
    u64 frame_ = 0;

    std::unique_ptr<Band[]> bands_;
    int nbands_ = 0;
    int nworkers_ = 0;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> quit_{false};
};
