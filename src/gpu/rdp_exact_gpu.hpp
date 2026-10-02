#pragma once
// The bit-exact RDP's pixels on the GPU (ExactAccel, see rdp_exact.hpp).
//
// ExactRdp still runs every command on the CPU - state, tiles, TMEM loads,
// triangle setup down to each scanline's span - and records the primitives
// (gpu/shaders/exact_layout.h). The GPU draws them with the same integer
// pipeline as ExactRdp::draw (shaders/exact.glsl), so RDRAM ends up holding
// exactly what the CPU would have drawn:
//
//  - The GPU keeps a copy of RDRAM and its ninth bits. Before drawing, the
//    64-byte blocks of the frame buffers the primitives draw into that
//    changed since the GPU last had them (CPU writes, primitives the CPU
//    drew) are uploaded; at a flush those frame buffers are read back and
//    what the GPU changed goes to RDRAM and ExactRdp's ninth bits.
//  - Primitives are binned by 8x8-pixel tile of their colour image.
//    exact_shade*.comp shade every (tile, primitive) pair at once - all that
//    doesn't read memory, in variants by state; then per run of primitives
//    into the same colour and depth image, exact_memory.comp runs each
//    pixel's primitives in drawing order: depth test, blender, store.
//  - submit() starts drawing what is recorded without waiting, so the GPU
//    works while the CPU emulates on; flush() waits and reads back. The
//    RDP front end flushes wherever RDRAM must hold the result (Sync Full,
//    a texture load from a frame buffer being drawn, primitives the CPU
//    draws, and - reported through jit::notify_read / the write hook - CPU
//    and DMA reads and writes of it), so RDRAM is all there is between
//    flushes (save states, the VI, the CPU see what they would with the CPU
//    RDP).
//  - Internal resolution S > 1 (as ExactRdp's UpStore): every primitive is
//    drawn a second time at S times the resolution into a high-resolution
//    copy of the frame buffers' 4 KB pages that lives on the GPU only
//    (exact_layout.h); copy- and fill-mode primitives repeat their
//    native pixels instead. What changes in RDRAM otherwise goes to every
//    sample (exact_apply.comp); the VI gets the copy through
//    fetch_upscaled().

#include "../rdp_exact.hpp"
#include "device.hpp"
#include "shaders/exact_layout.h"

#include <memory>
#include <vector>

namespace gpu {

class ExactRdpGpu final : public ExactAccel {
public:
    explicit ExactRdpGpu(std::shared_ptr<Device> device);
    ~ExactRdpGpu() override;

    void record(const ExactRecord& r) override;
    u32 pending() const override { return static_cast<u32>(prims_.size() / EXACT_PRIM_WORDS); }
    bool dirty() const override { return pending() != 0 || unsynced_; }
    void submit(ExactRdp& front, u8* rdram, size_t rdram_size) override;
    void flush(ExactRdp& front, u8* rdram, size_t rdram_size) override;
    bool touches(u64 paddr, u64 len) const override;
    void invalidate(u64 paddr, u64 len) override;
    void invalidate_all() override;
    void set_scale(u32 scale) override;
    void fetch_upscaled(u64 lo, u64 hi, UpStore& up) override;
    void store_upscaled(ExactRdp& front, u64 lo, u64 hi, const UpStore& up, const u8* rdram) override;

private:
    // A run of primitives into one colour and depth image.
    struct Pass {
        u32 first = 0, count = 0; // primitives
        u32 fb_index = 0, fb_width = 0, fb_fmt = 0, z_index = 0;
        bool uses_z = false;
        s32 min_row = 0x7fffffff, max_row = -1;
    };
    // The bins of every pass at one scale (see bin()).
    struct Binned;
    bool ensure_memory(size_t rdram_size);
    SDL_GPUBuffer* make_buffer(u32 bytes, const char* name);
    void reset_recording();
    void protect(u64 lo, u64 hi, const u8* rdram);
    void bin(u32 scale, u32 prims_at, const std::vector<u8>& variant, Binned& out);
    // Waits for what was submitted and brings RDRAM up to date.
    void sync(ExactRdp& front, u8* rdram);
    void drop_slots();
    // Grows the high-resolution copy's buffers to slots_used_ slots (keeping
    // what they hold, in `copy`); false if that failed.
    bool grow_slots(SDL_GPUCopyPass* copy);
    void report_stats();

    std::shared_ptr<Device> dev_;
    SDL_GPUDevice* gpu_;

    // What the GPU's copy of RDRAM holds, per 64-byte block when valid_
    // (and not drawn into since: gpu_dirty_).
    size_t size_ = 0;
    std::vector<u8> ref_, ref_hidden_, valid_, gpu_dirty_;
    u64 epoch_ = ~0ull;
    // Submitted and not read back yet: what was drawn into, and whether the
    // last walked pixel's memory colour is on its way (tail_down_).
    bool unsynced_ = false;
    std::vector<std::pair<u64, u64>> dirty_ranges_;
    bool tail_queued_ = false;

    // What the recorded primitives see of RDRAM: what it held when the first
    // of them reaching each block was recorded (pre_, in the blocks snap_
    // marks, which cycle_runs_ list) - the CPU may write it before they are
    // drawn, but its writes come after them. Reads of those blocks, and of
    // the ones drawing (gpu_dirty_), are reported (jit::watch_reads,
    // flag_runs_ lists what is watched).
    std::vector<u8> pre_, snap_;
    std::vector<std::pair<u64, u64>> cycle_runs_, flag_runs_;

    // ---- Internal resolution: the high-resolution copy's slots (one per
    // RDRAM page that frame buffers use, page_slot_ the page table) and the
    // pages that got one since the last submission.
    u32 scale_ = 1;
    std::vector<u32> page_slot_;
    std::vector<u32> new_pages_;
    u32 slots_used_ = 0, slots_cap_ = 0;
    bool pages_dirty_ = true;
    SDL_GPUBuffer* up_color_ = nullptr;  // slots * S * S pages of bytes
    SDL_GPUBuffer* up_hidden_ = nullptr; // slots * S * S half pages of ninth bits (a byte per halfword)
    SDL_GPUBuffer* pages_buf_ = nullptr; // page_slot_
    SDL_GPUTransferBuffer* fetch_down_ = nullptr;
    u32 fetch_bytes_ = 0;
    u32 slot_bytes() const { return scale_ * scale_ * (1u << EXACT_PAGE_SHIFT); }

    // ---- Recorded since the last submission (exact_layout.h; offsets
    // relative to each array until submit() lays them out in one buffer).
    std::vector<u32> states_, prims_, spans_, tmem_;
    std::vector<Pass> passes_;
    u32 last_state_ = ~0u;
    u64 last_tmem_gen_ = ~0ull;
    u32 last_tmem_ = ~0u;
    u32 tail_slots_ = 0;
    s32 last_tail_slot_ = -1;

    // ---- GPU resources
    SDL_GPUBuffer* rdram_buf_ = nullptr;  // RDRAM bytes
    SDL_GPUBuffer* hidden_buf_ = nullptr; // a byte of ninth bits per halfword
    SDL_GPUBuffer* data_buf_ = nullptr;
    u32 data_bytes_ = 0;
    SDL_GPUBuffer* tails_buf_ = nullptr;  // per walking primitive (and the debug log)
    SDL_GPUBuffer* shaded_buf_ = nullptr; // exact_shade*.comp's output
    u32 shaded_bytes_ = 0;
    SDL_GPUTransferBuffer* up_ = nullptr;
    u32 up_bytes_ = 0;
    SDL_GPUTransferBuffer* down_ = nullptr;
    u32 down_bytes_ = 0;
    SDL_GPUTransferBuffer* tail_down_ = nullptr; // the last tail and the debug log

    // Scratch.
    std::vector<std::pair<u64, u64>> ranges_;
    std::vector<std::pair<u32, u32>> runs_;
    std::vector<u32> data_, tile_count_, tile_first_, entries_, pairs_;

    // ORBIT64_GPU_STATS=1: per-second figures on stderr.
    struct Stats {
        bool on = false;
        u64 submits = 0, syncs = 0, prims = 0, passes = 0, chunks = 0, up_bytes = 0, down_bytes = 0, entries = 0;
        u64 variant[Device::kExactVariants] = {};
        double cpu_ms = 0, wait_ms = 0;
        u64 last = 0;
    } stats_;
};

// Makes the GPU back end while `device` has its pipelines.
ExactAccelFactory make_exact_accel_factory(std::shared_ptr<Device> device);

} // namespace gpu
