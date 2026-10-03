#pragma once
// Bit-exact low-level RDP: draws the RDP's own commands (what a real
// microcode on the low-level RSP sends) the way the hardware does, in
// integer arithmetic - edge walker, 8-sample coverage, attribute
// interpolation, perspective division through the hardware reciprocal table,
// TMEM and its loads, texture filtering, colour combiner, blender with its
// divider, dithering, the 18-bit compressed depth buffer and the RDRAM's
// hidden ninth bits (coverage and depth slope). Only at native resolution.
//
// The pixel pipeline follows parallel-rdp by Themaister (MIT licence, see
// rdp_exact.cpp), which reproduces Angrylion's reference RDP bit for bit.
//
// Internal resolution (set_scale), the way parallel-rdp upscales: RDRAM
// still gets the native, bit-exact picture the game sees, and every
// primitive is drawn a second time at `scale` times the resolution into a
// separate copy of memory (UpStore) that only the VI shows. Copy- and
// fill-mode primitives are drawn natively and their pixels repeated.

#include "common.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

// The high-resolution copy of memory: pixel (x, y) of a frame buffer at
// `scale` times its size is sample (x % scale, y % scale) of native pixel
// (x / scale, y / scale), kept in slice (y % scale) * scale + x % scale, a
// whole RDRAM-sized array addressed like RDRAM. The OS only backs the pages
// that are touched (the frame buffers).
struct UpStore {
    UpStore(u32 scale, size_t rdram_size);
    ~UpStore();
    UpStore(const UpStore&) = delete;
    UpStore& operator=(const UpStore&) = delete;
    u8* color(u32 slice) const { return color_ + slice * size; }         // RDRAM bytes
    u8* hidden(u32 slice) const { return hidden_ + slice * (size / 2); } // ninth bits per halfword
    u32 scale;
    size_t size;
    // RDRAM as it was when the copy last agreed with it (same layout): where
    // RDRAM differs, the CPU wrote it since (see ExactRdp::up_sync_before).
    u8* ref;

private:
    u8 *color_, *hidden_;
    size_t color_bytes_, hidden_bytes_, ref_bytes_;
};

// A primitive as the bit-exact RDP hands it to an accelerated back end
// (ExactAccel) instead of drawing it: words laid out as in
// gpu/shaders/exact_layout.h.
struct ExactRecord {
    const u32* state; // EXACT_STATE_WORDS (ES_TMEM left for the back end)
    const u32* prim;  // EXACT_PRIM_WORDS (EP_STATE, EP_SPANS and EP_TAIL_SLOT too)
    const s32* spans; // prim[EP_ROWS] spans of EXACT_SPAN_WORDS
    const u16* tmem;  // 2048 halfwords, when the primitive reads TMEM (else null)
    u64 tmem_gen;     // changes whenever TMEM does
    const u8* rdram;  // RDRAM as it is when the primitive is recorded
    size_t rdram_size;
};
class ExactRecorder {
public:
    virtual ~ExactRecorder() = default;
    virtual void record(const ExactRecord& r) = 0;
};

class ExactRdp {
public:
    ExactRdp();
    void reset();

    // ---- Internal resolution (see above). 1 = native only.
    void set_scale(u32 scale) {
        scale_ = scale < 1 ? 1 : scale;
        up_.reset();
    }
    u32 scale() const { return scale_; }
    void share_upscaled_with(const ExactRdp& o) {
        scale_ = o.scale_;
        up_ = o.up_;
    }
    // The high-resolution copy, allocated once RDRAM's size is known.
    UpStore* upscaled(size_t rdram_size);
    const UpStore* upscaled() const { return up_.get(); }
    // Around drawing into [lo, hi): what the CPU changed there since goes to
    // every sample (before), and the copy then agrees with RDRAM (after).
    void up_sync_before(u64 lo, u64 hi, const u8* rdram, size_t rdram_size);
    void up_sync_after(u64 lo, u64 hi, const u8* rdram, size_t rdram_size);

    // Number of 64-bit words of the command whose first byte is `op` (6 bits).
    static u32 command_words(u32 op);
    // Runs one command; `w` holds its 32-bit words (w[0] = the upper half of
    // the first 64-bit word). Returns true for Sync Full.
    bool command(const u32* w, u8* rdram, size_t rdram_size);

    // The ninth bits (2 per RDRAM halfword; 4 = never written by the RDP).
    const std::vector<u8>& hidden() const { return hs_->bits; }
    // Allocates the ninth bits for an RDRAM of that size (all "never written").
    void ensure_hidden(size_t rdram_size) {
        if (hs_->bits.size() == rdram_size / 2) return;
        ++hs_->epoch;
        hs_->bits.assign(rdram_size / 2, 4);
        hs_->word.assign(rdram_size / 2, 0);
        hs_->watched.assign(rdram_size / 64, 0);
    }
    // For tools (trace replay) and the high-level renderer: set halfword h's
    // ninth bits as RDP-written (valid while it holds `word`).
    void force_hidden(size_t h, u8 bits, u16 word) {
        if (h < hs_->bits.size()) {
            if (hs_->bits[h] != bits || hs_->word[h] != word) ++hs_->epoch;
            hs_->bits[h] = bits;
            hs_->word[h] = word;
        }
    }

    // force_hidden() for a renderer that draws on several threads: it leaves the epoch alone (a shared counter
    // every pixel wrote to), and the caller calls bump_hidden_epoch() once when the pixels are in.
    void set_hidden(size_t h, u8 bits, u16 word) {
        if (h < hs_->bits.size()) {
            hs_->bits[h] = bits;
            hs_->word[h] = word;
        }
    }
    void bump_hidden_epoch() { ++hs_->epoch; }

    // ---- Accelerated back end: primitives are recorded (see ExactRecord)
    // instead of drawn; everything else runs here as usual.
    void set_recorder(ExactRecorder* r) { recorder_ = r; }
    // Internal resolution S > 1 of the back end: primitives are recorded with
    // their spans at S times the resolution too, and what is drawn here is
    // drawn natively (the back end repeats it in its high-resolution copy,
    // which up_fetch, when set, brings into upscaled() for the VI).
    void set_record_scale(u32 s) { record_scale_ = s; }
    void set_native_draws(bool on) { native_draws_ = on; }
    using UpFetch = std::function<void(u64 lo, u64 hi, const u8* rdram, size_t rdram_size)>;
    void set_up_fetch(UpFetch f) { up_fetch_ = std::move(f); }
    // Whether a Fill Rectangle `w` would read the frame buffer as the
    // previous identical one left it (see fill_rectangle()): it is drawn here.
    bool fill_reads_stale(const u32* w) const;
    // The delayed memory colour of 2-cycle mode (see draw()), which the back
    // end hands over from the last primitive it walked.
    void set_prev_mem(const s32 m[4]) { std::copy(m, m + 4, prev_mem_); }
    // Lanes: the delayed memory colour as this copy has it is the one the
    // next serial primitive starts from.
    void publish_tail() {
        std::lock_guard<std::mutex> lk(hs_->tail_mutex);
        hs_->tail_serial = prim_count_;
        std::copy(prev_mem_, prev_mem_ + 4, hs_->tail_mem);
    }
    // The back end drew RDRAM halfwords [h, h + n), which now hold what
    // `rdram` (all of RDRAM) has there, leaving ninth bits `bits` (0-3 each).
    void gpu_wrote_hidden(size_t h, size_t n, const u8* bits, const u8* rdram);
    // The ninth bits of halfwords [h, h + n) as hidden_at() has them, into `out`.
    void hidden_range(size_t h, size_t n, const u8* rdram, u8* out) const;
    // Bumped whenever the ninth bits change other than by drawing or a
    // CPU write (cpu_wrote): reset, allocation, loading, the HLE renderer.
    u64 hidden_epoch() const { return hs_->epoch; }

    // ---- Lanes (ExactRdpLanes): this copy draws only the rows y with
    // y % lanes == lane, sharing the ninth bits with the other copies.
    void set_lane(u32 lane, u32 lanes) {
        lane_ = lane;
        lanes_ = lanes;
    }
    void share_hidden_with(const ExactRdp& o) { hs_ = o.hs_; }
    // Everything but the lane and the shared ninth bits.
    void copy_state_from(const ExactRdp& o) {
        const u32 lane = lane_, lanes = lanes_;
        auto hs = hs_;
        ExactRecorder* rec = recorder_;
        *this = o;
        lane_ = lane;
        lanes_ = lanes;
        hs_ = std::move(hs);
        recorder_ = rec;
    }
    // The next primitive is drawn whole by lane 0 (and skipped by the others).
    void set_serial(bool on) { serial_ = on; }
    // Whether drawing `cmd` needs the previous row's last pixel (2-cycle,
    // the first blender cycle reading memory) under the current modes.
    bool draw_needs_serial() const;
    // The ninth bits of RDRAM halfword `h`, which holds `word` now.
    u8 hidden_at(size_t h, u16 word) const {
        // What the RDP didn't write - or what was written over since - has the ninth bits a CPU write gives each
        // byte: its least significant bit.
        const HiddenStore& st = *hs_;
        if (h >= st.bits.size() || (st.bits[h] & 4) || st.word[h] != word)
            return static_cast<u8>((((word >> 8) & 1) << 1) | (word & 1));
        return st.bits[h];
    }
    // force_hidden() for halfwords [h, h + n) at once (a fill rectangle).
    void fill_hidden(size_t h, size_t n, u8 bits, u16 word) {
        if (h >= hs_->bits.size() || n == 0) return;
        n = std::min(n, hs_->bits.size() - h);
        ++hs_->epoch;
        std::fill_n(hs_->bits.begin() + static_cast<std::ptrdiff_t>(h), n, bits);
        std::fill_n(hs_->word.begin() + static_cast<std::ptrdiff_t>(h), n, word);
    }
    // The CPU (or a DMA) wrote [paddr, paddr + len): those bytes' ninth bits
    // are their least significant bits again.
    void cpu_wrote(u32 paddr, u32 len);
    const std::vector<u8>& watched_pages() const { return hs_->watched; }

    template <class S> void serialize(S& s) {
        s(tmem_, hs_->bits, hs_->word);
        for (Tile& t : tiles_)
            s(t.slo, t.shi, t.tlo, t.thi, t.offset, t.stride, t.fmt, t.size, t.palette, t.mask_s, t.shift_s,
              t.mask_t, t.shift_t, t.flags, t.raw_mask_s, t.raw_mask_t);
        s(other_h_, other_l_, combine_w0_, combine_w1_, fill_color_, fog_color_, blend_color_, prim_color_, env_color_);
        s(prim_min_level_, prim_lod_frac_, prim_depth_, prim_dz_, convert_, key_center_, key_scale_, key_width_);
        s(scissor_xlo_, scissor_ylo_, scissor_xhi_, scissor_yhi_, scissor_field_, scissor_odd_);
            s(ti_addr_, ti_width_, ti_size_, ti_fmt_, ci_addr_, ci_width_, ci_fmt_, zi_addr_, prev_mem_);
                if constexpr (S::loading) rebuild_watched();
            }
            // State added to save states later (a section of its own): the primitive
            // counter the noise is seeded from, the stale-rectangle memory and the
            // delayed memory colour handed between lanes.
            template <class S> void serialize_extra(S& s) {
                s(prim_count_, stale_.w0, stale_.w1, stale_.idx, stale_.pre, stale_.n, stale_.valid);
                s(hs_->tail_serial, hs_->tail_mem);
            }
            // Lanes: which lane's stale-rectangle memory is the real one (the lane
            // that draws its row reads memory into it; the others don't).
            u32 stale_owner(u32 lanes) const { return stale_.valid ? ((stale_.w1 & 0xfff) >> 2) % lanes : 0; }

private:
    // ---- State set by commands
    struct Tile {
        u32 slo{0}, shi{0}, tlo{0}, thi{0}; // 10.2
        u32 offset{0}, stride{0};           // TMEM bytes
        u8 fmt{0}, size{0}, palette{0};
        u8 mask_s{0}, shift_s{0}, mask_t{0}, shift_t{0}; // masks clamped to 10
        u8 flags{0};                        // TILE_* below
        u8 raw_mask_s{0}, raw_mask_t{0};
    };
    std::array<Tile, 8> tiles_{};
    std::array<u16, 2048> tmem_{}; // halfword h = TMEM bytes 2h (high) and 2h + 1
    struct HiddenStore {
        std::vector<u8> bits;  // the ninth bits of every RDRAM halfword (2 bits each; 4 = never written by the RDP)
        std::vector<u16> word; // the halfword as the RDP wrote it (if it changed since, the CPU wrote it)
        std::vector<u8> watched; // 64-byte pages the RDP wrote ninth bits in (jit::watch_rdram)
        u64 epoch{0};            // see hidden_epoch()
        // Lanes: the delayed memory colour (prev_mem_) after the last row of
        // the latest primitive that walked pixels, for a serial primitive.
        std::mutex tail_mutex;
        u32 tail_serial{0};
        s32 tail_mem[4]{};
    };
    std::shared_ptr<HiddenStore> hs_ = std::make_shared<HiddenStore>();
    u32 lane_{0}, lanes_{1};
    bool serial_{false};
    u32 prim_count_{0};
    u32 scale_{1};
    ExactRecorder* recorder_{nullptr};
    u64 tmem_gen_{0};
    u32 record_scale_{1};
    bool native_draws_{false};
    UpFetch up_fetch_;
    std::shared_ptr<UpStore> up_;
    s32 prev_mem_up_[4]{};    // prev_mem_ of the high-resolution pass

    u32 other_h_{0}, other_l_{0};
    u32 combine_w0_{0}, combine_w1_{0};
    u32 fill_color_{0}, fog_color_{0}, blend_color_{0}, prim_color_{0}, env_color_{0};
    u8 prim_min_level_{0}, prim_lod_frac_{0};
    s32 prim_depth_{0};
    u16 prim_dz_{0};
    std::array<s32, 6> convert_{};
    std::array<u8, 3> key_center_{}, key_scale_{};
    std::array<u16, 3> key_width_{};
    u32 scissor_xlo_{0}, scissor_ylo_{0}, scissor_xhi_{0}, scissor_yhi_{0};
    bool scissor_field_{false}, scissor_odd_{false};
    u32 ti_addr_{0}, ti_width_{1};
    u8 ti_size_{0}, ti_fmt_{0};
    u32 ci_addr_{0}, ci_width_{1};
    u8 ci_fmt_{0}; // FbFmt
    u32 zi_addr_{0};
    // 2-cycle mode: the memory colour of the last pixel walked (see draw()).
    s32 prev_mem_[4]{};

    // After loading the ninth bits: the pages holding RDP-written ones are
    // watched for CPU writes again.
    void rebuild_watched();

    // ---- Primitives
    struct Setup {
        s32 xh{0}, xm{0}, xl{0};
        s32 yh{0}, ym{0}, yl{0};
        s32 dxhdy{0}, dxmdy{0}, dxldy{0};
        u32 flags{0};
        u32 tile{0}; // tile | max LOD level << 3
    };
    struct Attr {
        s32 rgba[4]{}, drgba_dx[4]{}, drgba_de[4]{}, drgba_dy[4]{};
        s32 stzw[4]{}, dstzw_dx[4]{}, dstzw_de[4]{}, dstzw_dy[4]{}; // s, t, z, w
    };
    void draw(Setup& setup, Attr attr, u8* rdram, size_t rdram_size);
    void fill_rectangle(const u32* w, u8* rdram, size_t rdram_size);
    // Back-to-back identical rectangles: without atomic_prim the command
    // processor runs ahead of the pixel pipeline, and the second one reads
    // the framebuffer as it was before the first (see fill_rectangle()).
    struct StaleRect {
        u32 w0{0}, w1{0};
        u32 idx[32]{};  // the footprint, RDRAM halfword indices
        u16 pre[32]{};  // and what memory held before the rectangle
        u8 n{0};
        bool valid{false};
    } stale_{};
    void texture_rectangle(const u32* w, bool flip, u8* rdram, size_t rdram_size);

    // ---- TMEM loads
    enum class LoadMode { Tile, Block, Tlut };
    void load(const u32* w, LoadMode mode, const u8* rdram, size_t rdram_size);
};

class RasterPool;

// A back end that draws the bit-exact RDP's primitives somewhere else (the
// GPU, gpu/rdp_exact_gpu.hpp) from what ExactRdp records, bit for bit as
// ExactRdp would. It keeps its own copy of the frame buffers' memory: before
// drawing it takes what changed in RDRAM, afterwards RDRAM (and the ninth
// bits) get what it drew, so between flushes RDRAM is all there is.
class ExactAccel : public ExactRecorder {
public:
    // Primitives recorded and not handed to the back end yet.
    virtual u32 pending() const = 0;
    // Anything recorded or drawing whose result isn't in RDRAM yet.
    virtual bool dirty() const = 0;
    // Starts drawing what is recorded, from RDRAM as it is now (the result
    // reaches RDRAM with the next flush).
    virtual void submit(ExactRdp& front, u8* rdram, size_t rdram_size) = 0;
    // Draws everything, into RDRAM; `front` (the recording ExactRdp) gets the
    // ninth bits and the delayed memory colour.
    virtual void flush(ExactRdp& front, u8* rdram, size_t rdram_size) = 0;
    // Whether recorded or drawing primitives use [paddr, paddr + len):
    // reading it (a texture load, the CPU, a DMA) or writing it (the CPU, a
    // DMA) has to wait for them (see ExactRdpLanes::cpu_read / cpu_wrote).
    virtual bool touches(u64 paddr, u64 len) const = 0;
    // RDRAM (or its ninth bits) in [paddr, paddr + len) changed in a way its
    // copy can't tell (ninth bits only).
    virtual void invalidate(u64 paddr, u64 len) = 0;
    virtual void invalidate_all() = 0;
    // Internal resolution (ExactRdp::set_scale): S > 1 draws every primitive
    // a second time at S times the resolution into a copy of memory of the
    // back end's own (as ExactRdp does into UpStore).
    virtual void set_scale(u32 scale) = 0;
    // Brings what the copy holds for RDRAM [lo, hi) into `up` (for the VI),
    // with up->ref saying which RDRAM it was drawn over; after a flush.
    virtual void fetch_upscaled(u64 lo, u64 hi, UpStore& up) = 0;
    // The other way round, after a primitive was drawn into RDRAM [lo, hi)
    // and `up` by the CPU: both become the back end's copy.
    virtual void store_upscaled(ExactRdp& front, u64 lo, u64 hi, const UpStore& up, const u8* rdram) = 0;
};
using ExactAccelFactory = std::function<std::unique_ptr<ExactAccel>()>;

// The bit-exact RDP on several threads. Every lane (an ExactRdp) runs every
// command but draws only its own rows, so the result is the same as one
// ExactRdp's. Commands are queued and run in batches; a batch is cut short
// where drawing order across rows matters: before a texture load from
// memory the batch may have drawn into, before the colour or depth image
// changes, and around primitives whose first blender cycle reads memory in
// 2-cycle mode (their pixels depend on the previous row's), which lane 0
// draws alone.
class ExactRdpLanes {
public:
    ExactRdpLanes();
    ~ExactRdpLanes();
    void reset();
    // Queues one command (see ExactRdp::command); true for Sync Full, by
    // which time everything before it has been drawn.
    bool command(const u32* w, u8* rdram, size_t rdram_size);
    // Draws everything queued.
    void flush();
    const ExactRdp& primary() const { return *lanes_[0]; }
    ExactRdp& primary() { return *lanes_[0]; }
    // A CPU/DMA write (see ExactRdp::cpu_wrote); what is queued is drawn first.
    void cpu_wrote(u32 paddr, u32 len);
    // A CPU/DMA read is about to happen (jit::notify_read): what the
    // accelerated back end draws there is finished first.
    void cpu_read(u32 paddr, u32 len);
    // Internal resolution (see ExactRdp::set_scale).
    void set_scale(u32 scale);
    u32 scale() const { return lanes_[0]->scale(); }
    // Draws primitives with `accel` (null: on the CPU lanes).
    void set_accel(std::unique_ptr<ExactAccel> accel);
    bool has_accel() const { return accel_ != nullptr; }
    bool accelerated() const { return accel_on_; }

    template <class S> void serialize(S& s) {
        flush();
        lanes_[0]->serialize(s);
        if constexpr (S::loading) {
            for (size_t i = 1; i < lanes_.size(); ++i) lanes_[i]->copy_state_from(*lanes_[0]);
            if (accel_) accel_->invalidate_all();
        }
    }
    template <class S> void serialize_extra(S& s) {
        flush();
        if constexpr (S::loading) {
            lanes_[0]->serialize_extra(s);
            for (size_t i = 1; i < lanes_.size(); ++i) lanes_[i]->copy_state_from(*lanes_[0]);
            } else {
                lanes_[accel_on_ ? 0 : lanes_[0]->stale_owner(static_cast<u32>(lanes_.size()))]->serialize_extra(s);
            }
            // What the queue knows of the state (which draws run alone, and when
            // a batch ends).
            s(other_h_, other_l_, ti_addr_, ti_width_, ti_size_, ci_addr_, ci_width_, ci_size_, zi_addr_, scissor_yhi_,
              scissor_xhi_);
        }

private:
    std::vector<std::unique_ptr<ExactRdp>> lanes_;
    std::unique_ptr<RasterPool> pool_;
    std::vector<u32> buf_;    // queued commands' words
    std::vector<u32> starts_; // where each starts in buf_
    u8* rdram_{nullptr};
    size_t rdram_size_{0};
    // What the queue needs to know of the state (ahead of the lanes).
    u32 other_h_{0}, other_l_{0};
    u32 ti_addr_{0}, ti_width_{1}, ti_size_{0};
    u32 ci_addr_{0}, ci_width_{1}, ci_size_{2}, zi_addr_{0};
    u32 scissor_yhi_{0}, scissor_xhi_{0}, scissor_ylo_{0};
    u64 written_lo_{~0ull}, written_hi_{0}; // RDRAM the queued draws can write
    // The same as the colour and the depth image ranges, which the
    // high-resolution copy is brought up to date in around each batch.
    std::vector<std::pair<u64, u64>> sync_ranges_;
    void add_sync_range(u64 lo, u64 hi);
    void draw_ranges(u64& lo, u64& hi, u64& zlo, u64& zhi) const;
    void run_serial(const u32* w, u32 nwords);

    // Accelerated (see set_accel): lane 0 runs every command at once,
    // recording primitives for accel_ - or drawing them itself where the
    // drawing order of pixels across rows matters (as run_serial()).
    std::unique_ptr<ExactAccel> accel_;
    bool accel_on_{false};
    // ORBIT64_GPU_STATS: why the back end had to finish (Sync Full, a
    // texture load, a primitive drawn here, a CPU/DMA read, a CPU/DMA write).
    u64 accel_flushes_[5]{};
    void note_flush(int why);
    void update_accel(); // accel_on_ follows accel_ and the scale
    bool command_accel(const u32* w, u8* rdram, size_t rdram_size);
    // Internal resolution: primitives drawn here (serial ones) draw into the
    // high-resolution copy too - the back end's, which comes here for that
    // (cpu_up_, the RDRAM ranges it came for) and goes back before the back
    // end draws again (release_up()).
    std::vector<std::pair<u64, u64>> cpu_up_;
    bool fetching_up_{false};
    void release_up();
};
