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

#include "common.hpp"
#include <array>
#include <memory>
#include <mutex>
#include <vector>

class ExactRdp {
public:
    ExactRdp();
    void reset();

    // Number of 64-bit words of the command whose first byte is `op` (6 bits).
    static u32 command_words(u32 op);
    // Runs one command; `w` holds its 32-bit words (w[0] = the upper half of
    // the first 64-bit word). Returns true for Sync Full.
    bool command(const u32* w, u8* rdram, size_t rdram_size);

    // The ninth bits (2 per RDRAM halfword; 4 = never written by the RDP).
    const std::vector<u8>& hidden() const { return hs_->bits; }
    // For tools (trace replay): set halfword h's ninth bits as RDP-written.
    void force_hidden(size_t h, u8 bits, u16 word) {
        if (h < hs_->bits.size()) {
            hs_->bits[h] = bits;
            hs_->word[h] = word;
        }
    }

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
        *this = o;
        lane_ = lane;
        lanes_ = lanes;
        hs_ = std::move(hs);
    }
    // The next primitive is drawn whole by lane 0 (and skipped by the others).
    void set_serial(bool on) { serial_ = on; }
    // Whether drawing `cmd` needs the previous row's last pixel (2-cycle,
    // the first blender cycle reading memory) under the current modes.
    bool draw_needs_serial() const;
    // The ninth bits of RDRAM halfword `h`, which holds `word` now.
    u8 hidden_at(size_t h, u16 word) const;
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
    }

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
    // A CPU/DMA write (see ExactRdp::cpu_wrote); what is queued is drawn first.
    void cpu_wrote(u32 paddr, u32 len);

    template <class S> void serialize(S& s) {
        flush();
        lanes_[0]->serialize(s);
        if constexpr (S::loading)
            for (size_t i = 1; i < lanes_.size(); ++i) lanes_[i]->copy_state_from(*lanes_[0]);
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
    u32 scissor_yhi_{0}, scissor_xhi_{0};
    u64 written_lo_{~0ull}, written_hi_{0}; // RDRAM the queued draws can write
    void run_serial(const u32* w, u32 nwords);
};
