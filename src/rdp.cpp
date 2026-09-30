#include "rdp.hpp"
#include "mi.hpp"
#include "raster_pool.hpp"
#include "jit/jit_invalidate.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <unordered_set>

extern int g_current_frame; // emulator.cpp

static u32 stat_tri_called = 0;
static u32 stat_incount[4] = {0, 0, 0, 0};
static u32 stat_rast_called = 0;
static u32 stat_cull_back = 0;
static u32 stat_cull_front = 0;
static u32 stat_scissor_reject = 0;
// Pixel counters (debug log). Native-pass bands count into their own
// RDP::PixelStats and add them here once their flush is done.
static RDP::PixelStats stat_pixels;

// SHIFT_S/T applied to a texture coordinate (used by the debugger capture;
// the rasterizer uses raster::TexUnit).
static inline f32 apply_tile_shift(f32 coord, u8 shift) {
    if (shift == 0) return coord;
    if (shift > 10) {
        return coord * static_cast<f32>(1 << (16 - shift));
    } else {
        return coord / static_cast<f32>(1 << shift);
    }
}

// Hands the pixels a draw produces at native resolution to write_pixel().
struct RDP::NativeSink {
    RDP& rdp;
    const DrawState& st;
    u8* rdram;
    size_t rdram_size;
    u32* shadow;
    size_t shadow_len;
    PixelStats& stats;
    void write(u32 x, u32 y, u32 color, const raster::PixelAux& aux = {}) {
        rdp.write_pixel(st, x, y, color, rdram, rdram_size, shadow, shadow_len, stats, aux);
    }
    bool occluded(u32 x, u32 y, const raster::PixelAux& aux) const {
        if (!st.z_compare || x >= st.fb_w) return false;
        const u32 zidx = st.zb_addr + (y * st.fb_w + x) * 2;
        if (zidx + 1 >= rdram_size) return false;
        const u16 word = static_cast<u16>((rdram[zidx] << 8) | rdram[zidx + 1]);
        return raster::depth_occluded(st, aux, word, rdp.hle_hidden_->hidden_at(zidx >> 1, word));
    }
};

Matrix4x4 Matrix4x4::identity() {
    Matrix4x4 res{};
    for (int i = 0; i < 4; ++i) res.m[i][i] = 1.0f;
    return res;
}

Matrix4x4 Matrix4x4::multiply(const Matrix4x4& a, const Matrix4x4& b) {
    Matrix4x4 res{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            res.m[i][j] = a.m[i][0] * b.m[0][j] +
                          a.m[i][1] * b.m[1][j] +
                          a.m[i][2] * b.m[2][j] +
                          a.m[i][3] * b.m[3][j];
        }
    }
    return res;
}

void Matrix4x4::transform_point(f32 x, f32 y, f32 z, f32& ox, f32& oy, f32& oz, f32& ow) const {
    ox = x * m[0][0] + y * m[1][0] + z * m[2][0] + m[3][0];
    oy = x * m[0][1] + y * m[1][1] + z * m[2][1] + m[3][1];
    oz = x * m[0][2] + y * m[1][2] + z * m[2][2] + m[3][2];
    ow = x * m[0][3] + y * m[1][3] + z * m[2][3] + m[3][3];
    if (ow == 0.0f) ow = 0.0001f;
}

MicrocodeType identify_ucode_banner(const u8* p, size_t len) {
    const std::string_view h(reinterpret_cast<const char*>(p), len);
    const size_t gfx = h.find("RSP Gfx ucode ");
    const size_t sw = h.find("RSP SW Version");
    if (sw != std::string_view::npos && sw < gfx)
        return h.substr(sw, 32).find("2.0G") != std::string_view::npos ? MicrocodeType::F3DGOLDEN
                                                                         : MicrocodeType::Fast3D;
    if (gfx == std::string_view::npos) return MicrocodeType::Auto;
    const std::string_view text = h.substr(gfx + 14, 48);
    // The version is the first "<digit>.<digit>" in it.
    int major = -1;
    for (size_t i = 0; i + 2 < text.size(); ++i)
        if (text[i] >= '0' && text[i] <= '9' && text[i + 1] == '.' && text[i + 2] >= '0' && text[i + 2] <= '9' &&
            (i == 0 || text[i - 1] == ' ')) {
            major = text[i] - '0';
            break;
        }
    const bool gbi2 = major >= 2;
    if (text.substr(0, 5) == "S2DEX") return gbi2 ? MicrocodeType::S2DEX2 : MicrocodeType::S2DEX;
    if (text.substr(0, 3) == "F3D" || text.substr(0, 3) == "L3D")
        return gbi2 ? MicrocodeType::F3DEX2 : MicrocodeType::F3DEX;
    return MicrocodeType::Auto;
}

bool ucode_banner_is_cbfd(const u8* p, size_t len) {
    const std::string_view h(reinterpret_cast<const char*>(p), len);
    const size_t gfx = h.find("RSP Gfx ucode ");
    return gfx != std::string_view::npos && h.substr(gfx + 14, 7) == "F3DEXBG";
}

bool ucode_banner_is_non(const u8* p, size_t len) {
    const std::string_view h(reinterpret_cast<const char*>(p), len);
    const size_t gfx = h.find("RSP Gfx ucode ");
    if (gfx == std::string_view::npos) return false;
    const std::string_view name = h.substr(gfx + 14, 16);
    return name.substr(0, name.find(' ')).find(".NoN") != std::string_view::npos;
}

RDP::RDP(bool geometry_only) : geometry_only_(geometry_only) {
    hle_hidden_ = &exact_.primary();
    reset();
    if (geometry_only_) return;
    // CPU and DMA writes to RDRAM reset the ninth bits the RDP left there.
    jit::set_write_hook(this, [](void* owner, u32 paddr, u32 len) {
        static_cast<RDP*>(owner)->exact_.cpu_wrote(paddr, len);
    });
    if (const char* e = std::getenv("ORBIT64_PICK")) std::sscanf(e, "%d,%d,%d", &pick_x_, &pick_y_, &pick_frame_);
    if (const char* e = std::getenv("ORBIT64_DL_TRACE")) {
        dl_trace_frame_ = std::atoi(e);
        if (const char* comma = std::strchr(e, ',')) dl_trace_count_ = std::max(1, std::atoi(comma + 1));
    }
}

RDP::~RDP() {
    jit::clear_write_hook_if(this);
}

void RDP::set_capture(bool on) {
    capture_enabled = on;
    if (!on) capture_frame.clear();
    if (on && !capture_shadow_ && !geometry_only_) {
        capture_shadow_ = std::make_unique<RDP>(true);
        capture_shadow_mi_ = std::make_unique<MI>();
    }
    if (capture_shadow_) capture_shadow_->set_capture(on);
}

std::vector<CapturedMesh> RDP::take_capture() {
    finish_texture_run();
    std::vector<CapturedMesh> out;
    out.swap(capture_frame);
    capture_tris = 0;
    // A frame drawn on the low-level RSP: what the geometry-only copy read.
    std::vector<CapturedMesh> shadow = capture_shadow_ ? capture_shadow_->take_capture() : std::vector<CapturedMesh>{};
    capture_from_shadow_ = out.empty() && !shadow.empty();
    return capture_from_shadow_ ? std::move(shadow) : std::move(out);
}

std::vector<CapturedTexture> RDP::take_textures() {
    std::vector<CapturedTexture> out, shadow;
    out.swap(capture_textures);
    if (capture_shadow_) shadow = capture_shadow_->take_textures();
    return capture_from_shadow_ ? std::move(shadow) : std::move(out);
}

void RDP::set_texture_capture_target(u32 vtx_phys) {
    capture_texture_vtx = vtx_phys;
    if (capture_shadow_) capture_shadow_->set_texture_capture_target(vtx_phys);
}

const Matrix4x4& RDP::get_projection() const {
    return capture_from_shadow_ && capture_shadow_ ? capture_shadow_->projection_matrix : projection_matrix;
}

RDP* RDP::capture_shadow() {
    return capture_enabled ? capture_shadow_.get() : nullptr;
}

void RDP::capture_display_list(u32 dl_addr, const u8* rdram, size_t rdram_size) {
    if (!capture_enabled || !capture_shadow_) return;
    // Geometry-only: nothing is written to RDRAM (see RDP(bool)).
    capture_shadow_->process_display_list(dl_addr, const_cast<u8*>(rdram), rdram_size, *capture_shadow_mi_);
}

void RDP::reset() {
    flush_native();
    dpc_start = 0;
    dpc_end = 0;
    raw_buf_.clear();
    raw_mode_ = false;
    raw_unbind_ = true;
    exact_.reset();
    if (const char* e = std::getenv("ORBIT64_RDP")) {
        env_exact_ = std::strcmp(e, "exact") == 0 ? 1 : std::strcmp(e, "fast") == 0 ? 0 : -1;
        if (env_exact_ >= 0) exact_mode_ = env_exact_ != 0;
    }
    dpc_current = 0;
    dpc_status = 0;
    dp_pending_ = false;
    dpc_clock = 0;
    dpc_bufbusy = 0;
    dpc_pipebusy = 0;
    dpc_tmem = 0;

    dps_tbist = 0;
    dps_test_mode = 0;
    dps_buftest_addr = 0;
    dps_buftest_data = 0;

    std::fill(segments.begin(), segments.end(), 0);
    modelview_stack.clear();
    modelview_stack.push_back(Matrix4x4::identity());
    projection_matrix = Matrix4x4::identity();
    combined_matrix = Matrix4x4::identity();
    combined_matrix_dirty = false;

    std::fill(tmem.begin(), tmem.end(), 0);
    std::fill(tmem_word_dxt_zero.begin(), tmem_word_dxt_zero.end(), false);
    for (auto& t : tiles) t = {};
    active_tile = 0;

    timg_addr = 0;
    timg_format = 0;
    timg_size = 0;
    timg_width = 0;

    color_image_addr = 0;
    color_image_format = 0;
    color_image_size = 2;
    color_image_width = 320;
    depth_image_addr = 0;

    fill_color = 0;
    prim_color = 0xFFFFFFFF;
    tex_max_level = prim_min_level = prim_lod_frac = 0;
    fog_mul = fog_ofs = 0;
    no_near_clip_ = false;
    env_color = 0xFFFFFFFF;
    blend_color = 0;
    fog_color = 0;
    prim_depth = 0;
    prim_dz = 0;
    geometry_mode = 0;
    texture_enabled = true;
    texture_scale_s = 1.0f;
    texture_scale_t = 1.0f;

    combine_mode_w0 = 0;
    combine_mode_w1 = 0;
    combine_mode_set = false;

    other_mode_l = 0x00000030; // Z_CMP | Z_UPD enabled by default
    other_mode_h = 0;

    vp_scale_x = 160.0f; vp_scale_y = 120.0f; vp_scale_z = 511.5f;
    vp_trans_x = 160.0f; vp_trans_y = 120.0f; vp_trans_z = 511.5f;

    ambient_light = {128, 128, 128, 0, 0, 0};
    lookat_x = {0, 0, 0, 1.0f, 0.0f, 0.0f};
    lookat_y = {0, 0, 0, 0.0f, 1.0f, 0.0f};
    lookat_set = false;
    dir_lights.clear();
    num_lights = 0;

    scissor_ulx = 0;
    scissor_uly = 0;
    scissor_lrx = 320;
    scissor_lry = 240;
    rdp_half1 = 0;
    rdp_half2 = 0;
    vtx_color_base = 0;
    dkr_mtx_offset = dkr_vtx_offset = dkr_vtx_index = dkr_mv_index = 0;
    dkr_billboard = false;
    dkr_mv.fill(Matrix4x4::identity());
    ucode_type = MicrocodeType::Auto;

    std::fill(std::begin(s2d_genstat), std::end(s2d_genstat), 0u);
    cbfd_ = cbfd_advanced_ = false;
    cbfd_normal_base_ = 0;
    std::fill(std::begin(cbfd_coord_mod_), std::end(cbfd_coord_mod_), 0.0f);
    cbfd_lights_ = {};
    cbfd_num_lights_ = 0;
    obj2d_matrix = Obj2DMatrix{};
    obj_render_mode = 0;
    s2d_pending_flag = 0;
    s2d_pending_sid = 0;
    s2d_pending_addr_lo = 0;
    s2d_pending_valid = false;

    internal_zbuffer.assign(640 * 480, 1e30f);
    if (capture_shadow_) capture_shadow_->reset();

    draw_state_dirty_ = true;
    ++tmem_gen_;
    // Start over from an empty set of high-resolution buffers.
    recreate_hires();
}

void RDP::state_loaded() {
    // Everything cached from the old TMEM, tiles and modes is stale, and the
    // high-resolution buffers show the old frames: start them over, as a
    // reset does. They fill in again from what the game draws next.
    flush_native();
    draw_state_dirty_ = true;
    ++tmem_gen_;
    tmem_dirty = true;
    tex_run.active = false;
    tex_last_tlut = ~0u;
    recreate_hires();
}

std::unique_ptr<HiResRenderer> RDP::make_hires(u32 scale) const {
    if (hires_factory_)
        if (auto r = hires_factory_(scale)) return r;
    return std::make_unique<CpuHiResRenderer>(scale);
}

void RDP::recreate_hires() {
    flush_native();
    hires_shadow_ = nullptr;
    if (!hires_) return;
    const u32 scale = hires_->scale();
    hires_.reset();
    hires_ = make_hires(scale);
}

// The depth buffer is in RDRAM, as the RDP has it; this clears the
// high-resolution pass's copies.
void RDP::clear_zbuffer() {
    flush_native();
    if (hires_) hires_->clear_depth();
}

void RDP::set_hires_scale(u32 scale) {
    flush_native();
    scale = std::clamp<u32>(scale, 1, HiResRenderer::kMaxScale);
    if (scale != exact_.scale()) exact_.set_scale(scale);
    if (scale == hires_scale()) return;
    hires_shadow_ = nullptr;
    hires_.reset();
    if (scale > 1) hires_ = make_hires(scale);
}

HiResTarget* RDP::hires_target(u8* rdram, size_t rdram_size) {
    hires_shadow_ = nullptr;
    hires_shadow_len_ = 0;
    if (!hires_ || color_image_addr >= rdram_size) return nullptr;
    // Filling the depth image clears the depth buffer; there is no colour to show.
    if (color_image_addr != 0 && color_image_addr == depth_image_addr) return nullptr;
    const u32 fb_w = color_image_width ? color_image_width : 320;
    HiResTarget* t = hires_->bind(color_image_addr, fb_w, color_image_size, rdram, rdram_size);
    if (t) {
        hires_shadow_ = t->shadow.data();
        hires_shadow_len_ = t->shadow.size();
    }
    return t;
}

const DrawState& RDP::draw_state() {
    if (draw_state_dirty_) {
        DrawState& s = draw_state_;
        s.other_mode_h = other_mode_h;
        s.other_mode_l = other_mode_l;
        s.combine_w0 = combine_mode_w0;
        s.combine_w1 = combine_mode_w1;
        s.prim_color = prim_color;
        s.env_color = env_color;
        s.blend_color = blend_color;
        s.fog_color = fog_color;
        for (int i = 0; i < 3; ++i) {
            s.key_center[i] = key_center_[i];
            s.key_scale[i] = key_scale_[i];
        }
        s.k4 = k4_;
        s.k5 = k5_;
        s.noise_seed = static_cast<u32>(draw_state_serial_);
        s.scissor_ulx = scissor_ulx;
        s.scissor_uly = scissor_uly;
        s.scissor_lrx = scissor_lrx;
        s.scissor_lry = scissor_lry;
        s.fb_addr = color_image_addr;
        s.zb_addr = depth_image_addr;
        s.prim_depth = static_cast<u16>(prim_depth);
        s.prim_dz = static_cast<u16>(prim_dz);
        s.fb_w = color_image_width ? color_image_width : 320;
        s.fb_size = color_image_size;
        s.combine_set = combine_mode_set;
        s.texture_enabled = texture_enabled;
        // The RDP interpolates whatever its triangle commands say; flat
        // shading is the microcode's job (zero gradients).
        s.smooth_shading = raw_mode_ || (current_ucode_active == MicrocodeType::F3DEX2 ? (geometry_mode & 0x00200000) != 0
                                                                                       : (geometry_mode & 0x00000200) != 0);
        s.active_tile = active_tile;
        s.max_level = tex_max_level;
        s.min_level = prim_min_level;
        s.prim_lod_frac = prim_lod_frac;
        s.tmem = tmem.data();
        s.tmem_dxt = tmem_word_dxt_zero.data();
        for (int i = 0; i < 8; ++i) s.tex[i].prepare(tiles[i]);
        s.finalize();
        draw_state_dirty_ = false;
        ++draw_state_serial_;
    }
    return draw_state_;
}

// Display list commands that can't change anything draw_state() captures
// (geometry, flow control, syncs, TMEM loads, draws). Everything else marks
// the draw state dirty.
static bool keeps_draw_state(u8 opcode) {
    switch (opcode) {
        case 0x00: case 0x01: case 0x02: case 0x03: case 0x04: case 0x05: case 0x06: case 0x07: case 0x08:
        case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16: case 0x17:
        case 0x18: case 0x19: case 0x1A: case 0x1B: case 0x1C: case 0x1D: case 0x1E: case 0x1F:
        case 0xB0: case 0xB1: case 0xB2: case 0xB3: case 0xB4: case 0xB5: case 0xB8: case 0xBC: case 0xBD:
        case 0xBE: case 0xBF: case 0xC0: case 0xD8: case 0xDA: case 0xDB: case 0xDC: case 0xDE: case 0xDF:
        case 0xE1: case 0xE4: case 0xE5: case 0xE6: case 0xE7: case 0xE8: case 0xE9:
        case 0xF0: case 0xF1: case 0xF3: case 0xF4: case 0xF6: case 0xF7: case 0xFD:
            return true;
        default:
            return false;
    }
}

u32 RDP::read_dpc_reg(u32 addr) const {
    u32 reg = (addr & 0x1F) >> 2;
    switch (reg) {
        case 0: return dpc_start;
        case 1: return dpc_end;
        case 2: return dpc_current;
        case 3: return dpc_status | (1 << 7); // CBUF_READY: commands are taken right away
        case 4: return dpc_clock;
        case 5: return dpc_bufbusy;
        case 6: return dpc_pipebusy;
        case 7: return dpc_tmem;
        default: return 0;
    }
}

void RDP::write_dpc_reg(u32 addr, u32 val, MI& mi, u8* rdram, size_t rdram_size) {
    u32 reg = (addr & 0x1F) >> 2;
    switch (reg) {
        case 0: // DPC_START_REG
            dpc_start = val & 0x00FFFFF8;
            dpc_current = dpc_start;
            break;
        case 1: // DPC_END_REG: the RDP runs [DPC_CURRENT, DPC_END)
            dpc_end = val & 0x00FFFFF8;
            if (!(dpc_status & (1 << 1))) process_rdp_commands(mi, rdram, rdram_size);
            break;
        case 2: // DPC_CURRENT_REG (read only)
            break;
        case 3: { // DPC_STATUS_REG
            if (val & (1 << 0)) dpc_status &= ~(1 << 0); // Clr xbus
            if (val & (1 << 1)) dpc_status |= (1 << 0);  // Set xbus
            if ((val & (1 << 2)) && (dpc_status & (1 << 1))) {  // Clr freeze
                dpc_status &= ~(1 << 1);
                // What the RDP held back while frozen is done now.
                if (dp_pending_) {
                    dp_pending_ = false;
                    mi.raise_interrupt(MIInterrupt::DP);
                }
                process_rdp_commands(mi, rdram, rdram_size);
            }
            if (val & (1 << 3)) dpc_status |= (1 << 1);  // Set freeze
            if (val & (1 << 4)) dpc_status &= ~(1 << 2); // Clr flush
            if (val & (1 << 5)) dpc_status |= (1 << 2);  // Set flush
            break;
        }
    }
}

namespace {
// Length in 64-bit words of the RDP command `op` (6 bits): triangles carry
// 4 edge words, then 8 shade, 8 texture and 2 depth words if they have them.
u32 rdp_command_words(u32 op) {
    if (op >= 0x08 && op <= 0x0F)
        return 4 + ((op & 4) ? 8 : 0) + ((op & 2) ? 8 : 0) + ((op & 1) ? 2 : 0);
    if (op == 0x24 || op == 0x25) return 2; // texture rectangles
    return 1;
}
} // namespace

// Runs the commands between DPC_CURRENT and DPC_END, read from RDRAM (or
// from DMEM with XBUS). They are run as soon as DPC_END says they are
// there; a command cut off at the end of the buffer waits for the rest.
void RDP::process_rdp_commands(MI& mi, u8* rdram, size_t rdram_size) {
    if (dpc_current >= dpc_end) return;
    const bool xbus = dpc_status & 1;
    if (xbus && !rsp_dmem_) return;
    for (u32 a = dpc_current; a + 8 <= dpc_end; a += 8) {
        u64 w = 0;
        for (u32 i = 0; i < 8; ++i) {
            const u8 b = xbus ? rsp_dmem_[(a + i) & 0xFFF] : (a + i < rdram_size ? rdram[a + i] : 0);
            w = (w << 8) | b;
        }
        raw_buf_.push_back(w);
    }
    dpc_current = dpc_end;

    if (!trace_checked_) {
        trace_checked_ = true;
        if (const char* path = std::getenv("ORBIT64_RDP_TRACE")) trace_ = std::make_unique<Trace>(path);
    }
    // (Only at the start of a frame: what the CPU changes while the RDP is
    // drawing one is left out, which keeps tracing fast.)
    // ORBIT64_RDP_TRACE_FINE=<frame>: from that frame on, before every batch.
    if (trace_ && (!raw_mode_ || raw_unbind_ || (trace_->fine_from >= 0 && g_current_frame >= trace_->fine_from)))
        trace_->before(rdram, rdram_size);

    if (!raw_mode_ || raw_unbind_) {
        // The CPU may have changed RDRAM since the RDP last drew.
        raw_mode_ = true;
        raw_unbind_ = false;
        draw_state_dirty_ = true;
        if (hires_) hires_->unbind();
    }
    size_t pos = 0;
    while (pos < raw_buf_.size()) {
        const u64 c0 = raw_buf_[pos];
        const u32 op = static_cast<u32>(c0 >> 56) & 0x3F;
        const u32 len = rdp_command_words(op);
        if (pos + len > raw_buf_.size()) break; // the rest comes with the next DPC_END
        const u64* cmd = &raw_buf_[pos];
        pos += len;
        ++raw_cmd_count_;
        const u32 w0 = static_cast<u32>(c0 >> 32), w1 = static_cast<u32>(c0);
        if (trace_) trace_->command(cmd, len);
        if (exact_mode_) {
            u32 words[44];
            for (u32 i = 0; i < len; ++i) {
                words[i * 2] = static_cast<u32>(cmd[i] >> 32);
                words[i * 2 + 1] = static_cast<u32>(cmd[i]);
            }
            if (exact_.command(words, rdram, rdram_size)) { // Sync Full
                if (trace_) trace_->after(rdram, rdram_size);
                raw_unbind_ = true;
                ++display_list_count;
                mi.raise_interrupt(MIInterrupt::DP);
            }
            continue;
        }
        const u8 opcode = static_cast<u8>(op | 0xC0); // the display lists' numbering
        if (!keeps_draw_state(opcode)) draw_state_dirty_ = true;

        if (op >= 0x08 && op <= 0x0F) {
            rdp_triangle(cmd, op, rdram, rdram_size);
            continue;
        }
        switch (op) {
            case 0x24: // Texture Rectangle
            case 0x25: { // Texture Rectangle Flip (edges in quarter pixels)
                u32 lrx = (w0 >> 12) & 0xFFF;
                u32 lry = w0 & 0xFFF;
                const u32 tile_idx = (w1 >> 24) & 0x7;
                u32 ulx = (w1 >> 12) & 0xFFF;
                u32 uly = w1 & 0xFFF;
                if (ulx > lrx) std::swap(ulx, lrx);
                if (uly > lry) std::swap(uly, lry);
                const u64 c1 = cmd[1];
                const f32 s = static_cast<s16>(c1 >> 48) / 32.0f;
                const f32 t = static_cast<s16>(c1 >> 32) / 32.0f;
                const f32 dsdx = static_cast<s16>(c1 >> 16) / 1024.0f;
                const f32 dtdy = static_cast<s16>(c1) / 1024.0f;
                rasterize_tex_rect(ulx, uly, lrx, lry, tile_idx, s, t, dsdx, dtdy, op == 0x25, rdram, rdram_size);
                break;
            }
            case 0x29: // Sync Full: everything drawn is in RDRAM, tell the CPU
                flush_native();
                if (hires_) hires_->flush();
                raw_unbind_ = true;
                ++display_list_count;
                mi.raise_interrupt(MIInterrupt::DP);
                break;
            default:
                // Syncs and no-ops do nothing here; everything else is the
                // same as in a display list.
                if (op >= 0x26) execute_rdp_op(opcode, w0, w1, rdram, rdram_size, true);
                break;
        }
    }
    raw_buf_.erase(raw_buf_.begin(), raw_buf_.begin() + static_cast<std::ptrdiff_t>(pos));
    if (trace_ && trace_->fine_from >= 0 && g_current_frame >= trace_->fine_from) trace_->after(rdram, rdram_size);
}

// ORBIT64_RDP_TRACE=<file>: records what the RDP's command buffer runs - its
// commands, and every change to RDRAM made by anything but the RDP in
// between (4 KB pages; the first record is all of RDRAM) - so that it can be
// replayed on another RDP implementation and compared with this one.
//   'M' u32 address, u32 length, bytes      RDRAM contents
//   'C' u32 words, the command's 32-bit words
// (u32s little-endian, RDRAM bytes in the console's order).
RDP::Trace::Trace(const char* path) : f(std::fopen(path, "wb")) {
    if (const char* e = std::getenv("ORBIT64_RDP_TRACE_FINE")) fine_from = std::atoi(e);
}
RDP::Trace::~Trace() {
    if (f) std::fclose(f);
}
void RDP::Trace::put32(u32 v) {
    const u8 b[4] = {static_cast<u8>(v), static_cast<u8>(v >> 8), static_cast<u8>(v >> 16), static_cast<u8>(v >> 24)};
    std::fwrite(b, 1, 4, f);
}
void RDP::Trace::before(const u8* rdram, size_t size) {
    if (!f) return;
    constexpr size_t kPage = 4096;
    if (shadow.size() != size) {
        shadow.assign(rdram, rdram + size);
        std::fputc('M', f);
        put32(0);
        put32(static_cast<u32>(size));
        std::fwrite(rdram, 1, size, f);
        return;
    }
    for (size_t p = 0; p < size; p += kPage) {
        if (std::memcmp(rdram + p, shadow.data() + p, kPage) == 0) continue;
        std::memcpy(shadow.data() + p, rdram + p, kPage);
        std::fputc('M', f);
        put32(static_cast<u32>(p));
        put32(static_cast<u32>(kPage));
        std::fwrite(rdram + p, 1, kPage, f);
    }
}
void RDP::Trace::command(const u64* cmd, u32 len) {
    if (!f) return;
    std::fputc('C', f);
    put32(len * 2);
    for (u32 i = 0; i < len; ++i) {
        put32(static_cast<u32>(cmd[i] >> 32));
        put32(static_cast<u32>(cmd[i]));
    }
}
void RDP::Trace::after(const u8* rdram, size_t size) {
    if (!f || shadow.size() != size) return;
    // What the RDP wrote is not recorded: the replay draws it.
    std::memcpy(shadow.data(), rdram, size);
    std::fflush(f);
}

// A triangle command: three edges (the major edge H from YH to YL, M above
// YM and L below it) and the shade, texture and depth planes, each an
// attribute's value where H crosses the top scanline plus its change per
// pixel to the right (DxDx) and per scanline along H (DxDe). It is drawn as
// the (up to six-cornered) polygon the edges enclose, with every attribute
// worked out at the corners from its plane, through the same pixel pipeline
// as everything else.
void RDP::rdp_triangle(const u64* cmd, u32 op, u8* rdram, size_t rdram_size) {
    stat_tri_called++;
    const u64 c0 = cmd[0];
    auto s14 = [](u64 v) { return static_cast<s32>(static_cast<u32>(v & 0x3FFF) << 18) >> 18; };
    const u32 level = static_cast<u32>(c0 >> 51) & 7;
    const u32 tile = static_cast<u32>(c0 >> 48) & 7;
    const f32 yl = s14(c0 >> 32) / 4.0f, ym = s14(c0 >> 16) / 4.0f, yh = s14(c0) / 4.0f;
    auto fx = [](u32 v) { return static_cast<s32>(v) / 65536.0f; };
    const f32 xl = fx(static_cast<u32>(cmd[1] >> 32)), dxldy = fx(static_cast<u32>(cmd[1]));
    const f32 xh = fx(static_cast<u32>(cmd[2] >> 32)), dxhdy = fx(static_cast<u32>(cmd[2]));
    const f32 xm = fx(static_cast<u32>(cmd[3] >> 32)), dxmdy = fx(static_cast<u32>(cmd[3]));
    if (!(yl > yh)) return;
    // H and M start at the top scanline YH is on; L starts at YM.
    const f32 y0 = std::floor(yh);
    const f32 ymc = std::clamp(ym, yh, yl);

    const bool shade = op & 4, tex = op & 2, zbuf = op & 1;
    if (texture_enabled != tex || active_tile != tile || tex_max_level != level) {
        texture_enabled = tex;
        active_tile = tile;
        tex_max_level = static_cast<u8>(level);
        draw_state_dirty_ = true;
    }

    // An attribute plane: value, DxDx, DxDe (s15.16 each).
    struct Plane {
        f32 v{0}, dx{0}, de{0};
        // At (x, y) (the rasterizer samples pixels at their upper-left corner, as the RDP does).
        f32 at(f32 x, f32 y, f32 xh, f32 dxhdy, f32 y0) const {
            return v + dx * (x - xh) + (de - dx * dxhdy) * (y - y0);
        }
    };
    // Four attributes of 8 words: integer parts of the values, DxDx; their
    // fractions; then DxDe, DxDy, their fractions.
    auto planes4 = [](const u64* w, Plane out[4]) {
        for (int i = 0; i < 4; ++i) {
            const int sh = 48 - 16 * i;
            auto val = [&](u64 hi, u64 lo) {
                const u32 v = (static_cast<u32>((hi >> sh) & 0xFFFF) << 16) | static_cast<u32>((lo >> sh) & 0xFFFF);
                return static_cast<s32>(v) / 65536.0f;
            };
            out[i].v = val(w[0], w[2]);
            out[i].dx = val(w[1], w[3]);
            out[i].de = val(w[4], w[6]);
        }
    };
    Plane rgba[4], stw[4], z;
    u32 next = 4;
    if (shade) { planes4(cmd + next, rgba); next += 8; }
    if (tex) { planes4(cmd + next, stw); next += 8; }
    if (zbuf) {
        z.v = fx(static_cast<u32>(cmd[next] >> 32));
        z.dx = fx(static_cast<u32>(cmd[next]));
        z.de = fx(static_cast<u32>(cmd[next + 1] >> 32));
    }

    // The corners, going round: H and M at the top, M and L at YM, L and H
    // at the bottom.
    struct P { f32 x, y; };
    P poly[6] = {
        {xh + dxhdy * (yh - y0), yh},  {xm + dxmdy * (yh - y0), yh},
        {xm + dxmdy * (ymc - y0), ymc}, {xl, ymc},
        {xl + dxldy * (yl - ymc), yl}, {xh + dxhdy * (yl - y0), yl},
    };
    if (ym > yl) { // no L part: M runs to the bottom
        poly[3] = poly[4] = {xm + dxmdy * (yl - y0), yl};
    }
    P pts[6];
    int n = 0;
    for (const P& p : poly) {
        if (n > 0 && std::fabs(p.x - pts[n - 1].x) < 0.125f && std::fabs(p.y - pts[n - 1].y) < 0.125f) continue;
        pts[n++] = p;
    }
    while (n > 1 && std::fabs(pts[n - 1].x - pts[0].x) < 0.125f && std::fabs(pts[n - 1].y - pts[0].y) < 0.125f) --n;
    if (n < 3) return;

    const bool persp = (other_mode_h >> 19) & 1;
    const bool zprim = (other_mode_l >> 2) & 1; // Z_SOURCE_PRIM
    Vertex v[6];
    for (int i = 0; i < n; ++i) {
        Vertex& o = v[i];
        const f32 x = pts[i].x, y = pts[i].y;
        o.sx = x;
        o.sy = y;
        if (shade) {
            auto c = [&](const Plane& p) { return static_cast<u8>(std::clamp(p.at(x, y, xh, dxhdy, y0), 0.0f, 255.0f)); };
            o.r = c(rgba[0]); o.g = c(rgba[1]); o.b = c(rgba[2]); o.a = c(rgba[3]);
        } else {
            o.r = o.g = o.b = o.a = 0;
        }
        o.w = 1.0f;
        if (tex) {
            const f32 s = stw[0].at(x, y, xh, dxhdy, y0) / 32.0f; // S, T are s10.5 texels
            const f32 t = stw[1].at(x, y, xh, dxhdy, y0) / 32.0f;
            if (persp) {
                // S/W, T/W, with W 1.0 at 0x8000: the rasterizer interpolates
                // u/w, v/w and 1/w, so w = 1 / W.
                const f32 wn = std::max(stw[2].at(x, y, xh, dxhdy, y0) / 32768.0f, 1.0f / 32768.0f);
                o.w = 1.0f / wn;
                o.u = s / wn;
                o.v = t / wn;
            } else {
                o.u = s;
                o.v = t;
            }
        }
        // Depth 0..0x7FFF; display lists use z / 1023 of the viewport's
        // 0..1023, which is the RDP's z / 32.
        const f32 zi = zprim ? static_cast<f32>(prim_depth & 0x7FFF) : (zbuf ? z.at(x, y, xh, dxhdy, y0) : 0.0f);
        o.sz = std::clamp(zi / (1023.0f * 32.0f), 0.0f, 1.0f);
    }
    for (int i = 1; i + 1 < n; ++i) {
        const f32 area = raster::triangle_area(v[0], v[i], v[i + 1]);
        if (std::fabs(area) < 1e-6f) continue;
        draw_triangle(v[0], v[i], v[i + 1], area, rdram, rdram_size);
    }
}

void RDP::finish_task(MI& mi) {
    mi.raise_interrupt(MIInterrupt::SP);
    // A frozen RDP (DPC_STATUS freeze, set by e.g. Rare's scheduler while it
    // queues the next frame) doesn't run the commands yet, so "RDP done"
    // only comes once it is unfrozen; Banjo-Kazooie waits for it there.
    // On the hardware the RDP raises it only when it runs a G_RDPFULLSYNC: a list
    // without one (Blast Corps' render-to-texture pass) must not produce it, or the
    // game's scheduler pops a task that is not there and its thread faults.
    const bool full_sync = full_sync_seen_;
    full_sync_seen_ = false;
    if (!full_sync) return;
    if (dpc_status & (1 << 1)) dp_pending_ = true;
    else mi.raise_interrupt(MIInterrupt::DP);
}

u32 RDP::read_dps_reg(u32 addr) const {
    u32 reg = (addr & 0xF) >> 2;
    switch (reg) {
        case 0: return dps_tbist;
        case 1: return dps_test_mode;
        case 2: return dps_buftest_addr;
        case 3: return dps_buftest_data;
        default: return 0;
    }
}

void RDP::write_dps_reg(u32 addr, u32 val) {
    u32 reg = (addr & 0xF) >> 2;
    switch (reg) {
        case 0: dps_tbist = val; break;
        case 1: dps_test_mode = val; break;
        case 2: dps_buftest_addr = val; break;
        case 3: dps_buftest_data = val; break;
    }
}

// The microcodes built on Fast3D's command set (vertex indices x10, F3D
// geometry mode bits, G_RDPHALF_CONT, ...).
static bool is_f3d_family(MicrocodeType u) {
    return u == MicrocodeType::Fast3D || u == MicrocodeType::F3DGOLDEN || u == MicrocodeType::F3DPD ||
           u == MicrocodeType::F3DDKR || u == MicrocodeType::F3DJFG || u == MicrocodeType::F3DWRUS;
}

u32 RDP::segment_to_physical(u32 seg_addr) const {
    u32 seg = (seg_addr >> 24) & 0x0F;
    return (segments[seg] + (seg_addr & 0x00FFFFFF)) & 0x00FFFFFF;
}

void RDP::update_combined_matrix() {
    if (combined_matrix_dirty) {
        const auto& mv = modelview_stack.empty() ? Matrix4x4::identity() : modelview_stack.back();
        // N64 uses row vectors: v_clip = v × MV × Proj, so combined = MV × Proj
        combined_matrix = Matrix4x4::multiply(mv, projection_matrix);
        combined_matrix_dirty = false;
    }
}

void RDP::execute_mtx(u32 w0, u32 w1, MicrocodeType ucode, const u8* rdram, size_t rdram_size) {
    u32 mtx_addr = segment_to_physical(w1);
    capture_mtx_addr = mtx_addr;
    if (mtx_addr + 64 <= rdram_size) {
        Matrix4x4 mat{};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int idx = (i * 4 + j) * 2;
                s16 int_part = static_cast<s16>((rdram[mtx_addr + idx] << 8) | rdram[mtx_addr + idx + 1]);
                u16 frac_part = static_cast<u16>((rdram[mtx_addr + 32 + idx] << 8) | rdram[mtx_addr + 32 + idx + 1]);
                mat.m[i][j] = int_part + (frac_part / 65536.0f);
            }
        }

        // The matrix format in RDRAM directly matches the row-vector convention of transform_point.
        // DO NOT transpose here!

        bool is_proj = false;
        bool is_load = false;
        bool is_push = false;

        if (ucode == MicrocodeType::F3DEX2) {
            // In F3DEX2: gsSPMatrix(m, p) -> gsDma2p(G_MTX, m, sizeof(Mtx), (p) ^ G_MTX_PUSH, 0)
            // Parameters are in w0 & 0xFF.
            // G_MTX_PUSH (0x01) was inverted with XOR in SDK macro, so bit 0 == 0 means PUSH!
            u8 param_byte = w0 & 0xFF;
            is_push = (param_byte & 0x01) == 0;
            is_load = (param_byte & 0x02) != 0;
            is_proj = (param_byte & 0x04) != 0;
        } else {
            // Fast3D / F3DEX: gsSPMatrix(m, p) -> gsDma1p(G_MTX, m, sizeof(Mtx), p)
            // Parameters are in (w0 >> 16) & 0xFF.
            // G_MTX_PROJECTION = 0x01, G_MTX_LOAD = 0x02, G_MTX_PUSH = 0x04
            u8 param_byte = (w0 >> 16) & 0xFF;
            is_proj = (param_byte & 0x01) != 0;
            is_load = (param_byte & 0x02) != 0;
            is_push = (param_byte & 0x04) != 0;
            if (stat_tri_called > 395000 && stat_tri_called < 420000) {
                std::cout << "[MTXFLAGDBG] tri=" << stat_tri_called << " w0=0x" << std::hex << w0
                          << " param=0x" << (int)param_byte << std::dec
                          << " proj=" << is_proj << " load=" << is_load << " push=" << is_push << "\n";
            }
        }

        if (is_proj) {
            if (stat_tri_called > 395000 && stat_tri_called < 420000) {
                std::cout << "[PROJRAW] tri=" << stat_tri_called << " load=" << is_load << ":\n";
                for (int r = 0; r < 4; ++r) {
                    std::cout << "  [" << mat.m[r][0] << ", " << mat.m[r][1] << ", " << mat.m[r][2] << ", " << mat.m[r][3] << "]\n";
                }
            }
            if (is_load) projection_matrix = mat;
            else projection_matrix = Matrix4x4::multiply(mat, projection_matrix);
        } else {
            if (is_push && modelview_stack.size() < 32) {
                modelview_stack.push_back(modelview_stack.back());
            }
            if (is_load) {
                if (!modelview_stack.empty()) modelview_stack.back() = mat;
                else modelview_stack.push_back(mat);
            } else {
                // Row-vector convention (v' = v*M): the incoming matrix transforms the
                // object-space point FIRST, so it must be the left operand, matching
                // the projection-matrix MUL branch above (mat * existing, not existing * mat).
                if (!modelview_stack.empty()) modelview_stack.back() = Matrix4x4::multiply(mat, modelview_stack.back());
                else modelview_stack.push_back(mat);
            }
        }
        combined_matrix_dirty = true;
    }
}

// F3DDKR/F3DJFG G_DMA_MTX (0x01): loads a matrix (at the matrix offset) into
// one of the model-view slots and makes it current. DKR picks the slot with
// bits 23..22; Jet Force Gemini with bits 19..16, and bit 23 multiplies the
// new matrix onto slot 0. The matrices already include the projection.
void RDP::dkr_dma_matrix(u32 w0, u32 w1, MicrocodeType ucode, const u8* rdram, size_t rdram_size) {
    if ((w0 & 0xFFFF) != 64) return; // not a whole matrix
    u32 index = (w0 >> 16) & 0xF;
    bool multiply = false;
    if (ucode == MicrocodeType::F3DJFG && index != 0) {
        multiply = ((w0 >> 23) & 1) != 0;
    } else {
        index = (w0 >> 22) & 0x3;
    }
    index &= 3;
    const u32 addr = (segment_to_physical(w1) + dkr_mtx_offset) & 0x00FFFFFF;
    capture_mtx_addr = addr;
    if (addr + 64 > rdram_size) return;
    Matrix4x4 mat{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const u32 idx = (i * 4 + j) * 2;
            const s16 int_part = static_cast<s16>((rdram[addr + idx] << 8) | rdram[addr + idx + 1]);
            const u16 frac_part = static_cast<u16>((rdram[addr + 32 + idx] << 8) | rdram[addr + 32 + idx + 1]);
            mat.m[i][j] = int_part + (frac_part / 65536.0f);
        }
    }
    dkr_mv[index] = multiply ? Matrix4x4::multiply(mat, dkr_mv[0]) : mat;
    dkr_select_matrix(index);
}

void RDP::dkr_select_matrix(u32 index) {
    dkr_mv_index = index & 3;
    projection_matrix = Matrix4x4::identity();
    modelview_stack.resize(1);
    modelview_stack.back() = dkr_mv[dkr_mv_index];
    combined_matrix_dirty = true;
}

// F3DDKR/F3DJFG G_DMA_TRI (0x05): (w0 >> 4) & 0xFFF triangles of 16 bytes -
// a flag byte (0x40 = drawn from both sides), the three vertex slots, then
// each corner's S and T (10.5) - after which vertices start over at slot 0.
void RDP::dkr_dma_triangles(u32 w0, u32 w1, u8* rdram, size_t rdram_size) {
    const u32 n = (w0 >> 4) & 0xFFF;
    u32 addr = segment_to_physical(w1);
    const u32 saved_mode = geometry_mode;
    // These microcodes have no G_TEXTURE: the triangles are textured
    // whenever the combiner uses a texel, at the coordinates they carry.
    if (!texture_enabled) {
        texture_enabled = true;
        draw_state_dirty_ = true;
    }
    for (u32 i = 0; i < n && addr + 16 <= rdram_size; ++i, addr += 16) {
        const u8* t = rdram + addr;
        const u32 vi[3] = {t[1], t[2], t[3]};
        if (vi[0] >= vertex_cache.size() || vi[1] >= vertex_cache.size() || vi[2] >= vertex_cache.size()) continue;
        for (int k = 0; k < 3; ++k) {
            const s16 st_s = static_cast<s16>((t[4 + k * 4] << 8) | t[5 + k * 4]);
            const s16 st_t = static_cast<s16>((t[6 + k * 4] << 8) | t[7 + k * 4]);
            vertex_cache[vi[k]].u = st_s / 32.0f;
            vertex_cache[vi[k]].v = st_t / 32.0f;
        }
        geometry_mode = (saved_mode & ~0x3000u) | ((t[0] & 0x40) ? 0u : 0x2000u); // G_CULL_BACK unless two-sided
        emit_triangle(vi[0], vi[1], vi[2], rdram, rdram_size);
    }
    geometry_mode = saved_mode;
    dkr_vtx_index = 0;
}

void RDP::execute_vtx(u32 w0, u32 w1, MicrocodeType ucode, const u8* rdram, size_t rdram_size) {
    update_combined_matrix();

    u32 count = 0;
    u32 dest = 0;

    if (ucode == MicrocodeType::F3DEX2) {
        // F3DEX2: bits 19..12 = count, bits 7..1 = end_idx * 2
        count = (w0 >> 12) & 0xFF;
        u32 end_idx = (w0 >> 1) & 0x7F;
        dest = (end_idx >= count) ? (end_idx - count) : 0;
    } else if (ucode == MicrocodeType::F3DEX) {
        // F3DEX: bits 15..10 = count, bits 23..16 = dest * 2
        count = (w0 >> 10) & 0x3F;
        if (count == 0) count = ((w0 >> 20) & 0x0F) + 1;
        dest = ((w0 >> 16) & 0xFF) / 2;
    } else if (ucode == MicrocodeType::F3DWRUS) {
        // Bits 15..9 = count, bits 23..16 = first slot * 5.
        count = (w0 >> 9) & 0x7F;
        dest = ((w0 >> 16) & 0xFF) / 5;
    } else if (ucode == MicrocodeType::F3DDKR || ucode == MicrocodeType::F3DJFG) {
        // Bits 23..19 = count (- 1 on DKR), 13..9 = first slot; bit 16 appends
        // to the vertices loaded since the last triangle list instead (after
        // vertex 0, the billboard origin, when billboarding).
        count = ((w0 >> 19) & 0x1F) + (ucode == MicrocodeType::F3DDKR ? 1 : 0);
        if (!(w0 & 0x10000)) dkr_vtx_index = 0;
        else if (dkr_billboard) dkr_vtx_index = 1;
        dest = dkr_vtx_index + ((w0 >> 9) & 0x1F);
        dkr_vtx_index += count;
    } else {
        // Fast3D: bits 23..20 = count - 1, bits 19..16 = dest
        count = ((w0 >> 20) & 0x0F) + 1;
        dest = (w0 >> 16) & 0x0F;
    }

    u32 vtx_addr = segment_to_physical(w1);
    // Perfect Dark's vertices are 12 bytes: x, y, z, a color index, s, t. The
    // index is a byte offset into the table set by opcode 0x07, whose 4-byte
    // entries hold the RGBA color (or the normal, when lit) that the other
    // microcodes keep in bytes 12-15 of the vertex itself.
    const bool pd = ucode == MicrocodeType::F3DPD;
    // Diddy Kong Racing's are 10 bytes: x, y, z, then the color (or normal and
    // alpha); texture coordinates come with each triangle (dkr_dma_triangles()).
    const bool dkr = ucode == MicrocodeType::F3DDKR || ucode == MicrocodeType::F3DJFG;
    if (dkr) vtx_addr = (vtx_addr + dkr_vtx_offset) & 0x00FFFFFF;
    const u32 stride = dkr ? 10 : pd ? 12 : 16;
    if (cbfd_ && ucode == MicrocodeType::F3DEX2) {
        // The light directions in model space (the inverse of the
        // modelview's rotation applied to them), to meet the model's normals.
        const auto& mv = modelview_stack.empty() ? Matrix4x4::identity() : modelview_stack.back();
        for (u32 l = 0; l < cbfd_lights_.size(); ++l) {
            const CbfdLight& c = cbfd_lights_[l];
            f32 x = mv.m[0][0] * c.x + mv.m[0][1] * c.y + mv.m[0][2] * c.z;
            f32 y = mv.m[1][0] * c.x + mv.m[1][1] * c.y + mv.m[1][2] * c.z;
            f32 z = mv.m[2][0] * c.x + mv.m[2][1] * c.y + mv.m[2][2] * c.z;
            const f32 len = std::sqrt(x * x + y * y + z * z);
            if (len > 0.0f) { x /= len; y /= len; z /= len; }
            cbfd_ldir_[l][0] = x; cbfd_ldir_[l][1] = y; cbfd_ldir_[l][2] = z;
        }
    }

    for (u32 i = 0; i < count; ++i) {
        u32 cur_vtx = vtx_addr + i * stride;
        if (cur_vtx + stride > rdram_size) break;
        u32 dest_idx = (dest + i) % vertex_cache.size();
        // The 4 color/normal bytes.
        const u8* col = rdram + cur_vtx + (dkr ? 6 : 12);
        if (pd) col = rdram + ((vtx_color_base + rdram[cur_vtx + 7]) & (static_cast<u32>(rdram_size) - 4));

        s16 vx = static_cast<s16>((rdram[cur_vtx + 0] << 8) | rdram[cur_vtx + 1]);
        s16 vy = static_cast<s16>((rdram[cur_vtx + 2] << 8) | rdram[cur_vtx + 3]);
        s16 vz = static_cast<s16>((rdram[cur_vtx + 4] << 8) | rdram[cur_vtx + 5]);
        s16 tu = dkr ? 0 : static_cast<s16>((rdram[cur_vtx + 8] << 8) | rdram[cur_vtx + 9]);
        s16 tv = dkr ? 0 : static_cast<s16>((rdram[cur_vtx + 10] << 8) | rdram[cur_vtx + 11]);

        Vertex& v = vertex_cache[dest_idx];
        raw_vertex[dest_idx] = {static_cast<f32>(vx), static_cast<f32>(vy), static_cast<f32>(vz), cur_vtx};
        combined_matrix.transform_point(vx, vy, vz, v.x, v.y, v.z, v.w);
        if (dkr && dkr_billboard && dest_idx != 0) {
            // Billboards are given relative to vertex 0, in clip space.
            const Vertex& o = vertex_cache[0];
            v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
        }
        compute_screen_coords(v);

        f32 tnx = 0.0f, tny = 0.0f, tnz = 1.0f;
        if ((geometry_mode & 0x00020000) || (geometry_mode & 0x00040000)) {
            s8 nx_i = static_cast<s8>(col[0]);
            s8 ny_i = static_cast<s8>(col[1]);
            s8 nz_i = static_cast<s8>(col[2]);
            f32 nx = nx_i / 127.0f;
            f32 ny = ny_i / 127.0f;
            f32 nz = nz_i / 127.0f;

            const auto& mv = modelview_stack.empty() ? Matrix4x4::identity() : modelview_stack.back();
            tnx = nx * mv.m[0][0] + ny * mv.m[1][0] + nz * mv.m[2][0];
            tny = nx * mv.m[0][1] + ny * mv.m[1][1] + nz * mv.m[2][1];
            tnz = nx * mv.m[0][2] + ny * mv.m[1][2] + nz * mv.m[2][2];
            f32 tlen = std::sqrt(tnx * tnx + tny * tny + tnz * tnz);
            if (tlen > 0.0f) { tnx /= tlen; tny /= tlen; tnz /= tlen; }
        }

        if (geometry_mode & 0x00040000) { // G_TEXTURE_GEN (spherical mapping)
            f32 dot_x = tnx, dot_y = tny;
            if (lookat_set) {
                s8 nx_i = static_cast<s8>(col[0]);
                s8 ny_i = static_cast<s8>(col[1]);
                s8 nz_i = static_cast<s8>(col[2]);
                f32 nx = nx_i / 127.0f;
                f32 ny = ny_i / 127.0f;
                f32 nz = nz_i / 127.0f;
                dot_x = nx * lookat_x.dx + ny * lookat_x.dy + nz * lookat_x.dz;
                dot_y = nx * lookat_y.dx + ny * lookat_y.dy + nz * lookat_y.dz;
            }
            v.u = (dot_x * 0.5f + 0.5f) * 32.0f * texture_scale_s;
            v.v = (dot_y * 0.5f + 0.5f) * 32.0f * texture_scale_t;
        } else {
            v.u = (tu / 32.0f) * texture_scale_s;
            v.v = (tv / 32.0f) * texture_scale_t;
        }

        if ((geometry_mode & 0x00020000) && cbfd_ && ucode == MicrocodeType::F3DEX2) {
            // F3DEXBG: the colour stays the vertex's own, scaled by the light;
            // the normal comes from the separate table (2 bytes per vertex
            // slot) and the low byte of the vertex's flag word.
            const u32 nb = (cbfd_normal_base_ + dest_idx * 2) & (static_cast<u32>(rdram_size) - 2);
            const f32 nx = static_cast<s8>(rdram[nb]) / 127.0f;
            const f32 ny = static_cast<s8>(rdram[nb + 1]) / 127.0f;
            const f32 nz = static_cast<s8>(rdram[cur_vtx + 7]) / 127.0f;
            const u32 nl = cbfd_num_lights_;
            const f32* m = cbfd_coord_mod_;
            const f32 px = (vx + m[8]) * m[12], py = (vy + m[9]) * m[13], pz = (vz + m[10]) * m[14];
            f32 lr = cbfd_lights_[nl].r, lg = cbfd_lights_[nl].g, lb = cbfd_lights_[nl].b;
            auto add = [&](const CbfdLight& l, f32 k) {
                if (k > 0.0f) { lr += l.r * k; lg += l.g * k; lb += l.b * k; }
            };
            auto point = [&](const CbfdLight& l) {
                const f32 dx = px - l.px, dy = py - l.py, dz = pz - l.pz;
                const f32 len = 2.0f * (dx * dx + dy * dy + dz * dz) / 65536.0f;
                return len > 0.0f ? std::min(1.0f, l.ca / len) : 1.0f;
            };
            if (static_cast<s8>(rdram[cur_vtx + 6]) < 0) {
                lr = lg = lb = 1.0f; // a negative flag word: not lit
            } else if (!cbfd_advanced_) {
                for (int l = static_cast<int>(nl) - 2; l >= 0; --l) add(cbfd_lights_[l], point(cbfd_lights_[l]));
            } else if (nl > 0) {
                auto dot = [&](int l) {
                    return std::min(1.0f, nx * cbfd_ldir_[l][0] + ny * cbfd_ldir_[l][1] + nz * cbfd_ldir_[l][2]);
                };
                add(cbfd_lights_[nl - 1], dot(static_cast<int>(nl) - 1));
                for (int l = static_cast<int>(nl) - 2; l >= 0; --l) {
                    f32 k = point(cbfd_lights_[l]);
                    if (geometry_mode & 0x00400000) k *= dot(l); // G_POINT_LIGHTING
                    add(cbfd_lights_[l], k);
                }
            }
            v.r = static_cast<u8>(col[0] * std::min(1.0f, lr));
            v.g = static_cast<u8>(col[1] * std::min(1.0f, lg));
            v.b = static_cast<u8>(col[2] * std::min(1.0f, lb));
            v.a = col[3];
        } else if (geometry_mode & 0x00020000) { // G_LIGHTING
            u8 ca = col[3];
            f32 lit_r = ambient_light.r;
            f32 lit_g = ambient_light.g;
            f32 lit_b = ambient_light.b;

            for (const auto& l : dir_lights) {
                f32 dot = tnx * l.dx + tny * l.dy + tnz * l.dz;
                if (dot > 0.0f) {
                    lit_r += l.r * dot;
                    lit_g += l.g * dot;
                    lit_b += l.b * dot;
                }
            }

            v.r = static_cast<u8>(std::clamp(lit_r, 0.0f, 255.0f));
            v.g = static_cast<u8>(std::clamp(lit_g, 0.0f, 255.0f));
            v.b = static_cast<u8>(std::clamp(lit_b, 0.0f, 255.0f));
            v.a = ca; // Alpha=0 is a valid, common value (fades, transparency) - must not be forced opaque
        } else {
            u8 cr = col[0];
            u8 cg = col[1];
            u8 cb = col[2];
            u8 ca = col[3];
            v.r = cr; v.g = cg; v.b = cb;
            v.a = ca;
        }
        // G_FOG: the shade alpha is the fog factor, z/w scaled and offset as
        // gSPFogPosition set it up (0 behind the eye), for the blender's A_SHADE.
        if ((geometry_mode & 0x00010000) && fog_supported(ucode)) {
            const f32 f = v.w > 0.0f ? (v.z / v.w) * fog_mul + fog_ofs : 0.0f;
            v.a = static_cast<u8>(std::clamp(f, 0.0f, 255.0f));
        }
    }
}

// The microcodes whose G_FOG puts the fog factor in the shade alpha.
bool RDP::fog_supported(MicrocodeType u) {
    return u == MicrocodeType::Fast3D || u == MicrocodeType::F3DEX || u == MicrocodeType::F3DEX2 ||
           u == MicrocodeType::F3DGOLDEN || u == MicrocodeType::F3DPD || u == MicrocodeType::F3DDKR ||
           u == MicrocodeType::F3DJFG;
}

void RDP::compute_screen_coords(Vertex& v) const {
    f32 inv_w = (v.w != 0.0f) ? (1.0f / v.w) : 1.0f;
    v.sx = vp_trans_x + (v.x * inv_w) * vp_scale_x;
    v.sy = vp_trans_y - (v.y * inv_w) * vp_scale_y;
    f32 scr_z = vp_trans_z + (v.z * inv_w) * vp_scale_z;
    v.sz = std::clamp(scr_z / 1023.0f, 0.0f, 1.0f);

    v.clip_flags = 0;
    if (v.x > v.w)  v.clip_flags |= 0x01; // Right
    if (v.x < -v.w) v.clip_flags |= 0x02; // Left
    if (v.y > v.w)  v.clip_flags |= 0x04; // Top
    if (v.y < -v.w) v.clip_flags |= 0x08; // Bottom
    if (v.z > v.w)  v.clip_flags |= 0x10; // Far
    if (v.z < -v.w) v.clip_flags |= 0x20; // Near
}

static Vertex lerp_vertex(const Vertex& a, const Vertex& b, f32 t) {
    Vertex res;
    res.x = a.x + t * (b.x - a.x);
    res.y = a.y + t * (b.y - a.y);
    res.z = a.z + t * (b.z - a.z);
    res.w = a.w + t * (b.w - a.w);
    res.u = a.u + t * (b.u - a.u);
    res.v = a.v + t * (b.v - a.v);
    res.r = static_cast<u8>(std::clamp(a.r + t * (b.r - a.r), 0.0f, 255.0f));
    res.g = static_cast<u8>(std::clamp(a.g + t * (b.g - a.g), 0.0f, 255.0f));
    res.b = static_cast<u8>(std::clamp(a.b + t * (b.b - a.b), 0.0f, 255.0f));
    res.a = static_cast<u8>(std::clamp(a.a + t * (b.a - a.a), 0.0f, 255.0f));
    return res;
}

template <typename PlaneFn>
static void clip_polygon_plane(const std::vector<Vertex>& in_poly, std::vector<Vertex>& out_poly, PlaneFn plane_eval) {
    out_poly.clear();
    if (in_poly.empty()) return;

    for (size_t i = 0; i < in_poly.size(); ++i) {
        const Vertex& cur = in_poly[i];
        const Vertex& prev = in_poly[(i + in_poly.size() - 1) % in_poly.size()];

        f32 d_cur = plane_eval(cur);
        f32 d_prev = plane_eval(prev);

        if (d_cur >= 0.0f) {
            if (d_prev < 0.0f) {
                f32 t = d_prev / (d_prev - d_cur);
                out_poly.push_back(lerp_vertex(prev, cur, t));
            }
            out_poly.push_back(cur);
        } else if (d_prev >= 0.0f) {
            f32 t = d_prev / (d_prev - d_cur);
            out_poly.push_back(lerp_vertex(prev, cur, t));
        }
    }
}

void RDP::finish_texture_run() {
    if (!tex_run.active) return;
    tex_run.active = false;
    if (tex_run.index < 0 || tex_run.index >= static_cast<s32>(capture_textures.size())) return;
    CapturedTexture& t = capture_textures[tex_run.index];
    // Decode the texel range the triangles actually use, so repeats and mirrors
    // are baked in. Large ranges (tiled terrain) are sampled with a stride to
    // keep the image at most 256 pixels per axis.
    s32 s0 = static_cast<s32>(std::floor(tex_run.min_s)), s1 = static_cast<s32>(std::ceil(tex_run.max_s));
    s32 t0 = static_cast<s32>(std::floor(tex_run.min_t)), t1 = static_cast<s32>(std::ceil(tex_run.max_t));
    s1 = std::clamp(s1, s0 + 1, s0 + 8192);
    t1 = std::clamp(t1, t0 + 1, t0 + 8192);
    const s32 step_s = std::max(1, (s1 - s0 + 255) / 256), step_t = std::max(1, (t1 - t0 + 255) / 256);
    const u32 w = static_cast<u32>((s1 - s0 + step_s - 1) / step_s), h = static_cast<u32>((t1 - t0 + step_t - 1) / step_t);
    u64 key = tex_run.key;
    for (s32 v : {s0, t0, s1, t1}) { key ^= static_cast<u32>(v); key *= 1099511628211ull; }
    auto cached = texture_decode_cache.find(key);
    if (cached != texture_decode_cache.end()) {
        t = cached->second;
        return;
    }
    t.key = key;
    t.origin_s = static_cast<f32>(s0);
    t.origin_t = static_cast<f32>(t0);
    t.width = w;
    t.height = h;
    t.span_s = static_cast<f32>(w * step_s);
    t.span_t = static_cast<f32>(h * step_t);
    t.argb.resize(static_cast<size_t>(w) * h);
    raster::TexUnit tu;
    tu.prepare(tex_run.tile);
    for (u32 y = 0; y < h; ++y)
        for (u32 x = 0; x < w; ++x)
            t.argb[y * w + x] = raster::fetch_texel(tu, tex_run.tmem.data(), tex_run.dxt.data(), tex_run.tlut,
                                                    s0 + static_cast<s32>(x) * step_s, t0 + static_cast<s32>(y) * step_t);
    if (texture_decode_cache.size() > 512) texture_decode_cache.clear();
    texture_decode_cache.emplace(key, t);
}

void RDP::emit_triangle(u32 a, u32 b, u32 c, u8* rdram, size_t rdram_size) {
    if (capture_enabled && capture_tris < 400000) {
        const Matrix4x4 mv = modelview_stack.empty() ? Matrix4x4::identity() : modelview_stack.back();
        if (capture_frame.empty() || std::memcmp(&capture_frame.back().modelview, &mv, sizeof(Matrix4x4)) != 0) {
            if (capture_frame.size() < 8192) {
                CapturedMesh m;
                m.modelview = mv;
                m.mtx_addr = capture_mtx_addr;
                m.vtx_addr = raw_vertex[a % raw_vertex.size()].src;
                m.dl_addr = capture_dl_addr;
                capture_frame.push_back(std::move(m));
            }
        }
        if (!capture_frame.empty()) {
            CapturedMesh& m = capture_frame.back();
            for (u32 idx : {a, b, c}) {
                const RawVertex& rv = raw_vertex[idx % raw_vertex.size()];
                m.pos.push_back(rv.x);
                m.pos.push_back(rv.y);
                m.pos.push_back(rv.z);
            }
            const Vertex& cv = vertex_cache[a % vertex_cache.size()];
            m.col.push_back((u32(cv.r) << 24) | (u32(cv.g) << 16) | (u32(cv.b) << 8) | u32(cv.a));
            capture_tris++;

            // Texture of the selected mesh: remember the active tile + TMEM for this
            // run of triangles and their texel coordinates.
            if (capture_texture_vtx != 0 && m.vtx_addr == capture_texture_vtx && texture_enabled) {
                const Tile& tile = tiles[active_tile & 7];
                const u32 tlut = (other_mode_h >> 14) & 0x3;
                u64 key = tex_last_key;
                if (tmem_dirty || tlut != tex_last_tlut || std::memcmp(&tile, &tex_last_tile, sizeof tile) != 0) {
                    key = 1469598103934665603ull;
                    auto mixb = [&](const void* p, size_t n) {
                        const u8* b = static_cast<const u8*>(p);
                        for (size_t i = 0; i < n; ++i) { key ^= b[i]; key *= 1099511628211ull; }
                    };
                    mixb(&tile, sizeof tile);
                    mixb(&tlut, sizeof tlut);
                    mixb(tmem.data(), tmem.size());
                    tex_last_key = key;
                    tex_last_tile = tile;
                    tex_last_tlut = tlut;
                    tmem_dirty = false;
                }
                if (!tex_run.active || key != tex_run.key) {
                    finish_texture_run();
                    tex_run.active = true;
                    tex_run.key = key;
                    tex_run.index = static_cast<s32>(capture_textures.size());
                    capture_textures.emplace_back();
                    tex_run.tile = tile;
                    tex_run.tlut = tlut;
                    tex_run.tmem = tmem;
                    tex_run.dxt = tmem_word_dxt_zero;
                    tex_run.min_s = tex_run.min_t = 1e9f;
                    tex_run.max_s = tex_run.max_t = -1e9f;
                }
                for (u32 idx : {a, b, c}) {
                    const Vertex& v = vertex_cache[idx % vertex_cache.size()];
                    f32 st = apply_tile_shift(v.u, tile.shift_s) - tile.sl / 4.0f;
                    f32 tt = apply_tile_shift(v.v, tile.shift_t) - tile.tl / 4.0f;
                    m.uv.push_back(st);
                    m.uv.push_back(tt);
                    tex_run.min_s = std::min(tex_run.min_s, st);
                    tex_run.max_s = std::max(tex_run.max_s, st);
                    tex_run.min_t = std::min(tex_run.min_t, tt);
                    tex_run.max_t = std::max(tex_run.max_t, tt);
                }
                m.tex.push_back(tex_run.index);
            } else if (capture_texture_vtx != 0 && m.vtx_addr == capture_texture_vtx) {
                for (int k = 0; k < 6; ++k) m.uv.push_back(0.0f);
                m.tex.push_back(-1);
            }
        }
    }
    if (pick_frame_ == -2 || (pick_frame_ >= 0 && g_current_frame == pick_frame_)) debug_pick(a, b, c);
    clip_and_rasterize_triangle(vertex_cache[a], vertex_cache[b], vertex_cache[c], rdram, rdram_size);
}

// ORBIT64_PICK=x,y,frame: every triangle of that frame that covers pixel
// x,y (in its unclipped screen position), with the state it is drawn with.
void RDP::debug_pick(u32 a, u32 b, u32 c) const {
    const Vertex& A = vertex_cache[a];
    const Vertex& B = vertex_cache[b];
    const Vertex& C = vertex_cache[c];
    if (A.w <= 0.0f || B.w <= 0.0f || C.w <= 0.0f) return; // (partly) behind the camera: not placed right
    const f32 px = pick_x_ + 0.5f, py = pick_y_ + 0.5f;
    auto edge = [](const Vertex& p, const Vertex& q, f32 x, f32 y) {
        return (q.sx - p.sx) * (y - p.sy) - (q.sy - p.sy) * (x - p.sx);
    };
    const f32 e0 = edge(A, B, px, py), e1 = edge(B, C, px, py), e2 = edge(C, A, px, py);
    if (!((e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0))) return;
    std::fprintf(stderr, "PICK frame=%d dl=%08x ucode=%d geom=%08x omh=%08x oml=%08x comb=%08x:%08x prim=%08x env=%08x fog=%08x fm=%d fo=%d tex=%d tile=%u\n",
                 g_current_frame, capture_dl_addr, static_cast<int>(current_ucode_active), geometry_mode, other_mode_h, other_mode_l,
                 combine_mode_w0, combine_mode_w1, prim_color, env_color, fog_color, fog_mul, fog_ofs, texture_enabled, active_tile);
    for (int i = 0; i < 2; ++i) {
        const Tile& t = tiles[(active_tile + i) & 7];
        std::fprintf(stderr, "  tile%d fmt=%u siz=%u line=%u tmem=%u pal=%u cms=%u/%u/%u/%u cmt=%u/%u/%u/%u sl=%u tl=%u sh=%u th=%u\n",
                     (active_tile + i) & 7, t.format, t.size, t.line, t.tmem, t.palette, t.clamp_s, t.mirror_s, t.mask_s,
                     t.shift_s, t.clamp_t, t.mirror_t, t.mask_t, t.shift_t, t.sl, t.tl, t.sh, t.th);
    }
    for (const Vertex* v : {&A, &B, &C})
        std::fprintf(stderr, "  v sx=%.1f sy=%.1f sz=%.4f w=%.2f uv=%.2f,%.2f rgba=%02x%02x%02x%02x\n", v->sx, v->sy, v->sz,
                     v->w, v->u, v->v, v->r, v->g, v->b, v->a);
    // ORBIT64_PICK_DUMP=<dir>: the draw's tiles, decoded, as PPM images (colour, then alpha).
    if (const char* dir = std::getenv("ORBIT64_PICK_DUMP")) {
        static int n = 0;
        const DrawState& st = const_cast<RDP*>(this)->draw_state();
        for (int i = 0; i < 2; ++i) {
            const raster::TexUnit& tu = st.tex[(active_tile + i) & 7];
            u32 w = 0, h = 0;
            raster::tex_cache_dims(tu, w, h);
            if (w * h == 0 || w * h > 1024 * 1024) continue;
            char path[512];
            std::snprintf(path, sizeof path, "%s/pick%03d_tile%d.ppm", dir, n, i);
            if (FILE* f = std::fopen(path, "wb")) {
                std::fprintf(f, "P6 %u %u 255\n", w * 2, h);
                for (u32 t = 0; t < h; ++t)
                    for (u32 k = 0; k < w * 2; ++k) {
                        const u32 c = raster::fetch_wrapped(tu, st.tmem, st.tmem_dxt, st.tlut_type, static_cast<s32>(k % w), static_cast<s32>(t));
                        const u8 px[3] = {static_cast<u8>(k < w ? c >> 16 : c >> 24), static_cast<u8>(k < w ? c >> 8 : c >> 24),
                                          static_cast<u8>(k < w ? c : c >> 24)};
                        std::fwrite(px, 1, 3, f);
                    }
                std::fclose(f);
            }
        }
        ++n;
    }
}

void RDP::clip_and_rasterize_triangle(Vertex v0, Vertex v1, Vertex v2, u8* rdram, size_t rdram_size) {
    if (geometry_only_) return;
    stat_tri_called++;

    const f32 NEAR_W = 0.1f;

    // Trivial rejection tests: if all 3 vertices are behind the near plane, drop the triangle
    if (v0.w < NEAR_W && v1.w < NEAR_W && v2.w < NEAR_W) return;

    // Trivial rejection against frustum sides is only valid when w >= NEAR_W for all vertices,
    // because dividing/multiplying inequalities by negative w flips the inequality signs.
    if (v0.w >= NEAR_W && v1.w >= NEAR_W && v2.w >= NEAR_W) {
        if (v0.x > v0.w && v1.x > v1.w && v2.x > v2.w) return;
        if (v0.x < -v0.w && v1.x < -v1.w && v2.x < -v2.w) return;
        if (v0.y > v0.w && v1.y > v1.w && v2.y > v2.w) return;
        if (v0.y < -v0.w && v1.y < -v1.w && v2.y < -v2.w) return;
        if (!no_near_clip_) {
            if (v0.z > v0.w && v1.z > v1.w && v2.z > v2.w) return;
            if (v0.z < -v0.w && v1.z < -v1.w && v2.z < -v2.w) return;
        }
    }

    // Trivial acceptance: if all 3 vertices are completely inside all frustum planes
    // A .NoN microcode leaves depth out of it (it is clamped instead).
    const bool zc = !no_near_clip_;
    bool v0_in = (v0.w >= NEAR_W && v0.x >= -v0.w && v0.x <= v0.w && v0.y >= -v0.w && v0.y <= v0.w && (!zc || (v0.z >= -v0.w && v0.z <= v0.w)));
    bool v1_in = (v1.w >= NEAR_W && v1.x >= -v1.w && v1.x <= v1.w && v1.y >= -v1.w && v1.y <= v1.w && (!zc || (v1.z >= -v1.w && v1.z <= v1.w)));
    bool v2_in = (v2.w >= NEAR_W && v2.x >= -v2.w && v2.x <= v2.w && v2.y >= -v2.w && v2.y <= v2.w && (!zc || (v2.z >= -v2.w && v2.z <= v2.w)));

    if (v0_in && v1_in && v2_in) {
        rasterize_triangle(v0, v1, v2, rdram, rdram_size);
        return;
    }

    // Straddling triangle: Sutherland-Hodgman 4D clipping
    std::vector<Vertex> poly1 = {v0, v1, v2};
    std::vector<Vertex> poly2;

    auto clip_against = [&](auto plane_eval) {
        clip_polygon_plane(poly1, poly2, plane_eval);
        poly1 = std::move(poly2);
    };

    clip_against([&](const Vertex& v) { return v.w - NEAR_W; });
    if (poly1.size() < 3) return;
    clip_against([&](const Vertex& v) { return v.w + v.x; });
    if (poly1.size() < 3) return;
    clip_against([&](const Vertex& v) { return v.w - v.x; });
    if (poly1.size() < 3) return;
    clip_against([&](const Vertex& v) { return v.w + v.y; });
    if (poly1.size() < 3) return;
    clip_against([&](const Vertex& v) { return v.w - v.y; });
    if (poly1.size() < 3) return;
    if (zc) {
        clip_against([&](const Vertex& v) { return v.w + v.z; });
        if (poly1.size() < 3) return;
        clip_against([&](const Vertex& v) { return v.w - v.z; });
        if (poly1.size() < 3) return;
    }

    for (auto& v : poly1) {
        compute_screen_coords(v);
    }

    for (size_t i = 1; i + 1 < poly1.size(); ++i) {
        rasterize_triangle(poly1[0], poly1[i], poly1[i + 1], rdram, rdram_size);
    }
}

void RDP::clip_and_rasterize_line(Vertex v0, Vertex v1, u8* rdram, size_t rdram_size) {
    if (geometry_only_) return;
    flush_native();
    const f32 NEAR_W = 0.1f;
    if (v0.w < NEAR_W && v1.w < NEAR_W) return;

    if (v0.w < NEAR_W) {
        f32 t = (NEAR_W - v0.w) / (v1.w - v0.w);
        v0 = lerp_vertex(v0, v1, t);
    } else if (v1.w < NEAR_W) {
        f32 t = (NEAR_W - v1.w) / (v0.w - v1.w);
        v1 = lerp_vertex(v1, v0, t);
    }

    compute_screen_coords(v0);
    compute_screen_coords(v1);

    int x0 = static_cast<int>(v0.sx);
    int y0 = static_cast<int>(v0.sy);
    int x1 = static_cast<int>(v1.sx);
    int y1 = static_cast<int>(v1.sy);

    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;

    int steps = std::max(dx, dy);
    int step = 0;

    const DrawState& st = draw_state();
    HiResTarget* hr = hires_target(rdram, rdram_size);
    hires_line_px_.clear();
    while (true) {
        f32 t = steps > 0 ? static_cast<f32>(step) / steps : 0.0f;
        f32 z = (1.0f - t) * v0.sz + t * v1.sz;
        u8 r = static_cast<u8>((1.0f - t) * v0.r + t * v1.r);
        u8 g = static_cast<u8>((1.0f - t) * v0.g + t * v1.g);
        u8 b = static_cast<u8>((1.0f - t) * v0.b + t * v1.b);
        u8 a = static_cast<u8>((1.0f - t) * v0.a + t * v1.a);
        u32 color = (static_cast<u32>(a) << 24) | (static_cast<u32>(r) << 16) | (static_cast<u32>(g) << 8) | b;
        write_pixel(st, x0, y0, color, z, rdram, rdram_size);
        // The high-resolution pass draws each line pixel as a block.
        if (hr && x0 >= 0 && y0 >= 0 && x0 < 4096 && y0 < static_cast<int>(kFbLines))
            hires_line_px_.push_back({static_cast<u16>(x0), static_cast<u16>(y0), color, z});

        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
        step++;
    }
    if (hr) hires_->pixels(hr, st, draw_state_serial_, hires_line_px_);
}

void RDP::execute_moveword(u32 w0, u32 w1, MicrocodeType current_ucode) {
    u8 type;
    u32 offset;
    if (current_ucode == MicrocodeType::F3DEX2) {
        type = (w0 >> 16) & 0xFF;
        offset = w0 & 0xFFFF;
    } else {
        type = w0 & 0xFF;
        offset = (w0 >> 8) & 0xFFFF;
    }

    if (type == 0x06) { // G_MW_SEGMENT: offset in bytes = segment * 4
        u32 seg_idx = offset / 4;
        if (seg_idx < segments.size()) {
            segments[seg_idx] = w1;
        }
    } else if (type == 0x02) { // G_MW_NUMLIGHT
        u32 raw = w1 & 0x7FFFFFFF;
        if (current_ucode == MicrocodeType::F3DEX2) {
            num_lights = (raw / 24 > 0) ? (raw / 24) : 0;
        } else {
            num_lights = (raw / 32 > 0) ? (raw / 32 - 1) : 0;
        }
        if (num_lights > 8) num_lights = 8;
        dir_lights.resize(num_lights);
    } else if (type == 0x0A) { // G_MW_LIGHTCOL
        u8 r = (w1 >> 24) & 0xFF;
        u8 g = (w1 >> 16) & 0xFF;
        u8 b = (w1 >> 8) & 0xFF;
        u32 slot = (current_ucode == MicrocodeType::F3DEX2) ? (offset / 24) : (offset / 32);
        if (slot < dir_lights.size()) {
            dir_lights[slot].r = r;
            dir_lights[slot].g = g;
            dir_lights[slot].b = b;
        } else if (slot == num_lights) {
            ambient_light.r = r;
            ambient_light.g = g;
            ambient_light.b = b;
        }
    } else if (type == 0x08 && is_s2dex_ucode(current_ucode)) { // G_MW_GENSTAT (S2DEX gSPSetStatus)
        u32 slot = offset / 4;
        if (slot < 4) s2d_genstat[slot] = w1;
    } else if (type == 0x08) { // G_MW_FOG: gSPFogPosition's multiplier and offset
        fog_mul = static_cast<s16>(w1 >> 16);
        fog_ofs = static_cast<s16>(w1 & 0xFFFF);
    } else if (type == 0x0C) { // G_MW_POINTS (Fast3D gsSPModifyVertex)
        u32 vtx_idx = offset / 40;
        u8 where = offset % 40;
        if (vtx_idx < vertex_cache.size()) {
            Vertex& v = vertex_cache[vtx_idx];
            switch (where) {
                case 0x10: // G_MWO_POINT_RGBA
                    v.r = (w1 >> 24) & 0xFF;
                    v.g = (w1 >> 16) & 0xFF;
                    v.b = (w1 >> 8) & 0xFF;
                    v.a = w1 & 0xFF;
                    break;
                case 0x14: // G_MWO_POINT_ST
                    v.u = (static_cast<s16>(w1 >> 16) / 32.0f) * texture_scale_s;
                    v.v = (static_cast<s16>(w1 & 0xFFFF) / 32.0f) * texture_scale_t;
                    break;
                case 0x18: // G_MWO_POINT_XYSCREEN
                    v.sx = static_cast<s16>(w1 >> 16) / 4.0f;
                    v.sy = static_cast<s16>(w1 & 0xFFFF) / 4.0f;
                    break;
                case 0x1C: // G_MWO_POINT_ZSCREEN
                    v.sz = std::clamp((static_cast<s16>(w1 >> 16)) / 1024.0f, 0.0f, 1.0f);
                    break;
            }
        }
    }
}

// The commands in which F3DEXBG (Conker's Bad Fur Day) differs from F3DEX2;
// false for the others.
bool RDP::execute_cbfd_command(u8 opcode, u32 w0, u32 w1, u8* rdram, size_t rdram_size) {
    if (opcode >= 0x10 && opcode <= 0x1F) { // G_TRI4: 5-bit vertex indices
        const u32 idx[12] = {(w0 >> 23) & 31, (w0 >> 18) & 31, (((w0 >> 15) & 7) << 2) | (w1 >> 30),
                             (w0 >> 10) & 31, (w0 >> 5) & 31,  w0 & 31,
                             (w1 >> 25) & 31, (w1 >> 20) & 31, (w1 >> 15) & 31,
                             (w1 >> 10) & 31, (w1 >> 5) & 31,  w1 & 31};
        for (int t = 0; t < 4; ++t) {
            const u32 a = idx[t * 3], b = idx[t * 3 + 1], c = idx[t * 3 + 2];
            if (a == b && b == c) continue; // an unused slot
            emit_triangle(a, b, c, rdram, rdram_size);
        }
        return true;
    }
    switch (opcode) {
        case 0xDD: // G_LOAD_UCODE: here it switches to the second lighting mode
            cbfd_advanced_ = true;
            return true;
        case 0xDB: { // G_MOVEWORD
            const u32 type = (w0 >> 16) & 0xFF;
            if (type == 0x02) { // G_MW_NUMLIGHT: 48-byte lights
                cbfd_num_lights_ = std::min<u32>(w1 / 48, 12);
                return true;
            }
            if (type == 0x10) { // G_MW_COORD_MOD: how light positions relate to vertices
                if (w0 & 8) return true;
                const u32 i = (w0 >> 1) & 3, pos = w0 & 0x30;
                f32* m = cbfd_coord_mod_;
                if (pos == 0) {
                    m[0 + i] = static_cast<s16>(w1 >> 16);
                    m[1 + i] = static_cast<s16>(w1);
                } else if (pos == 0x10) {
                    m[4 + i] = static_cast<f32>(w1 >> 16) / 65536.0f;
                    m[5 + i] = static_cast<f32>(w1 & 0xFFFF) / 65536.0f;
                    m[12 + i] = m[0 + i] + m[4 + i];
                    m[13 + i] = m[1 + i] + m[5 + i];
                } else if (pos == 0x20) {
                    m[8 + i] = static_cast<s16>(w1 >> 16);
                    m[9 + i] = static_cast<s16>(w1);
                }
                return true;
            }
            return false;
        }
        case 0xDC: { // G_MOVEMEM
            const u32 idx = w0 & 0xFF;
            const u32 src = segment_to_physical(w1);
            if (idx == 14) { // G_MV_NORMALES: the per-vertex normals
                cbfd_normal_base_ = src;
                return true;
            }
            if (idx != 10 || src + 40 > rdram_size) return false;
            const u32 n = ((w0 >> 5) & 0x3FFF) / 48;
            auto dir = [&](u32 o) { return static_cast<s8>(rdram[src + o]) / 127.0f; };
            if (n < 2) { // LookAt X / Y
                Light& la = n == 0 ? lookat_x : lookat_y;
                la = {rdram[src + 0], rdram[src + 1], rdram[src + 2], dir(8), dir(9), dir(10)};
                lookat_set = true;
                return true;
            }
            if (n - 2 >= cbfd_lights_.size()) return true;
            CbfdLight& l = cbfd_lights_[n - 2];
            l.r = rdram[src + 0] / 255.0f;
            l.g = rdram[src + 1] / 255.0f;
            l.b = rdram[src + 2] / 255.0f;
            f32 x = static_cast<s8>(rdram[src + 8]), y = static_cast<s8>(rdram[src + 9]), z = static_cast<s8>(rdram[src + 10]);
            const f32 len = std::sqrt(x * x + y * y + z * z);
            if (len > 0.0f) { x /= len; y /= len; z /= len; }
            l.x = x; l.y = y; l.z = z;
            auto s16at = [&](u32 o) { return static_cast<f32>(static_cast<s16>((rdram[src + o] << 8) | rdram[src + o + 1])); };
            l.px = s16at(32);
            l.py = s16at(34);
            l.pz = s16at(36);
            l.ca = rdram[src + 12] / 16.0f;
            return true;
        }
        default:
            return false;
    }
}

void RDP::process_display_list(u32 dl_addr, u8* rdram, size_t rdram_size, MI& mi) {
    display_list_count++;
    dir_lights.clear();
    num_lights = 0;

    // The CPU may have changed RDRAM (and the microcode) since the last list.
    draw_state_dirty_ = true;
    raw_mode_ = false;
    if (hires_) hires_->unbind();
    // Whatever this list draws at high resolution can start right away.
    // ... and the queued native draws have to be in RDRAM before the CPU runs on.
    struct FlushOnExit {
        RDP& rdp;
        HiResRenderer* hr;
        ~FlushOnExit() {
            rdp.flush_native();
            if (hr) hr->flush();
        }
    } flush_on_exit{*this, hires_.get()};

    MicrocodeType current_ucode = ucode_type;
    // S2DEX keeps its status words in DMEM, which every task (and every
    // microcode load) starts from the microcode's data: zero.
    std::fill(std::begin(s2d_genstat), std::end(s2d_genstat), 0u);

    // The banner sniff (rsp.cpp / G_LOAD_UCODE below) can find a real, but
    // *stale or unrelated*, ucode credit string sitting in RDRAM: some ROMs
    // keep several microcode blobs resident even though only one is actually
    // driving a given display list (e.g. Dr. Mario 64 keeps a genuine
    // "RSP Gfx ucode S2DEX 1.07" banner around while a given task's actual
    // command stream is really F3DEX2). Blindly trusting that banner makes
    // every opcode of an unrelated F3D display list get misread as an S2DEX
    // command, drawing garbage or nothing (F3D's own G_VTX/G_TRI* opcodes
    // alias exactly the low-numbered opcodes S2DEX repurposes for BG/OBJ
    // drawing). So require *positive* evidence before trusting it: every
    // S2DEX opcode that carries a DMA pointer (BG/OBJ draws) is emitted with
    // a fixed zero low-24-bit word, so a well-formed one is proof the stream
    // really is S2DEX, and G_GEOMETRYMODE (0xD9, not part of the S2DEX
    // command set in either GBI generation) or a malformed low-24-bit word
    // on one of those same opcodes is proof it isn't. Ambiguous/no evidence
    // defaults to "not S2DEX", since a real S2DEX list is normally saturated
    // with BG/OBJ commands and should confirm itself almost immediately.
    if (current_ucode == MicrocodeType::S2DEX || current_ucode == MicrocodeType::S2DEX2) {
        bool confirmed_s2dex = false;
        bool confirmed_not_s2dex = false;
        u32 scan_pc = dl_addr;
        int depth = 0; // how many G_DL branches we've followed while scanning
        std::vector<u32> scan_stack;
        for (int i = 0; i < 400 && depth < 64 && !confirmed_not_s2dex; ++i) {
            u32 phys = segment_to_physical(scan_pc);
            if (phys + 8 > rdram_size) break;
            u8 op = rdram[phys];
            u32 w0 = (static_cast<u32>(rdram[phys + 0]) << 24) |
                     (static_cast<u32>(rdram[phys + 1]) << 16) |
                     (static_cast<u32>(rdram[phys + 2]) << 8)  |
                      static_cast<u32>(rdram[phys + 3]);
            u32 w1 = (static_cast<u32>(rdram[phys + 4]) << 24) |
                     (static_cast<u32>(rdram[phys + 5]) << 16) |
                     (static_cast<u32>(rdram[phys + 6]) << 8)  |
                      static_cast<u32>(rdram[phys + 7]);
            const bool gbi2 = current_ucode == MicrocodeType::S2DEX2;
            if (op == 0xD9) { confirmed_not_s2dex = true; break; }
            bool is_zero_l_opcode = gbi2
                ? (op == 0x01 || op == 0x02 || op == 0x09 || op == 0x0a)   // OBJ_RECT/SPRITE, BG_1CYC/COPY
                : (op == 0x01 || op == 0x02 || op == 0x03 || op == 0x04);  // BG_1CYC/COPY, OBJ_RECT/SPRITE
            if (is_zero_l_opcode) {
                if ((w0 & 0x00FFFFFF) != 0) { confirmed_not_s2dex = true; break; }
                if (w1 != 0) { confirmed_s2dex = true; break; } // a real DMA pointer, not just a zeroed-out slot
            }
            // Past a G_LOAD_UCODE the list is some other microcode's.
            if (op == (gbi2 ? 0xDD : 0xAF)) break;
            if (op == (gbi2 ? 0xDE : 0x06)) { // G_DL: most top-level S2DEX/F3D lists are mostly
                               // branches to sub-lists, so follow it to actually
                               // sample real content instead of scanning pointers.
                if (((w0 >> 16) & 0xFF) == 0) scan_stack.push_back(scan_pc + 8);
                scan_pc = w1;
                depth++;
                continue;
            }
            if (op == (gbi2 ? 0xDF : 0xB8)) { // G_ENDDL: back to the caller
                if (scan_stack.empty()) break;
                scan_pc = scan_stack.back();
                scan_stack.pop_back();
                continue;
            }
            scan_pc += 8;
        }
        if (!confirmed_s2dex || confirmed_not_s2dex) {
            current_ucode = MicrocodeType::Auto;
            ucode_type = MicrocodeType::Auto; // correct the persistent guess too
        }
    }

    if (current_ucode == MicrocodeType::Auto) {
        u32 scan_pc = dl_addr;
        for (int i = 0; i < 512; ++i) {
            u32 phys = segment_to_physical(scan_pc);
            if (phys + 8 > rdram_size) break;
            u8 op = rdram[phys];
            u32 w0 = (static_cast<u32>(rdram[phys + 0]) << 24) |
                     (static_cast<u32>(rdram[phys + 1]) << 16) |
                     (static_cast<u32>(rdram[phys + 2]) << 8)  |
                      static_cast<u32>(rdram[phys + 3]);
            // F3DEX2 unique opcodes
            if (op == 0xDF || op == 0xDE || op == 0xDA || op == 0xD9 ||
                op == 0xD8 || op == 0xD7 || op == 0x05 || op == 0x07 ||
                op == 0xE2 || op == 0xE3) {
                current_ucode = MicrocodeType::F3DEX2;
                break;
            }
            // F3DEX 1 unique opcodes
            if (op == 0xB1 || op == 0xB0 || op == 0xAF) {
                current_ucode = MicrocodeType::F3DEX;
                break;
            }
            // G_VTX (0x04) in Fast3D vs F3DEX 1
            if (op == 0x04) {
                // In Fast3D, length is n * 16, so low 4 bits are 0.
                // In F3DEX 1, length is n * 16 - 1, so low 4 bits are 0xF.
                if ((w0 & 0x0F) == 0x0F) {
                    current_ucode = MicrocodeType::F3DEX;
                    break;
                } else if ((w0 & 0x0F) == 0x00 && (w0 & 0xFFFF) > 0) {
                    current_ucode = MicrocodeType::Fast3D;
                    break;
                }
            }
            scan_pc += 8;
        }
        if (current_ucode == MicrocodeType::Auto) {
            current_ucode = MicrocodeType::Fast3D;
        }
    }
    current_ucode_active = current_ucode;

    std::vector<u32> dl_stack;
    // G_DMA_DL lists (F3DDKR/F3DJFG) end after a number of commands instead
    // of at a G_ENDDL: {dl_stack depth inside them, commands left}.
    struct CountedDl { size_t depth; u32 left; };
    std::vector<CountedDl> counted_dl;
    u32 pc = dl_addr;

    const u32 MAX_COMMANDS = 100000;
    u32 cmd_count = 0;

    while (cmd_count++ < MAX_COMMANDS) {
        while (!counted_dl.empty()) {
            const CountedDl& c = counted_dl.back();
            if (c.depth > dl_stack.size()) { // it ended at a G_ENDDL instead
                counted_dl.pop_back();
            } else if (c.depth == dl_stack.size() && c.left == 0) { // all its commands ran
                pc = dl_stack.back();
                dl_stack.pop_back();
                counted_dl.pop_back();
            } else {
                break;
            }
        }
        u32 phys_pc = segment_to_physical(pc);
        if (phys_pc + 8 > rdram_size) break;
        capture_dl_addr = phys_pc;

        u32 w0 = (static_cast<u32>(rdram[phys_pc + 0]) << 24) |
                 (static_cast<u32>(rdram[phys_pc + 1]) << 16) |
                 (static_cast<u32>(rdram[phys_pc + 2]) << 8)  |
                  static_cast<u32>(rdram[phys_pc + 3]);

        u32 w1 = (static_cast<u32>(rdram[phys_pc + 4]) << 24) |
                 (static_cast<u32>(rdram[phys_pc + 5]) << 16) |
                 (static_cast<u32>(rdram[phys_pc + 6]) << 8)  |
                  static_cast<u32>(rdram[phys_pc + 7]);

        pc += 8;
        u8 opcode = (w0 >> 24) & 0xFF;
        if (!counted_dl.empty() && counted_dl.back().depth == dl_stack.size()) counted_dl.back().left--;
        if (dl_trace_frame_ >= 0 && g_current_frame >= dl_trace_frame_ && g_current_frame < dl_trace_frame_ + dl_trace_count_)
            std::fprintf(stderr, "DL %d %08x: %08x %08x\n", g_current_frame, phys_pc, w0, w1);
        if (opcode == 0xE9) full_sync_seen_ = true;

        if (is_s2dex_ucode(current_ucode) &&
            execute_s2dex_command(opcode, w0, w1, current_ucode, pc, dl_stack, rdram, rdram_size)) {
            draw_state_dirty_ = true;
            continue;
        }
        if (!keeps_draw_state(opcode)) draw_state_dirty_ = true;
        if (cbfd_ && current_ucode == MicrocodeType::F3DEX2 && execute_cbfd_command(opcode, w0, w1, rdram, rdram_size))
            continue;

        switch (opcode) {
            case 0x00: // G_SPNOOP
                break;

            case 0x01: {
                if (current_ucode == MicrocodeType::F3DEX2) {
                    execute_vtx(w0, w1, current_ucode, rdram, rdram_size);
                } else if (current_ucode == MicrocodeType::F3DDKR || current_ucode == MicrocodeType::F3DJFG) {
                    dkr_dma_matrix(w0, w1, current_ucode, rdram, rdram_size);
                } else {
                    execute_mtx(w0, w1, current_ucode, rdram, rdram_size);
                }
                break;
            }

            case 0xDA: { // G_MTX (F3DEX2)
                execute_mtx(w0, w1, current_ucode, rdram, rdram_size);
                break;
            }

            case 0x02: {
                if (current_ucode == MicrocodeType::F3DEX2) {
                    // G_MODIFYVTX (F3DEX2)
                    u32 vtx_idx = (w0 & 0xFFFF) / 2;
                    u8 where = (w0 >> 16) & 0xFF;
                    if (vtx_idx < vertex_cache.size()) {
                        Vertex& v = vertex_cache[vtx_idx];
                        switch (where) {
                            case 0x10: // G_MWO_POINT_RGBA
                                v.r = (w1 >> 24) & 0xFF;
                                v.g = (w1 >> 16) & 0xFF;
                                v.b = (w1 >> 8) & 0xFF;
                                v.a = w1 & 0xFF;
                                break;
                            case 0x14: // G_MWO_POINT_ST
                                v.u = (static_cast<s16>(w1 >> 16) / 32.0f) * texture_scale_s;
                                v.v = (static_cast<s16>(w1 & 0xFFFF) / 32.0f) * texture_scale_t;
                                break;
                            case 0x18: // G_MWO_POINT_XYSCREEN
                                v.sx = static_cast<s16>(w1 >> 16) / 4.0f;
                                v.sy = static_cast<s16>(w1 & 0xFFFF) / 4.0f;
                                break;
                            case 0x1C: // G_MWO_POINT_ZSCREEN
                                v.sz = std::clamp((static_cast<s16>(w1 >> 16)) / 1024.0f, 0.0f, 1.0f);
                                break;
                        }
                    }
                } else {
                    // G_POPMTX (Fast3D older)
                    if (modelview_stack.size() > 1) {
                        modelview_stack.pop_back();
                        combined_matrix_dirty = true;
                    }
                }
                break;
            }

            case 0xD8: { // G_POPMTX (F3DEX / F3DEX2)
                u32 pop_count = 1;
                if (current_ucode == MicrocodeType::F3DEX2 && w1 >= 64) {
                    pop_count = w1 / 64;
                }
                while (pop_count-- > 0 && modelview_stack.size() > 1) {
                    modelview_stack.pop_back();
                }
                combined_matrix_dirty = true;
                break;
            }

            case 0xBD: { // G_POPMTX (Fast3D) or G_MOVEWORD (F3DGOLDEN)
                if (current_ucode == MicrocodeType::F3DGOLDEN) {
                    execute_moveword(w0, w1, current_ucode);
                    break;
                }
                if (modelview_stack.size() > 1) {
                    modelview_stack.pop_back();
                    combined_matrix_dirty = true;
                }
                break;
            }

            case 0x03: { // G_CULLDL in F3DEX2, G_MOVEMEM in Fast3D
                if (current_ucode == MicrocodeType::F3DEX2) {
                    u32 vstart = (w0 & 0xFFFF) / 2;
                    u32 vend = (w1 & 0xFFFF) / 2;
                    if (vend < vertex_cache.size() && vstart <= vend) {
                        bool all_left = true, all_right = true, all_bottom = true, all_top = true, all_far = true, all_near = true;
                        for (u32 i = vstart; i <= vend; ++i) {
                            const auto& v = vertex_cache[i];
                            if (v.x >= -v.w) all_left = false;
                            if (v.x <= v.w) all_right = false;
                            if (v.y >= -v.w) all_bottom = false;
                            if (v.y <= v.w) all_top = false;
                            if (v.z <= v.w) all_far = false;
                            if (v.w >= 0.1f) all_near = false;
                        }
                        if (all_left || all_right || all_top || all_bottom || all_far || all_near) {
                            if (!dl_stack.empty()) {
                                pc = dl_stack.back();
                                dl_stack.pop_back();
                            } else {
                                finish_task(mi);
                                return;
                            }
                        }
                    }
                    break;
                }
                [[fallthrough]];
            }
            case 0xDC: { // G_MOVEMEM (F3DEX / F3DEX2)
                u32 src_addr = segment_to_physical(w1);
                bool is_viewport = false;
                bool is_light = false;
                u32 light_n = 0;

                if (current_ucode == MicrocodeType::F3DEX2) {
                    u8 idx = w0 & 0xFF;
                    is_viewport = (idx == 8);
                    if (idx == 14) { // G_MV_MATRIX (gSPForceMatrix): the whole model-view-projection
                        if (src_addr + 64 <= rdram_size) {
                            for (int i = 0; i < 4; ++i) {
                                for (int j = 0; j < 4; ++j) {
                                    const u32 k = (i * 4 + j) * 2;
                                    const s16 ip = static_cast<s16>((rdram[src_addr + k] << 8) | rdram[src_addr + k + 1]);
                                    const u16 fp = static_cast<u16>((rdram[src_addr + 32 + k] << 8) | rdram[src_addr + 33 + k]);
                                    combined_matrix.m[i][j] = ip + fp / 65536.0f;
                                }
                            }
                            // Used as it is until the next G_MTX/G_POPMTX.
                            combined_matrix_dirty = false;
                        }
                    }
                    if (idx == 10) { // G_MV_LIGHT
                        u32 ofs = ((w0 >> 8) & 0xFF) * 8;
                        if (ofs == 0) { // G_MVO_LOOKATX
                            if (src_addr + 16 <= rdram_size) {
                                s8 dx = static_cast<s8>(rdram[src_addr + 8]);
                                s8 dy = static_cast<s8>(rdram[src_addr + 9]);
                                s8 dz = static_cast<s8>(rdram[src_addr + 10]);
                                lookat_x = {rdram[src_addr + 0], rdram[src_addr + 1], rdram[src_addr + 2],
                                            dx / 127.0f, dy / 127.0f, dz / 127.0f};
                                lookat_set = true;
                            }
                        } else if (ofs == 24) { // G_MVO_LOOKATY
                            if (src_addr + 16 <= rdram_size) {
                                s8 dx = static_cast<s8>(rdram[src_addr + 8]);
                                s8 dy = static_cast<s8>(rdram[src_addr + 9]);
                                s8 dz = static_cast<s8>(rdram[src_addr + 10]);
                                lookat_y = {rdram[src_addr + 0], rdram[src_addr + 1], rdram[src_addr + 2],
                                            dx / 127.0f, dy / 127.0f, dz / 127.0f};
                                lookat_set = true;
                            }
                        } else if (ofs >= 48) {
                            is_light = true;
                            light_n = (ofs / 24) - 1;
                        }
                    }
                } else {
                    u8 type_idx = (w0 >> 16) & 0xFF;
                    if (type_idx == 0) type_idx = w0 & 0xFF;

                    is_viewport = (type_idx == 0x80) || (type_idx == 8);
                    // G_MV_MATRIX_1..4 (gSPForceMatrix): bytes 0-15, 16-31,
                    // 32-47 and 48-63 of the whole model-view-projection,
                    // used as it is until the next G_MTX/G_POPMTX.
                    const int piece = type_idx == 0x9E ? 0 : type_idx == 0x98 ? 1 : type_idx == 0x9A ? 2
                                    : type_idx == 0x9C ? 3 : -1;
                    if (piece >= 0 && src_addr + 16 <= rdram_size) {
                        std::copy_n(rdram + src_addr, 16, forced_mtx.begin() + piece * 16);
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 4; ++j) {
                                const u32 k = (i * 4 + j) * 2;
                                const s16 ip = static_cast<s16>((forced_mtx[k] << 8) | forced_mtx[k + 1]);
                                const u16 fp = static_cast<u16>((forced_mtx[32 + k] << 8) | forced_mtx[33 + k]);
                                combined_matrix.m[i][j] = ip + fp / 65536.0f;
                            }
                        }
                        combined_matrix_dirty = false;
                    }
                    if (type_idx == 0x84) { // LookAtX
                        if (src_addr + 16 <= rdram_size) {
                            s8 dx = static_cast<s8>(rdram[src_addr + 8]);
                            s8 dy = static_cast<s8>(rdram[src_addr + 9]);
                            s8 dz = static_cast<s8>(rdram[src_addr + 10]);
                            lookat_x = {rdram[src_addr + 0], rdram[src_addr + 1], rdram[src_addr + 2],
                                        dx / 127.0f, dy / 127.0f, dz / 127.0f};
                            lookat_set = true;
                        }
                    } else if (type_idx == 0x82) { // LookAtY
                        if (src_addr + 16 <= rdram_size) {
                            s8 dx = static_cast<s8>(rdram[src_addr + 8]);
                            s8 dy = static_cast<s8>(rdram[src_addr + 9]);
                            s8 dz = static_cast<s8>(rdram[src_addr + 10]);
                            lookat_y = {rdram[src_addr + 0], rdram[src_addr + 1], rdram[src_addr + 2],
                                        dx / 127.0f, dy / 127.0f, dz / 127.0f};
                            lookat_set = true;
                        }
                    } else if (type_idx >= 0x86 && type_idx <= 0x94) {
                        is_light = true;
                        light_n = (type_idx - 0x86) / 2 + 1;
                    } else if (type_idx == 10) {
                        u32 ofs = ((w0 >> 8) & 0xFF) * 8;
                        if (ofs == 0) {
                            if (src_addr + 16 <= rdram_size) {
                                s8 dx = static_cast<s8>(rdram[src_addr + 8]);
                                s8 dy = static_cast<s8>(rdram[src_addr + 9]);
                                s8 dz = static_cast<s8>(rdram[src_addr + 10]);
                                lookat_x = {rdram[src_addr + 0], rdram[src_addr + 1], rdram[src_addr + 2],
                                            dx / 127.0f, dy / 127.0f, dz / 127.0f};
                                lookat_set = true;
                            }
                        } else if (ofs == 24) {
                            if (src_addr + 16 <= rdram_size) {
                                s8 dx = static_cast<s8>(rdram[src_addr + 8]);
                                s8 dy = static_cast<s8>(rdram[src_addr + 9]);
                                s8 dz = static_cast<s8>(rdram[src_addr + 10]);
                                lookat_y = {rdram[src_addr + 0], rdram[src_addr + 1], rdram[src_addr + 2],
                                            dx / 127.0f, dy / 127.0f, dz / 127.0f};
                                lookat_set = true;
                            }
                        } else if (ofs >= 48) {
                            is_light = true;
                            light_n = (ofs / 24) - 1;
                        }
                    }
                }

                if (is_viewport) {
                    if (src_addr + 16 <= rdram_size) {
                        s16 vscale_x = static_cast<s16>((rdram[src_addr + 0] << 8) | rdram[src_addr + 1]);
                        s16 vscale_y = static_cast<s16>((rdram[src_addr + 2] << 8) | rdram[src_addr + 3]);
                        s16 vscale_z = static_cast<s16>((rdram[src_addr + 4] << 8) | rdram[src_addr + 5]);
                        s16 vtrans_x = static_cast<s16>((rdram[src_addr + 8] << 8) | rdram[src_addr + 9]);
                        s16 vtrans_y = static_cast<s16>((rdram[src_addr + 10] << 8) | rdram[src_addr + 11]);
                        s16 vtrans_z = static_cast<s16>((rdram[src_addr + 12] << 8) | rdram[src_addr + 13]);

                        // X and Y have 2 fraction bits; Z has none (G_MAXZ / 2
                        // each maps the clip range onto the RDP's 0..1023).
                        vp_scale_x = vscale_x / 4.0f;
                        vp_scale_y = std::abs(vscale_y / 4.0f);
                        vp_scale_z = (vscale_z != 0) ? static_cast<f32>(vscale_z) : 511.5f;
                        vp_trans_x = vtrans_x / 4.0f;
                        vp_trans_y = vtrans_y / 4.0f;
                        vp_trans_z = (vtrans_z != 0) ? static_cast<f32>(vtrans_z) : 511.5f;
                    }
                } else if (is_light) {
                    if (src_addr + 16 <= rdram_size) {
                        u8 r = rdram[src_addr + 0];
                        u8 g = rdram[src_addr + 1];
                        u8 b = rdram[src_addr + 2];
                        s8 dx = static_cast<s8>(rdram[src_addr + 8]);
                        s8 dy = static_cast<s8>(rdram[src_addr + 9]);
                        s8 dz = static_cast<s8>(rdram[src_addr + 10]);

                        bool is_ambient = (num_lights == 0) || (light_n > num_lights) || (dx == 0 && dy == 0 && dz == 0);
                        if (is_ambient) {
                            ambient_light = {r, g, b, 0, 0, 0};
                        } else {
                            f32 len = std::sqrt(static_cast<f32>(dx * dx + dy * dy + dz * dz));
                            f32 ndx = (len > 0.0f) ? (dx / len) : 0.0f;
                            f32 ndy = (len > 0.0f) ? (dy / len) : 0.0f;
                            f32 ndz = (len > 0.0f) ? (dz / len) : 1.0f;
                            u32 slot = (light_n >= 1) ? (light_n - 1) : 0;
                            if (dir_lights.size() <= slot) {
                                dir_lights.resize(slot + 1);
                            }
                            dir_lights[slot] = {r, g, b, ndx, ndy, ndz};
                        }
                    }
                }
                break;
            }

            case 0x04: {
                if (current_ucode == MicrocodeType::F3DEX2) {
                    // G_BRANCH_Z in F3DEX2
                    u32 vtx = (w0 & 0xFFF) / 2;
                    u32 zval = w1;
                    if (vtx < vertex_cache.size()) {
                        f32 v_z = vertex_cache[vtx].sz * 1023.0f;
                        f32 thresh = (zval > 1024) ? (static_cast<f32>(zval) / 65536.0f) : static_cast<f32>(zval);
                        if (v_z <= thresh) {
                            if (rdp_half1 != 0) {
                                pc = rdp_half1;
                            }
                        }
                    }
                    break;
                }
                execute_vtx(w0, w1, current_ucode, rdram, rdram_size);
                break;
            }

            case 0xB0: { // G_BRANCH_Z (F3DEX)
                u32 vtx = (w0 & 0xFFF) / 2;
                u32 zval = w1;
                if (vtx < vertex_cache.size()) {
                    f32 v_z = vertex_cache[vtx].sz * 1023.0f;
                    f32 thresh = (zval > 1024) ? (static_cast<f32>(zval) / 65536.0f) : static_cast<f32>(zval);
                    if (v_z <= thresh) {
                        if (rdp_half1 != 0) {
                            pc = rdp_half1;
                        }
                    }
                }
                break;
            }

            case 0x05: { // G_TRI1 (F3DEX2) / G_DMA_TRI (F3DDKR/F3DJFG)
                if (current_ucode == MicrocodeType::F3DDKR || current_ucode == MicrocodeType::F3DJFG) {
                    dkr_dma_triangles(w0, w1, rdram, rdram_size);
                    break;
                }
                u32 v0 = ((w0 >> 16) & 0xFF) / 2;
                u32 v1 = ((w0 >> 8) & 0xFF) / 2;
                u32 v2 = (w0 & 0xFF) / 2;
                if (v0 < vertex_cache.size() && v1 < vertex_cache.size() && v2 < vertex_cache.size()) {
                    emit_triangle(v0, v1, v2, rdram, rdram_size);
                }
                break;
            }

            case 0xBF: { // G_TRI1 (Fast3D or F3DEX; F3DWRUS: slots x5) / G_DMA_OFFSETS (F3DDKR/F3DJFG)
                if (current_ucode == MicrocodeType::F3DDKR || current_ucode == MicrocodeType::F3DJFG) {
                    dkr_mtx_offset = w0 & 0x00FFFFFF;
                    dkr_vtx_offset = w1 & 0x00FFFFFF;
                    break;
                }
                u32 div = current_ucode == MicrocodeType::F3DWRUS ? 5 : is_f3d_family(current_ucode) ? 10 : 2;
                u32 v0 = ((w1 >> 16) & 0xFF) / div;
                u32 v1 = ((w1 >> 8) & 0xFF) / div;
                u32 v2 = (w1 & 0xFF) / div;
                if (v0 < vertex_cache.size() && v1 < vertex_cache.size() && v2 < vertex_cache.size()) {
                    emit_triangle(v0, v1, v2, rdram, rdram_size);
                }
                break;
            }

            case 0xB1: { // G_TRI4 (F3DGOLDEN, F3DPD) / G_TRI2 (Fast3D or F3DEX)
                if (current_ucode == MicrocodeType::F3DGOLDEN || current_ucode == MicrocodeType::F3DPD) {
                    while (w1 != 0) {
                        u32 v0 = w1 & 0x0F;
                        w1 >>= 4;
                        u32 v1 = w1 & 0x0F;
                        w1 >>= 4;
                        u32 v2 = w0 & 0x0F;
                        w0 >>= 4;
                        if (v0 < vertex_cache.size() && v1 < vertex_cache.size() && v2 < vertex_cache.size()) {
                            emit_triangle(v0, v1, v2, rdram, rdram_size);
                        }
                    }
                    break;
                }
                u32 div = current_ucode == MicrocodeType::F3DWRUS ? 5 : (current_ucode == MicrocodeType::Fast3D) ? 10 : 2;
                u32 v0 = ((w0 >> 16) & 0xFF) / div;
                u32 v1 = ((w0 >> 8) & 0xFF) / div;
                u32 v2 = (w0 & 0xFF) / div;
                u32 v3 = ((w1 >> 16) & 0xFF) / div;
                u32 v4 = ((w1 >> 8) & 0xFF) / div;
                u32 v5 = (w1 & 0xFF) / div;
                if (v0 < vertex_cache.size() && v1 < vertex_cache.size() && v2 < vertex_cache.size()) {
                    emit_triangle(v0, v1, v2, rdram, rdram_size);
                }
                if (v3 < vertex_cache.size() && v4 < vertex_cache.size() && v5 < vertex_cache.size()) {
                    emit_triangle(v3, v4, v5, rdram, rdram_size);
                }
                break;
            }

            case 0xB2: { // G_MODIFYVTX (F3DEX) / G_RDPHALF_CONT (Fast3D)
                if (is_f3d_family(current_ucode)) {
                    rdp_half2 = w1;
                    break;
                }
                u32 vtx_idx = (w0 & 0xFFFF) / 2;
                u8 where = (w0 >> 16) & 0xFF;
                if (vtx_idx < vertex_cache.size()) {
                    Vertex& v = vertex_cache[vtx_idx];
                    switch (where) {
                        case 0x10: // G_MWO_POINT_RGBA
                            v.r = (w1 >> 24) & 0xFF;
                            v.g = (w1 >> 16) & 0xFF;
                            v.b = (w1 >> 8) & 0xFF;
                            v.a = w1 & 0xFF;
                            break;
                        case 0x14: // G_MWO_POINT_ST
                            v.u = (static_cast<s16>(w1 >> 16) / 32.0f) * texture_scale_s;
                            v.v = (static_cast<s16>(w1 & 0xFFFF) / 32.0f) * texture_scale_t;
                            break;
                        case 0x18: // G_MWO_POINT_XYSCREEN
                            v.sx = static_cast<s16>(w1 >> 16) / 4.0f;
                            v.sy = static_cast<s16>(w1 & 0xFFFF) / 4.0f;
                            break;
                        case 0x1C: // G_MWO_POINT_ZSCREEN
                            v.sz = std::clamp((static_cast<s16>(w1 >> 16)) / 1024.0f, 0.0f, 1.0f);
                            break;
                    }
                }
                break;
            }

            case 0xBE: { // G_CULLDL (Fast3D / F3DEX)
                u32 vstart = 0, vend = 0;
                if (is_f3d_family(current_ucode)) {
                    vstart = ((w0 & 0xFFFF) / 40) & 0x0F;
                    vend = (((w1 & 0xFFFF) / 40) > 0) ? (((w1 & 0xFFFF) / 40) - 1) & 0x0F : 0;
                } else {
                    vstart = (w0 & 0xFFFF) / 2;
                    vend = (w1 & 0xFFFF) / 2;
                }
                if (vend >= vstart && vend < vertex_cache.size()) {
                    u32 clip_all = 0x3F;
                    for (u32 i = vstart; i <= vend; ++i) {
                        clip_all &= vertex_cache[i].clip_flags;
                    }
                    if (clip_all != 0) {
                        if (!dl_stack.empty()) {
                            pc = dl_stack.back();
                            dl_stack.pop_back();
                        } else {
                            finish_task(mi);
                            return;
                        }
                    }
                }
                break;
            }

            case 0xB5: { // G_LINE3D (Fast3D / F3DEX) / G_QUAD (F3DWRUS)
                if (current_ucode == MicrocodeType::F3DWRUS) {
                    const u32 q0 = ((w1 >> 24) & 0xFF) / 5, q1 = ((w1 >> 16) & 0xFF) / 5;
                    const u32 q2 = ((w1 >> 8) & 0xFF) / 5, q3 = (w1 & 0xFF) / 5;
                    if (q0 < vertex_cache.size() && q1 < vertex_cache.size() && q2 < vertex_cache.size() &&
                        q3 < vertex_cache.size()) {
                        emit_triangle(q0, q1, q2, rdram, rdram_size);
                        emit_triangle(q0, q2, q3, rdram, rdram_size);
                    }
                    break;
                }
                u32 div = (is_f3d_family(current_ucode)) ? 10 : 2;
                u32 v0 = ((w1 >> 16) & 0xFF) / div;
                u32 v1 = ((w1 >> 8) & 0xFF) / div;
                if (v0 < vertex_cache.size() && v1 < vertex_cache.size()) {
                    clip_and_rasterize_line(vertex_cache[v0], vertex_cache[v1], rdram, rdram_size);
                }
                break;
            }

            case 0x06: { // G_DL in Fast3D & F3DEX, G_TRI2 in F3DEX2
                if (current_ucode != MicrocodeType::F3DEX2) {
                    u32 target = w1;
                    u8 flag = (w0 >> 16) & 0xFF;
                    if (flag == 0) {
                        dl_stack.push_back(pc);
                    }
                    pc = target;
                } else {
                    u32 v0 = ((w0 >> 16) & 0xFF) / 2;
                    u32 v1 = ((w0 >> 8) & 0xFF) / 2;
                    u32 v2 = (w0 & 0xFF) / 2;
                    u32 v3 = ((w1 >> 16) & 0xFF) / 2;
                    u32 v4 = ((w1 >> 8) & 0xFF) / 2;
                    u32 v5 = (w1 & 0xFF) / 2;
                    if (v0 < vertex_cache.size() && v1 < vertex_cache.size() && v2 < vertex_cache.size()) {
                        emit_triangle(v0, v1, v2, rdram, rdram_size);
                    }
                    if (v3 < vertex_cache.size() && v4 < vertex_cache.size() && v5 < vertex_cache.size()) {
                        emit_triangle(v3, v4, v5, rdram, rdram_size);
                    }
                }
                break;
            }

            case 0x07: { // G_QUAD (F3DEX2) / vertex color table (F3DPD) / G_DMA_DL (F3DDKR/F3DJFG)
                if (current_ucode == MicrocodeType::F3DDKR || current_ucode == MicrocodeType::F3DJFG) {
                    // Runs (w0 >> 16) & 0xFF commands at w1, then comes back.
                    dl_stack.push_back(pc);
                    counted_dl.push_back({dl_stack.size(), (w0 >> 16) & 0xFF});
                    pc = w1;
                    break;
                }
                if (current_ucode == MicrocodeType::F3DPD) {
                    vtx_color_base = segment_to_physical(w1);
                    break;
                }
                u32 v0 = ((w0 >> 16) & 0xFF) / 2;
                u32 v1 = ((w0 >> 8) & 0xFF) / 2;
                u32 v2 = (w0 & 0xFF) / 2;
                u32 v3 = ((w1 >> 16) & 0xFF) / 2;
                u32 v4 = ((w1 >> 8) & 0xFF) / 2;
                u32 v5 = (w1 & 0xFF) / 2;
                if (v0 < vertex_cache.size() && v1 < vertex_cache.size() && v2 < vertex_cache.size()) {
                    emit_triangle(v0, v1, v2, rdram, rdram_size);
                }
                if (v3 < vertex_cache.size() && v4 < vertex_cache.size() && v5 < vertex_cache.size()) {
                    emit_triangle(v3, v4, v5, rdram, rdram_size);
                }
                break;
            }

            case 0x08: { // G_LINE3D (F3DEX2)
                u32 v0 = ((w0 >> 16) & 0xFF) / 2;
                u32 v1 = ((w0 >> 8) & 0xFF) / 2;
                if (v0 < vertex_cache.size() && v1 < vertex_cache.size()) {
                    clip_and_rasterize_line(vertex_cache[v0], vertex_cache[v1], rdram, rdram_size);
                }
                break;
            }

            case 0xDE: { // G_DL (F3DEX / F3DEX2)
                u32 target = w1;
                u8 flag = (w0 >> 16) & 0xFF;
                if (flag == 0) {
                    dl_stack.push_back(pc);
                }
                pc = target;
                break;
            }

            case 0xAF: // G_LOAD_UCODE (F3DEX)
            case 0xDD: { // G_LOAD_UCODE (F3DEX2)
                auto check_header = [&](u32 addr) -> bool {
                    u32 ucode_phys = segment_to_physical(addr);
                    if (ucode_phys + 256 <= rdram_size) {
                        std::string header(reinterpret_cast<const char*>(&rdram[ucode_phys]), 256);
                        // See the matching comment in rsp.cpp's banner detection: the
                        // GBI-1/GBI-2 check must stay local to *this* banner match --
                        // searching the whole window for "fifo 2" independently can
                        // pick up an unrelated, adjacent ucode's banner.
                        size_t s2dex_anchor = header.find("ucode S2DEX");
                        if (s2dex_anchor != std::string::npos) {
                            std::string local = header.substr(s2dex_anchor, 40);
                            bool is_gbi2 = local.find("S2DEX2") != std::string::npos || local.find("fifo 2") != std::string::npos;
                            current_ucode = is_gbi2 ? MicrocodeType::S2DEX2 : MicrocodeType::S2DEX;
                            current_ucode_active = current_ucode;
                            return true;
                        } else if (header.find("F3DEX 2") != std::string::npos ||
                            header.find("F3DEX2") != std::string::npos ||
                            header.find("fifo 2") != std::string::npos ||
                            header.find("F3DZEX") != std::string::npos) {
                            current_ucode = MicrocodeType::F3DEX2;
                            current_ucode_active = current_ucode;
                            return true;
                        } else if (header.find("F3DEX") != std::string::npos ||
                                   header.find("F3DLX") != std::string::npos) {
                            current_ucode = MicrocodeType::F3DEX;
                            current_ucode_active = current_ucode;
                            return true;
                        } else if (header.find("Fast3D") != std::string::npos) {
                            current_ucode = MicrocodeType::Fast3D;
                            current_ucode_active = current_ucode;
                            return true;
                        }
                    }
                    return false;
                };
                // The banner is in the data segment (RDPHALF_1), often past
                // the first 256 bytes: look through all of it first.
                if (rdp_half1 != 0) {
                    const u32 data_phys = segment_to_physical(rdp_half1);
                    const size_t dsize = std::min<size_t>((w0 & 0xFFFF) + 1, 0x800);
                    if (data_phys < rdram_size) {
                        const MicrocodeType t = identify_ucode_banner(
                            rdram + data_phys, std::min(dsize, rdram_size - data_phys));
                        if (t != MicrocodeType::Auto) {
                            current_ucode = current_ucode_active = t;
                            no_near_clip_ = ucode_banner_is_non(rdram + data_phys, std::min(dsize, rdram_size - data_phys));
                            std::fill(std::begin(s2d_genstat), std::end(s2d_genstat), 0u);
                            break;
                        }
                    }
                }
                if (!check_header(w1) && rdp_half1 != 0) {
                    check_header(rdp_half1);
                }
                break;
            }

            case 0xB8: // G_ENDDL (Fast3D)
            case 0xDF: { // G_ENDDL (F3DEX / F3DEX2)
                if (!dl_stack.empty()) {
                    pc = dl_stack.back();
                    dl_stack.pop_back();
                } else {
                    // Finished entire display list!
                    std::cout << "[RDP Stats] Triangles called: " << stat_tri_called
                              << " | incount: [0]=" << stat_incount[0] << " [1]=" << stat_incount[1] 
                              << " [2]=" << stat_incount[2] << " [3]=" << stat_incount[3]
                              << " | rasterize called: " << stat_rast_called
                              << " | cull_back: " << stat_cull_back << " cull_front: " << stat_cull_front
                              << " | scissor_rej: " << stat_scissor_reject
                              << " | pixels: drawn=" << stat_pixels.drawn << " z_fail=" << stat_pixels.z_fail
                              << " a_fail=" << stat_pixels.a_fail << "\n";
                    finish_task(mi);
                    return;
                }
                break;
            }

            case 0xBC: // G_MOVEWORD (Fast3D / F3DEX)
            case 0xDB: { // G_MOVEWORD (F3DEX2)
                if ((current_ucode == MicrocodeType::F3DDKR || current_ucode == MicrocodeType::F3DJFG) && opcode == 0xBC) {
                    // Two indices of their own: billboarding on/off, and which
                    // model-view slot is current.
                    if ((w0 & 0xFF) == 0x02) { dkr_billboard = (w1 & 1) != 0; break; }
                    if ((w0 & 0xFF) == 0x0A) { dkr_select_matrix((w1 >> 6) & 3); break; }
                }
                execute_moveword(w0, w1, current_ucode);
                break;
            }

            case 0xBB: // G_TEXTURE (Fast3D)
            case 0xD7: { // G_TEXTURE (F3DEX / F3DEX2)
                active_tile = (w0 >> 8) & 0x7;
                tex_max_level = (w0 >> 11) & 0x7;
                texture_scale_s = ((w1 >> 16) & 0xFFFF) / 65536.0f;
                texture_scale_t = (w1 & 0xFFFF) / 65536.0f;
                if (texture_scale_s == 0.0f) texture_scale_s = 1.0f;
                if (texture_scale_t == 0.0f) texture_scale_t = 1.0f;
                texture_enabled = ((w0 & 0xFF) != 0);
                break;
            }

            case 0xB7: // G_SETGEOMETRYMODE
                geometry_mode |= w1;
                break;

            case 0xB6: // G_CLEARGEOMETRYMODE
                geometry_mode &= ~w1;
                break;

            case 0xD9: // G_GEOMETRYMODE (F3DEX2)
                geometry_mode = (geometry_mode & (w0 & 0x00FFFFFF)) | (w1 & 0x00FFFFFF);
                break;

            case 0xB9: { // G_SETOTHERMODE_L (Fast3D)
                u32 shift = (w0 >> 8) & 0xFF;
                u32 len = w0 & 0xFF;
                u32 mask = (len >= 32) ? 0xFFFFFFFFU : (((1ULL << len) - 1) << shift);
                other_mode_l = (other_mode_l & ~mask) | (w1 & mask);
                break;
            }
            case 0xE2: { // G_SETOTHERMODE_L (F3DEX2)
                u32 len = (w0 & 0xFF) + 1;
                u32 shift = 32 - ((w0 >> 8) & 0xFF) - len;
                u32 mask = (len >= 32) ? 0xFFFFFFFFU : (((1ULL << len) - 1) << shift);
                other_mode_l = (other_mode_l & ~mask) | (w1 & mask);
                break;
            }

            case 0xBA: { // G_SETOTHERMODE_H (Fast3D)
                u32 shift = (w0 >> 8) & 0xFF;
                u32 len = w0 & 0xFF;
                u32 mask = (len >= 32) ? 0xFFFFFFFFU : (((1ULL << len) - 1) << shift);
                other_mode_h = (other_mode_h & ~mask) | (w1 & mask);
                break;
            }
            case 0xE3: { // G_SETOTHERMODE_H (F3DEX2)
                u32 len = (w0 & 0xFF) + 1;
                u32 shift = 32 - ((w0 >> 8) & 0xFF) - len;
                u32 mask = (len >= 32) ? 0xFFFFFFFFU : (((1ULL << len) - 1) << shift);
                other_mode_h = (other_mode_h & ~mask) | (w1 & mask);
                break;
            }

            case 0xC0: // G_NOOP
            case 0xE6: // G_RDPLOADSYNC
            case 0xE7: // G_RDPPIPESYNC
            case 0xE8: // G_RDPTILESYNC
            case 0xE9: // G_RDPFULLSYNC
                break;

            case 0xB4: // G_RDPHALF_1 (Fast3D/F3DEX)
            case 0xE1: // G_RDPHALF_1 (F3DEX2)
                rdp_half1 = w1;
                break;

            case 0xB3: // G_RDPHALF_2 (Fast3D/F3DEX)
            case 0xF1: // G_RDPHALF_2 (F3DEX2)
                rdp_half2 = w1;
                break;

            case 0xE4: // G_TEXRECT
            case 0xE5: { // G_TEXRECTFLIP
                bool flip = (opcode == 0xE5);
                // Edges in quarter pixels.
                u32 lrx = (w0 >> 12) & 0xFFF;
                u32 lry = w0 & 0xFFF;
                u32 tile_idx = (w1 >> 24) & 0x7;
                u32 ulx = (w1 >> 12) & 0xFFF;
                u32 uly = w1 & 0xFFF;

                if (ulx > lrx) std::swap(ulx, lrx);
                if (uly > lry) std::swap(uly, lry);

                f32 s = 0.0f, t = 0.0f, dsdx = 1.0f, dtdy = 1.0f;

                phys_pc = segment_to_physical(pc);
                if (phys_pc + 8 <= rdram_size) {
                    u32 nw0 = (static_cast<u32>(rdram[phys_pc + 0]) << 24) |
                              (static_cast<u32>(rdram[phys_pc + 1]) << 16) |
                              (static_cast<u32>(rdram[phys_pc + 2]) << 8)  |
                               static_cast<u32>(rdram[phys_pc + 3]);
                    u32 nw1 = (static_cast<u32>(rdram[phys_pc + 4]) << 24) |
                              (static_cast<u32>(rdram[phys_pc + 5]) << 16) |
                              (static_cast<u32>(rdram[phys_pc + 6]) << 8)  |
                               static_cast<u32>(rdram[phys_pc + 7]);

                    u8 next_op = (nw0 >> 24) & 0xFF;
                    if (next_op == 0xB4 || next_op == 0xB3 || next_op == 0xB2 ||
                        next_op == 0xE1 || next_op == 0xF1) {
                        s = static_cast<s16>((nw1 >> 16) & 0xFFFF) / 32.0f;
                        t = static_cast<s16>(nw1 & 0xFFFF) / 32.0f;
                        pc += 8;

                        u32 phys_pc2 = segment_to_physical(pc);
                        if (phys_pc2 + 8 <= rdram_size) {
                            u32 nnw0 = (static_cast<u32>(rdram[phys_pc2 + 0]) << 24) |
                                       (static_cast<u32>(rdram[phys_pc2 + 1]) << 16) |
                                       (static_cast<u32>(rdram[phys_pc2 + 2]) << 8)  |
                                        static_cast<u32>(rdram[phys_pc2 + 3]);
                            u32 nnw1 = (static_cast<u32>(rdram[phys_pc2 + 4]) << 24) |
                                       (static_cast<u32>(rdram[phys_pc2 + 5]) << 16) |
                                       (static_cast<u32>(rdram[phys_pc2 + 6]) << 8)  |
                                        static_cast<u32>(rdram[phys_pc2 + 7]);
                            u8 next_op2 = (nnw0 >> 24) & 0xFF;
                            if (next_op2 == 0xB3 || next_op2 == 0xB2 || next_op2 == 0xF1) {
                                dsdx = static_cast<s16>((nnw1 >> 16) & 0xFFFF) / 1024.0f;
                                dtdy = static_cast<s16>(nnw1 & 0xFFFF) / 1024.0f;
                                pc += 8;
                            }
                        }
                    } else {
                        s = static_cast<s16>((nw0 >> 16) & 0xFFFF) / 32.0f;
                        t = static_cast<s16>(nw0 & 0xFFFF) / 32.0f;
                        dsdx = static_cast<s16>((nw1 >> 16) & 0xFFFF) / 1024.0f;
                        dtdy = static_cast<s16>(nw1 & 0xFFFF) / 1024.0f;
                        pc += 8;
                    }
                }
                rasterize_tex_rect(ulx, uly, lrx, lry, tile_idx, s, t, dsdx, dtdy, flip, rdram, rdram_size);
                break;
            }

            case 0xEA: case 0xEB: case 0xEC: case 0xED: case 0xEE: case 0xEF:
            case 0xF0: case 0xF2: case 0xF3: case 0xF4: case 0xF5: case 0xF6: case 0xF7:
            case 0xF8: case 0xF9: case 0xFA: case 0xFB: case 0xFC: case 0xFD: case 0xFE: case 0xFF:
                execute_rdp_op(opcode, w0, w1, rdram, rdram_size, false);
                break;

            default: {
                static std::unordered_set<u8> logged_unknown_opcodes;
                if (logged_unknown_opcodes.insert(opcode).second) {
                    std::cerr << "[RDP] Unknown DL opcode: 0x" << std::hex << (int)opcode 
                              << " at seg PC: 0x" << (pc - 8) << " phys: 0x" << phys_pc 
                              << " w0: 0x" << w0 << " w1: 0x" << w1 
                              << " ucode: " << (int)current_ucode << std::dec << "\n";
                }
                break;
            }
        }
    }

    std::cout << "[RDP Stats] Triangles called: " << stat_tri_called
              << " | incount: [0]=" << stat_incount[0] << " [1]=" << stat_incount[1] 
              << " [2]=" << stat_incount[2] << " [3]=" << stat_incount[3]
              << " | rasterize called: " << stat_rast_called
              << " | cull_back: " << stat_cull_back << " cull_front: " << stat_cull_front
              << " | scissor_rej: " << stat_scissor_reject
              << " | pixels: drawn=" << stat_pixels.drawn << " z_fail=" << stat_pixels.z_fail
              << " a_fail=" << stat_pixels.a_fail << "\n";

    finish_task(mi);
}

// The RDP's own commands (0x24-0x3F, here with the display lists' 0xC0 on
// top): tile and TMEM loads, scissor, colours, modes, images and fill
// rectangles. Display lists pass them through to the RDP unchanged, apart
// from segmented image addresses; `raw` commands come from the RDP command
// buffer (process_rdp_commands()) and carry physical ones.
bool RDP::execute_rdp_op(u8 opcode, u32 w0, u32 w1, u8* rdram, size_t rdram_size, bool raw) {
    auto image_addr = [&](u32 a) { return raw ? (a & 0x00FFFFFF) : segment_to_physical(a); };
    switch (opcode) {
        case 0xED: { // G_SETSCISSOR
            scissor_ulx = ((w0 >> 12) & 0xFFF) / 4;
            scissor_uly = (w0 & 0xFFF) / 4;
            scissor_lrx = ((w1 >> 12) & 0xFFF) / 4;
            scissor_lry = (w1 & 0xFFF) / 4;
            if (scissor_lrx <= scissor_ulx) scissor_lrx = 320;
            if (scissor_lry <= scissor_uly) scissor_lry = 240;
            break;
        }

        case 0xF0: { // G_LOADTLUT
            tmem_dirty = true;
            ++tmem_gen_;
            u32 tile_idx = (w1 >> 24) & 0x7;
            // Entries uls..lrs of row ult of the image. Games normally load
            // from 0,0, but Perfect Dark stores each palette right after its
            // texture and loads it with an offset from the same image address.
            u32 uls = ((w0 >> 12) & 0xFFF) >> 2;
            u32 ult = (w0 & 0xFFF) >> 2;
            u32 lrs = ((w1 >> 12) & 0xFFF) >> 2;
            u32 count = lrs >= uls ? lrs - uls + 1 : 1;
            u32 src = timg_addr + (ult * timg_width + uls) * 2;
            native_before_read(src, static_cast<u64>(src) + count * 2);
            u32 start_word = tiles[tile_idx].tmem;
            // TMEM keeps palettes packed, entry i at 0x800 + 2i
            // (raster::lookup_tlut); the hardware quadruples each entry
            // into its own 64-bit word, so the load at word 256 + n
            // (gDPLoadTLUT_pal16 with palette n / 16) starts at entry n.
            u32 tmem_dest = start_word >= 256 ? 0x800 + (start_word - 256) * 2 : start_word * 8;
            start_word = tmem_dest / 8;
            u32 bytes = count * 2;

            for (u32 b = 0; b < bytes && (tmem_dest + b < tmem.size()) && (src + b < rdram_size); ++b) {
                tmem[tmem_dest + b] = rdram[src + b];
            }
            u32 num_words = (bytes + 7) / 8;
            for (u32 w = 0; w < num_words && (start_word + w < 512); ++w) {
                tmem_word_dxt_zero[start_word + w] = false;
            }
            break;
        }

        case 0xF2: { // G_SETTILESIZE
            u32 tile_idx = (w1 >> 24) & 0x7;
            Tile& t = tiles[tile_idx];
            t.sl = (w0 >> 12) & 0xFFF;
            t.tl = w0 & 0xFFF;
            t.sh = (w1 >> 12) & 0xFFF;
            t.th = w1 & 0xFFF;
            break;
        }

        case 0xF3: { // G_LOADBLOCK
            tmem_dirty = true;
            ++tmem_gen_;
            u32 tile_idx = (w1 >> 24) & 0x7;
            u32 lrs = (w1 >> 12) & 0xFFF;
            u32 dxt = w1 & 0xFFF;

            u32 src_addr = timg_addr;
            u32 words = lrs + 1;
            u32 start_word = tiles[tile_idx].tmem;
            u32 tmem_dest = start_word * 8;
            native_before_read(src_addr, static_cast<u64>(src_addr) + static_cast<u64>(words) * 4);

            if (timg_size == 3) {
                u32 texels = words;
                for (u32 i = 0; i < texels; ++i) {
                    u32 dram_idx = src_addr + i * 4;
                    u32 rg_dest = tmem_dest + i * 2;
                    u32 ba_dest = rg_dest + 0x800;
                    if (dram_idx + 3 < rdram_size && ba_dest + 1 < tmem.size()) {
                        tmem[rg_dest + 0] = rdram[dram_idx + 0];
                        tmem[rg_dest + 1] = rdram[dram_idx + 1];
                        tmem[ba_dest + 0] = rdram[dram_idx + 2];
                        tmem[ba_dest + 1] = rdram[dram_idx + 3];
                    }
                }
                u32 num_words = (texels * 2 + 7) / 8;
                bool is_dxt_zero = (dxt == 0);
                for (u32 w = 0; w < num_words && (start_word + w < 512); ++w) {
                    tmem_word_dxt_zero[start_word + w] = is_dxt_zero;
                }
                for (u32 w = 0; w < num_words && (start_word + 256 + w < 512); ++w) {
                    tmem_word_dxt_zero[start_word + 256 + w] = is_dxt_zero;
                }
            } else {
                u32 bytes;
                switch (timg_size) {
                    case 0: bytes = (words + 1) / 2; break; // 4-bit
                    case 1: bytes = words; break;           // 8-bit
                    case 2: bytes = words * 2; break;       // 16-bit
                    default: bytes = words; break;
                }
                for (u32 b = 0; b < bytes && (tmem_dest + b < tmem.size()) && (src_addr + b < rdram_size); ++b) {
                    tmem[tmem_dest + b] = rdram[src_addr + b];
                }
                u32 num_words = (bytes + 7) / 8;
                bool is_dxt_zero = (dxt == 0);
                for (u32 w = 0; w < num_words && (start_word + w < 512); ++w) {
                    tmem_word_dxt_zero[start_word + w] = is_dxt_zero;
                }
            }
            break;
        }

        case 0xF4: { // G_LOADTILE
            tmem_dirty = true;
            ++tmem_gen_;
            u32 tile_idx = (w1 >> 24) & 0x7;
            u32 uls = (w0 >> 12) & 0xFFF;
            u32 ult = w0 & 0xFFF;
            u32 lrs = (w1 >> 12) & 0xFFF;
            u32 lrt = w1 & 0xFFF;

            u32 start_s = uls / 4;
            u32 start_t = ult / 4;
            u32 end_s = lrs / 4;
            u32 end_t = lrt / 4;

            u32 num_rows = (end_t >= start_t) ? (end_t - start_t + 1) : 0;
            u32 num_texels = (end_s >= start_s) ? (end_s - start_s + 1) : 0;
            {
                // Rows start_t .. end_t of a timg_width-wide image, at most 4 bytes a texel.
                const u64 stride = static_cast<u64>(timg_width) * 4 + 4;
                native_before_read(timg_addr,
                                   static_cast<u64>(timg_addr) + (static_cast<u64>(end_t) + 1) * stride +
                                       (static_cast<u64>(start_s) + num_texels) * 4);
            }

            Tile& tile = tiles[tile_idx];
            u32 tmem_dest = tile.tmem * 8;

            if (timg_size == 3) {
                u32 dram_stride = timg_width * 4;
                u32 tmem_stride = tile.line > 0 ? (tile.line * 8) : (num_texels * 2);

                for (u32 row = 0; row < num_rows; ++row) {
                    u32 dram_row_start = timg_addr + (start_t + row) * dram_stride + (start_s * 4);
                    u32 tmem_row_rg = tmem_dest + row * tmem_stride;
                    u32 tmem_row_ba = tmem_row_rg + 0x800;

                    for (u32 col = 0; col < num_texels; ++col) {
                        u32 dram_idx = dram_row_start + col * 4;
                        u32 rg_idx = tmem_row_rg + col * 2;
                        u32 ba_idx = tmem_row_ba + col * 2;

                        if (dram_idx + 3 < rdram_size && ba_idx + 1 < tmem.size()) {
                            tmem[rg_idx + 0] = rdram[dram_idx + 0];
                            tmem[rg_idx + 1] = rdram[dram_idx + 1];
                            tmem[ba_idx + 0] = rdram[dram_idx + 2];
                            tmem[ba_idx + 1] = rdram[dram_idx + 3];
                        }
                    }
                    u32 rg_w0 = tmem_row_rg / 8;
                    u32 rg_w1 = (tmem_row_rg + num_texels * 2 + 7) / 8;
                    for (u32 w = rg_w0; w < rg_w1 && w < 512; ++w) tmem_word_dxt_zero[w] = false;
                    u32 ba_w0 = tmem_row_ba / 8;
                    u32 ba_w1 = (tmem_row_ba + num_texels * 2 + 7) / 8;
                    for (u32 w = ba_w0; w < ba_w1 && w < 512; ++w) tmem_word_dxt_zero[w] = false;
                }
            } else {
                u32 bpp_shift = (timg_size == 2) ? 1 : 0;
                u32 dram_stride = (timg_size == 0) ? ((timg_width + 1) / 2) : (timg_width << bpp_shift);
                u32 row_bytes = (timg_size == 0) ? ((num_texels + 1) / 2) : (num_texels << bpp_shift);
                u32 tmem_stride = tile.line > 0 ? (tile.line * 8) : row_bytes;

                for (u32 row = 0; row < num_rows; ++row) {
                    u32 dram_row_start = timg_addr + (start_t + row) * dram_stride + 
                                         ((timg_size == 0) ? (start_s / 2) : (start_s << bpp_shift));
                    u32 tmem_row_start = tmem_dest + row * tmem_stride;

                    for (u32 b = 0; b < row_bytes; ++b) {
                        if (tmem_row_start + b < tmem.size() && dram_row_start + b < rdram_size) {
                            tmem[tmem_row_start + b] = rdram[dram_row_start + b];
                        }
                    }
                    u32 w0 = tmem_row_start / 8;
                    u32 w1 = (tmem_row_start + row_bytes + 7) / 8;
                    for (u32 w = w0; w < w1 && w < 512; ++w) tmem_word_dxt_zero[w] = false;
                }
            }
            break;
        }

        case 0xF5: { // G_SETTILE
            u32 tile_idx = (w1 >> 24) & 0x7;
            Tile& t = tiles[tile_idx];
            t.format = (w0 >> 21) & 0x7;
            t.size = (w0 >> 19) & 0x3;
            t.line = (w0 >> 9) & 0x1FF;
            t.tmem = w0 & 0x1FF;
            t.palette = (w1 >> 20) & 0xF;
            t.clamp_t = (w1 >> 19) & 0x1;
            t.mirror_t = (w1 >> 18) & 0x1;
            t.mask_t = (w1 >> 14) & 0xF;
            t.shift_t = (w1 >> 10) & 0xF;
            t.clamp_s = (w1 >> 9) & 0x1;
            t.mirror_s = (w1 >> 8) & 0x1;
            t.mask_s = (w1 >> 4) & 0xF;
            t.shift_s = w1 & 0xF;
            break;
        }

        case 0xF6: { // G_FILLRECT
            u32 lrx = ((w0 >> 12) & 0xFFF) / 4;
            u32 lry = (w0 & 0xFFF) / 4;
            u32 ulx = ((w1 >> 12) & 0xFFF) / 4;
            u32 uly = (w1 & 0xFFF) / 4;
            // Only in FILL (and COPY) mode is this a plain fill with the
            // fill colour. In 1- and 2-cycle mode the rectangle goes
            // through the combiner and blender like a texture rectangle -
            // e.g. OoT's fade overlays (PRIM colour, alpha 0 = no change),
            // which cleared the whole screen to black when taken as fills.
            // (Edges in quarter pixels; texture coordinates all 0.)
            const u32 cycle_type = (other_mode_h >> 20) & 0x3;
            if (cycle_type < 2)
                rasterize_tex_rect((w1 >> 12) & 0xFFF, w1 & 0xFFF, (w0 >> 12) & 0xFFF, w0 & 0xFFF, 0, 0.0f, 0.0f, 0.0f, 0.0f,
                                   false, rdram, rdram_size);
            else rasterize_fill_rect(ulx, uly, lrx, lry, rdram, rdram_size);
            break;
        }

        case 0xEA: // G_SETKEYGB
            key_center_[1] = static_cast<u8>(w1 >> 24);
            key_scale_[1] = static_cast<u8>(w1 >> 16);
            key_center_[2] = static_cast<u8>(w1 >> 8);
            key_scale_[2] = static_cast<u8>(w1);
            break;
        case 0xEB: // G_SETKEYR
            key_center_[0] = static_cast<u8>(w1 >> 8);
            key_scale_[0] = static_cast<u8>(w1);
            break;
        case 0xEC: // G_SETCONVERT (K0..K3 are for YUV textures; K4, K5 are combiner inputs)
            k4_ = static_cast<u16>((w1 >> 9) & 0x1FF);
            k5_ = static_cast<u16>(w1 & 0x1FF);
            break;

        case 0xEE: { // G_SETPRIMDEPTH
            prim_depth = (w1 >> 16) & 0xFFFF;
            prim_dz = w1 & 0xFFFF;
            break;
        }

        case 0xEF: { // G_RDPSETOTHERMODE
            other_mode_h = w0 & 0x00FFFFFF;
            other_mode_l = w1;
            break;
        }

        case 0xF7: // G_SETFILLCOLOR
            fill_color = w1;
            break;

        case 0xF8: // G_SETFOGCOLOR
            fog_color = w1;
            break;

        case 0xF9: // G_SETBLENDCOLOR
            blend_color = w1;
            break;

        case 0xFA: // G_SETPRIMCOLOR (and the minimum LOD level / LOD fraction)
            prim_color = w1;
            prim_min_level = (w0 >> 8) & 0x1F;
            prim_lod_frac = w0 & 0xFF;
            break;

        case 0xFB: // G_SETENVCOLOR
            env_color = w1;
            break;

        case 0xFC: // G_SETCOMBINE
            combine_mode_w0 = w0;
            combine_mode_w1 = w1;
            combine_mode_set = true;
            break;

        case 0xFD: { // G_SETTIMG
            timg_format = (w0 >> 21) & 0x7;
            timg_size = (w0 >> 19) & 0x3;
            timg_width = (w0 & 0xFFF) + 1;
            timg_addr = image_addr(w1);
            break;
        }

        case 0xFE: { // G_SETDEPTHIMAGE
            depth_image_addr = image_addr(w1);
            break;
        }

        case 0xFF: { // G_SETCOLORIMAGE
            flush_native(); // the queue's bands and depth rows assume one colour image
            color_image_format = (w0 >> 21) & 0x7;
            color_image_size = (w0 >> 19) & 0x3;
            color_image_width = (w0 & 0xFFF) + 1;
            color_image_addr = image_addr(w1);
            if (hires_) hires_->unbind();
            break;
        }

        default:
            return false;
    }
    return true;
}

void RDP::rasterize_fill_rect(u32 ulx, u32 uly, u32 lrx, u32 lry, u8* rdram, size_t rdram_size) {
    if (geometry_only_) return;
    flush_native();
    if (color_image_addr != 0 && color_image_addr == depth_image_addr) {
        clear_zbuffer();
    }
    if (color_image_addr >= rdram_size) return;

    u32 fb_w = color_image_width ? color_image_width : 320;
    u32 max_x = std::min({lrx, fb_w, scissor_lrx});
    u32 max_y = std::min({lry, kMaxFbLines, scissor_lry});
    u32 start_x = std::max(ulx, scissor_ulx);
    u32 start_y = std::max(uly, scissor_uly);
    // FILL/COPY rectangles include their lower-right pixel (the SDK macros pass
    // x+w-1); an exclusive loop drops the last column and row.
    u32 cycle_type = (other_mode_h >> 20) & 0x3;
    if (cycle_type == 2 || cycle_type == 3) {
        max_x = std::min({lrx + 1, fb_w, scissor_lrx});
        max_y = std::min({lry + 1, kMaxFbLines, scissor_lry});
    }

    if (start_x >= max_x || start_y >= max_y) return;

    HiResTarget* hr = hires_target(rdram, rdram_size);
    hle_hidden_->ensure_hidden(rdram_size);
    if (color_image_size == 2) { // 16-bit
        u16 color16 = static_cast<u16>(fill_color & 0xFFFF);
        for (u32 y = start_y; y < max_y; ++y) {
            for (u32 x = start_x; x < max_x; ++x) {
                u32 idx = color_image_addr + (y * fb_w + x) * 2;
                if (idx + 1 < rdram_size) {
                    rdram[idx + 0] = (color16 >> 8) & 0xFF;
                    rdram[idx + 1] = color16 & 0xFF;
                    hle_hidden_->force_hidden(idx >> 1, (color16 & 1) ? 3 : 0, color16); // coverage: all or nothing
                    if (y * fb_w + x < hires_shadow_len_) hires_shadow_[y * fb_w + x] = color16;
                }
            }
        }
        if (hr) hires_->fill_rect(hr, start_x, start_y, max_x, max_y, fb_pixel_to_argb(color16, 2));
    } else if (color_image_size == 1) { // 8-bit: byte k of the fill colour for address k mod 4
        for (u32 y = start_y; y < max_y; ++y) {
            for (u32 x = start_x; x < max_x; ++x) {
                const u32 idx = color_image_addr + y * fb_w + x;
                if (idx < rdram_size) rdram[idx] = static_cast<u8>(fill_color >> (((idx & 3) ^ 3) * 8));
            }
        }
    } else if (color_image_size == 3) { // 32-bit
        for (u32 y = start_y; y < max_y; ++y) {
            for (u32 x = start_x; x < max_x; ++x) {
                u32 idx = color_image_addr + (y * fb_w + x) * 4;
                if (idx + 3 < rdram_size) {
                    rdram[idx + 0] = (fill_color >> 24) & 0xFF;
                    rdram[idx + 1] = (fill_color >> 16) & 0xFF;
                    rdram[idx + 2] = (fill_color >> 8) & 0xFF;
                    rdram[idx + 3] = fill_color & 0xFF;
                    if (y * fb_w + x < hires_shadow_len_) hires_shadow_[y * fb_w + x] = fill_color;
                }
            }
        }
        if (hr) hires_->fill_rect(hr, start_x, start_y, max_x, max_y, fb_pixel_to_argb(fill_color, 4));
    }
}

void RDP::write_pixel(const DrawState& st, u32 x, u32 y, u32 color, f32 z, u8* rdram, size_t rdram_size) {
    raster::PixelAux aux;
    aux.z = raster::depth18(z);
    write_pixel(st, x, y, color, rdram, rdram_size, hires_shadow_, hires_shadow_len_, stat_pixels, aux);
}

// Thread-safe for pixels of distinct rows of one colour image (the native
// pass's bands): it only touches that pixel's RDRAM (colour and depth),
// ninth bits and shadow entry.
void RDP::write_pixel(const DrawState& st, u32 x, u32 y, u32 color, u8* rdram, size_t rdram_size,
                      u32* shadow, size_t shadow_len, PixelStats& stats, const raster::PixelAux& aux) {
    u32 fb_w = st.fb_w;
    if (x >= fb_w || y >= kMaxFbLines) return;
    u32 eff_lrx = (st.scissor_lrx > st.scissor_ulx) ? st.scissor_lrx : fb_w;
    u32 eff_lry = (st.scissor_lry > st.scissor_uly) ? st.scissor_lry : kMaxFbLines;
    if (x < st.scissor_ulx || x >= eff_lrx || y < st.scissor_uly || y >= eff_lry) return;

    const u32 pixel_idx = y * fb_w + x;
    raster::MemPixel mem;
    raster::PixelResult res;
    // The depth buffer: 16 bits per pixel, the same width as the colour image.
    const u32 zidx = st.zb_addr + pixel_idx * 2;
    const bool z_ok = (st.z_compare || st.z_update) && zidx + 1 < rdram_size;
    if (z_ok) {
        mem.zword = static_cast<u16>((rdram[zidx] << 8) | rdram[zidx + 1]);
        mem.zhidden = hle_hidden_->hidden_at(zidx >> 1, mem.zword);
    }
    if (st.fb_size == 2) { // 16-bit RGBA 5-5-5-1, coverage in the alpha bit and the two ninth bits
        const u32 idx = st.fb_addr + pixel_idx * 2;
        if (idx + 1 >= rdram_size) return;
        const u16 word = static_cast<u16>((rdram[idx] << 8) | rdram[idx + 1]);
        mem.r = (word >> 8) & 0xF8;
        mem.g = (word >> 3) & 0xF8;
        mem.b = (word << 2) & 0xF8;
        mem.cvg = ((word & 1) << 2) | hle_hidden_->hidden_at(idx >> 1, word);
        if (!raster::pixel_backend(st, color, aux, mem, res, true, x, y)) {
            stats.z_fail++;
            return;
        }
        const u16 p = static_cast<u16>(((res.r >> 3) << 11) | ((res.g >> 3) << 6) | ((res.b >> 3) << 1) | (res.cvg >> 2));
        rdram[idx] = static_cast<u8>(p >> 8);
        rdram[idx + 1] = static_cast<u8>(p);
        hle_hidden_->force_hidden(idx >> 1, res.cvg & 3, p);
        if (pixel_idx < shadow_len) shadow[pixel_idx] = p;
    } else if (st.fb_size == 3) { // 32-bit RGBA, coverage in the upper 3 bits of alpha
        const u32 idx = st.fb_addr + pixel_idx * 4;
        if (idx + 3 >= rdram_size) return;
        mem.r = rdram[idx];
        mem.g = rdram[idx + 1];
        mem.b = rdram[idx + 2];
        mem.cvg = rdram[idx + 3] >> 5;
        if (!raster::pixel_backend(st, color, aux, mem, res, true, x, y)) {
            stats.z_fail++;
            return;
        }
        const u8 a = static_cast<u8>(res.cvg << 5);
        rdram[idx] = res.r;
        rdram[idx + 1] = res.g;
        rdram[idx + 2] = res.b;
        rdram[idx + 3] = a;
        if (pixel_idx < shadow_len)
            shadow[pixel_idx] = (static_cast<u32>(res.r) << 24) | (static_cast<u32>(res.g) << 16) | (static_cast<u32>(res.b) << 8) | a;
    } else if (st.fb_size == 1) { // 8-bit intensity: every byte a pixel, coverage always full
        const u32 idx = st.fb_addr + pixel_idx;
        if (idx >= rdram_size) return;
        mem.r = mem.g = mem.b = rdram[idx];
        mem.cvg = 7;
        if (!raster::pixel_backend(st, color, aux, mem, res, true, x, y)) {
            stats.z_fail++;
            return;
        }
        // The RDP stores the red channel in even bytes and green in odd ones;
        // an odd byte's ninth bits follow its lowest bit.
        const u8 c = (idx & 1) ? res.g : res.r;
        rdram[idx] = c;
        const u32 h = idx & ~1u;
        if (idx & 1) hle_hidden_->force_hidden(h >> 1, (c & 1) * 3, static_cast<u16>((rdram[h] << 8) | c));
    } else {
        return;
    }
    if (res.z_write && z_ok) {
        rdram[zidx] = static_cast<u8>(res.zword >> 8);
        rdram[zidx + 1] = static_cast<u8>(res.zword);
        hle_hidden_->force_hidden(zidx >> 1, res.zhidden, res.zword);
    }
    stats.drawn++;
}

raster::TexCache* RDP::native_tex_cache(const raster::TexUnit& tu, u32 tlut_type) {
    u32 w = 0, h = 0;
    raster::tex_cache_dims(tu, w, h);
    // Decoding costs one fetch per texel up front; far larger than any
    // real texture means a degenerate tile, cheaper to sample directly.
    if (static_cast<u64>(w) * h > 256 * 256) return nullptr;
    u64 key_a = 0, key_b = 0;
    raster::tex_cache_key(tu, tlut_type, key_a, key_b);
    for (size_t i = native_tex_used_; i-- > 0;) {
        NativeTex& e = *native_tex_[i];
        if (e.gen == tmem_gen_ && e.key_a == key_a && e.key_b == key_b) return &e.cache;
    }
    if (native_tex_used_ == native_tex_.size()) native_tex_.push_back(std::make_unique<NativeTex>());
    NativeTex& e = *native_tex_[native_tex_used_++];
    e.gen = tmem_gen_;
    e.key_a = key_a;
    e.key_b = key_b;
    e.texels.resize(static_cast<size_t>(w) * h);
    e.cache.w = w;
    e.cache.h = h;
    e.cache.texels = e.texels.data();
    e.cache.state.store(0, std::memory_order_relaxed); // filled by the first sample (raster::tex_table)
    return &e.cache;
}

DrawState* RDP::native_snapshot(u32 tile0, bool use0, bool use1) {
    const DrawState& live = draw_state();
    if (!native_state_ || native_state_serial_ != draw_state_serial_ || native_state_gen_ != tmem_gen_) {
        if (!native_tmem_ || native_tmem_gen_ != tmem_gen_) {
            if (native_tmems_used_ == native_tmems_.size()) native_tmems_.push_back(std::make_unique<TmemSnapshot>());
            TmemSnapshot& t = *native_tmems_[native_tmems_used_++];
            t.data = tmem;
            t.dxt = tmem_word_dxt_zero;
            native_tmem_ = &t;
            native_tmem_gen_ = tmem_gen_;
        }
        if (native_states_used_ == native_states_.size()) native_states_.push_back(std::make_unique<DrawState>());
        DrawState& s = *native_states_[native_states_used_++];
        s = live;
        s.tmem = native_tmem_->data.data();
        s.tmem_dxt = native_tmem_->dxt.data();
        for (raster::TexUnit& tu : s.tex) tu.cache = nullptr;
        native_state_ = &s;
        native_state_serial_ = draw_state_serial_;
        native_state_gen_ = tmem_gen_;
    }
    DrawState& s = *native_state_;
    for (u32 i = 0; i < 2; ++i) {
        if (!(i == 0 ? use0 : use1)) continue;
        raster::TexUnit& tu = s.tex[(tile0 + i) & 7];
        if (!tu.cache) tu.cache = native_tex_cache(tu, s.tlut_type);
    }
    return &s;
}

void RDP::queue_native(NativeCmd& cmd, u8* rdram, size_t rdram_size) {
    const DrawState& st = *cmd.st;
    // Bytes of the colour image (and depth buffer) this command's rows can write.
    const u32 bpp = st.fb_size == 3 ? 4 : 2;
    auto clamp32 = [](u64 v) { return static_cast<u32>(std::min<u64>(v, 0xFFFFFFFFu)); };
    const u64 lo = st.fb_addr + static_cast<u64>(cmd.y_first) * st.fb_w * bpp;
    const u64 hi = st.fb_addr + (static_cast<u64>(cmd.y_last) + 1) * st.fb_w * bpp;
    const bool depth = st.z_compare || st.z_update;
    const u64 zlo = st.zb_addr + static_cast<u64>(cmd.y_first) * st.fb_w * 2;
    const u64 zhi = st.zb_addr + (static_cast<u64>(cmd.y_last) + 1) * st.fb_w * 2;
    if (native_queue_.empty()) {
        native_fb_lo_ = clamp32(lo);
        native_fb_hi_ = clamp32(hi);
        native_z_lo_ = native_z_hi_ = 0;
        native_work_ = 0;
    } else {
        native_fb_lo_ = std::min(native_fb_lo_, clamp32(lo));
        native_fb_hi_ = std::max(native_fb_hi_, clamp32(hi));
    }
    if (depth) {
        if (native_z_hi_ == native_z_lo_) {
            native_z_lo_ = clamp32(zlo);
            native_z_hi_ = clamp32(zhi);
        } else {
            native_z_lo_ = std::min(native_z_lo_, clamp32(zlo));
            native_z_hi_ = std::max(native_z_hi_, clamp32(zhi));
        }
    }
    native_work_ += static_cast<u64>(cmd.y_last - cmd.y_first + 1) * st.fb_w;
    native_rdram_ = rdram;
    native_rdram_size_ = rdram_size;
    native_queue_.push_back(cmd);
}

void RDP::flush_native() {
    if (native_queue_.empty()) return;
    hle_hidden_->ensure_hidden(native_rdram_size_);
    constexpr s32 kBandRows = 8;
    constexpr u32 kBands = (kMaxFbLines + kBandRows - 1) / kBandRows;
    // Waking the workers costs more than a handful of small draws.
    bool threaded = native_work_ >= 20000;
    if (threaded && !raster_pool_) {
        // Started on first use: one band worker per spare hardware thread
        // (the emulation thread draws bands too), capped - a frame only has
        // 240 (at most 480) rows to split.
        const unsigned hw = std::thread::hardware_concurrency();
        raster_pool_ = std::make_unique<RasterPool>(std::min(hw > 1 ? hw - 1 : 0u, 15u));
    }
    threaded = threaded && raster_pool_->workers() > 0;
    const u32 bands = threaded ? kBands : 1;
    std::array<PixelStats, kBands> band_stats{};
    auto draw_band = [&](u32 band) {
        const s32 row_begin = threaded ? static_cast<s32>(band) * kBandRows : 0;
        const s32 row_end = threaded ? row_begin + kBandRows : INT_MAX;
        PixelStats& stats = band_stats[band];
        for (const NativeCmd& c : native_queue_) {
            if (c.y_last < row_begin || c.y_first >= row_end) continue;
            NativeSink sink{*this, *c.st, native_rdram_, native_rdram_size_, c.shadow, c.shadow_len, stats};
            if (c.kind == NativeCmd::Kind::Triangle) {
                raster::triangle(*c.st, c.v[0], c.v[1], c.v[2], c.area, 1, row_begin, row_end, sink);
            } else {
                raster::tex_rect(*c.st, c.ulx, c.uly, c.lrx, c.lry, c.tile, c.s, c.t, c.dsdx, c.dtdy, c.flip, 1,
                                 row_begin, row_end, sink);
            }
        }
    };
    if (threaded) raster_pool_->run(bands, draw_band);
    else draw_band(0);
    for (const PixelStats& b : band_stats) {
        stat_pixels.drawn += b.drawn;
        stat_pixels.z_fail += b.z_fail;
        stat_pixels.a_fail += b.a_fail;
    }

    native_queue_.clear();
    native_states_used_ = 0;
    native_tmems_used_ = 0;
    native_state_ = nullptr;
    native_tmem_ = nullptr;
    // Decoded textures of the current TMEM contents stay (moved to the front);
    // older ones can't be sampled again.
    size_t kept = 0;
    for (size_t i = 0; i < native_tex_used_; ++i) {
        if (native_tex_[i]->gen == tmem_gen_) std::swap(native_tex_[kept++], native_tex_[i]);
    }
    native_tex_used_ = kept;
}

void RDP::rasterize_tex_rect(u32 ulx, u32 uly, u32 lrx, u32 lry, u32 tile_idx, f32 s, f32 t, f32 dsdx, f32 dtdy, bool flip, u8* rdram, size_t rdram_size) {
    if (geometry_only_) return;
    ++tex_rect_count_;
    if (color_image_addr >= rdram_size) return;
    if ((pick_frame_ == -2 || (pick_frame_ >= 0 && g_current_frame == pick_frame_)) &&
        static_cast<s32>(ulx) <= pick_x_ && pick_x_ < static_cast<s32>(lrx) && static_cast<s32>(uly) <= pick_y_ && pick_y_ < static_cast<s32>(lry)) {
        std::fprintf(stderr, "PICKRECT frame=%d rect=%u,%u-%u,%u tile=%u st=%.2f,%.2f d=%.3f,%.3f comb=%08x:%08x omh=%08x oml=%08x prim=%08x env=%08x\n",
                     g_current_frame, ulx, uly, lrx, lry, tile_idx, s, t, dsdx, dtdy, combine_mode_w0, combine_mode_w1, other_mode_h,
                     other_mode_l, prim_color, env_color);
        const Tile& tl = tiles[tile_idx & 7];
        std::fprintf(stderr, "  tile%u fmt=%u siz=%u line=%u tmem=%u pal=%u cms=%u/%u/%u/%u cmt=%u/%u/%u/%u sl=%u tl=%u sh=%u th=%u\n",
                     tile_idx & 7, tl.format, tl.size, tl.line, tl.tmem, tl.palette, tl.clamp_s, tl.mirror_s, tl.mask_s, tl.shift_s,
                     tl.clamp_t, tl.mirror_t, tl.mask_t, tl.shift_t, tl.sl, tl.tl, tl.sh, tl.th);
        std::fprintf(stderr, "  tmem[0..32]:");
        for (int i = 0; i < 32; ++i) std::fprintf(stderr, " %02x", tmem[tl.tmem * 8 + i]);
        std::fprintf(stderr, "\n  tlut[0..16]:");
        for (int i = 0; i < 16; ++i) std::fprintf(stderr, " %02x%02x", tmem[0x800 + 2 * i], tmem[0x800 + 2 * i + 1]);
        std::fprintf(stderr, "\n");
    }
    const DrawState& live = draw_state();
    HiResTarget* hr = hires_target(rdram, rdram_size);
    // Same tiles raster::tex_rect() samples.
    const bool combined = live.combine_set && !live.copy_mode;
    NativeCmd cmd{};
    cmd.kind = NativeCmd::Kind::TexRect;
    cmd.st = native_snapshot(tile_idx, !combined || live.need_tex0, combined && live.need_tex1);
    // The rows raster::tex_rect() can draw.
    const raster::TexRectSetup rs = raster::tex_rect_setup(live, ulx, uly, lrx, lry, dsdx, dtdy, flip);
    cmd.y_first = std::min(rs.y0, static_cast<s32>(kMaxFbLines));
    cmd.y_last = std::min(rs.y1, static_cast<s32>(kMaxFbLines) - 1);
    cmd.shadow = hires_shadow_;
    cmd.shadow_len = hires_shadow_len_;
    cmd.ulx = ulx; cmd.uly = uly; cmd.lrx = lrx; cmd.lry = lry; cmd.tile = tile_idx;
    cmd.s = s; cmd.t = t; cmd.dsdx = dsdx; cmd.dtdy = dtdy; cmd.flip = flip;
    if (cmd.y_first <= cmd.y_last) queue_native(cmd, rdram, rdram_size);
    if (hr) hires_->tex_rect(hr, live, draw_state_serial_, tmem_gen_, ulx, uly, lrx, lry, tile_idx, s, t, dsdx, dtdy, flip);
}

void RDP::rasterize_triangle(const Vertex& in0, const Vertex& in1, const Vertex& in2, u8* rdram, size_t rdram_size) {
    stat_rast_called++;
    ++triangle_count_;
    if (color_image_addr >= rdram_size) return;

    if (in0.w <= 0.0001f || in1.w <= 0.0001f || in2.w <= 0.0001f) return;
    // The microcode hands the RDP screen positions in quarter pixels.
    auto snap = [](Vertex v) {
        v.sx = std::nearbyint(v.sx * 4.0f) * 0.25f;
        v.sy = std::nearbyint(v.sy * 4.0f) * 0.25f;
        return v;
    };
    const Vertex v0 = snap(in0), v1 = snap(in1), v2 = snap(in2);

    // Backface culling: signed 2D area
    f32 area = (v1.sx - v0.sx) * (v2.sy - v0.sy) - (v2.sx - v0.sx) * (v1.sy - v0.sy);
    bool cull_front = false;
    bool cull_back = false;
    if (current_ucode_active == MicrocodeType::F3DEX2) {
        cull_front = (geometry_mode & 0x00000200) != 0;
        cull_back  = (geometry_mode & 0x00000400) != 0;
    } else {
        cull_front = (geometry_mode & 0x00001000) != 0;
        cull_back  = (geometry_mode & 0x00002000) != 0;
    }

    // Both bits set (G_CULL_BOTH) draws normally on the real microcode, as the low-level RSP
    // shows: Mortal Kombat 4's menu sets it and every triangle used to be culled.
    if (cull_front && cull_back) cull_front = cull_back = false;
    if (cull_back && area >= 0.0f) { stat_cull_back++; return; }
    if (cull_front && area <= 0.0f) { stat_cull_front++; return; }
    if (std::abs(area) < 0.001f) return;
    draw_triangle(v0, v1, v2, area, rdram, rdram_size);
}

void RDP::draw_triangle(const Vertex& v0, const Vertex& v1, const Vertex& v2, f32 area, u8* rdram, size_t rdram_size) {
    const DrawState& live = draw_state();
    HiResTarget* hr = hires_target(rdram, rdram_size);
    f32 min_x, max_x, min_y, max_y;
    if (!raster::triangle_bounds(live, v0, v1, v2, 1, min_x, max_x, min_y, max_y)) {
        stat_scissor_reject++;
        return;
    }
    NativeCmd cmd{};
    cmd.kind = NativeCmd::Kind::Triangle;
    // Same tiles raster::triangle() samples.
    const bool combined = live.combine_set;
    const bool textured = live.texture_enabled;
    cmd.st = native_snapshot(live.active_tile, textured && (!combined || live.need_tex0),
                             textured && combined && live.need_tex1);
    cmd.y_first = static_cast<s32>(min_y); // exactly the rows raster::triangle() visits
    cmd.y_last = static_cast<s32>(max_y);
    cmd.shadow = hires_shadow_;
    cmd.shadow_len = hires_shadow_len_;
    cmd.v[0] = v0;
    cmd.v[1] = v1;
    cmd.v[2] = v2;
    cmd.area = area;
    queue_native(cmd, rdram, rdram_size);
    if (hr) hires_->triangle(hr, live, draw_state_serial_, tmem_gen_, v0, v1, v2, area);
}

// ===========================================================================
// S2DEX / S2DEX2 microcode (HLE)
// ===========================================================================
//
// S2DEX is a 2D sprite/background microcode "derived from F3DEX" (per the
// N64 programming manual, ch.25.5): it reuses F3DEX's shared RDP-level GBI
// (gDPSetTextureImage, gDPSetTile, scissoring, sync, colour/combine setup,
// gSPDisplayList/gSPEndDisplayList/gSPBranchList, gSPSegment, ...) and only
// replaces the 3D-primitive opcodes (matrices, vertices, triangles, lights)
// with its own low-numbered opcodes for BG planes, 2D sprites/rectangles,
// a single (stack-less) 2D transform matrix, TMEM texture loading, and a
// status-word-based conditional display-list branch.
//
// Because the replaced opcodes reuse the same numeric values as F3D's own
// vertex/triangle/matrix commands (0x01-0x0b, 0xB0-0xB2, 0xC1-0xC4, 0xDA,
// 0xDC, 0xE4), they can't be added as ordinary switch cases in
// process_display_list's shared switch without colliding. Instead,
// execute_s2dex_command() intercepts every opcode up front whenever the
// active microcode is S2DEX/S2DEX2, consuming the ones that are S2DEX
// opcodes (returning true) and falling through (returning false) for the
// genuinely shared opcodes (G_DL, G_ENDDL, G_SETTIMG/G_SETTILE/..., sync,
// scissor, combine/colour setup) which the common switch already decodes
// correctly for both GBI generations.
//
// Sprites/rectangles (G_OBJ_SPRITE/RECTANGLE/RECTANGLE_R) draw from texel
// data that a prior G_OBJ_LOADTXTR-family command already placed in TMEM, so
// those reuse the existing tile/TMEM/sample_texture/rasterize_tex_rect
// machinery, just like a normal F3D G_TEXRECT. Backgrounds (G_BG_1CYC /
// G_BG_COPY) are conceptually a texture "streamed" through TMEM in slices by
// the real microcode purely because TMEM is only 4KB; since our HLE can
// address RDRAM directly, s2dex_draw_bg() skips that streaming and samples
// the source image directly out of RDRAM instead, which is visually
// equivalent without needing to reproduce the RSP's TMEM-slicing algorithm.

namespace {

inline u16 s2d_read_u16(const u8* rdram, u32 addr) {
    return (static_cast<u16>(rdram[addr]) << 8) | static_cast<u16>(rdram[addr + 1]);
}
inline s16 s2d_read_s16(const u8* rdram, u32 addr) {
    return static_cast<s16>(s2d_read_u16(rdram, addr));
}
inline u32 s2d_read_u32(const u8* rdram, u32 addr) {
    return (static_cast<u32>(rdram[addr + 0]) << 24) | (static_cast<u32>(rdram[addr + 1]) << 16) |
           (static_cast<u32>(rdram[addr + 2]) << 8)  |  static_cast<u32>(rdram[addr + 3]);
}
inline s32 s2d_read_s32(const u8* rdram, u32 addr) {
    return static_cast<s32>(s2d_read_u32(rdram, addr));
}

// Fetches one texel directly out of a plain row-major RDRAM image (used by
// BG drawing, which bypasses TMEM entirely). Mirrors fetch_texel()'s format
// decode (see above), but with a linear RDRAM stride instead of TMEM's
// tile/line addressing -- CI textures still look their palette up in TMEM,
// since a real G_OBJ_LOADTXTR TLUT load (or plain G_LOADTLUT) already placed
// it there.
// row_stride_bytes must be the real hardware row stride (see
// s2d_bg_row_stride_bytes below) -- it is *not* generally image_w_texels
// (or half/double of it): the S2DEX microcode derives it via a lossy
// fixed-point computation on the raw (undivided) imageW field, and BG
// assets with a non-"round" imageW rely on that exact truncation, so a
// naively-recomputed stride shifts every row by a texel or so (visible as
// a diagonal shear across the whole image).
u32 fetch_bg_texel_raw(const u8* rdram, size_t rdram_size, const std::array<u8, 4096>& tmem,
                        u32 image_addr, u8 fmt, u8 siz, u8 pal, u32 tlut_type,
                        u32 row_stride_bytes, s32 ix, s32 iy) {
    if (row_stride_bytes == 0) return 0;
    if (siz == 0) { // 4-bit
        u32 offset = image_addr + static_cast<u32>(iy) * row_stride_bytes + (static_cast<u32>(ix) / 2);
        if (offset >= rdram_size) return 0;
        u8 byte_val = rdram[offset];
        u8 val = (ix & 1) ? (byte_val & 0xF) : ((byte_val >> 4) & 0xF);
        if (fmt == 2) { // CI4
            if (tlut_type == 0) { u8 i = static_cast<u8>(val * 17); return (static_cast<u32>(i) << 24) | (i << 16) | (i << 8) | i; }
            return raster::lookup_tlut(tmem.data(), pal * 16u + val, tlut_type);
        } else if (fmt == 3) { // IA4
            u8 i = static_cast<u8>(((val >> 1) & 0x7) * 255 / 7);
            u8 a = (val & 1) ? 255 : 0;
            return (static_cast<u32>(a) << 24) | (static_cast<u32>(i) << 16) | (static_cast<u32>(i) << 8) | i;
        }
        u8 i = static_cast<u8>(val * 17);
        return (static_cast<u32>(i) << 24) | (i << 16) | (i << 8) | i;
    } else if (siz == 1) { // 8-bit
        u32 offset = image_addr + static_cast<u32>(iy) * row_stride_bytes + static_cast<u32>(ix);
        if (offset >= rdram_size) return 0;
        u8 val = rdram[offset];
        if (fmt == 2) { // CI8
            if (tlut_type == 0) return 0xFF000000u | (static_cast<u32>(val) << 16) | (val << 8) | val;
            return raster::lookup_tlut(tmem.data(), val, tlut_type);
        } else if (fmt == 3) { // IA8
            u8 i = static_cast<u8>(((val >> 4) & 0xF) * 17);
            u8 a = static_cast<u8>((val & 0xF) * 17);
            return (static_cast<u32>(a) << 24) | (static_cast<u32>(i) << 16) | (static_cast<u32>(i) << 8) | i;
        }
        return (static_cast<u32>(val) << 24) | (val << 16) | (val << 8) | val;
    } else if (siz == 2) { // 16-bit
        u32 offset = image_addr + static_cast<u32>(iy) * row_stride_bytes + static_cast<u32>(ix) * 2;
        if (offset + 1 >= rdram_size) return 0;
        u16 p = (static_cast<u16>(rdram[offset]) << 8) | static_cast<u16>(rdram[offset + 1]);
        if (fmt == 3) { // IA16
            u8 i = (p >> 8) & 0xFF, a = p & 0xFF;
            return (static_cast<u32>(a) << 24) | (static_cast<u32>(i) << 16) | (static_cast<u32>(i) << 8) | i;
        }
        return raster::rgba16_to_rgba32(p);
    } else { // 32-bit: plain R,G,B,A bytes in RDRAM (no TMEM bank split -- that's a TMEM-only quirk)
        u32 offset = image_addr + static_cast<u32>(iy) * row_stride_bytes + static_cast<u32>(ix) * 4;
        if (offset + 3 >= rdram_size) return 0;
        u8 r = rdram[offset], g = rdram[offset + 1], b = rdram[offset + 2], a = rdram[offset + 3];
        return (static_cast<u32>(a) << 24) | (static_cast<u32>(r) << 16) | (static_cast<u32>(g) << 8) | b;
    }
}

// The BG image's row byte-stride, replicating the exact fixed-point
// computation guS2DEmuBgRect1Cyc (us2dex_emu.c) performs on the *raw*
// (undivided) imageW field:
//   imageSrcW      = imageWraw << 3;                 // raw field, not imageW/4
//   imageSrcWsize  = (imageSrcW / TMEMSHIFT[siz]) << 3;
// This is a genuinely lossy computation (integer division truncates), and
// BG assets whose imageW isn't a "round" multiple for their format/size
// rely on that exact truncation -- so this can differ from the naive
// texel_count * bytes_per_texel stride by a texel or so per row, which
// shows up as a diagonal shear across the whole image if not replicated.
u32 s2d_bg_row_stride_bytes(u16 image_w_raw, u8 siz) {
    static const u32 TMEM_SHIFT[4] = {0x200, 0x100, 0x80, 0x40};
    u32 image_src_w = static_cast<u32>(image_w_raw) << 3;
    return (image_src_w / TMEM_SHIFT[siz & 0x3]) << 3;
}

} // namespace

// gSPObjLoadTxtr: what the microcode does with a uObjTxtr (24 bytes: type,
// image, three u16 parameters, sid, flag, mask). The load is skipped when
// status word sid/4 already says this texture is there ((status & mask) ==
// flag), and afterwards the status word takes the flag bits. The loads
// themselves are plain RDP loads of 8-bit texels (TLUTs: 16-bit), whatever
// the last gDPSetTextureImage said:
//   TXTRBLOCK: LOADBLOCK of (tsize + 1) 64-bit words, DXT = tline
//   TXTRTILE:  LOADTILE of theight/4 + 1 rows of (twidth + 1) * 2 bytes,
//              which is the TMEM line as well: one contiguous copy
//   TLUT:      LOADTLUT of pnum + 1 entries to TMEM word phead (>= 256)
void RDP::s2dex_load_txtr(u32 tx_addr, u8* rdram, size_t rdram_size) {
    if (tx_addr + 24 > rdram_size) return;
    const u32 type = s2d_read_u32(rdram, tx_addr);
    const u32 image_ptr = segment_to_physical(s2d_read_u32(rdram, tx_addr + 4));
    const u16 p0 = s2d_read_u16(rdram, tx_addr + 8);
    const u16 p1 = s2d_read_u16(rdram, tx_addr + 10);
    const u16 p2 = s2d_read_u16(rdram, tx_addr + 12);
    const u32 slot = (s2d_read_u16(rdram, tx_addr + 14) >> 2) & 3;
    const u32 flag = s2d_read_u32(rdram, tx_addr + 16);
    const u32 mask = s2d_read_u32(rdram, tx_addr + 20);
    if ((s2d_genstat[slot] & mask) == flag) return;
    s2d_genstat[slot] = (s2d_genstat[slot] & ~mask) | (flag & mask);

    auto copy = [&](u32 dest, u32 bytes, bool dxt_zero) {
        if (dest >= tmem.size()) return;
        bytes = std::min<u32>(bytes, static_cast<u32>(tmem.size()) - dest);
        native_before_read(image_ptr, static_cast<u64>(image_ptr) + bytes);
        tmem_dirty = true;
        ++tmem_gen_;
        for (u32 i = 0; i < bytes; ++i) tmem[dest + i] = image_ptr + i < rdram_size ? rdram[image_ptr + i] : 0;
        for (u32 w = dest / 8; w < (dest + bytes + 7) / 8 && w < 512; ++w) tmem_word_dxt_zero[w] = dxt_zero;
    };
    if (type == 0x00000030) { // G_OBJLT_TLUT (packed palette layout, see G_LOADTLUT)
        copy(p0 >= 256 ? 0x800 + (p0 - 256) * 2u : p0 * 8u, (p1 + 1u) * 2, false);
    } else if (type == 0x00fc1034) { // G_OBJLT_TXTRTILE
        copy(p0 * 8u, ((p2 >> 2) + 1u) * ((p1 + 1u) * 2), false);
    } else { // G_OBJLT_TXTRBLOCK (0x00001033)
        copy(p0 * 8u, (p1 + 1u) * 8, p2 == 0);
    }
}

RDP::S2DObjSpriteInfo RDP::s2dex_setup_obj_tile(u32 sp_addr, const u8* rdram, size_t rdram_size) {
    S2DObjSpriteInfo info{};
    if (sp_addr + 24 > rdram_size) return info;

    s16 objX = s2d_read_s16(rdram, sp_addr + 0);
    u16 scaleWraw = s2d_read_u16(rdram, sp_addr + 2);
    u16 imageWraw = s2d_read_u16(rdram, sp_addr + 4);
    s16 objY = s2d_read_s16(rdram, sp_addr + 8);
    u16 scaleHraw = s2d_read_u16(rdram, sp_addr + 10);
    u16 imageHraw = s2d_read_u16(rdram, sp_addr + 12);
    u16 imageStride = s2d_read_u16(rdram, sp_addr + 16);
    u16 imageAdrs   = s2d_read_u16(rdram, sp_addr + 18);
    u8  imageFmt  = rdram[sp_addr + 20];
    u8  imageSiz  = rdram[sp_addr + 21];
    u8  imagePal  = rdram[sp_addr + 22];
    u8  imageFlags = rdram[sp_addr + 23];

    info.objX = objX / 4.0f;
    info.objY = objY / 4.0f;
    info.scaleW = (scaleWraw ? scaleWraw : 1024) / 1024.0f;
    info.scaleH = (scaleHraw ? scaleHraw : 1024) / 1024.0f;
    info.imageW = imageWraw / 32.0f;
    info.imageH = imageHraw / 32.0f;
    info.imageFlags = imageFlags;

    Tile& t = tiles[0];
    t = Tile{};
    t.format = imageFmt;
    t.size = imageSiz;
    t.line = imageStride;
    t.tmem = imageAdrs;
    t.palette = imagePal;
    t.clamp_s = t.clamp_t = 1;
    t.mask_s = t.mask_t = 0;
    u32 tex_w = std::max<u32>(1, static_cast<u32>(info.imageW));
    u32 tex_h = std::max<u32>(1, static_cast<u32>(info.imageH));
    t.sl = 0; t.tl = 0;
    t.sh = static_cast<u16>((tex_w - 1) * 4);
    t.th = static_cast<u16>((tex_h - 1) * 4);

    active_tile = 0;
    texture_enabled = true;
    draw_state_dirty_ = true;
    return info;
}

void RDP::s2dex_draw_obj_rect(u32 sp_addr, bool use_matrix, u8* rdram, size_t rdram_size) {
    if (geometry_only_) return;
    S2DObjSpriteInfo info = s2dex_setup_obj_tile(sp_addr, rdram, rdram_size);
    if (info.imageW <= 0.0f || info.imageH <= 0.0f) return;

    f32 obj_w = info.imageW / info.scaleW;
    f32 obj_h = info.imageH / info.scaleH;

    f32 screen_x0, screen_y0, screen_w, screen_h;
    if (use_matrix) {
        f32 base_x = (obj2d_matrix.baseScaleX != 0.0f) ? obj2d_matrix.baseScaleX : 1.0f;
        f32 base_y = (obj2d_matrix.baseScaleY != 0.0f) ? obj2d_matrix.baseScaleY : 1.0f;
        screen_x0 = obj2d_matrix.X + info.objX / base_x;
        screen_y0 = obj2d_matrix.Y + info.objY / base_y;
        screen_w = obj_w / base_x;
        screen_h = obj_h / base_y;
    } else {
        screen_x0 = info.objX;
        screen_y0 = info.objY;
        screen_w = obj_w;
        screen_h = obj_h;
    }
    if (screen_w <= 0.0f || screen_h <= 0.0f) return;

    bool flipS = (info.imageFlags & 0x01) != 0;
    bool flipT = (info.imageFlags & 0x10) != 0;

    f32 dsdx = info.imageW / screen_w;
    f32 dtdy = info.imageH / screen_h;

    f32 fx0 = screen_x0, fy0 = screen_y0;
    f32 fx1 = screen_x0 + screen_w, fy1 = screen_y0 + screen_h;

    f32 clip_x0 = std::max(fx0, static_cast<f32>(scissor_ulx));
    f32 clip_y0 = std::max(fy0, static_cast<f32>(scissor_uly));
    f32 clip_x1 = std::min(fx1, static_cast<f32>(scissor_lrx));
    f32 clip_y1 = std::min(fy1, static_cast<f32>(scissor_lry));
    if (clip_x1 <= clip_x0 || clip_y1 <= clip_y0) return;

    f32 skip_left = clip_x0 - fx0;
    f32 skip_top  = clip_y0 - fy0;
    f32 skip_right = fx1 - clip_x1;
    f32 skip_bottom = fy1 - clip_y1;

    f32 s0 = flipS ? (info.imageW - skip_right * dsdx) : (skip_left * dsdx);
    f32 t0 = flipT ? (info.imageH - skip_bottom * dtdy) : (skip_top * dtdy);
    f32 dsdx_signed = flipS ? -dsdx : dsdx;
    f32 dtdy_signed = flipT ? -dtdy : dtdy;
    // Sample from the centre of the first texel to avoid an off-by-one at
    // the mirrored edge (floor(imageW - epsilon) must land on imageW-1).
    if (flipS) s0 -= 0.001f;
    if (flipT) t0 -= 0.001f;

    // In quarter pixels, as the microcode hands them to the RDP.
    u32 ulx = static_cast<u32>(clip_x0 * 4.0f);
    u32 uly = static_cast<u32>(clip_y0 * 4.0f);
    u32 lrx = static_cast<u32>(clip_x1 * 4.0f);
    u32 lry = static_cast<u32>(clip_y1 * 4.0f);

    rasterize_tex_rect(ulx, uly, lrx, lry, 0, s0, t0, dsdx_signed, dtdy_signed, false, rdram, rdram_size);
}

void RDP::s2dex_draw_obj_sprite(u32 sp_addr, u8* rdram, size_t rdram_size) {
    if (geometry_only_) return;
    S2DObjSpriteInfo info = s2dex_setup_obj_tile(sp_addr, rdram, rdram_size);
    if (info.imageW <= 0.0f || info.imageH <= 0.0f) return;

    f32 obj_w = info.imageW / info.scaleW;
    f32 obj_h = info.imageH / info.scaleH;
    bool flipS = (info.imageFlags & 0x01) != 0;
    bool flipT = (info.imageFlags & 0x10) != 0;

    f32 ox0 = info.objX, oy0 = info.objY;
    f32 ox1 = info.objX + obj_w, oy1 = info.objY + obj_h;

    f32 u0 = flipS ? info.imageW : 0.0f, u1 = flipS ? 0.0f : info.imageW;
    f32 v0 = flipT ? info.imageH : 0.0f, v1 = flipT ? 0.0f : info.imageH;

    auto transform = [&](f32 ox, f32 oy, f32 u, f32 v) {
        Vertex vert{};
        vert.sx = obj2d_matrix.A * ox + obj2d_matrix.B * oy + obj2d_matrix.X;
        vert.sy = obj2d_matrix.C * ox + obj2d_matrix.D * oy + obj2d_matrix.Y;
        vert.sz = 0.0f;
        vert.x = vert.y = vert.z = 0.0f;
        vert.w = 1.0f;
        vert.u = u;
        vert.v = v;
        vert.r = vert.g = vert.b = vert.a = 255;
        return vert;
    };

    Vertex p00 = transform(ox0, oy0, u0, v0);
    Vertex p10 = transform(ox1, oy0, u1, v0);
    Vertex p11 = transform(ox1, oy1, u1, v1);
    Vertex p01 = transform(ox0, oy1, u0, v1);

    // Shade is white: the combiner set up by the game (prim/env colour
    // tints, fades) applies as it does to rectangles.
    draw_state_dirty_ = true;
    rasterize_triangle(p00, p10, p11, rdram, rdram_size);
    rasterize_triangle(p00, p11, p01, rdram, rdram_size);
}

void RDP::s2dex_draw_bg(u32 bg_addr, bool scaled, u8* rdram, size_t rdram_size) {
    if (geometry_only_) return;
    flush_native(); // draws synchronously, and reads its image straight from RDRAM
    if (bg_addr + 32 > rdram_size) return;

    u16 imageX = s2d_read_u16(rdram, bg_addr + 0);
    u16 imageWraw = s2d_read_u16(rdram, bg_addr + 2);
    s16 frameX = s2d_read_s16(rdram, bg_addr + 4);
    u16 frameWraw = s2d_read_u16(rdram, bg_addr + 6);
    u16 imageY = s2d_read_u16(rdram, bg_addr + 8);
    u16 imageHraw = s2d_read_u16(rdram, bg_addr + 10);
    s16 frameY = s2d_read_s16(rdram, bg_addr + 12);
    u16 frameHraw = s2d_read_u16(rdram, bg_addr + 14);
    // NOTE: uObjBg_t's "imagePtr" field is declared `u64 *`, but on the N64's
    // 32-bit MIPS target a pointer is 4 bytes regardless of what it points
    // to -- it is NOT an 8-byte field. Every offset below matches the real
    // (4+2+2+2+2+2+2+2 = 16, +4-byte pointer = 20, ...) 40-byte layout from
    // gs2dex.h; treating the pointer as 8 bytes shifts every subsequent
    // field (format/size/palette/flip/scale) by 4 bytes and reads garbage.
    u32 imagePtr = segment_to_physical(s2d_read_u32(rdram, bg_addr + 16));
    u8 imageFmt = rdram[bg_addr + 22];
    u8 imageSiz = rdram[bg_addr + 23];
    u16 imagePal = s2d_read_u16(rdram, bg_addr + 24);
    u16 imageFlip = s2d_read_u16(rdram, bg_addr + 26);

    f32 scaleW = 1024.0f, scaleH = 1024.0f;
    if (scaled && bg_addr + 40 <= rdram_size) {
        u16 sw = s2d_read_u16(rdram, bg_addr + 28);
        u16 sh = s2d_read_u16(rdram, bg_addr + 30);
        scaleW = sw ? sw : 1024.0f;
        scaleH = sh ? sh : 1024.0f;
    }
    f32 sW = scaleW / 1024.0f, sH = scaleH / 1024.0f;

    f32 frameX0 = frameX / 4.0f, frameW = frameWraw / 4.0f;
    f32 frameY0 = frameY / 4.0f, frameH = frameHraw / 4.0f;
    f32 imageX0 = imageX / 32.0f, imageY0 = imageY / 32.0f;
    f32 imageW = imageWraw / 4.0f, imageH = imageHraw / 4.0f;
    if (imageW <= 0.0f || imageH <= 0.0f || frameW <= 0.0f || frameH <= 0.0f) return;

    bool flipS = (imageFlip & 0x01) != 0; // G_BG_FLAG_FLIPS

    f32 fx0 = frameX0, fx1 = frameX0 + frameW;
    f32 fy0 = frameY0, fy1 = frameY0 + frameH;
    f32 clip_x0 = std::max(fx0, static_cast<f32>(scissor_ulx));
    f32 clip_y0 = std::max(fy0, static_cast<f32>(scissor_uly));
    f32 clip_x1 = std::min(fx1, static_cast<f32>(scissor_lrx));
    f32 clip_y1 = std::min(fy1, static_cast<f32>(scissor_lry));
    if (clip_x1 <= clip_x0 || clip_y1 <= clip_y0) return;

    f32 skip_left = clip_x0 - fx0;
    f32 skip_top = clip_y0 - fy0;
    f32 skip_right = fx1 - clip_x1;

    // img_x_for(x)=img_x_start + x*sW assuming no flip; a flip mirrors which
    // screen column maps to the image's left edge.
    f32 img_x_start = imageX0 + (flipS ? skip_right : skip_left) * sW;
    f32 img_y_start = imageY0 + skip_top * sH;

    u32 image_w_texels = std::max<u32>(1, static_cast<u32>(imageW));
    u32 image_h_texels = std::max<u32>(1, static_cast<u32>(imageH));
    u32 row_stride_bytes = s2d_bg_row_stride_bytes(imageWraw, imageSiz);

    u32 out_w = static_cast<u32>(clip_x1 - clip_x0);
    u32 out_h = static_cast<u32>(clip_y1 - clip_y0);
    u32 screen_x0 = static_cast<u32>(clip_x0);
    u32 screen_y0 = static_cast<u32>(clip_y0);

    u32 tlut_type = (other_mode_h >> 14) & 0x3;
    bool bilerp = (obj_render_mode & 0x08) != 0; // G_OBJRM_BILERP; BG only interpolates horizontally

    const DrawState& st = draw_state();
    HiResTarget* hr = hires_target(rdram, rdram_size);
    if (hr) hires_bg_px_.assign(static_cast<size_t>(out_w) * out_h, 0);

    // The source column (and filter weight) of each output column is the same
    // on every row: work it out once instead of per pixel (two divisions each).
    bg_columns_.resize(out_w);
    BgColumn* columns = bg_columns_.data();
    for (u32 x = 0; x < out_w; ++x) {
        u32 col = flipS ? (out_w - 1 - x) : x;
        f32 fxc = img_x_start + static_cast<f32>(col) * sW;
        f32 fx_floor = std::floor(fxc);
        s32 ix = static_cast<s32>(fx_floor) % static_cast<s32>(image_w_texels);
        if (ix < 0) ix += image_w_texels;
        columns[x] = {ix, (ix + 1) % static_cast<s32>(image_w_texels), fxc - fx_floor};
    }
    // The combiner only depends on the texel here, and backgrounds are mostly
    // runs of one colour: reuse the previous result for a repeated input.
    u32 last_in = 0, last_out = 0;
    bool have_last = false;

    for (u32 y = 0; y < out_h; ++y) {
        f32 fyc = img_y_start + static_cast<f32>(y) * sH;
        s32 iy = static_cast<s32>(std::floor(fyc)) % static_cast<s32>(image_h_texels);
        if (iy < 0) iy += image_h_texels;

        for (u32 x = 0; x < out_w; ++x) {
            const s32 ix = columns[x].ix;

            u32 color;
            if (bilerp) {
                s32 ix2 = columns[x].ix2;
                f32 frac = columns[x].frac;
                u32 c0 = fetch_bg_texel_raw(rdram, rdram_size, tmem, imagePtr, imageFmt, imageSiz,
                                             static_cast<u8>(imagePal), tlut_type, row_stride_bytes, ix, iy);
                u32 c1 = fetch_bg_texel_raw(rdram, rdram_size, tmem, imagePtr, imageFmt, imageSiz,
                                             static_cast<u8>(imagePal), tlut_type, row_stride_bytes, ix2, iy);
                u8 out[4];
                for (int i = 0; i < 4; ++i) {
                    int shift = 24 - i * 8;
                    u8 a = (c0 >> shift) & 0xFF, b = (c1 >> shift) & 0xFF;
                    out[i] = static_cast<u8>(a + frac * (static_cast<f32>(b) - static_cast<f32>(a)));
                }
                color = (static_cast<u32>(out[0]) << 24) | (static_cast<u32>(out[1]) << 16) |
                        (static_cast<u32>(out[2]) << 8) | out[3];
            } else {
                color = fetch_bg_texel_raw(rdram, rdram_size, tmem, imagePtr, imageFmt, imageSiz,
                                            static_cast<u8>(imagePal), tlut_type, row_stride_bytes, ix, iy);
            }

            if (st.combine_set) {
                if (!have_last || color != last_in) {
                    last_in = color;
                    last_out = raster::combine(st, color, color, 0, 0, 0, 0);
                    have_last = true;
                }
                color = last_out;
            }
            write_pixel(st, screen_x0 + x, screen_y0 + y, color, 0.0f, rdram, rdram_size);
            if (hr) hires_bg_px_[static_cast<size_t>(y) * out_w + x] = color;
        }
    }
    // The high-resolution pass draws each background pixel as a block.
    if (hr) hires_->blit(hr, st, draw_state_serial_, screen_x0, screen_y0, out_w, out_h, hires_bg_px_.data());
}

bool RDP::execute_s2dex_command(u8 opcode, u32 w0, u32 w1, MicrocodeType ucode,
                                 u32& pc, std::vector<u32>& dl_stack,
                                 u8* rdram, size_t rdram_size) {
    bool gbi2 = (ucode == MicrocodeType::S2DEX2);

    enum class Op {
        None, BgCyc, BgCopy, ObjRect, ObjSprite, ObjMoveMem, SelectDL, ObjRenderMode,
        ObjRectR, ObjLoadTxtr, ObjLdTxSprite, ObjLdTxRect, ObjLdTxRectR, RdpHalf0
    };
    Op op = Op::None;

    if (gbi2) {
        switch (opcode) {
            case 0x01: op = Op::ObjRect; break;
            case 0x02: op = Op::ObjSprite; break;
            case 0x04: op = Op::SelectDL; break;
            case 0x05: op = Op::ObjLoadTxtr; break;
            case 0x06: op = Op::ObjLdTxSprite; break;
            case 0x07: op = Op::ObjLdTxRect; break;
            case 0x08: op = Op::ObjLdTxRectR; break;
            case 0x09: op = Op::BgCyc; break;
            case 0x0a: op = Op::BgCopy; break;
            case 0x0b: op = Op::ObjRenderMode; break;
            case 0xda: op = Op::ObjRectR; break;
            case 0xdc: op = Op::ObjMoveMem; break;
            case 0xe4: op = Op::RdpHalf0; break;
            default: return false;
        }
    } else {
        switch (opcode) {
            case 0x01: op = Op::BgCyc; break;
            case 0x02: op = Op::BgCopy; break;
            case 0x03: op = Op::ObjRect; break;
            case 0x04: op = Op::ObjSprite; break;
            case 0x05: op = Op::ObjMoveMem; break;
            case 0xb0: op = Op::SelectDL; break;
            case 0xb1: op = Op::ObjRenderMode; break;
            case 0xb2: op = Op::ObjRectR; break;
            case 0xc1: op = Op::ObjLoadTxtr; break;
            case 0xc2: op = Op::ObjLdTxSprite; break;
            case 0xc3: op = Op::ObjLdTxRect; break;
            case 0xc4: op = Op::ObjLdTxRectR; break;
            case 0xe4: op = Op::RdpHalf0; break;
            default: return false;
        }
    }

    switch (op) {
        case Op::BgCyc:
            s2dex_draw_bg(segment_to_physical(w1), true, rdram, rdram_size);
            return true;
        case Op::BgCopy:
            s2dex_draw_bg(segment_to_physical(w1), false, rdram, rdram_size);
            return true;
        case Op::ObjRect:
            s2dex_draw_obj_rect(segment_to_physical(w1), false, rdram, rdram_size);
            return true;
        case Op::ObjRectR:
            s2dex_draw_obj_rect(segment_to_physical(w1), true, rdram, rdram_size);
            return true;
        case Op::ObjSprite:
            s2dex_draw_obj_sprite(segment_to_physical(w1), rdram, rdram_size);
            return true;
        case Op::ObjMoveMem: {
            u32 index = (w0 >> 16) & 0xFF;
            u32 addr = segment_to_physical(w1);
            if (index == 23 && addr + 24 <= rdram_size) { // full uObjMtx
                obj2d_matrix.A = s2d_read_s32(rdram, addr + 0) / 65536.0f;
                obj2d_matrix.B = s2d_read_s32(rdram, addr + 4) / 65536.0f;
                obj2d_matrix.C = s2d_read_s32(rdram, addr + 8) / 65536.0f;
                obj2d_matrix.D = s2d_read_s32(rdram, addr + 12) / 65536.0f;
                obj2d_matrix.X = s2d_read_s16(rdram, addr + 16) / 4.0f;
                obj2d_matrix.Y = s2d_read_s16(rdram, addr + 18) / 4.0f;
                u16 bx = s2d_read_u16(rdram, addr + 20), by = s2d_read_u16(rdram, addr + 22);
                obj2d_matrix.baseScaleX = (bx ? bx : 1024) / 1024.0f;
                obj2d_matrix.baseScaleY = (by ? by : 1024) / 1024.0f;
            } else if (index == 7 && addr + 8 <= rdram_size) { // uObjSubMtx
                obj2d_matrix.X = s2d_read_s16(rdram, addr + 0) / 4.0f;
                obj2d_matrix.Y = s2d_read_s16(rdram, addr + 2) / 4.0f;
                u16 bx = s2d_read_u16(rdram, addr + 4), by = s2d_read_u16(rdram, addr + 6);
                obj2d_matrix.baseScaleX = (bx ? bx : 1024) / 1024.0f;
                obj2d_matrix.baseScaleY = (by ? by : 1024) / 1024.0f;
            }
            return true;
        }
        case Op::ObjRenderMode:
            obj_render_mode = w1;
            return true;
        case Op::ObjLoadTxtr:
            s2dex_load_txtr(segment_to_physical(w1), rdram, rdram_size);
            return true;
        case Op::ObjLdTxSprite: {
            u32 addr = segment_to_physical(w1);
            s2dex_load_txtr(addr, rdram, rdram_size);
            s2dex_draw_obj_sprite(addr + 24, rdram, rdram_size);
            return true;
        }
        case Op::ObjLdTxRect: {
            u32 addr = segment_to_physical(w1);
            s2dex_load_txtr(addr, rdram, rdram_size);
            s2dex_draw_obj_rect(addr + 24, false, rdram, rdram_size);
            return true;
        }
        case Op::ObjLdTxRectR: {
            u32 addr = segment_to_physical(w1);
            s2dex_load_txtr(addr, rdram, rdram_size);
            s2dex_draw_obj_rect(addr + 24, true, rdram, rdram_size);
            return true;
        }
        case Op::RdpHalf0:
            s2d_pending_flag = w1;
            s2d_pending_sid = (w0 >> 16) & 0xFF;
            s2d_pending_addr_lo = w0 & 0xFFFF;
            s2d_pending_valid = true;
            return true;
        case Op::SelectDL: {
            if (!s2d_pending_valid) return true; // malformed stream: ignore
            s2d_pending_valid = false;
            u32 mask = w1;
            bool push = ((w0 >> 16) & 0xFF) != 0;
            u32 addr_hi = w0 & 0xFFFF;
            u32 target = s2d_pending_addr_lo | (addr_hi << 16); // pc stays in segment space, like G_DL's target
            u32 slot = s2d_pending_sid / 4;
            if (slot >= 4) return true;

            if ((s2d_genstat[slot] & mask) == (s2d_pending_flag & mask)) {
                // Condition true: per spec, do nothing.
                return true;
            }
            s2d_genstat[slot] = (s2d_genstat[slot] & ~mask) | (s2d_pending_flag & mask);
            if (push) dl_stack.push_back(pc);
            pc = target;
            return true;
        }
        default:
            return false;
    }
}
