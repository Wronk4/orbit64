#pragma once

#include "common.hpp"
#include "hires.hpp"
#include "raster.hpp"
#include <vector>
#include <array>
#include <memory>
#include <unordered_map>

struct Matrix4x4 {
    float m[4][4]{};

    static Matrix4x4 identity();
    static Matrix4x4 multiply(const Matrix4x4& a, const Matrix4x4& b);
    void transform_point(f32 x, f32 y, f32 z, f32& ox, f32& oy, f32& oz, f32& ow) const;
};

// Geometry captured for the debugger: one entry per run of triangles drawn with
// the same model-view matrix. Positions are the raw object-space vertices the
// game submitted, so any game using an F3D-family microcode is supported.
struct CapturedMesh {
    Matrix4x4 modelview{};
    u32 mtx_addr{0};   // RDRAM (physical) address of the last model-view matrix loaded
    u32 vtx_addr{0};   // RDRAM (physical) address of the first vertex buffer used
    u32 dl_addr{0};    // RDRAM (physical) address of the display list being executed
    std::vector<f32> pos; // 9 floats per triangle (object space)
    std::vector<u32> col; // RGBA8 per triangle (shaded vertex colour of the first vertex)
    // Textures are only captured for the mesh selected in the debugger (see
    // set_texture_capture_target); otherwise these stay empty.
    std::vector<s32> tex; // per triangle: index into the frame's CapturedTexture list, -1 = untextured
    std::vector<f32> uv;  // 6 floats per triangle: texel coordinates relative to the texture origin
};

// A texture as the RDP samples it for one run of triangles: the active tile
// decoded from TMEM over the texel range those triangles use (wrap, mirror
// and clamp already applied, exactly like the rasterizer).
struct CapturedTexture {
    u64 key{0};                 // identifies tile state + TMEM contents + texel range
    u32 width{0}, height{0};    // image size in pixels
    f32 origin_s{0}, origin_t{0};
    f32 span_s{1}, span_t{1};   // texels covered by the image (> size when a large range was downsampled)
    std::vector<u32> argb;
};

enum class MicrocodeType {
    Auto,
    Fast3D,
    F3DEX,
    F3DEX2,
    S2DEX,
    S2DEX2,
    F3DGOLDEN
};


class MI;

class RDP {
public:
    RDP();
    ~RDP();

    void reset();

    // Internal resolution: 1 draws only into RDRAM at the game's frame buffer
    // size; 2..8 also draws everything at that many times the size (hires.hpp).
    void set_hires_scale(u32 scale);
    u32 hires_scale() const { return hires_ ? hires_->scale() : 1; }
    HiResRenderer* hires() { return hires_.get(); }
    // Runs the high-resolution pass at any scale, including 1 (for tests; see tools/rdp_check.cpp).
    void force_hires(u32 scale) { hires_.reset(); hires_ = std::make_unique<HiResRenderer>(scale); }

    void set_ucode_type(MicrocodeType type) { ucode_type = type; }
    MicrocodeType get_ucode_type() const { return ucode_type; }

    // Frontend status queries (read-only).
    MicrocodeType get_active_ucode() const { return current_ucode_active; }
    bool has_processed_display_list() const { return display_list_count != 0; }
    u64 get_display_list_count() const { return display_list_count; }
    u32 get_segment(int index) const { return segments[index & 15]; }

    // Debugger geometry capture (see CapturedMesh).
    void set_capture(bool on) { capture_enabled = on; if (!on) capture_frame.clear(); }
    std::vector<CapturedMesh> take_capture() {
        finish_texture_run();
        std::vector<CapturedMesh> out;
        out.swap(capture_frame);
        capture_tris = 0;
        return out;
    }
    std::vector<CapturedTexture> take_textures() { std::vector<CapturedTexture> out; out.swap(capture_textures); return out; }
    // Physical address of the vertex buffer whose meshes should have their textures captured (0 = none).
    void set_texture_capture_target(u32 vtx_phys) { capture_texture_vtx = vtx_phys; }
    const Matrix4x4& get_projection() const { return projection_matrix; }

    // MMIO register access for DPC (0x04100000) and DPS (0x04200000)
    u32 read_dpc_reg(u32 addr) const;
    void write_dpc_reg(u32 addr, u32 val, MI& mi, u8* rdram, size_t rdram_size);

    u32 read_dps_reg(u32 addr) const;
    void write_dps_reg(u32 addr, u32 val);

    // Process a display list starting at segmented address in RDRAM
    void process_display_list(u32 dl_addr, u8* rdram, size_t rdram_size, MI& mi);

    void clear_zbuffer();

    // TMEM access
    u8* get_tmem() { return tmem.data(); }

private:
    // DPC registers
    u32 dpc_start{0};
    u32 dpc_end{0};
    u32 dpc_current{0};
    u32 dpc_status{0};
    u32 dpc_clock{0};
    u32 dpc_bufbusy{0};
    u32 dpc_pipebusy{0};
    u32 dpc_tmem{0};

    // DPS registers
    u32 dps_tbist{0};
    u32 dps_test_mode{0};
    u32 dps_buftest_addr{0};
    u32 dps_buftest_data{0};

    // Display List State
    std::array<u32, 16> segments{};
    std::vector<Matrix4x4> modelview_stack;
    Matrix4x4 projection_matrix{};
    Matrix4x4 combined_matrix{};
    bool combined_matrix_dirty{true};

    // Vertex cache
    std::array<Vertex, 80> vertex_cache{};

    // Debugger capture state
    struct RawVertex { f32 x{0}, y{0}, z{0}; u32 src{0}; };
    std::array<RawVertex, 80> raw_vertex{};
    bool capture_enabled{false};
    std::vector<CapturedMesh> capture_frame;
    u32 capture_mtx_addr{0};
    u32 capture_dl_addr{0};
    size_t capture_tris{0};
    void emit_triangle(u32 a, u32 b, u32 c, u8* rdram, size_t rdram_size);

    // Texture capture (debugger)
    u32 capture_texture_vtx{0};
    std::vector<CapturedTexture> capture_textures;
    std::unordered_map<u64, CapturedTexture> texture_decode_cache; // decoded images reused across frames
    struct TextureRun {
        bool active{false};
        u64 key{0};
        s32 index{-1};
        Tile tile{};
        u32 tlut{0};
        std::array<u8, 4096> tmem{};
        std::array<bool, 512> dxt{};
        f32 min_s{0}, max_s{0}, min_t{0}, max_t{0};
    } tex_run;
    void finish_texture_run();
    // Set by every TMEM load so the capture only re-hashes TMEM when it changed.
    bool tmem_dirty{true};
    Tile tex_last_tile{};
    u32 tex_last_tlut{~0u};
    u64 tex_last_key{0};

    // Viewport
    f32 vp_scale_x{160.0f}, vp_scale_y{120.0f}, vp_scale_z{511.5f};
    f32 vp_trans_x{160.0f}, vp_trans_y{120.0f}, vp_trans_z{511.5f};

    // TMEM (4KB)
    std::array<u8, 4096> tmem{};
    std::array<bool, 512> tmem_word_dxt_zero{};
    std::array<Tile, 8> tiles{};
    u32 active_tile{0};

    // Texture image settings
    u32 timg_addr{0};
    u8 timg_format{0};
    u8 timg_size{0};
    u16 timg_width{0};

    // Framebuffer & Depthbuffer settings
    u32 color_image_addr{0};
    u8 color_image_format{0};
    u8 color_image_size{2}; // 16-bit default
    u16 color_image_width{320};

    u32 depth_image_addr{0};

    // Render colors & modes
    u32 fill_color{0};
    u32 prim_color{0xFFFFFFFF};
    u32 env_color{0xFFFFFFFF};
    u32 blend_color{0};
    u32 fog_color{0};
    u32 prim_depth{0};
    u32 prim_dz{0};
    u32 geometry_mode{0};
    bool texture_enabled{true};
    f32 texture_scale_s{1.0f};
    f32 texture_scale_t{1.0f};

    // Combiner state
    u32 combine_mode_w0{0};
    u32 combine_mode_w1{0};
    bool combine_mode_set{false};

    // Other modes (render mode, z-compare, alpha compare, cycle type)
    u32 other_mode_l{0x00000030}; // Default Z_CMP | Z_UPD
    u32 other_mode_h{0};

    // Lighting state
    struct Light {
        u8 r{255}, g{255}, b{255};
        f32 dx{0.0f}, dy{1.0f}, dz{0.0f};
    };
    Light ambient_light{128, 128, 128, 0, 0, 0};
    Light lookat_x{0, 0, 0, 1.0f, 0.0f, 0.0f};
    Light lookat_y{0, 0, 0, 0.0f, 1.0f, 0.0f};
    bool lookat_set{false};
    std::vector<Light> dir_lights;
    u32 num_lights{0};

    // Scissor clipping
    u32 scissor_ulx{0};
    u32 scissor_uly{0};
    u32 scissor_lrx{320};
    u32 scissor_lry{240};

    // Microcode type & helper registers
    MicrocodeType ucode_type{MicrocodeType::Auto};
    MicrocodeType current_ucode_active{MicrocodeType::Fast3D};
    u64 display_list_count{0};
    u32 rdp_half1{0};
    u32 rdp_half2{0};

    // S2DEX / S2DEX2 state -------------------------------------------------
    // The four RSP "general status" words used by G_MOVEWORD(G_MW_GENSTAT)
    // and consulted/updated by G_SELECT_DL / G_SELECT_BRANCH_DL (sid must be
    // one of 0, 4, 8, 12 -- indexed here as sid/4).
    u32 s2d_genstat[4]{};

    // The single 2D transform matrix (no stack, no push/pop) used by
    // G_OBJ_SPRITE (full A,B,C,D,X,Y) and G_OBJ_RECTANGLE_R (X,Y,BaseScale only).
    struct Obj2DMatrix {
        f32 A{1.0f}, B{0.0f}, C{0.0f}, D{1.0f};
        f32 X{0.0f}, Y{0.0f};
        f32 baseScaleX{1.0f}, baseScaleY{1.0f};
    } obj2d_matrix{};

    u32 obj_render_mode{0};

    // G_SELECT_DL is split across two consecutive commands: G_RDPHALF_0
    // stashes flag/sid/low-address, then G_SELECT_DL supplies mask/push/high-address.
    u32 s2d_pending_flag{0};
    u32 s2d_pending_sid{0};
    u32 s2d_pending_addr_lo{0};
    bool s2d_pending_valid{false};

    bool is_s2dex_ucode(MicrocodeType t) const {
        return t == MicrocodeType::S2DEX || t == MicrocodeType::S2DEX2;
    }

    // Returns true if the opcode was fully handled (S2DEX/S2DEX2-specific or
    // overlaps numerically with an F3D-family opcode reused for a different
    // purpose); false lets the caller fall through to the shared RDP-command
    // switch (G_DL, G_ENDDL, scissor, image/tile setup, sync, etc.).
    bool execute_s2dex_command(u8 opcode, u32 w0, u32 w1, MicrocodeType ucode,
                                u32& pc, std::vector<u32>& dl_stack,
                                u8* rdram, size_t rdram_size);

    struct S2DObjSpriteInfo {
        f32 objX{0}, objY{0};     // s10.2 -> pixels
        f32 scaleW{1}, scaleH{1}; // u5.10 -> ratio (1.0 = no scale)
        f32 imageW{0}, imageH{0}; // u10.5 -> texels
        u8 imageFlags{0};
    };
    S2DObjSpriteInfo s2dex_setup_obj_tile(u32 sp_addr, const u8* rdram, size_t rdram_size);
    void s2dex_draw_obj_rect(u32 sp_addr, bool use_matrix, u8* rdram, size_t rdram_size);
    void s2dex_draw_obj_sprite(u32 sp_addr, u8* rdram, size_t rdram_size);
    void s2dex_draw_bg(u32 bg_addr, bool scaled, u8* rdram, size_t rdram_size);
    void s2dex_load_txtr(u32 tx_addr, u8* rdram, size_t rdram_size);
    void s2dex_tmem_load_block(u32 tmem_dest_words, u32 src_addr, u32 lrs, u8* rdram, size_t rdram_size);
    void s2dex_tmem_load_tile(u32 tmem_dest_words, u32 src_addr, u32 texel_w, u32 texel_h, u8* rdram, size_t rdram_size);

    // Internal Z-buffer (for depth testing)
    std::vector<f32> internal_zbuffer;

    u32 segment_to_physical(u32 seg_addr) const;
    void update_combined_matrix();

    void execute_mtx(u32 w0, u32 w1, MicrocodeType ucode, const u8* rdram, size_t rdram_size);
    void execute_vtx(u32 w0, u32 w1, MicrocodeType ucode, const u8* rdram, size_t rdram_size);
    void execute_moveword(u32 w0, u32 w1, MicrocodeType ucode);

    // Primitive rasterizers
    void compute_screen_coords(Vertex& v) const;
    void clip_and_rasterize_triangle(Vertex v0, Vertex v1, Vertex v2, u8* rdram, size_t rdram_size);
    void clip_and_rasterize_line(Vertex v0, Vertex v1, u8* rdram, size_t rdram_size);
    void rasterize_fill_rect(u32 ulx, u32 uly, u32 lrx, u32 lry, u8* rdram, size_t rdram_size);
    void rasterize_tex_rect(u32 ulx, u32 uly, u32 lrx, u32 lry, u32 tile_idx, f32 s, f32 t, f32 dsdx, f32 dtdy, bool flip, u8* rdram, size_t rdram_size);
    void rasterize_triangle(const Vertex& v0, const Vertex& v1, const Vertex& v2, u8* rdram, size_t rdram_size);

    struct NativeSink;
    void write_pixel(const DrawState& st, u32 x, u32 y, u32 color, f32 z, u8* rdram, size_t rdram_size);

    // Pixel pipeline state (raster.hpp) for the next draw. Display list
    // commands that can change it mark it dirty; see process_display_list().
    const DrawState& draw_state();
    DrawState draw_state_{};
    bool draw_state_dirty_{true};
    u64 draw_state_serial_{0};
    u64 tmem_gen_{0}; // bumped by every TMEM load

    // Internal-resolution pass (nullptr at native resolution).
    std::unique_ptr<HiResRenderer> hires_;
    // High-resolution buffer of the current colour image, or nullptr. Its
    // shadow receives every RDRAM pixel write_pixel() makes.
    HiResTarget* hires_target(u8* rdram, size_t rdram_size);
    u32* hires_shadow_{nullptr};
    size_t hires_shadow_len_{0};
    std::vector<HiResRenderer::Pixel> hires_line_px_;
    std::vector<u32> hires_bg_px_;
};
