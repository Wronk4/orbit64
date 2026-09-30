#pragma once

#include "common.hpp"
#include "hires.hpp"
#include "raster.hpp"
#include "rdp_exact.hpp"
#include <vector>
#include <array>
#include <memory>
#include <unordered_map>
#include <cstdio>

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
    F3DGOLDEN,
    F3DPD,  // Perfect Dark: Fast3D with 12-byte vertices that index a color table
    F3DDKR, // Diddy Kong Racing: DMA-loaded matrices, 10-byte vertices, textured triangle lists
    F3DJFG, // Jet Force Gemini: F3DDKR with a different vertex count and matrix multiply
    F3DWRUS // Wave Race 64 (US): Fast3D with vertex slots x5, 32 of them, and G_QUAD at 0xB5
};

// Identifies a graphics microcode from its credit string ("RSP Gfx ucode
// F3DEX.NoN 1.23 Yoshitaka Yasumoto Nintendo.", "RSP SW Version: 2.0G ...")
// in its data segment: the first one in [p, p + len). A 2.x version means the
// GBI-2 (F3DEX2 / S2DEX2) command set. Auto when there is none.
MicrocodeType identify_ucode_banner(const u8* p, size_t len);
// Whether that first credit string is Conker's Bad Fur Day's F3DEXBG, an
// F3DEX2 with point lights, per-vertex normals kept apart and G_TRI4.
bool ucode_banner_is_cbfd(const u8* p, size_t len);
// Whether it is a ".NoN" build (F3DEX.NoN, F3DZEX.NoN, ...), which doesn't
// clip triangles against the near (or far) plane but clamps their depth.
bool ucode_banner_is_non(const u8* p, size_t len);

class MI;

class RDP {
public:
    // geometry_only: a copy that only reads display lists, for the debugger's
    // geometry capture while graphics run on the low-level RSP (see
    // capture_display_list): it never draws, and nothing it does reaches RDRAM.
    explicit RDP(bool geometry_only = false);
    ~RDP();

    void reset();

    // Internal resolution: 1 draws only into RDRAM at the game's frame buffer
    // size; 2..8 also draws everything at that many times the size (hires.hpp).
    void set_hires_scale(u32 scale);
    u32 hires_scale() const { return hires_ ? hires_->scale() : 1; }
    HiResRenderer* hires() { return hires_.get(); }
    // Runs the high-resolution pass at any scale, including 1 (for tests; see tools/rdp_check.cpp).
    void force_hires(u32 scale) { hires_.reset(); hires_ = make_hires(scale); }
    // Who draws the high-resolution pass (the GPU renderer, src/gpu/); the
    // CPU renderer without one. Takes effect with the next renderer made.
    void set_hires_factory(HiResFactory f) { hires_factory_ = std::move(f); }
    // Starts the high-resolution pass over (after a factory change).
    void recreate_hires();

    void set_ucode_type(MicrocodeType type) { ucode_type = type; }
    void set_no_near_clip(bool on) { no_near_clip_ = on; }
    void set_cbfd(bool on) {
        if (on && !cbfd_) cbfd_advanced_ = false;
        cbfd_ = on;
    }
    MicrocodeType get_ucode_type() const { return ucode_type; }

    // Frontend status queries (read-only).
    MicrocodeType get_active_ucode() const { return current_ucode_active; }
    bool has_processed_display_list() const { return display_list_count != 0; }
    u64 get_display_list_count() const { return display_list_count; }
    // Primitives sent to the rasterizer so far (tools/game_probe.cpp).
    u64 get_triangle_count() const { return triangle_count_; }
    u64 get_tex_rect_count() const { return tex_rect_count_; }
    u32 get_segment(int index) const { return segments[index & 15]; }

    // Debugger geometry capture (see CapturedMesh).
    void set_capture(bool on);
    bool capture_on() const { return capture_enabled; }
    std::vector<CapturedMesh> take_capture();
    std::vector<CapturedTexture> take_textures();
    // Physical address of the vertex buffer whose meshes should have their textures captured (0 = none).
    void set_texture_capture_target(u32 vtx_phys);
    const Matrix4x4& get_projection() const;
    // A graphics task running on the low-level RSP draws through RDP
    // commands, which carry no 3D geometry: its display list is read here
    // too, by the geometry-only copy, so the capture still sees the scene.
    // The RSP identifies the microcode on capture_shadow() first (null while
    // the capture is off).
    RDP* capture_shadow();
    void capture_display_list(u32 dl_addr, const u8* rdram, size_t rdram_size);

    // MMIO register access for DPC (0x04100000) and DPS (0x04200000)
    u32 read_dpc_reg(u32 addr) const;
    void write_dpc_reg(u32 addr, u32 val, MI& mi, u8* rdram, size_t rdram_size);
    // End of a graphics task: SP interrupt, and DP unless the RDP is frozen.
    void finish_task(MI& mi);

    u32 read_dps_reg(u32 addr) const;
    void write_dps_reg(u32 addr, u32 val);

    // The RSP's data memory, where the RDP reads its commands from when
    // DPC_STATUS says XBUS (microcodes that keep the RDP's buffer in DMEM).
    void set_rsp_dmem(const u8* dmem) { rsp_dmem_ = dmem; }
    // RDP command-buffer commands run so far (low-level graphics).
    u64 get_rdp_command_count() const { return raw_cmd_count_; }
    bool drawing_rdp_commands() const { return raw_mode_; }
    // Low-level graphics: draw the RDP's commands bit-exactly (rdp_exact.hpp)
    // instead of through the high-level renderer.
    // ORBIT64_RDP=exact or fast overrides it.
    void set_exact(bool on) { exact_mode_ = env_exact_ >= 0 ? env_exact_ != 0 : on; }
    bool exact() const { return exact_mode_; }
    // Whether the frame being shown was drawn by the bit-exact RDP.
    bool exact_drawing() const { return exact_mode_ && raw_mode_; }
    // The bit-exact RDP (its ninth bits, and its high-resolution copy at an
    // internal resolution, which set_hires_scale() sets for it too).
    const ExactRdp* exact_rdp() const { return &exact_.primary(); }
    ExactRdp* exact_rdp() { return &exact_.primary(); }

    // Process a display list starting at segmented address in RDRAM
    void process_display_list(u32 dl_addr, u8* rdram, size_t rdram_size, MI& mi);

    void clear_zbuffer();

    // TMEM access
    u8* get_tmem() { return tmem.data(); }

    // Pixel pipeline counters (debug log).
    struct PixelStats {
        u32 drawn{0}, z_fail{0}, a_fail{0};
    };

    // Save states (savestate.hpp): the microcode's and the RDP's state
    // between display lists. Only valid between them (queued draws are
    // flushed at the end of every list); state_loaded() then drops what was
    // derived from the replaced state.
    template <class S> void serialize(S& s) {
        s(dpc_start, dpc_end, dpc_current, dpc_status, dpc_clock, dpc_bufbusy, dpc_pipebusy, dpc_tmem, dp_pending_);
        s(dps_tbist, dps_test_mode, dps_buftest_addr, dps_buftest_data);
        s(segments, modelview_stack, projection_matrix, combined_matrix, combined_matrix_dirty, vertex_cache);
        s(vp_scale_x, vp_scale_y, vp_scale_z, vp_trans_x, vp_trans_y, vp_trans_z);
        s(tmem, tmem_word_dxt_zero, tiles, active_tile, timg_addr, timg_format, timg_size, timg_width);
        s(color_image_addr, color_image_format, color_image_size, color_image_width, depth_image_addr);
        s(fill_color, prim_color, env_color, blend_color, fog_color, prim_depth, prim_dz, geometry_mode,
          texture_enabled, texture_scale_s, texture_scale_t);
        s(combine_mode_w0, combine_mode_w1, combine_mode_set, other_mode_l, other_mode_h);
        s(ambient_light, lookat_x, lookat_y, lookat_set, dir_lights, num_lights);
        s(scissor_ulx, scissor_uly, scissor_lrx, scissor_lry);
        s(ucode_type, current_ucode_active, display_list_count, rdp_half1, rdp_half2, vtx_color_base, forced_mtx);
        s(dkr_mtx_offset, dkr_vtx_offset, dkr_vtx_index, dkr_mv_index, dkr_billboard, dkr_mv);
        s(s2d_genstat, obj2d_matrix, obj_render_mode, s2d_pending_flag, s2d_pending_sid, s2d_pending_addr_lo,
          s2d_pending_valid);
        s(tex_max_level, prim_min_level, prim_lod_frac, fog_mul, fog_ofs, no_near_clip_);
        s(cbfd_, cbfd_advanced_, cbfd_normal_base_, cbfd_coord_mod_, cbfd_lights_, cbfd_num_lights_);
        s.fixed(internal_zbuffer);
        // G_MTX pushes a copy of the top of the stack, which must exist.
        if (modelview_stack.empty() || modelview_stack.size() > 32) s.fail("the RDP matrix stack is invalid");
    }
    void state_loaded();
    // The command buffer's state (its own save state section, added later).
    template <class S> void serialize_commands(S& s) {
        s(raw_buf_, raw_mode_, raw_unbind_, raw_cmd_count_);
        exact_.serialize(s);
    }
    template <class S> void serialize_exact_extra(S& s) { exact_.serialize_extra(s); }
    // Draws everything queued so far (into RDRAM and the high-resolution buffers).
    void flush_pending() {
        flush_native();
        exact_.flush();
        if (hires_) hires_->flush();
    }

private:
    // ---- The RDP's own command buffer (DPC_START / DPC_END) ------------
    // What the low-level RSP (a real microcode) sends: the RDP commands
    // themselves, triangles as edge and attribute coefficients.
    const u8* rsp_dmem_{nullptr};
    std::vector<u64> raw_buf_;      // commands read but not run yet (a command can straddle two buffers)
    bool raw_mode_{false};          // drawing RDP commands, not a display list
    bool raw_unbind_{true};         // the CPU may have touched RDRAM since the last full sync
    u64 raw_cmd_count_{0};
    struct Trace {
        std::FILE* f;
        std::vector<u8> shadow; // RDRAM as the replay has it
        int fine_from{-1};
        explicit Trace(const char* path);
        ~Trace();
        void put32(u32 v);
        void before(const u8* rdram, size_t size);
        void command(const u64* cmd, u32 len);
        void after(const u8* rdram, size_t size);
    };
    std::unique_ptr<Trace> trace_;
    bool trace_checked_{false};
    bool exact_mode_{true};
    int env_exact_{-1};
    ExactRdpLanes exact_;
    void process_rdp_commands(MI& mi, u8* rdram, size_t rdram_size);
    void rdp_triangle(const u64* cmd, u32 op, u8* rdram, size_t rdram_size);
    // Shared by display lists and the command buffer, see its definition.
    bool execute_rdp_op(u8 opcode, u32 w0, u32 w1, u8* rdram, size_t rdram_size, bool raw);

    // DPC registers
    u32 dpc_start{0};
    u32 dpc_end{0};
    u32 dpc_current{0};
    u32 dpc_status{0};
    bool dp_pending_{false}; // task finished while frozen: DP interrupt on unfreeze
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
    bool geometry_only_{false};
    std::unique_ptr<RDP> capture_shadow_; // geometry-only copy (capture_display_list)
    std::unique_ptr<MI> capture_shadow_mi_; // takes its interrupts
    bool capture_from_shadow_{false};     // what the last take_capture() returned
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

        template <class S> void serialize(S& s) { s(r, g, b, dx, dy, dz); }
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
    u64 triangle_count_{0};
    u64 tex_rect_count_{0};
    u32 rdp_half1{0};
    u32 rdp_half2{0};
    // Fast3D/F3DEX gSPForceMatrix: the model-view-projection matrix arrives in
    // four 16-byte G_MOVEMEM pieces (G_MV_MATRIX_1..4), gathered here.
    std::array<u8, 64> forced_mtx{};
    u32 vtx_color_base{0}; // F3DPD: where vertex colors/normals live (set by opcode 0x07)
    // F3DDKR/F3DJFG: offsets added to matrix and vertex addresses, where the
    // next appended vertex goes, the model-view slot in use (and the slots
    // themselves), and whether vertices are billboarded around vertex 0.
    u32 dkr_mtx_offset{0};
    u32 dkr_vtx_offset{0};
    u32 dkr_vtx_index{0};
    u32 dkr_mv_index{0};
    bool dkr_billboard{false};
    std::array<Matrix4x4, 4> dkr_mv{};

    // S2DEX / S2DEX2 state -------------------------------------------------
    // The four RSP "general status" words used by G_MOVEWORD(G_MW_GENSTAT)
    // and consulted/updated by G_SELECT_DL / G_SELECT_BRANCH_DL (sid must be
    // one of 0, 4, 8, 12 -- indexed here as sid/4).
    u32 s2d_genstat[4]{};
    u8 tex_max_level{0};  // G_TEXTURE: mip-map levels past the first
    u8 prim_min_level{0}; // G_SETPRIMCOLOR: minimum level of detail (1/32 texels)
    u8 prim_lod_frac{0};  // G_SETPRIMCOLOR: PRIM_LOD_FRAC
    s16 fog_mul{0}, fog_ofs{0}; // G_MW_FOG
    bool no_near_clip_{false};  // a .NoN microcode: no near/far clipping
    static bool fog_supported(MicrocodeType u);

    // F3DEXBG (Conker's Bad Fur Day). Lights 0..n-1 (the last one
    // directional, the others point lights), then the ambient colour.
    struct CbfdLight {
        f32 r, g, b;    // 0..1
        f32 x, y, z;    // direction, normalised
        f32 px, py, pz; // position
        f32 ca;         // point light strength
    };
    bool cbfd_{false};
    bool cbfd_advanced_{false}; // its G_LOAD_UCODE turns on the second lighting mode
    u32 cbfd_normal_base_{0};
    f32 cbfd_coord_mod_[16]{};
    std::array<CbfdLight, 13> cbfd_lights_{};
    u32 cbfd_num_lights_{0};
    f32 cbfd_ldir_[13][3]{}; // their directions in model space (execute_vtx)
    bool execute_cbfd_command(u8 opcode, u32 w0, u32 w1, u8* rdram, size_t rdram_size);

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

    // Internal Z-buffer (for depth testing)
    std::vector<f32> internal_zbuffer;

    u32 segment_to_physical(u32 seg_addr) const;
    void update_combined_matrix();

    void execute_mtx(u32 w0, u32 w1, MicrocodeType ucode, const u8* rdram, size_t rdram_size);
    void execute_vtx(u32 w0, u32 w1, MicrocodeType ucode, const u8* rdram, size_t rdram_size);
    // F3DDKR/F3DJFG commands (see process_display_list()).
    void dkr_dma_matrix(u32 w0, u32 w1, MicrocodeType ucode, const u8* rdram, size_t rdram_size);
    void dkr_select_matrix(u32 index);
    void dkr_dma_triangles(u32 w0, u32 w1, u8* rdram, size_t rdram_size);
    void execute_moveword(u32 w0, u32 w1, MicrocodeType ucode);

    // Primitive rasterizers
    void compute_screen_coords(Vertex& v) const;
    void clip_and_rasterize_triangle(Vertex v0, Vertex v1, Vertex v2, u8* rdram, size_t rdram_size);
    void clip_and_rasterize_line(Vertex v0, Vertex v1, u8* rdram, size_t rdram_size);
    void rasterize_fill_rect(u32 ulx, u32 uly, u32 lrx, u32 lry, u8* rdram, size_t rdram_size);
    void rasterize_tex_rect(u32 ulx, u32 uly, u32 lrx, u32 lry, u32 tile_idx, f32 s, f32 t, f32 dsdx, f32 dtdy, bool flip, u8* rdram, size_t rdram_size);
    void rasterize_triangle(const Vertex& v0, const Vertex& v1, const Vertex& v2, u8* rdram, size_t rdram_size);
    // Draws a triangle as it is (no culling); `area` is triangle_area().
    void draw_triangle(const Vertex& v0, const Vertex& v1, const Vertex& v2, f32 area, u8* rdram, size_t rdram_size);

    struct NativeSink;
    // Draws one pixel through the current colour image's blending/depth/
    // coverage rules; the short form uses hires_shadow_ and the global counters.
    void write_pixel(const DrawState& st, u32 x, u32 y, u32 color, f32 z, u8* rdram, size_t rdram_size);
    void write_pixel(const DrawState& st, u32 x, u32 y, u32 color, f32 z, u8* rdram, size_t rdram_size,
                     u32* shadow, size_t shadow_len, PixelStats& stats, u8 shade_a);

    // Pixel pipeline state (raster.hpp) for the next draw. Display list
    // commands that can change it mark it dirty; see process_display_list().
    const DrawState& draw_state();
    DrawState draw_state_{};
    bool draw_state_dirty_{true};
    // ORBIT64_DL_TRACE=<frame>[,<count>]: print every display list command
    // of those frames to stderr (debugging).
    int dl_trace_frame_ = -1;
    int dl_trace_count_ = 1;
    int pick_x_ = 0, pick_y_ = 0, pick_frame_ = -1; // ORBIT64_PICK
    void debug_pick(u32 a, u32 b, u32 c) const;
    u64 draw_state_serial_{0};
    u64 tmem_gen_{0}; // bumped by every TMEM load

    // ---- Deferred, multi-threaded native pass --------------------------
    // Triangles and texture rectangles aren't drawn when their command is
    // processed but queued, each with a snapshot of the DrawState and TMEM it
    // saw, and drawn by flush_native(): the rows are split into bands and
    // every band - on its own thread - runs the whole queue in order for its
    // rows only. A pixel's result only depends on earlier draws to that same
    // pixel (blending, depth), so this produces exactly what drawing each
    // command immediately would. flush_native() must run before anything
    // else touches the frame buffer or depth buffer, or reads RDRAM the
    // queue may still write (see native_before_read()), and at the end of
    // every display list.
    struct NativeCmd {
        enum class Kind : u8 { Triangle, TexRect } kind;
        const DrawState* st;
        s32 y_first, y_last;     // rows the command can touch (inclusive)
        u32* shadow;             // hires_shadow_ at queue time
        size_t shadow_len;
        Vertex v[3];             // Triangle
        f32 area;
        u32 ulx, uly, lrx, lry, tile; // TexRect
        f32 s, t, dsdx, dtdy;
        bool flip;
    };
    std::vector<NativeCmd> native_queue_;
    u32 native_fb_lo_{0}, native_fb_hi_{0}; // RDRAM bytes the queue can write: [lo, hi)
    u64 native_work_{0};                    // rough pixel count of the queue, to skip threading tiny flushes
    u8* native_rdram_{nullptr};
    size_t native_rdram_size_{0};
    // Snapshots the queue points at (recycled by flush_native()).
    struct TmemSnapshot {
        std::array<u8, 4096> data;
        std::array<bool, 512> dxt;
    };
    std::vector<std::unique_ptr<DrawState>> native_states_;
    size_t native_states_used_{0};
    std::vector<std::unique_ptr<TmemSnapshot>> native_tmems_;
    size_t native_tmems_used_{0};
    DrawState* native_state_{nullptr}; // snapshot of the current draw_state_serial_/tmem_gen_, if taken
    u64 native_state_serial_{~0ull}, native_state_gen_{~0ull};
    const TmemSnapshot* native_tmem_{nullptr};
    u64 native_tmem_gen_{~0ull};
    std::unique_ptr<class RasterPool> raster_pool_;
    // The current state as a snapshot, with decoded-texel tables attached to
    // tile0 (if use0) and tile0 + 1 (if use1).
    DrawState* native_snapshot(u32 tile0, bool use0, bool use1);
    void queue_native(NativeCmd& cmd, u8* rdram, size_t rdram_size);
    void flush_native();
    // Flushes if [lo, hi) of RDRAM - about to be read - may still be written by the queue.
    void native_before_read(u64 lo, u64 hi) {
        if (!native_queue_.empty() && lo < native_fb_hi_ && hi > native_fb_lo_) flush_native();
    }

    // Decoded-texel tables (raster::TexCache): a textured draw samples a
    // flat ARGB array - filled by the first sample, through the very same
    // fetch_wrapped() - instead of decoding TMEM for every texel it reads,
    // four times per pixel when filtering. Keyed by TMEM generation and tile
    // settings; entries of older generations are recycled by flush_native().
    struct NativeTex {
        u64 gen{0}, key_a{0}, key_b{0};
        raster::TexCache cache;
        std::vector<u32> texels;
    };
    std::vector<std::unique_ptr<NativeTex>> native_tex_;
    size_t native_tex_used_{0};
    raster::TexCache* native_tex_cache(const raster::TexUnit& tu, u32 tlut_type);

    // Internal-resolution pass (nullptr at native resolution).
    std::unique_ptr<HiResRenderer> hires_;
    HiResFactory hires_factory_;
    std::unique_ptr<HiResRenderer> make_hires(u32 scale) const;
    // High-resolution buffer of the current colour image, or nullptr. Its
    // shadow receives every RDRAM pixel write_pixel() makes.
    HiResTarget* hires_target(u8* rdram, size_t rdram_size);
    u32* hires_shadow_{nullptr};
    size_t hires_shadow_len_{0};
    std::vector<HiResRenderer::Pixel> hires_line_px_;
    std::vector<u32> hires_bg_px_;
    // s2dex_draw_bg(): source column and horizontal filter weight per output column.
    struct BgColumn {
        s32 ix, ix2;
        f32 frac;
    };
    std::vector<BgColumn> bg_columns_;
};
