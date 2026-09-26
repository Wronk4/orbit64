// Object Viewer, 3D Object Inspector and Player Viewer.
//
// Objects are the geometry groups the game draws each frame: every run of
// triangles submitted with one model-view matrix. Because this is read from
// the graphics microcode rather than from game-specific structures, it works
// in any game that renders through an F3D-family microcode.
//
// World space is estimated by treating the largest mesh of the frame (almost
// always the level / stage) as drawn at the world origin, so its model-view
// matrix is the camera's view matrix: world = modelview * inverse(view).

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "app.hpp"
#include "platform.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <unordered_map>

namespace ui {

namespace {

constexpr float kPi = 3.14159265358979f;

using Mat = float[4][4];

bool invert(const Mat& m, Mat& out) {
    // Gauss-Jordan elimination on [m | I].
    float a[4][8];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 8; ++c) a[r][c] = c < 4 ? m[r][c] : (c - 4 == r ? 1.0f : 0.0f);
    for (int c = 0; c < 4; ++c) {
        int piv = c;
        for (int r = c + 1; r < 4; ++r) if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
        if (std::fabs(a[piv][c]) < 1e-9f) return false;
        if (piv != c) for (int k = 0; k < 8; ++k) std::swap(a[c][k], a[piv][k]);
        float inv = 1.0f / a[c][c];
        for (int k = 0; k < 8; ++k) a[c][k] *= inv;
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            float f = a[r][c];
            for (int k = 0; k < 8; ++k) a[r][k] -= f * a[c][k];
        }
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) out[r][c] = a[r][c + 4];
    return true;
}

void multiply(const Mat& a, const Mat& b, Mat& out) {
    Mat t;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            t[i][j] = 0;
            for (int k = 0; k < 4; ++k) t[i][j] += a[i][k] * b[k][j];
        }
    std::memcpy(out, t, sizeof t);
}

// Row-vector convention (v' = v * M), as used by the N64 microcodes.
void decompose(const Mat& m, float pos[3], float rot_deg[3], float scale[3]) {
    for (int i = 0; i < 3; ++i) {
        pos[i] = m[3][i];
        scale[i] = std::sqrt(m[i][0] * m[i][0] + m[i][1] * m[i][1] + m[i][2] * m[i][2]);
    }
    float r[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r[i][j] = scale[i] > 1e-6f ? m[i][j] / scale[i] : 0.0f;
    float pitch = std::asin(std::clamp(-r[2][1], -1.0f, 1.0f));
    float yaw = std::atan2(r[2][0], r[2][2]);
    float roll = std::atan2(r[0][1], r[1][1]);
    rot_deg[0] = pitch * 180.0f / kPi;
    rot_deg[1] = yaw * 180.0f / kPi;
    rot_deg[2] = roll * 180.0f / kPi;
}

std::string hex32(std::uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof b, "%08X", v);
    return b;
}

// Physical capture addresses shown as CPU (KSEG0) addresses, like the memory tools.
std::uint32_t kseg0(std::uint32_t phys) { return phys ? (0x80000000u | phys) : 0; }

const ImVec4 kAxisCol[3] = {ImVec4(0.95f, 0.36f, 0.38f, 1), ImVec4(0.40f, 0.85f, 0.45f, 1), ImVec4(0.38f, 0.60f, 1.0f, 1)};

void vec3_row(const char* label, const float v[3], const char* fmt, float w) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(p.x, p.y + dp(3)), col(g_pal.text_faint), label);
    float lw = dp(64);
    float cw = (w - lw) / 3;
    for (int i = 0; i < 3; ++i) {
        float x = p.x + lw + i * cw;
        dl->AddRectFilled(ImVec2(x, p.y + dp(4)), ImVec2(x + dp(3), p.y + dp(16)), col(kAxisCol[i]), dp(1));
        char b[32];
        std::snprintf(b, sizeof b, fmt, v[i]);
        text_ellipsis(dl, g_fonts.mono_small, ImVec2(x + dp(8), p.y + dp(3)), cw - dp(10), col(g_pal.text), b);
    }
    ImGui::Dummy(ImVec2(w, dp(22)));
}

void kv_row(const char* k, const std::string& v, float w, const ImVec4* vc = nullptr) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(p.x, p.y + dp(2)), col(g_pal.text_faint), k);
    ImVec2 ts = g_fonts.mono_small->CalcTextSizeA(font_px(g_fonts.mono_small), FLT_MAX, 0, v.c_str());
    dl->AddText(g_fonts.mono_small, font_px(g_fonts.mono_small), ImVec2(p.x + w - ts.x, p.y + dp(3)), col(vc ? *vc : g_pal.text), v.c_str());
    ImGui::Dummy(ImVec2(w, dp(20)));
}

void empty_state(Icon icon, const char* title, const char* text) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float cy = p.y + std::max(dp(50), avail.y * 0.30f);
    ImVec2 c(p.x + avail.x * 0.5f, cy);
    dl->AddCircleFilled(c, dp(28), col(g_pal.bg3), 32);
    draw_icon(dl, icon, c, dp(24), col(g_pal.text_faint));
    ImVec2 ts = g_fonts.body_bold->CalcTextSizeA(font_px(g_fonts.body_bold), FLT_MAX, 0, title);
    dl->AddText(g_fonts.body_bold, font_px(g_fonts.body_bold), ImVec2(c.x - ts.x * 0.5f, cy + dp(40)), col(g_pal.text), title);
    ImVec2 ts2 = g_fonts.small->CalcTextSizeA(font_px(g_fonts.small), avail.x - dp(40), avail.x - dp(40), text);
    dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(c.x - ts2.x * 0.5f, cy + dp(64)), col(g_pal.text_dim), text, nullptr,
                avail.x - dp(40));
    ImGui::SetCursorScreenPos(ImVec2(p.x, cy + dp(74) + ts2.y + dp(14)));
    ImGui::Dummy(ImVec2(0, 0)); // validates the cursor move
}

} // namespace

// ---------------------------------------------------------------------------
// Scene analysis
// ---------------------------------------------------------------------------

void App::compute_scene_objects() {
    dbg_.objects.clear();
    if (!dbg_.snap) return;
    dbg_.objects_frame = dbg_.snap->frame;
    if (!dbg_.snap->meshes || dbg_.snap->meshes->empty()) return;
    const auto& meshes = *dbg_.snap->meshes;

    // Reference (view) matrix: the mesh with the most triangles.
    size_t ref = 0;
    for (size_t i = 1; i < meshes.size(); ++i)
        if (meshes[i].col.size() > meshes[ref].col.size()) ref = i;
    Mat inv_view;
    bool world = dbg_.world_space && invert(meshes[ref].modelview.m, inv_view);
    float cam_pos[3] = {0, 0, 0};
    if (world) for (int i = 0; i < 3; ++i) cam_pos[i] = inv_view[3][i];

    std::unordered_map<std::uint32_t, int> instances;
    dbg_.objects.reserve(meshes.size());
    for (size_t i = 0; i < meshes.size(); ++i) {
        const CapturedMesh& m = meshes[i];
        SceneObject o;
        o.index = static_cast<int>(i);
        o.mtx_addr = kseg0(m.mtx_addr);
        o.vtx_addr = kseg0(m.vtx_addr);
        o.dl_addr = kseg0(m.dl_addr);
        o.triangles = static_cast<int>(m.col.size());
        o.reference = i == ref;
        o.instance = instances[o.vtx_addr]++;
        if (world) multiply(m.modelview.m, inv_view, o.world);
        else std::memcpy(o.world, m.modelview.m, sizeof o.world);
        decompose(o.world, o.pos, o.rot, o.scale);
        float d2 = 0;
        for (int k = 0; k < 3; ++k) d2 += (o.pos[k] - cam_pos[k]) * (o.pos[k] - cam_pos[k]);
        o.distance = std::sqrt(d2);
        dbg_.objects.push_back(o);
    }

    // Player tracking: derive velocity from the tracked object's movement.
    if (dbg_.player_source == 0 && dbg_.player.valid) {
        if (const SceneObject* p = find_object(dbg_.player)) {
            std::uint64_t f = dbg_.snap->frame;
            if (dbg_.player_found && f > dbg_.player_frame) {
                float dt = static_cast<float>(f - dbg_.player_frame);
                for (int k = 0; k < 3; ++k) dbg_.player_vel[k] = (p->pos[k] - dbg_.player_pos[k]) / dt;
            }
            std::copy(p->pos, p->pos + 3, dbg_.player_pos);
            dbg_.player_frame = f;
            dbg_.player_found = true;
            dbg_.player.instance = p->instance;
            dbg_.player_trail.push_back({p->pos[0], p->pos[1], p->pos[2]});
            while (dbg_.player_trail.size() > 240) dbg_.player_trail.pop_front();
        } else {
            dbg_.player_found = false;
        }
    }
    // Remember where the selected object is so it can be followed if draw order changes.
    if (const SceneObject* s = find_object(dbg_.selected)) {
        std::copy(s->pos, s->pos + 3, dbg_.selected_last_pos);
        dbg_.selected.instance = s->instance;
    }
}

const SceneObject* App::find_object(const ObjectKey& key) const {
    if (!key.valid) return nullptr;
    const SceneObject* best = nullptr;
    int candidates = 0;
    for (const auto& o : dbg_.objects) {
        if (o.vtx_addr != key.vtx_addr) continue;
        candidates++;
        if (o.instance == key.instance) best = &o;
    }
    if (candidates > 1) {
        // Several instances of one model (e.g. coins): follow the one closest to its last position.
        const float* last = &key == &dbg_.player ? dbg_.player_pos : dbg_.selected_last_pos;
        float best_d = FLT_MAX;
        for (const auto& o : dbg_.objects) {
            if (o.vtx_addr != key.vtx_addr) continue;
            float d = 0;
            for (int k = 0; k < 3; ++k) d += (o.pos[k] - last[k]) * (o.pos[k] - last[k]);
            if (d < best_d) { best_d = d; best = &o; }
        }
    }
    return best;
}

SDL_Texture* App::debug_texture(const CapturedTexture& t) {
    if (t.width == 0 || t.height == 0 || t.argb.empty()) return nullptr;
    const std::uint64_t h = t.key;
    if (auto it = dbg_textures_.find(h); it != dbg_textures_.end()) return it->second;
    SDL_Texture* tex = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC,
                                         static_cast<int>(t.width), static_cast<int>(t.height));
    if (!tex) return nullptr;
    SDL_UpdateTexture(tex, nullptr, t.argb.data(), static_cast<int>(t.width) * 4);
    SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND);
    SDL_SetTextureScaleMode(tex, settings_.filter == 0 ? SDL_SCALEMODE_NEAREST : SDL_SCALEMODE_LINEAR);
    dbg_textures_[h] = tex;
    return tex;
}

void App::select_object(const SceneObject& o, bool open_inspector) {
    bool changed = !dbg_.selected.valid || dbg_.selected.vtx_addr != o.vtx_addr || dbg_.selected.instance != o.instance;
    dbg_.selected = {o.vtx_addr, o.instance, true};
    std::copy(o.pos, o.pos + 3, dbg_.selected_last_pos);
    if (changed) {
        dbg_.cam.fitted = false;
        dbg_.mesh_pos.clear();
        dbg_.mesh_col.clear();
        dbg_.mesh_uv.clear();
        dbg_.mesh_tex.clear();
        dbg_.mesh_texture_count = 0;
    }
    if (open_inspector) open_tool(DebugTool::ObjectInspector);
}

// ---------------------------------------------------------------------------
// Object Viewer
// ---------------------------------------------------------------------------

void App::tool_object_viewer() {
    const float w = ImGui::GetContentRegionAvail().x;
    if (!dbg_.snap) {
        spinner(dp(10), dp(2), col(g_pal.accent));
        return;
    }
    // Header: summary, coordinate space, filter
    const float row_x0 = ImGui::GetCursorPosX();
    size_t tris = 0;
    for (const auto& o : dbg_.objects) tris += o.triangles;
    ImGui::PushFont(g_fonts.body_bold);
    ImGui::Text("%zu objects", dbg_.objects.size());
    ImGui::PopFont();
    ImGui::SameLine(0, dp(8));
    ImGui::PushFont(g_fonts.small);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(2));
    ImGui::TextColored(g_pal.text_faint, "%zu triangles \xC2\xB7 frame %llu", tris, static_cast<unsigned long long>(dbg_.snap->frame));
    ImGui::PopFont();

    const char* spaces[] = {"World", "Camera"};
    int space = dbg_.world_space ? 0 : 1;
    float right = row_x0 + w;
    float filter_w = std::min(dp(170), w * 0.3f);
    ImGui::SameLine(right - dp(170) - filter_w - dp(8));
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - dp(4));
    if (segmented("space", spaces, 2, &space, dp(170))) {
        dbg_.world_space = space == 0;
        compute_scene_objects();
    }
    if (ImGui::IsItemHovered())
        tooltip("World: positions relative to the level (the largest mesh is treated as drawn at the origin).\n"
                "Camera: positions exactly as submitted to the RSP, relative to the camera.");
    ImGui::SameLine(0, dp(8));
    search_field("objfilter", dbg_.object_filter, sizeof dbg_.object_filter, "Address\xE2\x80\xA6", filter_w);
    ImGui::Dummy(dp(0, 2));

    if (dbg_.objects.empty()) {
        empty_state(Icon::Layers, "No 3D geometry this frame",
                    "Objects appear here when the game draws triangles through the graphics microcode.");
        return;
    }

    // Detail strip for the selection
    const SceneObject* sel = find_object(dbg_.selected);
    const float detail_h = sel ? dp(86) : 0.0f;

    ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Hideable;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, dp(8, 5));
    if (ImGui::BeginTable("##objects", 8, flags, ImVec2(0, -detail_h))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, dp(34));
        ImGui::TableSetupColumn("Object", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Tris", ImGuiTableColumnFlags_WidthFixed, dp(50));
        ImGui::TableSetupColumn("X", ImGuiTableColumnFlags_WidthFixed, dp(70));
        ImGui::TableSetupColumn("Y", ImGuiTableColumnFlags_WidthFixed, dp(70));
        ImGui::TableSetupColumn("Z", ImGuiTableColumnFlags_WidthFixed, dp(70));
        ImGui::TableSetupColumn("Yaw", ImGuiTableColumnFlags_WidthFixed, dp(56));
        ImGui::TableSetupColumn("Scale", ImGuiTableColumnFlags_WidthFixed, dp(56));
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::PushStyleColor(ImGuiCol_Text, g_pal.text_faint);
        ImGui::TableHeadersRow();
        ImGui::PopStyleColor();
        ImGui::PopFont();

        std::string filt = dbg_.object_filter;
        std::transform(filt.begin(), filt.end(), filt.begin(), [](unsigned char c) { return std::toupper(c); });
        if (filt.rfind("0X", 0) == 0) filt = filt.substr(2);
        std::vector<int> rows;
        for (const auto& o : dbg_.objects) {
            if (!filt.empty() && hex32(o.vtx_addr).find(filt) == std::string::npos && hex32(o.mtx_addr).find(filt) == std::string::npos)
                continue;
            rows.push_back(o.index);
        }
        ImGuiListClipper clip;
        clip.Begin(static_cast<int>(rows.size()));
        while (clip.Step()) {
            for (int r = clip.DisplayStart; r < clip.DisplayEnd; ++r) {
                const SceneObject& o = dbg_.objects[rows[r]];
                bool is_sel = sel && sel->index == o.index;
                bool is_player = dbg_.player.valid && dbg_.player.vtx_addr == o.vtx_addr && dbg_.player.instance == o.instance;
                ImGui::TableNextRow(0, dp(26));
                ImGui::TableNextColumn();
                ImGui::PushID(o.index);
                if (ImGui::Selectable("##row", is_sel, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap,
                                      ImVec2(0, dp(18))))
                    select_object(o, true);
                if (ImGui::IsItemHovered()) tooltip("Click to open in the 3D Object Inspector");
                ImGui::PopID();
                ImGui::SameLine(0, 0);
                ImGui::PushFont(g_fonts.mono_small);
                ImGui::TextColored(g_pal.text_faint, "%d", o.index);
                ImGui::PopFont();
                ImGui::TableNextColumn();
                ImGui::PushFont(g_fonts.mono_small);
                ImGui::TextColored(is_sel ? g_pal.accent_hover : g_pal.text, "%s", hex32(o.vtx_addr).c_str());
                ImGui::PopFont();
                if (o.reference) {
                    ImGui::SameLine(0, dp(6));
                    badge("LEVEL", g_pal.info);
                    if (ImGui::IsItemHovered()) tooltip("Largest mesh of the frame, used as the world reference");
                }
                if (is_player) {
                    ImGui::SameLine(0, dp(6));
                    badge("PLAYER", g_pal.success);
                }
                auto num = [&](const char* fmt, float v, const ImVec4& c) {
                    ImGui::TableNextColumn();
                    ImGui::PushFont(g_fonts.mono_small);
                    ImGui::TextColored(c, fmt, v);
                    ImGui::PopFont();
                };
                ImGui::TableNextColumn();
                ImGui::PushFont(g_fonts.mono_small);
                ImGui::TextColored(g_pal.text_dim, "%d", o.triangles);
                ImGui::PopFont();
                num("%.1f", o.pos[0], g_pal.text);
                num("%.1f", o.pos[1], g_pal.text);
                num("%.1f", o.pos[2], g_pal.text);
                num("%.0f\xC2\xB0", o.rot[1], g_pal.text_dim);
                num("%.2f", (o.scale[0] + o.scale[1] + o.scale[2]) / 3.0f, g_pal.text_dim);
            }
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();

    if (sel) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(ImVec2(p.x, p.y + dp(6)), ImVec2(p.x + w, p.y + detail_h), col(g_pal.bg2), dp(8));
        ImGui::SetCursorScreenPos(ImVec2(p.x + dp(12), p.y + dp(14)));
        ImGui::BeginGroup();
        ImGui::PushFont(g_fonts.body_bold);
        ImGui::Text("Object %d", sel->index);
        ImGui::PopFont();
        ImGui::SameLine(0, dp(10));
        ImGui::PushFont(g_fonts.mono_small);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(2));
        ImGui::TextColored(g_pal.text_dim, "vtx %s  mtx %s  dl %s", hex32(sel->vtx_addr).c_str(), hex32(sel->mtx_addr).c_str(),
                           hex32(sel->dl_addr).c_str());
        ImGui::PopFont();
        if (button("3D Inspector", Icon::Cube, ButtonKind::Primary, 0, true, dp(32))) open_tool(DebugTool::ObjectInspector);
        ImGui::SameLine(0, dp(8));
        if (button("Track as Player", Icon::User, ButtonKind::Subtle, 0, true, dp(32))) {
            dbg_.player = dbg_.selected;
            dbg_.player_source = 0;
            dbg_.player_found = false;
            dbg_.player_trail.clear();
            open_tool(DebugTool::PlayerViewer);
        }
        ImGui::SameLine(0, dp(8));
        if (button("Matrix in Memory", Icon::Pencil, ButtonKind::Ghost, 0, sel->mtx_addr != 0, dp(32)))
            open_in_editor(sel->mtx_addr, MemType::S16);
        ImGui::EndGroup();
    }
}

// ---------------------------------------------------------------------------
// OBJ export
// ---------------------------------------------------------------------------

// Saves the currently cached mesh (dbg_.mesh_pos / col / uv / tex) together
// with its textures to an OBJ + MTL pair.  Textures are written as BMP files
// next to the OBJ (SDL_SaveBMP is always available without extra libraries).
// Returns true on success, false on any I/O error.
bool App::export_mesh_obj(const std::filesystem::path& obj_path) {
    if (dbg_.mesh_pos.empty()) return false;

    namespace fs = std::filesystem;
    const fs::path dir  = obj_path.parent_path();
    const fs::path stem = obj_path.stem();
    const fs::path mtl_path = dir / (stem.string() + ".mtl");

    // Create output directory if it doesn't exist.
    {
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) return false;
    }

    const size_t ntri = dbg_.mesh_col.size();
    const bool   has_tex = dbg_.mesh_tex.size() == ntri && dbg_.mesh_uv.size() == ntri * 6;

    // Sanity: mesh_pos must contain exactly 9 floats per triangle (3 verts × 3 coords).
    if (dbg_.mesh_pos.size() != ntri * 9) return false;

    // ---- Collect unique textures and assign names --------------------------
    // Map SDL_Texture* -> material name + BMP filename.
    // File names are prefixed with the OBJ stem so exports of different
    // objects into the same folder don't clobber each other's textures.
    std::unordered_map<void*, std::string> tex_to_mat;
    std::vector<std::pair<void*, std::string>> tex_list; // ordered
    {
        int idx = 0;
        for (size_t t = 0; t < ntri; ++t) {
            void* ptr = has_tex ? dbg_.mesh_tex[t] : nullptr;
            if (ptr && tex_to_mat.find(ptr) == tex_to_mat.end()) {
                char name[64];
                std::snprintf(name, sizeof name, "%s_tex%d", stem.string().c_str(), idx++);
                tex_to_mat[ptr] = name;
                tex_list.push_back({ptr, name});
            }
        }
    }

    // ---- Write BMP texture files -------------------------------------------
    // Try to use the cached CapturedTexture pixel data first (from the
    // snapshot).  Fall back to reading the SDL_Texture through a render
    // target when the raw data is no longer available.
    //
    // Build a lookup from SDL_Texture* to CapturedTexture* so we can grab
    // the original ARGB pixels when possible.
    std::unordered_map<void*, const CapturedTexture*> tex_data;
    if (dbg_.snap && dbg_.snap->textures) {
        const auto& texs = *dbg_.snap->textures;
        for (const auto& ct : texs) {
            // debug_texture() returns the same pointer for the same key,
            // so this mapping is stable within a frame.
            if (auto it = dbg_textures_.find(ct.key); it != dbg_textures_.end())
                tex_data[it->second] = &ct;
        }
    }

    for (auto& [raw_tex, mat_name] : tex_list) {
        SDL_Texture* sdl_tex = static_cast<SDL_Texture*>(raw_tex);
        const int tw = sdl_tex ? sdl_tex->w : 0, th = sdl_tex ? sdl_tex->h : 0;
        if (tw <= 0 || th <= 0) continue;

        SDL_Surface* surf = SDL_CreateSurface(tw, th, SDL_PIXELFORMAT_ARGB8888);
        if (!surf) continue;

        // Prefer original pixel data when available.
        const CapturedTexture* ct = nullptr;
        if (auto it = tex_data.find(raw_tex); it != tex_data.end()) ct = it->second;

        if (ct && static_cast<int>(ct->width) == tw && static_cast<int>(ct->height) == th
            && ct->argb.size() == static_cast<size_t>(tw) * th) {
            // Copy directly — no GPU round-trip.
            SDL_LockSurface(surf);
            for (int y = 0; y < th; ++y) {
                auto* dst = reinterpret_cast<std::uint32_t*>(
                    static_cast<std::uint8_t*>(surf->pixels) + y * surf->pitch);
                const std::uint32_t* src = ct->argb.data() + y * tw;
                std::memcpy(dst, src, tw * 4);
            }
            SDL_UnlockSurface(surf);
        } else {
            // Fall back: render the texture into a target and read back.
            SDL_Texture* rt = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ARGB8888,
                                                SDL_TEXTUREACCESS_TARGET, tw, th);
            if (rt) {
                // Disable blending on the source texture so we get the raw
                // pixels, not premultiplied-blended-over-black.
                SDL_BlendMode prev_blend = SDL_BLENDMODE_NONE;
                SDL_GetTextureBlendMode(sdl_tex, &prev_blend);
                SDL_SetTextureBlendMode(sdl_tex, SDL_BLENDMODE_NONE);

                SDL_SetRenderTarget(renderer_, rt);
                SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 0);
                SDL_RenderClear(renderer_);
                SDL_RenderTexture(renderer_, sdl_tex, nullptr, nullptr);
                if (SDL_Surface* read = SDL_RenderReadPixels(renderer_, nullptr)) {
                    SDL_BlitSurface(read, nullptr, surf, nullptr);
                    SDL_DestroySurface(read);
                }
                SDL_SetRenderTarget(renderer_, nullptr);
                SDL_DestroyTexture(rt);

                // Restore the original blend mode.
                SDL_SetTextureBlendMode(sdl_tex, prev_blend);
            }
        }

        fs::path bmp_path = dir / (mat_name + ".bmp");
        SDL_SaveBMP(surf, platform::path_to_utf8(bmp_path).c_str());
        SDL_DestroySurface(surf);
    }

    // ---- Write MTL file ----------------------------------------------------
    {
        std::ofstream mtl(mtl_path);
        if (!mtl.is_open()) return false;
        mtl << "# Exported from Orbit64 N64 emulator\n";

        // One material per unique texture.
        for (auto& [raw_tex, mat_name] : tex_list) {
            mtl << "\nnewmtl " << mat_name << "\n";
            mtl << "Ka 1.0 1.0 1.0\n";
            mtl << "Kd 1.0 1.0 1.0\n";
            mtl << "Ks 0.0 0.0 0.0\n";
            mtl << "d 1.0\n";
            mtl << "map_Kd " << mat_name << ".bmp\n";
        }
        // Fallback material for untextured / vertex-coloured triangles.
        mtl << "\nnewmtl vcol\n";
        mtl << "Ka 1.0 1.0 1.0\n";
        mtl << "Kd 1.0 1.0 1.0\n";
        mtl << "Ks 0.0 0.0 0.0\n";
        mtl << "d 1.0\n";
    }

    // ---- Write OBJ file ----------------------------------------------------
    {
        std::ofstream obj(obj_path);
        if (!obj.is_open()) return false;

        obj << "# Exported from Orbit64 N64 emulator\n";
        obj << "# Object: " << hex32(dbg_.selected.vtx_addr) << "  Triangles: " << ntri << "\n\n";
        obj << "mtllib " << stem.string() << ".mtl\n\n";

        // -- Positions (v) ---------------------------------------------------
        // 9 floats per triangle: v0.xyz v1.xyz v2.xyz  → nvert "v" lines.
        for (size_t i = 0; i < ntri * 9; i += 3) {
            // Flip Y so the model sits right-side up in most DCC tools.
            obj << "v " << dbg_.mesh_pos[i + 0] << " " << -dbg_.mesh_pos[i + 1]
                << " " << dbg_.mesh_pos[i + 2] << "\n";
        }
        obj << "\n";

        // -- Texture coordinates (vt) ----------------------------------------
        // Only emit for textured triangles.  We need a separate vt index
        // counter because not every vertex has a UV.
        // vt_index[t] = 1-based vt index of the first vert of triangle t
        // (the next two are vt_index[t]+1 and +2).  0 = not textured.
        std::vector<size_t> vt_base(ntri, 0);
        if (has_tex) {
            size_t vt_count = 0;
            for (size_t t = 0; t < ntri; ++t) {
                void* tex = dbg_.mesh_tex[t];
                if (!tex) continue;
                vt_base[t] = vt_count + 1; // 1-based
                for (int k = 0; k < 3; ++k) {
                    float u = dbg_.mesh_uv[t * 6 + k * 2 + 0];
                    float v = dbg_.mesh_uv[t * 6 + k * 2 + 1];
                    // OBJ V-axis is flipped vs. the N64 texture-space origin.
                    obj << "vt " << u << " " << (1.0f - v) << "\n";
                    ++vt_count;
                }
            }
            if (vt_count > 0) obj << "\n";
        }

        // -- Faces (f) -------------------------------------------------------
        // Group by material.  Textured faces use "f v/vt", untextured use "f v".
        obj << "o " << stem.string() << "\n";
        void* cur_mat = reinterpret_cast<void*>(static_cast<uintptr_t>(1)); // sentinel
        for (size_t t = 0; t < ntri; ++t) {
            void* tex = has_tex ? dbg_.mesh_tex[t] : nullptr;

            // Emit usemtl on material change.
            if (tex != cur_mat) {
                cur_mat = tex;
                if (tex && tex_to_mat.count(tex))
                    obj << "\nusemtl " << tex_to_mat[tex] << "\n";
                else
                    obj << "\nusemtl vcol\n";
            }

            // v indices are 1-based, flat layout: tri t → base = t*3+1
            size_t v0 = t * 3 + 1;
            if (vt_base[t]) {
                // Textured: f v/vt v/vt v/vt
                size_t uv0 = vt_base[t];
                obj << "f " << v0     << "/" << uv0     << " "
                             << v0 + 1 << "/" << uv0 + 1 << " "
                             << v0 + 2 << "/" << uv0 + 2 << "\n";
            } else {
                // Untextured: f v v v
                obj << "f " << v0 << " " << v0 + 1 << " " << v0 + 2 << "\n";
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 3D Object Inspector
// ---------------------------------------------------------------------------

void App::tool_object_inspector() {
    const SceneObject* obj = find_object(dbg_.selected);
    if (!dbg_.selected.valid) {
        empty_state(Icon::Cube, "No object selected", "Pick an object in the Object Viewer to inspect its 3D model here.");
        float bw = dp(190);
        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - bw) * 0.5f + ImGui::GetCursorPosX());
        if (button("Open Object Viewer", Icon::Layers, ButtonKind::Primary, bw)) open_tool(DebugTool::ObjectViewer);
        return;
    }

    // Keep the last seen mesh so the model stays visible when the object is culled.
    dbg_.mesh_live = obj != nullptr && dbg_.snap && dbg_.snap->meshes;
    if (dbg_.mesh_live && (dbg_.mesh_frame != dbg_.snap->frame || dbg_.mesh_pos.empty())) {
        const CapturedMesh& m = (*dbg_.snap->meshes)[obj->index];
        dbg_.mesh_pos = m.pos;
        dbg_.mesh_col = m.col;
        dbg_.mesh_frame = dbg_.snap->frame;
        // Textures (captured only for the inspected object; they appear a frame after selecting it).
        const size_t n = m.col.size();
        if (m.tex.size() == n && m.uv.size() == n * 6 && dbg_.snap->textures) {
            const auto& texs = *dbg_.snap->textures;
            // Bound GPU memory. Only done here, before any texture of this mesh is
            // resolved, so no triangle can keep a pointer to a destroyed texture.
            if (dbg_textures_.size() > 256) {
                for (auto& [k, tex] : dbg_textures_) SDL_DestroyTexture(tex);
                dbg_textures_.clear();
            }
            dbg_.mesh_uv.assign(n * 6, 0.0f);
            dbg_.mesh_tex.assign(n, nullptr);
            std::vector<SDL_Texture*> resolved(texs.size(), nullptr);
            std::vector<bool> done(texs.size(), false);
            int count = 0;
            for (size_t t = 0; t < n; ++t) {
                int ti = m.tex[t];
                if (ti < 0 || ti >= static_cast<int>(texs.size())) continue;
                if (!done[ti]) {
                    resolved[ti] = debug_texture(texs[ti]);
                    done[ti] = true;
                    count += resolved[ti] ? 1 : 0;
                }
                const CapturedTexture& ct = texs[ti];
                dbg_.mesh_tex[t] = resolved[ti];
                for (int k = 0; k < 3; ++k) {
                    dbg_.mesh_uv[t * 6 + k * 2 + 0] = (m.uv[t * 6 + k * 2 + 0] - ct.origin_s) / ct.span_s;
                    dbg_.mesh_uv[t * 6 + k * 2 + 1] = (m.uv[t * 6 + k * 2 + 1] - ct.origin_t) / ct.span_t;
                }
            }
            dbg_.mesh_texture_count = count;
        } else if (m.tex.empty()) {
            dbg_.mesh_uv.clear();
            dbg_.mesh_tex.clear();
            dbg_.mesh_texture_count = 0;
        }
    }
    const size_t ntri = dbg_.mesh_col.size();

    // Bounds
    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (size_t i = 0; i + 2 < dbg_.mesh_pos.size(); i += 3)
        for (int k = 0; k < 3; ++k) {
            mn[k] = std::min(mn[k], dbg_.mesh_pos[i + k]);
            mx[k] = std::max(mx[k], dbg_.mesh_pos[i + k]);
        }
    if (ntri == 0) for (int k = 0; k < 3; ++k) mn[k] = mx[k] = 0;
    float radius = 0.5f * std::sqrt((mx[0] - mn[0]) * (mx[0] - mn[0]) + (mx[1] - mn[1]) * (mx[1] - mn[1]) + (mx[2] - mn[2]) * (mx[2] - mn[2]));
    radius = std::max(radius, 1.0f);
    const float fov = 45.0f * kPi / 180.0f;
    auto fit = [&]() {
        for (int k = 0; k < 3; ++k) dbg_.cam.target[k] = (mn[k] + mx[k]) * 0.5f;
        dbg_.cam.distance = radius / std::sin(fov * 0.5f) * 1.15f;
        dbg_.cam.yaw = 0.6f;
        dbg_.cam.pitch = 0.35f;
        dbg_.cam.fitted = true;
    };
    if (!dbg_.cam.fitted && ntri) fit();

    // ---- Toolbar row
    const float W = ImGui::GetContentRegionAvail().x;
    const float row_x0 = ImGui::GetCursorPosX();
    ImGui::PushFont(g_fonts.body_bold);
    if (obj) ImGui::Text("Object %d", obj->index);
    else ImGui::TextColored(g_pal.text_dim, "Object");
    ImGui::PopFont();
    ImGui::SameLine(0, dp(8));
    ImGui::PushFont(g_fonts.mono_small);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + dp(2));
    ImGui::TextColored(g_pal.text_dim, "%s", hex32(dbg_.selected.vtx_addr).c_str());
    ImGui::PopFont();
    ImGui::SameLine(0, dp(8));
    if (dbg_.mesh_live) badge("LIVE", g_pal.success);
    else badge("NOT DRAWN THIS FRAME", g_pal.warning);
    ImGui::SameLine(0, dp(6));
    badge("LINKED TO OBJECT VIEWER", g_pal.accent);

    // 7 icon buttons: Export, Prev, Next, Tex, Wire, Grid, Reset
    float bx = row_x0 + W - dp(30) * 7 - dp(4) * 6;
    ImGui::SameLine(bx);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - dp(6));
    auto nav = [&](int dir) {
        if (dbg_.objects.empty()) return;
        int cur = obj ? obj->index : 0;
        int n = static_cast<int>(dbg_.objects.size());
        select_object(dbg_.objects[(cur + dir + n) % n], false);
    };
    // Export to OBJ button
    const bool can_export = !dbg_.mesh_pos.empty();
    if (icon_button("export", Icon::Download, dp(30), "Export OBJ (Wavefront + textures)", false, can_export)) {
        // Build a default filename from the object's vertex-buffer address.
        std::string def_name = "obj_" + hex32(dbg_.selected.vtx_addr) + ".obj";
        auto result = platform::native_save_file("Export 3D Model as OBJ", def_name, {"obj"});
        if (result.has_value()) {
            if (!result->empty()) {
                // Ensure the path ends with .obj
                std::filesystem::path out_path = *result;
                if (out_path.extension() != ".obj") out_path += ".obj";
                if (export_mesh_obj(out_path)) {
                    toast("Saved " + platform::path_to_utf8(out_path.filename()), ToastKind::Success);
                } else {
                    toast("Export failed — could not write files", ToastKind::Error);
                }
            }
            // empty result = user cancelled, do nothing
        } else {
            // Native dialog not available; fall back to saving in the config dir.
            namespace fs = std::filesystem;
            fs::path out_path = platform::config_dir() / def_name;
            if (export_mesh_obj(out_path)) {
                toast("Saved to " + platform::path_to_utf8(out_path), ToastKind::Success);
            } else {
                toast("Export failed — could not write files", ToastKind::Error);
            }
        }
    }
    if (ImGui::IsItemHovered() && !can_export)
        tooltip("Capture a mesh first — select an object in the Object Viewer");
    ImGui::SameLine(0, dp(4));
    if (icon_button("prev", Icon::ChevronLeft, dp(30), "Previous object")) nav(-1);
    ImGui::SameLine(0, dp(4));
    if (icon_button("next", Icon::ChevronRight, dp(30), "Next object")) nav(1);
    ImGui::SameLine(0, dp(4));
    if (icon_button("tex", Icon::Image, dp(30), "Textures", dbg_.show_textures)) dbg_.show_textures = !dbg_.show_textures;
    ImGui::SameLine(0, dp(4));
    if (icon_button("wire", Icon::Layers, dp(30), "Wireframe", dbg_.wireframe)) dbg_.wireframe = !dbg_.wireframe;
    ImGui::SameLine(0, dp(4));
    if (icon_button("grid", Icon::Grid, dp(30), "Ground grid", dbg_.show_grid)) dbg_.show_grid = !dbg_.show_grid;
    ImGui::SameLine(0, dp(4));
    if (icon_button("reset", Icon::Target, dp(30), "Reset camera (double-click the view)")) fit();

    // ---- Layout: viewport + data panel
    const bool side = W > dp(600);
    const float panel_w = side ? dp(220) : W;
    const float avail_h = ImGui::GetContentRegionAvail().y;
    const float vw = side ? W - panel_w - dp(12) : W;
    const float vh = side ? avail_h : std::max(dp(200), avail_h - dp(230));

    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImVec2 p1(p0.x + vw, p0.y + vh);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGui::InvisibleButton("##viewport3d", ImVec2(vw, vh),
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    ImGuiIO& io = ImGui::GetIO();

    // Camera basis
    auto basis = [&](float eye[3], float f[3], float r[3], float u[3]) {
        float cp = std::cos(dbg_.cam.pitch), sp = std::sin(dbg_.cam.pitch);
        float cy = std::cos(dbg_.cam.yaw), sy = std::sin(dbg_.cam.yaw);
        float dir[3] = {cp * sy, sp, cp * cy};
        for (int k = 0; k < 3; ++k) {
            eye[k] = dbg_.cam.target[k] + dir[k] * dbg_.cam.distance;
            f[k] = -dir[k];
        }
        r[0] = f[2]; r[1] = 0; r[2] = -f[0]; // cross(f, up)
        float rl = std::sqrt(r[0] * r[0] + r[2] * r[2]);
        if (rl < 1e-6f) { r[0] = 1; rl = 1; }
        r[0] /= rl; r[2] /= rl;
        u[0] = r[1] * f[2] - r[2] * f[1];
        u[1] = r[2] * f[0] - r[0] * f[2];
        u[2] = r[0] * f[1] - r[1] * f[0];
        // ensure up points up
        if (u[1] < 0) { for (int k = 0; k < 3; ++k) { u[k] = -u[k]; r[k] = -r[k]; } }
    };

    // Mouse: left = orbit, right/middle or Shift+left = pan, wheel = zoom, double-click = reset.
    if (active && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
        bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Right) || ImGui::IsMouseDown(ImGuiMouseButton_Middle) || io.KeyShift;
        if (pan) {
            float e[3], f[3], r[3], u[3];
            basis(e, f, r, u);
            float k = dbg_.cam.distance * 0.0016f / g_scale;
            for (int i = 0; i < 3; ++i) dbg_.cam.target[i] += (-r[i] * io.MouseDelta.x + u[i] * io.MouseDelta.y) * k;
        } else {
            dbg_.cam.yaw -= io.MouseDelta.x * 0.01f;
            dbg_.cam.pitch = std::clamp(dbg_.cam.pitch + io.MouseDelta.y * 0.01f, -1.55f, 1.55f);
        }
    }
    if (hovered && io.MouseWheel != 0.0f)
        dbg_.cam.distance = std::clamp(dbg_.cam.distance * std::pow(0.87f, io.MouseWheel), radius * 0.05f, radius * 60.0f);
    if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) fit();

    // Background
    int v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(p0, p1, IM_COL32_WHITE, dp(8));
    ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, p0, ImVec2(p0.x, p1.y), col(mix(g_pal.bg3, g_pal.bg1, 0.3f)),
                                                  col(g_pal.bg0));
    dl->AddRect(p0, p1, col(active ? g_pal.accent : g_pal.border), dp(8));
    dl->PushClipRect(ImVec2(p0.x + 1, p0.y + 1), ImVec2(p1.x - 1, p1.y - 1), true);

    float eye[3], f[3], r[3], u[3];
    basis(eye, f, r, u);
    const float focal = vh * 0.5f / std::tan(fov * 0.5f);
    const ImVec2 c((p0.x + p1.x) * 0.5f, (p0.y + p1.y) * 0.5f);
    const float near_z = dbg_.cam.distance * 0.01f;
    auto project = [&](const float* v, ImVec2& out, float& z) {
        float d[3] = {v[0] - eye[0], v[1] - eye[1], v[2] - eye[2]};
        z = d[0] * f[0] + d[1] * f[1] + d[2] * f[2];
        if (z < near_z) return false;
        float x = d[0] * r[0] + d[1] * r[1] + d[2] * r[2];
        float y = d[0] * u[0] + d[1] * u[1] + d[2] * u[2];
        out = ImVec2(c.x + x / z * focal, c.y - y / z * focal);
        return true;
    };

    // Ground grid under the model
    if (dbg_.show_grid) {
        float step = std::pow(10.0f, std::floor(std::log10(radius * 0.5f)));
        if (radius / step > 8) step *= 2.5f;
        int n = 10;
        float gy = mn[1];
        float cx = std::round(dbg_.cam.target[0] / step) * step, cz = std::round(dbg_.cam.target[2] / step) * step;
        for (int i = -n; i <= n; ++i) {
            for (int axis = 0; axis < 2; ++axis) {
                float a[3], b[3];
                if (axis == 0) { a[0] = b[0] = cx + i * step; a[2] = cz - n * step; b[2] = cz + n * step; }
                else { a[2] = b[2] = cz + i * step; a[0] = cx - n * step; b[0] = cx + n * step; }
                a[1] = b[1] = gy;
                ImVec2 sa, sb;
                float za, zb;
                if (project(a, sa, za) && project(b, sb, zb))
                    dl->AddLine(sa, sb, col(g_pal.text_faint, i == 0 ? 0.35f : 0.12f), 1.0f);
            }
        }
    }

    // Model: painter's algorithm, flat shaded, anti-aliasing off to avoid seams.
    if (ntri) {
        float light[3];
        for (int k = 0; k < 3; ++k) light[k] = -f[k] + u[k] * 0.6f - r[k] * 0.35f;
        float ll = std::sqrt(light[0] * light[0] + light[1] * light[1] + light[2] * light[2]);
        for (float& l : light) l /= ll;

        struct Tri { ImVec2 s[3]; float depth; ImU32 color; SDL_Texture* tex; ImVec2 uv[3]; };
        const bool textured = dbg_.show_textures && dbg_.mesh_tex.size() == ntri && dbg_.mesh_uv.size() == ntri * 6;
        std::vector<Tri> tris;
        tris.reserve(std::min<size_t>(ntri, 120000));
        for (size_t t = 0; t < ntri && t < 120000; ++t) {
            const float* v = &dbg_.mesh_pos[t * 9];
            Tri tr;
            float z[3];
            bool ok = true;
            for (int k = 0; k < 3; ++k) ok &= project(v + k * 3, tr.s[k], z[k]);
            if (!ok) continue;
            tr.depth = z[0] + z[1] + z[2];
            float e1[3] = {v[3] - v[0], v[4] - v[1], v[5] - v[2]}, e2[3] = {v[6] - v[0], v[7] - v[1], v[8] - v[2]};
            float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
            float nl = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            float shade = 0.35f;
            if (nl > 1e-9f) shade = 0.30f + 0.70f * std::fabs((n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) / nl);
            std::uint32_t rgba = dbg_.mesh_col[t];
            // Lift very dark materials (e.g. a black moustache) so their shape stays readable on the dark viewport.
            float cr = 0.16f + 0.84f * ((rgba >> 24) & 0xFF) / 255.0f, cg = 0.16f + 0.84f * ((rgba >> 16) & 0xFF) / 255.0f,
                  cb = 0.16f + 0.84f * ((rgba >> 8) & 0xFF) / 255.0f;
            tr.tex = textured ? static_cast<SDL_Texture*>(dbg_.mesh_tex[t]) : nullptr;
            if (tr.tex) {
                // Textured: the RDP's shade colour modulates the texture (like the common
                // MODULATE combiner), with a little extra lighting for depth.
                float l = 0.65f + 0.35f * shade;
                float r0 = ((rgba >> 24) & 0xFF) / 255.0f, g0 = ((rgba >> 16) & 0xFF) / 255.0f, b0 = ((rgba >> 8) & 0xFF) / 255.0f;
                tr.color = ImGui::ColorConvertFloat4ToU32(ImVec4(r0 * l, g0 * l, b0 * l, 1.0f));
                for (int k = 0; k < 3; ++k) tr.uv[k] = ImVec2(dbg_.mesh_uv[t * 6 + k * 2], dbg_.mesh_uv[t * 6 + k * 2 + 1]);
            } else {
                tr.color = ImGui::ColorConvertFloat4ToU32(ImVec4(cr * shade, cg * shade, cb * shade, 1.0f));
            }
            tris.push_back(tr);
        }
        std::sort(tris.begin(), tris.end(), [](const Tri& a, const Tri& b) { return a.depth > b.depth; });
        ImDrawListFlags old = dl->Flags;
        dl->Flags &= ~ImDrawListFlags_AntiAliasedFill;
        for (const auto& t : tris) {
            if (dbg_.wireframe) continue;
            if (t.tex) {
                dl->PushTextureID((ImTextureID)(intptr_t)t.tex);
                dl->PrimReserve(3, 3);
                for (int k = 0; k < 3; ++k) dl->PrimVtx(t.s[k], t.uv[k], t.color);
                dl->PopTextureID();
            } else {
                dl->AddTriangleFilled(t.s[0], t.s[1], t.s[2], t.color);
            }
        }
        dl->Flags = old;
        if (dbg_.wireframe)
            for (const auto& t : tris) dl->AddTriangle(t.s[0], t.s[1], t.s[2], col(g_pal.accent_hover, 0.8f), 1.0f);
    } else {
        const char* msg = "Model not captured yet";
        ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2(c.x - ts.x * 0.5f, c.y - ts.y * 0.5f), col(g_pal.text_dim), msg);
    }

    // Axis gizmo
    {
        ImVec2 g(p0.x + dp(38), p1.y - dp(38));
        dl->AddCircleFilled(g, dp(28), col(g_pal.bg0, 0.7f), 32);
        const char* names[3] = {"X", "Y", "Z"};
        for (int a = 0; a < 3; ++a) {
            float axis[3] = {a == 0 ? 1.0f : 0, a == 1 ? 1.0f : 0, a == 2 ? 1.0f : 0};
            ImVec2 e(g.x + (axis[0] * r[0] + axis[1] * r[1] + axis[2] * r[2]) * dp(20),
                     g.y - (axis[0] * u[0] + axis[1] * u[1] + axis[2] * u[2]) * dp(20));
            dl->AddLine(g, e, col(kAxisCol[a]), dp(2));
            dl->AddText(g_fonts.small_bold, font_px(g_fonts.small_bold), ImVec2(e.x - dp(3), e.y - dp(7)), col(kAxisCol[a]), names[a]);
        }
    }
    // Hint
    const char* hint = "Drag to rotate \xC2\xB7 Right-drag or Shift+drag to pan \xC2\xB7 Scroll to zoom";
    ImVec2 hs = g_fonts.small->CalcTextSizeA(font_px(g_fonts.small), FLT_MAX, 0, hint);
    if (hs.x < vw - dp(100))
        dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(p1.x - hs.x - dp(12), p1.y - hs.y - dp(10)), col(g_pal.text_faint), hint);
    char stats[64];
    if (dbg_.mesh_texture_count > 0)
        std::snprintf(stats, sizeof stats, "%zu triangles \xC2\xB7 %zu vertices \xC2\xB7 %d textures", ntri, ntri * 3, dbg_.mesh_texture_count);
    else
        std::snprintf(stats, sizeof stats, "%zu triangles \xC2\xB7 %zu vertices", ntri, ntri * 3);
    dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(p0.x + dp(12), p0.y + dp(10)), col(g_pal.text_dim), stats);
    dl->PopClipRect();

    // ---- Data panel
    if (side) {
        ImGui::SetCursorScreenPos(ImVec2(p1.x + dp(12), p0.y));
    } else {
        ImGui::SetCursorScreenPos(ImVec2(p0.x, p1.y + dp(10)));
    }
    ImGui::BeginChild("##inspector_data", ImVec2(panel_w, side ? vh : 0), ImGuiChildFlags_None);
    const float pw = ImGui::GetContentRegionAvail().x;
    auto section = [&](const char* s) {
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::TextColored(g_pal.text_faint, "%s", s);
        ImGui::PopFont();
    };
    if (obj) {
        section(dbg_.world_space ? "TRANSFORM (WORLD)" : "TRANSFORM (CAMERA)");
        vec3_row("Position", obj->pos, "%.1f", pw);
        vec3_row("Rotation", obj->rot, "%.1f\xC2\xB0", pw);
        vec3_row("Scale", obj->scale, "%.3f", pw);
        ImGui::Dummy(dp(0, 4));
        section("OBJECT");
        kv_row("Draw index", std::to_string(obj->index), pw);
        kv_row("Vertex buffer", hex32(obj->vtx_addr), pw);
        kv_row("Matrix", hex32(obj->mtx_addr), pw);
        kv_row("Display list", hex32(obj->dl_addr), pw);
        char d[32];
        std::snprintf(d, sizeof d, "%.1f", obj->distance);
        kv_row("Camera distance", d, pw);
        kv_row("Triangles", std::to_string(ntri), pw);
        kv_row("Textures", std::to_string(dbg_.mesh_texture_count), pw);
        ImGui::Dummy(dp(0, 4));
        if (button("Export OBJ", Icon::Download, ButtonKind::Primary, pw, can_export)) {
            std::string def_name = "obj_" + hex32(dbg_.selected.vtx_addr) + ".obj";
            auto result = platform::native_save_file("Export 3D Model as OBJ", def_name, {"obj"});
            if (result.has_value()) {
                if (!result->empty()) {
                    std::filesystem::path out_path = *result;
                    if (out_path.extension() != ".obj") out_path += ".obj";
                    if (export_mesh_obj(out_path))
                        toast("Saved " + platform::path_to_utf8(out_path.filename()), ToastKind::Success);
                    else
                        toast("Export failed — could not write files", ToastKind::Error);
                }
            } else {
                namespace fs = std::filesystem;
                fs::path out_path = platform::config_dir() / def_name;
                if (export_mesh_obj(out_path))
                    toast("Saved to " + platform::path_to_utf8(out_path), ToastKind::Success);
                else
                    toast("Export failed — could not write files", ToastKind::Error);
            }
        }
        if (button("Track as Player", Icon::User, ButtonKind::Subtle, pw)) {
            dbg_.player = dbg_.selected;
            dbg_.player_source = 0;
            dbg_.player_found = false;
            dbg_.player_trail.clear();
            open_tool(DebugTool::PlayerViewer);
        }
        if (button("Vertices in Memory", Icon::Pencil, ButtonKind::Ghost, pw)) open_in_editor(obj->vtx_addr, MemType::S16);
    } else {
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + pw);
        ImGui::TextColored(g_pal.text_dim, "The object isn't drawn in the current frame (off screen or hidden). Its last captured model is shown.");
        ImGui::PopTextWrapPos();
    }
    ImGui::EndChild();
}

// ---------------------------------------------------------------------------
// Player Viewer
// ---------------------------------------------------------------------------

void App::tool_player_viewer() {
    const float w = ImGui::GetContentRegionAvail().x;
    const char* sources[] = {"Tracked Object", "Memory Address"};
    if (segmented("psource", sources, 2, &dbg_.player_source, w)) {
        dbg_.player_found = false;
        dbg_.player_trail.clear();
    }
    ImGui::Dummy(dp(0, 4));

    float pos[3] = {}, vel[3] = {};
    bool have = false;
    std::uint32_t addr = 0;

    if (dbg_.player_source == 0) {
        const SceneObject* p = find_object(dbg_.player);
        if (!dbg_.player.valid) {
            empty_state(Icon::User, "No player selected",
                        "Select the player's model in the Object Viewer (usually the object near the centre of the screen that "
                        "moves with your input) and choose Track as Player.");
            float bw = dp(210);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (w - bw) * 0.5f);
            if (dbg_.selected.valid) {
                if (button("Track Selected Object", Icon::Target, ButtonKind::Primary, bw)) {
                    dbg_.player = dbg_.selected;
                    dbg_.player_found = false;
                }
            } else if (button("Open Object Viewer", Icon::Layers, ButtonKind::Primary, bw)) {
                open_tool(DebugTool::ObjectViewer);
            }
            return;
        }
        if (p) {
            have = true;
            std::copy(p->pos, p->pos + 3, pos);
            std::copy(dbg_.player_vel, dbg_.player_vel + 3, vel);
            addr = p->mtx_addr;
        }
    } else {
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_dim, "Address of the X position (F32); Y and Z follow at +4 and +8.");
        ImGui::PopFont();
        ImGui::SetNextItemWidth(w - dp(110));
        bool enter = ImGui::InputTextWithHint("##paddr", "e.g. 8033B1AC", dbg_.player_addr_text, sizeof dbg_.player_addr_text,
                                              ImGuiInputTextFlags_CharsHexadecimal | ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine(0, dp(8));
        if (button("Set", Icon::Check, ButtonKind::Subtle, dp(102), true, dp(32)) || enter) {
            std::uint32_t a;
            if (parse_address(dbg_.player_addr_text, a)) {
                dbg_.player_addr = a;
                dbg_.player_found = false;
                dbg_.player_trail.clear();
            } else {
                toast("Enter a hexadecimal address", ToastKind::Warning);
            }
        }
        ImGui::PushFont(g_fonts.small);
        ImGui::TextColored(g_pal.text_faint, "Tip: find it with Memory Search (F32, value changes when you move).");
        ImGui::PopFont();
        if (dbg_.player_addr && dbg_.snap && dbg_.snap->valid(dbg_.player_addr, 12)) {
            have = true;
            addr = dbg_.player_addr;
            for (int k = 0; k < 3; ++k) pos[k] = dbg_.snap->f32(addr + k * 4);
            std::uint64_t f = dbg_.snap->frame;
            if (dbg_.player_found && f > dbg_.player_frame) {
                float dt = static_cast<float>(f - dbg_.player_frame);
                for (int k = 0; k < 3; ++k) dbg_.player_vel[k] = (pos[k] - dbg_.player_pos[k]) / dt;
            }
            if (!dbg_.player_found || f != dbg_.player_frame) {
                dbg_.player_trail.push_back({pos[0], pos[1], pos[2]});
                while (dbg_.player_trail.size() > 240) dbg_.player_trail.pop_front();
            }
            std::copy(pos, pos + 3, dbg_.player_pos);
            dbg_.player_frame = f;
            dbg_.player_found = true;
            std::copy(dbg_.player_vel, dbg_.player_vel + 3, vel);
        }
        if (!dbg_.player_addr) return;
    }

    // ---- Summary cards
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float h = dp(58);
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(g_pal.bg2), dp(8));
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(g_pal.border), dp(8));
    ImVec2 ic(p.x + dp(29), p.y + h * 0.5f);
    dl->AddCircleFilled(ic, dp(17), col(have ? g_pal.success : g_pal.warning, 0.16f), 24);
    draw_icon(dl, Icon::User, ic, dp(18), col(have ? g_pal.success : g_pal.warning));
    dl->AddText(g_fonts.body_bold, font_px(g_fonts.body_bold), ImVec2(p.x + dp(56), p.y + dp(10)), col(g_pal.text),
                have ? "Player" : "Player not drawn this frame");
    std::string a = std::string(dbg_.player_source == 0 ? "matrix " : "address ") + hex32(addr);
    dl->AddText(g_fonts.mono_small, font_px(g_fonts.mono_small), ImVec2(p.x + dp(56), p.y + dp(33)), col(g_pal.text_dim), a.c_str());
    ImGui::Dummy(ImVec2(w, h + dp(6)));
    if (!have) {
        // keep showing last known values
        std::copy(dbg_.player_pos, dbg_.player_pos + 3, pos);
    }

    auto section = [&](const char* s) {
        ImGui::Dummy(dp(0, 2));
        ImGui::PushFont(g_fonts.small_bold);
        ImGui::TextColored(g_pal.text_faint, "%s", s);
        ImGui::PopFont();
    };
    section("POSITION");
    vec3_row("X  Y  Z", pos, "%.3f", w);
    section("VELOCITY (PER FRAME)");
    vec3_row("dX dY dZ", vel, "%.3f", w);
    float hspeed = std::sqrt(vel[0] * vel[0] + vel[2] * vel[2]);
    float speed = std::sqrt(hspeed * hspeed + vel[1] * vel[1]);
    char b[48];
    std::snprintf(b, sizeof b, "%.3f", hspeed);
    kv_row("Horizontal speed", b, w);
    std::snprintf(b, sizeof b, "%.3f", speed);
    kv_row("Total speed", b, w);
    float heading = hspeed > 1e-4f ? std::atan2(vel[0], vel[2]) * 180.0f / kPi : 0.0f;
    std::snprintf(b, sizeof b, "%.1f\xC2\xB0", heading);
    kv_row("Movement heading", b, w);

    if (dbg_.player_source == 0) {
        if (const SceneObject* o = find_object(dbg_.player)) {
            section("MODEL");
            vec3_row("Rotation", o->rot, "%.1f\xC2\xB0", w);
            vec3_row("Scale", o->scale, "%.3f", w);
            kv_row("Vertex buffer", hex32(o->vtx_addr), w);
            kv_row("Triangles", std::to_string(o->triangles), w);
        }
    } else if (dbg_.snap && dbg_.snap->valid(addr, 64)) {
        section("NEARBY VALUES");
        if (ImGui::BeginTable("##pstruct", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            for (int i = -4; i < 12; ++i) {
                std::uint32_t aa = addr + i * 4;
                ImGui::TableNextRow();
                ImGui::PushFont(g_fonts.mono_small);
                ImGui::TableNextColumn();
                ImGui::TextColored(i >= 0 && i < 3 ? g_pal.accent_hover : g_pal.text_faint, "%c%02X", i < 0 ? '-' : '+', std::abs(i * 4));
                ImGui::TableNextColumn();
                ImGui::TextColored(g_pal.text_dim, "%08X", dbg_.snap->u32(aa));
                ImGui::TableNextColumn();
                ImGui::TextColored(g_pal.text, "%.4g", dbg_.snap->f32(aa));
                ImGui::PopFont();
            }
            ImGui::EndTable();
        }
    }

    // ---- Top-down trail
    section("PATH (TOP-DOWN, LAST 4 SECONDS)");
    float mh = std::max(dp(120), ImGui::GetContentRegionAvail().y - dp(4));
    ImVec2 m0 = ImGui::GetCursorScreenPos(), m1(m0.x + w, m0.y + mh);
    dl->AddRectFilled(m0, m1, col(g_pal.bg0), dp(8));
    dl->AddRect(m0, m1, col(g_pal.border), dp(8));
    if (dbg_.player_trail.size() >= 2) {
        float mnx = FLT_MAX, mxx = -FLT_MAX, mnz = FLT_MAX, mxz = -FLT_MAX;
        for (const auto& t : dbg_.player_trail) {
            mnx = std::min(mnx, t[0]); mxx = std::max(mxx, t[0]);
            mnz = std::min(mnz, t[2]); mxz = std::max(mxz, t[2]);
        }
        float span = std::max({mxx - mnx, mxz - mnz, 50.0f});
        float cxw = (mnx + mxx) * 0.5f, czw = (mnz + mxz) * 0.5f;
        float sc = (std::min(w, mh) - dp(30)) / span;
        ImVec2 mc((m0.x + m1.x) * 0.5f, (m0.y + m1.y) * 0.5f);
        std::vector<ImVec2> pts;
        for (const auto& t : dbg_.player_trail) pts.push_back(ImVec2(mc.x + (t[0] - cxw) * sc, mc.y + (t[2] - czw) * sc));
        for (size_t i = 1; i < pts.size(); ++i)
            dl->AddLine(pts[i - 1], pts[i], col(g_pal.accent, 0.2f + 0.8f * i / pts.size()), dp(2));
        dl->AddCircleFilled(pts.back(), dp(5), col(g_pal.success), 16);
        char sb[32];
        std::snprintf(sb, sizeof sb, "%.0f units", span);
        dl->AddText(g_fonts.small, font_px(g_fonts.small), ImVec2(m0.x + dp(10), m1.y - dp(22)), col(g_pal.text_faint), sb);
    } else {
        const char* msg = "Move the player to draw its path";
        ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2((m0.x + m1.x - ts.x) * 0.5f, (m0.y + m1.y - ts.y) * 0.5f), col(g_pal.text_faint), msg);
    }
    ImGui::Dummy(ImVec2(w, mh));
}

} // namespace ui
