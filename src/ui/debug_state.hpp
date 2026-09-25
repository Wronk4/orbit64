#pragma once
// State of the DEBUG / MEMORY tools.
//
// Everything here is game-independent: the tools read RDRAM and CPU state
// from DebugSnapshot, and "objects" are the geometry groups the game submits
// to the RSP each frame (one per model-view matrix), which works for any game
// using an F3D-family graphics microcode.

#include "emu_core.hpp"
#include "icons.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace ui {

enum class DebugTool {
    ObjectViewer, ObjectInspector, PlayerViewer, Registers, FrameControl, Resolution,
    MemorySearch, RamWatch, MemoryEditor, FreezeList,
    Count
};
constexpr int kDebugToolCount = static_cast<int>(DebugTool::Count);

struct DebugToolInfo {
    const char* title;
    const char* description;
    Icon icon;
    bool memory; // listed in the MEMORY section
};
const DebugToolInfo& debug_tool_info(DebugTool t);

// ---- Memory value types -----------------------------------------------------
enum class MemType { U8, U16, U32, S8, S16, S32, F32, Count };
constexpr int kMemTypeCount = static_cast<int>(MemType::Count);
extern const char* const kMemTypeNames[kMemTypeCount];
int mem_type_size(MemType t);
// Reads a value from the snapshot (big-endian) as raw bits.
std::uint32_t mem_read_raw(const DebugSnapshot& s, std::uint32_t addr, MemType t);
std::string mem_format(std::uint32_t raw, MemType t);
double mem_as_double(std::uint32_t raw, MemType t);
// Parses user input ("123", "-5", "0x1F", "1.5") into big-endian bytes.
bool mem_parse(const std::string& text, MemType t, std::vector<std::uint8_t>& out);
// Parses "8033B1AC", "0x8033B1AC" or a physical offset; returns a KSEG0 address.
bool parse_address(const std::string& text, std::uint32_t& out);

// ---- Scene objects (captured geometry) ---------------------------------------
struct SceneObject {
    int index = 0;                  // draw order within the frame
    std::uint32_t mtx_addr = 0, vtx_addr = 0, dl_addr = 0;
    int triangles = 0;
    float world[4][4] = {};         // model matrix in the chosen reference space
    float pos[3] = {}, rot[3] = {}, scale[3] = {};
    float distance = 0;             // from the camera
    bool reference = false;         // mesh used as the world reference (usually level geometry)
    int instance = 0;               // n-th object sharing this vertex buffer
};

struct ObjectKey {
    std::uint32_t vtx_addr = 0;
    int instance = 0;
    bool valid = false;
};

struct Watch {
    std::string name;
    std::uint32_t addr = 0;
    MemType type = MemType::U32;
    std::uint32_t last = 0;
    double changed_at = -10.0;
};

struct Freeze {
    bool enabled = true;
    std::string name;
    std::uint32_t addr = 0;
    MemType type = MemType::U32;
    std::string value;
};

struct DebugState {
    bool open[kDebugToolCount] = {};
    bool capture_active = false, capture_ram = false, capture_geometry = false;
    std::shared_ptr<const DebugSnapshot> snap;
    CpuSnapshot prev_cpu;                    // previous frame (change highlighting)
    std::uint64_t cpu_frame = 0;

    // Object Viewer
    std::vector<SceneObject> objects;
    std::uint64_t objects_frame = ~0ull;
    bool world_space = true;
    char object_filter[32] = {};
    ObjectKey selected;
    float selected_last_pos[3] = {};

    // 3D Object Inspector
    struct Camera {
        float yaw = 0.6f, pitch = 0.35f, distance = 1.0f;
        float target[3] = {};
        bool fitted = false;
    } cam;
    bool wireframe = false, show_grid = true, show_axes = true;
    std::vector<float> mesh_pos;             // cached mesh of the selected object
    std::vector<std::uint32_t> mesh_col;
    std::vector<float> mesh_uv;              // 6 normalized UVs per triangle
    std::vector<void*> mesh_tex;             // SDL_Texture* per triangle (nullptr = untextured)
    int mesh_texture_count = 0;
    bool show_textures = true;
    std::uint64_t mesh_frame = 0;
    bool mesh_live = false;                  // drawn in the latest frame

    // Player Viewer
    int player_source = 0;                   // 0 = tracked object, 1 = memory address
    ObjectKey player;
    float player_pos[3] = {}, player_prev[3] = {}, player_vel[3] = {};
    std::uint64_t player_frame = 0;
    bool player_found = false;
    std::deque<std::array<float, 3>> player_trail;
    char player_addr_text[16] = {};
    std::uint32_t player_addr = 0;

    // Registers
    int reg_tab = 0;

    // Memory Search
    int search_type = static_cast<int>(MemType::S32);
    int search_mode = 0;
    char search_value[32] = {};
    bool search_started = false;
    std::vector<std::uint8_t> search_prev;   // RDRAM at the previous scan
    std::vector<std::uint8_t> search_mask;   // 1 = still a candidate (per aligned slot)
    std::vector<std::uint32_t> search_results; // listed candidates (when few enough)
    std::size_t search_count = 0;
    int search_scans = 0;

    // RAM Watch
    std::vector<Watch> watches;
    char watch_name[48] = {}, watch_addr[16] = {};
    int watch_type = static_cast<int>(MemType::U32);

    // Memory Editor
    char edit_addr_text[16] = "80000000";
    std::uint32_t edit_addr = 0x80000000;
    int edit_type = static_cast<int>(MemType::U32);
    char edit_value[32] = {};
    bool edit_scroll = true;

    // Freeze List
    std::vector<Freeze> freezes;
    char freeze_name[48] = {}, freeze_addr[16] = {}, freeze_value[32] = {};
    int freeze_type = static_cast<int>(MemType::U32);
    bool freezes_dirty = true;

    std::string lists_game;                  // game the watch/freeze lists belong to
};

} // namespace ui
