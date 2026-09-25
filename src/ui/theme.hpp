#pragma once
// Design tokens: colours, type scale, spacing and the DPI scale factor.

#include "imgui.h"

namespace ui {

struct Fonts {
    ImFont* body = nullptr;     // 15px regular
    ImFont* body_bold = nullptr;// 15px bold
    ImFont* small = nullptr;    // 12.5px regular
    ImFont* small_bold = nullptr;
    ImFont* title = nullptr;    // 20px bold
    ImFont* display = nullptr;  // 30px bold
    ImFont* mono = nullptr;     // 13px monospace (status bar, tech data)
    ImFont* mono_small = nullptr;
};

struct Palette {
    ImVec4 bg0, bg1, bg2, bg3, bg4;  // darkest -> lightest surfaces
    ImVec4 border, border_strong;
    ImVec4 text, text_dim, text_faint;
    ImVec4 accent, accent_hover, accent_active, accent_soft, accent2;
    ImVec4 success, warning, danger, info;
    ImVec4 overlay;
};

extern Fonts g_fonts;
extern Palette g_pal;
extern float g_scale; // logical UI scale (DPI * user preference)

inline float dp(float v) { return v * g_scale; }
inline ImVec2 dp(float x, float y) { return ImVec2(x * g_scale, y * g_scale); }

ImU32 col(const ImVec4& c, float alpha_mul = 1.0f);
ImVec4 mix(const ImVec4& a, const ImVec4& b, float t);
ImVec4 with_alpha(const ImVec4& c, float a);

// Accent presets selectable in Settings > General.
struct AccentPreset {
    const char* name;
    ImVec4 color;
    ImVec4 secondary;
};
extern const AccentPreset kAccentPresets[];
extern const int kAccentPresetCount;

void apply_palette(int accent_index);
void apply_style();
// Builds the font atlas at `raster_scale` (UI scale * framebuffer scale) so
// text is crisp on Retina / high-DPI Windows / fractional-scaled Linux.
void build_fonts(float ui_scale, float framebuffer_scale);

} // namespace ui
