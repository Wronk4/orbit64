#include "theme.hpp"
#include "fonts_embedded.hpp"

#include <algorithm>

namespace ui {

Fonts g_fonts;
Palette g_pal;
float g_scale = 1.0f;

static ImVec4 hex(unsigned rgb, float a = 1.0f) {
    return ImVec4(((rgb >> 16) & 0xFF) / 255.0f, ((rgb >> 8) & 0xFF) / 255.0f, (rgb & 0xFF) / 255.0f, a);
}

const AccentPreset kAccentPresets[] = {
    {"Violet", hex(0x8B7BFF), hex(0x4F3FD0)},
    {"Azure", hex(0x4C9BFF), hex(0x2463C9)},
    {"Emerald", hex(0x2FCB86), hex(0x138457)},
    {"Crimson", hex(0xF2555A), hex(0xA8262C)},
    {"Amber", hex(0xF5A524), hex(0xB86E0A)},
};
const int kAccentPresetCount = static_cast<int>(sizeof(kAccentPresets) / sizeof(kAccentPresets[0]));

ImU32 col(const ImVec4& c, float alpha_mul) {
    return ImGui::ColorConvertFloat4ToU32(ImVec4(c.x, c.y, c.z, c.w * alpha_mul));
}

ImVec4 mix(const ImVec4& a, const ImVec4& b, float t) {
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

ImVec4 with_alpha(const ImVec4& c, float a) { return ImVec4(c.x, c.y, c.z, a); }

void apply_palette(int accent_index) {
    accent_index = std::clamp(accent_index, 0, kAccentPresetCount - 1);
    Palette& p = g_pal;
    p.bg0 = hex(0x0C0D11);
    p.bg1 = hex(0x121419);
    p.bg2 = hex(0x181B22);
    p.bg3 = hex(0x20242D);
    p.bg4 = hex(0x2A2F3A);
    p.border = hex(0x252932);
    p.border_strong = hex(0x343945);
    p.text = hex(0xE9EBF1);
    p.text_dim = hex(0x9CA2B0);
    p.text_faint = hex(0x646B7A);
    p.accent = kAccentPresets[accent_index].color;
    p.accent2 = kAccentPresets[accent_index].secondary;
    p.accent_hover = mix(p.accent, hex(0xFFFFFF), 0.12f);
    p.accent_active = mix(p.accent, hex(0x000000), 0.15f);
    p.accent_soft = with_alpha(p.accent, 0.16f);
    p.success = hex(0x3DD68C);
    p.warning = hex(0xF5B94A);
    p.danger = hex(0xF2555A);
    p.info = hex(0x5AA9FF);
    p.overlay = hex(0x05060A, 0.62f);
    apply_style();
}

void apply_style() {
    ImGuiStyle style;
    ImGui::StyleColorsDark(&style);
    const Palette& p = g_pal;

    style.WindowPadding = ImVec2(16, 16);
    style.FramePadding = ImVec2(10, 7);
    style.ItemSpacing = ImVec2(10, 8);
    style.ItemInnerSpacing = ImVec2(8, 6);
    style.CellPadding = ImVec2(10, 7);
    style.ScrollbarSize = 10;
    style.GrabMinSize = 12;
    style.WindowBorderSize = 0;
    style.ChildBorderSize = 0;
    style.PopupBorderSize = 1;
    style.FrameBorderSize = 0;
    style.WindowRounding = 10;
    style.ChildRounding = 8;
    style.FrameRounding = 6;
    style.PopupRounding = 8;
    style.ScrollbarRounding = 6;
    style.GrabRounding = 6;
    style.TabRounding = 6;
    style.SeparatorTextBorderSize = 1;
    style.WindowMenuButtonPosition = ImGuiDir_None;
    style.DisplaySafeAreaPadding = ImVec2(0, 0);
    style.AntiAliasedLinesUseTex = true;

    ImVec4* c = style.Colors;
    c[ImGuiCol_Text] = p.text;
    c[ImGuiCol_TextDisabled] = p.text_faint;
    c[ImGuiCol_WindowBg] = p.bg0;
    c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg] = p.bg2;
    c[ImGuiCol_Border] = p.border;
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = p.bg3;
    c[ImGuiCol_FrameBgHovered] = p.bg4;
    c[ImGuiCol_FrameBgActive] = mix(p.bg4, p.accent, 0.15f);
    c[ImGuiCol_TitleBg] = p.bg1;
    c[ImGuiCol_TitleBgActive] = p.bg1;
    c[ImGuiCol_TitleBgCollapsed] = p.bg1;
    c[ImGuiCol_MenuBarBg] = p.bg1;
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ScrollbarGrab] = p.bg4;
    c[ImGuiCol_ScrollbarGrabHovered] = mix(p.bg4, p.text_faint, 0.5f);
    c[ImGuiCol_ScrollbarGrabActive] = p.text_faint;
    c[ImGuiCol_CheckMark] = ImVec4(1, 1, 1, 1);
    c[ImGuiCol_SliderGrab] = p.accent;
    c[ImGuiCol_SliderGrabActive] = p.accent_hover;
    c[ImGuiCol_Button] = p.bg3;
    c[ImGuiCol_ButtonHovered] = p.bg4;
    c[ImGuiCol_ButtonActive] = mix(p.bg4, p.accent, 0.2f);
    c[ImGuiCol_Header] = p.accent_soft;
    c[ImGuiCol_HeaderHovered] = with_alpha(p.accent, 0.22f);
    c[ImGuiCol_HeaderActive] = with_alpha(p.accent, 0.30f);
    c[ImGuiCol_Separator] = p.border;
    c[ImGuiCol_SeparatorHovered] = p.accent;
    c[ImGuiCol_SeparatorActive] = p.accent;
    c[ImGuiCol_ResizeGrip] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ResizeGripHovered] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ResizeGripActive] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_Tab] = p.bg2;
    c[ImGuiCol_TabHovered] = p.bg4;
    c[ImGuiCol_TabSelected] = p.bg3;
    c[ImGuiCol_TableHeaderBg] = p.bg1;
    c[ImGuiCol_TableBorderStrong] = p.border;
    c[ImGuiCol_TableBorderLight] = p.border;
    c[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(1, 1, 1, 0.018f);
    c[ImGuiCol_TextSelectedBg] = with_alpha(p.accent, 0.35f);
    c[ImGuiCol_NavCursor] = p.accent;
    c[ImGuiCol_ModalWindowDimBg] = p.overlay;
    c[ImGuiCol_PlotLines] = p.accent;
    c[ImGuiCol_PlotHistogram] = p.accent;

    style.ScaleAllSizes(g_scale);
    ImGui::GetStyle() = style;
}

void build_fonts(float ui_scale, float fb_scale) {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();

    const float raster = ui_scale * fb_scale;
    static const ImWchar ranges[] = {
        0x0020, 0x017F, // Basic Latin, Latin-1, Latin Extended-A (Polish, Czech, ...)
        0x2010, 0x203A, // Dashes, bullets, ellipsis, primes, single guillemets
        0x2190, 0x2193, // Arrows
        0x2212, 0x2212, // Minus sign
        0,
    };

    auto load = [&](const fonts::Blob& blob, float px) {
        ImFontConfig cfg;
        cfg.FontDataOwnedByAtlas = false; // data lives in the binary
        cfg.OversampleH = 2;
        cfg.OversampleV = 1;
        cfg.PixelSnapH = false;
        cfg.RasterizerDensity = 1.0f;
        ImFont* f = io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(blob.data), static_cast<int>(blob.size),
                                                   px * raster, &cfg, ranges);
        return f;
    };

    g_fonts.body = load(fonts::font_ui_regular, 15.0f);
    g_fonts.body_bold = load(fonts::font_ui_bold, 15.0f);
    g_fonts.small = load(fonts::font_ui_regular, 12.5f);
    g_fonts.small_bold = load(fonts::font_ui_bold, 12.5f);
    g_fonts.title = load(fonts::font_ui_bold, 20.0f);
    g_fonts.display = load(fonts::font_ui_bold, 30.0f);
    // Cousine lacks a few typographic glyphs (e.g. the ellipsis); merge them from Lato.
    auto load_mono = [&](float px) {
        ImFont* f = load(fonts::font_mono, px);
        static const ImWchar extra[] = {0x2026, 0x2026, 0};
        ImFontConfig cfg;
        cfg.MergeMode = true;
        cfg.FontDataOwnedByAtlas = false;
        cfg.OversampleH = 2;
        io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(fonts::font_ui_regular.data),
                                       static_cast<int>(fonts::font_ui_regular.size), px * raster, &cfg, extra);
        return f;
    };
    g_fonts.mono = load_mono(13.0f);
    g_fonts.mono_small = load_mono(11.5f);
    io.FontDefault = g_fonts.body;
    // Fonts are rasterised at physical resolution; scale back to logical units.
    io.FontGlobalScale = 1.0f / fb_scale;
    io.Fonts->Build();
}

} // namespace ui
