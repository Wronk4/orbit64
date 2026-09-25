#include "widgets.hpp"

#include "imgui_internal.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ui {

// ---------------------------------------------------------------------------
// Animation
// ---------------------------------------------------------------------------

float anim(ImGuiID id, float target, float speed) {
    ImGuiStorage* st = ImGui::GetStateStorage();
    // Keep a separate "initialised" flag so the first frame snaps to target.
    // Values, not Get*Ref() pointers: ImGuiStorage is a sorted vector, so
    // adding one key can reallocate it and leave a pointer to another key's
    // slot dangling - writing through it corrupted the heap.
    ImGuiID init_id = id ^ 0x5bd1e995u;
    if (!st->GetBool(init_id, false)) {
        st->SetBool(init_id, true);
        st->SetFloat(id, target);
        return target;
    }
    float v = st->GetFloat(id, target);
    float dt = ImGui::GetIO().DeltaTime;
    float k = 1.0f - std::exp(-speed * dt);
    v += (target - v) * k;
    if (std::fabs(v - target) < 0.001f) v = target;
    st->SetFloat(id, v);
    return v;
}

float anim(const char* str_id, float target, float speed) { return anim(ImGui::GetID(str_id), target, speed); }

float font_px(ImFont* f) { return f->FontSize * ImGui::GetIO().FontGlobalScale; }

// ---------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------

bool icon_button(const char* id, Icon icon, float size, const char* tip, bool active, bool enabled, ButtonKind kind) {
    ImGui::PushID(id);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 sz(size, size);
    ImGui::BeginDisabled(!enabled);
    bool pressed = ImGui::InvisibleButton(id, sz); // id doubles as a readable label for tooling
    ImGui::EndDisabled();
    bool hovered = enabled && ImGui::IsItemHovered();
    bool held = enabled && ImGui::IsItemActive();
    float h = anim(ImGui::GetID("hover"), hovered ? 1.0f : 0.0f, 18.0f);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const Palette& p = g_pal;
    ImVec4 bg(0, 0, 0, 0), fg = p.text_dim;
    switch (kind) {
        case ButtonKind::Primary:
            bg = mix(p.accent, p.accent_hover, h);
            if (held) bg = p.accent_active;
            fg = ImVec4(1, 1, 1, 1);
            break;
        case ButtonKind::Danger:
            bg = with_alpha(p.danger, 0.10f + 0.12f * h);
            fg = p.danger;
            break;
        case ButtonKind::Subtle:
            bg = mix(p.bg3, p.bg4, h);
            fg = mix(p.text_dim, p.text, h);
            break;
        case ButtonKind::Ghost:
        default:
            bg = with_alpha(p.bg4, 0.9f * h);
            fg = mix(p.text_dim, p.text, h);
            if (active) {
                bg = mix(p.accent_soft, with_alpha(p.accent, 0.24f), h);
                fg = p.accent_hover;
            }
            break;
    }
    if (held && kind != ButtonKind::Primary) bg = mix(bg, p.bg4, 0.6f);
    if (!enabled) fg = p.text_faint;
    if (bg.w > 0.001f) dl->AddRectFilled(pos, ImVec2(pos.x + size, pos.y + size), col(bg), dp(7));
    draw_icon(dl, icon, ImVec2(pos.x + size * 0.5f, pos.y + size * 0.5f + (held ? 0.5f : 0.0f)), size * 0.5f, col(fg));
    if (tip && hovered) tooltip(tip);
    ImGui::PopID();
    return pressed;
}

bool button(const char* label, Icon icon, ButtonKind kind, float width, bool enabled, float height) {
    ImGui::PushID(label);
    const char* label_end = ImGui::FindRenderedTextEnd(label);
    ImFont* font = g_fonts.body_bold;
    ImGui::PushFont(font);
    ImVec2 ts = ImGui::CalcTextSize(label, label_end);
    ImGui::PopFont();
    float icon_sz = dp(15);
    float pad_x = dp(14);
    float gap = (icon != Icon::None && ts.x > 0) ? dp(8) : 0.0f;
    float w = width > 0 ? width : ts.x + pad_x * 2 + (icon != Icon::None ? icon_sz + gap : 0);
    float hgt = height > 0 ? height : dp(34);

    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::BeginDisabled(!enabled);
    bool pressed = ImGui::InvisibleButton(label, ImVec2(w, hgt)); // label doubles as a readable id for tooling
    ImGui::EndDisabled();
    bool hovered = enabled && ImGui::IsItemHovered();
    bool held = enabled && ImGui::IsItemActive();
    float h = anim(ImGui::GetID("hover"), hovered ? 1.0f : 0.0f, 18.0f);

    const Palette& p = g_pal;
    ImVec4 bg, fg, border(0, 0, 0, 0);
    switch (kind) {
        case ButtonKind::Primary:
            bg = held ? p.accent_active : mix(p.accent, p.accent_hover, h);
            fg = ImVec4(1, 1, 1, 1);
            break;
        case ButtonKind::Danger:
            bg = held ? mix(p.danger, ImVec4(0, 0, 0, 1), 0.15f) : mix(with_alpha(p.danger, 0.85f), p.danger, h);
            fg = ImVec4(1, 1, 1, 1);
            break;
        case ButtonKind::Ghost:
            bg = with_alpha(p.bg4, h * 0.8f);
            fg = mix(p.text_dim, p.text, h);
            break;
        case ButtonKind::Subtle:
        default:
            bg = held ? p.bg4 : mix(p.bg3, p.bg4, h);
            fg = p.text;
            border = p.border_strong;
            break;
    }
    if (!enabled) {
        bg = with_alpha(bg, bg.w * 0.4f);
        fg = p.text_faint;
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 end(pos.x + w, pos.y + hgt);
    if (kind == ButtonKind::Primary && enabled) {
        // Soft glow under primary actions.
        dl->AddRectFilled(ImVec2(pos.x + dp(2), pos.y + dp(3)), ImVec2(end.x - dp(2), end.y + dp(3)),
                          col(p.accent, 0.18f + 0.12f * h), dp(8));
    }
    dl->AddRectFilled(pos, end, col(bg), dp(8));
    if (border.w > 0) dl->AddRect(pos, end, col(border, 0.6f), dp(8));

    float content_w = ts.x + (icon != Icon::None ? icon_sz + gap : 0);
    float x = pos.x + (w - content_w) * 0.5f;
    float cy = pos.y + hgt * 0.5f;
    if (icon != Icon::None) {
        draw_icon(dl, icon, ImVec2(x + icon_sz * 0.5f, cy), icon_sz, col(fg));
        x += icon_sz + gap;
    }
    dl->AddText(font, font_px(font), ImVec2(x, cy - ts.y * 0.5f), col(fg), label, label_end);
    ImGui::PopID();
    return pressed;
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

bool toggle(const char* id, bool* v) {
    ImGui::PushID(id);
    float w = dp(38), h = dp(22);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    bool pressed = ImGui::InvisibleButton("##toggle", ImVec2(w, h));
    if (pressed) *v = !*v;
    bool hovered = ImGui::IsItemHovered();
    float t = anim(ImGui::GetID("t"), *v ? 1.0f : 0.0f, 16.0f);
    float hv = anim(ImGui::GetID("h"), hovered ? 1.0f : 0.0f, 16.0f);
    const Palette& p = g_pal;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec4 track = mix(mix(p.bg4, mix(p.bg4, p.text_faint, 0.35f), hv), p.accent, t);
    dl->AddRectFilled(pos, ImVec2(pos.x + w, pos.y + h), col(track), h * 0.5f);
    float r = h * 0.5f - dp(3);
    float cx = pos.x + h * 0.5f + (w - h) * t;
    dl->AddCircleFilled(ImVec2(cx, pos.y + h * 0.5f + dp(1)), r, IM_COL32(0, 0, 0, 60), 24);
    dl->AddCircleFilled(ImVec2(cx, pos.y + h * 0.5f), r, IM_COL32(255, 255, 255, 255), 24);
    ImGui::PopID();
    return pressed;
}

bool segmented(const char* id, const char* const* items, int count, int* current, float width, const Icon* icons) {
    ImGui::PushID(id);
    ImFont* font = g_fonts.small_bold;
    float pad = dp(3);
    float h = dp(30);
    float item_w;
    if (width <= 0) {
        float maxw = 0;
        for (int i = 0; i < count; ++i) maxw = std::max(maxw, font->CalcTextSizeA(font_px(font), FLT_MAX, 0, items[i]).x);
        item_w = maxw + dp(24) + (icons ? dp(20) : 0);
        width = item_w * count + pad * 2;
    } else {
        item_w = (width - pad * 2) / count;
    }
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##seg", ImVec2(width, h));
    bool changed = false;
    ImVec2 mouse = ImGui::GetIO().MousePos;
    bool hovered = ImGui::IsItemHovered();
    int hover_idx = hovered ? std::clamp(static_cast<int>((mouse.x - pos.x - pad) / item_w), 0, count - 1) : -1;
    if (ImGui::IsItemClicked() && hover_idx >= 0 && hover_idx != *current) {
        *current = hover_idx;
        changed = true;
    }
    const Palette& p = g_pal;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + width, pos.y + h), col(p.bg1), dp(8));
    dl->AddRect(pos, ImVec2(pos.x + width, pos.y + h), col(p.border), dp(8));
    float sel_x = anim(ImGui::GetID("sel"), static_cast<float>(*current), 16.0f);
    ImVec2 a(pos.x + pad + sel_x * item_w, pos.y + pad);
    dl->AddRectFilled(a, ImVec2(a.x + item_w, pos.y + h - pad), col(p.bg4), dp(6));
    for (int i = 0; i < count; ++i) {
        float x0 = pos.x + pad + i * item_w;
        ImVec2 ts = font->CalcTextSizeA(font_px(font), FLT_MAX, 0, items[i]);
        float content = ts.x + (icons ? dp(20) : 0);
        float x = x0 + (item_w - content) * 0.5f;
        ImVec4 fc = (i == *current) ? p.text : (i == hover_idx ? mix(p.text_dim, p.text, 0.5f) : p.text_dim);
        if (icons) {
            draw_icon(dl, icons[i], ImVec2(x + dp(7), pos.y + h * 0.5f), dp(14), col(fc));
            x += dp(20);
        }
        dl->AddText(font, font_px(font), ImVec2(x, pos.y + (h - ts.y) * 0.5f), col(fc), items[i]);
    }
    ImGui::PopID();
    return changed;
}

bool slider_float(const char* id, float* v, float vmin, float vmax, const char* fmt, float width) {
    ImGui::PushID(id);
    const Palette& p = g_pal;
    char value[32];
    std::snprintf(value, sizeof value, fmt, *v);
    ImFont* font = g_fonts.mono;
    float label_w = dp(52);
    float track_w = width - label_w;
    float h = dp(24);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##slider", ImVec2(track_w, h));
    bool changed = false;
    bool active = ImGui::IsItemActive();
    bool hovered = ImGui::IsItemHovered();
    float knob_r = dp(7);
    float x0 = pos.x + knob_r, x1 = pos.x + track_w - knob_r;
    if (active) {
        float t = std::clamp((ImGui::GetIO().MousePos.x - x0) / (x1 - x0), 0.0f, 1.0f);
        float nv = vmin + t * (vmax - vmin);
        if (nv != *v) { *v = nv; changed = true; }
    }
    if (hovered && !active && ImGui::GetIO().MouseWheel != 0.0f) {
        *v = std::clamp(*v + ImGui::GetIO().MouseWheel * (vmax - vmin) * 0.02f, vmin, vmax);
        changed = true;
    }
    float t = (*v - vmin) / (vmax - vmin);
    float cy = pos.y + h * 0.5f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(ImVec2(x0, cy - dp(2)), ImVec2(x1, cy + dp(2)), col(p.bg4), dp(2));
    float kx = x0 + t * (x1 - x0);
    dl->AddRectFilled(ImVec2(x0, cy - dp(2)), ImVec2(kx, cy + dp(2)), col(p.accent), dp(2));
    float grow = anim(ImGui::GetID("k"), (hovered || active) ? 1.0f : 0.0f, 18.0f);
    if (grow > 0.01f) dl->AddCircleFilled(ImVec2(kx, cy), knob_r + dp(5) * grow, col(p.accent, 0.18f * grow), 24);
    dl->AddCircleFilled(ImVec2(kx, cy), knob_r, IM_COL32(255, 255, 255, 255), 24);
    ImVec2 ts = font->CalcTextSizeA(font_px(font), FLT_MAX, 0, value);
    dl->AddText(font, font_px(font), ImVec2(pos.x + width - ts.x, cy - ts.y * 0.5f), col(p.text_dim), value);
    ImGui::SameLine(0, 0);
    ImGui::Dummy(ImVec2(label_w, h));
    ImGui::PopID();
    return changed;
}

bool slider_int(const char* id, int* v, int vmin, int vmax, const char* fmt, float width) {
    // Reuse the float slider with integer formatting.
    float f = static_cast<float>(*v);
    std::string ffmt = fmt;
    size_t pos = ffmt.find("%d");
    if (pos != std::string::npos) ffmt.replace(pos, 2, "%.0f");
    slider_float(id, &f, static_cast<float>(vmin), static_cast<float>(vmax), ffmt.c_str(), width);
    int nv = static_cast<int>(std::lround(f));
    if (nv == *v) return false;
    *v = nv;
    return true;
}

bool combo(const char* id, int* current, const char* const* items, int count, float width) {
    ImGui::PushID(id);
    ImGui::SetNextItemWidth(width);
    bool changed = false;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, dp(10, 7));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, dp(8));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(6, 6));
    ImGui::PushStyleColor(ImGuiCol_Button, g_pal.bg3);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, g_pal.bg4);
    const char* preview = (*current >= 0 && *current < count) ? items[*current] : "";
    bool open = ImGui::BeginCombo("##combo", preview, ImGuiComboFlags_HeightLarge | ImGuiComboFlags_NoArrowButton);
    {
        ImVec2 mn = ImGui::GetItemRectMin(), mx = ImGui::GetItemRectMax();
        bool hov = ImGui::IsItemHovered();
        draw_icon(ImGui::GetWindowDrawList(), open ? Icon::ChevronUp : Icon::ChevronDown, ImVec2(mx.x - dp(16), (mn.y + mx.y) * 0.5f),
                  dp(14), col(hov || open ? g_pal.text : g_pal.text_dim));
    }
    if (open) {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, dp(4, 2));
        for (int i = 0; i < count; ++i) {
            bool sel = i == *current;
            ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0, 0.5f));
            if (ImGui::Selectable(items[i], sel, 0, ImVec2(0, dp(26)))) {
                *current = i;
                changed = true;
            }
            ImGui::PopStyleVar();
            if (sel) {
                ImVec2 mx = ImGui::GetItemRectMax();
                draw_icon(ImGui::GetWindowDrawList(), Icon::Check, ImVec2(mx.x - dp(12), (ImGui::GetItemRectMin().y + mx.y) * 0.5f),
                          dp(13), col(g_pal.accent));
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::PopStyleVar();
        ImGui::EndCombo();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
    ImGui::PopID();
    return changed;
}

bool search_field(const char* id, char* buf, size_t buf_size, const char* hint, float width) {
    ImGui::PushID(id);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(dp(32), dp(8)));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, dp(8));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, g_pal.bg2);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, g_pal.bg3);
    ImGui::SetNextItemWidth(width);
    ImGui::SetNextItemAllowOverlap(); // the clear button sits inside the field
    bool changed = ImGui::InputTextWithHint("##search", hint, buf, buf_size);
    bool focused = ImGui::IsItemActive();
    ImVec2 mn = ImGui::GetItemRectMin(), mx = ImGui::GetItemRectMax();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float f = anim(ImGui::GetID("focus"), focused ? 1.0f : 0.0f, 16.0f);
    dl->AddRect(mn, mx, col(mix(g_pal.border, g_pal.accent, f)), dp(8), 0, dp(1));
    draw_icon(dl, Icon::Search, ImVec2(pos.x + dp(16), (mn.y + mx.y) * 0.5f), dp(14), col(g_pal.text_faint));
    if (buf[0]) {
        ImVec2 save = ImGui::GetCursorScreenPos();
        float s = dp(20);
        ImGui::SetCursorScreenPos(ImVec2(mx.x - s - dp(6), (mn.y + mx.y) * 0.5f - s * 0.5f));
        if (icon_button("clear", Icon::Close, s, "Clear search")) {
            buf[0] = 0;
            changed = true;
        }
        ImGui::SetCursorScreenPos(save);
    }
    ImGui::PopID();
    return changed;
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

void section_title(const char* title, const char* subtitle) {
    ImGui::PushFont(g_fonts.title);
    ImGui::TextUnformatted(title);
    ImGui::PopFont();
    if (subtitle) {
        ImGui::PushStyleColor(ImGuiCol_Text, g_pal.text_dim);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x);
        ImGui::TextUnformatted(subtitle);
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    }
    ImGui::Dummy(dp(0, 6));
}

namespace {
struct RowState {
    ImVec2 start;
    float left_bottom;
    float width;
};
RowState g_row;
} // namespace

void row_begin(const char* label, const char* description, float control_width) {
    ImGui::PushID(label);
    g_row.start = ImGui::GetCursorScreenPos();
    g_row.width = ImGui::GetContentRegionAvail().x;
    float text_w = std::max(dp(120), g_row.width - control_width - dp(28));

    ImGui::BeginGroup();
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + text_w);
    ImGui::PushFont(g_fonts.body_bold);
    ImGui::TextUnformatted(label);
    ImGui::PopFont();
    if (description && *description) {
        ImGui::PushFont(g_fonts.small);
        ImGui::PushStyleColor(ImGuiCol_Text, g_pal.text_dim);
        ImGui::TextUnformatted(description);
        ImGui::PopStyleColor();
        ImGui::PopFont();
    }
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
    g_row.left_bottom = ImGui::GetItemRectMax().y;
    float left_h = g_row.left_bottom - g_row.start.y;
    float ctrl_h = dp(30);
    float y = g_row.start.y + std::max(0.0f, (left_h - ctrl_h) * 0.5f);
    ImGui::SetCursorScreenPos(ImVec2(g_row.start.x + g_row.width - control_width, y));
}

void row_end() {
    float bottom = std::max(g_row.left_bottom, ImGui::GetItemRectMax().y);
    ImGui::SetCursorScreenPos(ImVec2(g_row.start.x, bottom + dp(12)));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    dl->AddLine(p, ImVec2(p.x + g_row.width, p.y), col(g_pal.border, 0.7f));
    ImGui::Dummy(ImVec2(g_row.width, dp(12)));
    ImGui::PopID();
}

void badge_at(ImDrawList* dl, ImVec2 pos, const char* text, const ImVec4& color, bool filled) {
    ImFont* f = g_fonts.small_bold;
    ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, text);
    ImVec2 end(pos.x + ts.x + dp(12), pos.y + ts.y + dp(4));
    dl->AddRectFilled(pos, end, col(color, filled ? 1.0f : 0.16f), dp(5));
    dl->AddText(f, font_px(f), ImVec2(pos.x + dp(6), pos.y + dp(2)), filled ? IM_COL32(255, 255, 255, 255) : col(color), text);
}

void badge(const char* text, const ImVec4& color, bool filled) {
    ImFont* f = g_fonts.small_bold;
    ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, text);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    badge_at(ImGui::GetWindowDrawList(), pos, text, color, filled);
    ImGui::Dummy(ImVec2(ts.x + dp(12), ts.y + dp(4)));
}

void kbd(const char* text) {
    ImFont* f = g_fonts.mono_small;
    ImVec2 ts = f->CalcTextSizeA(font_px(f), FLT_MAX, 0, text);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 end(pos.x + ts.x + dp(12), pos.y + ts.y + dp(7));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(end.x, end.y), col(g_pal.border_strong), dp(5));
    dl->AddRectFilled(pos, ImVec2(end.x, end.y - dp(2)), col(g_pal.bg3), dp(5));
    dl->AddText(f, font_px(f), ImVec2(pos.x + dp(6), pos.y + dp(2.5f)), col(g_pal.text), text);
    ImGui::Dummy(ImVec2(end.x - pos.x, end.y - pos.y));
}

void spinner(float radius, float thickness, ImU32 color) {
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(radius * 2, radius * 2));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float t = static_cast<float>(ImGui::GetTime());
    ImVec2 c(pos.x + radius, pos.y + radius);
    dl->AddCircle(c, radius - thickness * 0.5f, (color & 0x00FFFFFF) | 0x30000000, 32, thickness);
    float a0 = t * 5.0f, a1 = a0 + 1.6f + std::sin(t * 2.0f) * 0.8f;
    dl->PathArcTo(c, radius - thickness * 0.5f, a0, a1, 24);
    dl->PathStroke(color, ImDrawFlags_None, thickness);
}

void tooltip(const char* text) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(10, 7));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, dp(6));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, g_pal.bg4);
    ImGui::PushStyleColor(ImGuiCol_Border, g_pal.border_strong);
    if (ImGui::BeginTooltip()) {
        ImGui::PushFont(g_fonts.small);
        ImGui::PushTextWrapPos(dp(320));
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::PopFont();
        ImGui::EndTooltip();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
}

void help_marker(const char* text) {
    ImVec2 pos = ImGui::GetCursorScreenPos();
    float s = dp(16);
    ImGui::Dummy(ImVec2(s, s));
    bool hov = ImGui::IsItemHovered();
    draw_icon(ImGui::GetWindowDrawList(), Icon::Info, ImVec2(pos.x + s * 0.5f, pos.y + s * 0.5f), s * 0.85f,
              col(hov ? g_pal.text : g_pal.text_faint));
    if (hov) tooltip(text);
}

void gradient_rect_v(ImDrawList* dl, ImVec2 mn, ImVec2 mx, ImU32 rgb, int a0, int a1, float rounding, ImDrawFlags flags) {
    int v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(mn, mx, rgb | IM_COL32_A_MASK, rounding, flags);
    const float h = std::max(1.0f, mx.y - mn.y);
    for (int i = v0; i < dl->VtxBuffer.Size; ++i) {
        ImDrawVert& v = dl->VtxBuffer[i];
        float t = std::clamp((v.pos.y - mn.y) / h, 0.0f, 1.0f);
        int a = static_cast<int>(a0 + (a1 - a0) * t);
        // Anti-aliased fringe vertices already carry alpha 0; keep them transparent.
        int fringe = (v.col >> IM_COL32_A_SHIFT) & 0xFF;
        a = a * fringe / 255;
        v.col = (rgb & ~IM_COL32_A_MASK) | (static_cast<ImU32>(a) << IM_COL32_A_SHIFT);
    }
}

void text_ellipsis(ImDrawList* dl, ImFont* font, ImVec2 pos, float max_width, ImU32 color, const char* text) {
    float px = font_px(font);
    ImVec2 ts = font->CalcTextSizeA(px, FLT_MAX, 0, text);
    if (ts.x <= max_width) {
        dl->AddText(font, px, pos, color, text);
        return;
    }
    // U+2026 when the font has it (the monospace font may not), else three dots.
    const char* ell = font->FindGlyphNoFallback(0x2026) ? "\xE2\x80\xA6" : "...";
    float ell_w = font->CalcTextSizeA(px, FLT_MAX, 0, ell).x;
    const char* end = text;
    const char* s = text;
    float w = 0;
    while (*s) {
        unsigned int c;
        int n = ImTextCharFromUtf8(&c, s, nullptr);
        float cw = font->CalcTextSizeA(px, FLT_MAX, 0, s, s + n).x;
        if (w + cw + ell_w > max_width) break;
        w += cw;
        s += n;
        end = s;
    }
    dl->AddText(font, px, pos, color, text, end);
    dl->AddText(font, px, ImVec2(pos.x + w, pos.y), color, ell);
}

// ---------------------------------------------------------------------------
// Modals
// ---------------------------------------------------------------------------

bool begin_modal(const char* id, const char* title, ImVec2 size, bool* open, Icon icon) {
    ImGuiStorage* st = ImGui::GetStateStorage();
    ImGuiID gid = ImGui::GetID(id);
    bool was_open = ImGui::IsPopupOpen(id);
    if (*open && !was_open) {
        ImGui::OpenPopup(id);
        st->SetFloat(gid + 1, static_cast<float>(ImGui::GetTime()));
    }
    if (!*open && !was_open) return false;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImVec2 max_sz(vp->WorkSize.x - dp(32), vp->WorkSize.y - dp(32));
    size.x = std::min(size.x, max_sz.x);
    size.y = std::min(size.y, max_sz.y);
    float t = std::clamp((static_cast<float>(ImGui::GetTime()) - st->GetFloat(gid + 1, 0.0f)) / 0.18f, 0.0f, 1.0f);
    float ease = 1.0f - (1.0f - t) * (1.0f - t) * (1.0f - t);
    ImVec2 center(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y * 0.5f + (1.0f - ease) * dp(14));
    ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);

    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ease);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, dp(12));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, g_pal.bg1);
    ImGui::PushStyleColor(ImGuiCol_Border, g_pal.border_strong);
    ImGui::PushStyleColor(ImGuiCol_ModalWindowDimBg, with_alpha(g_pal.overlay, g_pal.overlay.w * ease));
    bool visible = ImGui::BeginPopupModal(id, nullptr,
                                          ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                              ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar |
                                              ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(4);
    if (!visible) return false;

    // Header
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    float hh = dp(56);
    float x = wp.x + dp(20);
    if (icon != Icon::None) {
        ImVec2 ic(x + dp(15), wp.y + hh * 0.5f);
        const ImVec4 tint = icon == Icon::Warning ? g_pal.warning : g_pal.accent_hover;
        dl->AddRectFilled(ImVec2(ic.x - dp(15), ic.y - dp(15)), ImVec2(ic.x + dp(15), ic.y + dp(15)), col(tint, 0.16f), dp(8));
        draw_icon(dl, icon, ic, dp(16), col(tint));
        x += dp(42);
    }
    ImFont* tf = g_fonts.title;
    dl->AddText(tf, font_px(tf) * 0.9f, ImVec2(x, wp.y + hh * 0.5f - font_px(tf) * 0.45f), col(g_pal.text), title);
    ImGui::SetCursorPos(ImVec2(size.x - dp(20) - dp(30), (hh - dp(30)) * 0.5f));
    if (icon_button("##modal_close", Icon::Close, dp(30), nullptr)) *open = false;
    dl->AddLine(ImVec2(wp.x, wp.y + hh), ImVec2(wp.x + size.x, wp.y + hh), col(g_pal.border));
    ImGui::SetCursorPos(ImVec2(0, hh));
    ImGui::Dummy(ImVec2(0, 0));
    ImGui::SetCursorPos(ImVec2(0, hh));

    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) && ImGui::IsKeyPressed(ImGuiKey_Escape, false) &&
        !ImGui::IsAnyItemActive())
        *open = false;
    if (!*open) ImGui::CloseCurrentPopup();
    return true;
}

void end_modal() { ImGui::EndPopup(); }

void modal_footer_begin(float height) {
    ImVec2 wp = ImGui::GetWindowPos();
    ImVec2 ws = ImGui::GetWindowSize();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float y = wp.y + ws.y - height;
    dl->AddRectFilled(ImVec2(wp.x, y), ImVec2(wp.x + ws.x, wp.y + ws.y), col(g_pal.bg0, 0.6f), dp(12),
                      ImDrawFlags_RoundCornersBottom);
    dl->AddLine(ImVec2(wp.x, y), ImVec2(wp.x + ws.x, y), col(g_pal.border));
    ImGui::SetCursorPos(ImVec2(dp(20), ws.y - height + (height - dp(34)) * 0.5f));
    ImGui::BeginGroup();
}

void modal_footer_end() {
    ImGui::Dummy(ImVec2(0, 0)); // validates the cursor move even when the footer is empty
    ImGui::EndGroup();
}

// ---------------------------------------------------------------------------
// Toasts
// ---------------------------------------------------------------------------

namespace {
struct Toast {
    std::string text;
    ToastKind kind;
    double start;
    float duration;
};
std::vector<Toast> g_toasts;
} // namespace

void toast(const std::string& text, ToastKind kind, float seconds) {
    g_toasts.push_back({text, kind, ImGui::GetTime(), seconds});
    if (g_toasts.size() > 4) g_toasts.erase(g_toasts.begin());
}

void draw_toasts(ImVec2 br) {
    double now = ImGui::GetTime();
    g_toasts.erase(std::remove_if(g_toasts.begin(), g_toasts.end(),
                                  [&](const Toast& t) { return now - t.start > t.duration + 0.35; }),
                   g_toasts.end());
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    float y = br.y;
    ImFont* f = g_fonts.body;
    for (int i = static_cast<int>(g_toasts.size()) - 1; i >= 0; --i) {
        const Toast& t = g_toasts[i];
        float age = static_cast<float>(now - t.start);
        float in = std::clamp(age / 0.25f, 0.0f, 1.0f);
        float out = std::clamp((t.duration + 0.35f - age) / 0.35f, 0.0f, 1.0f);
        float a = std::min(in, out);
        float ease = 1.0f - (1.0f - a) * (1.0f - a);
        ImVec2 ts = f->CalcTextSizeA(font_px(f), dp(360), dp(360), t.text.c_str());
        float w = ts.x + dp(56), h = std::max(dp(44), ts.y + dp(24));
        float x = br.x - w + (1.0f - ease) * dp(24);
        ImVec2 mn(x, y - h), mx(x + w, y);
        ImVec4 accent = t.kind == ToastKind::Success ? g_pal.success
                      : t.kind == ToastKind::Warning ? g_pal.warning
                      : t.kind == ToastKind::Error   ? g_pal.danger
                                                     : g_pal.accent;
        Icon icon = t.kind == ToastKind::Success ? Icon::Check
                  : (t.kind == ToastKind::Warning || t.kind == ToastKind::Error) ? Icon::Warning
                                                                                 : Icon::Info;
        dl->AddRectFilled(ImVec2(mn.x, mn.y + dp(4)), ImVec2(mx.x, mx.y + dp(6)), IM_COL32(0, 0, 0, (int)(90 * a)), dp(10));
        dl->AddRectFilled(mn, mx, col(g_pal.bg3, a), dp(10));
        dl->AddRect(mn, mx, col(g_pal.border_strong, a), dp(10));
        dl->AddRectFilled(ImVec2(mn.x, mn.y + dp(10)), ImVec2(mn.x + dp(3), mx.y - dp(10)), col(accent, a), dp(2));
        draw_icon(dl, icon, ImVec2(mn.x + dp(24), (mn.y + mx.y) * 0.5f), dp(16), col(accent, a));
        dl->AddText(f, font_px(f), ImVec2(mn.x + dp(42), (mn.y + mx.y) * 0.5f - ts.y * 0.5f), col(g_pal.text, a),
                    t.text.c_str(), nullptr, dp(360));
        y -= (h + dp(8)) * ease;
    }
}

} // namespace ui
