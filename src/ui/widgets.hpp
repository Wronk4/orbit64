#pragma once
// Reusable styled widgets built on top of Dear ImGui.

#include "icons.hpp"
#include "theme.hpp"

#include <string>

namespace ui {

// ---- Animation ---------------------------------------------------------
// Critically-damped approach of a per-ID float towards `target`.
float anim(ImGuiID id, float target, float speed = 14.0f);
float anim(const char* str_id, float target, float speed = 14.0f);

// ---- Buttons -------------------------------------------------------------
enum class ButtonKind { Ghost, Subtle, Primary, Danger };

bool icon_button(const char* id, Icon icon, float size, const char* tooltip = nullptr, bool active = false,
                 bool enabled = true, ButtonKind kind = ButtonKind::Ghost);
bool button(const char* label, Icon icon = Icon::None, ButtonKind kind = ButtonKind::Subtle, float width = 0.0f,
            bool enabled = true, float height = 0.0f);

// ---- Inputs --------------------------------------------------------------
bool toggle(const char* id, bool* v);
bool segmented(const char* id, const char* const* items, int count, int* current, float width = 0.0f,
               const Icon* icons = nullptr);
bool slider_float(const char* id, float* v, float min, float max, const char* fmt, float width);
bool slider_int(const char* id, int* v, int min, int max, const char* fmt, float width);
bool combo(const char* id, int* current, const char* const* items, int count, float width);
bool search_field(const char* id, char* buf, size_t buf_size, const char* hint, float width);

// ---- Layout helpers ------------------------------------------------------
void section_title(const char* title, const char* subtitle = nullptr);
// A settings row: label + description on the left, control on the right.
// Returns the x position where the control should start; call row_end() after.
void row_begin(const char* label, const char* description, float control_width);
void row_end();
void badge(const char* text, const ImVec4& color, bool filled = false);
void badge_at(ImDrawList* dl, ImVec2 pos, const char* text, const ImVec4& color, bool filled = false);
void kbd(const char* text);
void spinner(float radius, float thickness, ImU32 color);
void tooltip(const char* text);
void help_marker(const char* text);

// Vertical alpha gradient over a (optionally rounded) rect: `rgb` fades from
// alpha a0 at the top to a1 at the bottom.
void gradient_rect_v(ImDrawList* dl, ImVec2 mn, ImVec2 mx, ImU32 rgb, int a0, int a1, float rounding = 0.0f,
                     ImDrawFlags flags = 0);

// Renders `text` clipped with an ellipsis to `max_width`.
void text_ellipsis(ImDrawList* dl, ImFont* font, ImVec2 pos, float max_width, ImU32 color, const char* text);
float font_px(ImFont* f);

// ---- Modals --------------------------------------------------------------
// Animated modal dialog with a custom header (icon, title, close button).
// Opens when *open becomes true, closes on Esc / close button.
bool begin_modal(const char* id, const char* title, ImVec2 size, bool* open, Icon icon = Icon::None);
void end_modal();
// Footer strip at the bottom of a modal; the caller lays out its buttons
// inside the begin/end pair.
void modal_footer_begin(float height);
void modal_footer_end();

// ---- Toasts --------------------------------------------------------------
enum class ToastKind { Info, Success, Warning, Error };
void toast(const std::string& text, ToastKind kind = ToastKind::Info, float seconds = 3.0f);
void draw_toasts(ImVec2 anchor_bottom_right);

} // namespace ui
