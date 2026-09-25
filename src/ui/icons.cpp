#include "icons.hpp"

#include <cmath>
#include <initializer_list>

namespace ui {

namespace {

constexpr float kPi = 3.14159265358979f;

struct Pen {
    ImDrawList* dl;
    ImVec2 c;
    float s;
    ImU32 col;
    float t; // stroke thickness in pixels

    ImVec2 p(float x, float y) const { return ImVec2(c.x + x * s, c.y + y * s); }

    void line(float x0, float y0, float x1, float y1) const { dl->AddLine(p(x0, y0), p(x1, y1), col, t); }

    void poly(std::initializer_list<ImVec2> pts, bool closed) const {
        dl->PathClear();
        for (const auto& q : pts) dl->PathLineTo(p(q.x, q.y));
        dl->PathStroke(col, closed ? ImDrawFlags_Closed : ImDrawFlags_None, t);
    }
    void fill(std::initializer_list<ImVec2> pts) const {
        dl->PathClear();
        for (const auto& q : pts) dl->PathLineTo(p(q.x, q.y));
        dl->PathFillConvex(col);
    }
    void rect(float x0, float y0, float x1, float y1, float r) const {
        dl->AddRect(p(x0, y0), p(x1, y1), col, r * s, 0, t);
    }
    void rect_fill(float x0, float y0, float x1, float y1, float r) const {
        dl->AddRectFilled(p(x0, y0), p(x1, y1), col, r * s);
    }
    void circle(float x, float y, float r) const { dl->AddCircle(p(x, y), r * s, col, 0, t); }
    void dot(float x, float y, float r) const { dl->AddCircleFilled(p(x, y), r * s, col); }
    void arc(float x, float y, float r, float a0, float a1) const {
        dl->PathClear();
        dl->PathArcTo(p(x, y), r * s, a0, a1);
        dl->PathStroke(col, ImDrawFlags_None, t);
    }
    // Small filled arrow head at the end of an arc, pointing along `dir` (radians).
    void arrow_head(float x, float y, float dir, float len) const {
        ImVec2 tip = p(x, y);
        float l = len * s;
        ImVec2 a(tip.x - std::cos(dir - 0.55f) * l, tip.y - std::sin(dir - 0.55f) * l);
        ImVec2 b(tip.x - std::cos(dir + 0.55f) * l, tip.y - std::sin(dir + 0.55f) * l);
        dl->AddTriangleFilled(tip, a, b, col);
    }
};

void star(const Pen& pen, bool filled) {
    ImVec2 pts[10];
    for (int i = 0; i < 10; ++i) {
        float r = (i % 2 == 0) ? 0.44f : 0.19f;
        float a = -kPi / 2 + i * kPi / 5;
        pts[i] = pen.p(std::cos(a) * r, std::sin(a) * r + 0.03f);
    }
    if (filled) {
        // Star is star-shaped around its centre, so a triangle fan is exact.
        ImVec2 ctr = pen.p(0, 0.03f);
        for (int i = 0; i < 10; ++i) pen.dl->AddTriangleFilled(ctr, pts[i], pts[(i + 1) % 10], pen.col);
    } else {
        pen.dl->AddPolyline(pts, 10, pen.col, ImDrawFlags_Closed, pen.t);
    }
}

void gear(const Pen& pen) {
    const int teeth = 8;
    pen.dl->PathClear();
    for (int i = 0; i < teeth; ++i) {
        float base = i * 2 * kPi / teeth;
        float w = kPi / teeth;
        float ro = 0.44f, ri = 0.33f;
        float a0 = base - w * 0.55f, a1 = base - w * 0.32f, a2 = base + w * 0.32f, a3 = base + w * 0.55f;
        pen.dl->PathLineTo(pen.p(std::cos(a0) * ri, std::sin(a0) * ri));
        pen.dl->PathLineTo(pen.p(std::cos(a1) * ro, std::sin(a1) * ro));
        pen.dl->PathLineTo(pen.p(std::cos(a2) * ro, std::sin(a2) * ro));
        pen.dl->PathLineTo(pen.p(std::cos(a3) * ri, std::sin(a3) * ri));
        float a4 = base + w * 1.45f;
        pen.dl->PathLineTo(pen.p(std::cos(a4) * ri, std::sin(a4) * ri));
    }
    pen.dl->PathStroke(pen.col, ImDrawFlags_Closed, pen.t);
    pen.circle(0, 0, 0.13f);
}

} // namespace

void draw_icon(ImDrawList* dl, Icon icon, ImVec2 center, float size, ImU32 col) {
    // Snap to pixel centre so 1-2px strokes stay sharp.
    center.x = std::floor(center.x) + 0.5f;
    center.y = std::floor(center.y) + 0.5f;
    Pen g{dl, center, size, col, std::fmax(1.25f, size * 0.085f)};

    switch (icon) {
        case Icon::None: break;
        case Icon::Play: g.fill({{-0.24f, -0.36f}, {0.38f, 0.0f}, {-0.24f, 0.36f}}); break;
        case Icon::Pause:
            g.rect_fill(-0.30f, -0.34f, -0.08f, 0.34f, 0.05f);
            g.rect_fill(0.08f, -0.34f, 0.30f, 0.34f, 0.05f);
            break;
        case Icon::Stop: g.rect_fill(-0.30f, -0.30f, 0.30f, 0.30f, 0.07f); break;
        case Icon::Reset:
            g.arc(0, 0.02f, 0.32f, -kPi * 0.5f, kPi * 1.15f);
            g.arrow_head(0.02f, -0.30f, 0.0f, 0.2f);
            break;
        case Icon::Refresh:
            g.arc(0, 0, 0.32f, -kPi * 0.95f, -kPi * 0.15f);
            g.arrow_head(0.29f, -0.14f, kPi * 0.4f, 0.18f);
            g.arc(0, 0, 0.32f, kPi * 0.05f, kPi * 0.85f);
            g.arrow_head(-0.29f, 0.14f, -kPi * 0.6f, 0.18f);
            break;
        case Icon::FastForward:
            g.fill({{-0.42f, -0.30f}, {0.0f, 0.0f}, {-0.42f, 0.30f}});
            g.fill({{0.0f, -0.30f}, {0.42f, 0.0f}, {0.0f, 0.30f}});
            break;
        case Icon::Step:
            g.fill({{-0.32f, -0.32f}, {0.18f, 0.0f}, {-0.32f, 0.32f}});
            g.rect_fill(0.2f, -0.32f, 0.32f, 0.32f, 0.03f);
            break;
        case Icon::Folder:
            g.poly({{-0.42f, -0.30f}, {-0.12f, -0.30f}, {-0.03f, -0.19f}, {0.42f, -0.19f}, {0.42f, 0.32f}, {-0.42f, 0.32f}}, true);
            break;
        case Icon::FolderOpen:
            g.poly({{-0.34f, 0.32f}, {-0.42f, 0.32f}, {-0.42f, -0.30f}, {-0.12f, -0.30f}, {-0.03f, -0.19f}, {0.32f, -0.19f}, {0.32f, -0.04f}}, false);
            g.poly({{-0.42f, 0.32f}, {-0.26f, -0.04f}, {0.48f, -0.04f}, {0.32f, 0.32f}}, true);
            break;
        case Icon::File:
            g.poly({{-0.30f, -0.42f}, {0.10f, -0.42f}, {0.32f, -0.20f}, {0.32f, 0.42f}, {-0.30f, 0.42f}}, true);
            g.poly({{0.10f, -0.42f}, {0.10f, -0.20f}, {0.32f, -0.20f}}, false);
            break;
        case Icon::Chip: // N64 cartridge silhouette
            g.poly({{-0.36f, -0.40f}, {0.36f, -0.40f}, {0.36f, 0.26f}, {0.28f, 0.26f}, {0.28f, 0.40f}, {-0.28f, 0.40f}, {-0.28f, 0.26f}, {-0.36f, 0.26f}}, true);
            g.rect(-0.22f, -0.26f, 0.22f, 0.08f, 0.04f);
            g.line(-0.16f, 0.26f, 0.16f, 0.26f);
            break;
        case Icon::Library:
            g.rect(-0.40f, -0.36f, -0.18f, 0.38f, 0.03f);
            g.rect(-0.12f, -0.36f, 0.10f, 0.38f, 0.03f);
            g.poly({{0.16f, -0.30f}, {0.34f, -0.36f}, {0.46f, 0.32f}, {0.28f, 0.38f}}, true);
            break;
        case Icon::Grid:
            g.rect(-0.38f, -0.38f, -0.06f, -0.06f, 0.06f);
            g.rect(0.06f, -0.38f, 0.38f, -0.06f, 0.06f);
            g.rect(-0.38f, 0.06f, -0.06f, 0.38f, 0.06f);
            g.rect(0.06f, 0.06f, 0.38f, 0.38f, 0.06f);
            break;
        case Icon::List:
            for (int i = -1; i <= 1; ++i) {
                g.dot(-0.34f, i * 0.26f, 0.055f);
                g.line(-0.18f, i * 0.26f, 0.40f, i * 0.26f);
            }
            break;
        case Icon::Settings: gear(g); break;
        case Icon::Fullscreen:
            g.poly({{-0.38f, -0.12f}, {-0.38f, -0.38f}, {-0.12f, -0.38f}}, false);
            g.poly({{0.12f, -0.38f}, {0.38f, -0.38f}, {0.38f, -0.12f}}, false);
            g.poly({{0.38f, 0.12f}, {0.38f, 0.38f}, {0.12f, 0.38f}}, false);
            g.poly({{-0.12f, 0.38f}, {-0.38f, 0.38f}, {-0.38f, 0.12f}}, false);
            break;
        case Icon::ExitFullscreen:
            g.poly({{-0.38f, -0.14f}, {-0.14f, -0.14f}, {-0.14f, -0.38f}}, false);
            g.poly({{0.14f, -0.38f}, {0.14f, -0.14f}, {0.38f, -0.14f}}, false);
            g.poly({{0.38f, 0.14f}, {0.14f, 0.14f}, {0.14f, 0.38f}}, false);
            g.poly({{-0.14f, 0.38f}, {-0.14f, 0.14f}, {-0.38f, 0.14f}}, false);
            break;
        case Icon::Camera:
            g.poly({{-0.42f, -0.20f}, {-0.18f, -0.20f}, {-0.10f, -0.34f}, {0.10f, -0.34f}, {0.18f, -0.20f}, {0.42f, -0.20f}, {0.42f, 0.32f}, {-0.42f, 0.32f}}, true);
            g.circle(0, 0.05f, 0.15f);
            break;
        case Icon::Gamepad:
            g.dl->PathClear();
            g.dl->PathArcTo(g.p(-0.26f, 0.0f), 0.2f * g.s, kPi * 0.5f, kPi * 1.5f);
            g.dl->PathArcTo(g.p(0.26f, 0.0f), 0.2f * g.s, -kPi * 0.5f, kPi * 0.5f);
            g.dl->PathStroke(g.col, ImDrawFlags_Closed, g.t);
            g.line(-0.34f, 0.0f, -0.18f, 0.0f);
            g.line(-0.26f, -0.08f, -0.26f, 0.08f);
            g.dot(0.22f, 0.05f, 0.045f);
            g.dot(0.32f, -0.05f, 0.045f);
            break;
        case Icon::Volume:
        case Icon::VolumeMute:
            g.fill({{-0.40f, -0.12f}, {-0.22f, -0.12f}, {-0.22f, 0.12f}, {-0.40f, 0.12f}});
            g.fill({{-0.22f, -0.12f}, {0.02f, -0.34f}, {0.02f, 0.34f}, {-0.22f, 0.12f}});
            if (icon == Icon::Volume) {
                g.arc(0.02f, 0, 0.18f, -kPi * 0.3f, kPi * 0.3f);
                g.arc(0.02f, 0, 0.34f, -kPi * 0.3f, kPi * 0.3f);
            } else {
                g.line(0.14f, -0.13f, 0.40f, 0.13f);
                g.line(0.14f, 0.13f, 0.40f, -0.13f);
            }
            break;
        case Icon::Speaker:
            g.rect(-0.30f, -0.42f, 0.30f, 0.42f, 0.08f);
            g.circle(0, 0.12f, 0.14f);
            g.dot(0, -0.24f, 0.06f);
            break;
        case Icon::Monitor:
            g.rect(-0.42f, -0.34f, 0.42f, 0.20f, 0.05f);
            g.line(0, 0.20f, 0, 0.36f);
            g.line(-0.2f, 0.37f, 0.2f, 0.37f);
            break;
        case Icon::Cpu:
            g.rect(-0.26f, -0.26f, 0.26f, 0.26f, 0.05f);
            g.rect_fill(-0.1f, -0.1f, 0.1f, 0.1f, 0.02f);
            for (int i = -1; i <= 1; ++i) {
                float o = i * 0.14f;
                g.line(o, -0.42f, o, -0.26f);
                g.line(o, 0.26f, o, 0.42f);
                g.line(-0.42f, o, -0.26f, o);
                g.line(0.26f, o, 0.42f, o);
            }
            break;
        case Icon::Info:
            g.circle(0, 0, 0.42f);
            g.line(0, -0.04f, 0, 0.22f);
            g.dot(0, -0.18f, 0.055f);
            break;
        case Icon::Warning:
            g.poly({{0.0f, -0.40f}, {0.44f, 0.36f}, {-0.44f, 0.36f}}, true);
            g.line(0, -0.12f, 0, 0.12f);
            g.dot(0, 0.24f, 0.045f);
            break;
        case Icon::Search:
            g.circle(-0.07f, -0.07f, 0.28f);
            g.line(0.14f, 0.14f, 0.40f, 0.40f);
            break;
        case Icon::Star: star(g, false); break;
        case Icon::StarFilled: star(g, true); break;
        case Icon::Sparkle:
            g.fill({{0, -0.44f}, {0.1f, -0.1f}, {0.44f, 0}, {0.1f, 0.1f}});
            g.fill({{0.44f, 0}, {0.1f, 0.1f}, {0, 0.44f}, {-0.1f, 0.1f}});
            g.fill({{0, 0.44f}, {-0.1f, 0.1f}, {-0.44f, 0}, {-0.1f, -0.1f}});
            g.fill({{-0.44f, 0}, {-0.1f, -0.1f}, {0, -0.44f}, {0.1f, -0.1f}});
            break;
        case Icon::Clock:
            g.circle(0, 0, 0.42f);
            g.poly({{0.0f, -0.24f}, {0.0f, 0.0f}, {0.17f, 0.12f}}, false);
            break;
        case Icon::Close:
            g.line(-0.28f, -0.28f, 0.28f, 0.28f);
            g.line(-0.28f, 0.28f, 0.28f, -0.28f);
            break;
        case Icon::Check: g.poly({{-0.34f, 0.02f}, {-0.10f, 0.26f}, {0.36f, -0.24f}}, false); break;
        case Icon::ChevronDown: g.poly({{-0.26f, -0.12f}, {0.0f, 0.14f}, {0.26f, -0.12f}}, false); break;
        case Icon::ChevronUp: g.poly({{-0.26f, 0.12f}, {0.0f, -0.14f}, {0.26f, 0.12f}}, false); break;
        case Icon::ChevronRight: g.poly({{-0.12f, -0.26f}, {0.14f, 0.0f}, {-0.12f, 0.26f}}, false); break;
        case Icon::ChevronLeft: g.poly({{0.12f, -0.26f}, {-0.14f, 0.0f}, {0.12f, 0.26f}}, false); break;
        case Icon::ArrowUp:
            g.line(0, -0.36f, 0, 0.36f);
            g.poly({{-0.26f, -0.1f}, {0.0f, -0.36f}, {0.26f, -0.1f}}, false);
            break;
        case Icon::More:
            g.dot(-0.30f, 0, 0.07f);
            g.dot(0, 0, 0.07f);
            g.dot(0.30f, 0, 0.07f);
            break;
        case Icon::Trash:
            g.line(-0.38f, -0.26f, 0.38f, -0.26f);
            g.poly({{-0.12f, -0.26f}, {-0.12f, -0.38f}, {0.12f, -0.38f}, {0.12f, -0.26f}}, false);
            g.poly({{-0.28f, -0.26f}, {-0.22f, 0.40f}, {0.22f, 0.40f}, {0.28f, -0.26f}}, false);
            break;
        case Icon::Plus:
            g.line(-0.32f, 0, 0.32f, 0);
            g.line(0, -0.32f, 0, 0.32f);
            break;
        case Icon::Minus: g.line(-0.32f, 0, 0.32f, 0); break;
        case Icon::Home:
            g.poly({{-0.42f, -0.02f}, {0.0f, -0.40f}, {0.42f, -0.02f}}, false);
            g.poly({{-0.30f, -0.12f}, {-0.30f, 0.38f}, {0.30f, 0.38f}, {0.30f, -0.12f}}, false);
            break;
        case Icon::Download:
            g.line(0, -0.40f, 0, 0.12f);
            g.poly({{-0.20f, -0.08f}, {0.0f, 0.12f}, {0.20f, -0.08f}}, false);
            g.poly({{-0.40f, 0.14f}, {-0.40f, 0.38f}, {0.40f, 0.38f}, {0.40f, 0.14f}}, false);
            break;
        case Icon::Drive:
            g.rect(-0.42f, -0.18f, 0.42f, 0.24f, 0.08f);
            g.dot(0.26f, 0.03f, 0.05f);
            g.line(-0.28f, 0.03f, 0.08f, 0.03f);
            break;
        case Icon::Sliders:
            for (int i = -1; i <= 1; ++i) {
                float y = i * 0.28f;
                g.line(-0.40f, y, 0.40f, y);
                float kx = (i == -1) ? 0.16f : (i == 0 ? -0.18f : 0.06f);
                g.dl->AddCircleFilled(g.p(kx, y), 0.1f * g.s, g.col);
            }
            break;
        case Icon::Keyboard:
            g.rect(-0.44f, -0.26f, 0.44f, 0.28f, 0.06f);
            for (int r = 0; r < 2; ++r)
                for (int k = 0; k < 5; ++k) g.dot(-0.28f + k * 0.14f, -0.10f + r * 0.13f, 0.035f);
            g.line(-0.18f, 0.16f, 0.18f, 0.16f);
            break;
        case Icon::Sidebar:
            g.rect(-0.42f, -0.34f, 0.42f, 0.34f, 0.06f);
            g.line(0.12f, -0.34f, 0.12f, 0.34f);
            break;
        case Icon::Globe:
            g.circle(0, 0, 0.42f);
            g.line(-0.42f, 0, 0.42f, 0);
            g.dl->AddEllipse(g.p(0, 0), ImVec2(0.18f * g.s, 0.42f * g.s), g.col, 0.0f, 0, g.t);
            break;
        case Icon::Wave: {
            ImVec2 pts[24];
            for (int i = 0; i < 24; ++i) {
                float x = -0.44f + i * (0.88f / 23.0f);
                pts[i] = g.p(x, std::sin(x * 9.0f) * 0.24f);
            }
            dl->AddPolyline(pts, 24, col, ImDrawFlags_None, g.t);
            break;
        }
        case Icon::Bug:
            g.dl->AddEllipse(g.p(0, 0.08f), ImVec2(0.22f * g.s, 0.30f * g.s), g.col, 0.0f, 0, g.t);
            g.line(0, -0.10f, 0, 0.38f);
            g.arc(0, -0.22f, 0.13f, kPi, kPi * 2.0f);
            for (int i = -1; i <= 1; ++i) {
                float y = 0.08f + i * 0.17f;
                g.line(-0.22f, y, -0.40f, y + i * 0.08f);
                g.line(0.22f, y, 0.40f, y + i * 0.08f);
            }
            break;
        case Icon::Cube:
            g.poly({{0.0f, -0.40f}, {0.38f, -0.20f}, {0.38f, 0.22f}, {0.0f, 0.42f}, {-0.38f, 0.22f}, {-0.38f, -0.20f}}, true);
            g.poly({{-0.38f, -0.20f}, {0.0f, 0.0f}, {0.38f, -0.20f}}, false);
            g.line(0.0f, 0.0f, 0.0f, 0.42f);
            break;
        case Icon::User:
            g.circle(0, -0.16f, 0.17f);
            g.arc(0, 0.46f, 0.36f, kPi * 1.1f, kPi * 1.9f);
            break;
        case Icon::Eye:
            g.dl->PathClear();
            g.dl->PathArcTo(g.p(0, 0.42f), 0.58f * g.s, kPi * 1.22f, kPi * 1.78f);
            g.dl->PathArcTo(g.p(0, -0.42f), 0.58f * g.s, kPi * 0.22f, kPi * 0.78f);
            g.dl->PathStroke(g.col, ImDrawFlags_Closed, g.t);
            g.circle(0, 0, 0.12f);
            break;
        case Icon::Pencil:
            g.poly({{0.22f, -0.40f}, {0.40f, -0.22f}, {-0.20f, 0.38f}, {-0.40f, 0.40f}, {-0.38f, 0.20f}}, true);
            g.line(0.10f, -0.28f, 0.28f, -0.10f);
            break;
        case Icon::Snowflake:
            for (int i = 0; i < 3; ++i) {
                float a = i * kPi / 3.0f + kPi / 2.0f;
                float cx = std::cos(a) * 0.42f, cy = std::sin(a) * 0.42f;
                g.line(-cx, -cy, cx, cy);
                for (float sgn : {1.0f, -1.0f}) {
                    float bx = cx * 0.6f * sgn, by = cy * 0.6f * sgn;
                    float px = -std::sin(a) * 0.12f, py = std::cos(a) * 0.12f;
                    g.line(bx, by, bx * 1.35f + px, by * 1.35f + py);
                    g.line(bx, by, bx * 1.35f - px, by * 1.35f - py);
                }
            }
            break;
        case Icon::Target:
            g.circle(0, 0, 0.30f);
            g.dot(0, 0, 0.07f);
            g.line(0, -0.44f, 0, -0.18f);
            g.line(0, 0.18f, 0, 0.44f);
            g.line(-0.44f, 0, -0.18f, 0);
            g.line(0.18f, 0, 0.44f, 0);
            break;
        case Icon::Image:
            g.rect(-0.42f, -0.34f, 0.42f, 0.34f, 0.06f);
            g.poly({{-0.42f, 0.24f}, {-0.12f, -0.06f}, {0.10f, 0.16f}, {0.22f, 0.04f}, {0.42f, 0.24f}}, false);
            g.dot(0.20f, -0.14f, 0.07f);
            break;
        case Icon::Hash:
            g.line(-0.12f, -0.40f, -0.20f, 0.40f);
            g.line(0.20f, -0.40f, 0.12f, 0.40f);
            g.line(-0.38f, -0.14f, 0.40f, -0.14f);
            g.line(-0.40f, 0.14f, 0.38f, 0.14f);
            break;
        case Icon::Layers:
            g.poly({{0.0f, -0.38f}, {0.42f, -0.16f}, {0.0f, 0.06f}, {-0.42f, -0.16f}}, true);
            g.poly({{-0.42f, 0.04f}, {0.0f, 0.26f}, {0.42f, 0.04f}}, false);
            g.poly({{-0.42f, 0.20f}, {0.0f, 0.42f}, {0.42f, 0.20f}}, false);
            break;
    }
}

void draw_logo(ImDrawList* dl, ImVec2 c, float size, ImU32 accent, ImU32 accent2) {
    float r = size * 0.5f;
    // Planet with a subtle two-tone fill.
    dl->AddCircleFilled(c, r * 0.62f, accent, 48);
    dl->AddCircleFilled(ImVec2(c.x + r * 0.12f, c.y + r * 0.12f), r * 0.48f, accent2, 48);
    dl->AddCircleFilled(ImVec2(c.x - r * 0.18f, c.y - r * 0.2f), r * 0.14f, IM_COL32(255, 255, 255, 90), 24);

    // Tilted orbit ring (front half drawn over the planet).
    const int n = 64;
    ImVec2 pts[n + 1];
    float tilt = -0.42f;
    for (int i = 0; i <= n; ++i) {
        float a = static_cast<float>(i) / n * 2 * kPi;
        float x = std::cos(a) * r * 0.98f, y = std::sin(a) * r * 0.36f;
        pts[i] = ImVec2(c.x + x * std::cos(tilt) - y * std::sin(tilt), c.y + x * std::sin(tilt) + y * std::cos(tilt));
    }
    dl->AddPolyline(pts, n / 2 + 1, IM_COL32(255, 255, 255, 235), ImDrawFlags_None, std::fmax(1.5f, size * 0.07f));
    // Satellite.
    dl->AddCircleFilled(pts[n * 5 / 8], size * 0.085f, IM_COL32(255, 255, 255, 255), 16);
}

} // namespace ui
