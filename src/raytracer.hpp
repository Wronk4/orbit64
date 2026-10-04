#pragma once
// Ray tracing for the HLE renderer: the triangles a frame drew, in a BVH, that
// the GPU traces shadow, ambient occlusion and lighting rays against per pixel
// (RDP::rt_pixel_flush, shaders/ext/rt_trace.comp). Positions are in the space
// the modelview puts them in, which is the same for every object of a frame.

#include "common.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace orbit64::rt {

struct RTVector3 {
    f32 x{0.0f}, y{0.0f}, z{0.0f};

    constexpr RTVector3() = default;
    constexpr RTVector3(f32 x_, f32 y_, f32 z_) : x(x_), y(y_), z(z_) {}

    constexpr RTVector3 operator+(const RTVector3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr RTVector3 operator-(const RTVector3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr RTVector3 operator-() const { return {-x, -y, -z}; }
    constexpr RTVector3 operator*(f32 s) const { return {x * s, y * s, z * s}; }
    constexpr RTVector3 operator*(const RTVector3& o) const { return {x * o.x, y * o.y, z * o.z}; }
    constexpr RTVector3 operator/(f32 s) const { f32 inv = 1.0f / s; return {x * inv, y * inv, z * inv}; }

    RTVector3& operator+=(const RTVector3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    RTVector3& operator*=(f32 s) { x *= s; y *= s; z *= s; return *this; }

    f32 dot(const RTVector3& o) const { return x * o.x + y * o.y + z * o.z; }
    RTVector3 cross(const RTVector3& o) const { return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x}; }

    f32 length_squared() const { return x * x + y * y + z * z; }
    f32 length() const { return std::sqrt(length_squared()); }

    RTVector3 normalized() const {
        f32 len = length();
        if (len > 1e-6f) return *this * (1.0f / len);
        return {0.0f, 0.0f, 1.0f};
    }
    bool finite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
};

inline RTVector3 operator*(f32 s, const RTVector3& v) { return v * s; }

struct RTRay {
    RTVector3 origin;
    RTVector3 dir;
    RTVector3 inv_dir;
    f32 t_min{0.0f};
    f32 t_max{1e30f};

    RTRay(const RTVector3& orig, const RTVector3& d, f32 min_t = 0.0f, f32 max_t = 1e30f)
        : origin(orig), dir(d), t_min(min_t), t_max(max_t) {
        inv_dir = {std::abs(d.x) > 1e-8f ? 1.0f / d.x : (d.x >= 0.0f ? 1e8f : -1e8f),
                   std::abs(d.y) > 1e-8f ? 1.0f / d.y : (d.y >= 0.0f ? 1e8f : -1e8f),
                   std::abs(d.z) > 1e-8f ? 1.0f / d.z : (d.z >= 0.0f ? 1e8f : -1e8f)};
    }
};

struct RTAABB {
    RTVector3 min{1e30f, 1e30f, 1e30f};
    RTVector3 max{-1e30f, -1e30f, -1e30f};

    void expand(const RTVector3& p) {
        min = {std::min(min.x, p.x), std::min(min.y, p.y), std::min(min.z, p.z)};
        max = {std::max(max.x, p.x), std::max(max.y, p.y), std::max(max.z, p.z)};
    }
    void expand(const RTAABB& b) {
        expand(b.min);
        expand(b.max);
    }
    RTVector3 center() const { return (min + max) * 0.5f; }

    // The entry distance along the ray, or a negative value on a miss.
    f32 hit(const RTRay& ray, f32 t_max) const {
        f32 tx1 = (min.x - ray.origin.x) * ray.inv_dir.x, tx2 = (max.x - ray.origin.x) * ray.inv_dir.x;
        f32 tmin = std::min(tx1, tx2), tmax = std::max(tx1, tx2);
        f32 ty1 = (min.y - ray.origin.y) * ray.inv_dir.y, ty2 = (max.y - ray.origin.y) * ray.inv_dir.y;
        tmin = std::max(tmin, std::min(ty1, ty2));
        tmax = std::min(tmax, std::max(ty1, ty2));
        f32 tz1 = (min.z - ray.origin.z) * ray.inv_dir.z, tz2 = (max.z - ray.origin.z) * ray.inv_dir.z;
        tmin = std::max(tmin, std::min(tz1, tz2));
        tmax = std::min(tmax, std::max(tz1, tz2));
        tmin = std::max(tmin, ray.t_min);
        tmax = std::min(tmax, t_max);
        return tmax >= tmin ? tmin : -1.0f;
    }
};

struct RTTriangle {
    RTVector3 v0, e1, e2; // a corner and the two edges from it
    RTVector3 normal;

    bool intersect(const RTRay& ray, f32 t_max, f32& out_t) const {
        // Möller-Trumbore, both faces.
        const RTVector3 h = ray.dir.cross(e2);
        const f32 a = e1.dot(h);
        if (std::abs(a) < 1e-12f) return false;
        const f32 f = 1.0f / a;
        const RTVector3 s = ray.origin - v0;
        const f32 u = f * s.dot(h);
        if (u < 0.0f || u > 1.0f) return false;
        const RTVector3 q = s.cross(e1);
        const f32 v = f * ray.dir.dot(q);
        if (v < 0.0f || u + v > 1.0f) return false;
        const f32 t = f * e2.dot(q);
        if (t < ray.t_min || t > t_max) return false;
        out_t = t;
        return true;
    }
};

// The game's lights for a lit object: ambient and up to two directional
// lights (0..255 per channel, directions toward the light), as the vertices
// were lit with.
struct RTLightSet {
    f32 ambient[3] = {};
    f32 dir[2][3] = {};
    f32 col[2][3] = {};
    u32 count = 0;
    bool operator==(const RTLightSet& o) const { return std::memcmp(this, &o, sizeof *this) == 0; }
};

// What per-pixel lighting needs of a triangle: smooth normals and the shade
// the game gave its corners (RGBA8, r in the low byte), so the pixel's own
// lighting can replace the colour interpolated across it.
struct RTTriangleAttr {
    RTVector3 n[3];
    u32 shade[3] = {};
    u32 light_set = 0;
    bool lit = false;
};

struct BVHNode {
    RTAABB bounds;
    u32 first{0}; // leaf: first triangle; inner: index of the right child (the left one follows the node)
    u32 count{0}; // triangles of a leaf, 0 for inner nodes
};

class RayTracingScene {
public:
    void clear();
    void add_triangle(const RTVector3& v0, const RTVector3& v1, const RTVector3& v2,
                      const RTTriangleAttr& attr = {});
    // Index of `ls` in the scene's light sets (the last one again if equal).
    u32 add_light_set(const RTLightSet& ls);
    void build_bvh();

    bool empty() const { return tris_.empty(); }
    size_t triangle_count() const { return tris_.size(); }
    bool ready() const { return built_ && !tris_.empty(); }
    // The scene's typical triangle edge length: the yardstick for ray biases
    // and occlusion radii, whatever units the game works in.
    f32 scale() const { return scale_; }

    // The BVH as 32-bit words for the GPU: nodes (min xyz, max xyz, first,
    // count; an inner node's left child follows it, `first` is the right
    // one), then triangles (kTriWords each: v0, e1, e2, face normal, the
    // corners' normals, their shades, light set, lit), then the light sets
    // (kLightSetWords each: ambient, count, then per light direction and
    // colour). Returns the triangles' offset; *lights_offset the sets'.
    static constexpr u32 kTriWords = 28, kLightSetWords = 16;
    u32 serialize(std::vector<u32>& out, u32* lights_offset = nullptr) const;

    bool intersect_any(const RTRay& ray) const;
    bool intersect_closest(const RTRay& ray, f32& hit_t, RTVector3& hit_normal) const;

    // The fraction of `samples` rays in a cone of `cone` radians around
    // light_dir that leave `pos` unblocked (soft shadows). `seed` rotates the
    // pattern so neighbouring vertices don't band.
    f32 compute_shadow(const RTVector3& pos, const RTVector3& normal, const RTVector3& light_dir,
                       f32 max_dist = 1e30f, int samples = 4, f32 cone = 0.05f, u32 seed = 0) const;
    // 1 for an open surface down to ~0.15 in a tight crease: cosine-weighted
    // rays over the hemisphere, nearer hits occluding more.
    f32 compute_ao(const RTVector3& pos, const RTVector3& normal, f32 radius, int samples = 8, u32 seed = 0) const;

private:
    std::vector<RTTriangle> tris_;
    std::vector<RTTriangleAttr> attrs_;
    std::vector<RTLightSet> light_sets_;
    std::vector<RTAABB> tri_bounds_;
    std::vector<RTVector3> centers_;
    std::vector<u32> order_;
    std::vector<BVHNode> nodes_;
    f32 scale_{1.0f};
    bool built_{false};

    void build(u32 node, u32 first, u32 count, int depth);
    f32 bias(const RTVector3& pos) const { return std::max(scale_ * 0.02f, pos.length() * 1e-4f); }
};

// Two scenes: the frame being drawn adds to one; the other is the last
// finished frame's.
class FrameScenes {
public:
    RayTracingScene& building() { return scenes_[cur_ ^ 1]; }
    const RayTracingScene& active() const { return scenes_[cur_]; }
    // The frame is over: its triangles become the scene to trace (unless it
    // drew none, which keeps the last one, as for a frame drawn twice).
    void end_frame() {
        RayTracingScene& b = scenes_[cur_ ^ 1];
        if (b.empty()) return;
        if (!b.ready()) b.build_bvh();
        cur_ ^= 1;
        scenes_[cur_ ^ 1].clear();
    }
    void clear() {
        scenes_[0].clear();
        scenes_[1].clear();
    }

private:
    RayTracingScene scenes_[2];
    int cur_ = 0;
};

} // namespace orbit64::rt
