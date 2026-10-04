#pragma once

#include "common.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>

namespace orbit64::rt {

struct RTVector3 {
    f32 x{0.0f}, y{0.0f}, z{0.0f};

    constexpr RTVector3() = default;
    constexpr RTVector3(f32 x_, f32 y_, f32 z_) : x(x_), y(y_), z(z_) {}

    constexpr RTVector3 operator+(const RTVector3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr RTVector3 operator-(const RTVector3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr RTVector3 operator*(f32 s) const { return {x * s, y * s, z * s}; }
    constexpr RTVector3 operator*(const RTVector3& o) const { return {x * o.x, y * o.y, z * o.z}; }
    constexpr RTVector3 operator/(f32 s) const { f32 inv = 1.0f / s; return {x * inv, y * inv, z * inv}; }

    RTVector3& operator+=(const RTVector3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    RTVector3& operator*=(f32 s) { x *= s; y *= s; z *= s; return *this; }

    f32 dot(const RTVector3& o) const { return x * o.x + y * o.y + z * o.z; }
    RTVector3 cross(const RTVector3& o) const {
        return {
            y * o.z - z * o.y,
            z * o.x - x * o.z,
            x * o.y - y * o.x
        };
    }

    f32 length_squared() const { return x * x + y * y + z * z; }
    f32 length() const { return std::sqrt(length_squared()); }

    RTVector3 normalized() const {
        f32 len = length();
        if (len > 1e-6f) return *this * (1.0f / len);
        return {0.0f, 0.0f, 1.0f};
    }
};

inline RTVector3 operator*(f32 s, const RTVector3& v) { return v * s; }

struct RTRay {
    RTVector3 origin;
    RTVector3 dir;
    RTVector3 inv_dir;
    f32 t_min{0.01f};
    f32 t_max{1e7f};

    RTRay(const RTVector3& orig, const RTVector3& d, f32 min_t = 0.05f, f32 max_t = 1e7f)
        : origin(orig), dir(d), t_min(min_t), t_max(max_t) {
        inv_dir = {
            std::abs(d.x) > 1e-8f ? 1.0f / d.x : (d.x >= 0.0f ? 1e8f : -1e8f),
            std::abs(d.y) > 1e-8f ? 1.0f / d.y : (d.y >= 0.0f ? 1e8f : -1e8f),
            std::abs(d.z) > 1e-8f ? 1.0f / d.z : (d.z >= 0.0f ? 1e8f : -1e8f)
        };
    }
};

struct RTAABB {
    RTVector3 min{ 1e9f,  1e9f,  1e9f};
    RTVector3 max{-1e9f, -1e9f, -1e9f};

    void expand(const RTVector3& p) {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        min.z = std::min(min.z, p.z);
        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
        max.z = std::max(max.z, p.z);
    }

    void expand(const RTAABB& b) {
        min.x = std::min(min.x, b.min.x);
        min.y = std::min(min.y, b.min.y);
        min.z = std::min(min.z, b.min.z);
        max.x = std::max(max.x, b.max.x);
        max.y = std::max(max.y, b.max.y);
        max.z = std::max(max.z, b.max.z);
    }

    RTVector3 center() const {
        return (min + max) * 0.5f;
    }

    bool intersect(const RTRay& ray, f32& t_near, f32& t_far) const {
        f32 tx1 = (min.x - ray.origin.x) * ray.inv_dir.x;
        f32 tx2 = (max.x - ray.origin.x) * ray.inv_dir.x;
        f32 tmin = std::min(tx1, tx2);
        f32 tmax = std::max(tx1, tx2);

        f32 ty1 = (min.y - ray.origin.y) * ray.inv_dir.y;
        f32 ty2 = (max.y - ray.origin.y) * ray.inv_dir.y;
        tmin = std::max(tmin, std::min(ty1, ty2));
        tmax = std::min(tmax, std::max(ty1, ty2));

        f32 tz1 = (min.z - ray.origin.z) * ray.inv_dir.z;
        f32 tz2 = (max.z - ray.origin.z) * ray.inv_dir.z;
        tmin = std::max(tmin, std::min(tz1, tz2));
        tmax = std::min(tmax, std::max(tz1, tz2));

        t_near = std::max(tmin, ray.t_min);
        t_far = std::min(tmax, ray.t_max);
        return tmax >= tmin && t_far >= t_near;
    }
};

struct RTTriangle {
    RTVector3 v0, v1, v2;
    RTVector3 normal;
    RTVector3 center;
    RTAABB bounds;
    u32 color{0xFFFFFFFF};

    void compute_derived() {
        RTVector3 e1 = v1 - v0;
        RTVector3 e2 = v2 - v0;
        normal = e1.cross(e2).normalized();
        center = (v0 + v1 + v2) * (1.0f / 3.0f);
        bounds = RTAABB{};
        bounds.expand(v0);
        bounds.expand(v1);
        bounds.expand(v2);
    }

    bool intersect(const RTRay& ray, f32& out_t) const {
        // Möller-Trumbore intersection algorithm
        const RTVector3 edge1 = v1 - v0;
        const RTVector3 edge2 = v2 - v0;
        const RTVector3 h = ray.dir.cross(edge2);
        const f32 a = edge1.dot(h);

        if (std::abs(a) < 1e-7f) return false;
        const f32 f = 1.0f / a;
        const RTVector3 s = ray.origin - v0;
        const f32 u = f * s.dot(h);
        if (u < 0.0f || u > 1.0f) return false;

        const RTVector3 q = s.cross(edge1);
        const f32 v = f * ray.dir.dot(q);
        if (v < 0.0f || u + v > 1.0f) return false;

        const f32 t = f * edge2.dot(q);
        if (t >= ray.t_min && t <= ray.t_max) {
            out_t = t;
            return true;
        }
        return false;
    }
};

struct RTLight {
    RTVector3 dir{0.0f, 1.0f, 0.0f};   // Direction toward light (directional)
    RTVector3 pos{0.0f, 0.0f, 0.0f};   // Position in scene (point light)
    RTVector3 color{1.0f, 1.0f, 1.0f}; // RGB 0..1
    f32 intensity{1.0f};
    f32 radius{200.0f};
    bool is_point{false};
};

struct BVHNode {
    RTAABB bounds;
    s32 left_child{-1};
    s32 right_child{-1};
    u32 first_prim{0};
    u32 prim_count{0};

    bool is_leaf() const { return prim_count > 0; }
};

class RayTracingScene {
public:
    RayTracingScene() = default;

    void clear();
    void add_triangle(const RTVector3& v0, const RTVector3& v1, const RTVector3& v2, u32 col = 0xFFFFFFFF);
    void build_bvh();

    bool empty() const { return triangles_.empty(); }
    size_t triangle_count() const { return triangles_.size(); }

    bool intersect_any(const RTRay& ray) const;
    bool intersect_closest(const RTRay& ray, f32& hit_t, RTVector3& hit_normal) const;

    // Ray tracing features
    f32 compute_shadow(const RTVector3& pos, const RTVector3& normal, const RTVector3& light_dir, f32 max_dist = 1000.0f) const;
    f32 compute_ao(const RTVector3& pos, const RTVector3& normal, f32 radius = 35.0f, int samples = 5) const;

    RTVector3 evaluate_lighting(
        const RTVector3& pos,
        const RTVector3& normal,
        const RTVector3& view_dir,
        const RTVector3& ambient_col,
        const std::vector<RTLight>& lights,
        const RTVector3& vertex_base_col,
        bool enable_shadows = true,
        bool enable_ao = true,
        bool enable_specular = true
    ) const;

private:
    std::vector<RTTriangle> triangles_;
    std::vector<u32> triangle_indices_;
    std::vector<BVHNode> nodes_;
    bool bvh_built_{false};

    s32 build_recursive(u32 first, u32 count, int depth);
};

} // namespace orbit64::rt
