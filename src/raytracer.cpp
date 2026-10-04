#include "raytracer.hpp"

#include <algorithm>
#include <cmath>

namespace orbit64::rt {

namespace {

constexpr f32 kPi = 3.14159265358979f;
constexpr u32 kLeafSize = 4;
constexpr int kMaxDepth = 40; // keeps the traversal stacks (64) deep enough
constexpr int kBins = 12;

void basis(const RTVector3& n, RTVector3& t, RTVector3& b) {
    // Frisvad / Duff et al.: an orthonormal basis without branches on the axis.
    const f32 sign = n.z >= 0.0f ? 1.0f : -1.0f;
    const f32 a = -1.0f / (sign + n.z);
    const f32 c = n.x * n.y * a;
    t = {1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x};
    b = {c, sign + n.y * n.y * a, -n.y};
}

// A low-discrepancy point in [0,1)^2 (R2 sequence), offset by the seed.
void r2(int i, u32 seed, f32& u, f32& v) {
    const f32 s = static_cast<f32>(seed & 0xFFFF) * (1.0f / 65536.0f);
    u = std::fmod(0.5f + s + 0.7548776662f * static_cast<f32>(i + 1), 1.0f);
    v = std::fmod(0.5f + s * 0.61803f + 0.5698402910f * static_cast<f32>(i + 1), 1.0f);
}

f32 axis_of(const RTVector3& v, int axis) { return axis == 0 ? v.x : axis == 1 ? v.y : v.z; }

f32 area(const RTAABB& b) {
    const RTVector3 d = b.max - b.min;
    if (d.x < 0.0f) return 0.0f;
    return d.x * d.y + d.y * d.z + d.z * d.x;
}

} // namespace

void RayTracingScene::clear() {
    tris_.clear();
    tri_bounds_.clear();
    centers_.clear();
    order_.clear();
    nodes_.clear();
    scale_ = 1.0f;
    built_ = false;
}

void RayTracingScene::add_triangle(const RTVector3& v0, const RTVector3& v1, const RTVector3& v2) {
    if (!v0.finite() || !v1.finite() || !v2.finite()) return;
    RTTriangle t;
    t.v0 = v0;
    t.e1 = v1 - v0;
    t.e2 = v2 - v0;
    const RTVector3 n = t.e1.cross(t.e2);
    const f32 len = n.length();
    if (!(len > 1e-9f)) return; // degenerate
    t.normal = n * (1.0f / len);
    RTAABB b;
    b.expand(v0);
    b.expand(v1);
    b.expand(v2);
    tris_.push_back(t);
    tri_bounds_.push_back(b);
    centers_.push_back((v0 + v1 + v2) * (1.0f / 3.0f));
    built_ = false;
}

void RayTracingScene::build_bvh() {
    nodes_.clear();
    order_.clear();
    built_ = true;
    if (tris_.empty()) return;
    const u32 n = static_cast<u32>(tris_.size());

    // Median edge length (of a sample of the triangles).
    std::vector<f32> edges;
    const u32 step = std::max<u32>(1, n / 512);
    for (u32 i = 0; i < n; i += step) edges.push_back(tris_[i].e1.length());
    std::nth_element(edges.begin(), edges.begin() + edges.size() / 2, edges.end());
    scale_ = std::max(edges[edges.size() / 2], 1e-3f);

    order_.resize(n);
    for (u32 i = 0; i < n; ++i) order_[i] = i;
    nodes_.reserve(n * 2);
    nodes_.emplace_back();
    build(0, 0, n, 0);

    // Triangles in leaf order, for locality.
    std::vector<RTTriangle> sorted(n);
    for (u32 i = 0; i < n; ++i) sorted[i] = tris_[order_[i]];
    tris_.swap(sorted);
}

void RayTracingScene::build(u32 node, u32 first, u32 count, int depth) {
    RTAABB bounds, cb;
    for (u32 i = first; i < first + count; ++i) {
        bounds.expand(tri_bounds_[order_[i]]);
        cb.expand(centers_[order_[i]]);
    }
    nodes_[node].bounds = bounds;
    if (count <= kLeafSize || depth >= kMaxDepth) {
        nodes_[node].first = first;
        nodes_[node].count = count;
        return;
    }

    // Binned surface area heuristic over the longest axis of the centroids.
    const RTVector3 ext = cb.max - cb.min;
    const int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
    const f32 lo = axis_of(cb.min, axis), span = axis_of(ext, axis);
    u32 mid = first + count / 2;
    if (span > 1e-9f) {
        RTAABB bin_box[kBins];
        u32 bin_count[kBins] = {};
        const f32 k = kBins / span;
        auto bin_of = [&](u32 tri) {
            return std::min(kBins - 1, static_cast<int>((axis_of(centers_[tri], axis) - lo) * k));
        };
        for (u32 i = first; i < first + count; ++i) {
            const int b = bin_of(order_[i]);
            bin_box[b].expand(tri_bounds_[order_[i]]);
            ++bin_count[b];
        }
        f32 left_area[kBins - 1];
        u32 left_n[kBins - 1];
        RTAABB acc;
        u32 an = 0;
        for (int i = 0; i < kBins - 1; ++i) {
            acc.expand(bin_box[i]);
            an += bin_count[i];
            left_area[i] = area(acc);
            left_n[i] = an;
        }
        f32 best = 1e30f;
        int best_split = -1;
        acc = RTAABB{};
        an = 0;
        for (int i = kBins - 1; i > 0; --i) {
            acc.expand(bin_box[i]);
            an += bin_count[i];
            if (left_n[i - 1] == 0 || an == 0) continue;
            const f32 cost = left_area[i - 1] * left_n[i - 1] + area(acc) * an;
            if (cost < best) {
                best = cost;
                best_split = i;
            }
        }
        if (best_split > 0) {
            auto it = std::partition(order_.begin() + first, order_.begin() + first + count,
                                     [&](u32 t) { return bin_of(t) < best_split; });
            mid = static_cast<u32>(it - order_.begin());
        }
    }
    if (mid == first || mid == first + count) {
        mid = first + count / 2;
        std::nth_element(order_.begin() + first, order_.begin() + mid, order_.begin() + first + count,
                         [&](u32 a, u32 b) { return axis_of(centers_[a], axis) < axis_of(centers_[b], axis); });
    }

    const u32 left = static_cast<u32>(nodes_.size());
    nodes_.emplace_back();
    build(left, first, mid - first, depth + 1);
    const u32 right = static_cast<u32>(nodes_.size());
    nodes_.emplace_back();
    build(right, mid, first + count - mid, depth + 1);
    nodes_[node].first = right;
    nodes_[node].count = 0;
}

u32 RayTracingScene::serialize(std::vector<u32>& out) const {
    out.clear();
    if (!ready()) return 0;
    auto f = [&](f32 v) { out.push_back(std::bit_cast<u32>(v)); };
    out.reserve(nodes_.size() * 8 + tris_.size() * 12);
    for (const BVHNode& n : nodes_) {
        f(n.bounds.min.x); f(n.bounds.min.y); f(n.bounds.min.z);
        f(n.bounds.max.x); f(n.bounds.max.y); f(n.bounds.max.z);
        out.push_back(n.first);
        out.push_back(n.count);
    }
    const u32 tri_offset = static_cast<u32>(out.size());
    for (const RTTriangle& t : tris_) {
        for (const RTVector3* v : {&t.v0, &t.e1, &t.e2, &t.normal}) {
            f(v->x); f(v->y); f(v->z);
        }
    }
    return tri_offset;
}

bool RayTracingScene::intersect_any(const RTRay& ray) const {
    if (!ready()) return false;
    u32 stack[64];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const u32 ni = stack[--sp];
        const BVHNode& node = nodes_[ni];
        if (node.bounds.hit(ray, ray.t_max) < 0.0f) continue;
        if (node.count) {
            f32 t;
            for (u32 i = node.first; i < node.first + node.count; ++i)
                if (tris_[i].intersect(ray, ray.t_max, t)) return true;
        } else if (sp < 62) {
            stack[sp++] = node.first;
            stack[sp++] = ni + 1;
        }
    }
    return false;
}

bool RayTracingScene::intersect_closest(const RTRay& ray, f32& hit_t, RTVector3& hit_normal) const {
    if (!ready()) return false;
    f32 closest = ray.t_max;
    const RTTriangle* best = nullptr;
    u32 stack[64];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const u32 ni = stack[--sp];
        const BVHNode& node = nodes_[ni];
        if (node.bounds.hit(ray, closest) < 0.0f) continue;
        if (node.count) {
            f32 t;
            for (u32 i = node.first; i < node.first + node.count; ++i)
                if (tris_[i].intersect(ray, closest, t)) {
                    closest = t;
                    best = &tris_[i];
                }
        } else if (sp < 62) {
            // Nearer child on top.
            const u32 l = ni + 1, r = node.first;
            const f32 tl = nodes_[l].bounds.hit(ray, closest), tr = nodes_[r].bounds.hit(ray, closest);
            if (tl >= 0.0f && tr >= 0.0f) {
                stack[sp++] = tl <= tr ? r : l;
                stack[sp++] = tl <= tr ? l : r;
            } else if (tl >= 0.0f) {
                stack[sp++] = l;
            } else if (tr >= 0.0f) {
                stack[sp++] = r;
            }
        }
    }
    if (!best) return false;
    hit_t = closest;
    hit_normal = best->normal;
    return true;
}

f32 RayTracingScene::compute_shadow(const RTVector3& pos, const RTVector3& normal, const RTVector3& light_dir,
                                    f32 max_dist, int samples, f32 cone, u32 seed) const {
    if (!ready()) return 1.0f;
    const RTVector3 l = light_dir.normalized();
    // Off the surface, toward the light's side of it.
    const RTVector3 n = normal.dot(l) >= 0.0f ? normal : -normal;
    const f32 eps = bias(pos);
    const RTVector3 origin = pos + n * eps + l * eps;
    RTVector3 t, b;
    basis(l, t, b);
    samples = std::clamp(samples, 1, 32);
    int open = 0;
    for (int i = 0; i < samples; ++i) {
        RTVector3 d = l;
        if (samples > 1) {
            f32 u, v;
            r2(i, seed, u, v);
            const f32 r = cone * std::sqrt(u), a = 2.0f * kPi * v;
            d = (l + t * (r * std::cos(a)) + b * (r * std::sin(a))).normalized();
        }
        if (!intersect_any(RTRay(origin, d, 0.0f, max_dist))) ++open;
    }
    return static_cast<f32>(open) / static_cast<f32>(samples);
}

f32 RayTracingScene::compute_ao(const RTVector3& pos, const RTVector3& normal, f32 radius, int samples,
                                u32 seed) const {
    if (!ready() || radius <= 0.0f) return 1.0f;
    const RTVector3 n = normal.normalized();
    const RTVector3 origin = pos + n * bias(pos);
    RTVector3 t, b;
    basis(n, t, b);
    samples = std::clamp(samples, 1, 64);
    f32 occlusion = 0.0f;
    for (int i = 0; i < samples; ++i) {
        f32 u, v;
        r2(i, seed, u, v);
        // Cosine-weighted direction.
        const f32 r = std::sqrt(u), a = 2.0f * kPi * v;
        const f32 x = r * std::cos(a), y = r * std::sin(a), z = std::sqrt(std::max(0.0f, 1.0f - u));
        const RTVector3 d = (t * x + b * y + n * z).normalized();
        f32 hit_t;
        RTVector3 hit_n;
        if (intersect_closest(RTRay(origin, d, 0.0f, radius), hit_t, hit_n)) {
            const f32 k = 1.0f - hit_t / radius;
            occlusion += 0.35f + 0.65f * k * k;
        }
    }
    return std::clamp(1.0f - occlusion / static_cast<f32>(samples), 0.15f, 1.0f);
}

} // namespace orbit64::rt
