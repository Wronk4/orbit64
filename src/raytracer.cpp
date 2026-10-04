#include "raytracer.hpp"
#include <algorithm>
#include <cmath>

namespace orbit64::rt {

namespace {

// Deterministic hemisphere directions for Ambient Occlusion
constexpr std::array<RTVector3, 8> kHemisphereSamples = {{
    {0.000f, 0.000f, 1.000f},
    {0.577f, 0.577f, 0.577f},
    {-0.577f, 0.577f, 0.577f},
    {0.577f, -0.577f, 0.577f},
    {-0.577f, -0.577f, 0.577f},
    {0.707f, 0.000f, 0.707f},
    {-0.707f, 0.000f, 0.707f},
    {0.000f, 0.707f, 0.707f}
}};

// Soft shadow jitter offsets
constexpr std::array<RTVector3, 4> kShadowJitters = {{
    {0.000f, 0.000f, 0.000f},
    {0.035f, 0.020f, -0.015f},
    {-0.025f, 0.035f, 0.020f},
    {0.015f, -0.030f, 0.035f}
}};

// Construct an orthonormal basis from a normal
void create_orthonormal_basis(const RTVector3& n, RTVector3& tangent, RTVector3& bitangent) {
    if (std::abs(n.x) > std::abs(n.z)) {
        tangent = RTVector3(-n.y, n.x, 0.0f).normalized();
    } else {
        tangent = RTVector3(0.0f, -n.z, n.y).normalized();
    }
    bitangent = n.cross(tangent).normalized();
}

} // namespace

void RayTracingScene::clear() {
    triangles_.clear();
    triangle_indices_.clear();
    nodes_.clear();
    bvh_built_ = false;
}

void RayTracingScene::add_triangle(const RTVector3& v0, const RTVector3& v1, const RTVector3& v2, u32 col) {
    // Filter non-finite or degenerate triangles
    if (!std::isfinite(v0.x) || !std::isfinite(v0.y) || !std::isfinite(v0.z) ||
        !std::isfinite(v1.x) || !std::isfinite(v1.y) || !std::isfinite(v1.z) ||
        !std::isfinite(v2.x) || !std::isfinite(v2.y) || !std::isfinite(v2.z)) {
        return;
    }

    RTTriangle tri;
    tri.v0 = v0;
    tri.v1 = v1;
    tri.v2 = v2;
    tri.color = col;
    tri.compute_derived();

    if (tri.normal.length_squared() < 1e-6f) return; // Degenerate

    triangles_.push_back(tri);
    bvh_built_ = false;
}

void RayTracingScene::build_bvh() {
    if (triangles_.empty()) {
        nodes_.clear();
        triangle_indices_.clear();
        bvh_built_ = true;
        return;
    }

    const u32 count = static_cast<u32>(triangles_.size());
    triangle_indices_.resize(count);
    for (u32 i = 0; i < count; ++i) {
        triangle_indices_[i] = i;
    }

    nodes_.clear();
    nodes_.reserve(count * 2);

    build_recursive(0, count, 0);
    bvh_built_ = true;
}

s32 RayTracingScene::build_recursive(u32 first, u32 count, int depth) {
    s32 node_idx = static_cast<s32>(nodes_.size());
    nodes_.emplace_back();

    RTAABB bbox;
    RTAABB centroid_box;
    for (u32 i = 0; i < count; ++i) {
        const auto& tri = triangles_[triangle_indices_[first + i]];
        bbox.expand(tri.bounds);
        centroid_box.expand(tri.center);
    }

    nodes_[node_idx].bounds = bbox;

    // Leaf condition: small primitive count or deep tree
    constexpr u32 kMaxLeafPrims = 4;
    constexpr int kMaxDepth = 24;
    if (count <= kMaxLeafPrims || depth >= kMaxDepth) {
        nodes_[node_idx].first_prim = first;
        nodes_[node_idx].prim_count = count;
        nodes_[node_idx].left_child = -1;
        nodes_[node_idx].right_child = -1;
        return node_idx;
    }

    // Split along longest axis of centroids
    RTVector3 extent = centroid_box.max - centroid_box.min;
    int axis = 0;
    if (extent.y > extent.x && extent.y > extent.z) axis = 1;
    else if (extent.z > extent.x && extent.z > extent.y) axis = 2;

    f32 split_coord = centroid_box.center().x;
    if (axis == 1) split_coord = centroid_box.center().y;
    else if (axis == 2) split_coord = centroid_box.center().z;

    // Partition
    u32 mid = first;
    for (u32 i = first; i < first + count; ++i) {
        const auto& tri = triangles_[triangle_indices_[i]];
        f32 c = tri.center.x;
        if (axis == 1) c = tri.center.y;
        else if (axis == 2) c = tri.center.z;

        if (c < split_coord) {
            std::swap(triangle_indices_[i], triangle_indices_[mid]);
            mid++;
        }
    }

    // If partition failed to split, fall back to median split
    if (mid == first || mid == first + count) {
        mid = first + count / 2;
        std::nth_element(
            triangle_indices_.begin() + first,
            triangle_indices_.begin() + mid,
            triangle_indices_.begin() + first + count,
            [&](u32 a, u32 b) {
                const auto& ta = triangles_[a];
                const auto& tb = triangles_[b];
                if (axis == 0) return ta.center.x < tb.center.x;
                if (axis == 1) return ta.center.y < tb.center.y;
                return ta.center.z < tb.center.z;
            }
        );
    }

    u32 left_count = mid - first;
    u32 right_count = count - left_count;

    s32 left_child = build_recursive(first, left_count, depth + 1);
    s32 right_child = build_recursive(mid, right_count, depth + 1);

    nodes_[node_idx].left_child = left_child;
    nodes_[node_idx].right_child = right_child;
    nodes_[node_idx].prim_count = 0; // Internal node

    return node_idx;
}

bool RayTracingScene::intersect_any(const RTRay& ray) const {
    if (!bvh_built_ || nodes_.empty()) return false;

    s32 stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        s32 curr_idx = stack[--stack_ptr];
        const auto& node = nodes_[curr_idx];

        f32 t_near, t_far;
        if (!node.bounds.intersect(ray, t_near, t_far)) {
            continue;
        }

        if (node.is_leaf()) {
            for (u32 i = 0; i < node.prim_count; ++i) {
                const auto& tri = triangles_[triangle_indices_[node.first_prim + i]];
                f32 t;
                if (tri.intersect(ray, t)) {
                    return true;
                }
            }
        } else {
            if (node.left_child >= 0) stack[stack_ptr++] = node.left_child;
            if (node.right_child >= 0) stack[stack_ptr++] = node.right_child;
        }
    }

    return false;
}

bool RayTracingScene::intersect_closest(const RTRay& ray, f32& hit_t, RTVector3& hit_normal) const {
    if (!bvh_built_ || nodes_.empty()) return false;

    f32 closest_t = ray.t_max;
    bool hit = false;
    RTVector3 normal{};

    s32 stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        s32 curr_idx = stack[--stack_ptr];
        const auto& node = nodes_[curr_idx];

        f32 t_near, t_far;
        if (!node.bounds.intersect(ray, t_near, t_far) || t_near >= closest_t) {
            continue;
        }

        if (node.is_leaf()) {
            for (u32 i = 0; i < node.prim_count; ++i) {
                const auto& tri = triangles_[triangle_indices_[node.first_prim + i]];
                f32 t;
                if (tri.intersect(ray, t) && t < closest_t) {
                    closest_t = t;
                    hit = true;
                    normal = tri.normal;
                }
            }
        } else {
            if (node.left_child >= 0) stack[stack_ptr++] = node.left_child;
            if (node.right_child >= 0) stack[stack_ptr++] = node.right_child;
        }
    }

    if (hit) {
        hit_t = closest_t;
        hit_normal = normal;
    }
    return hit;
}

f32 RayTracingScene::compute_shadow(const RTVector3& pos, const RTVector3& normal, const RTVector3& light_dir, f32 max_dist) const {
    if (triangles_.empty() || !bvh_built_) return 1.0f;

    // Normal offset bias to avoid self-intersection
    const RTVector3 origin = pos + normal * 0.15f;

    RTVector3 tangent, bitangent;
    create_orthonormal_basis(light_dir, tangent, bitangent);

    int unoccluded_rays = 0;
    for (const auto& jitter : kShadowJitters) {
        RTVector3 dir = (light_dir + tangent * jitter.x + bitangent * jitter.y + light_dir * jitter.z).normalized();
        RTRay shadow_ray(origin, dir, 0.05f, max_dist);
        if (!intersect_any(shadow_ray)) {
            unoccluded_rays++;
        }
    }

    return static_cast<f32>(unoccluded_rays) / static_cast<f32>(kShadowJitters.size());
}

f32 RayTracingScene::compute_ao(const RTVector3& pos, const RTVector3& normal, f32 radius, int samples) const {
    if (triangles_.empty() || !bvh_built_) return 1.0f;

    const RTVector3 origin = pos + normal * 0.15f;
    RTVector3 tangent, bitangent;
    create_orthonormal_basis(normal, tangent, bitangent);

    f32 occlusion = 0.0f;
    int test_samples = std::clamp(samples, 1, static_cast<int>(kHemisphereSamples.size()));

    for (int i = 0; i < test_samples; ++i) {
        const auto& s = kHemisphereSamples[i];
        RTVector3 dir = (tangent * s.x + bitangent * s.y + normal * s.z).normalized();
        RTRay ao_ray(origin, dir, 0.05f, radius);

        f32 hit_t;
        RTVector3 hit_norm;
        if (intersect_closest(ao_ray, hit_t, hit_norm)) {
            // Closer intersections occlude more strongly
            f32 dist_factor = std::clamp(1.0f - (hit_t / radius), 0.0f, 1.0f);
            f32 angle_factor = std::max(0.0f, normal.dot(dir));
            occlusion += dist_factor * angle_factor;
        }
    }

    f32 ao = 1.0f - (occlusion / static_cast<f32>(test_samples));
    return std::clamp(ao, 0.18f, 1.0f);
}

RTVector3 RayTracingScene::evaluate_lighting(
    const RTVector3& pos,
    const RTVector3& normal,
    const RTVector3& view_dir,
    const RTVector3& ambient_col,
    const std::vector<RTLight>& lights,
    const RTVector3& vertex_base_col,
    bool enable_shadows,
    bool enable_ao,
    bool enable_specular
) const {
    const RTVector3 n = normal.normalized();
    const RTVector3 v = view_dir.normalized();

    // 1. Ray-Traced Ambient Occlusion
    f32 ao = 1.0f;
    if (enable_ao && !triangles_.empty() && bvh_built_) {
        ao = compute_ao(pos, n, 40.0f, 5);
    }

    // Subtle sky-ground hemispherical ambient gradient
    f32 hemi_factor = n.y * 0.25f + 0.75f; // [0.5..1.0]
    RTVector3 lit_ambient = ambient_col * (ao * hemi_factor);

    RTVector3 diffuse_acc{0.0f, 0.0f, 0.0f};
    RTVector3 specular_acc{0.0f, 0.0f, 0.0f};

    // 2. Direct Lighting (Directional & Point lights) with Ray-Traced Shadows
    for (const auto& light : lights) {
        RTVector3 l_dir;
        f32 attenuation = 1.0f;
        f32 max_dist = 1e6f;

        if (light.is_point) {
            RTVector3 to_light = light.pos - pos;
            f32 dist = to_light.length();
            if (dist > light.radius || dist < 1e-4f) continue;

            l_dir = to_light / dist;
            max_dist = dist;
            // Physically-plausible inverse-square falloff
            attenuation = 1.0f / (1.0f + 0.005f * dist * dist);
        } else {
            l_dir = light.dir.normalized();
        }

        f32 n_dot_l = n.dot(l_dir);
        if (n_dot_l <= 0.0f) continue;

        // Trace shadow ray
        f32 shadow = 1.0f;
        if (enable_shadows && !triangles_.empty() && bvh_built_) {
            shadow = compute_shadow(pos, n, l_dir, max_dist);
        }

        if (shadow <= 0.001f) continue;

        // Diffuse contribution
        f32 diff_factor = n_dot_l * shadow * attenuation;
        diffuse_acc += light.color * (diff_factor * light.intensity);

        // 3. Specular Reflection (Blinn-Phong + Schlick's Fresnel approximation)
        if (enable_specular) {
            RTVector3 h = (l_dir + v).normalized();
            f32 n_dot_h = std::max(0.0f, n.dot(h));
            if (n_dot_h > 0.0f) {
                // High specular power for glossy look
                f32 spec = std::pow(n_dot_h, 32.0f);
                // Fresnel term (F0 = 0.04 for dielectric surfaces)
                f32 v_dot_h = std::max(0.0f, v.dot(h));
                f32 fresnel = 0.04f + (1.0f - 0.04f) * std::pow(1.0f - v_dot_h, 5.0f);
                specular_acc += light.color * (spec * fresnel * shadow * attenuation * light.intensity);
            }
        }
    }

    // Combine diffuse modulated by base vertex color, plus specular
    RTVector3 final_col = (lit_ambient + diffuse_acc) * vertex_base_col + specular_acc;

    return {
        std::clamp(final_col.x, 0.0f, 1.0f),
        std::clamp(final_col.y, 0.0f, 1.0f),
        std::clamp(final_col.z, 0.0f, 1.0f)
    };
}

} // namespace orbit64::rt
