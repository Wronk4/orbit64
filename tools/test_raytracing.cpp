// Unit tests of the HLE ray tracer (src/raytracer.cpp): cmake --build <dir> --target test_raytracing
#undef NDEBUG // the checks are asserts, in every build type
#include "../src/raytracer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace orbit64::rt;

void test_basic_intersection() {
    std::cout << "[Test] Running test_basic_intersection...\n";
    RayTracingScene scene;

    // A simple quad/triangle on the XY plane at z = 10
    // Triangle: (0,0,10), (10,0,10), (0,10,10)
    scene.add_triangle({0, 0, 10}, {10, 0, 10}, {0, 10, 10});
    scene.build_bvh();

    assert(!scene.empty());
    assert(scene.triangle_count() == 1);

    // Ray shooting from (2, 2, 0) in +Z direction: should hit
    RTRay ray_hit({2, 2, 0}, {0, 0, 1});
    assert(scene.intersect_any(ray_hit));

    f32 hit_t = 0.0f;
    RTVector3 hit_norm;
    assert(scene.intersect_closest(ray_hit, hit_t, hit_norm));
    assert(std::abs(hit_t - 10.0f) < 1e-3f);
    assert(std::abs(hit_norm.z) > 0.9f);

    // Ray shooting in opposite direction (-Z): should miss
    RTRay ray_miss({2, 2, 0}, {0, 0, -1});
    assert(!scene.intersect_any(ray_miss));

    // Ray shooting outside triangle: should miss
    RTRay ray_outside({20, 20, 0}, {0, 0, 1});
    assert(!scene.intersect_any(ray_outside));

    std::cout << "  -> test_basic_intersection PASSED\n";
}

void test_shadows() {
    std::cout << "[Test] Running test_shadows...\n";
    RayTracingScene scene;

    // Ground plane at y = 0
    scene.add_triangle({-100, 0, -100}, {100, 0, -100}, {100, 0, 100});
    scene.add_triangle({-100, 0, -100}, {100, 0, 100}, {-100, 0, 100});

    // Occluder floating above at y = 20, covering x in [-10, 10], z in [-10, 10]
    scene.add_triangle({-10, 20, -10}, {10, 20, -10}, {10, 20, 10});
    scene.add_triangle({-10, 20, -10}, {10, 20, 10}, {-10, 20, 10});
    scene.build_bvh();

    // Light shining directly from above (0, 1, 0)
    RTVector3 light_dir(0, 1, 0);

    // Point directly under occluder at (0, 0.1, 0)
    f32 shadow_under = scene.compute_shadow({0, 0.1f, 0}, {0, 1, 0}, light_dir);
    // Point far away at (50, 0.1, 50)
    f32 shadow_outside = scene.compute_shadow({50, 0.1f, 50}, {0, 1, 0}, light_dir);

    std::cout << "  Shadow under occluder: " << shadow_under << " (expected 0.0)\n";
    std::cout << "  Shadow outside occluder: " << shadow_outside << " (expected 1.0)\n";

    assert(shadow_under < 0.1f);
    assert(shadow_outside > 0.9f);

    std::cout << "  -> test_shadows PASSED\n";
}

void test_ambient_occlusion() {
    std::cout << "[Test] Running test_ambient_occlusion...\n";
    RayTracingScene scene;

    // Corner: floor at y = 0 and wall at x = 0
    // Floor: x in [0, 50], z in [-50, 50]
    scene.add_triangle({0, 0, -50}, {50, 0, -50}, {50, 0, 50});
    scene.add_triangle({0, 0, -50}, {50, 0, 50}, {0, 0, 50});

    // Wall: y in [0, 50], z in [-50, 50]
    scene.add_triangle({0, 0, -50}, {0, 50, -50}, {0, 50, 50});
    scene.add_triangle({0, 0, -50}, {0, 50, 50}, {0, 0, 50});
    scene.build_bvh();

    // Point deep in the corner at (0.5, 0.5, 0)
    f32 ao_corner = scene.compute_ao({0.5f, 0.5f, 0}, {0, 1, 0}, 30.0f, 8);
    // Point far out in open space at (40, 0.5, 0)
    f32 ao_open = scene.compute_ao({40.0f, 0.5f, 0}, {0, 1, 0}, 30.0f, 8);

    std::cout << "  AO in corner: " << ao_corner << "\n";
    std::cout << "  AO in open space: " << ao_open << "\n";

    assert(ao_corner < ao_open);
    assert(ao_open > 0.85f);

    std::cout << "  -> test_ambient_occlusion PASSED\n";
}

void test_scale_independence() {
    std::cout << "[Test] Running test_scale_independence...\n";
    // The shadow test's scene in units 1000 times larger (games differ wildly).
    for (f32 k : {0.01f, 1.0f, 1000.0f}) {
        RayTracingScene scene;
        scene.add_triangle(RTVector3{-100, 0, -100} * k, RTVector3{100, 0, -100} * k, RTVector3{100, 0, 100} * k);
        scene.add_triangle(RTVector3{-100, 0, -100} * k, RTVector3{100, 0, 100} * k, RTVector3{-100, 0, 100} * k);
        scene.add_triangle(RTVector3{-10, 20, -10} * k, RTVector3{10, 20, -10} * k, RTVector3{10, 20, 10} * k);
        scene.add_triangle(RTVector3{-10, 20, -10} * k, RTVector3{10, 20, 10} * k, RTVector3{-10, 20, 10} * k);
        scene.build_bvh();
        const f32 under = scene.compute_shadow(RTVector3{0, 0, 0} * k, {0, 1, 0}, {0, 1, 0});
        const f32 open = scene.compute_shadow(RTVector3{50, 0, 50} * k, {0, 1, 0}, {0, 1, 0});
        // A point on the lit ground must not shadow itself.
        const f32 self = scene.compute_shadow(RTVector3{60, 0, -60} * k, {0, 1, 0}, RTVector3{1, 1, 0}.normalized());
        std::cout << "  scale " << k << ": under " << under << ", open " << open << ", self " << self << "\n";
        assert(under < 0.1f);
        assert(open > 0.9f);
        assert(self > 0.9f);
    }
    std::cout << "  -> test_scale_independence PASSED\n";
}

void test_frame_scenes() {
    std::cout << "[Test] Running test_frame_scenes...\n";
    FrameScenes fs;
    assert(!fs.active().ready());
    fs.building().add_triangle({0, 0, 10}, {10, 0, 10}, {0, 10, 10});
    assert(!fs.active().ready()); // nothing to trace until the frame ends
    fs.end_frame();
    assert(fs.active().ready() && fs.active().triangle_count() == 1);
    assert(fs.building().empty());
    fs.end_frame(); // a frame that drew nothing keeps the last scene
    assert(fs.active().triangle_count() == 1);
    std::cout << "  -> test_frame_scenes PASSED\n";
}

void test_many_triangles() {
    std::cout << "[Test] Running test_many_triangles...\n";
    // A 100 x 100 grid of quads: the BVH must agree with brute force.
    RayTracingScene scene;
    for (int z = 0; z < 100; ++z)
        for (int x = 0; x < 100; ++x) {
            const f32 h = std::sin(x * 0.3f) * std::cos(z * 0.2f) * 3.0f;
            scene.add_triangle({f32(x), h, f32(z)}, {f32(x + 1), h, f32(z)}, {f32(x + 1), h, f32(z + 1)});
            scene.add_triangle({f32(x), h, f32(z)}, {f32(x + 1), h, f32(z + 1)}, {f32(x), h, f32(z + 1)});
        }
    scene.build_bvh();
    int hits = 0;
    for (int i = 0; i < 1000; ++i) {
        const f32 x = 0.5f + (i % 37) * 2.6f, z = 0.5f + (i % 41) * 2.4f;
        f32 t;
        RTVector3 n;
        if (scene.intersect_closest(RTRay({x, 50, z}, {0, -1, 0}), t, n)) {
            ++hits;
            assert(t > 46.0f && t < 54.0f);
        }
    }
    std::cout << "  " << hits << " / 1000 rays hit\n";
    assert(hits == 1000);
    std::cout << "  -> test_many_triangles PASSED\n";
}

int main() {
    std::cout << "=== Running HLE Ray Tracing Test Suite ===\n";
    test_basic_intersection();
    test_shadows();
    test_ambient_occlusion();
    test_scale_independence();
    test_frame_scenes();
    test_many_triangles();
    std::cout << "=== All Ray Tracing Tests Passed Successfully! ===\n";
    return 0;
}
