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

void test_lighting_evaluation() {
    std::cout << "[Test] Running test_lighting_evaluation...\n";
    RayTracingScene scene;

    // Add sphere-like / box mesh
    scene.add_triangle({-5, 0, -5}, {5, 0, -5}, {0, 5, 0});
    scene.build_bvh();

    RTVector3 pos(0, 0, -10);
    RTVector3 norm(0, 0, -1);
    RTVector3 view(0, 0, -1);
    RTVector3 amb(0.2f, 0.2f, 0.2f);

    std::vector<RTLight> lights;
    RTLight sun;
    sun.dir = {0, 0, -1};
    sun.color = {1.0f, 1.0f, 1.0f};
    sun.intensity = 1.0f;
    sun.is_point = false;
    lights.push_back(sun);

    RTVector3 base_col(0.8f, 0.1f, 0.1f); // red
    auto lit = scene.evaluate_lighting(pos, norm, view, amb, lights, base_col);

    std::cout << "  Evaluated lighting RGB: (" << lit.x << ", " << lit.y << ", " << lit.z << ")\n";
    assert(lit.x > 0.5f); // Red component illuminated
    assert(lit.x > lit.y); // Dominantly red
    assert(lit.z >= 0.0f && lit.z <= 1.0f);

    std::cout << "  -> test_lighting_evaluation PASSED\n";
}

int main() {
    std::cout << "=== Running HLE Ray Tracing Test Suite ===\n";
    test_basic_intersection();
    test_shadows();
    test_ambient_occlusion();
    test_lighting_evaluation();
    std::cout << "=== All Ray Tracing Tests Passed Successfully! ===\n";
    return 0;
}
