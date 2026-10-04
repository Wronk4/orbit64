// Shared by rt_trace.comp and rt_apply.comp; the C++ side is RtParams in
// src/gpu/rdp_gpu.cpp.

// The pass's parameters (std140: vec4s only).
//   ip0..ip3  rows of the inverse projection: clip -> modelview space
//   vp        viewport scale x, y, translation x, y
//   vpz       viewport scale z, translation z, 1 / S, the scene's scale
//   sun       toward the main light, shadow cone (radians)
//   eye       the camera, ambient occlusion radius
//   dims      width, height (pixels at scale S), word offsets of the BVH's nodes and triangles
//   misc      frame number, hemisphere rays, shadow rays, the grid (pixels)
//             shadow and hemisphere rays are traced on: one pixel per
//             grid x grid block, spread to the rest by rt_apply.comp
//   strength  how dark full shadow and full occlusion make a pixel, specular
//             highlight strength, per-pixel lighting on
//   ofs       word offset of the light sets (RTLightSet, 16 words each),
//             the dispatch: 0 every pixel (eye rays, lighting, reflections),
//             1 the grid's pixels (shadow and hemisphere rays)
//   pr0..pr3  rows of the projection: modelview space -> clip
//   refl      reflection strength of water and of metal, ripples, time
//   gi        global illumination strength, bounce ray length
#define RT_PARAMS                                                                                    \
    layout(std140, UNIFORM(0)) uniform Params {                                                      \
        vec4 ip0, ip1, ip2, ip3;                                                                     \
        vec4 vp, vpz, sun, eye;                                                                      \
        uvec4 dims, misc;                                                                            \
        vec4 strength;                                                                               \
        uvec4 ofs;                                                                                   \
        vec4 pr0, pr1, pr2, pr3;                                                                     \
        vec4 refl;                                                                                   \
        vec4 gi;                                                                                     \
    } P;

// Per pixel in the occlusion buffer (8 words): shadow | ao << 16, distance,
// then for per-pixel lighting halves of the light without the main light,
// the main light's, and the colour the game lit the corners with (all
// 0..1): (base r, g), (base b, main r), (main g, b), (shade r, g),
// (shade b, 1 when there is lighting), then the reflection to blend in
// (RGBA8: colour, alpha the amount), the light bounced off what is around
// (RGB8), 1 when shadow and hemisphere rays were traced here.
#define OCC_WORDS 10u

// RayTracingScene::kTriWords and the flags in a triangle's word 25.
#define TRI_WORDS 36u
#define TRI_LIT 1u
#define TRI_METAL 2u
#define TRI_WATER 4u
#define TRI_CUTOUT 8u // word 26: draw state (texture), 27-32: s, t of the corners

// The RDP's compressed depth (raster.comp).
int z_decompress(int z) {
    int exponent = z >> 11, mantissa = z & 0x7FF;
    int shift = max(6 - exponent, 0);
    int base = 0x40000 - (0x40000 >> exponent);
    return (mantissa << shift) + base;
}
