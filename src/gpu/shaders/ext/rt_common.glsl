// Shared by rt_trace.comp and rt_apply.comp; the C++ side is RtParams in
// src/gpu/rdp_gpu.cpp.

// The pass's parameters (std140: vec4s only).
//   ip0..ip3  rows of the inverse projection: clip -> modelview space
//   vp        viewport scale x, y, translation x, y
//   vpz       viewport scale z, translation z, 1 / S, the scene's scale
//   sun       toward the main light, shadow cone (radians)
//   eye       the camera, ambient occlusion radius
//   dims      width, height (pixels at scale S), word offsets of the BVH's nodes and triangles
//   misc      frame number, AO rays, shadow rays, filter step (pixels)
//   strength  how dark full shadow and full occlusion make a pixel
#define RT_PARAMS                                                                                    \
    layout(std140, UNIFORM(0)) uniform Params {                                                      \
        vec4 ip0, ip1, ip2, ip3;                                                                     \
        vec4 vp, vpz, sun, eye;                                                                      \
        uvec4 dims, misc;                                                                            \
        vec4 strength;                                                                               \
    } P;

// The RDP's compressed depth (raster.comp).
int z_decompress(int z) {
    int exponent = z >> 11, mantissa = z & 0x7FF;
    int shift = max(6 - exponent, 0);
    int base = 0x40000 - (0x40000 >> exponent);
    return (mantissa << shift) + base;
}
