// Resource binding for SDL_GPU compute shaders. Each shader defines how many
// resources of each kind it has before including this file:
//   NUM_UNIFORM, NUM_RO_BUF, NUM_RW_TEX
//
// SPIR-V (Vulkan, and HLSL for Direct3D 12 through SPIRV-Cross) wants
//   set 0: read-only storage buffers, set 1: read-write storage textures then
//   read-write storage buffers, set 2: uniform buffers.
// MSL (built with --msl-decoration-binding, which keeps the binding number and
// drops the set) wants [[buffer(n)]] uniform buffers, then read-only storage
// buffers, then read-write storage buffers, and [[texture(n)]] storage
// textures. TARGET_MSL switches to those numbers.

#ifdef TARGET_MSL
#define RO_BUFFER(i) set = 0, binding = (NUM_UNIFORM + (i))
#define RW_BUFFER(i) set = 0, binding = (NUM_UNIFORM + NUM_RO_BUF + (i))
#define RW_IMAGE(i) set = 3, binding = (i)
#define UNIFORM(i) set = 0, binding = (i)
#else
#define RO_BUFFER(i) set = 0, binding = (i)
#define RW_BUFFER(i) set = 1, binding = (NUM_RW_TEX + (i))
#define RW_IMAGE(i) set = 1, binding = (i)
#define UNIFORM(i) set = 2, binding = (i)
#endif
