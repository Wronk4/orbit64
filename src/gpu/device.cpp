#include "device.hpp"
#include "shaders_gen.hpp"

#include <SDL3/SDL.h>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <d3dcompiler.h>
#endif

namespace gpu {

SDL_GPUDevice* create_device() {
    const bool debug = std::getenv("ORBIT64_GPU_DEBUG") != nullptr;
    auto make = [debug](bool spirv, bool msl, bool d3d, const char* name) {
        SDL_PropertiesID props = SDL_CreateProperties();
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_SPIRV_BOOLEAN, spirv);
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_MSL_BOOLEAN, msl);
        // SDL's own renderer brings DXIL shaders; ours are compiled to DXBC.
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_DXIL_BOOLEAN, d3d);
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_DXBC_BOOLEAN, d3d);
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_DEBUGMODE_BOOLEAN, debug);
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_PREFERLOWPOWER_BOOLEAN, false);
        if (name) SDL_SetStringProperty(props, SDL_PROP_GPU_DEVICE_CREATE_NAME_STRING, name);
        SDL_GPUDevice* d = SDL_CreateGPUDeviceWithProperties(props);
        SDL_DestroyProperties(props);
        return d;
    };
    if (const char* name = std::getenv("ORBIT64_GPU_DRIVER"); name && *name) return make(true, true, true, name);
#if defined(_WIN32)
    // SDL prefers Direct3D 12, where the shaders have to be compiled at run
    // time; Vulkan runs the same SPIR-V as on Linux. Direct3D 12 is the
    // fallback for machines without a Vulkan driver.
    if (SDL_GPUDevice* d = make(true, false, false, nullptr)) return d;
    return make(true, false, true, nullptr);
#else
    return make(true, true, false, nullptr);
#endif
}

namespace {

#if defined(_WIN32)
// HLSL -> DXBC (shader model 5.1) with the compiler every Windows 10/11 ships.
std::vector<Uint8> compile_dxbc(const char* hlsl, std::string& error) {
    static HMODULE lib = LoadLibraryA("d3dcompiler_47.dll");
    if (!lib) {
        error = "d3dcompiler_47.dll is missing";
        return {};
    }
    auto compile = reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(GetProcAddress(lib, "D3DCompile")));
    if (!compile) {
        error = "D3DCompile is missing";
        return {};
    }
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(hlsl, std::strlen(hlsl), "orbit64", nullptr, nullptr, "main", "cs_5_1",
                               D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    std::vector<Uint8> out;
    if (SUCCEEDED(hr) && code) {
        const auto* p = static_cast<const Uint8*>(code->GetBufferPointer());
        out.assign(p, p + code->GetBufferSize());
    } else {
        error = errors ? std::string(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize())
                       : "D3DCompile failed";
    }
    if (code) code->Release();
    if (errors) errors->Release();
    return out;
}
#endif

struct Resources {
    Uint32 ro_buffers = 0, rw_buffers = 0, rw_textures = 0, uniforms = 1;
    Uint32 tx = 8, ty = 8;
};

} // namespace

static SDL_GPUComputePipeline* make_pipeline(SDL_GPUDevice* dev, const shaders::Blob& blob, const Resources& r,
                                             const char* name, std::string& error) {
    SDL_GPUComputePipelineCreateInfo ci{};
    ci.num_readonly_storage_buffers = r.ro_buffers;
    ci.num_readwrite_storage_buffers = r.rw_buffers;
    ci.num_readwrite_storage_textures = r.rw_textures;
    ci.num_uniform_buffers = r.uniforms;
    ci.threadcount_x = r.tx;
    ci.threadcount_y = r.ty;
    ci.threadcount_z = 1;
    const SDL_GPUShaderFormat formats = SDL_GetGPUShaderFormats(dev);
    std::vector<Uint8> dxbc;
    if (formats & SDL_GPU_SHADERFORMAT_MSL) {
        ci.format = SDL_GPU_SHADERFORMAT_MSL;
        ci.code = reinterpret_cast<const Uint8*>(blob.msl);
        ci.code_size = std::strlen(blob.msl);
        ci.entrypoint = "main0";
    } else if (formats & SDL_GPU_SHADERFORMAT_SPIRV) {
        ci.format = SDL_GPU_SHADERFORMAT_SPIRV;
        ci.code = blob.spirv;
        ci.code_size = blob.spirv_size;
        ci.entrypoint = "main";
    } else if (formats & SDL_GPU_SHADERFORMAT_DXBC) {
#if defined(_WIN32)
        dxbc = compile_dxbc(blob.hlsl, error);
        if (dxbc.empty()) {
            error = std::string(name) + ": " + error;
            return nullptr;
        }
        ci.format = SDL_GPU_SHADERFORMAT_DXBC;
        ci.code = dxbc.data();
        ci.code_size = dxbc.size();
        ci.entrypoint = "main";
#endif
    }
    if (!ci.code) {
        error = "no shaders for this graphics driver";
        return nullptr;
    }
    ci.props = SDL_CreateProperties();
    SDL_SetStringProperty(ci.props, SDL_PROP_GPU_COMPUTEPIPELINE_CREATE_NAME_STRING, name);
    SDL_GPUComputePipeline* p = SDL_CreateGPUComputePipeline(dev, &ci);
    SDL_DestroyProperties(ci.props);
    if (!p) error = std::string(name) + ": " + SDL_GetError();
    return p;
}

Device::Device(SDL_GPUDevice* device) : device_(device) {
    if (!device_) {
        error_ = "no GPU device";
        return;
    }
    Resources raster;
    raster.ro_buffers = 2;
    raster.rw_buffers = 2;
    Resources compose;
    compose.ro_buffers = 2;
    compose.rw_textures = 1;
    Resources fill;
    fill.rw_buffers = 1;
    fill.tx = 64;
    fill.ty = 1;
    Resources decode;
    decode.ro_buffers = 1;
    decode.rw_buffers = 1;
    decode.tx = 256;
    decode.ty = 1;
    raster_ = make_pipeline(device_, shaders::raster, raster, "orbit64 raster", error_);
    if (raster_) compose_ = make_pipeline(device_, shaders::compose, compose, "orbit64 compose", error_);
    if (compose_) fill_ = make_pipeline(device_, shaders::fill, fill, "orbit64 fill", error_);
    if (fill_) decode_ = make_pipeline(device_, shaders::decode, decode, "orbit64 decode", error_);
    if (!ok()) SDL_Log("GPU renderer unavailable: %s", error_.c_str());
}

Device::~Device() {
    if (raster_) SDL_ReleaseGPUComputePipeline(device_, raster_);
    if (compose_) SDL_ReleaseGPUComputePipeline(device_, compose_);
    if (fill_) SDL_ReleaseGPUComputePipeline(device_, fill_);
    if (decode_) SDL_ReleaseGPUComputePipeline(device_, decode_);
}

} // namespace gpu
