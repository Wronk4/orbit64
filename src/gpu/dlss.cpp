#include "dlss.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#if defined(ORBIT64_DLSS) && defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"
#endif

namespace gpu {

namespace {

struct ModeInfo {
    const char* name;
    float ratio; // output / input
};
const ModeInfo kModeInfo[Dlss::kModes] = {
    {"Off", 1.0f},     {"DLAA", 1.0f},        {"Quality", 1.5f},
    {"Balanced", 1.724f}, {"Performance", 2.0f}, {"Ultra Performance", 3.0f},
};

} // namespace

const char* Dlss::mode_name(int mode) { return kModeInfo[std::clamp(mode, 0, kModes - 1)].name; }

void Dlss::output_size(int mode, int w, int h, int& ow, int& oh) {
    const float r = kModeInfo[std::clamp(mode, 0, kModes - 1)].ratio;
    ow = static_cast<int>(std::lround(w * r));
    oh = static_cast<int>(std::lround(h * r));
}

#if defined(ORBIT64_DLSS) && defined(_WIN32)

using Microsoft::WRL::ComPtr;

struct Dlss::Impl {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    UINT64 fence_value = 0;
    NVSDK_NGX_Parameter* params = nullptr;
    bool ngx = false;

    // Per size and mode.
    NVSDK_NGX_Handle* feature = nullptr;
    int w = 0, h = 0, ow = 0, oh = 0, mode = 0;
    ComPtr<ID3D12Resource> color, depth, motion, output, upload, readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT up_fp{}, rb_fp{};
    bool first = true;

    ~Impl() {
        wait();
        if (feature) NVSDK_NGX_D3D12_ReleaseFeature(feature);
        if (params) NVSDK_NGX_D3D12_DestroyParameters(params);
        if (ngx) NVSDK_NGX_D3D12_Shutdown1(device.Get());
        if (event) CloseHandle(event);
    }

    void wait() {
        if (!queue || !fence) return;
        queue->Signal(fence.Get(), ++fence_value);
        if (fence->GetCompletedValue() < fence_value) {
            fence->SetEventOnCompletion(fence_value, event);
            WaitForSingleObject(event, INFINITE);
        }
    }

    ComPtr<ID3D12Resource> texture(DXGI_FORMAT fmt, int tw, int th, D3D12_RESOURCE_FLAGS flags,
                                   D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = static_cast<UINT64>(tw);
        d.Height = static_cast<UINT>(th);
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = fmt;
        d.SampleDesc.Count = 1;
        d.Flags = flags;
        ComPtr<ID3D12Resource> r;
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r))))
            return nullptr;
        return r;
    }
    ComPtr<ID3D12Resource> buffer(D3D12_HEAP_TYPE type, UINT64 size) {
        D3D12_HEAP_PROPERTIES hp{type};
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = size;
        d.Height = 1;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> r;
        const D3D12_RESOURCE_STATES st =
            type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST;
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, IID_PPV_ARGS(&r))))
            return nullptr;
        return r;
    }
    void barrier(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = r;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = from;
        b.Transition.StateAfter = to;
        list->ResourceBarrier(1, &b);
    }
    void copy_in(ID3D12Resource* dst, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp) {
        D3D12_TEXTURE_COPY_LOCATION d{dst, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        d.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION s{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        s.PlacedFootprint = fp;
        list->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
    }
    bool submit(std::string& error) {
        if (FAILED(list->Close())) {
            error = "command list";
            return false;
        }
        ID3D12CommandList* l[] = {list.Get()};
        queue->ExecuteCommandLists(1, l);
        wait();
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        return true;
    }

    // Resources and the DLSS feature for a size and mode.
    bool setup(int nw, int nh, int nmode, std::string& error) {
        if (feature && nw == w && nh == h && nmode == mode) return true;
        wait();
        if (feature) NVSDK_NGX_D3D12_ReleaseFeature(feature);
        feature = nullptr;
        w = nw;
        h = nh;
        mode = nmode;
        Dlss::output_size(mode, w, h, ow, oh);
        const NVSDK_NGX_PerfQuality_Value pq[Dlss::kModes] = {
            NVSDK_NGX_PerfQuality_Value_DLAA,       NVSDK_NGX_PerfQuality_Value_DLAA,
            NVSDK_NGX_PerfQuality_Value_MaxQuality, NVSDK_NGX_PerfQuality_Value_Balanced,
            NVSDK_NGX_PerfQuality_Value_MaxPerf,    NVSDK_NGX_PerfQuality_Value_UltraPerformance};
        // The input has to lie in the range DLSS takes for that output.
        unsigned rw = 0, rh = 0, maxw = 0, maxh = 0, minw = 0, minh = 0;
        float sharp = 0.0f;
        if (NVSDK_NGX_SUCCEED(NGX_DLSS_GET_OPTIMAL_SETTINGS(params, ow, oh, pq[mode], &rw, &rh, &maxw, &maxh, &minw,
                                                            &minh, &sharp)) &&
            rw && rh && (static_cast<unsigned>(w) < minw || static_cast<unsigned>(w) > maxw ||
                         static_cast<unsigned>(h) < minh || static_cast<unsigned>(h) > maxh)) {
            ow = static_cast<int>(std::lround(double(ow) * w / rw));
            oh = static_cast<int>(std::lround(double(oh) * h / rh));
        }

        color = texture(DXGI_FORMAT_B8G8R8A8_UNORM, w, h, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        depth = texture(DXGI_FORMAT_R32_FLOAT, w, h, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        motion = texture(DXGI_FORMAT_R16G16_FLOAT, w, h, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        output = texture(DXGI_FORMAT_R8G8B8A8_UNORM, ow, oh, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (!color || !depth || !motion || !output) {
            error = "out of video memory";
            return false;
        }
        UINT64 up_size = 0, rb_size = 0;
        D3D12_RESOURCE_DESC cd = color->GetDesc(), od = output->GetDesc();
        device->GetCopyableFootprints(&cd, 0, 1, 0, &up_fp, nullptr, nullptr, &up_size);
        device->GetCopyableFootprints(&od, 0, 1, 0, &rb_fp, nullptr, nullptr, &rb_size);
        upload = buffer(D3D12_HEAP_TYPE_UPLOAD, up_size);
        readback = buffer(D3D12_HEAP_TYPE_READBACK, rb_size);
        if (!upload || !readback) {
            error = "out of memory";
            return false;
        }
        // Depth and motion vectors: zero (4 bytes a texel each, like the colour).
        void* p = nullptr;
        upload->Map(0, nullptr, &p);
        std::memset(p, 0, static_cast<size_t>(up_size));
        upload->Unmap(0, nullptr);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT zf = up_fp;
        zf.Footprint.Format = DXGI_FORMAT_R32_FLOAT;
        copy_in(depth.Get(), zf);
        zf.Footprint.Format = DXGI_FORMAT_R16G16_FLOAT;
        copy_in(motion.Get(), zf);
        barrier(depth.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        barrier(motion.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        NVSDK_NGX_DLSS_Create_Params cp{};
        cp.Feature.InWidth = static_cast<unsigned>(w);
        cp.Feature.InHeight = static_cast<unsigned>(h);
        cp.Feature.InTargetWidth = static_cast<unsigned>(ow);
        cp.Feature.InTargetHeight = static_cast<unsigned>(oh);
        cp.Feature.InPerfQualityValue = pq[mode];
        cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        const NVSDK_NGX_Result r = NGX_D3D12_CREATE_DLSS_EXT(list.Get(), 1, 1, &feature, params, &cp);
        if (NVSDK_NGX_FAILED(r)) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "DLSS feature creation failed (0x%08x) for %dx%d -> %dx%d", unsigned(r), w, h,
                          ow, oh);
            error = buf;
            feature = nullptr;
            submit(error);
            return false;
        }
        first = true;
        return submit(error);
    }
};

Dlss::Dlss() = default;
Dlss::~Dlss() = default;

bool Dlss::init() {
    if (tried_) return available_;
    tried_ = true;
    auto im = std::make_unique<Impl>();

    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
        error_ = "DXGI unavailable";
        return false;
    }
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) !=
                     DXGI_ERROR_NOT_FOUND;
         ++i) {
        DXGI_ADAPTER_DESC1 d{};
        adapter->GetDesc1(&d);
        if (d.VendorId == 0x10DE) { // NVIDIA
            char name[128] = {};
            WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, name, sizeof name - 1, nullptr, nullptr);
            adapter_ = name;
            break;
        }
        adapter.Reset();
    }
    if (!adapter) {
        error_ = "DLSS needs an NVIDIA RTX graphics card";
        return false;
    }
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&im->device)))) {
        error_ = "Direct3D 12 unavailable on " + adapter_;
        return false;
    }
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(im->device->CreateCommandQueue(&qd, IID_PPV_ARGS(&im->queue))) ||
        FAILED(im->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&im->alloc))) ||
        FAILED(im->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, im->alloc.Get(), nullptr,
                                             IID_PPV_ARGS(&im->list))) ||
        FAILED(im->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&im->fence)))) {
        error_ = "Direct3D 12 setup failed";
        return false;
    }
    im->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    // NGX logs into a folder of its own; the temporary one will do.
    wchar_t tmp[MAX_PATH] = L".";
    GetTempPathW(MAX_PATH, tmp);
    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init_with_ProjectID("4c0b6f5e-5d3a-4f6e-9a77-0b17a64e0064",
                                                             NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", tmp,
                                                             im->device.Get());
    if (NVSDK_NGX_FAILED(r)) {
        char buf[96];
        std::snprintf(buf, sizeof buf, "NGX initialisation failed (0x%08x): driver too old?", unsigned(r));
        error_ = buf;
        return false;
    }
    im->ngx = true;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_D3D12_GetCapabilityParameters(&im->params)) || !im->params) {
        error_ = "NGX capability query failed";
        return false;
    }
    int ok = 0, needs_driver = 0;
    NVSDK_NGX_Parameter_GetI(im->params, NVSDK_NGX_Parameter_SuperSampling_Available, &ok);
    NVSDK_NGX_Parameter_GetI(im->params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs_driver);
    if (!ok) {
        error_ = needs_driver ? "DLSS needs a newer NVIDIA driver"
                              : "DLSS unavailable (nvngx_dlss.dll missing or GPU without Tensor cores)";
        return false;
    }
    impl_ = std::move(im);
    available_ = true;
    return true;
}

bool Dlss::evaluate(const std::uint32_t* argb, int w, int h, int mode, bool temporal, std::vector<std::uint32_t>& out,
                    int& ow, int& oh) {
    if (!available_ || !argb || w <= 0 || h <= 0 || mode <= Off || mode >= kModes) return false;
    Impl& im = *impl_;
    std::string err;
    if (!im.setup(w, h, mode, err)) {
        if (!err.empty()) error_ = err;
        return false;
    }
    // Colour in: ARGB8888 words are B8G8R8A8 texels.
    std::uint8_t* up = nullptr;
    if (FAILED(im.upload->Map(0, nullptr, reinterpret_cast<void**>(&up)))) return false;
    const UINT pitch = im.up_fp.Footprint.RowPitch;
    for (int y = 0; y < h; ++y) std::memcpy(up + im.up_fp.Offset + size_t(y) * pitch, argb + size_t(y) * w, size_t(w) * 4);
    im.upload->Unmap(0, nullptr);
    if (!im.first)
        im.barrier(im.color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    im.copy_in(im.color.Get(), im.up_fp);
    im.barrier(im.color.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    NVSDK_NGX_D3D12_DLSS_Eval_Params ep{};
    ep.Feature.pInColor = im.color.Get();
    ep.Feature.pInOutput = im.output.Get();
    ep.pInDepth = im.depth.Get();
    ep.pInMotionVectors = im.motion.Get();
    ep.InRenderSubrectDimensions = {static_cast<unsigned>(w), static_cast<unsigned>(h)};
    ep.InReset = (temporal && !im.first) ? 0 : 1;
    ep.InMVScaleX = ep.InMVScaleY = 1.0f;
    const NVSDK_NGX_Result r = NGX_D3D12_EVALUATE_DLSS_EXT(im.list.Get(), im.feature, im.params, &ep);
    im.first = false;
    if (NVSDK_NGX_FAILED(r)) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "DLSS evaluation failed (0x%08x)", unsigned(r));
        error_ = buf;
        im.submit(err);
        return false;
    }

    // Output back to memory.
    im.barrier(im.output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst{im.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    dst.PlacedFootprint = im.rb_fp;
    D3D12_TEXTURE_COPY_LOCATION src{im.output.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    src.SubresourceIndex = 0;
    im.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    im.barrier(im.output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!im.submit(err)) return false;

    const std::uint8_t* rb = nullptr;
    if (FAILED(im.readback->Map(0, nullptr, reinterpret_cast<void**>(const_cast<std::uint8_t**>(&rb))))) return false;
    ow = im.ow;
    oh = im.oh;
    out.resize(size_t(ow) * oh);
    const UINT rp = im.rb_fp.Footprint.RowPitch;
    for (int y = 0; y < oh; ++y) {
        const std::uint8_t* row = rb + im.rb_fp.Offset + size_t(y) * rp;
        std::uint32_t* o = out.data() + size_t(y) * ow;
        for (int x = 0; x < ow; ++x)
            o[x] = 0xFF000000u | (std::uint32_t(row[x * 4]) << 16) | (std::uint32_t(row[x * 4 + 1]) << 8) | row[x * 4 + 2];
    }
    D3D12_RANGE none{0, 0};
    im.readback->Unmap(0, &none);
    return true;
}

#else // no DLSS SDK

struct Dlss::Impl {};
Dlss::Dlss() = default;
Dlss::~Dlss() = default;
bool Dlss::init() {
    tried_ = true;
    error_ = "this build has no DLSS (Windows, with the DLSS SDK in third_party/dlss: tools/fetch_dlss.py)";
    return false;
}
bool Dlss::evaluate(const std::uint32_t*, int, int, int, bool, std::vector<std::uint32_t>&, int&, int&) { return false; }

#endif

} // namespace gpu
