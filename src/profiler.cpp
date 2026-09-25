#include "profiler.hpp"

#if defined(_WIN32) && (defined(__x86_64__) || defined(_M_X64))
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace profiler {
namespace {

constexpr int kMaxDepth = 24;

std::atomic<bool> g_running{false};
std::thread g_thread;
HANDLE g_target = nullptr;
std::string g_out;
std::vector<std::vector<uint64_t>> g_samples;
uint64_t g_stack_lo = 0, g_stack_hi = 0;
uint64_t g_exe_lo = 0, g_exe_hi = 0;

void sample_loop(unsigned interval_us) {
    // A high-resolution waitable timer: plain Sleep() rounds up to the
    // ~15.6 ms scheduler tick, far too coarse for sampling.
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    while (g_running.load(std::memory_order_relaxed)) {
        // Nothing between SuspendThread and ResumeThread may allocate or take
        // a lock the target could be holding (the heap lock in particular) -
        // that deadlocks. The chain goes into a fixed buffer and is copied
        // into g_samples only after the thread runs again.
        uint64_t chain[kMaxDepth + 1];
        int depth = 0;
        if (SuspendThread(g_target) != static_cast<DWORD>(-1)) {
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            if (GetThreadContext(g_target, &ctx)) {
                chain[depth++] = ctx.Rip;
                // Walk the stack with the unwind tables (.pdata) - GCC's Win64
                // frame pointer doesn't point at the saved one, so an RBP walk
                // doesn't work. A function without an entry is a leaf (return
                // address at [rsp]) if it is in the executable; anything else
                // without one (JIT-compiled code) ends the walk.
                for (int i = 0; i < kMaxDepth; ++i) {
                    DWORD64 image_base = 0;
                    PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
                    if (fn) {
                        PVOID handler_data = nullptr;
                        DWORD64 establisher = 0;
                        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, fn, &ctx, &handler_data, &establisher, nullptr);
                    } else {
                        if (ctx.Rip < g_exe_lo || ctx.Rip >= g_exe_hi) break;
                        if (ctx.Rsp < g_stack_lo || ctx.Rsp + 8 > g_stack_hi) break;
                        ctx.Rip = *reinterpret_cast<const uint64_t*>(ctx.Rsp);
                        ctx.Rsp += 8;
                    }
                    if (ctx.Rip == 0 || ctx.Rsp < g_stack_lo || ctx.Rsp >= g_stack_hi) break;
                    chain[depth++] = ctx.Rip;
                }
            }
            ResumeThread(g_target);
            if (depth > 0) g_samples.emplace_back(chain, chain + depth);
        }
        if (timer) {
            LARGE_INTEGER due;
            due.QuadPart = -static_cast<LONGLONG>(interval_us) * 10; // relative, 100 ns units
            SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
            WaitForSingleObject(timer, INFINITE);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(interval_us));
        }
    }
    if (timer) CloseHandle(timer);
}

} // namespace

bool start(const std::string& out_path, unsigned interval_us) {
    if (g_running) return false;
    g_out = out_path;
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &g_target,
                         THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0)) {
        return false;
    }
    ULONG_PTR lo = 0, hi = 0;
    GetCurrentThreadStackLimits(&lo, &hi);
    g_stack_lo = lo;
    g_stack_hi = hi;
    const auto* exe = reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(exe + reinterpret_cast<const IMAGE_DOS_HEADER*>(exe)->e_lfanew);
    g_exe_lo = reinterpret_cast<uintptr_t>(exe);
    g_exe_hi = g_exe_lo + nt->OptionalHeader.SizeOfImage;
    g_samples.clear();
    g_samples.reserve(1 << 16);
    g_running = true;
    g_thread = std::thread(sample_loop, interval_us);
    return true;
}

void stop() {
    if (!g_running) return;
    g_running = false;
    g_thread.join();
    CloseHandle(g_target);
    g_target = nullptr;

    FILE* f = std::fopen(g_out.c_str(), "w");
    if (!f) return;
    // Header: where the executable was loaded, so addresses can be mapped
    // back to its (preferred-base) symbol table despite ASLR.
    std::fprintf(f, "base %llx\n", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr))));
    for (const auto& chain : g_samples) {
        for (size_t i = 0; i < chain.size(); ++i) std::fprintf(f, i ? " %llx" : "%llx", static_cast<unsigned long long>(chain[i]));
        std::fputc('\n', f);
    }
    std::fclose(f);
    std::printf("[Profiler] %zu samples written to %s\n", g_samples.size(), g_out.c_str());
}

} // namespace profiler

#else

namespace profiler {
bool start(const std::string&, unsigned) { return false; }
void stop() {}
} // namespace profiler

#endif
