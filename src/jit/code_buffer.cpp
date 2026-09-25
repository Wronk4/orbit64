#include "code_buffer.hpp"
#include <cstdlib>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#define ORBIT64_MAP_JIT 1
#endif

namespace {
constexpr size_t kDefaultCapacity = 16 * 1024 * 1024; // 16 MB of code
}

CodeBuffer::CodeBuffer() {
    allocate(kDefaultCapacity);
}

CodeBuffer::~CodeBuffer() {
    release();
}

void CodeBuffer::allocate(size_t capacity) {
#if defined(_WIN32)
    base_ = static_cast<u8*>(VirtualAlloc(nullptr, capacity, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!base_) throw std::runtime_error("CodeBuffer: VirtualAlloc failed");
#else
    void* p = MAP_FAILED;
#if defined(ORBIT64_MAP_JIT)
    p = mmap(nullptr, capacity, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT, -1, 0);
    map_jit_ = p != MAP_FAILED;
#endif
    if (p == MAP_FAILED) p = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("CodeBuffer: mmap failed");
    base_ = static_cast<u8*>(p);
#endif
    capacity_ = capacity;
    size_ = 0;
    flushed_ = 0;
    executable_ = false;
}

void CodeBuffer::release() {
    if (!base_) return;
#if defined(_WIN32)
    VirtualFree(base_, 0, MEM_RELEASE);
#else
    munmap(base_, capacity_);
#endif
    base_ = nullptr;
    capacity_ = 0;
    size_ = 0;
}

void CodeBuffer::reset() {
    size_ = 0;
    flushed_ = 0;
    make_writable();
}

void CodeBuffer::make_writable() {
#if defined(ORBIT64_MAP_JIT)
    if (map_jit_) {
        // Always toggle: the protection is per thread, so executable_ (shared)
        // can't tell whether *this* thread still has to flip - the arena is
        // e.g. reset on the UI thread by Emulator::load_rom() and then filled
        // on the emulator thread. The toggle itself costs next to nothing.
        pthread_jit_write_protect_np(0);
        executable_ = false;
        return;
    }
#endif
    if (!executable_) return;
#if defined(_WIN32)
    DWORD old_protect;
    VirtualProtect(base_, capacity_, PAGE_READWRITE, &old_protect);
#else
    mprotect(base_, capacity_, PROT_READ | PROT_WRITE);
#endif
    executable_ = false;
}

void CodeBuffer::make_executable() {
#if defined(ORBIT64_MAP_JIT)
    if (map_jit_) {
        pthread_jit_write_protect_np(1);
        flush_icache();
        executable_ = true;
        return;
    }
#endif
    if (executable_) return;
#if defined(_WIN32)
    DWORD old_protect;
    VirtualProtect(base_, capacity_, PAGE_EXECUTE_READ, &old_protect);
#else
    mprotect(base_, capacity_, PROT_READ | PROT_EXEC);
#endif
    flush_icache();
    executable_ = true;
}

// The arena is append-only between resets, so everything below flushed_ is
// unchanged since it was last made visible - only the new tail needs it.
void CodeBuffer::flush_icache() {
    if (size_ > flushed_) {
#if defined(__APPLE__)
        sys_icache_invalidate(base_ + flushed_, size_ - flushed_);
#elif defined(__GNUC__) || defined(__clang__)
        __builtin___clear_cache(reinterpret_cast<char*>(base_ + flushed_), reinterpret_cast<char*>(base_ + size_));
#endif
    }
    flushed_ = size_;
}
