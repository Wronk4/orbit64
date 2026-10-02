#include "jit_invalidate.hpp"

namespace jit {
namespace {
void* g_owner = nullptr;
InvalidateFn g_fn = nullptr;
void* g_watch_owner = nullptr;
WatchFn g_watch_fn = nullptr;
void* g_write_owner = nullptr;
WriteFn g_write_fn = nullptr;
void* g_read_watch_owner = nullptr;
ReadWatchFn g_read_watch_fn = nullptr;
void* g_read_owner = nullptr;
ReadFn g_read_fn = nullptr;
}

const u8* g_page_flags = nullptr;
u32 g_page_count = 0;
bool g_read_checks = false;

namespace {
void* g_codegen_owner = nullptr;
CodegenFn g_codegen_fn = nullptr;
}
void set_codegen_hook(void* owner, CodegenFn fn) {
    g_codegen_owner = owner;
    g_codegen_fn = fn;
}
void clear_codegen_hook_if(void* owner) {
    if (g_codegen_owner == owner) {
        g_codegen_owner = nullptr;
        g_codegen_fn = nullptr;
    }
}
void set_read_checks(bool on) {
    if (on == g_read_checks) return;
    g_read_checks = on;
    if (on && g_codegen_fn) g_codegen_fn(g_codegen_owner);
}

void set_read_watch_hook(void* owner, ReadWatchFn fn) {
    g_read_watch_owner = owner;
    g_read_watch_fn = fn;
}
void clear_read_watch_hook_if(void* owner) {
    if (g_read_watch_owner == owner) {
        g_read_watch_owner = nullptr;
        g_read_watch_fn = nullptr;
    }
}
void watch_reads(u32 paddr, u32 len, bool on) {
    if (g_read_watch_fn && len) g_read_watch_fn(g_read_watch_owner, paddr, len, on);
}
void set_read_hook(void* owner, ReadFn fn) {
    g_read_owner = owner;
    g_read_fn = fn;
}
void clear_read_hook_if(void* owner) {
    if (g_read_owner == owner) {
        g_read_owner = nullptr;
        g_read_fn = nullptr;
    }
}
void notify_read_slow(u32 paddr, u32 len) {
    if (g_read_fn) g_read_fn(g_read_owner, paddr, len);
}

void set_watch_hook(void* owner, WatchFn fn) {
    g_watch_owner = owner;
    g_watch_fn = fn;
}
void clear_watch_hook_if(void* owner) {
    if (g_watch_owner == owner) {
        g_watch_owner = nullptr;
        g_watch_fn = nullptr;
    }
}
void watch_rdram(u32 paddr, u32 len) {
    if (g_watch_fn) g_watch_fn(g_watch_owner, paddr, len);
}
void set_write_hook(void* owner, WriteFn fn) {
    g_write_owner = owner;
    g_write_fn = fn;
}
void clear_write_hook_if(void* owner) {
    if (g_write_owner == owner) {
        g_write_owner = nullptr;
        g_write_fn = nullptr;
    }
}

void set_invalidate_hook(void* owner, InvalidateFn fn) {
    g_owner = owner;
    g_fn = fn;
}

void clear_invalidate_hook() {
    g_owner = nullptr;
    g_fn = nullptr;
}

void clear_invalidate_hook_if(void* owner) {
    if (g_owner == owner) clear_invalidate_hook();
}

void notify_code_write(u32 paddr, u32 len) {
    if (g_fn) g_fn(g_owner, paddr, len);
    if (g_write_fn) g_write_fn(g_write_owner, paddr, len);
}

} // namespace jit
