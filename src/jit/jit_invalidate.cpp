#include "jit_invalidate.hpp"

namespace jit {
namespace {
void* g_owner = nullptr;
InvalidateFn g_fn = nullptr;
void* g_watch_owner = nullptr;
WatchFn g_watch_fn = nullptr;
void* g_write_owner = nullptr;
WriteFn g_write_fn = nullptr;
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
