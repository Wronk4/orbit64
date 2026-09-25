#include "jit_invalidate.hpp"

namespace jit {
namespace {
void* g_owner = nullptr;
InvalidateFn g_fn = nullptr;
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
}

} // namespace jit
