#pragma once
// Tiny process-wide hook so RDRAM writes that could clobber cached JIT code
// (regular CPU stores, PI/SI/SP DMA, the frontend's memory pokes) can
// invalidate the recompiler's block cache without threading a Recompiler*
// through Bus/PI/SI/RSP constructors. There is exactly one Emulator (and one
// Recompiler) alive at a time in this application, so a single global slot is
// fine; Emulator wires it up in its constructor and clears it in its
// destructor.
//
// Invalidation itself is all-or-nothing (see Recompiler::invalidate_all), but
// it only happens when the reported range overlaps compiled code - so every
// writer must report exactly the bytes it wrote. A conservative "all of
// RDRAM" would drop the whole cache on every call (it used to, on every
// SP/PI/SI register write, several times a frame).

#include "../common.hpp"

namespace jit {

// `paddr`/`len` is the physical RDRAM range just written; the recompiler
// uses it to skip invalidation entirely when the write didn't touch a page
// any cached block was compiled from (see Recompiler::request_invalidate).
using InvalidateFn = void (*)(void* owner, u32 paddr, u32 len);

void set_invalidate_hook(void* owner, InvalidateFn fn);
void clear_invalidate_hook();
// Clears the hook only if `owner` is still the currently-registered one -
// safe to call from a destructor even if a newer instance has since replaced
// the registration (e.g. Emulator was torn down and rebuilt).
void clear_invalidate_hook_if(void* owner);

// Called from Bus::write8/16/32/64, from the PI/SI/SP DMA paths that poke
// RDRAM directly, and from the frontend's memory pokes. No-ops when no
// recompiler is active (interpreter-only mode, or no ROM loaded yet).
// `paddr`/`len` describe the bytes just written.
void notify_code_write(u32 paddr, u32 len);

// RDRAM the RDP keeps ninth bits for (rdp_exact.hpp): the recompiler's
// compiled stores to watched memory take the slow path, so every CPU (and
// DMA) write to it reaches the write hook, which gives the bytes written the
// ninth bits a CPU write leaves.
using WatchFn = void (*)(void* owner, u32 paddr, u32 len);
void set_watch_hook(void* owner, WatchFn fn); // who marks watched memory (the recompiler)
void clear_watch_hook_if(void* owner);
void watch_rdram(u32 paddr, u32 len);
using WriteFn = void (*)(void* owner, u32 paddr, u32 len);
void set_write_hook(void* owner, WriteFn fn); // who is told about writes (the RDP)
void clear_write_hook_if(void* owner);

} // namespace jit
