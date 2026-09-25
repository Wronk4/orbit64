#pragma once
// AArch64 code generator for the recompiler (backend_a64.cpp). Internal to
// src/jit: only recompiler.cpp calls into it.

#include "recompiler.hpp"
#include <functional>
#include <vector>

class CPU;
class Bus;
class CodeBuffer;

// Emits the chaining runtime (entry trampoline, chain exit, dispatcher) at
// the current end of `code`. `jcache_bits`: log2 of the dispatcher table's
// entry count. False if the arena has no room.
bool emit_a64_runtime(CodeBuffer& code, Recompiler::ChainRuntime& rt, u32 jcache_bits);

struct A64BlockEnv {
    const Recompiler::ChainRuntime* rt = nullptr;
    // Whether blocks may jump to each other at all (Recompiler::set_chaining).
    bool chain = true;
    // The block is reached through the TLB (Recompiler::map_blocks_): it stops
    // at the end of its 4 KB page and gets to other pages only through the
    // dispatcher.
    bool mapped = false;
    // Entry of the compiled, linkable block at a physical address, or null.
    std::function<const void*(u32 paddr)> link_target;
};

// Compiles the block starting at physical address `start_paddr` into `code`
// (see Recompiler::compile_block). A null fn means "not JIT-able here".
// Branches to blocks not compiled yet are appended to `pending`.
Recompiler::Block compile_block_a64(CPU& cpu, Bus& bus, u32 start_paddr, CodeBuffer& code, bool delay_slot,
                                    const A64BlockEnv& env, std::vector<Recompiler::PendingLink>& pending);
