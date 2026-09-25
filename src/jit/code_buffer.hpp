#pragma once
// A growable arena of executable memory for JIT-compiled blocks.
//
// Code is written while the arena is writable, then made executable before
// it is ever run (W^X). How that toggle happens depends on the platform:
//
// - Apple Silicon: the arena is a MAP_JIT mapping and each thread flips its
//   own view of it with pthread_jit_write_protect_np() - a register write,
//   no syscall. A process without the hardened runtime needs no entitlement
//   for that (with it, com.apple.security.cs.allow-jit - which the mprotect
//   path below could not do without either). The mprotect path is kept only
//   as a fallback in case the MAP_JIT mapping is refused.
// - Everywhere else: mprotect/VirtualProtect, i.e. two syscalls per
//   compiled block - but only over the page(s) the new block is written to,
//   never the whole arena. Flipping all 16 MB each time made a compile cost
//   tens of microseconds on Windows (it scales with the page count), which
//   made the JIT slower than the interpreter in games that compile a lot.
//
// Either way only the bytes appended since the previous make_executable()
// have their instruction-cache lines invalidated; flushing the whole used
// arena instead made every compile O(arena size).
//
// A dual RW/RX mapping of one shared-memory object (write through one alias,
// execute through the other, no toggling at all) was tried here too, but
// current macOS flatly SIGBUSes when executing from a MAP_SHARED
// PROT_EXEC-only mapping - reads/writes through it work fine, execution
// doesn't. So this stays a single mapping with an explicit toggle.

#include "../common.hpp"
#include <vector>

class CodeBuffer {
public:
    CodeBuffer();
    ~CodeBuffer();

    CodeBuffer(const CodeBuffer&) = delete;
    CodeBuffer& operator=(const CodeBuffer&) = delete;

    // Resets the arena, discarding all previously emitted code. Every
    // pointer returned by write_ptr() before this call becomes invalid.
    void reset();

    // Call before emitting bytes for a new block. Returns the writable
    // pointer to append at; the buffer is left in "writable" state.
    u8* write_ptr() { return base_ + size_; }
    size_t remaining() const { return capacity_ - size_; }

    // Appends `n` bytes (used by the assemblers to flush their instruction
    // stream).
    void commit(size_t n) { size_ += n; }

    // Makes the pages the last make_writable() opened executable again and
    // flushes instruction caches for everything appended since the last
    // call. Must be called before calling any function pointer into the
    // arena, and again after any further writes. With MAP_JIT this affects
    // the whole arena but only for the calling thread, so it has to happen
    // on the thread that runs the code.
    void make_executable();

    // Makes the next `n` bytes at write_ptr() writable so they can be
    // appended (only the pages they touch; with MAP_JIT the whole arena, for
    // the calling thread).
    void make_writable(size_t n);

    // Rewrites one 4-byte instruction word of code that is already executable
    // (the AArch64 backend retargets a branch to a block compiled after it)
    // and makes that word visible to instruction fetch. The arena is left
    // executable, as after make_executable().
    void patch32(u8* where, u32 word);

    // Identity on this (single-mapping) implementation; exists so callers
    // can be written the same way a dual-mapping backend would need them to
    // be, in case a future platform-specific backend brings that back.
    u8* exec_ptr(u8* write_ptr_value) const { return write_ptr_value; }

    u8* base() { return base_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

private:
    u8* base_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
    size_t flushed_ = 0; // [0, flushed_) is already visible to instruction fetch
    size_t page_size_ = 4096;
    size_t writable_lo_ = 0, writable_hi_ = 0; // page range make_writable() last opened
    [[maybe_unused]] bool map_jit_ = false; // Apple Silicon MAP_JIT arena (per-thread W^X toggle)

    void allocate(size_t capacity);
    void release();
    void flush_icache();
};
