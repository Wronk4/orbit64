#pragma once

#include "common.hpp"
#include <algorithm>
#include <vector>
#include <array>

class Cartridge;
class PIF;
class Controller;
class MI;
class VI;
class AI;
class PI;
class SI;
class RSP;
class RDP;
class CPU;

enum class TLBResult {
    SUCCESS,
    MISS,
    INVALID,
    MODIFIED
};

struct TLBEntry {
    u32 page_mask{0};
    u64 entry_hi{0};
    u64 entry_lo0{0};
    u64 entry_lo1{0};
    bool initialized{false};

    template <class S> void serialize(S& s) { s(page_mask, entry_hi, entry_lo0, entry_lo1, initialized); }
};

class Bus {
public:
    Bus(Cartridge& cart, PIF& pif, Controller controllers[4],
        MI& mi, VI& vi, AI& ai, PI& pi, SI& si, RSP& rsp, RDP& rdp);

    void reset();

    // Virtual to Physical address translation
    TLBResult translate_vaddr(u64 vaddr, u32& paddr, bool is_write, u8 current_asid = 0);

    // Physical memory access
    u8 read8(u32 paddr);
    u16 read16(u32 paddr);
    u32 read32(u32 paddr);
    u64 read64(u32 paddr);

    void write8(u32 paddr, u8 val);
    void write16(u32 paddr, u16 val);
    void write32(u32 paddr, u32 val);
    void write64(u32 paddr, u64 val);

    // Virtual memory access (with translation)
    u8 read_v8(u64 vaddr, TLBResult& result, u8 current_asid = 0);
    u16 read_v16(u64 vaddr, TLBResult& result, u8 current_asid = 0);
    u32 read_v32(u64 vaddr, TLBResult& result, u8 current_asid = 0);
    u64 read_v64(u64 vaddr, TLBResult& result, u8 current_asid = 0);

    void write_v8(u64 vaddr, u8 val, TLBResult& result, u8 current_asid = 0);
    void write_v16(u64 vaddr, u16 val, TLBResult& result, u8 current_asid = 0);
    void write_v32(u64 vaddr, u32 val, TLBResult& result, u8 current_asid = 0);
    void write_v64(u64 vaddr, u64 val, TLBResult& result, u8 current_asid = 0);

    u8* get_rdram() { return rdram.data(); }
    const u8* get_rdram() const { return rdram.data(); }
    size_t get_rdram_size() const { return rdram.size(); }
    // The RDRAM the CPU sees: 8 MB with the Expansion Pak, 4 MB without.
    // Above it the CPU reads zeros and its writes go nowhere, which is how
    // libultra's osGetMemSize() finds out. The buffer itself stays 8 MB.
    // The JIT compiles this into its blocks: flush it after a change.
    void set_ram_limit(u32 bytes) { ram_limit_ = std::min<u32>(bytes, RDRAM_SIZE); }
    u32 get_ram_limit() const { return ram_limit_; }

    MI& get_mi() { return mi; }
    const MI& get_mi() const { return mi; }

    // TLB management
    void set_tlb_entry(size_t index, const TLBEntry& entry);
    const TLBEntry& get_tlb_entry(size_t index) const;
    // Changes whenever a TLB entry is written, so the recompiler knows when
    // translations it cached (virtual pc -> compiled block) may be stale.
    u32 tlb_generation() const { return tlb_gen_; }

    // Save states (savestate.hpp).
    template <class S> void serialize(S& s) {
        s.fixed(rdram);
        s(tlb_entries, ri_mode, ri_config, ri_current_load, ri_select, ri_refresh, ri_latency, ri_error, ri_werror);
        if constexpr (S::loading) tlb_gen_++; // the TLB may be different now
    }

private:
    Cartridge& cart;
    PIF& pif;
    Controller* controllers;
    MI& mi;
    VI& vi;
    AI& ai;
    PI& pi;
    SI& si;
    RSP& rsp;
    RDP& rdp;

    std::vector<u8> rdram;
    u32 ram_limit_{RDRAM_SIZE};
    std::array<TLBEntry, 32> tlb_entries{};
    u32 tlb_gen_ = 0;
    // Recent translations of 4 KB pages (every TLB page is a multiple of
    // that), good while the TLB (tlb_gen_) and the ASID stay the same:
    // translate_vaddr() otherwise searches all 32 entries on every access
    // to mapped memory.
    struct TlbCacheEntry {
        u32 vpn = ~0u;  // virtual address >> 12
        u32 ppage = 0;  // physical address of the page
        u32 gen = 0;
        u8 asid = 0;
        bool writable = false;
    };
    std::array<TlbCacheEntry, 64> tlb_cache_{};

    // RI registers
    u32 ri_mode{0};
    u32 ri_config{0};
    u32 ri_current_load{0};
    u32 ri_select{0};
    u32 ri_refresh{0};
    u32 ri_latency{0};
    u32 ri_error{0};
    u32 ri_werror{0};
};
