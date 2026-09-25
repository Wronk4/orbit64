#pragma once

#include "common.hpp"
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

    MI& get_mi() { return mi; }
    const MI& get_mi() const { return mi; }

    // TLB management
    void set_tlb_entry(size_t index, const TLBEntry& entry);
    const TLBEntry& get_tlb_entry(size_t index) const;

    // Save states (savestate.hpp).
    template <class S> void serialize(S& s) {
        s.fixed(rdram);
        s(tlb_entries, ri_mode, ri_config, ri_current_load, ri_select, ri_refresh, ri_latency, ri_error, ri_werror);
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
    std::array<TLBEntry, 32> tlb_entries{};

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
