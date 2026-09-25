#pragma once

#include "common.hpp"

namespace MIInterrupt {
    constexpr u32 SP = 1 << 0;
    constexpr u32 SI = 1 << 1;
    constexpr u32 AI = 1 << 2;
    constexpr u32 VI = 1 << 3;
    constexpr u32 PI = 1 << 4;
    constexpr u32 DP = 1 << 5;
}

class MI {
public:
    MI();

    void reset();

    u32 read_reg(u32 addr) const;
    void write_reg(u32 addr, u32 val);

    void raise_interrupt(u32 intr_mask);
    void clear_interrupt(u32 intr_mask);

    bool is_interrupt_asserted() const {
        return (intr & intr_mask) != 0;
    }

    u32 get_intr() const { return intr; }
    u32 get_intr_mask() const { return intr_mask; }

private:
    u32 mode{0};
    u32 version{0x02020102};
    u32 intr{0};
    u32 intr_mask{0};
};
