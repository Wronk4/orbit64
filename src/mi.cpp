#include "mi.hpp"

MI::MI() {
    reset();
}

void MI::reset() {
    mode = 0;
    version = 0x02020102;
    intr = 0;
    intr_mask = 0;
}

u32 MI::read_reg(u32 addr) const {
    u32 reg = (addr & 0xF) >> 2;
    switch (reg) {
        case 0: return mode;
        case 1: return version;
        case 2: return intr;
        case 3: return intr_mask;
        default: return 0;
    }
}

void MI::write_reg(u32 addr, u32 val) {
    u32 reg = (addr & 0xF) >> 2;
    switch (reg) {
        case 0: { // MI_MODE_REG
            mode = (mode & ~0x7F) | (val & 0x7F);
            if (val & (1 << 7))  mode &= ~(1 << 7);  // Clr init mode
            if (val & (1 << 8))  mode |= (1 << 7);   // Set init mode
            if (val & (1 << 9))  mode &= ~(1 << 8);  // Clr ebus test mode
            if (val & (1 << 10)) mode |= (1 << 8);   // Set ebus test mode
            if (val & (1 << 11)) clear_interrupt(MIInterrupt::DP); // Clr DP interrupt
            if (val & (1 << 12)) mode &= ~(1 << 9);  // Clr RDRAM reg mode
            if (val & (1 << 13)) mode |= (1 << 9);   // Set RDRAM reg mode
            break;
        }
        case 1: // MI_VERSION_REG (read only)
            break;
        case 2: // MI_INTR_REG (read only)
            break;
        case 3: { // MI_INTR_MASK_REG
            if (val & (1 << 0)) intr_mask &= ~MIInterrupt::SP;
            if (val & (1 << 1)) intr_mask |= MIInterrupt::SP;
            if (val & (1 << 2)) intr_mask &= ~MIInterrupt::SI;
            if (val & (1 << 3)) intr_mask |= MIInterrupt::SI;
            if (val & (1 << 4)) intr_mask &= ~MIInterrupt::AI;
            if (val & (1 << 5)) intr_mask |= MIInterrupt::AI;
            if (val & (1 << 6)) intr_mask &= ~MIInterrupt::VI;
            if (val & (1 << 7)) intr_mask |= MIInterrupt::VI;
            if (val & (1 << 8)) intr_mask &= ~MIInterrupt::PI;
            if (val & (1 << 9)) intr_mask |= MIInterrupt::PI;
            if (val & (1 << 10)) intr_mask &= ~MIInterrupt::DP;
            if (val & (1 << 11)) intr_mask |= MIInterrupt::DP;
            break;
        }
    }
}

void MI::raise_interrupt(u32 intr_bit) {
    intr |= intr_bit;
}

void MI::clear_interrupt(u32 intr_bit) {
    intr &= ~intr_bit;
}
