#include "cpu.hpp"
#include "bus.hpp"
#include "mi.hpp"
#include <iostream>
#include <cmath>

CPU::CPU(Bus& bus) : bus(bus) {
    reset();
}

void CPU::reset(u32 entry_point) {
    std::fill(gpr.begin(), gpr.end(), 0);
    std::fill(cp0.begin(), cp0.end(), 0);
    std::fill(fpr.begin(), fpr.end(), 0);

    pc = entry_point;
    cur_pc = entry_point;
    in_delay_slot = false;
    delay_slot_active = false;
    branch_target = 0;
    pc_overridden = false;
    hi = 0;
    lo = 0;
    fcsr = 0;

    // Standard CP0 state after IPL3 boot
    cp0[CP0Reg::CONFIG] = 0x0006E463;
    cp0[CP0Reg::STATUS] = 0x34000000; // CU1=1, CU0=1, FR=1
    cp0[CP0Reg::PRID]   = 0x00000B22; // VR4300 revision
    cp0[CP0Reg::RANDOM] = 31;

    // Standard initial registers
    gpr[20] = 0x00000000; // s4
    gpr[22] = 0x0000003F; // s6 (seed)
    gpr[29] = 0x8033B400; // sp
    gpr[31] = entry_point; // ra
}

void CPU::trigger_exception(u32 exc_code, u64 vector) {
    if (vector == 0x80000180) {
        bool bev = (cp0[CP0Reg::STATUS] & (1 << 22)) != 0;
        vector = bev ? 0xBFC00380ULL : 0x80000180ULL;
    }

    if (!(cp0[CP0Reg::STATUS] & 2)) { // If EXL is not already set
        if (delay_slot_active) {
            cp0[CP0Reg::EPC] = cur_pc - 4;
            cp0[CP0Reg::CAUSE] |= (1ULL << 31); // BD bit
        } else {
            cp0[CP0Reg::EPC] = cur_pc;
            cp0[CP0Reg::CAUSE] &= ~(1ULL << 31);
        }
        cp0[CP0Reg::STATUS] |= 2; // Set EXL
    }

    cp0[CP0Reg::CAUSE] = (cp0[CP0Reg::CAUSE] & ~0x7C) | ((exc_code & 0x1F) << 2);

    in_delay_slot = false;
    delay_slot_active = false;
    pc = vector;
    pc_overridden = true;
}

void CPU::trigger_tlb_exception(u64 fault_vaddr, TLBResult result, bool is_write) {
    u64 status = cp0[CP0Reg::STATUS];
    bool bev = (status & (1 << 22)) != 0;
    bool exl = (status & 2) != 0;

    cp0[CP0Reg::BAD_VADDR] = fault_vaddr;
    cp0[CP0Reg::ENTRY_HI] = (cp0[CP0Reg::ENTRY_HI] & 0xFFULL) | (fault_vaddr & 0xFFFFE000ULL);
    cp0[CP0Reg::CONTEXT] = (cp0[CP0Reg::CONTEXT] & 0xFFFFFFFFFF800000ULL) | (((fault_vaddr >> 13) & 0x7FFFFULL) << 4);

    u32 exc_code = 2; // EXC_TLBL
    if (result == TLBResult::MODIFIED) {
        exc_code = 1; // EXC_MOD
    } else if (is_write) {
        exc_code = 3; // EXC_TLBS
    }

    u64 vector;
    bool is_kuseg = (fault_vaddr < 0x80000000ULL);
    if (result == TLBResult::MISS && !exl && is_kuseg) {
        vector = bev ? 0xBFC00200ULL : 0x80000000ULL;
    } else {
        vector = bev ? 0xBFC00380ULL : 0x80000180ULL;
    }


    if (!exl) {
        if (delay_slot_active) {
            cp0[CP0Reg::EPC] = cur_pc - 4;
            cp0[CP0Reg::CAUSE] |= (1ULL << 31);
        } else {
            cp0[CP0Reg::EPC] = cur_pc;
            cp0[CP0Reg::CAUSE] &= ~(1ULL << 31);
        }
        cp0[CP0Reg::STATUS] |= 2;
    }

    cp0[CP0Reg::CAUSE] = (cp0[CP0Reg::CAUSE] & ~0x7C) | ((exc_code & 0x1F) << 2);

    in_delay_slot = false;
    delay_slot_active = false;
    pc = vector;
    pc_overridden = true;
}

void CPU::trigger_interrupt() {
    u64 vector = 0x80000180ULL;
    bool bev = (cp0[CP0Reg::STATUS] & (1 << 22)) != 0;
    if (bev) vector = 0xBFC00380ULL;

    if (!(cp0[CP0Reg::STATUS] & 2)) { // If EXL is not already set
        // For external interrupts between instructions, the interrupted instruction
        // is the instruction about to be executed next (pc)
        cp0[CP0Reg::EPC] = pc;
        cp0[CP0Reg::CAUSE] &= ~(1ULL << 31); // BD bit = 0
        cp0[CP0Reg::STATUS] |= 2; // Set EXL
    }

    cp0[CP0Reg::CAUSE] = (cp0[CP0Reg::CAUSE] & ~0x7C); // ExcCode = 0 (Interrupt)

    in_delay_slot = false;
    delay_slot_active = false;
    pc = vector;
    pc_overridden = true;
}

void CPU::step_timer(u32 cycles) {
    u32 prev_count = static_cast<u32>(cp0[CP0Reg::COUNT]);
    cp0[CP0Reg::COUNT] = (prev_count + (cycles / 2)) & 0xFFFFFFFF;

    u32 cur_count = static_cast<u32>(cp0[CP0Reg::COUNT]);
    u32 compare = static_cast<u32>(cp0[CP0Reg::COMPARE]);

    // Check if compare was reached or passed in this interval
    u32 delta = cur_count - prev_count;
    if ((cur_count - compare) < delta) {
        if (!(cp0[CP0Reg::CAUSE] & (1 << 15))) {
            cp0[CP0Reg::CAUSE] |= (1 << 15); // IP7 (Timer interrupt)
        }
    }
}

void CPU::check_interrupts() {
    // An interrupt is never taken between a branch and its delay slot
    if (in_delay_slot) {
        return;
    }

    if (bus.get_mi().is_interrupt_asserted()) {
        cp0[CP0Reg::CAUSE] |= (1 << 10); // IP2 (MI external interrupt)
    } else {
        cp0[CP0Reg::CAUSE] &= ~(1 << 10);
    }

    u64 status = cp0[CP0Reg::STATUS];
    u64 cause = cp0[CP0Reg::CAUSE];

    // IE=1, EXL=0, ERL=0
    if ((status & 1) && !(status & 2) && !(status & 4)) {
        u8 im = (status >> 8) & 0xFF;
        u8 ip = (cause >> 8) & 0xFF;
        if (im & ip) {
            trigger_interrupt();
        }
    }
}

u32 CPU::step() {
    gpr[0] = 0;

    // VR4300 Random register decrements each instruction down to Wired, then wraps to 31
    u64 wired = cp0[CP0Reg::WIRED] & 0x1F;
    u64 random = cp0[CP0Reg::RANDOM] & 0x1F;
    if (random <= wired) {
        cp0[CP0Reg::RANDOM] = 31;
    } else {
        cp0[CP0Reg::RANDOM] = random - 1;
    }

    cur_pc = pc;
    // Check if the instruction we are about to execute is in a branch delay slot
    delay_slot_active = in_delay_slot;
    u64 next_pc = delay_slot_active ? branch_target : (cur_pc + 4);
    in_delay_slot = false;
    pc_overridden = false;

    TLBResult fetch_res = TLBResult::SUCCESS;
    u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
    u32 instr = bus.read_v32(pc, fetch_res, asid);
    if (fetch_res != TLBResult::SUCCESS) {
        trigger_tlb_exception(pc, fetch_res, false);
        return 1;
    }

    execute(instr, cur_pc);
    gpr[0] = 0;

    if (in_delay_slot) {
        pc = cur_pc + 4;
    } else if (!pc_overridden) {
        pc = next_pc;
    }

    step_timer(2);
    check_interrupts();

    return 2;
}

void CPU::execute(u32 instr, u64 cur_pc) {
    u8 op = (instr >> 26) & 0x3F;
    u8 rs = (instr >> 21) & 0x1F;
    u8 rt = (instr >> 16) & 0x1F;
    u8 rd = (instr >> 11) & 0x1F;
    u8 shamt = (instr >> 6) & 0x1F;
    u8 funct = instr & 0x3F;

    u16 imm = instr & 0xFFFF;
    s64 simm = sign_extend_16_64(static_cast<s16>(imm));
    u32 target = (instr & 0x03FFFFFF) << 2;

    switch (op) {
        case 0x00: { // SPECIAL
            switch (funct) {
                case 0x00: gpr[rd] = sign_extend_32_64(static_cast<s32>(gpr[rt] << shamt)); break; // SLL
                case 0x02: gpr[rd] = sign_extend_32_64(static_cast<s32>(static_cast<u32>(gpr[rt]) >> shamt)); break; // SRL
                case 0x03: gpr[rd] = sign_extend_32_64(static_cast<s32>(gpr[rt]) >> shamt); break; // SRA
                case 0x04: gpr[rd] = sign_extend_32_64(static_cast<s32>(gpr[rt] << (gpr[rs] & 0x1F))); break; // SLLV
                case 0x06: gpr[rd] = sign_extend_32_64(static_cast<s32>(static_cast<u32>(gpr[rt]) >> (gpr[rs] & 0x1F))); break; // SRLV
                case 0x07: gpr[rd] = sign_extend_32_64(static_cast<s32>(gpr[rt]) >> (gpr[rs] & 0x1F)); break; // SRAV
                case 0x08: { // JR
                    in_delay_slot = true;
                    branch_target = gpr[rs];
                    break;
                }
                case 0x09: { // JALR
                    gpr[rd] = cur_pc + 8;
                    in_delay_slot = true;
                    branch_target = gpr[rs];
                    break;
                }
                case 0x0A: if (gpr[rt] == 0) gpr[rd] = gpr[rs]; break; // MOVZ
                case 0x0B: if (gpr[rt] != 0) gpr[rd] = gpr[rs]; break; // MOVN
                case 0x0C: trigger_exception(8); break; // SYSCALL
                case 0x0D: trigger_exception(9); break; // BREAK
                case 0x0F: break; // SYNC
                case 0x10: gpr[rd] = hi; break; // MFHI
                case 0x11: hi = gpr[rs]; break; // MTHI
                case 0x12: gpr[rd] = lo; break; // MFLO
                case 0x13: lo = gpr[rs]; break; // MTLO
                case 0x14: gpr[rd] = gpr[rt] << (gpr[rs] & 0x3F); break; // DSLLV
                case 0x16: gpr[rd] = gpr[rt] >> (gpr[rs] & 0x3F); break; // DSRLV
                case 0x17: gpr[rd] = static_cast<s64>(gpr[rt]) >> (gpr[rs] & 0x3F); break; // DSRAV
                case 0x18: { // MULT
                    s64 res = static_cast<s64>(static_cast<s32>(gpr[rs])) * static_cast<s64>(static_cast<s32>(gpr[rt]));
                    lo = sign_extend_32_64(static_cast<s32>(res));
                    hi = sign_extend_32_64(static_cast<s32>(res >> 32));
                    break;
                }
                case 0x19: { // MULTU
                    u64 res = static_cast<u64>(static_cast<u32>(gpr[rs])) * static_cast<u64>(static_cast<u32>(gpr[rt]));
                    lo = sign_extend_32_64(static_cast<s32>(res));
                    hi = sign_extend_32_64(static_cast<s32>(res >> 32));
                    break;
                }
                case 0x1A: { // DIV
                    s32 num = static_cast<s32>(gpr[rs]);
                    s32 den = static_cast<s32>(gpr[rt]);
                    if (den != 0) {
                        // INT32_MIN / -1 overflows; hardware truncates to INT32_MIN and rem 0
                        if (num == INT32_MIN && den == -1) {
                            lo = sign_extend_32_64(INT32_MIN);
                            hi = 0;
                        } else {
                            lo = sign_extend_32_64(num / den);
                            hi = sign_extend_32_64(num % den);
                        }
                    } else {
                        // Division by zero: hardware-documented (undefined by ISA, fixed on VR4300)
                        lo = sign_extend_32_64(num >= 0 ? -1 : 1);
                        hi = sign_extend_32_64(num);
                    }
                    break;
                }
                case 0x1B: { // DIVU
                    u32 num = static_cast<u32>(gpr[rs]);
                    u32 den = static_cast<u32>(gpr[rt]);
                    if (den != 0) {
                        lo = sign_extend_32_64(static_cast<s32>(num / den));
                        hi = sign_extend_32_64(static_cast<s32>(num % den));
                    } else {
                        lo = sign_extend_32_64(-1);
                        hi = sign_extend_32_64(static_cast<s32>(num));
                    }
                    break;
                }
                case 0x1C: { // DMULT
                    __int128 res = static_cast<__int128>(static_cast<s64>(gpr[rs])) * static_cast<__int128>(static_cast<s64>(gpr[rt]));
                    lo = static_cast<u64>(res);
                    hi = static_cast<u64>(res >> 64);
                    break;
                }
                case 0x1D: { // DMULTU
                    unsigned __int128 res = static_cast<unsigned __int128>(gpr[rs]) * static_cast<unsigned __int128>(gpr[rt]);
                    lo = static_cast<u64>(res);
                    hi = static_cast<u64>(res >> 64);
                    break;
                }
                case 0x1E: { // DDIV
                    s64 num = static_cast<s64>(gpr[rs]);
                    s64 den = static_cast<s64>(gpr[rt]);
                    if (den != 0) {
                        if (num == INT64_MIN && den == -1) {
                            lo = static_cast<u64>(INT64_MIN);
                            hi = 0;
                        } else {
                            lo = num / den;
                            hi = num % den;
                        }
                    } else {
                        lo = static_cast<u64>(num >= 0 ? -1 : 1);
                        hi = static_cast<u64>(num);
                    }
                    break;
                }
                case 0x1F: { // DDIVU
                    u64 num = gpr[rs];
                    u64 den = gpr[rt];
                    if (den != 0) {
                        lo = num / den;
                        hi = num % den;
                    } else {
                        lo = static_cast<u64>(-1);
                        hi = num;
                    }
                    break;
                }
                case 0x20: { // ADD (traps on 32-bit signed overflow)
                    s32 a = static_cast<s32>(gpr[rs]);
                    s32 b = static_cast<s32>(gpr[rt]);
                    s32 res;
                    if (__builtin_add_overflow(a, b, &res)) {
                        trigger_exception(12); // EXC_OV
                    } else {
                        gpr[rd] = sign_extend_32_64(res);
                    }
                    break;
                }
                case 0x21: gpr[rd] = sign_extend_32_64(static_cast<s32>(gpr[rs] + gpr[rt])); break; // ADDU
                case 0x22: { // SUB (traps on 32-bit signed overflow)
                    s32 a = static_cast<s32>(gpr[rs]);
                    s32 b = static_cast<s32>(gpr[rt]);
                    s32 res;
                    if (__builtin_sub_overflow(a, b, &res)) {
                        trigger_exception(12); // EXC_OV
                    } else {
                        gpr[rd] = sign_extend_32_64(res);
                    }
                    break;
                }
                case 0x23: gpr[rd] = sign_extend_32_64(static_cast<s32>(gpr[rs] - gpr[rt])); break; // SUBU
                case 0x24: gpr[rd] = gpr[rs] & gpr[rt]; break; // AND
                case 0x25: gpr[rd] = gpr[rs] | gpr[rt]; break; // OR
                case 0x26: gpr[rd] = gpr[rs] ^ gpr[rt]; break; // XOR
                case 0x27: gpr[rd] = ~(gpr[rs] | gpr[rt]); break; // NOR
                case 0x2A: gpr[rd] = (static_cast<s64>(gpr[rs]) < static_cast<s64>(gpr[rt])) ? 1 : 0; break; // SLT
                case 0x2B: gpr[rd] = (gpr[rs] < gpr[rt]) ? 1 : 0; break; // SLTU
                case 0x2C: { // DADD (traps on 64-bit signed overflow)
                    s64 a = static_cast<s64>(gpr[rs]);
                    s64 b = static_cast<s64>(gpr[rt]);
                    s64 res;
                    if (__builtin_add_overflow(a, b, &res)) {
                        trigger_exception(12); // EXC_OV
                    } else {
                        gpr[rd] = static_cast<u64>(res);
                    }
                    break;
                }
                case 0x2D: gpr[rd] = gpr[rs] + gpr[rt]; break; // DADDU
                case 0x2E: { // DSUB (traps on 64-bit signed overflow)
                    s64 a = static_cast<s64>(gpr[rs]);
                    s64 b = static_cast<s64>(gpr[rt]);
                    s64 res;
                    if (__builtin_sub_overflow(a, b, &res)) {
                        trigger_exception(12); // EXC_OV
                    } else {
                        gpr[rd] = static_cast<u64>(res);
                    }
                    break;
                }
                case 0x2F: gpr[rd] = gpr[rs] - gpr[rt]; break; // DSUBU
                case 0x30: if (static_cast<s64>(gpr[rs]) >= static_cast<s64>(gpr[rt])) trigger_exception(13); break; // TGE
                case 0x31: if (gpr[rs] >= gpr[rt]) trigger_exception(13); break; // TGEU
                case 0x32: if (static_cast<s64>(gpr[rs]) < static_cast<s64>(gpr[rt])) trigger_exception(13); break; // TLT
                case 0x33: if (gpr[rs] < gpr[rt]) trigger_exception(13); break; // TLTU
                case 0x34: if (gpr[rs] == gpr[rt]) trigger_exception(13); break; // TEQ
                case 0x36: if (gpr[rs] != gpr[rt]) trigger_exception(13); break; // TNE
                case 0x38: gpr[rd] = gpr[rt] << shamt; break; // DSLL
                case 0x3A: gpr[rd] = gpr[rt] >> shamt; break; // DSRL
                case 0x3B: gpr[rd] = static_cast<s64>(gpr[rt]) >> shamt; break; // DSRA
                case 0x3C: gpr[rd] = gpr[rt] << (shamt + 32); break; // DSLL32
                case 0x3E: gpr[rd] = gpr[rt] >> (shamt + 32); break; // DSRL32
                case 0x3F: gpr[rd] = static_cast<s64>(gpr[rt]) >> (shamt + 32); break; // DSRA32
                default: trigger_exception(10); break; // Reserved Instruction
            }
            break;
        }

        case 0x01: { // REGIMM
            switch (rt) {
                case 0x08: if (static_cast<s64>(gpr[rs]) >= simm) trigger_exception(13); break; // TGEI
                case 0x09: if (gpr[rs] >= static_cast<u64>(simm)) trigger_exception(13); break; // TGEIU
                case 0x0A: if (static_cast<s64>(gpr[rs]) < simm) trigger_exception(13); break; // TLTI
                case 0x0B: if (gpr[rs] < static_cast<u64>(simm)) trigger_exception(13); break; // TLTIU
                case 0x0C: if (static_cast<s64>(gpr[rs]) == simm) trigger_exception(13); break; // TEQI
                case 0x0E: if (static_cast<s64>(gpr[rs]) != simm) trigger_exception(13); break; // TNEI
                case 0x00: { // BLTZ
                    if (static_cast<s64>(gpr[rs]) < 0) {
                        in_delay_slot = true;
                        branch_target = cur_pc + 4 + (simm << 2);
                    }
                    break;
                }
                case 0x01: { // BGEZ
                    if (static_cast<s64>(gpr[rs]) >= 0) {
                        in_delay_slot = true;
                        branch_target = cur_pc + 4 + (simm << 2);
                    }
                    break;
                }
                case 0x02: { // BLTZL
                    if (static_cast<s64>(gpr[rs]) < 0) {
                        in_delay_slot = true;
                        branch_target = cur_pc + 4 + (simm << 2);
                    } else {
                        pc = cur_pc + 8;
                        pc_overridden = true;
                    }
                    break;
                }
                case 0x03: { // BGEZL
                    if (static_cast<s64>(gpr[rs]) >= 0) {
                        in_delay_slot = true;
                        branch_target = cur_pc + 4 + (simm << 2);
                    } else {
                        pc = cur_pc + 8;
                        pc_overridden = true;
                    }
                    break;
                }
                case 0x10: { // BLTZAL
                    gpr[31] = cur_pc + 8;
                    if (static_cast<s64>(gpr[rs]) < 0) {
                        in_delay_slot = true;
                        branch_target = cur_pc + 4 + (simm << 2);
                    }
                    break;
                }
                case 0x11: { // BGEZAL
                    gpr[31] = cur_pc + 8;
                    if (static_cast<s64>(gpr[rs]) >= 0) {
                        in_delay_slot = true;
                        branch_target = cur_pc + 4 + (simm << 2);
                    }
                    break;
                }
                case 0x12: { // BLTZALL
                    gpr[31] = cur_pc + 8;
                    if (static_cast<s64>(gpr[rs]) < 0) {
                        in_delay_slot = true;
                        branch_target = cur_pc + 4 + (simm << 2);
                    } else {
                        pc = cur_pc + 8;
                        pc_overridden = true;
                    }
                    break;
                }
                case 0x13: { // BGEZALL
                    gpr[31] = cur_pc + 8;
                    if (static_cast<s64>(gpr[rs]) >= 0) {
                        in_delay_slot = true;
                        branch_target = cur_pc + 4 + (simm << 2);
                    } else {
                        pc = cur_pc + 8;
                        pc_overridden = true;
                    }
                    break;
                }
                default: trigger_exception(10); break; // Reserved Instruction
            }
            break;
        }

        case 0x02: { // J
            in_delay_slot = true;
            branch_target = (cur_pc & 0xFFFFFFFF00000000ULL) | ((cur_pc & 0xF0000000ULL) | target);
            break;
        }

        case 0x03: { // JAL
            gpr[31] = cur_pc + 8;
            in_delay_slot = true;
            branch_target = (cur_pc & 0xFFFFFFFF00000000ULL) | ((cur_pc & 0xF0000000ULL) | target);
            break;
        }

        case 0x04: { // BEQ
            if (gpr[rs] == gpr[rt]) {
                in_delay_slot = true;
                branch_target = cur_pc + 4 + (simm << 2);
            }
            break;
        }

        case 0x05: { // BNE
            if (gpr[rs] != gpr[rt]) {
                in_delay_slot = true;
                branch_target = cur_pc + 4 + (simm << 2);
            }
            break;
        }

        case 0x06: { // BLEZ
            if (static_cast<s64>(gpr[rs]) <= 0) {
                in_delay_slot = true;
                branch_target = cur_pc + 4 + (simm << 2);
            }
            break;
        }

        case 0x07: { // BGTZ
            if (static_cast<s64>(gpr[rs]) > 0) {
                in_delay_slot = true;
                branch_target = cur_pc + 4 + (simm << 2);
            }
            break;
        }

        case 0x08: { // ADDI (traps on 32-bit signed overflow)
            s32 a = static_cast<s32>(gpr[rs]);
            s32 b = static_cast<s32>(simm);
            s32 res;
            if (__builtin_add_overflow(a, b, &res)) {
                trigger_exception(12); // EXC_OV
            } else {
                gpr[rt] = sign_extend_32_64(res);
            }
            break;
        }
        case 0x09: gpr[rt] = sign_extend_32_64(static_cast<s32>(gpr[rs] + simm)); break; // ADDIU
        case 0x0A: gpr[rt] = (static_cast<s64>(gpr[rs]) < simm) ? 1 : 0; break; // SLTI
        case 0x0B: gpr[rt] = (gpr[rs] < static_cast<u64>(simm)) ? 1 : 0; break; // SLTIU
        case 0x0C: gpr[rt] = gpr[rs] & imm; break; // ANDI
        case 0x0D: gpr[rt] = gpr[rs] | imm; break; // ORI
        case 0x0E: gpr[rt] = gpr[rs] ^ imm; break; // XORI
        case 0x0F: gpr[rt] = sign_extend_32_64(static_cast<s32>(imm << 16)); break; // LUI

        case 0x10: { // COP0
            if (rs == 0x00) { // MFC0
                gpr[rt] = sign_extend_32_64(static_cast<s32>(cp0[rd]));
            } else if (rs == 0x01) { // DMFC0
                gpr[rt] = cp0[rd];
            } else if (rs == 0x04) { // MTC0
                cp0[rd] = static_cast<u32>(gpr[rt]);
                if (rd == CP0Reg::COMPARE) {
                    cp0[CP0Reg::CAUSE] &= ~(1 << 15); // Clear IP7
                }
                if (rd == CP0Reg::WIRED) {
                    cp0[CP0Reg::RANDOM] = 31;
                }
            } else if (rs == 0x05) { // DMTC0
                cp0[rd] = gpr[rt];
                if (rd == CP0Reg::WIRED) {
                    cp0[CP0Reg::RANDOM] = 31;
                }
            } else if (rs == 0x10) { // CO / TLB / ERET
                if (funct == 0x18) { // ERET
                    if (cp0[CP0Reg::STATUS] & 4) { // ERL
                        cp0[CP0Reg::STATUS] &= ~4;
                        pc = cp0[CP0Reg::ERROR_EPC];
                    } else {
                        cp0[CP0Reg::STATUS] &= ~2; // Clear EXL
                        pc = cp0[CP0Reg::EPC];
                    }
                    in_delay_slot = false;
                    delay_slot_active = false;
                    pc_overridden = true;
                } else {
                    execute_tlb(funct);
                }
            } else {
                trigger_exception(10); // Reserved Instruction (no CFC0/CTC0/BC0 on VR4300)
            }
            break;
        }

        case 0x11: { // COP1 (FPU)
            if (rs == 0x00) { // MFC1
                gpr[rt] = sign_extend_32_64(static_cast<s32>(get_fpr32(rd)));
            } else if (rs == 0x01) { // DMFC1
                gpr[rt] = get_fpr64(rd);
            } else if (rs == 0x02) { // CFC1
                gpr[rt] = sign_extend_32_64(static_cast<s32>(fcsr));
            } else if (rs == 0x04) { // MTC1
                set_fpr32(rd, static_cast<u32>(gpr[rt]));
            } else if (rs == 0x05) { // DMTC1
                set_fpr64(rd, gpr[rt]);
            } else if (rs == 0x06) { // CTC1
                fcsr = static_cast<u32>(gpr[rt]);
            } else if (rs == 0x08) { // BC1
                bool cond = (fcsr & (1 << 23)) != 0;
                bool is_true = (rt & 1) != 0;
                bool is_likely = (rt & 2) != 0;
                bool taken = (cond == is_true);

                if (is_likely) {
                    if (taken) {
                        in_delay_slot = true;
                        branch_target = cur_pc + 4 + (simm << 2);
                    } else {
                        pc = cur_pc + 8;
                        pc_overridden = true;
                    }
                } else {
                    if (taken) {
                        in_delay_slot = true;
                        branch_target = cur_pc + 4 + (simm << 2);
                    }
                }
            } else if (rs == 0x10 || rs == 0x11 || rs == 0x14 || rs == 0x15) { // S / D / W / L
                execute_fpu_op(instr);
            } else {
                trigger_exception(10); // Reserved Instruction
            }
            break;
        }

        case 0x14: { // BEQL
            if (gpr[rs] == gpr[rt]) {
                in_delay_slot = true;
                branch_target = cur_pc + 4 + (simm << 2);
            } else {
                pc = cur_pc + 8;
                pc_overridden = true;
            }
            break;
        }
        case 0x15: { // BNEL
            if (gpr[rs] != gpr[rt]) {
                in_delay_slot = true;
                branch_target = cur_pc + 4 + (simm << 2);
            } else {
                pc = cur_pc + 8;
                pc_overridden = true;
            }
            break;
        }
        case 0x16: { // BLEZL
            if (static_cast<s64>(gpr[rs]) <= 0) {
                in_delay_slot = true;
                branch_target = cur_pc + 4 + (simm << 2);
            } else {
                pc = cur_pc + 8;
                pc_overridden = true;
            }
            break;
        }
        case 0x17: { // BGTZL
            if (static_cast<s64>(gpr[rs]) > 0) {
                in_delay_slot = true;
                branch_target = cur_pc + 4 + (simm << 2);
            } else {
                pc = cur_pc + 8;
                pc_overridden = true;
            }
            break;
        }

        case 0x18: { // DADDI (traps on 64-bit signed overflow)
            s64 a = static_cast<s64>(gpr[rs]);
            s64 res;
            if (__builtin_add_overflow(a, simm, &res)) {
                trigger_exception(12); // EXC_OV
            } else {
                gpr[rt] = static_cast<u64>(res);
            }
            break;
        }
        case 0x19: gpr[rt] = gpr[rs] + simm; break; // DADDIU

        case 0x1A: { // LDL
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u64 dword = bus.read_v64(addr & ~7, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~7, res, false);
                break;
            }
            u32 shift = (addr & 7) * 8;
            u64 mask = 0xFFFFFFFFFFFFFFFFULL << shift;
            gpr[rt] = (gpr[rt] & ~mask) | (dword << shift);
            break;
        }
        case 0x1B: { // LDR
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u64 dword = bus.read_v64(addr & ~7, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~7, res, false);
                break;
            }
            u32 shift = (7 - (addr & 7)) * 8;
            u64 mask = 0xFFFFFFFFFFFFFFFFULL >> shift;
            gpr[rt] = (gpr[rt] & ~mask) | (dword >> shift);
            break;
        }

        case 0x20: { // LB
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u8 byte = bus.read_v8(addr, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, false);
                break;
            }
            gpr[rt] = sign_extend_8_64(static_cast<s8>(byte));
            break;
        }
        case 0x21: { // LH
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u16 hword = bus.read_v16(addr, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, false);
                break;
            }
            gpr[rt] = sign_extend_16_64(static_cast<s16>(hword));
            break;
        }
        case 0x22: { // LWL
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u32 word = bus.read_v32(addr & ~3, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~3, res, false);
                break;
            }
            u32 shift = (addr & 3) * 8;
            u32 mask = 0xFFFFFFFFU << shift;
            gpr[rt] = sign_extend_32_64(static_cast<s32>((static_cast<u32>(gpr[rt]) & ~mask) | (word << shift)));
            break;
        }
        case 0x23: { // LW
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u32 word = bus.read_v32(addr, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, false);
                break;
            }
            gpr[rt] = sign_extend_32_64(static_cast<s32>(word));
            break;
        }
        case 0x24: { // LBU
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u8 byte = bus.read_v8(addr, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, false);
                break;
            }
            gpr[rt] = byte;
            break;
        }
        case 0x25: { // LHU
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u16 hword = bus.read_v16(addr, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, false);
                break;
            }
            gpr[rt] = hword;
            break;
        }
        case 0x26: { // LWR
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u32 word = bus.read_v32(addr & ~3, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~3, res, false);
                break;
            }
            u32 shift = (3 - (addr & 3)) * 8;
            u32 mask = 0xFFFFFFFFU >> shift;
            gpr[rt] = sign_extend_32_64(static_cast<s32>((static_cast<u32>(gpr[rt]) & ~mask) | (word >> shift)));
            break;
        }
        case 0x27: { // LWU
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u32 word = bus.read_v32(addr, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, false);
                break;
            }
            gpr[rt] = word;
            break;
        }

        case 0x28: { // SB
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            bus.write_v8(addr, static_cast<u8>(gpr[rt]), res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, true);
                break;
            }
            break;
        }
        case 0x29: { // SH
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            bus.write_v16(addr, static_cast<u16>(gpr[rt]), res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, true);
                break;
            }
            break;
        }
        case 0x2A: { // SWL
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u32 orig = bus.read_v32(addr & ~3, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~3, res, false);
                break;
            }
            u32 shift = (addr & 3) * 8;
            u32 mask = 0xFFFFFFFFU >> shift;
            bus.write_v32(addr & ~3, (orig & ~mask) | (static_cast<u32>(gpr[rt]) >> shift), res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~3, res, true);
                break;
            }
            break;
        }
        case 0x2B: { // SW
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            bus.write_v32(addr, static_cast<u32>(gpr[rt]), res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, true);
                break;
            }
            break;
        }
        case 0x2C: { // SDL
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u64 orig = bus.read_v64(addr & ~7, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~7, res, false);
                break;
            }
            u32 shift = (addr & 7) * 8;
            u64 mask = 0xFFFFFFFFFFFFFFFFULL >> shift;
            bus.write_v64(addr & ~7, (orig & ~mask) | (gpr[rt] >> shift), res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~7, res, true);
                break;
            }
            break;
        }
        case 0x2D: { // SDR
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u64 orig = bus.read_v64(addr & ~7, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~7, res, false);
                break;
            }
            u32 shift = (7 - (addr & 7)) * 8;
            u64 mask = 0xFFFFFFFFFFFFFFFFULL << shift;
            bus.write_v64(addr & ~7, (orig & ~mask) | (gpr[rt] << shift), res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~7, res, true);
                break;
            }
            break;
        }
        case 0x2E: { // SWR
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u32 orig = bus.read_v32(addr & ~3, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~3, res, false);
                break;
            }
            u32 shift = (3 - (addr & 3)) * 8;
            u32 mask = 0xFFFFFFFFU << shift;
            bus.write_v32(addr & ~3, (orig & ~mask) | (static_cast<u32>(gpr[rt]) << shift), res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr & ~3, res, true);
                break;
            }
            break;
        }

        case 0x2F: break; // CACHE (nop)

        case 0x30: { // LL
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u32 word = bus.read_v32(addr, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, false);
                break;
            }
            cp0[CP0Reg::LLADDR] = addr;
            gpr[rt] = sign_extend_32_64(static_cast<s32>(word));
            break;
        }
        case 0x31: { // LWC1
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u32 word = bus.read_v32(addr, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, false);
                break;
            }
            set_fpr32(rt, word);
            break;
        }
        case 0x34: { // LLD
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u64 dword = bus.read_v64(addr, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, false);
                break;
            }
            cp0[CP0Reg::LLADDR] = addr;
            gpr[rt] = dword;
            break;
        }
        case 0x35: { // LDC1
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u64 dword = bus.read_v64(addr, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, false);
                break;
            }
            set_fpr64(rt, dword);
            break;
        }
        case 0x37: { // LD
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            u64 dword = bus.read_v64(addr, res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, false);
                break;
            }
            gpr[rt] = dword;
            break;
        }
        case 0x38: { // SC
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            bus.write_v32(addr, static_cast<u32>(gpr[rt]), res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, true);
                break;
            }
            gpr[rt] = 1;
            break;
        }
        case 0x39: { // SWC1
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            bus.write_v32(addr, get_fpr32(rt), res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, true);
                break;
            }
            break;
        }
        case 0x3C: { // SCD
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            bus.write_v64(addr, gpr[rt], res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, true);
                break;
            }
            gpr[rt] = 1;
            break;
        }
        case 0x3D: { // SDC1
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            bus.write_v64(addr, get_fpr64(rt), res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, true);
                break;
            }
            break;
        }
        case 0x3F: { // SD
            u64 addr = gpr[rs] + simm;
            u8 asid = cp0[CP0Reg::ENTRY_HI] & 0xFF;
            TLBResult res = TLBResult::SUCCESS;
            bus.write_v64(addr, gpr[rt], res, asid);
            if (res != TLBResult::SUCCESS) {
                trigger_tlb_exception(addr, res, true);
                break;
            }
            break;
        }

        default: trigger_exception(10); break; // Reserved Instruction (COP2, LWC2/SWC2/LDC2/SDC2, etc.)
    }
}

u32 CPU::get_fpr32(size_t reg) const {
    return static_cast<u32>(fpr[reg & 31]);
}

void CPU::set_fpr32(size_t reg, u32 val) {
    fpr[reg & 31] = val;
}

u64 CPU::get_fpr64(size_t reg) const {
    bool fr = (cp0[CP0Reg::STATUS] & (1 << 26)) != 0;
    if (fr) {
        return fpr[reg & 31];
    } else {
        // In 32-bit FPU mode (FR=0):
        // Odd register holds upper 32 bits (bits 63..32)
        // Even register holds lower 32 bits (bits 31..0)
        u64 hi = static_cast<u32>(fpr[(reg & 31) | 1]);
        u64 lo = static_cast<u32>(fpr[(reg & 31) & ~1]);
        return (hi << 32) | lo;
    }
}

void CPU::set_fpr64(size_t reg, u64 val) {
    bool fr = (cp0[CP0Reg::STATUS] & (1 << 26)) != 0;
    if (fr) {
        fpr[reg & 31] = val;
    } else {
        u32 hi = static_cast<u32>(val >> 32);
        u32 lo = static_cast<u32>(val & 0xFFFFFFFF);
        fpr[(reg & 31) | 1] = hi;
        fpr[(reg & 31) & ~1] = lo;
    }
}

// MIPS C.cond.fmt predicate. The low 3 bits of the condition field select the
// relations that make the compare true (bit0 = unordered, bit1 = equal,
// bit2 = less); bit 3 only picks the signalling variant and does not change the
// predicate. Spelling the cases out one by one is what dropped C.LT/C.LE before.
static inline bool fp_condition(double a, double b, u32 cond) {
    if (std::isnan(a) || std::isnan(b)) return (cond & 1) != 0;
    return (((cond & 4) != 0) && a < b) || (((cond & 2) != 0) && a == b);
}

void CPU::jit_fpu_compare_s(size_t fs, size_t ft, u32 cond) {
    float s_val, t_val;
    u32 fs_raw = get_fpr32(fs), ft_raw = get_fpr32(ft);
    std::memcpy(&s_val, &fs_raw, 4);
    std::memcpy(&t_val, &ft_raw, 4);
    if (fp_condition(s_val, t_val, cond)) fcsr |= (1 << 23);
    else fcsr &= ~(1 << 23);
}

void CPU::jit_fpu_compare_d(size_t fs, size_t ft, u32 cond) {
    double s_val, t_val;
    u64 fs_raw = get_fpr64(fs), ft_raw = get_fpr64(ft);
    std::memcpy(&s_val, &fs_raw, 8);
    std::memcpy(&t_val, &ft_raw, 8);
    if (fp_condition(s_val, t_val, cond)) fcsr |= (1 << 23);
    else fcsr &= ~(1 << 23);
}

void CPU::execute_fpu_op(u32 instr) {
    u8 fmt = (instr >> 21) & 0x1F;
    u8 ft = (instr >> 16) & 0x1F;
    u8 fs = (instr >> 11) & 0x1F;
    u8 fd = (instr >> 6) & 0x1F;
    u8 funct = instr & 0x3F;

    if (fmt == 0x10) { // Single precision (.S)
        float s_val, t_val;
        u32 fs_raw = get_fpr32(fs);
        u32 ft_raw = get_fpr32(ft);
        std::memcpy(&s_val, &fs_raw, 4);
        std::memcpy(&t_val, &ft_raw, 4);
        float d_val = 0.0f;

        switch (funct) {
            case 0x00: d_val = s_val + t_val; break; // ADD.S
            case 0x01: d_val = s_val - t_val; break; // SUB.S
            case 0x02: d_val = s_val * t_val; break; // MUL.S
            case 0x03: d_val = (t_val != 0.0f) ? (s_val / t_val) : 0.0f; break; // DIV.S
            case 0x04: d_val = std::sqrt(s_val); break; // SQRT.S
            case 0x05: d_val = std::abs(s_val); break; // ABS.S
            case 0x06: d_val = s_val; break; // MOV.S
            case 0x07: d_val = -s_val; break; // NEG.S
            case 0x08: { // ROUND.L.S
                s64 i_val = static_cast<s64>(std::round(s_val));
                set_fpr64(fd, static_cast<u64>(i_val));
                return;
            }
            case 0x09: { // TRUNC.L.S
                s64 i_val = static_cast<s64>(std::trunc(s_val));
                set_fpr64(fd, static_cast<u64>(i_val));
                return;
            }
            case 0x0A: { // CEIL.L.S
                s64 i_val = static_cast<s64>(std::ceil(s_val));
                set_fpr64(fd, static_cast<u64>(i_val));
                return;
            }
            case 0x0B: { // FLOOR.L.S
                s64 i_val = static_cast<s64>(std::floor(s_val));
                set_fpr64(fd, static_cast<u64>(i_val));
                return;
            }
            case 0x0C: { // ROUND.W.S
                s32 i_val = static_cast<s32>(std::round(s_val));
                set_fpr32(fd, static_cast<u32>(i_val));
                return;
            }
            case 0x0D: { // TRUNC.W.S
                s32 i_val = static_cast<s32>(std::trunc(s_val));
                set_fpr32(fd, static_cast<u32>(i_val));
                return;
            }
            case 0x0E: { // CEIL.W.S
                s32 i_val = static_cast<s32>(std::ceil(s_val));
                set_fpr32(fd, static_cast<u32>(i_val));
                return;
            }
            case 0x0F: { // FLOOR.W.S
                s32 i_val = static_cast<s32>(std::floor(s_val));
                set_fpr32(fd, static_cast<u32>(i_val));
                return;
            }
            case 0x25: { // CVT.L.S
                s64 i_val = static_cast<s64>(s_val);
                set_fpr64(fd, static_cast<u64>(i_val));
                return;
            }
            case 0x21: { // CVT.D.S
                double dbl = static_cast<double>(s_val);
                u64 raw;
                std::memcpy(&raw, &dbl, 8);
                set_fpr64(fd, raw);
                return;
            }
            case 0x24: { // CVT.W.S
                s32 i_val = static_cast<s32>(s_val);
                set_fpr32(fd, static_cast<u32>(i_val));
                return;
            }
            default:
                if ((funct & 0x30) == 0x30) { // C.cond.S
                    bool c = fp_condition(s_val, t_val, funct & 0x07);
                    if (c) fcsr |= (1 << 23);
                    else fcsr &= ~(1 << 23);
                    return;
                }
                trigger_exception(10); // Reserved Instruction
                return;
        }
        u32 raw;
        std::memcpy(&raw, &d_val, 4);
        set_fpr32(fd, raw);

    } else if (fmt == 0x11) { // Double precision (.D)
        double s_val, t_val;
        u64 fs_raw = get_fpr64(fs);
        u64 ft_raw = get_fpr64(ft);
        std::memcpy(&s_val, &fs_raw, 8);
        std::memcpy(&t_val, &ft_raw, 8);
        double d_val = 0.0;

        switch (funct) {
            case 0x00: d_val = s_val + t_val; break; // ADD.D
            case 0x01: d_val = s_val - t_val; break; // SUB.D
            case 0x02: d_val = s_val * t_val; break; // MUL.D
            case 0x03: d_val = (t_val != 0.0) ? (s_val / t_val) : 0.0; break; // DIV.D
            case 0x04: d_val = std::sqrt(s_val); break; // SQRT.D
            case 0x05: d_val = std::abs(s_val); break; // ABS.D
            case 0x06: d_val = s_val; break; // MOV.D
            case 0x07: d_val = -s_val; break; // NEG.D
            case 0x08: { // ROUND.L.D
                s64 i_val = static_cast<s64>(std::round(s_val));
                set_fpr64(fd, static_cast<u64>(i_val));
                return;
            }
            case 0x09: { // TRUNC.L.D
                s64 i_val = static_cast<s64>(std::trunc(s_val));
                set_fpr64(fd, static_cast<u64>(i_val));
                return;
            }
            case 0x0A: { // CEIL.L.D
                s64 i_val = static_cast<s64>(std::ceil(s_val));
                set_fpr64(fd, static_cast<u64>(i_val));
                return;
            }
            case 0x0B: { // FLOOR.L.D
                s64 i_val = static_cast<s64>(std::floor(s_val));
                set_fpr64(fd, static_cast<u64>(i_val));
                return;
            }
            case 0x0C: { // ROUND.W.D
                s32 i_val = static_cast<s32>(std::round(s_val));
                set_fpr32(fd, static_cast<u32>(i_val));
                return;
            }
            case 0x0D: { // TRUNC.W.D
                s32 i_val = static_cast<s32>(std::trunc(s_val));
                set_fpr32(fd, static_cast<u32>(i_val));
                return;
            }
            case 0x0E: { // CEIL.W.D
                s32 i_val = static_cast<s32>(std::ceil(s_val));
                set_fpr32(fd, static_cast<u32>(i_val));
                return;
            }
            case 0x0F: { // FLOOR.W.D
                s32 i_val = static_cast<s32>(std::floor(s_val));
                set_fpr32(fd, static_cast<u32>(i_val));
                return;
            }
            case 0x25: { // CVT.L.D
                s64 i_val = static_cast<s64>(s_val);
                set_fpr64(fd, static_cast<u64>(i_val));
                return;
            }
            case 0x20: { // CVT.S.D
                float flt = static_cast<float>(s_val);
                u32 raw;
                std::memcpy(&raw, &flt, 4);
                set_fpr32(fd, raw);
                return;
            }
            case 0x24: { // CVT.W.D
                s32 i_val = static_cast<s32>(s_val);
                set_fpr32(fd, static_cast<u32>(i_val));
                return;
            }
            default:
                if ((funct & 0x30) == 0x30) { // C.cond.D
                    bool c = fp_condition(s_val, t_val, funct & 0x07);
                    if (c) fcsr |= (1 << 23);
                    else fcsr &= ~(1 << 23);
                    return;
                }
                trigger_exception(10); // Reserved Instruction
                return;
        }
        u64 raw;
        std::memcpy(&raw, &d_val, 8);
        set_fpr64(fd, raw);

    } else if (fmt == 0x14) { // Word (.W)
        s32 val = static_cast<s32>(get_fpr32(fs));
        if (funct == 0x20) { // CVT.S.W
            float flt = static_cast<float>(val);
            u32 raw;
            std::memcpy(&raw, &flt, 4);
            set_fpr32(fd, raw);
        } else if (funct == 0x21) { // CVT.D.W
            double dbl = static_cast<double>(val);
            u64 raw;
            std::memcpy(&raw, &dbl, 8);
            set_fpr64(fd, raw);
        } else {
            trigger_exception(10); // Reserved Instruction
        }
    } else if (fmt == 0x15) { // Long (.L)
        s64 val = static_cast<s64>(get_fpr64(fs));
        if (funct == 0x20) { // CVT.S.L
            float flt = static_cast<float>(val);
            u32 raw;
            std::memcpy(&raw, &flt, 4);
            set_fpr32(fd, raw);
        } else if (funct == 0x21) { // CVT.D.L
            double dbl = static_cast<double>(val);
            u64 raw;
            std::memcpy(&raw, &dbl, 8);
            set_fpr64(fd, raw);
        } else {
            trigger_exception(10); // Reserved Instruction
        }
    }
}

void CPU::execute_tlb(u32 funct) {
    switch (funct) {
        case 0x01: { // TLBR
            size_t idx = cp0[CP0Reg::INDEX] & 0x1F;
            const auto& entry = bus.get_tlb_entry(idx);
            cp0[CP0Reg::PAGE_MASK] = entry.page_mask;
            cp0[CP0Reg::ENTRY_HI]  = entry.entry_hi;
            cp0[CP0Reg::ENTRY_LO0] = entry.entry_lo0;
            cp0[CP0Reg::ENTRY_LO1] = entry.entry_lo1;
            break;
        }
        case 0x02: { // TLBWI
            size_t idx = cp0[CP0Reg::INDEX] & 0x1F;
            TLBEntry entry{};
            entry.page_mask = static_cast<u32>(cp0[CP0Reg::PAGE_MASK]);
            entry.entry_hi  = cp0[CP0Reg::ENTRY_HI];
            entry.entry_lo0 = cp0[CP0Reg::ENTRY_LO0];
            entry.entry_lo1 = cp0[CP0Reg::ENTRY_LO1];
            entry.initialized = true;
            bus.set_tlb_entry(idx, entry);
            break;
        }
        case 0x06: { // TLBWR
            size_t idx = cp0[CP0Reg::RANDOM] & 0x1F;
            TLBEntry entry{};
            entry.page_mask = static_cast<u32>(cp0[CP0Reg::PAGE_MASK]);
            entry.entry_hi  = cp0[CP0Reg::ENTRY_HI];
            entry.entry_lo0 = cp0[CP0Reg::ENTRY_LO0];
            entry.entry_lo1 = cp0[CP0Reg::ENTRY_LO1];
            entry.initialized = true;
            bus.set_tlb_entry(idx, entry);
            break;
        }
        case 0x08: { // TLBP
            // Probe TLB for entry matching EntryHi
            u64 entry_hi = cp0[CP0Reg::ENTRY_HI];
            u8 asid = entry_hi & 0xFF;
            bool found = false;
            for (size_t i = 0; i < 32; ++i) {
                const auto& entry = bus.get_tlb_entry(i);
                if (!entry.initialized) continue;
                u32 mask = 0xFFFFE000 & ~entry.page_mask;
                if ((entry.entry_hi & mask) == (entry_hi & mask)) {
                    bool is_global = (entry.entry_lo0 & 1) && (entry.entry_lo1 & 1);
                    if (is_global || ((entry.entry_hi & 0xFF) == asid)) {
                        cp0[CP0Reg::INDEX] = i;
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                cp0[CP0Reg::INDEX] = 0x80000000ULL;
            }
            break;
        }
        default: trigger_exception(10); break; // Reserved Instruction
    }
}
