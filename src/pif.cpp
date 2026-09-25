#include "pif.hpp"
#include <iostream>

PIF::PIF() {
    reset(CICType::CIC_6102);
}

void PIF::reset(CICType cic) {
    cic_type = cic;
    std::fill(ram.begin(), ram.end(), 0);
    std::fill(rom.begin(), rom.end(), 0);

    // Boot handshake defaults
    // What the PIF reports about the CIC after boot: 0x24 = 00 0x ss 3F,
    // x = 06 for 6101 else 02, ss = the CIC's seed.
    ram[0x24] = 0x00;
    ram[0x25] = (cic == CICType::CIC_6101) ? 0x06 : 0x02;
    ram[0x26] = cic == CICType::CIC_6103 ? 0x78 : cic == CICType::CIC_6105 ? 0x91 : cic == CICType::CIC_6106 ? 0x85 : 0x3F;
    ram[0x27] = 0x3F;
    ram[63] = 0x00;
}

u8 PIF::read_ram(u32 addr) const {
    u32 offset = addr & 0x3F;
    return ram[offset];
}

void PIF::write_ram(u32 addr, u8 val) {
    u32 offset = addr & 0x3F;
    ram[offset] = val;
}

u8 PIF::read_rom(u32 addr) const {
    u32 offset = addr & (PIF_ROM_SIZE - 1);
    return rom[offset];
}

void PIF::process_commands(Controller controllers[4], Cartridge& cartridge) {
    u8 cmd_byte = ram[63];

    // Check for CIC challenge request
    if (cmd_byte & 0x02) {
        process_cic_challenge(cartridge);
        ram[63] &= ~0x02;
    }

    // Check for Joybus commands
    if ((cmd_byte & 0x01) || cmd_byte == 0x00 || (cmd_byte & 0x08)) {
        int idx = 0;
        int channel = 0;

        while (idx < 63 && channel < 5) {
            u8 cmd = ram[idx++];

            if (cmd == 0x00) {
                // Skip channel
                channel++;
                continue;
            }
            if (cmd == 0xFD) {
                // Channel reset
                channel++;
                continue;
            }
            if (cmd == 0xFE) {
                // End of frame
                break;
            }
            if (cmd == 0xFF) {
                // NOP
                continue;
            }

            u8 tx_len = cmd & 0x3F;
            if (idx >= 63) break;
            u8 rx_len = ram[idx++] & 0x3F;

            if (tx_len == 0 && rx_len == 0) continue;
            if (idx + tx_len > 63) break;

            u8 sub_cmd = ram[idx];

            if (channel < 4) {
                auto& ctrl = controllers[channel];
                if (ctrl.is_plugged_in()) {
                    if (sub_cmd == 0x00 || sub_cmd == 0xFF) {
                        // Info/Reset command
                        if (idx + tx_len + 2 < 64) {
                            ram[idx + tx_len + 0] = 0x05; // Standard controller
                            ram[idx + tx_len + 1] = 0x00;
                            ram[idx + tx_len + 2] = 0x01; // Controller Pak connected
                        }
                    } else if (sub_cmd == 0x01) {
                        // Read controller status
                        if (idx + tx_len + 3 < 64) {
                            u16 btn = ctrl.get_buttons();
                            ram[idx + tx_len + 0] = static_cast<u8>((btn >> 8) & 0xFF);
                            ram[idx + tx_len + 1] = static_cast<u8>(btn & 0xFF);
                            ram[idx + tx_len + 2] = static_cast<u8>(ctrl.get_stick_x());
                            ram[idx + tx_len + 3] = static_cast<u8>(ctrl.get_stick_y());
                        }
                    } else if (sub_cmd == 0x02) {
                        // Read Controller Pak (MemPak)
                        // Address in ram[idx+1], ram[idx+2]
                        if (idx + tx_len + rx_len <= 64) {
                            // Clear returned data + valid CRC (0x00)
                            std::fill_n(&ram[idx + tx_len], rx_len, 0x00);
                        }
                    } else if (sub_cmd == 0x03) {
                        // Write Controller Pak (MemPak)
                        if (idx + tx_len < 64) {
                            ram[idx + tx_len] = 0x00; // CRC
                        }
                    }
                } else {
                    // Controller disconnected: set error bit in rx_len
                    ram[idx - 1] |= 0x80;
                }
            } else if (channel == 4) {
                // Cartridge EEPROM
                SaveType st = cartridge.get_save_type();
                if (st == SaveType::EEPROM_4K || st == SaveType::EEPROM_16K) {
                    if (sub_cmd == 0x00 || sub_cmd == 0xFF) {
                        if (idx + tx_len + 2 < 64) {
                            ram[idx + tx_len + 0] = 0x00;
                            ram[idx + tx_len + 1] = (st == SaveType::EEPROM_16K) ? 0xC0 : 0x80;
                            ram[idx + tx_len + 2] = 0x00;
                        }
                    } else if (sub_cmd == 0x04) {
                        // Read EEPROM block (8 bytes)
                        u8 block = ram[idx + 1];
                        if (idx + tx_len + 8 <= 64) {
                            cartridge.read_eeprom(block, &ram[idx + tx_len]);
                        }
                    } else if (sub_cmd == 0x05) {
                        // Write EEPROM block (8 bytes)
                        u8 block = ram[idx + 1];
                        if (idx + 2 + 8 <= 64) {
                            cartridge.write_eeprom(block, &ram[idx + 2]);
                        }
                        if (idx + tx_len < 64) {
                            ram[idx + tx_len] = 0x00;
                        }
                    }
                } else {
                    ram[idx - 1] |= 0x80;
                }
            }

            idx += tx_len + rx_len;
            channel++;
        }
        ram[63] = 0x00;
    }
}

void PIF::process_cic_challenge(Cartridge& cartridge) {
    if (cartridge.get_cic_type() == CICType::CIC_6105) {
        solve_cic_6105_challenge();
    } else {
        // Dummy challenge: invert bytes
        for (int i = 48; i < 63; ++i) {
            ram[i] = ~ram[i];
        }
    }
}

void PIF::solve_cic_6105_challenge() {
    // Standard CIC 6105 challenge calculation
    static const u8 lut[32] = {
        4, 7, 10, 7, 14, 5, 14, 1, 12, 15, 8, 15, 6, 13, 6, 9,
        4, 11, 2, 11, 14, 9, 14, 13, 12, 3, 0, 3, 6, 1, 6, 5
    };

    u8 key = 0xB;
    for (int i = 48; i < 63; ++i) {
        u8 val = ram[i];
        u8 hi = (val >> 4) & 0x0F;
        u8 lo = val & 0x0F;

        hi = (lut[(hi + key) & 0x1F] + 1) & 0x0F;
        key = hi;
        lo = (lut[(lo + key) & 0x1F] + 1) & 0x0F;
        key = lo;

        ram[i] = (hi << 4) | lo;
    }
}
