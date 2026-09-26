#pragma once

#include "common.hpp"
#include "transfer_pak.hpp"
#include <string>

namespace Button {
    constexpr u16 A        = 0x8000;
    constexpr u16 B        = 0x4000;
    constexpr u16 Z        = 0x2000;
    constexpr u16 START    = 0x1000;
    constexpr u16 D_UP     = 0x0800;
    constexpr u16 D_DOWN   = 0x0400;
    constexpr u16 D_LEFT   = 0x0200;
    constexpr u16 D_RIGHT  = 0x0100;
    constexpr u16 L        = 0x0020;
    constexpr u16 R        = 0x0010;
    constexpr u16 C_UP     = 0x0008;
    constexpr u16 C_DOWN   = 0x0004;
    constexpr u16 C_LEFT   = 0x0002;
    constexpr u16 C_RIGHT  = 0x0001;
}

// What is plugged into the controller's accessory slot.
enum class Accessory : u8 { None, ControllerPak, RumblePak, TransferPak };

class Controller {
public:
    Controller();

    void reset();

    void set_button(u16 button, bool pressed);
    void set_stick(s8 x, s8 y);

    u16 get_buttons() const { return buttons; }
    s8 get_stick_x() const { return stick_x; }
    s8 get_stick_y() const { return stick_y; }

    bool is_plugged_in() const { return plugged_in; }
    void set_plugged_in(bool plug) { plugged_in = plug; }

    Accessory accessory() const { return accessory_; }
    void set_accessory(Accessory a) { accessory_ = a; if (a != Accessory::RumblePak) rumble_ = false; }
    // Rumble Pak motor, switched by the game through the accessory slot.
    bool rumble() const { return rumble_; }
    void set_rumble(bool on) { rumble_ = on; }
    // The Transfer Pak and its Game Boy cartridge (used while accessory() is TransferPak).
    TransferPak& transfer_pak() { return tpak_; }

    // Helper for command string input (e.g. "start", "a", "b", "up", etc.)
    void press_named_button(const std::string& name, bool pressed);

private:
    u16 buttons{0};
    s8 stick_x{0};
    s8 stick_y{0};
    bool plugged_in{true};
    Accessory accessory_{Accessory::None};
    bool rumble_{false};
    TransferPak tpak_;
};
