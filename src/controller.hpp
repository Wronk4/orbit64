#pragma once

#include "common.hpp"
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

    // Helper for command string input (e.g. "start", "a", "b", "up", etc.)
    void press_named_button(const std::string& name, bool pressed);

private:
    u16 buttons{0};
    s8 stick_x{0};
    s8 stick_y{0};
    bool plugged_in{true};
};
