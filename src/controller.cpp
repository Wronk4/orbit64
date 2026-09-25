#include "controller.hpp"

Controller::Controller() {
    reset();
}

void Controller::reset() {
    buttons = 0;
    stick_x = 0;
    stick_y = 0;
    plugged_in = true;
}

void Controller::set_button(u16 button, bool pressed) {
    if (pressed) {
        buttons |= button;
    } else {
        buttons &= ~button;
    }
}

void Controller::set_stick(s8 x, s8 y) {
    stick_x = x;
    stick_y = y;
}

void Controller::press_named_button(const std::string& name, bool pressed) {
    if (name == "a" || name == "A") set_button(Button::A, pressed);
    else if (name == "b" || name == "B") set_button(Button::B, pressed);
    else if (name == "z" || name == "Z") set_button(Button::Z, pressed);
    else if (name == "start" || name == "START") set_button(Button::START, pressed);
    else if (name == "dup" || name == "up") set_button(Button::D_UP, pressed);
    else if (name == "ddown" || name == "down") set_button(Button::D_DOWN, pressed);
    else if (name == "dleft" || name == "left") set_button(Button::D_LEFT, pressed);
    else if (name == "dright" || name == "right") set_button(Button::D_RIGHT, pressed);
    else if (name == "l" || name == "L") set_button(Button::L, pressed);
    else if (name == "r" || name == "R") set_button(Button::R, pressed);
    else if (name == "cup") set_button(Button::C_UP, pressed);
    else if (name == "cdown") set_button(Button::C_DOWN, pressed);
    else if (name == "cleft") set_button(Button::C_LEFT, pressed);
    else if (name == "cright") set_button(Button::C_RIGHT, pressed);
    else if (name == "stick_up") set_stick(0, pressed ? 80 : 0);
    else if (name == "stick_down") set_stick(0, pressed ? -80 : 0);
    else if (name == "stick_left") set_stick(pressed ? -80 : 0, 0);
    else if (name == "stick_right") set_stick(pressed ? 80 : 0, 0);
}
