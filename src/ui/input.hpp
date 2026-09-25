#pragma once
// Maps host input (keyboard + SDL GameController devices, which covers
// XInput, DirectInput, PlayStation, Switch Pro and most USB pads on all three
// desktop platforms) onto N64 controller state.

#include "emu_core.hpp"
#include "settings.hpp"

#include <SDL.h>
#include <string>
#include <vector>

namespace ui {

struct GamepadInfo {
    SDL_JoystickID id;
    SDL_GameController* handle;
    std::string name;
};

class InputManager {
public:
    ~InputManager();

    void open_all();
    void handle_event(const SDL_Event& e);

    const std::vector<GamepadInfo>& gamepads() const { return pads_; }
    std::string device_name(int device) const;

    // Builds controller state for a port. `keyboard_enabled` is false while a
    // text field has focus, so typing in the UI doesn't move Mario.
    ControllerSnapshot poll(const PortConfig& cfg, bool keyboard_enabled) const;

    // Rebinding capture: returns true once a key / pad input was captured.
    void begin_capture(int port, int input);
    void cancel_capture() { capture_port_ = -1; }
    bool capturing() const { return capture_port_ >= 0; }
    int capture_port() const { return capture_port_; }
    int capture_input() const { return capture_input_; }
    bool consume_capture(int& port, int& input, int& key, int& pad);

    static std::string key_label(int scancode);
    static std::string pad_label(int binding);

private:
    SDL_GameController* pad_for_device(int device) const;
    float pad_value(SDL_GameController* pad, int binding) const;

    std::vector<GamepadInfo> pads_;
    int capture_port_ = -1;
    int capture_input_ = -1;
    bool captured_ = false;
    int captured_key_ = -1;
    int captured_pad_ = -1;
};

} // namespace ui
