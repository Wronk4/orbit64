#pragma once
// Maps host input (keyboard + SDL gamepad devices, which covers
// XInput, DirectInput, PlayStation, Switch Pro and most USB pads on all three
// desktop platforms) onto N64 controller state.

#include "emu_core.hpp"
#include "settings.hpp"

#include <SDL3/SDL.h>
#include <string>
#include <vector>

namespace ui {

struct GamepadInfo {
    SDL_JoystickID id;
    SDL_Gamepad* handle;
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

    // Rumble Pak: runs the motors of the port's gamepad at `strength`
    // (0..1, 0 stops them). Call every frame; unchanged requests are cheap.
    void rumble(int port, const PortConfig& cfg, float strength);
    // Whether `device` (PortConfig::device) is a gamepad that can vibrate.
    bool can_rumble(int device) const;
    // A short pulse at the port's strength (settings "Test" button).
    void test_rumble(const PortConfig& cfg);

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
    SDL_Gamepad* pad_for_device(int device) const;
    float pad_value(SDL_Gamepad* pad, int binding) const;

    std::vector<GamepadInfo> pads_;
    struct RumbleState {
        SDL_Gamepad* pad = nullptr;
        float strength = 0.0f;
        std::uint64_t sent_ms = 0;
    };
    RumbleState rumble_[4];
    int capture_port_ = -1;
    int capture_input_ = -1;
    bool captured_ = false;
    int captured_key_ = -1;
    int captured_pad_ = -1;
};

} // namespace ui
