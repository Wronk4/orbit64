#include "input.hpp"
#include "../controller.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>

namespace ui {

InputManager::~InputManager() {
    for (auto& p : pads_) SDL_GameControllerClose(p.handle);
}

void InputManager::open_all() {
    for (int i = 0; i < SDL_NumJoysticks(); ++i) {
        if (!SDL_IsGameController(i)) continue;
        SDL_JoystickID id = SDL_JoystickGetDeviceInstanceID(i);
        bool known = std::any_of(pads_.begin(), pads_.end(), [&](const GamepadInfo& g) { return g.id == id; });
        if (known) continue;
        if (SDL_GameController* gc = SDL_GameControllerOpen(i)) {
            const char* name = SDL_GameControllerName(gc);
            pads_.push_back({id, gc, name ? name : "Gamepad"});
        }
    }
}

void InputManager::handle_event(const SDL_Event& e) {
    switch (e.type) {
        case SDL_CONTROLLERDEVICEADDED: open_all(); break;
        case SDL_CONTROLLERDEVICEREMOVED:
            pads_.erase(std::remove_if(pads_.begin(), pads_.end(),
                                       [&](const GamepadInfo& g) {
                                           if (g.id != e.cdevice.which) return false;
                                           SDL_GameControllerClose(g.handle);
                                           return true;
                                       }),
                        pads_.end());
            break;
        case SDL_KEYDOWN:
            if (capture_port_ >= 0 && !e.key.repeat) {
                if (e.key.keysym.scancode == SDL_SCANCODE_ESCAPE) { cancel_capture(); break; }
                captured_ = true;
                captured_key_ = e.key.keysym.scancode;
                captured_pad_ = -2; // unchanged
            }
            break;
        case SDL_CONTROLLERBUTTONDOWN:
            if (capture_port_ >= 0) {
                captured_ = true;
                captured_key_ = -2;
                captured_pad_ = e.cbutton.button;
            }
            break;
        case SDL_CONTROLLERAXISMOTION:
            if (capture_port_ >= 0 && std::abs(e.caxis.value) > 24000) {
                captured_ = true;
                captured_key_ = -2;
                captured_pad_ = kPadAxisBase + e.caxis.axis * 2 + (e.caxis.value > 0 ? 1 : 0);
            }
            break;
        default: break;
    }
}

std::string InputManager::device_name(int device) const {
    if (device == 0) return "Keyboard";
    if (device - 1 < static_cast<int>(pads_.size())) return pads_[device - 1].name;
    return "Gamepad " + std::to_string(device) + " (not connected)";
}

void InputManager::begin_capture(int port, int input) {
    capture_port_ = port;
    capture_input_ = input;
    captured_ = false;
}

bool InputManager::consume_capture(int& port, int& input, int& key, int& pad) {
    if (!captured_ || capture_port_ < 0) return false;
    port = capture_port_;
    input = capture_input_;
    key = captured_key_;
    pad = captured_pad_;
    captured_ = false;
    capture_port_ = -1;
    return true;
}

SDL_GameController* InputManager::pad_for_device(int device) const {
    if (device <= 0 || device - 1 >= static_cast<int>(pads_.size())) return nullptr;
    return pads_[device - 1].handle;
}

float InputManager::pad_value(SDL_GameController* pad, int binding) const {
    if (!pad || binding < 0) return 0.0f;
    if (binding < kPadAxisBase)
        return SDL_GameControllerGetButton(pad, static_cast<SDL_GameControllerButton>(binding)) ? 1.0f : 0.0f;
    int axis = (binding - kPadAxisBase) / 2;
    bool positive = ((binding - kPadAxisBase) % 2) == 1;
    float v = SDL_GameControllerGetAxis(pad, static_cast<SDL_GameControllerAxis>(axis)) / 32767.0f;
    return positive ? std::max(0.0f, v) : std::max(0.0f, -v);
}

ControllerSnapshot InputManager::poll(const PortConfig& cfg, bool keyboard_enabled) const {
    ControllerSnapshot s;
    s.plugged = cfg.plugged;
    if (!cfg.plugged) return s;

    static const std::uint16_t kMasks[] = {
        Button::A, Button::B, Button::Z, Button::START, Button::L, Button::R,
        Button::C_UP, Button::C_DOWN, Button::C_LEFT, Button::C_RIGHT,
        Button::D_UP, Button::D_DOWN, Button::D_LEFT, Button::D_RIGHT,
    };

    float values[kN64InputCount] = {};
    if (cfg.device == 0) {
        if (keyboard_enabled) {
            const Uint8* keys = SDL_GetKeyboardState(nullptr);
            for (int i = 0; i < kN64InputCount; ++i) {
                int sc = cfg.keys[i];
                if (sc > 0 && sc < SDL_NUM_SCANCODES && keys[sc]) values[i] = 1.0f;
            }
        }
    } else if (SDL_GameController* pad = pad_for_device(cfg.device)) {
        for (int i = 0; i < kN64InputCount; ++i) values[i] = pad_value(pad, cfg.pad[i]);
    }

    for (int i = 0; i < 14; ++i) {
        // Analog bindings used as buttons (e.g. triggers, C-stick) need a threshold.
        if (values[i] > 0.5f) s.buttons |= kMasks[i];
    }

    auto shape = [&](float v) {
        if (v < cfg.deadzone) return 0.0f;
        return std::min(1.0f, (v - cfg.deadzone) / (1.0f - cfg.deadzone));
    };
    float x = shape(values[(int)N64Input::StickRight]) - shape(values[(int)N64Input::StickLeft]);
    float y = shape(values[(int)N64Input::StickUp]) - shape(values[(int)N64Input::StickDown]);
    // The real N64 stick range is roughly ±80 on each axis.
    const float range = 80.0f * cfg.sensitivity;
    s.stick_x = static_cast<std::int8_t>(std::clamp(x * range, -127.0f, 127.0f));
    s.stick_y = static_cast<std::int8_t>(std::clamp(y * range, -127.0f, 127.0f));
    return s;
}

std::string InputManager::key_label(int scancode) {
    if (scancode <= 0) return "Unbound";
    const char* n = SDL_GetScancodeName(static_cast<SDL_Scancode>(scancode));
    return (n && *n) ? n : "Key " + std::to_string(scancode);
}

std::string InputManager::pad_label(int b) {
    if (b < 0) return "Unbound";
    if (b < kPadAxisBase) {
        const char* n = SDL_GameControllerGetStringForButton(static_cast<SDL_GameControllerButton>(b));
        std::string s = n ? n : "button";
        if (!s.empty()) s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
        return s;
    }
    int axis = (b - kPadAxisBase) / 2;
    bool pos = ((b - kPadAxisBase) % 2) == 1;
    switch (axis) {
        case SDL_CONTROLLER_AXIS_LEFTX: return pos ? "Left Stick Right" : "Left Stick Left";
        case SDL_CONTROLLER_AXIS_LEFTY: return pos ? "Left Stick Down" : "Left Stick Up";
        case SDL_CONTROLLER_AXIS_RIGHTX: return pos ? "Right Stick Right" : "Right Stick Left";
        case SDL_CONTROLLER_AXIS_RIGHTY: return pos ? "Right Stick Down" : "Right Stick Up";
        case SDL_CONTROLLER_AXIS_TRIGGERLEFT: return "Left Trigger";
        case SDL_CONTROLLER_AXIS_TRIGGERRIGHT: return "Right Trigger";
        default: return "Axis " + std::to_string(axis);
    }
}

} // namespace ui
