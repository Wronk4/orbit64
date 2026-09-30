#include "input.hpp"
#include "../controller.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>

namespace ui {

namespace {
// The N64 stick moves inside an octagonal gate: about ±80 along the axes and
// (±70, ±70) in the corners. Returns the distance from the centre to that
// gate along the unit direction (ux, uy).
float n64_gate_radius(float ux, float uy) {
    constexpr float kAxis = 80.0f, kCorner = 70.0f;
    constexpr float kQuarter = 1.57079633f / 2.0f; // 45 degrees
    float ang = std::atan2(uy, ux);
    if (ang < 0) ang += 4.0f * 1.57079633f;
    const int k = static_cast<int>(ang / kQuarter) % 8;
    auto vertex = [&](int i, float& x, float& y) {
        const float a = (i % 8) * kQuarter;
        const float r = (i % 2 == 0) ? kAxis : kCorner * 1.41421356f;
        x = std::cos(a) * r;
        y = std::sin(a) * r;
    };
    float ax, ay, bx, by;
    vertex(k, ax, ay);
    vertex(k + 1, bx, by);
    // Where the ray t * (ux, uy) crosses the gate edge from A to B.
    const float ex = bx - ax, ey = by - ay;
    const float den = ux * ey - uy * ex;
    return den != 0.0f ? (ax * ey - ay * ex) / den : kAxis;
}
} // namespace

InputManager::~InputManager() {
    for (auto& p : pads_) SDL_CloseGamepad(p.handle);
}

void InputManager::open_all() {
    int count = 0;
    SDL_JoystickID* ids = SDL_GetGamepads(&count);
    for (int i = 0; ids && i < count; ++i) {
        const SDL_JoystickID id = ids[i];
        bool known = std::any_of(pads_.begin(), pads_.end(), [&](const GamepadInfo& g) { return g.id == id; });
        if (known) continue;
        if (SDL_Gamepad* gc = SDL_OpenGamepad(id)) {
            const char* name = SDL_GetGamepadName(gc);
            pads_.push_back({id, gc, name ? name : "Gamepad"});
        }
    }
    SDL_free(ids);
}

void InputManager::handle_event(const SDL_Event& e) {
    switch (e.type) {
        case SDL_EVENT_GAMEPAD_ADDED: open_all(); break;
        case SDL_EVENT_GAMEPAD_REMOVED:
            pads_.erase(std::remove_if(pads_.begin(), pads_.end(),
                                       [&](const GamepadInfo& g) {
                                           if (g.id != e.gdevice.which) return false;
                                           for (auto& r : rumble_) if (r.pad == g.handle) r = {};
                                           SDL_CloseGamepad(g.handle);
                                           return true;
                                       }),
                        pads_.end());
            break;
        case SDL_EVENT_KEY_DOWN:
            if (capture_port_ >= 0 && !e.key.repeat) {
                if (e.key.scancode == SDL_SCANCODE_ESCAPE) { cancel_capture(); break; }
                captured_ = true;
                captured_key_ = e.key.scancode;
                captured_pad_ = -2; // unchanged
            }
            break;
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
            if (capture_port_ >= 0) {
                captured_ = true;
                captured_key_ = -2;
                captured_pad_ = e.gbutton.button;
            }
            break;
        case SDL_EVENT_GAMEPAD_AXIS_MOTION:
            if (capture_port_ >= 0 && std::abs(e.gaxis.value) > 24000) {
                captured_ = true;
                captured_key_ = -2;
                captured_pad_ = kPadAxisBase + e.gaxis.axis * 2 + (e.gaxis.value > 0 ? 1 : 0);
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

SDL_Gamepad* InputManager::pad_for_device(int device) const {
    if (device <= 0 || device - 1 >= static_cast<int>(pads_.size())) return nullptr;
    return pads_[device - 1].handle;
}

bool InputManager::can_rumble(int device) const {
    SDL_Gamepad* pad = pad_for_device(device);
    return pad && SDL_GetBooleanProperty(SDL_GetGamepadProperties(pad), SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN, false);
}

void InputManager::rumble(int port, const PortConfig& cfg, float strength) {
    RumbleState& r = rumble_[port & 3];
    SDL_Gamepad* pad = cfg.plugged && strength > 0.0f ? pad_for_device(cfg.device) : nullptr;
    // Stop the pad that was vibrating when it changes or the motor turns off.
    if (r.pad && r.pad != pad) {
        SDL_RumbleGamepad(r.pad, 0, 0, 0);
        r = {};
    }
    if (!pad) return;
    // Rumble requests expire, so a running motor is renewed now and then; a
    // stalled UI thread then can't leave the pad vibrating.
    const std::uint64_t now = SDL_GetTicks();
    if (r.pad == pad && r.strength == strength && now - r.sent_ms < 100) return;
    const auto v = static_cast<Uint16>(std::clamp(strength, 0.0f, 1.0f) * 0xFFFF);
    SDL_RumbleGamepad(pad, v, v, 250);
    r = {pad, strength, now};
}

void InputManager::test_rumble(const PortConfig& cfg) {
    if (SDL_Gamepad* pad = pad_for_device(cfg.device)) {
        const auto v = static_cast<Uint16>(std::clamp(cfg.rumble_strength, 0, 100) * 0xFFFF / 100);
        SDL_RumbleGamepad(pad, v, v, 400);
    }
}

float InputManager::pad_value(SDL_Gamepad* pad, int binding) const {
    if (!pad || binding < 0) return 0.0f;
    if (binding < kPadAxisBase)
        return SDL_GetGamepadButton(pad, static_cast<SDL_GamepadButton>(binding)) ? 1.0f : 0.0f;
    int axis = (binding - kPadAxisBase) / 2;
    bool positive = ((binding - kPadAxisBase) % 2) == 1;
    float v = SDL_GetGamepadAxis(pad, static_cast<SDL_GamepadAxis>(axis)) / 32767.0f;
    return positive ? std::max(0.0f, v) : std::max(0.0f, -v);
}

ControllerSnapshot InputManager::poll(const PortConfig& cfg, bool keyboard_enabled) const {
    ControllerSnapshot s;
    s.plugged = cfg.plugged;
    s.pak = cfg.pak;
    if (cfg.pak == 3) s.gb_rom = cfg.gb_rom;
    if (!cfg.plugged) return s;

    static const std::uint16_t kMasks[] = {
        Button::A, Button::B, Button::Z, Button::START, Button::L, Button::R,
        Button::C_UP, Button::C_DOWN, Button::C_LEFT, Button::C_RIGHT,
        Button::D_UP, Button::D_DOWN, Button::D_LEFT, Button::D_RIGHT,
    };

    float values[kN64InputCount] = {};
    if (cfg.device == 0) {
        if (keyboard_enabled) {
            const bool* keys = SDL_GetKeyboardState(nullptr);
            for (int i = 0; i < kN64InputCount; ++i) {
                int sc = cfg.keys[i];
                if (sc > 0 && sc < SDL_SCANCODE_COUNT && keys[sc]) values[i] = 1.0f;
            }
        }
    } else if (SDL_Gamepad* pad = pad_for_device(cfg.device)) {
        for (int i = 0; i < kN64InputCount; ++i) values[i] = pad_value(pad, cfg.pad[i]);
    }

    for (int i = 0; i < 14; ++i) {
        // Analog bindings used as buttons (e.g. triggers, C-stick) need a threshold.
        if (values[i] > 0.5f) s.buttons |= kMasks[i];
    }

    // Radial dead zone: nothing inside the circle, so diagonals don't snap to
    // the axes. The travel past it is stretched back to 0..1 (unless rescaling
    // is off), keeping the direction. With the octagonal gate on, the unit
    // circle is mapped onto the N64's gate; off, each axis reaches ±80 on its
    // own (a square range, as the host stick reports it).
    const float x = values[(int)N64Input::StickRight] - values[(int)N64Input::StickLeft];
    const float y = values[(int)N64Input::StickUp] - values[(int)N64Input::StickDown];
    const float mag = std::sqrt(x * x + y * y);
    float sx = 0.0f, sy = 0.0f;
    const float dz = std::clamp(cfg.deadzone, 0.0f, 0.95f);
    if (mag > dz) {
        const float m = cfg.deadzone_rescale ? (mag - dz) / (1.0f - dz) : mag;
        const float ux = x / mag, uy = y / mag;
        if (cfg.octagon) {
            const float r = n64_gate_radius(ux, uy) * std::min(1.0f, m);
            sx = ux * r;
            sy = uy * r;
        } else {
            sx = std::clamp(ux * m, -1.0f, 1.0f) * 80.0f;
            sy = std::clamp(uy * m, -1.0f, 1.0f) * 80.0f;
        }
        sx *= cfg.sensitivity;
        sy *= cfg.sensitivity;
    }
    s.stick_x = static_cast<std::int8_t>(std::clamp(std::round(sx), -127.0f, 127.0f));
    s.stick_y = static_cast<std::int8_t>(std::clamp(std::round(sy), -127.0f, 127.0f));
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
        const char* n = SDL_GetGamepadStringForButton(static_cast<SDL_GamepadButton>(b));
        std::string s = n ? n : "button";
        if (!s.empty()) s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
        return s;
    }
    int axis = (b - kPadAxisBase) / 2;
    bool pos = ((b - kPadAxisBase) % 2) == 1;
    switch (axis) {
        case SDL_GAMEPAD_AXIS_LEFTX: return pos ? "Left Stick Right" : "Left Stick Left";
        case SDL_GAMEPAD_AXIS_LEFTY: return pos ? "Left Stick Down" : "Left Stick Up";
        case SDL_GAMEPAD_AXIS_RIGHTX: return pos ? "Right Stick Right" : "Right Stick Left";
        case SDL_GAMEPAD_AXIS_RIGHTY: return pos ? "Right Stick Down" : "Right Stick Up";
        case SDL_GAMEPAD_AXIS_LEFT_TRIGGER: return "Left Trigger";
        case SDL_GAMEPAD_AXIS_RIGHT_TRIGGER: return "Right Trigger";
        default: return "Axis " + std::to_string(axis);
    }
}

} // namespace ui
