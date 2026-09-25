// Save states: slots, menus, shortcuts' actions and result toasts.
//
// Each game keeps its states in its own folder under the configuration
// directory, named after the ROM header (title and checksums), so they follow
// the game even when the ROM file is renamed or moved. EmuCore does the work
// on the emulation thread and reports back through poll_state_event().

#include "app.hpp"
#include "platform.hpp"
#include "../savestate.hpp"

#include "imgui.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>

namespace ui {

namespace fs = std::filesystem;

fs::path App::states_dir() const {
    // A UI test keeps its states with its captures, away from the player's.
    const fs::path root = ui_test_dir_.empty() ? platform::config_dir() / "states"
                                               : platform::utf8_to_path(ui_test_dir_) / "states";
    if (!core_.loaded()) return root;
    std::string name = current_rom_.internal_name.empty() ? current_rom_.display_title : current_rom_.internal_name;
    for (char& c : name) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                        std::strchr(" .,-_()'!&+", c) != nullptr;
        if (!ok) c = '_';
    }
    while (!name.empty() && (name.back() == ' ' || name.back() == '.')) name.pop_back();
    if (name.empty()) name = "Game";
    char crc[24];
    std::snprintf(crc, sizeof crc, " [%08X-%08X]", current_rom_.crc1, current_rom_.crc2);
    return root / platform::utf8_to_path(name + crc);
}

fs::path App::state_path(int slot) const {
    return states_dir() / ("slot" + std::to_string(slot) + ".state");
}

void App::save_state(int slot) {
    if (!core_.loaded()) {
        toast("Start a game to save its state", ToastKind::Warning);
        return;
    }
    core_.save_state(state_path(slot)); // the toast comes with the result
}

void App::load_state(int slot) {
    if (!core_.loaded()) {
        toast("Start a game to load a state", ToastKind::Warning);
        return;
    }
    std::error_code ec;
    if (!fs::exists(state_path(slot), ec)) {
        toast("Slot " + std::to_string(slot) + " is empty", ToastKind::Warning);
        return;
    }
    core_.load_state(state_path(slot));
}

void App::undo_load_state() {
    if (core_.can_undo_load()) core_.undo_load_state();
}

void App::select_state_slot(int slot) {
    state_slot_ = std::clamp(slot, 1, kStateSlots);
    if (!core_.loaded()) return;
    state_slots_checked_ = -1.0;
    refresh_state_slots();
    const StateSlot& s = state_slots_[state_slot_];
    toast("State slot " + std::to_string(state_slot_) + (s.exists ? " \xE2\x80\x94 saved " + s.when : " \xE2\x80\x94 empty"));
}

void App::poll_state_events() {
    StateEvent ev;
    while (core_.poll_state_event(ev)) {
        // " to slot 3" (empty once the game has been stopped)
        std::string slot;
        for (int i = 1; i <= kStateSlots; ++i)
            if (ev.path == state_path(i)) slot = " slot " + std::to_string(i);
        switch (ev.op) {
        case StateEvent::Op::Save:
            state_slots_checked_ = -1.0;
            if (ev.ok) toast("State saved" + (slot.empty() ? "" : " to" + slot), ToastKind::Success, 2.0f);
            else toast("Couldn't save the state: " + ev.error, ToastKind::Error, 5.0f);
            break;
        case StateEvent::Op::Load:
            if (ev.ok) toast("State loaded" + (slot.empty() ? "" : " from" + slot), ToastKind::Success, 2.0f);
            else toast("Couldn't load" + (slot.empty() ? std::string(" the state") : slot) + ": " + ev.error,
                       ToastKind::Error, 5.0f);
            break;
        case StateEvent::Op::UndoLoad:
            if (ev.ok) toast("Back to where you were before loading", ToastKind::Success, 2.5f);
            else toast("Couldn't undo the load: " + ev.error, ToastKind::Error, 5.0f);
            break;
        }
    }
}

void App::clear_state_slots() {
    for (StateSlot& s : state_slots_) {
        if (s.thumb) SDL_DestroyTexture(s.thumb);
        s = StateSlot{};
    }
    state_slots_checked_ = -1.0;
}

// Menus call this every frame they're open; the files are only looked at
// twice a second, and a slot's header only re-read when its file changed.
void App::refresh_state_slots() {
    const double now = ImGui::GetTime();
    if (state_slots_checked_ >= 0.0 && now - state_slots_checked_ < 0.5) return;
    state_slots_checked_ = now;
    for (int i = 1; i <= kStateSlots; ++i) {
        StateSlot& s = state_slots_[i];
        std::error_code ec;
        const fs::path p = state_path(i);
        const auto mtime = fs::last_write_time(p, ec);
        if (ec) {
            if (s.thumb) SDL_DestroyTexture(s.thumb);
            s = StateSlot{};
            continue;
        }
        if (s.exists && s.mtime == mtime) continue;
        if (s.thumb) SDL_DestroyTexture(s.thumb);
        s = StateSlot{};
        savestate::FileInfo info;
        std::string err;
        if (!savestate::read_file(p, nullptr, info, err)) continue;
        s.exists = true;
        s.mtime = mtime;
        const std::time_t t = static_cast<std::time_t>(info.created);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char when[40];
        std::strftime(when, sizeof when, "%d %b %Y, %H:%M", &tm);
        s.when = when;
        if (!info.thumb.empty()) {
            s.thumb = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC,
                                        static_cast<int>(info.thumb_w), static_cast<int>(info.thumb_h));
            if (s.thumb) {
                SDL_UpdateTexture(s.thumb, nullptr, info.thumb.data(), static_cast<int>(info.thumb_w) * 4);
                SDL_SetTextureScaleMode(s.thumb, SDL_ScaleModeLinear);
            }
        }
    }
}

// Picture and time of a slot, while its menu item is hovered.
void App::state_slot_tooltip(int slot) {
    const StateSlot& s = state_slots_[slot];
    if (!s.exists || !ImGui::IsItemHovered()) return;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, dp(8, 8));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, dp(6));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, g_pal.bg4);
    ImGui::PushStyleColor(ImGuiCol_Border, g_pal.border_strong);
    if (ImGui::BeginTooltip()) {
        if (s.thumb) ImGui::Image((ImTextureID)(intptr_t)s.thumb, ImVec2(dp(224), dp(168)));
        ImGui::PushFont(g_fonts.small);
        ImGui::Text("Slot %d \xC2\xB7 %s", slot, s.when.c_str());
        ImGui::PopFont();
        ImGui::EndTooltip();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
}

void App::draw_state_menu() {
    using platform::shortcut_label;
    const bool loaded = core_.loaded();
    if (loaded) refresh_state_slots();
    const std::string slot = std::to_string(state_slot_);
    if (ImGui::MenuItem(("Save State (Slot " + slot + ")").c_str(), shortcut_label(true, false, false, "S").c_str(), false,
                        loaded))
        save_state(state_slot_);
    if (ImGui::MenuItem(("Load State (Slot " + slot + ")").c_str(), shortcut_label(true, true, false, "L").c_str(), false,
                        loaded && state_slots_[state_slot_].exists))
        load_state(state_slot_);
    state_slot_tooltip(state_slot_);
    if (ImGui::BeginMenu("Save State to Slot", loaded)) {
        for (int i = 1; i <= kStateSlots; ++i) {
            const StateSlot& s = state_slots_[i];
            if (ImGui::MenuItem(("Slot " + std::to_string(i)).c_str(), s.exists ? s.when.c_str() : "Empty")) {
                state_slot_ = i;
                save_state(i);
            }
            state_slot_tooltip(i);
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Load State from Slot", loaded)) {
        for (int i = 1; i <= kStateSlots; ++i) {
            const StateSlot& s = state_slots_[i];
            if (ImGui::MenuItem(("Slot " + std::to_string(i)).c_str(), s.exists ? s.when.c_str() : "Empty", false,
                                s.exists)) {
                state_slot_ = i;
                load_state(i);
            }
            state_slot_tooltip(i);
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Current Slot")) {
        for (int i = 1; i <= kStateSlots; ++i) {
            const std::string key = std::to_string(i);
            if (ImGui::MenuItem(("Slot " + key).c_str(), shortcut_label(true, false, false, key.c_str()).c_str(),
                                state_slot_ == i))
                select_state_slot(i);
        }
        ImGui::EndMenu();
    }
    if (ImGui::MenuItem("Undo Load State", nullptr, false, loaded && core_.can_undo_load())) undo_load_state();
}

void App::reveal_states_folder() {
    // The running game's folder once it has one, otherwise all of them.
    std::error_code ec;
    fs::path dir = states_dir();
    if (core_.loaded() && !fs::is_directory(dir, ec)) dir = dir.parent_path();
    fs::create_directories(dir, ec);
    platform::reveal_in_file_manager(dir);
}

} // namespace ui
