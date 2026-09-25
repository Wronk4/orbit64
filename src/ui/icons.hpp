#pragma once
// Vector icon set drawn directly with ImDrawList primitives.
//
// Icons are resolution independent (crisp at any DPI / UI scale), need no
// icon font or image assets and render identically on every platform.

#include "imgui.h"

namespace ui {

enum class Icon {
    None,
    Play, Pause, Stop, Reset, FastForward, Step,
    FolderOpen, Folder, File, Chip, Library, Grid, List,
    Settings, Fullscreen, ExitFullscreen, Camera, Gamepad, Volume, VolumeMute,
    Monitor, Cpu, Info, Search, Star, StarFilled, Clock, Close, Check,
    ChevronDown, ChevronRight, ChevronLeft, ChevronUp, More, Trash, Plus, Minus, Refresh,
    Home, Download, Drive, Sliders, Keyboard, Speaker, Sidebar, ArrowUp, Warning, Globe, Sparkle,
    Wave, Layers,
    Bug, Cube, User, Eye, Pencil, Snowflake, Target, Hash, Image,
};

// Draws `icon` centred at `center` inside a box of `size` pixels.
void draw_icon(ImDrawList* dl, Icon icon, ImVec2 center, float size, ImU32 color);

// Draws the Orbit64 logo mark.
void draw_logo(ImDrawList* dl, ImVec2 center, float size, ImU32 accent, ImU32 accent2);

} // namespace ui
