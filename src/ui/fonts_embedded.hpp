#pragma once
#include <cstddef>

namespace ui::fonts {
struct Blob {
    const unsigned char* data;
    std::size_t size;
};
extern const Blob font_ui_regular;
extern const Blob font_ui_bold;
extern const Blob font_mono;
} // namespace ui::fonts
