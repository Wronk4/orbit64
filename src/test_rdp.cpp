#include "rdp.hpp"
#include "mi.hpp"
#include <iostream>
#include <fstream>
#include <vector>

static void save_bmp(const std::string& filename, const u32* pixels, int width, int height) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return;

    u16 bfType = 0x4D42;
    u32 bfOffBits = 54;
    u32 bfSize = bfOffBits + width * height * 4;
    u32 bfReserved = 0;
    out.write(reinterpret_cast<const char*>(&bfType), 2);
    out.write(reinterpret_cast<const char*>(&bfSize), 4);
    out.write(reinterpret_cast<const char*>(&bfReserved), 4);
    out.write(reinterpret_cast<const char*>(&bfOffBits), 4);

    u32 biSize = 40;
    s32 biWidth = width;
    s32 biHeight = height; // bottom-up
    u16 biPlanes = 1;
    u16 biBitCount = 32;
    u32 biCompression = 0;
    u32 biSizeImage = width * height * 4;
    s32 biXPelsPerMeter = 2835;
    s32 biYPelsPerMeter = 2835;
    u32 biClrUsed = 0;
    u32 biClrImportant = 0;
    out.write(reinterpret_cast<const char*>(&biSize), 4);
    out.write(reinterpret_cast<const char*>(&biWidth), 4);
    out.write(reinterpret_cast<const char*>(&biHeight), 4);
    out.write(reinterpret_cast<const char*>(&biPlanes), 2);
    out.write(reinterpret_cast<const char*>(&biBitCount), 2);
    out.write(reinterpret_cast<const char*>(&biCompression), 4);
    out.write(reinterpret_cast<const char*>(&biSizeImage), 4);
    out.write(reinterpret_cast<const char*>(&biXPelsPerMeter), 4);
    out.write(reinterpret_cast<const char*>(&biYPelsPerMeter), 4);
    out.write(reinterpret_cast<const char*>(&biClrUsed), 4);
    out.write(reinterpret_cast<const char*>(&biClrImportant), 4);

    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            u32 pixel = pixels[y * width + x];
            u8 b = pixel & 0xFF;
            u8 g = (pixel >> 8) & 0xFF;
            u8 r = (pixel >> 16) & 0xFF;
            u8 a = (pixel >> 24) & 0xFF;
            out.put(b); out.put(g); out.put(r); out.put(a);
        }
    }
}

int main() {
    std::ifstream file("mk64_race.ram", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open mk64_race.ram\n";
        return 1;
    }
    std::vector<u8> rdram(8 * 1024 * 1024);
    file.read(reinterpret_cast<char*>(rdram.data()), rdram.size());
    std::cout << "Loaded mk64_race.ram (" << rdram.size() << " bytes)\n";

    RDP rdp;
    rdp.set_ucode_type(MicrocodeType::F3DEX);
    MI mi;

    // Run display list at 0x140490
    std::cout << "Executing display list at 0x140490...\n";
    rdp.process_display_list(0x140490, rdram.data(), rdram.size(), mi);

    // Mario Kart 64 color image framebuffer address is 0x36a780
    u32 fb_addr = 0x36a780;
    std::vector<u32> pixels(320 * 240);
    for (int y = 0; y < 240; ++y) {
        for (int x = 0; x < 320; ++x) {
            u32 idx = fb_addr + (y * 320 + x) * 2;
            u16 p = (static_cast<u16>(rdram[idx]) << 8) | rdram[idx + 1];
            u8 r = ((p >> 11) & 0x1F) * 255 / 31;
            u8 g = ((p >> 6) & 0x1F) * 255 / 31;
            u8 b = ((p >> 1) & 0x1F) * 255 / 31;
            pixels[y * 320 + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
    }

    save_bmp("test_rdp_out.bmp", pixels.data(), 320, 240);
    std::cout << "Saved test_rdp_out.bmp\n";
    return 0;
}
