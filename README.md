# Orbit64: Nintendo 64 emulator

A desktop frontend built on SDL2 and Dear ImGui for the N64 emulation core in `src/`.
It runs on **Windows, macOS and Linux** from a single codebase.

## Building

The only external dependency is **SDL2 ≥ 2.0.18**. Dear ImGui (`third_party/imgui`), stb_image (`third_party/stb`) and the fonts
(`src/ui/fonts_embedded.cpp`) are included in the repository.

| Platform | Install SDL2 | Build |
|---|---|---|
| Windows (MSVC) | `vcpkg install sdl2` | `cmake -S . -B out -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake` then `cmake --build out --config Release` |
| Windows (MSYS2/MinGW) | `pacman -S mingw-w64-x86_64-SDL2` | `cmake -S . -B out -G Ninja && cmake --build out` |
| macOS | `brew install sdl2` | `cmake -S . -B out && cmake --build out` or `make` |
| Linux | `apt install libsdl2-dev`, `dnf install SDL2-devel` or `pacman -S sdl2` | `cmake -S . -B out && cmake --build out` or `make` |

CMake produces `orbit64` (`orbit64.exe`). The Makefile produces `bin/n64`.

On Windows the Makefile also works with [w64devkit](https://github.com/skeeto/w64devkit) and the
`SDL2-devel-*-mingw` package: `build.bat` puts `tools\w64devkit` (or `%W64DEVKIT%`) on `PATH` and runs `make`, and
`SDL2_DIR` points at the package's `x86_64-w64-mingw32` folder (default `tools/SDL2-2.30.12/x86_64-w64-mingw32`).

## Dynamic recompiler

The default CPU core translates MIPS code to x86-64 (Windows / Linux / macOS) or AArch64 native code (`src/jit/`),
falling back to the interpreter for whatever it doesn't handle. Two tools keep it honest:

```
make jit_selftest              # runs ~240 guest programs on both cores and diffs registers, exceptions and memory
tools/jit_bench.sh 600 --stats # every ROM in the project folder: fps, CPU vs RSP/RDP time, JIT coverage, both cores
```

`--cpu interp|jit` and `--jit-stats` also work on their own in `--headless` mode.

## Performance work

- **Profiling (Windows x64):** `--profile samples.txt` in `--headless` mode samples the emulation thread about every
  millisecond; `python tools/prof_report.py bin/n64.exe samples.txt` prints self and inclusive time per function, and
  `--lines <function>` breaks one function down per source line (build with `-g` for that, e.g.
  `make BUILD_DIR=build_prof BIN_DIR=bin_prof CXX="g++ -g"`).
- **Native RDP pass:** triangles and texture rectangles are queued and drawn by row bands on all cores
  (`RDP::flush_native`, `src/raster_pool.*`), with decoded-texture caches; the output is bit-identical to drawing them one
  by one. Check any change to it with `make rdp_check` against a build without the change (`--frame-log` finds the
  first frame that differs).

## Running

```
orbit64 [rom.z64]                       # GUI (shows the library when no ROM is given)
orbit64 rom.z64 --headless 300 --screenshot out.bmp   # core test mode, no window
orbit64 rom.z64 --headless 300 --scale 4 --screenshot out.bmp   # same, rendered at 4x internal resolution
orbit64 --ui-test <dir>                 # scripted tour of every screen, saves captures to <dir>
```

Settings, the library cache and thumbnails are stored in the per-user config folder:
`%APPDATA%\Orbit64`, `~/Library/Application Support/Orbit64` or `~/.local/share/Orbit64`.
Set `ORBIT64_CONFIG_DIR` to use a portable folder instead.

**Box art:** put a No-Intro / RetroArch box art folder (for example `Named_Boxarts`, or any folder with "box" or
"cover" in its name) inside a ROM folder and it is detected automatically. You can also choose a folder in
Settings › Library. Games that have never been started show their box art. After a game is played, its card shows
a screenshot from the last session instead.

## Save states

**Emulation › Save State / Load State** (also in the game screen's right-click menu) save the whole machine to one of
nine slots per game and load it back. The shortcuts are Cmd/Ctrl+S (or F2) to save and Cmd/Ctrl+Shift+L (or F4) to
load the current slot, and Cmd/Ctrl+1…9 to choose the slot. Hover over a slot in the menu to see its screenshot and
when it was saved. *Undo Load State* returns to the moment before the last load. Saving and loading also work while
the game is paused.

States are stored in `<config>/states/<ROM title> [<CRC1>-<CRC2>]/slotN.state` (File › Show Save States Folder).
Because the folder is named after the ROM header, states stay with the game when the ROM file is renamed or moved.
A state includes the cartridge's save memory (EEPROM/SRAM), so after a load the `.sav` file matches the loaded state
the next time it is written. A state only loads in the game it was saved from and in a build with the same state
format (`savestate::kStateVersion`). Other states are refused with a message, and the running game is not changed.

The state is written by a component's `serialize()` (`src/savestate.hpp`). The format is compressed with zlib from
stb. The emulation thread only copies the state, which takes about 1 ms, and compressing and writing the file happens
in the background. `make savestate_check` builds the regression test:

```
bin/savestate_check rom.z64 --at 900 --after 300 [--int] [--scale 2]
```

It checks that a loaded state continues frame for frame exactly like the machine it was saved from. It compares RDRAM,
the image and the full state in a fresh emulator and in one that was already running. It also checks that saving a
loaded state gives the same bytes and that a damaged state is refused.

## Internal resolution

Settings › Graphics › *Internal resolution* (also in View › Internal Resolution and in Debug › Screen Resolution)
renders the game at 2× to 8× its own frame buffer size. This is not an upscale of the finished image. The software
RDP draws every triangle, texture rectangle and fill a second time at the higher resolution, with its own depth buffer
and 8 bits per colour channel, and the displayed frame is built from that copy.

The game itself still gets its normal frame buffer in RDRAM, so frame buffer effects and CPU reads keep working.
Pixels that the CPU or a DMA changed after the RDP drew them are shown from RDRAM, at native resolution. The
high-resolution pass (`src/raster.*`, `src/hires.*`) runs on all CPU cores in horizontal bands, while emulation carries on.

## Debug and memory tools

Open the **Debug** menu (or the bug button in the toolbar). Every tool works with any game:

- **Object Viewer / 3D Object Inspector / Player Viewer.** The frontend records every group of triangles that the game
  draws with one model-view matrix through the graphics microcode (Fast3D, F3DEX and F3DEX2). World coordinates are
  estimated by treating the largest mesh of the frame (usually the level) as the world origin. The player is whichever
  object you choose with *Track as Player*, or an X/Y/Z address you found with Memory Search.
- **Registers, Frame Control, Screen Resolution.** Show CPU/COP0/FPU state and let you pause, advance one frame, set an
  FPS limit, use turbo and change the output scale.
- **Memory Search, RAM Watch, Memory Editor, Freeze List.** Work on RDRAM. Watch and freeze lists are saved per game in
  `<config>/debug/`.

## Frontend layout (`src/ui`)

| File | Responsibility |
|---|---|
| `platform.*` | Everything OS-specific: config/user folders, drives and volumes, native file dialogs (Win32 / AppleScript / zenity or kdialog), "Show in Explorer/Finder", Ctrl vs Cmd, DPI scale |
| `emu_core.*` | Runs the emulator on its own thread and exposes the frame buffer, input, audio, save states and live stats (FPS, microcode, RSP/RDP activity, VI/AI rates) |
| `input.*` | Keyboard and SDL GameController mapping with rebinding |
| `boxart.*` | Box art matching (No-Intro / RetroArch `Named_Boxarts` folders) and asynchronous PNG/JPEG loading (stb_image) |
| `library.*`, `rom_info.*` | Background ROM scanning, header parsing, favorites, recently played list, play time |
| `settings.*` | INI-based persistent settings |
| `theme.*`, `widgets.*`, `icons.*` | Design tokens, custom widgets, animations and a vector icon set |
| `file_browser.*` | Built-in cross-platform ROM and folder picker with ROM header preview |
| `app*.cpp` | Window and main loop, menu/toolbar/status bar, library, game screen, settings, dialogs, save state slots (`app_states.cpp`) |
| `app_debug*.cpp`, `debug_state.hpp` | DEBUG / MEMORY tools |

Rendering goes through `SDL_Renderer`, so SDL picks Direct3D on Windows, Metal on macOS and OpenGL on Linux.
The frontend has no platform-specific rendering code.
