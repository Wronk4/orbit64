# Orbit64: Nintendo 64 emulator

> **Work in progress.** Orbit64 is under active development. Many games are playable, but others may have graphical
> or audio glitches, crash or not start. Features and the save state format can change between releases.

A desktop frontend built on SDL3 and Dear ImGui for the N64 emulation core in `src/`. Graphics go through SDL_GPU:
Metal on macOS, Vulkan on Linux and Windows (Direct3D 12 on Windows machines without Vulkan).
It runs on **Windows, macOS and Linux** from a single codebase.

## Folders

| Folder | What is in it |
|---|---|
| `src/` | The emulator: CPU + JIT (`src/jit/`), RSP/RDP, audio, save states; the desktop frontend is `src/ui/` |
| `tools/` | Test and analysis tools (`jit_bench.sh`, `audio_report.py`, `*_check.cpp`, profiler report) |
| `third_party/`, `assets/` | Dear ImGui, stb; icons and other bundled files |
| `docs/` | Notes on the emulator's internals |
| `site/` | The project page on GitHub Pages (https://wronk4.github.io/orbit64/, published by `.github/workflows/pages.yml` from `main` or `nightly`) |
| `roms/` | **Your ROMs** (subfolders are fine). Ignored by git except its README |
| `bin/`, `build/` | Build output (Makefile); `out/` for CMake |
| `test_output/` | What the test tools write (screenshots, `--wav` captures, reports). Safe to delete |
| `reference/` | Local reference material such as the N64 SDK. Never committed |

## Downloads

Every push to GitHub is built for Windows, macOS, Linux and FreeBSD by `.github/workflows/build.yml`. Each push to `main`
publishes a new release on the Releases page (`v1.0.<commits on main>`, the first two numbers come from
`project(VERSION)` in `CMakeLists.txt`) with `orbit64-windows-x64.zip`, `orbit64-windows-arm64.zip`, `orbit64-macos-universal.zip`,
`orbit64-linux-x64.tar.gz`, `orbit64-linux-arm64.tar.gz`, `orbit64-freebsd-x64.tar.gz` and `orbit64-freebsd-arm64.tar.gz`; pushing a tag such as `v2.0.0` publishes that version. Day-to-day work goes to the
`nightly` branch, whose builds (like those of any other branch) are only kept as artifacts of their run in the Actions
tab. The macOS binary is not notarized: run `xattr -dr com.apple.quarantine orbit64` once after unpacking.

## Building

The only external dependency is **SDL3 ≥ 3.4**. Dear ImGui (`third_party/imgui`), stb_image (`third_party/stb`), the fonts
(`src/ui/fonts_embedded.cpp`) and the prebuilt GPU shaders (`src/gpu/shaders_gen.cpp`) are included in the repository.

| Platform | Install SDL3 (optional with CMake) | Build |
|---|---|---|
| Windows (MSVC) | `vcpkg install sdl3` | `cmake -S . -B out -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake` then `cmake --build out --config Release` |
| Windows (MSYS2/MinGW) | `pacman -S mingw-w64-x86_64-SDL3` | `cmake -S . -B out -G Ninja && cmake --build out` |
| macOS | `brew install sdl3` | `cmake -S . -B out && cmake --build out` or `make` |
| Linux | `apt install libsdl3-dev`, `dnf install SDL3-devel` or `pacman -S sdl3` | `cmake -S . -B out && cmake --build out` or `make` |

When CMake finds no SDL3 3.4 (or with `-DORBIT64_VENDOR_SDL3=ON`) it downloads and builds SDL3 itself. `-DORBIT64_SDL3_STATIC=ON`
links that copy statically (a single self-contained executable, as the release builds do). CMake produces
`orbit64` (`orbit64.exe`, with `SDL3.dll` next to it). The Makefile produces `bin/n64` and uses `pkg-config sdl3`; on
Windows with a MinGW toolchain such as [w64devkit](https://github.com/skeeto/w64devkit), `SDL3_DIR` points at the
`x86_64-w64-mingw32` folder of the `SDL3-devel-*-mingw` package.

The GPU shaders are GLSL (`src/gpu/shaders/*.comp`). After editing one, `make shaders` (or
`python3 tools/gen_shaders.py`) rebuilds `src/gpu/shaders_gen.cpp` for every format SDL_GPU takes; it needs
`glslangValidator` and `spirv-cross` (`brew install glslang spirv-cross`, `apt install glslang-tools spirv-cross` or the
Vulkan SDK).

## Dynamic recompiler

The default CPU core translates MIPS code to x86-64 (Windows / Linux / macOS) or AArch64 native code (`src/jit/`),
falling back to the interpreter for whatever it doesn't handle. Two tools keep it honest:

```
make jit_selftest              # runs ~240 guest programs on both cores and diffs registers, exceptions and memory
tools/jit_bench.sh 600 --stats # every ROM in roms/: fps, CPU vs RSP/RDP time, JIT coverage, both cores
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

## Compatibility testing

`python tools/compat_sweep.py` runs every ROM in `roms/` headless for 30 s of game time (pressing START/A a few times
to get past title screens) and writes `test_output/compat/report.html`: status, speed, graphics and audio microcode,
screenshots and the captured sound per game, plus `sheet.png` with all screenshots and `results.json`.
`--compare <old results.json>` marks what changed since an earlier sweep (status, and whether the game ran differently:
runs are deterministic, so a different RDRAM hash means different behaviour); `--filter <text>` runs a subset.

With a big library, `python tools/test_set.py` picks a small test set out of a full sweep's `results.json` - every game
that doesn't simply run, the slowest ones, the noisiest ones, one per microcode, CIC and save type, and a core of games
whose fixes must not regress - into `tools/test_set.txt`; `compat_sweep.py --list tools/test_set.txt` runs just those.

### Game status page

`python tools/game_status.py` answers a different question: how far does each game get. It needs `make game_probe`
(`bin/game_probe`), which plays a ROM by itself. From the same save state it tries the game with no input, with A, with
START and with the stick held, and compares the pictures: whatever changes is the game reacting to that input, so it
presses A or START only where the game reacts, moves the cursor of a menu before A, and finally checks whether the stick
moves the character or the camera. Every game ends up with one status - `INGAME`, `MENU`, `INTRO_TITLE`, `BLACK_SCREEN`
or `CRASH_ERROR` - plus a finer detail (`freeze`, `playable`, ...).

Each game runs on the high-level RSP first (`--rsp hle`); the ones that don't reach gameplay, and did start a graphics
task, are run again on the low-level RSP and RDP (`--rsp lle-gfx`), and the better result counts. Identical dumps
(same CRCs) run once. Interrupting with Ctrl+C is safe: run the same command again and it continues where it stopped
(`--fresh` starts over, `--redo MENU,INTRO_TITLE` or `--redo CUT` repeat the games the probe could not get through, `--hle-only`
skips the low-level pass). `--report-only` rebuilds the reports from the saved runs.

The verdict is automatic and can be wrong (a game that needs a choice in a menu may stay `MENU`), so it can be corrected by
hand in `tools/game_status_review.json`, keyed by the ROM's CRCs, and such games are marked "checked" on the page.
`--publish` writes the result page to `site/compatibility/`, which GitHub Pages publishes at
<https://wronk4.github.io/orbit64/compatibility/>; without it the page goes to `test_output/game_status/web/`, next to
`review/` (screenshots and verdicts, ten games per sheet) and `results.json`.

## RSP

The graphics and audio tasks of libultra games are emulated at a high level (`src/rdp.cpp`, `src/ahle.*`). Every other
task a game starts on the RSP runs on a low-level interpreter of the RSP's scalar and vector units
(`src/rsp_core.*`), which passes all of the RSP sub-tests of Nintendo's N64 diagnostics cartridge.
`ORBIT64_RSP=lle` runs every task on it and `ORBIT64_RSP=lle-audio` only the audio ones (their output is within
rounding of the high-level audio, a good check of both). `ORBIT64_RSP_DUMP=<dir>` saves the IMEM and DMEM each
low-level task starts with, and `make rsp_test && bin/rsp_test <dir>/*.bin` runs such images on their own.

## Audio

- **Microcode HLE** (`src/ahle.*`): the audio command lists of the libultra ABI 1 microcode (and its GoldenEye /
  Diddy Kong Racing variant), n_audio (Rare and many third-party games) and the Nintendo EAD microcodes (Mario Kart 64,
  Star Fox 64 / F-Zero X, Zelda / Yoshi / 1080) are run with the microcode's own DMEM layouts, rounding and saturation.
  Per-voice state (ADPCM history, resampler phase, envelopes, filters) lives in RDRAM where the game points it, as on
  hardware. Factor 5's MusyX synthesizer (v1: Rogue Squadron; v2: Resident Evil 2, Indiana Jones, Battle for Naboo)
  is emulated too: PCM16/ADPCM voices, envelopes, resampling, delay/reverb.
- **Output** (`src/audio_stream.*`): the samples the game hands to the DAC go through a lock-free queue to the audio
  callback, which resamples them to the device rate with a 32-tap windowed-sinc filter. The pitch is never bent to
  absorb clock drift: at normal speed the emulator runs up to 3% faster or slower to keep the queue on target
  (Settings › Audio shows the resulting latency and any dropouts).
- **Checks:** `--wav out.wav` in `--headless` mode saves the raw DAC stream (native rate) with a hash of it;
  `python tools/audio_report.py out.wav --png spec.png` reports level, clipping and clicks and draws a spectrogram;
  `make audio_check` tests the resampler (tones at every native/device rate pair) and simulates the pacing against a
  drifting device clock. `ORBIT64_PACING_LOG=<file>` makes the GUI log frame timing, queue level and dropouts once a
  second.

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
Pixels that the CPU or a DMA changed after the RDP drew them are shown from RDRAM, at native resolution.

The high-resolution pass runs on the GPU (`src/gpu/`): compute shaders run the same pixel pipeline as the CPU renderer
(combiner, blender, depth, TMEM decoding), one invocation per output pixel over the primitives binned to its 8x8 tile,
and the finished frame stays in video memory for the window to show. Without a usable GPU driver (or with Settings ›
Graphics › Video backend › Compatibility, or `ORBIT64_HIRES=cpu`) it runs on all CPU cores in horizontal bands instead
(`src/raster.*`, `src/hires.*`).

`make gpu_check` builds a checker that runs a ROM once per renderer and compares every K-th frame:
`bin/gpu_check rom.z64 --scale 4 --frames 1200 --every 40 --mash start [--shots dir [--all]]`, or `--bench` to time the
GPU renderer alone. `--shots` saves the worst frame of each renderer and a diff, `--all` every compared frame.

Texture rectangles at 2× and up keep their S/T coordinates inside the range the native rectangle samples. Games that
draw 2D art in strips, each with its own texture load (Mario Kart 64's menus), would otherwise show seams between the
strips, because bilinear filtering at sub-pixel positions reaches texels that were never loaded. `ORBIT64_GPU_DRIVER=vulkan` picks the driver (on macOS through MoltenVK, with
`SDL_VULKAN_LIBRARY=/opt/homebrew/lib/libvulkan.1.dylib`); `ORBIT64_GPU_STATS=1` prints per-second figures.

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
| `input.*` | Keyboard and SDL gamepad mapping with rebinding |
| `boxart.*` | Box art matching (No-Intro / RetroArch `Named_Boxarts` folders) and asynchronous PNG/JPEG loading (stb_image) |
| `library.*`, `rom_info.*` | Background ROM scanning, header parsing, favorites, recently played list, play time |
| `settings.*` | INI-based persistent settings |
| `theme.*`, `widgets.*`, `icons.*` | Design tokens, custom widgets, animations and a vector icon set |
| `file_browser.*` | Built-in cross-platform ROM and folder picker with ROM header preview |
| `app*.cpp` | Window and main loop, menu/toolbar/status bar, library, game screen, settings, dialogs, save state slots (`app_states.cpp`) |
| `app_debug*.cpp`, `debug_state.hpp` | DEBUG / MEMORY tools |

The UI is drawn with SDL's GPU `SDL_Renderer` on the same `SDL_GPUDevice` as the RDP renderer (`gpu::create_device()`
in `src/gpu/device.cpp`: Metal, Vulkan or Direct3D 12), so frames rendered on the GPU are shown without a copy back to
the CPU. When no GPU device can be created, SDL's default renderer is used. The frontend has no platform-specific
rendering code.

## Thanks

Thank you to [Mupen64Plus](https://mupen64plus.org/) and everyone who has worked on it since Hacktarux's original
Mupen64. For well over a decade they have built a free, portable N64 emulator in the open and shared what they learned
about the console. That work taught a generation of emulator authors and homebrew developers how the Nintendo 64
works, and every N64 emulator written since, this one included, benefits from it. Thanks as well to the wider N64
emulation and homebrew community.
