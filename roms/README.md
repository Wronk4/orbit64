# roms/

Put your ROMs here (`.z64`, `.n64`, `.v64`). Everything in this folder except
this file is ignored by git, so nothing you drop in here gets committed.

- Subfolders are fine: e.g. `roms/usa/`, `roms/europe/`, `roms/homebrew/`.
- Save files (`.sav`) are written next to the ROM they belong to.
- To see them in the game library, add this folder in Settings > Library.
- `tools/jit_bench.sh` and a headless run without a ROM argument
  (`bin/n64 --headless 600`) use the ROMs from here.
