#!/bin/sh
# Headless JIT benchmark / regression check: runs every ROM in roms/
# with both CPU cores and prints fps, the final RDRAM hash and (for the
# JIT) coverage stats.
#   tools/jit_bench.sh [frames] [--no-interp] [--stats]
# The two hashes of a ROM matching means the JIT reproduced the interpreter's
# state exactly; the JIT batches COUNT/interrupt checks per block, so a
# mismatch is not automatically a bug - compare screenshots (test_output/jit_bench/) too.
cd "$(dirname "$0")/.." || exit 1
FRAMES=${1:-600}
INTERP=1
STATS=
for a in "$@"; do
    case "$a" in
        --no-interp) INTERP= ;;
        --stats) STATS=--jit-stats ;;
    esac
done
OUT=test_output/jit_bench
mkdir -p "$OUT"
for rom in roms/*.z64 roms/*.n64 roms/*.v64; do
    [ -f "$rom" ] || continue
    name=$(basename "$rom" | sed 's/[^A-Za-z0-9]/_/g' | cut -c1-24)
    echo "=== $rom"
    ./bin/n64 "$rom" --headless "$FRAMES" --no-save --cpu jit $STATS --screenshot "$OUT/${name}_jit.bmp" 2>&1 |
        grep -E '^\[Main\] (core=|time split)|^\[JIT\]'
    if [ -n "$INTERP" ]; then
        ./bin/n64 "$rom" --headless "$FRAMES" --no-save --cpu interp $STATS --screenshot "$OUT/${name}_interp.bmp" 2>&1 |
            grep -E '^\[Main\] (core=|time split)'
        if cmp -s "$OUT/${name}_jit.bmp" "$OUT/${name}_interp.bmp"; then
            echo "    screenshots: identical"
        else
            echo "    screenshots: DIFFER"
        fi
    fi
done
