#!/bin/sh
# Headless JIT benchmark / regression check: runs every ROM in the project
# folder with both CPU cores and prints fps, the final RDRAM hash and (for the
# JIT) coverage stats.
#   tools/jit_bench.sh [frames] [--no-interp] [--stats]
# The two hashes of a ROM matching means the JIT reproduced the interpreter's
# state exactly; the JIT batches COUNT/interrupt checks per block, so a
# mismatch is not automatically a bug - compare screenshots (bench_out/) too.
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
mkdir -p bench_out
for rom in *.z64 *.n64 *.v64; do
    [ -f "$rom" ] || continue
    name=$(echo "$rom" | sed 's/[^A-Za-z0-9]/_/g' | cut -c1-24)
    echo "=== $rom"
    ./bin/n64 "$rom" --headless "$FRAMES" --no-save --cpu jit $STATS --screenshot "bench_out/${name}_jit.bmp" 2>&1 |
        grep -E '^\[Main\] (core=|time split)|^\[JIT\]'
    if [ -n "$INTERP" ]; then
        ./bin/n64 "$rom" --headless "$FRAMES" --no-save --cpu interp $STATS --screenshot "bench_out/${name}_interp.bmp" 2>&1 |
            grep -E '^\[Main\] (core=|time split)'
        if cmp -s "bench_out/${name}_jit.bmp" "bench_out/${name}_interp.bmp"; then
            echo "    screenshots: identical"
        else
            echo "    screenshots: DIFFER"
        fi
    fi
done
