#!/usr/bin/env python3
"""Picks a small test set out of a full compatibility sweep.

    python tools/test_set.py [--from test_output/compat/results.json] [--out tools/test_set.txt]
    python tools/compat_sweep.py --list tools/test_set.txt [--compare old.json]

A full sweep of a big library takes a long time; this chooses the ROMs that
cover it: a core of games whose fixes must not regress, every game that doesn't
simply run, the slowest ones, the ones with the most warnings or audio clicks,
and one per graphics microcode, audio microcode, CIC and save type.
Each line of the list is a ROM file name with the reason after '#';
rerun this after a full sweep to refresh it.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Games that exercised a specific fix; matched by a piece of the file name.
CORE = ["Super Mario 64", "Ocarina of Time", "Mario Kart 64", "GoldenEye", "Perfect Dark", "Banjo-Kazooie",
        "Donkey Kong 64", "Conker", "Diddy Kong Racing", "Jet Force Gemini", "Paper Mario", "Star Fox 64",
        "F-Zero X", "Resident Evil 2", "Worms", "Rainbow Six", "Pokemon Snap", "Turok - Dinosaur Hunter",
        "Mario Party 3", "1080", "Super Smash Bros", "Quake"]

SLOWEST = 10
NOISY = 3  # most warnings / most audio clicks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default=os.path.join(ROOT, "test_output", "compat", "results.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "tools", "test_set.txt"))
    args = ap.parse_args()
    with open(args.src, encoding="utf-8") as f:
        results = json.load(f)
    if len(results) < 40:
        print(f"{args.src} has only {len(results)} ROMs - build the set from a full sweep")
    chosen = {}  # file name -> reasons

    def pick(r, why):
        chosen.setdefault(os.path.basename(r["rom"]), []).append(why)

    runs = [r for r in results if r["status"] == "runs"]
    for piece in CORE:
        hits = [r for r in results if piece.lower() in os.path.basename(r["rom"]).lower()]
        if hits:
            pick(sorted(hits, key=lambda r: os.path.basename(r["rom"]))[0], "core")
        else:
            print(f"core game not in the sweep: {piece}")
    for r in results:
        if r["status"] != "runs":
            pick(r, r["status"])
    for r in sorted(runs, key=lambda r: r.get("fps", 0))[:SLOWEST]:
        pick(r, f"slow ({r.get('fps', 0):.0f} fps)")
    for r in sorted(runs, key=lambda r: -r.get("warnings", 0))[:NOISY]:
        if r.get("warnings", 0):
            pick(r, f"{r['warnings']} warnings")
    for r in sorted(runs, key=lambda r: -r["audio"].get("clicks_per_s", 0))[:NOISY]:
        pick(r, f"audio clicks ({r['audio'].get('clicks_per_s', 0):.1f}/s)")
    # Coverage: the fastest running game for each microcode, CIC and save chip
    # not already in the set.
    for key, label in (("gfx_ucode", "gfx"), ("audio_abi", "audio ABI"), ("cic", "CIC"), ("save", "save")):
        values = sorted({r.get(key, "?") for r in results})
        for v in values:
            have = [r for r in results if r.get(key) == v and os.path.basename(r["rom"]) in chosen]
            if have:
                continue
            cands = sorted((r for r in runs if r.get(key) == v), key=lambda r: -r.get("fps", 0))
            if cands:
                pick(cands[0], f"{label} {v}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# Test set chosen by tools/test_set.py from {len(results)} ROMs; one ROM file name per line.\n")
        for name in sorted(chosen, key=str.lower):
            f.write(f"{name}  # {', '.join(chosen[name])}\n")
    print(f"{len(chosen)} of {len(results)} ROMs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
