#!/usr/bin/env python3
"""How well does OCR read emulator screens? Scores every engine / enlargement / picture version on
tools/ocr_bench/*.png against what a person reads in them (tools/ocr_bench_labels.json).

    python tools/ocr_bench.py                     # the standard comparison
    python tools/ocr_bench.py --engines rapid --scales 3 4 --variants raw stretch
    python tools/ocr_bench.py --show 22           # what is read on one picture, line by line

A phrase counts as found when its letters and digits appear in the text that was read, allowing one wrong
character per five (so "EXIT AND KEEP CHANGES" read as "EXIT AND KEEP CHANGE5" is found). Recall is the
share of phrases found; "menu" recall is the same for the words a decision rests on.
"""
import argparse
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocr_engines as oe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def approx_in(needle: str, hay: str, max_err: int) -> bool:
    """Whether `needle` occurs in `hay` with at most max_err substitutions / insertions / deletions."""
    if not needle:
        return True
    if max_err == 0:
        return needle in hay
    prev = [0] * (len(hay) + 1)  # Sellers: a match may start anywhere
    for i, nc in enumerate(needle, 1):
        cur = [i] + [0] * len(hay)
        for j, hc in enumerate(hay, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (nc != hc))
        prev = cur
    return min(prev) <= max_err


def found(phrase: str, texts) -> bool:
    p = oe.norm(phrase)
    if not p:
        return True
    err = len(p) // 5
    return any(approx_in(p, oe.norm(t), err) for t in texts)


def dict_copy(b):
    import copy
    return copy.copy(b)


def load():
    labels = json.load(open(os.path.join(HERE, "ocr_bench_labels.json"), encoding="utf-8"))
    items = []
    for k, v in labels.items():
        if k.startswith("_"):
            continue
        path = os.path.join(HERE, "ocr_bench", f"{int(k):02d}.png")
        if os.path.exists(path):
            items.append((int(k), v, path))
    return items


_CACHE = {}
_TIME = {}


def cached_read(k, path, engine, scale, vname):
    key = (k, engine, scale, vname)
    if key not in _CACHE:
        t0 = time.time()
        _CACHE[key] = oe.read_one(Image.open(path), engine, scale, vname)
        _TIME[key] = time.time() - t0
    return _CACHE[key]


def score(items, engines, scales, variant_names, verbose=False):
    tot = hit = mtot = mhit = 0
    t0 = time.time()
    misses = []
    for k, v, path in items:
        boxes = oe.merge([dict_copy(b) for en in engines for sc in scales for vn in variant_names
                          for b in cached_read(k, path, en, sc, vn)])
        # the lines, and also the whole text in reading order: a phrase split over two boxes is still read
        texts = [b.text for b in boxes] + [" ".join(b.text for b in boxes)]
        for ph in v["text"]:
            tot += 1
            ok = found(ph, texts)
            hit += ok
            if not ok:
                misses.append((k, v["name"], ph))
        for ph in v.get("menu", []):
            mtot += 1
            mhit += found(ph, texts)
    dt = sum(_TIME.get((k, en, sc, vn), 0) for k, _, _ in items for en in engines for sc in scales for vn in variant_names) / max(1, len(items))
    return hit / tot, mhit / max(1, mtot), dt, misses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", nargs="*", help="engines to try (default: all that work)")
    ap.add_argument("--scales", nargs="*", type=int, default=[2, 3, 4])
    ap.add_argument("--variants", nargs="*", help="picture versions (raw stretch invert); default: each alone and all together")
    ap.add_argument("--show", type=int, help="print what is read on one picture")
    ap.add_argument("--misses", action="store_true", help="list the phrases that were not found")
    args = ap.parse_args()

    engines = args.engines or oe.available_engines()
    items = load()
    print(f"{len(items)} pictures, engines: {', '.join(engines)}")
    if args.show is not None:
        for k, v, path in items:
            if k == args.show:
                for b in oe.read_screen(Image.open(path), engines, args.scales, args.variants):
                    print(f"{b.conf:.2f} [{b.engine}/{b.variant}] ({b.x0:.0f},{b.y0:.0f})-({b.x1:.0f},{b.y1:.0f}) {b.text}")
                print("expected:", v["text"])
        return 0
    combos = []
    names = args.variants or ["raw", "stretch", "invert"]
    for en in engines:
        for sc in args.scales:
            for vn in names:
                combos.append(([en], [sc], [vn]))
            if len(names) > 1:
                combos.append(([en], [sc], names))
    print(f"{'engine':8} {'scale':>5} {'versions':22} {'recall':>7} {'menu':>6} {'s/pic':>6}")
    for en, sc, vn in combos:
        r, m, dt, miss = score(items, en, sc, vn)
        print(f"{en[0]:8} {sc[0]:>5} {'+'.join(vn):22} {100 * r:6.1f}% {100 * m:5.0f}% {dt:6.2f}", flush=True)
        if args.misses and len(vn) == len(names):
            for k, name, ph in miss:
                print(f"      missed #{k} {name}: {ph}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
