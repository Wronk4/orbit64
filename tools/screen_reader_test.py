#!/usr/bin/env python3
"""Does tools/screen_reader.py decide sensibly? Runs it on the pictures in tools/ocr_bench/ and compares with
tools/ocr_bench_decisions.json: the kind of screen, and whether one of the first candidates presses an acceptable
button (and, where given, aims at the right option).

    python tools/screen_reader_test.py [-v]
"""
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_reader as sr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = {sr.A: "A", sr.START: "START", sr.B: "B"}


def final_button(macro):
    """The last button a macro presses."""
    for buttons, _, _, _ in reversed(macro):
        for mask, name in FINAL.items():
            if buttons & mask:
                return name
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    verbose = "-v" in sys.argv
    truth = json.load(open(os.path.join(HERE, "ocr_bench_decisions.json"), encoding="utf-8"))
    ok_kind = ok_press = n = 0
    for k, exp in truth.items():
        if k.startswith("_"):
            continue
        path = os.path.join(HERE, "ocr_bench", (k if k.startswith("dec_") else f"{int(k):02d}") + ".png")
        if not os.path.exists(path):
            continue
        n += 1
        adv = sr.advise(Image.open(path))
        kind_ok = adv.kind in exp["kind"]
        top = adv.candidates[:3]
        if exp["press"]:
            press_ok = any(final_button(c.macro) in exp["press"] and (not exp.get("target") or exp["target"].lower() in c.name.lower() or c.name in ("A",)
                                                                       and adv.options and adv.selected is not None
                                                                       and exp["target"].lower() in adv.options[adv.selected].text.lower())
                           for c in top)
        else:
            press_ok = not top or adv.kind == "gameplay" or all(c.score < 0.7 for c in top)
        if exp.get("notes"):
            press_ok = press_ok and bool(adv.notes)
        ok_kind += kind_ok
        ok_press += press_ok
        mark = "ok  " if kind_ok and press_ok else "FAIL"
        print(f"{mark} {k:18} kind {adv.kind:11} (want {'/'.join(exp['kind'])}) "
              f"first: {top[0].name if top else '-'}")
        if verbose or not (kind_ok and press_ok):
            print("       text:", " | ".join(adv.lines)[:200])
            for c in top:
                print(f"       {c.score:.2f} {c.name}: {c.why}")
    print(f"\nkind right in {ok_kind}/{n}, a sensible press among the first three in {ok_press}/{n}")
    return 0 if ok_kind == n and ok_press == n else 1


if __name__ == "__main__":
    sys.exit(main())
