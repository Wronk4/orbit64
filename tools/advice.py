"""Answers the prober's requests for advice (see the comment above ask_advice() in game_probe.cpp).

The prober saves the screen as <dir>/req_<n>.png and waits for <dir>/resp_<n>.txt. `serve(dir)` looks for such
requests, reads the screen (tools/screen_reader.py: OCR + rules) and writes the answer: what kind of screen it is
and short controller macros to try, best first. Everything that was asked and answered is kept in <dir>/log.jsonl.
"""
import json
import os
import time

from PIL import Image

import screen_reader as sr


def format_reply(adv) -> str:
    """The answer in the prober's text format."""
    lines = [f"kind {adv.kind}", f"conf {adv.confidence:.3f}"]
    for c in adv.candidates[:6]:
        segs = ";".join(f"{b:x},{x},{y},{f}" for b, x, y, f in c.macro)
        name = c.name.replace("\t", " ").replace("\n", " ")
        lines.append(f"cand {c.score:.3f}\t{name}\t{segs}")
    return "\n".join(lines) + "\n"


_SEEN = {}  # directory -> [(small picture, reply, entry)]: a screen that comes again is not read again


def _small(img):
    import numpy as np
    return np.asarray(img.convert("L").resize((48, 36), Image.BILINEAR), dtype=np.int16)


def answer(png_path: str):
    """(reply text, log entry) for one screen."""
    t0 = time.time()
    import numpy as np
    key = os.path.dirname(png_path)
    try:
        img = Image.open(png_path)
        img.load()
        small = _small(img)
        for sm, reply, entry in _SEEN.get(key, []):
            if float(np.abs(sm - small).mean()) < 2.0:
                e = dict(entry)
                e["cached"] = True
                e["seconds"] = round(time.time() - t0, 2)
                return reply, e
    except Exception:
        img = None
    try:
        adv = sr.advise(img if img is not None else Image.open(png_path))
    except Exception as e:  # a broken picture or engine must not stop the prober: it just gets no advice
        return "kind error\nconf 0\n", {"error": f"{type(e).__name__}: {e}"}
    entry = {"kind": adv.kind, "conf": round(adv.confidence, 3), "lines": adv.lines[:16], "selected": adv.selected,
             "options": [o.text for o in adv.options][:8], "notes": adv.notes[:6],
             "cands": [(round(c.score, 2), c.name) for c in adv.candidates[:4]], "seconds": round(time.time() - t0, 2)}
    reply = format_reply(adv)
    if img is not None:
        _SEEN.setdefault(key, []).append((small, reply, entry))
        del _SEEN[key][:-6]
    return reply, entry


def serve(dirpath: str) -> int:
    """Answers every request in the directory that has no answer yet; returns how many."""
    n = 0
    try:
        names = os.listdir(dirpath)
    except OSError:
        return 0
    for name in names:
        if not (name.startswith("req_") and name.endswith(".png") and not name.endswith(".tmp.png")):
            continue
        k = name[4:-4]
        resp = os.path.join(dirpath, f"resp_{k}.txt")
        if os.path.exists(resp):
            continue
        reply, entry = answer(os.path.join(dirpath, name))
        entry["n"] = int(k) if k.isdigit() else k
        tmp = resp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(reply)
        os.replace(tmp, resp)  # whole file or nothing
        with open(os.path.join(dirpath, "log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        n += 1
    return n
