#!/usr/bin/env python3
"""Game status report: how far every ROM gets, for the compatibility page.

    python tools/game_status.py [--filter text] [--list tools/test_set.txt] [--publish]

The statuses come from the QA tool (tools/qa.py, see tools/QA.md): the recorded route of every game
(tools/routes/index.json: how far it gets and how) and the last `qa.py run` (test_output/qa/results.json:
a crash or a freeze while the route was played, the screen the OCR reader saw at its end). Nothing is played
here, so the page can be rebuilt in seconds: run `python tools/qa.py record` (once) and `python tools/qa.py run`
first, so that the data is current.

    python tools/game_status.py --probe-runs [--jobs 6] [--fresh]   # the older way: every game in bin/game_probe again

With --probe-runs each ROM is played by bin/game_probe (tools/game_probe.cpp) on its own, from save states, and
classified from its measurements. From either source every game gets exactly one status:

    BLACK_SCREEN  nothing (or only black) was ever shown
    INTRO_TITLE   logos, intro, title screen, attract demo: no control yet
    MENU          menus: the game reacts to buttons / cursor, not gameplay
    INGAME        gameplay: the stick moves the character or the camera
    CRASH_ERROR   the emulator crashed or timed out, the game froze, or
                  the picture is broken

plus a finer `detail` (e.g. "title screen", "freeze", "playable").

The automatic verdict can be overruled after looking at the screenshots:
tools/game_status_review.json maps a ROM's CRC ("CRC1-CRC2", see results.json)
to {"status": ..., "detail": ..., "note": ...}. The review sheets in
review/ show every game's pictures next to its automatic verdict for that.

Output (test_output/game_status/ unless --out):
    results.json         everything, per ROM image (identical dumps once)
    review/sheet_NN.png  screenshots + verdicts, 10 games per sheet
    web/                 index.html + img/ - a self-contained page for the site
    games/<slug>/        the probe's raw output (probe.json, PNGs, log)

Needs Pillow. Save files next to the ROMs are neither read nor written.
"""
import argparse
import concurrent.futures as cf
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "bin", "game_probe.exe" if os.name == "nt" else "game_probe")
REVIEW_FILE = os.path.join(ROOT, "tools", "game_status_review.json")

STATUSES = ["INGAME", "MENU", "INTRO_TITLE", "BLACK_SCREEN", "CRASH_ERROR"]
REACT = 0.004     # share of the picture that must differ for "the game reacted"
BIG_STICK = 0.10  # the stick moving this much of the picture: camera / character


# ---------------------------------------------------------------------------
# ROMs

def rom_header(path):
    """(CRC1, CRC2, internal name) from the header, whatever the byte order."""
    with open(path, "rb") as f:
        h = bytearray(f.read(0x40))
    if len(h) < 0x40:
        return None
    if h[:4] == b"\x37\x80\x40\x12":  # .v64: 16-bit byte swapped
        for i in range(0, 0x40, 2):
            h[i], h[i + 1] = h[i + 1], h[i]
    elif h[:4] == b"\x40\x12\x37\x80":  # .n64: 32-bit little endian
        for i in range(0, 0x40, 4):
            h[i:i + 4] = h[i:i + 4][::-1]
    elif h[:4] != b"\x80\x37\x12\x40":
        return None
    crc = "%08X-%08X" % (int.from_bytes(h[0x10:0x14], "big"), int.from_bytes(h[0x14:0x18], "big"))
    name = h[0x20:0x34].decode("ascii", "replace").strip("\x00 ")
    return crc, name


def find_roms(filt, listed):
    """One entry per distinct ROM image: identical dumps (same CRCs) run once."""
    by_crc = {}
    for d, _, files in os.walk(os.path.join(ROOT, "roms")):
        for f in sorted(files, key=str.lower):
            if not f.lower().endswith((".z64", ".n64", ".v64")):
                continue
            if filt and filt.lower() not in f.lower():
                continue
            if listed is not None and f not in listed:
                continue
            path = os.path.join(d, f)
            hdr = rom_header(path)
            if not hdr:
                print(f"skipping {f}: not an N64 ROM", file=sys.stderr)
                continue
            crc, internal = hdr
            e = by_crc.setdefault(crc, {"crc": crc, "internal": internal, "files": []})
            e["files"].append(os.path.relpath(path, ROOT))
    roms = list(by_crc.values())
    for e in roms:
        # Prefer the GoodN64 "[!]" (verified) name for display.
        e["files"].sort(key=lambda p: (0 if "[!]" in p or "(!)" in p else 1, len(p)))
        e["rom"] = e["files"][0]
        e["slug"] = slug(e["rom"])
        e.update(display_name(os.path.basename(e["rom"])))
    return sorted(roms, key=lambda e: e["name"].lower())


def slug(path):
    return re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(os.path.basename(path))[0]).strip("_")[:60]


REGIONS = {"U": "USA", "E": "Europe", "J": "Japan", "JU": "Japan/USA", "UE": "USA/Europe", "A": "Australia",
           "G": "Germany", "F": "France", "S": "Spain", "I": "Italy", "B": "Brazil", "USA": "USA",
           "Europe": "Europe", "Japan": "Japan"}


def display_name(fname):
    """'Legend of Zelda, The - Ocarina of Time (U) (V1.2) [!].z64' -> name, region, version."""
    base = os.path.splitext(fname)[0]
    tags = re.findall(r"\(([^)]*)\)", base)
    name = re.sub(r"\s*[\(\[].*$", "", base).strip()
    # GoodN64 file names write the apostrophe as an underscore: Yoshi_s Story, Cruis_n USA, Goin_ Quackers, _99.
    name = re.sub(r"(?<=\w)_(?=[A-Za-z](?![A-Za-z]))|_(?=\s)|(?<=\s)_(?=\d)", "'", name).replace("_", " ")
    m = re.match(r"^(.*), (The|A|An)( - .*)?$", name)  # "Legend of Zelda, The - X" -> "The Legend of Zelda - X"
    if m:
        name = f"{m.group(2)} {m.group(1)}{m.group(3) or ''}"
    region = next((REGIONS[t] for t in tags if t in REGIONS), "")
    # "(V1.2)" (GoodN64) or "(Rev 1)" (No-Intro): the revisions of one game are different dumps and the page lists each
    version = next((t for t in tags if re.match(r"^V\d", t) or re.match(r"^Rev ?\w", t)), "")
    return {"name": name, "region": region, "version": version}


# ---------------------------------------------------------------------------
# Probing

# Exit codes of a process stopped from outside: Ctrl+C / taskkill on Windows, SIGINT / SIGTERM / SIGKILL elsewhere.
INTERRUPTED = {0xC000013A, 0xFFFFFFFF, -1, -2, -9, -15, 1, 130, 137, 143}  # 1: TerminateProcess


def as_done(futs):
    """Like as_completed, but waits in short slices: a blocking wait isn't interrupted by Ctrl+C on Windows."""
    remaining = set(futs)
    while remaining:
        done, remaining = cf.wait(remaining, timeout=0.5, return_when=cf.FIRST_COMPLETED)
        yield from done


STOP = threading.Event()   # set on Ctrl+C: probes that get killed then are not results
_children, _children_lock = set(), threading.Lock()


def spawn(cmd, timeout):
    """Runs a probe; returns (exit code, stdout, stderr, timed out). Tracked, so Ctrl+C can kill it."""
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    with _children_lock:
        _children.add(p)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err, False
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return None, out or "", err or "", True
    finally:
        with _children_lock:
            _children.discard(p)


def kill_children():
    with _children_lock:
        for p in list(_children):
            p.kill()


def run_probe(e, args, outdir, mode="hle"):
    gdir = os.path.join(outdir, "games", e["slug"], mode)
    # run.json is written when the probe process has ended (crashes and
    # timeouts included), so a game that has it is done; a game that was
    # cut off in the middle has none and runs again.
    if not args.fresh and os.path.exists(os.path.join(gdir, "run.json")):
        return load_run(e, gdir)
    # The run goes into a work folder that replaces the old result only when it
    # ended properly: a run killed from outside must not destroy what was there.
    work = gdir + ".new"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    cmd = [EXE, os.path.join(ROOT, e["rom"]), "--out", work, "--frames", str(args.frames),
           "--max-seconds", str(args.max_seconds), "--rsp", mode]
    t0 = time.time()
    rc, out, err, timed_out = spawn(cmd, args.max_seconds + 150)
    # The core prints a lot to stderr; keep the start and the end.
    err_lines = err.splitlines()
    if len(err_lines) > 400:
        err_lines = err_lines[:200] + [f"... {len(err_lines) - 400} lines ..."] + err_lines[-200:]
    with open(os.path.join(work, "log.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + f"\n\nexit={rc} timeout={timed_out} wall={time.time() - t0:.1f}s\n\n--- stdout ---\n"
                + out + "\n--- stderr ---\n" + "\n".join(err_lines))
    # A probe that was stopped from outside (Ctrl+C, taskkill) isn't a result:
    # no run.json, so the game runs again next time.
    if rc in INTERRUPTED or STOP.is_set():
        shutil.rmtree(work, ignore_errors=True)
        return load_run(e, gdir)
    with open(os.path.join(work, "run.json"), "w") as f:
        json.dump({"rc": rc, "timeout": timed_out, "wall_s": round(time.time() - t0, 1)}, f)
    shutil.rmtree(gdir, ignore_errors=True)
    os.replace(work, gdir)
    return load_run(e, gdir)


def load_run(e, gdir):
    r = dict(e)
    r["dir"] = gdir
    r["mode"] = os.path.basename(gdir)
    try:
        with open(os.path.join(gdir, "run.json")) as f:
            r.update(json.load(f))
    except OSError:
        r.update(rc=None, timeout=False, wall_s=0)
    try:
        with open(os.path.join(gdir, "probe.json")) as f:
            r["probe"] = json.load(f)
    except (OSError, ValueError):
        r["probe"] = None
    return r


# ---------------------------------------------------------------------------
# Classification

def classify(r):
    """(status, detail, confidence, reasons) from the probe's measurements."""
    p = r["probe"]
    if r.get("timeout"):
        return "CRASH_ERROR", "emulator timeout", "high", ["the probe did not finish in time"]
    if p is None or r.get("rc") not in (0, None):
        return "CRASH_ERROR", "emulator crash", "high", [f"exit code {r.get('rc')}, no result"]
    fin, fr, tl = p["final"], p["final_react"], p["timeline"]
    react_a, react_s = fr["a"] > REACT, fr["start"] > REACT
    stick = max(fr["stick"], fr.get("stick_with_a", 0))
    react_k = stick > REACT
    reacts = react_a or react_s or react_k
    why = [f"final reaction: A {fr['a']:.1%}, START {fr['start']:.1%}, stick {stick:.1%} of the picture"]
    content = [t for t in tl if t.get("colors", 0) >= 6] + ([1] if p["boot"]["colors"] >= 6 else [])
    black = fin["lit"] < 0.01 and fin["colors"] <= 4
    still = fin["moving"] == 0
    frozen = not reacts and still and fin["gfx_tasks"] == 0 and fin.get("vi_swaps", 0) == 0

    # No display list at all: black, unless the game draws with the CPU (Namco Museum 64).
    if p["display_lists"] == 0 and not content:
        return "BLACK_SCREEN", "no display list", "high", why + ["the game never sent a display list"]
    if not content and (black or fin["colors"] < 6):
        d = "nothing drawn, game stopped" if frozen else "nothing drawn"
        return "BLACK_SCREEN", d, "high", why + ["no picture with content during the whole run"]
    if frozen:
        why.append(f"last {fin['frames']} frames: no graphics task, no change, no reaction")
        return "CRASH_ERROR", "freeze", "high", why
    if black:
        if reacts:
            return "BLACK_SCREEN", "black picture, game reacts", "medium", why
        return "BLACK_SCREEN", "went black", "medium", why + ["pictures earlier, black at the end"]
    tris = fin["triangles"] / max(fin["frames"], 1)
    if stick >= BIG_STICK:
        conf = "high" if p.get("ingame_frame", -1) > 0 or fr.get("stick_with_a", 0) >= BIG_STICK else "medium"
        return "INGAME", "playable", conf, why + [f"the stick moves {stick:.0%} of the picture"]
    if react_k:
        if tris > 600 and stick >= 0.03:
            return "INGAME", "playable", "low", why + [f"{tris:.0f} triangles per frame, small stick reaction"]
        return "MENU", "cursor moves", "medium", why + ["the stick changes a small part of the picture"]
    presses = p.get("presses_a", 0) + p.get("presses_start", 0)
    if react_a or react_s:
        if presses == 0:
            return "INTRO_TITLE", "title screen", "medium", why + ["first screen that reacts to a button"]
        return "MENU", "reacts to buttons", "low", why + [f"{presses} presses so far, the stick does nothing"]
    if fin["moving"] > 0.2:
        return "INTRO_TITLE", "intro / demo (no input)", "low", why + ["moving picture, no reaction to input"]
    return "INTRO_TITLE", "static screen (no input)", "low", why + ["still picture, no reaction to input"]


def load_review():
    try:
        with open(REVIEW_FILE, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return {}


MODES = ["hle", "lle-gfx"]
MODE_LABEL = {"hle": "HLE", "lle-gfx": "LLE"}
RANK = {"INGAME": 4, "MENU": 3, "INTRO_TITLE": 2, "BLACK_SCREEN": 1, "CRASH_ERROR": 0}


def combine(e, runs):
    """One result from the runs of one ROM in the different RSP/RDP modes.

    The best status wins; on a tie the high-level mode (faster, and what
    most people run). `modes` keeps every mode's own verdict for the report.
    """
    scored = []
    for m in MODES:
        if m in runs:
            st, det, conf, why = classify(runs[m])
            scored.append((RANK[st], m == "hle", m, st, det))
    best = max(scored, key=lambda t: (t[0], t[1]))
    r = dict(runs[best[2]])
    r["modes"] = {m: {"status": st, "detail": det} for _, _, m, st, det in scored}
    if len(scored) == 1:
        r["works_in"] = [best[2]] if best[3] == "INGAME" else []
    else:
        top = max(t[0] for t in scored)
        r["works_in"] = [m for rk, _, m, _, _ in scored if rk == top] if top == RANK["INGAME"] else [best[2]]
    return r


def finish(r, review):
    st, det, conf, why = classify(r)
    r["auto"] = {"status": st, "detail": det, "confidence": conf, "reasons": why}
    r["needs_lle"] = r["mode"] != "hle" and r["modes"].get("hle", {}).get("status") != st
    rv = review.get(r["crc"])
    if rv:
        r["status"], r["detail"] = rv["status"], rv.get("detail", det)
        r["note"] = rv.get("note", "")
        r["source"] = "review"
    else:
        r["status"], r["detail"], r["note"], r["source"] = st, det, "", "auto"
    p = r["probe"] or {}
    r["shots"] = pick_shots(r)
    if p:
        r["seconds_played"] = round(p.get("frames", 0) / 60, 1)
        r["probe_fps"] = round(p.get("emulated_frames", 0) / max(p.get("seconds", 1), 1e-3))
    return r


def pick_shots(r):
    """Screenshots for the pages: boot, a few along the way, the end."""
    p = r["probe"]
    if not p:
        return []
    have = set(p.get("shots", []))
    tl = sorted((s for s in have if re.match(r"^t\d+\.png$", s)), key=lambda s: int(s[1:-4]))
    picks = []
    if "boot.png" in have:
        picks.append("boot.png")
    if tl:
        n = len(tl)
        for i in sorted({n // 4, n // 2, (3 * n) // 4}):
            picks.append(tl[min(i, n - 1)])
    for s in ("first_play.png", "final.png", "final_stick.png"):
        if s in have:
            picks.append(s)
    return picks


# ---------------------------------------------------------------------------
# Review sheets

COLORS = {"INGAME": (46, 160, 67), "MENU": (56, 120, 220), "INTRO_TITLE": (150, 100, 220),
          "BLACK_SCREEN": (120, 120, 120), "CRASH_ERROR": (210, 50, 50)}


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def review_sheets(results, outdir, per_sheet=10):
    d = os.path.join(outdir, "review")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    TW, TH, LW = 160, 120, 330
    f1, f2 = font(14), font(12)
    for s in range(0, len(results), per_sheet):
        chunk = results[s:s + per_sheet]
        im = Image.new("RGB", (LW + 7 * (TW + 4), len(chunk) * (TH + 8)), (24, 24, 28))
        dr = ImageDraw.Draw(im)
        for i, r in enumerate(chunk):
            y = i * (TH + 8)
            a = r["auto"]
            dr.text((6, y + 4), f"#{s + i + 1} {r['name']}"[:44], font=f1, fill=(255, 255, 255))
            dr.text((6, y + 22), f"{r['region']} {r['version']}  {r['crc']}", font=f2, fill=(170, 170, 170))
            dr.rectangle((6, y + 40, 16, y + 50), fill=COLORS[a["status"]])
            dr.text((22, y + 38), f"{MODE_LABEL[r['mode']]}: {a['status']} / {a['detail']} ({a['confidence']})",
                    font=f2, fill=(255, 255, 255))
            if len(r["modes"]) > 1:
                dr.text((170, y + 22), "  ".join(f"{MODE_LABEL[m]}={v['status'][:6]}" for m, v in r["modes"].items()),
                        font=f2, fill=(120, 200, 255))
            if r["source"] == "review":
                dr.text((22, y + 54), f"review: {r['status']} / {r['detail']}", font=f2, fill=(255, 220, 90))
            p = r["probe"] or {}
            fr = p.get("final_react", {})
            if fr:
                dr.text((6, y + 72), "A %.1f%%  START %.1f%%  stick %.1f%% / %.1f%%" % (
                    100 * fr["a"], 100 * fr["start"], 100 * fr["stick"], 100 * fr.get("stick_with_a", 0)),
                    font=f2, fill=(170, 170, 170))
                dr.text((6, y + 88), "played %ss, A x%d START x%d, ingame@%s" % (
                    r.get("seconds_played"), p.get("presses_a", 0), p.get("presses_start", 0),
                    p.get("ingame_frame")), font=f2, fill=(170, 170, 170))
            for k, shot in enumerate(r["shots"][-7:]):
                try:
                    t = Image.open(os.path.join(r["dir"], shot)).convert("RGB").resize((TW, TH))
                except OSError:
                    continue
                x = LW + k * (TW + 4)
                im.paste(t, (x, y + 4))
                dr.text((x + 2, y + 4), shot[:-4], font=f2, fill=(255, 255, 0))
        im.save(os.path.join(d, "sheet_%02d.png" % (s // per_sheet + 1)))


# ---------------------------------------------------------------------------
# Web page

LABELS = {"INGAME": "In-game", "MENU": "Menu", "INTRO_TITLE": "Intro / title", "BLACK_SCREEN": "Black screen",
          "CRASH_ERROR": "Crash / error"}


def web_shots(r):
    """The pictures the page uses: [0] is the big one, the rest a strip below it."""
    have = {x: os.path.join(r["dir"], x) for x in r["shots"] if os.path.exists(os.path.join(r["dir"], x))}
    if not have:
        return []
    tl = sorted((x for x in have if re.match(r"^t\d+\.png$", x)), key=lambda x: int(x[1:-4]))
    main = "final.png" if "final.png" in have else (tl[-1] if tl else next(iter(have)))
    strip = []
    for x in ("boot.png", tl[len(tl) // 2] if tl else None, "final_stick.png"):
        if x and x in have and x != main and x not in strip:
            strip.append(x)
    return [main] + strip


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def web_report(results, outdir, args):
    """Writes the page: test_output/game_status/web, or site/compatibility with --publish."""
    d = os.path.join(ROOT, "site", "compatibility") if args.publish else os.path.join(outdir, "web")
    img = os.path.join(d, "img")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(img)
    for f in ("logo.svg", "favicon.ico", "icon.svg"):  # the page is self-contained (also next to the site's own files)
        shutil.copy(os.path.join(ROOT, "site", f), d)
    games = []
    for r in results:
        shots = []
        for s in web_shots(r):
            dst = f"{r['slug']}__{s[:-4]}.webp"
            full = f"{r['slug']}__{s[:-4]}_full.webp"
            im = Image.open(os.path.join(r["dir"], s)).convert("RGB")
            big = im.copy()
            big.thumbnail((640, 480))  # for the screenshot viewer
            big.save(os.path.join(img, full), "WEBP", quality=82, method=6)
            im.thumbnail((256, 192))
            im.save(os.path.join(img, dst), "WEBP", quality=70, method=6)
            shots.append({"src": "img/" + dst, "full": "img/" + full, "label": shot_label(s)})
        games.append({"name": r["name"], "region": r["region"], "version": r["version"], "crc": r["crc"],
                      "internal": r["internal"], "files": [os.path.basename(f) for f in r["files"]],
                      "status": r["status"], "detail": r["detail"], "note": r["note"], "source": r["source"],
                      "mode": MODE_LABEL[r["mode"]], "modes": {MODE_LABEL[m]: v for m, v in r["modes"].items()},
                      "shots": shots})
    counts = {s: sum(1 for g in games if g["status"] == s) for s in STATUSES}
    data = {"generated": time.strftime("%Y-%m-%d"), "build": git_head(), "games": games, "counts": counts,
            "lle_tried": sum(1 for g in games if "LLE" in g["modes"]),
            "lle_gain": sum(1 for g in games if g["mode"] == "LLE" and g["status"] == "INGAME")}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(d, "compat.json"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(PAGE_FILE, encoding="utf-8") as f:
        page = f.read()
    with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
        label = {"INGAME": "in-game", "MENU": "menu", "INTRO_TITLE": "intro / title", "BLACK_SCREEN": "black screen",
                 "CRASH_ERROR": "crash / error"}
        # for readers without JavaScript (and crawlers that do not run it): every game as plain text
        static = "<ul>" + "".join("<li>%s%s (%s) - %s</li>" % (html.escape(g["name"]), " " + html.escape(g["version"]) if g["version"] else "",
                                                               html.escape(g["region"]), label[g["status"]]) for g in games) + "</ul>"
        f.write(page.replace("<!--STATIC-->", static).replace("/*DATA*/", text.replace("</", "<\\/")))
    print(f"page: {d} ({sum(os.path.getsize(os.path.join(img, x)) for x in os.listdir(img)) / 1e6:.1f} MB of images)")


def shot_label(s):
    if s == "boot.png":
        return "start"
    if s == "final.png":
        return "end of run"
    if s == "final_stick.png":
        return "stick held"
    if s == "first_play.png":
        return "first reaction to the stick"
    m = re.match(r"^t(\d+)\.png$", s)
    if s.startswith("hle_"):
        return "during play"
    if s == "route_end.png":
        return "end of the route"
    return f"{int(m.group(1)) / 60:.0f} s" if m else s


PAGE_FILE = os.path.join(ROOT, "tools", "game_status_page.html")


# ---------------------------------------------------------------------------
# Status from the QA tool (tools/qa.py): the recorded routes and the last `qa.py run`

QA_DIR = os.path.join(ROOT, "test_output", "qa")
ROUTES_INDEX = os.path.join(ROOT, "tools", "routes", "index.json")
# What the OCR reader (tools/screen_reader.py) saw on the last screen of the route.
MENU_KINDS = {"menu", "options", "prompt", "pause", "dialog", "game_over", "error_dialog", "warning", "credits"}
TITLE_KINDS = {"press_start", "press_a", "legal", "no_text", "unknown"}


def _load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def classify_qa(route, g, probe):
    """(status, detail, confidence, reasons) from a game's recorded route and its last QA run.

    route: tools/routes/index.json entry ({} when none), g: the game's entry in test_output/qa/results.json,
    probe: the probe's own measurements of the recording (probe.json) or None."""
    codes = {x["code"]: x for x in (g or {}).get("reasons", [])}
    reached = (route or {}).get("reached", "")
    kind = ((g or {}).get("screen") or {}).get("kind", "")
    m = (g or {}).get("metrics", {})
    why = []
    if reached:
        why.append(f"recorded route reaches '{reached}' in {route.get('frames', 0)} frames")
    # The emulator crashed or froze while the route was played.
    if "cpu_lost" in codes:
        return "CRASH_ERROR", "crash", "high", why + [codes["cpu_lost"]["text"]]
    if "stalled" in codes:
        return "CRASH_ERROR", "freeze", "high", why + [codes["stalled"]["text"]]
    if not reached:
        p = probe or {}
        if p and p.get("display_lists", 1) == 0:
            return "BLACK_SCREEN", "no display list", "high", ["the game never sent a display list"]
        failed = (route or {}).get("failed", "")
        if failed == "timeout":
            return "CRASH_ERROR", "emulator timeout", "medium", ["the probe did not get anywhere in time"]
        return "BLACK_SCREEN", "nothing drawn", "medium", ["no route could be recorded: " + (failed or "no data")]
    if m and m.get("lit", 1) < 0.01 and m.get("colors", 99) <= 4:
        return "BLACK_SCREEN", "went black", "medium", why + ["the picture at the end of the route is black"]
    verified = (g or {}).get("status") in ("ok", "warn") and "route_lost" not in codes
    conf = "high" if verified else "medium"
    if reached == "ingame" or kind == "gameplay":
        return "INGAME", "playable", conf if reached == "ingame" else "low", why + ["the stick moves the picture"]
    if reached == "scene":
        if kind in MENU_KINDS:
            return "MENU", "menu over a 3D scene", "medium", why
        return "INTRO_TITLE", "scene / intro (no control confirmed)", "low", why
    # 'big' and 'furthest': the screen decides between a menu and a title
    if kind in MENU_KINDS:
        return "MENU", "menu" if kind != "game_over" else "game over screen", "medium", why + [f"the screen reads as '{kind}'"]
    if reached == "big":
        return "MENU", "reacts to the stick", "low", why + ["the stick changes a big part of the picture"]
    return "INTRO_TITLE", "title / intro screen" if kind in TITLE_KINDS else "intro", "low", why


def results_from_qa(roms, review):
    """The same result records as finish() makes, from the QA data instead of probe runs of their own."""
    index = _load_json(ROUTES_INDEX, {})
    qa = {x["crc"]: x for x in _load_json(os.path.join(QA_DIR, "results.json"), {}).get("games", [])}
    results = []
    for e in roms:
        g = qa.get(e["crc"])
        route = index.get(e["crc"], {})
        if not g and not route:
            continue
        rec = os.path.join(QA_DIR, "record", e["crc"])
        probe = _load_json(os.path.join(rec, "probe.json"), None)
        st, det, conf, why = classify_qa(route, g, probe)
        runs = os.path.join(QA_DIR, "runs", e["crc"])
        names = ["route_end.png"] + [f"hle_{i:02d}.png" for i in (2, 5, 8)]
        shots = [s for s in names if os.path.exists(os.path.join(runs, s))]
        d = runs
        if not shots:  # no QA run of this game: the probe's own pictures from the recording
            d = rec
            have = set((probe or {}).get("shots", []))
            shots = [s for s in ("final.png", "boot.png", "first_play.png") if s in have]
        r = dict(e)
        r.update(dir=d, mode="hle", rc=0, timeout=False, wall_s=(route or {}).get("seconds", 0), probe=probe, shots=shots,
                 modes={"hle": {"status": st, "detail": det}}, works_in=["hle"] if st == "INGAME" else [])
        r["auto"] = {"status": st, "detail": det, "confidence": conf, "reasons": why}
        rv = review.get(e["crc"])
        if rv:
            r["status"], r["detail"], r["note"], r["source"] = rv["status"], rv.get("detail", det), rv.get("note", ""), "review"
        else:
            r["status"], r["detail"], r["note"], r["source"] = st, det, "", "auto"
        results.append(r)
    return results


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--filter", default="")
    ap.add_argument("--list", help="only ROM file names listed in this file (tools/test_set.txt)")
    ap.add_argument("--frames", type=int, default=4200, help="game time to play after boot (frames)")
    ap.add_argument("--max-seconds", type=float, default=420, help="wall time per game (the probe stops exploring at 70%%)")
    ap.add_argument("--out", default=os.path.join(ROOT, "test_output", "game_status"))
    ap.add_argument("--fresh", action="store_true", help="run everything again (default: continue, skipping games that are done)")
    ap.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)  # the default now
    ap.add_argument("--both", action="store_true", help="run every game in LLE too (default: only those HLE doesn't get into the game)")
    ap.add_argument("--redo", default="", help="repeat the HLE runs of games whose status is one of these, e.g. MENU,INTRO_TITLE; "
                    "CUT = games (not in-game) whose run hit the time limit")
    ap.add_argument("--redo-list", help="run the games named in this file again, in every mode (tools/fix_list.txt: the games "
                    "that didn't get into the game, after a fix); the rest of the results stay as they are")
    ap.add_argument("--hle-only", action="store_true", help="skip the LLE runs")
    ap.add_argument("--publish", action="store_true", help="write the page to site/compatibility/ (GitHub Pages) instead of test_output")
    ap.add_argument("--report-only", action="store_true", help="don't run anything, rebuild the reports")
    ap.add_argument("--probe-runs", action="store_true", help="the old way: run every game in the probe again (default: the status "
                    "comes from the QA tool's routes and last run, tools/qa.py)")
    args = ap.parse_args()

    listed = None
    if args.list:
        with open(args.list, encoding="utf-8") as f:
            listed = {l.split("#", 1)[0].strip() for l in f if l.split("#", 1)[0].strip()}
    roms = find_roms(args.filter, listed)
    os.makedirs(args.out, exist_ok=True)
    review = load_review()

    runs = {e["crc"]: {} for e in roms}  # crc -> mode -> run
    use_qa = not args.probe_runs
    if use_qa:
        if not os.path.exists(ROUTES_INDEX):
            sys.exit("no routes yet: python tools/qa.py record (or use --probe-runs)")
    elif not args.report_only:
        if not os.path.exists(EXE):
            sys.exit(f"{EXE} is missing: make game_probe")
        t0 = time.time()
        if args.redo_list:
            with open(args.redo_list, encoding="utf-8") as f:
                names = {l.split("#", 1)[0].strip() for l in f if l.split("#", 1)[0].strip()}
            again_n = 0
            for e in roms:
                if names & {os.path.basename(x) for x in e["files"]}:
                    for m in MODES:
                        rj = os.path.join(args.out, "games", e["slug"], m, "run.json")
                        if os.path.exists(rj):
                            os.remove(rj)
                    again_n += 1
            print(f"--redo-list: {again_n} of {len(names)} listed games will run again", flush=True)
        wanted = {x.strip().upper() for x in args.redo.split(",") if x.strip()}
        if wanted:
            redone = 0
            for e in roms:
                gd = os.path.join(args.out, "games", e["slug"], "hle")
                rj = os.path.join(gd, "run.json")
                if not os.path.exists(rj):
                    continue
                run = load_run(e, gd)
                st = classify(run)[0]
                # CUT: the probe ran out of time before the game was played through.
                cut = "CUT" in wanted and st != "INGAME" and (run["probe"] or {}).get("out_of_time")
                if st in wanted or cut:
                    os.remove(rj)
                    redone += 1
            print(f"--redo {','.join(sorted(wanted))}: {redone} HLE runs will be repeated", flush=True)

        def phase(title, todo, mode):
            def gdir(e):
                return os.path.join(args.out, "games", e["slug"], mode)

            # Games with a run.json are done: continue where the last run stopped.
            ready = [e for e in todo if not args.fresh and os.path.exists(os.path.join(gdir(e), "run.json"))]
            pending = [e for e in todo if e not in ready]
            for e in ready:
                runs[e["crc"]][mode] = load_run(e, gdir(e))
            total, finished = len(todo), len(ready)
            print(f"{title}: {total} games, {finished} already done, {len(pending)} to run ({args.jobs} at a time)", flush=True)
            if not pending:
                return
            t1 = time.time()
            ex = cf.ThreadPoolExecutor(args.jobs)
            try:
                futs = [ex.submit(run_probe, e, args, args.out, mode) for e in pending]
                for n, fu in enumerate(as_done(futs), 1):
                    r = fu.result()
                    if not os.path.exists(os.path.join(r["dir"], "run.json")):
                        continue  # stopped from outside: not a result, it runs again next time
                    runs[r["crc"]][mode] = r
                    st = classify(r)
                    finished += 1
                    el = time.time() - t1
                    eta = f", ~{el / n * (len(pending) - n) / 60:.0f} min left" if n >= 3 else ""
                    print(f"[{mode} {finished}/{total}, {el / 60:.0f} min{eta}] {r['name']}: {st[0]} / {st[1]}", flush=True)
                ex.shutdown()
            except KeyboardInterrupt:
                STOP.set()
                ex.shutdown(wait=False, cancel_futures=True)
                kill_children()
                print(f"\nStopped: {finished}/{total} {mode} games are done. Run the same command again to continue.", flush=True)
                os._exit(130)

        # 1. everything on the high-level emulation.
        phase("HLE", roms, "hle")
        # 2. what HLE doesn't get into the game runs on the low-level RSP/RDP
        #    too (all games with --both), and the better result counts.
        def worth_lle(e):
            r = runs[e["crc"]]["hle"]
            if args.both:
                return True
            if classify(r)[0] == "INGAME":
                return False
            # A game that never started a graphics task stalls before the RSP
            # is involved: LLE (RSP + RDP) can't change that.
            p = r["probe"] or {}
            return not (p and p.get("display_lists", 1) == 0 and p.get("gfx_ucode_tasks", 1) == 0)

        again = [e for e in roms if worth_lle(e)]
        skipped = sum(1 for e in roms if not worth_lle(e) and classify(runs[e["crc"]]["hle"])[0] != "INGAME")
        if skipped:
            print(f"LLE skipped for {skipped} games that never started a graphics task in HLE", flush=True)
        if not args.hle_only and again:
            phase("LLE", again, "lle-gfx")
    else:
        for e in roms:
            for m in MODES:
                gdir = os.path.join(args.out, "games", e["slug"], m)
                if os.path.exists(os.path.join(gdir, "run.json")):
                    runs[e["crc"]][m] = load_run(e, gdir)
    if use_qa:
        results = results_from_qa(roms, review)
    else:
        results = [combine(e, runs[e["crc"]]) for e in roms if runs[e["crc"]]]
        results = [finish(r, review) for r in results]
    results.sort(key=lambda r: r["name"].lower())

    slim = []
    for r in results:
        p = r["probe"] or {}
        slim.append({k: r[k] for k in ("name", "region", "version", "crc", "internal", "files", "status", "detail",
                                       "note", "source", "auto", "shots", "rc", "timeout", "wall_s", "mode", "modes", "works_in")}
                    | {"seconds_played": r.get("seconds_played"), "probe_fps": r.get("probe_fps"),
                       "dir": os.path.relpath(r["dir"], args.out).replace("\\", "/"),
                       "final_react": p.get("final_react"), "ingame_frame": p.get("ingame_frame"),
                       "presses_a": p.get("presses_a"), "presses_start": p.get("presses_start")})
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=1, ensure_ascii=False)
    review_sheets(results, args.out)
    web_report(results, args.out, args)

    counts = {s: sum(1 for r in results if r["status"] == s) for s in STATUSES}
    print("\n" + "  ".join(f"{s}={n}" for s, n in counts.items()))
    print(f"better in LLE than in HLE: {sum(1 for r in results if r['mode'] != 'hle')}")
    print(f"reviewed: {sum(1 for r in results if r['source'] == 'review')}/{len(results)}")
    print(f"-> {os.path.join(args.out, 'web', 'index.html')}")


if __name__ == "__main__":
    main()
