#!/usr/bin/env python3
"""Compatibility sweep: runs every ROM in roms/ headless and reports how it did.

    python tools/compat_sweep.py [--frames 1800] [--jobs 4] [--filter text] [--cpu jit|interp]

For each ROM (roms/ and its subfolders) the emulator runs --frames frames with
a few START/A presses to get past title screens, taking screenshots along the
way and capturing the audio (--wav). Results go to test_output/compat/:

    report.html   one row per game: status, speed, microcodes, screenshots, audio
    sheet.png     every game's screenshots on one image
    results.json  the raw numbers, for comparing two sweeps (--compare old.json)

Needs Pillow and numpy. Save files are neither read nor written (--no-save).
"""
import argparse
import concurrent.futures as cf
import html
import json
import os
import re
import subprocess
import sys
import time
import wave

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = [240, 600, 900, 1200, 1500]  # frames (scaled with --frames)
PRESSES = [(480, "START"), (840, "START"), (900, "A"), (1080, "A"), (1260, "START"), (1320, "A")]


def find_roms(filt):
    roms = []
    for d, _, files in os.walk(os.path.join(ROOT, "roms")):
        for f in files:
            if f.lower().endswith((".z64", ".n64", ".v64")) and (not filt or filt.lower() in f.lower()):
                roms.append(os.path.join(d, f))
    return sorted(roms, key=lambda p: os.path.basename(p).lower())


def slug(path):
    return re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(os.path.basename(path))[0]).strip("_")[:48]


def analyse_image(path):
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    small = a[:: max(1, a.shape[0] // 60), :: max(1, a.shape[1] // 80)].reshape(-1, 3)
    colors = len({tuple(c) for c in small})
    return {"w": im.width, "h": im.height, "colors": colors, "mean": float(a.mean())}


def analyse_audio(path):
    if not os.path.exists(path) or os.path.getsize(path) <= 44:
        return {"seconds": 0.0, "rms_db": -240.0, "clicks_per_s": 0.0}
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64).reshape(-1, 2) / 32768.0
    if len(x) == 0:
        return {"seconds": 0.0, "rms_db": -240.0, "clicks_per_s": 0.0}
    mono = x.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono * mono)))
    # Clicks: second differences far above the local level (as tools/audio_report.py).
    d2 = np.abs(np.diff(mono, 2))
    win = max(64, rate // 100)
    step = win // 4
    pad = np.pad(d2, (win // 2, win // 2), mode="edge")
    centers = np.arange(0, len(d2), step)
    med = np.array([np.median(pad[c:c + win]) for c in centers]) if len(centers) else np.zeros(1)
    local = np.interp(np.arange(len(d2)), centers, med) if len(centers) else np.zeros(len(d2))
    hits = np.nonzero(d2 > np.maximum(local * 12.0, 0.02))[0]
    events = 0
    last = -10 ** 9
    for h in hits:
        if h - last > rate // 1000:
            events += 1
        last = h
    secs = len(x) / rate
    return {"seconds": secs, "rms_db": 20 * np.log10(rms + 1e-12), "clicks_per_s": events / max(secs, 1e-9),
            "rate": rate}


def run_one(rom, args, outdir):
    name = slug(rom)
    gdir = os.path.join(outdir, name)
    os.makedirs(gdir, exist_ok=True)
    scale = args.frames / 1800.0
    shots = [max(1, int(f * scale)) for f in SHOTS]
    cmd = [args.exe, rom, "--headless", str(args.frames), "--no-save", "--cpu", args.cpu,
           "--wav", os.path.join(gdir, "audio.wav")]
    for f in shots:
        cmd += ["--screenshot-at", f"{f}:{os.path.join(gdir, f'shot_{f:05d}.bmp')}"]
    for f, btn in PRESSES:
        cmd += ["--press", f"{int(f * scale)}:{btn}"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=args.timeout)
        rc, out, err, timed_out = p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        rc, timed_out = None, True
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    wall = time.time() - t0
    with open(os.path.join(gdir, "log.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n--- stdout ---\n" + out + "\n--- stderr ---\n" + err)

    r = {"rom": os.path.relpath(rom, ROOT), "name": name, "rc": rc, "timeout": timed_out, "wall_s": round(wall, 1)}
    m = re.search(r"core=\S+ frames=(\d+) time=([\d.]+)s \(([\d.]+) fps\) rdram_hash=(\w+)", out)
    if m:
        r.update(fps=float(m.group(3)), rdram_hash=m.group(4))
    m = re.search(r'\[Main\] info: (.*)', out)
    if m:
        for k, v in re.findall(r'(\w+)=("[^"]*"|\S+)', m.group(1)):
            r[k] = v.strip('"')
    # Messages the core prints about things it doesn't handle.
    unk = [l for l in (out + err).splitlines()
           if re.search(r"unknown|unimplemented|unhandled|unsupported|invalid", l, re.I)]
    r["warnings"] = len(unk)
    r["warning_samples"] = sorted(set(re.sub(r"0x[0-9a-fA-F]+|\d+", "#", l.strip())[:120] for l in unk))[:6]
    r["shots"] = []
    for f in shots:
        bmp = os.path.join(gdir, f"shot_{f:05d}.bmp")
        info = analyse_image(bmp)
        if info:
            png = bmp[:-4] + ".png"
            Image.open(bmp).convert("RGB").save(png)
            os.remove(bmp)
            info["file"] = os.path.relpath(png, outdir).replace("\\", "/")
            info["frame"] = f
        r["shots"].append(info)
    r["audio"] = analyse_audio(os.path.join(gdir, "audio.wav"))

    # Classification (a first guess; the screenshots have the final word).
    shots_ok = [s for s in r["shots"] if s]
    varied = [s for s in shots_ok if s["colors"] >= 6]
    if timed_out:
        r["status"] = "timeout"
    elif rc != 0 or not m:
        r["status"] = "crash"
    elif int(r.get("display_lists", "0")) == 0 and not varied:
        r["status"] = "no picture"
    elif not varied:
        r["status"] = "blank screen"
    elif int(r.get("audio_tasks", "0")) > 0 and r["audio"]["rms_db"] < -70:
        r["status"] = "silent"
    else:
        r["status"] = "runs"
    return r


STATUS_COLOR = {"runs": "#2e7d32", "silent": "#b26a00", "blank screen": "#b26a00", "no picture": "#c62828",
                "crash": "#c62828", "timeout": "#c62828"}
ABI = {"-1": "none", "0": "ABI 1", "1": "n_audio", "2": "EAD (MK)", "3": "EAD (SF/FZ)", "4": "EAD (Zelda)",
       "5": "MusyX"}


def write_report(results, outdir, args, compare):
    rows = []
    for r in results:
        a = r["audio"]
        shots = "".join(f'<a href="{s["file"]}"><img src="{s["file"]}" title="frame {s["frame"]}"></a>' if s
                        else '<span class="miss">no shot</span>' for s in r["shots"])
        change = ""
        if compare and r["name"] in compare:
            o = compare[r["name"]]
            if o.get("status") != r["status"]:
                change = f'<div class="chg">was: {html.escape(o.get("status", "?"))}</div>'
            if o.get("rdram_hash") and o.get("rdram_hash") != r.get("rdram_hash"):
                change += '<div class="chg">RDRAM differs</div>'
        warn = "<br>".join(html.escape(w) for w in r["warning_samples"])
        rows.append(f"""<tr>
<td><b>{html.escape(os.path.basename(r['rom']))}</b><div class="sub">{html.escape(r.get('title', ''))}</div></td>
<td><span class="st" style="background:{STATUS_COLOR.get(r['status'], '#555')}">{r['status']}</span>{change}</td>
<td>{r.get('fps', 0):.0f} fps<div class="sub">{r['wall_s']} s</div></td>
<td>{html.escape(r.get('gfx_ucode', '?'))}<div class="sub">{r.get('display_lists', '?')} DLs</div></td>
<td>{ABI.get(r.get('audio_abi', '-1'), r.get('audio_abi', '?'))}<div class="sub">{r.get('ai_rate', '?')} Hz</div></td>
<td>{a['rms_db']:.0f} dB<div class="sub">{a['clicks_per_s']:.1f} clicks/s</div>
<audio controls preload="none" src="{r['name']}/audio.wav"></audio></td>
<td class="shots">{shots}</td>
<td class="sub">{r['warnings']}<br>{warn}</td></tr>""")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = " &middot; ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    page = f"""<!doctype html><meta charset="utf-8"><title>Orbit64 compatibility</title>
<style>
body{{font:14px system-ui,sans-serif;background:#16161c;color:#ddd;margin:16px}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #333;padding:6px;vertical-align:top;text-align:left}}
.sub{{color:#999;font-size:12px}} .st{{color:#fff;border-radius:4px;padding:2px 6px;white-space:nowrap}}
.shots img{{width:160px;height:120px;image-rendering:pixelated;margin-right:4px;border:1px solid #333}}
.chg{{color:#ffb74d;font-size:12px}} audio{{width:160px;height:28px;margin-top:4px}} .miss{{color:#c62828}}
</style>
<h1>Orbit64 compatibility sweep</h1>
<p>{len(results)} ROMs, {args.frames} frames each, CPU: {args.cpu}. {summary}. Generated {time.strftime('%Y-%m-%d %H:%M')}.</p>
<table><tr><th>ROM</th><th>Status</th><th>Speed</th><th>Graphics</th><th>Audio ucode</th><th>Sound</th>
<th>Screenshots</th><th>Warnings</th></tr>
{''.join(rows)}</table>"""
    with open(os.path.join(outdir, "report.html"), "w", encoding="utf-8") as f:
        f.write(page)


def write_sheet(results, outdir):
    cols = len(SHOTS)
    w, h, label = 160, 120, 180
    sheet = Image.new("RGB", (label + cols * w, len(results) * h), (22, 22, 28))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(results):
        d.text((4, i * h + 4), os.path.basename(r["rom"])[:28], fill=(230, 230, 230))
        d.text((4, i * h + 20), r["status"], fill=(255, 180, 80) if r["status"] != "runs" else (120, 220, 120))
        d.text((4, i * h + 36), f'{r.get("gfx_ucode", "?")} / {ABI.get(r.get("audio_abi", "-1"), "?")}',
               fill=(170, 170, 170))
        for j, s in enumerate(r["shots"]):
            if s:
                im = Image.open(os.path.join(outdir, s["file"])).convert("RGB").resize((w, h))
                sheet.paste(im, (label + j * w, i * h))
    sheet.save(os.path.join(outdir, "sheet.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=1800)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument("--filter", default="")
    ap.add_argument("--cpu", default="jit", choices=["jit", "interp"])
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(ROOT, "test_output", "compat"))
    ap.add_argument("--compare", default="", help="results.json of an earlier sweep")
    ap.add_argument("--exe", default=os.path.join(ROOT, "bin", "n64.exe" if os.name == "nt" else "n64"))
    args = ap.parse_args()

    roms = find_roms(args.filter)
    if not roms:
        print("No ROMs found in roms/")
        return 1
    os.makedirs(args.out, exist_ok=True)
    compare = {}
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            compare = {r["name"]: r for r in json.load(f)}
    print(f"{len(roms)} ROMs, {args.jobs} at a time -> {args.out}")
    results = []
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(run_one, rom, args, args.out): rom for rom in roms}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"  {r['status']:<13} {r.get('fps', 0):6.0f} fps  {r.get('gfx_ucode', '?'):<10} "
                  f"{ABI.get(r.get('audio_abi', '-1'), '?'):<12} {os.path.basename(r['rom'])}", flush=True)
    results.sort(key=lambda r: os.path.basename(r["rom"]).lower())
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    write_report(results, args.out, args, compare)
    write_sheet(results, args.out)
    print(f"Report: {os.path.join(args.out, 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
