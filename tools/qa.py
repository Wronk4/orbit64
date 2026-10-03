#!/usr/bin/env python3
"""Game QA: does every game get to its gameplay, and is it drawn right there?

    python tools/qa.py record  [--list tools/test_set.txt | --filter text] [--jobs 4] [--retry-failed] [--force]
    python tools/qa.py run     [smoke|render|deep] [--list ... | --filter ...] [--jobs 4] [--no-lle] [--fresh]
    python tools/qa.py report                          # rebuild report.html from the last run
    python tools/qa.py routes                          # which games have a route, how far it gets

How it works
------------
record   plays each ROM with tools/game_probe (experiments from save states: which button does the game
         react to?) and writes the winning input as a *route*: tools/routes/<CRC>.route, a few hundred bytes
         per game (commit them). A checkpoint save state at the end goes to test_output/qa/checkpoints.
run      replays the routes from power-on (seconds, not minutes - the emulator is deterministic), checks the
         game is still playable there, plays it for a while and judges the picture: tools/game_qa.cpp. With
         LLE every picture is compared with what the real microcode draws from the same machine state, so
         anything HLE draws differently stands out (culled geometry, wrong textures, missing sprites).
         A route a fix made stale (timing changed) is re-recorded on the spot ("healed").
report   test_output/qa/report.html: every game with a verdict, the reasons, the numbers and the HLE / LLE /
         difference pictures of its worst moment; sortable, filterable; what changed since the last run.

Tiers (run): smoke = route + a short look (about 15 s per game), render = + gameplay samples and the HLE/LLE
comparison (the default), deep = longer play, more samples.

Verdicts: FAIL (crash, route lost, CPU jumped into nothing, nothing drawn in gameplay), WARN (HLE differs a lot
from LLE, geometry mostly culled, snow in the picture, silent audio, slow), OK. A game that is worse than in the
previous run is also marked REGRESSION, a better one FIXED.

Needs Pillow and numpy for the report. Save files next to the ROMs are neither read nor written.
"""
import argparse
import base64
import concurrent.futures as cf
import hashlib
import html
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game_status as gs  # noqa: E402  (ROM discovery, process handling shared with the status page)

ROOT = gs.ROOT
EXE_SUFFIX = ".exe" if os.name == "nt" else ""
PROBE = os.path.join(ROOT, "bin", "game_probe" + EXE_SUFFIX)
QA = os.path.join(ROOT, "bin", "game_qa" + EXE_SUFFIX)
ROUTES = os.path.join(ROOT, "tools", "routes")
OUT = os.path.join(ROOT, "test_output", "qa")

TIERS = {
    #          play frames, sample every, LLE frames, wall limit (s), tap the other buttons too
    "smoke":  dict(frames=60,  sample=30, lle=0,   limit=120, mash=False),
    "render": dict(frames=300, sample=30, lle=120, limit=240, mash=True),
    "deep":   dict(frames=900, sample=60, lle=300, limit=600, mash=True),
}

ENV = dict(os.environ, SDL_VIDEODRIVER="offscreen", SDL_AUDIODRIVER="dummy")


def spawn(cmd, timeout):
    """gs.spawn with the offscreen environment (a test must never open a window)."""
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         errors="replace", env=ENV)
    with gs._children_lock:
        gs._children.add(p)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err, False
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return None, out or "", err or "", True
    finally:
        with gs._children_lock:
            gs._children.discard(p)


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def exe_stamp():
    """Changes whenever the emulator (game_qa links all of it) is rebuilt: results from before are stale."""
    try:
        return int(os.path.getmtime(QA))
    except OSError:
        return 0


def read_route_head(path):
    """The header of a route file: {key: value}; None if missing or damaged."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    if not lines or not lines[0].startswith("orbit-route 1"):
        return None
    head = {}
    frames = 0
    for ln in lines[1:]:
        k, _, v = ln.partition(" ")
        if k == "in":
            a = v.split()
            frames = max(frames, int(a[0]) + int(a[1]))
        else:
            head[k] = v
    head["frames"] = frames
    return head if frames else None


# ---------------------------------------------------------------------------
# Recording

def route_path(e):
    return os.path.join(ROUTES, e["crc"] + ".route")


def frontier_path(e):
    """Where the search got to (any class): the starting point of the next `record --improve`. Local, not committed."""
    return os.path.join(OUT, "frontier", e["crc"] + ".route")


RANK = {"furthest": 0, "big": 1, "scene": 2, "ingame": 3}


def probe_stop(rdir):
    """Why the probe stopped: 'dead' (the game did nothing for 12 steps: a stall), 'time', 'framecap', or ''."""
    pj = load_json(os.path.join(rdir, "probe.json"))
    if not pj:
        return ""
    if pj.get("dead_steps", 0) >= 12:
        return "dead"
    if pj.get("stagnant"):
        return "stagnant"
    if pj.get("out_of_time"):
        return "time"
    if pj.get("route_reached") != "ingame":
        return "framecap"
    return ""


def truncate_route(src, dst, frames):
    """A copy of a route file cut at `frames` (its inputs and pictures up to there)."""
    out = []
    with open(src, encoding="utf-8") as f:
        for ln in f.read().splitlines():
            k, _, v = ln.partition(" ")
            if k == "in":
                a = v.split()
                first, n = int(a[0]), int(a[1])
                if first >= frames:
                    continue
                if first + n > frames:
                    a[1] = str(frames - first)
                out.append("in " + " ".join(a))
            elif k == "sig":
                if int(v.split()[0]) <= frames:
                    out.append(ln)
            else:
                out.append(ln)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def spawn_with_advice(cmd, timeout, advdir):
    """Runs the probe and answers its requests for advice (tools/advice.py) while it runs."""
    import advice
    os.makedirs(advdir, exist_ok=True)
    outp, errp = os.path.join(advdir, "stdout.txt"), os.path.join(advdir, "stderr.txt")
    with open(outp, "w", encoding="utf-8") as fo, open(errp, "w", encoding="utf-8") as fe:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fo, stderr=fe, text=True, errors="replace", env=ENV)
        with gs._children_lock:
            gs._children.add(p)
        t0 = time.time()
        timed_out = False
        try:
            while p.poll() is None:
                if advice.serve(advdir) == 0:
                    time.sleep(0.05)
                if time.time() - t0 > timeout:
                    p.kill()
                    timed_out = True
                    break
            p.wait()
        finally:
            with gs._children_lock:
                gs._children.discard(p)
    with open(outp, encoding="utf-8", errors="replace") as fo, open(errp, encoding="utf-8", errors="replace") as fe:
        return (None if timed_out else p.returncode), fo.read(), fe.read(), timed_out


def record_one(e, args, improve=False, protect=None, resume=None):
    """Runs the probe on one ROM; returns an index entry. With `improve` the search goes on from the end of
    the route the game already has, and the old route is only replaced by one that gets further.
    With `protect` (the header of a route that no longer works) a new route that reaches a lower class than
    that one is not installed: the old route stays as the evidence of what used to work."""
    rdir = os.path.join(OUT, "record", e["crc"])
    shutil.rmtree(rdir, ignore_errors=True)
    os.makedirs(rdir, exist_ok=True)
    os.makedirs(ROUTES, exist_ok=True)
    os.makedirs(os.path.join(OUT, "checkpoints"), exist_ok=True)
    new_route = os.path.join(rdir, "new.route")
    cmd = [PROBE, os.path.join(ROOT, e["rom"]), "--out", rdir, "--route", new_route,
           "--checkpoint", os.path.join(OUT, "checkpoints", e["crc"] + ".state"),
           "--max-seconds", str(args.max_seconds)]
    old = read_route_head(route_path(e)) if improve else None
    new_frontier = os.path.join(rdir, "new.frontier")
    cmd += ["--frontier", new_frontier]
    index = load_json(os.path.join(ROUTES, "index.json"), {})
    ent = index.get(e["crc"], {})
    attempts = ent.get("attempts", 1 if ent else 0)
    # a different way of choosing every attempt: which action wins when several change the picture
    cmd += ["--strategy", str(attempts % 4)]
    if old:
        # go on from where the search got to last time if that is further than the route that is kept - but
        # not from a dead end: after a search that went in circles the next one starts well before the circle
        fr = read_route_head(frontier_path(e))
        start = frontier_path(e) if fr and fr["frames"] > old["frames"] else route_path(e)
        if ent.get("stop") == "stagnant" and start == frontier_path(e):
            back = os.path.join(OUT, "record", e["crc"] + ".back.route")
            os.makedirs(os.path.dirname(back), exist_ok=True)
            sigs = [int(ln.split()[1]) for ln in open(start, encoding="utf-8").read().splitlines() if ln.startswith("sig ")]
            cut = max([f for f in sigs if f <= 0.6 * fr["frames"]] or [old["frames"]])
            truncate_route(start, back, max(cut, old["frames"]))
            start = back
        cmd += ["--continue-route", start]
    elif resume:
        cmd += ["--continue-route", resume]
    t0 = time.time()
    if getattr(args, "advice", True):
        # reading the screen is slow next to the emulation: the time spent waiting for it is not counted by the probe
        rc, out, err, timed_out = spawn_with_advice(cmd + ["--advice", os.path.join(rdir, "advice")], args.max_seconds * 3 + 150,
                                                     os.path.join(rdir, "advice"))
    else:
        rc, out, err, timed_out = spawn(cmd, args.max_seconds + 150)
    entry = {"name": e["name"], "rom": e["rom"], "seconds": round(time.time() - t0, 1), "head": git_head()}
    if STOP_REQUESTED():
        return None
    head = read_route_head(new_route) if rc == 0 else None
    entry["stop"] = probe_stop(rdir) if rc == 0 else ""   # (also when the old route is kept: it tells the next search where it stands)
    nf = read_route_head(new_frontier) if rc == 0 else None
    if nf:
        cur = read_route_head(frontier_path(e))
        if not cur or nf["frames"] > cur["frames"]:
            os.makedirs(os.path.dirname(frontier_path(e)), exist_ok=True)
            shutil.copyfile(new_frontier, frontier_path(e))
        entry["frontier"] = nf["frames"]
    if head is None:
        entry["failed"] = "timeout" if timed_out else f"probe exit {rc}"
        with open(os.path.join(rdir, "log.txt"), "w", encoding="utf-8") as f:
            f.write((out or "")[-4000:] + "\n--- stderr ---\n" + (err or "")[-4000:])
        return entry
    # better: a higher class, or - in the same class - an end in a moving 3D scene instead of a menu / pause
    better = old is None or RANK.get(head.get("reached"), 0) > RANK.get(old.get("reached"), 0) or (
        RANK.get(head.get("reached"), 0) == RANK.get(old.get("reached"), 0) and head.get("ends") == "scene" and old.get("ends") != "scene")
    if old and not better:
        entry.update(reached=old.get("reached"), frames=old["frames"], play=old.get("play"), kept="no better route found")
        return entry
    if protect and RANK.get(head.get("reached"), 0) < RANK.get(protect.get("reached"), 0):
        entry.update(reached=protect.get("reached"), frames=protect["frames"], play=protect.get("play"),
                     kept=f"re-recording reached only {head.get('reached')}", worse=head.get("reached"))
        return entry
    shutil.copyfile(new_route, route_path(e))
    entry.update(reached=head.get("reached"), frames=head["frames"], play=head.get("play"), stop=probe_stop(rdir))
    return entry


def backfill_stops(index):
    """Routes recorded before the stop reason was kept: read it from the probe's own output."""
    changed = False
    for crc, ent in index.items():
        if "stop" in ent or "reached" not in ent:
            continue
        ent["stop"] = probe_stop(os.path.join(OUT, "record", crc))
        changed = True
    return changed


def STOP_REQUESTED():
    return gs.STOP.is_set()


def old_probe_route(probe, step=45):
    """A route rebuilt from an old probe.json (tools/game_status.py's output): its timeline lists, per step,
    what was pressed. A tap is A / START held for 6 frames, a stick step is down for half, then right; an
    UNSTICK is a START tap in a step of 90 frames. Only games the old probe took to gameplay are rebuilt. NAV steps (stick + A in a direction) did not record the direction: such
    a game cannot be rebuilt. Returns (inputs, reached, ingame_frame) or None."""
    tl = probe.get("timeline") or []
    boot = tl[0]["frame"] if tl else 240
    if probe.get("ingame_frame", -1) >= 0:
        reached, stop_frame = "ingame", probe["frames"]
    else:
        # "big" (the stick moved much of the picture once) is not trusted from an old run: at a logo, the stick
        # only has to change a game's timing for two runs to differ. Only the six-steps-in-a-row evidence counts.
        return None
    inputs = [(0, 0, 0)] * boot
    for i, t in enumerate(tl):
        start = t["frame"]
        if start >= stop_frame:
            break
        end = tl[i + 1]["frame"] if i + 1 < len(tl) else probe["frames"]
        n = min(end, stop_frame) - start
        act = t["action"]
        if act == "NAV":
            return None
        for f in range(n):
            if act == "A":
                inputs.append((0x8000 if f < 6 else 0, 0, 0))
            elif act in ("START", "UNSTICK"):
                inputs.append((0x1000 if f < 6 else 0, 0, 0))
            elif act == "stick":
                inputs.append((0, 0, -80) if f < step // 2 else (0, 80, 0))
            else:
                inputs.append((0, 0, 0))
    return inputs, reached, probe.get("ingame_frame", -1)


def write_route_file(path, e, inputs, reached, play, ingame_frame, boot=240):
    runs = []
    for i, v in enumerate(inputs):
        if runs and runs[-1][2] == v:
            runs[-1][1] += 1
        else:
            runs.append([i, 1, v])
    lines = ["orbit-route 1", f"rom {os.path.basename(e['rom'])}", f"crc {e['crc']}", f"title {e['internal']}",
             "state_version 0", "rsp hle", f"boot {boot}", f"reached {reached}", f"play {play}", f"ingame_frame {ingame_frame}"]
    lines += [f"in {a} {n} {v[0]:x} {v[1]} {v[2]}" for a, n, v in runs]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def seed_one(e, args, old_dir):
    """Tries the route the old probe found for this game; installs it if it still gets to gameplay."""
    pj = load_json(os.path.join(old_dir, "probe.json"))
    if not pj:
        return None
    built = old_probe_route(pj, pj.get("step", 45))
    if not built:
        return {"name": e["name"], "skipped": "no usable route in the old run"}
    inputs, reached, ingame = built
    fr = pj.get("final_react", {})
    play = "stick_a" if fr.get("stick", 1) < 0.10 <= fr.get("stick_with_a", 0) else "stick"
    cand_dir = os.path.join(OUT, "seed", e["crc"])
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(cand_dir, exist_ok=True)
    cand = os.path.join(cand_dir, "cand.route")
    write_route_file(cand, e, inputs, reached, play, ingame)
    rc, out, err, to = spawn([QA, os.path.join(ROOT, e["rom"]), "--route", cand, "--out", cand_dir, "--frames", "30",
                              "--sample", "30", "--max-seconds", "120"], 300)
    data = load_json(os.path.join(cand_dir, "qa.json")) if rc == 0 else None
    if STOP_REQUESTED():
        return None
    if not data:
        return {"name": e["name"], "skipped": "the replay failed"}
    if not data["route"]["route_ok"]:
        return {"name": e["name"], "skipped": f"the old route no longer works (stick moves {pct(data['route']['playable_diff'])})"}
    old = read_route_head(route_path(e))
    if old and RANK.get(old.get("reached"), 0) > RANK[reached]:
        return {"name": e["name"], "skipped": f"already has a better route ({old.get('reached')})"}
    if old and RANK.get(old.get("reached"), 0) == RANK[reached] and old["frames"] <= len(inputs):
        return {"name": e["name"], "skipped": "already has an equally good route"}
    os.makedirs(ROUTES, exist_ok=True)
    shutil.copyfile(cand, route_path(e))
    return {"name": e["name"], "reached": reached, "frames": len(inputs), "was": old.get("reached") if old else None}


def cmd_seed(args):
    """Routes from what the old probe already found (test_output/game_status): no new search, only a replay."""
    if not os.path.exists(QA):
        print("build the tools first: make game_probe game_qa", file=sys.stderr)
        return 2
    results = load_json(os.path.join(ROOT, "test_output", "game_status", "results.json"), [])
    olddir = {g["crc"]: os.path.join(ROOT, "test_output", "game_status", g["dir"]) for g in results if g.get("crc") and g.get("dir")}
    roms = [e for e in select_roms(args) if e["crc"] in olddir and os.path.isdir(olddir[e["crc"]])]
    print(f"{len(roms)} games have an old run, {args.jobs} at a time")
    index_path = os.path.join(ROUTES, "index.json")
    index = load_json(index_path, {})
    n = done = 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(seed_one, e, args, olddir[e["crc"]]): e for e in roms}
        try:
            for fut in gs.as_done(futs):
                e = futs[fut]
                r = fut.result()
                done += 1
                if not r:
                    continue
                if "reached" in r:
                    n += 1
                    index[e["crc"]] = {"name": e["name"], "rom": e["rom"], "reached": r["reached"], "frames": r["frames"],
                                       "seeded": True, "stop": "", "seconds": 0, "head": git_head()}
                    save_json(index_path, index)
                    print(f"[{done}/{len(roms)}] seeded {e['name']}: {r['reached']} in {r['frames']} frames"
                          + (f" (was {r['was']})" if r["was"] else ""), flush=True)
                elif args.verbose:
                    print(f"[{done}/{len(roms)}] {e['name']}: {r['skipped']}", flush=True)
        except KeyboardInterrupt:
            gs.STOP.set()
            gs.kill_children()
            return 1
    print(f"{n} routes installed from the old runs")
    return 0


def cmd_record(args):
    roms = select_roms(args)
    index_path = os.path.join(ROUTES, "index.json")
    index = load_json(index_path, {})
    todo = []
    for e in roms:
        head = read_route_head(route_path(e))
        ent = index.get(e["crc"], {})
        if args.improve:
            # games whose route stops short of gameplay, a few tries each
            # (a game that stalls keeps its route: it is the recipe that reproduces the stall)
            if head and head.get("reached") != "ingame" and ent.get("attempts", 1) < args.max_attempts and ent.get("stop") != "dead":
                todo.append(e)
        elif args.force or (head is None and ("failed" not in ent or args.retry_failed)):
            todo.append(e)
    print(f"{len(roms)} ROMs, {len(todo)} to record, {args.jobs} at a time")
    done = 0
    lock = threading.Lock()
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(record_one, e, args, args.improve): e for e in todo}
        try:
            for fut in gs.as_done(futs):
                e = futs[fut]
                entry = fut.result()
                if entry is None:
                    continue
                with lock:
                    # a route recorded before attempts were counted was one attempt
                    entry["attempts"] = index.get(e["crc"], {}).get("attempts", 1 if e["crc"] in index else 0) + 1
                    index[e["crc"]] = entry
                    save_json(index_path, index)
                    done += 1
                    state = entry.get("failed") or f'{entry["reached"]} in {entry["frames"]} frames{" (kept the old route)" if entry.get("kept") else ""}'
                    print(f"[{done}/{len(todo)}] {e['name']}: {state} ({entry['seconds']} s)", flush=True)
        except KeyboardInterrupt:
            gs.STOP.set()
            gs.kill_children()
            print("interrupted - run again to continue")
            return 1
    if backfill_stops(index):
        save_json(index_path, index)
    reached = {}
    for e in roms:
        r = (index.get(e["crc"], {}) or {}).get("reached") or ("failed" if "failed" in index.get(e["crc"], {}) else "none")
        reached[r] = reached.get(r, 0) + 1
    print("routes:", ", ".join(f"{k} {v}" for k, v in sorted(reached.items())))
    return 0


# ---------------------------------------------------------------------------
# Running

def select_roms(args):
    listed = None
    if args.list:
        with open(args.list, encoding="utf-8") as f:
            listed = {ln.split("#")[0].strip() for ln in f if ln.split("#")[0].strip()}
    return gs.find_roms(args.filter, listed)


def run_dir(e):
    return os.path.join(OUT, "runs", e["crc"])


def qa_once(e, tier, args, with_checkpoint=False):
    """One game_qa run; returns (qa.json dict or None, exit code, log tail, timed out)."""
    t = TIERS[tier]
    rdir = run_dir(e)
    shutil.rmtree(rdir, ignore_errors=True)
    os.makedirs(rdir, exist_ok=True)
    cmd = [QA, os.path.join(ROOT, e["rom"]), "--route", route_path(e), "--out", rdir,
           "--frames", str(t["frames"]), "--sample", str(t["sample"]), "--max-seconds", str(t["limit"])]
    if t["mash"]:
        cmd.append("--mash")
    lle = t["lle"] and not args.no_lle
    if lle:
        cmd += ["--lle", "--lle-frames", str(t["lle"])]
    if with_checkpoint:
        cmd += ["--checkpoint", os.path.join(OUT, "checkpoints", e["crc"] + ".state")]
    rc, out, err, timed_out = spawn(cmd, t["limit"] + 120)
    log = (err or "")[-3000:]
    data = load_json(os.path.join(rdir, "qa.json")) if rc == 0 else None
    return data, rc, log, timed_out


def run_one(e, tier, args):
    """Replays a game's route and judges the result; re-records a stale route once."""
    t0 = time.time()
    res = {"crc": e["crc"], "name": e["name"], "region": e["region"], "version": e["version"], "rom": e["rom"], "tier": tier}
    if read_route_head(route_path(e)) is None:
        if not args.record_missing:
            res.update(status="none", reasons=[reason("no_route", "no route recorded (qa.py record)", "info")])
            return res
        entry = record_one(e, args)
        if entry is None:
            return None
        if "failed" in entry:
            res.update(status="fail", reasons=[reason("record_failed", f"no route: the probe failed ({entry['failed']})", "fail")])
            return res
        res["healed"] = "recorded"
    index = load_json(os.path.join(ROUTES, "index.json"), {})
    stop = (index.get(e["crc"]) or {}).get("stop", "")
    data, rc, log, timed_out = qa_once(e, tier, args, with_checkpoint=args.checkpoint)
    if STOP_REQUESTED():
        return None
    if data is not None and data["route"].get("trimmed_to", 0) > 0 and not args.no_heal:
        # The route's last steps spoiled it (an A press that opened a menu): game_qa found an earlier end
        # where the stick works. Keep the route cut there.
        cut = data["route"]["trimmed_to"]
        truncate_route(route_path(e), route_path(e), cut)
        res["healed"] = f"trimmed to frame {cut}"
        index = load_json(os.path.join(ROUTES, "index.json"), {})
        if e["crc"] in index:
            index[e["crc"]]["frames"] = cut
            save_json(os.path.join(ROUTES, "index.json"), index)
    if data is not None and not data["route"]["route_ok"] and not args.no_heal:
        # The route no longer gets to a playable game: something changed the timing. Find a new one -
        # but a new route that gets less far than the old one is not a "heal", it is a regression:
        # the old route stays, and keeps failing until the game is fixed.
        old = read_route_head(route_path(e))
        # The route's pictures say where the game parted from the recording: everything up to the last
        # picture that still matched is good, so the search goes on from there instead of from power-on.
        resume = None
        drift = data["route"].get("drift", {})
        if drift.get("first", -1) >= 0 and drift.get("last_ok", 0) >= 300:
            resume = os.path.join(OUT, "record", e["crc"] + ".prefix.route")
            os.makedirs(os.path.dirname(resume), exist_ok=True)
            truncate_route(route_path(e), resume, drift["last_ok"])
        entry = record_one(e, args, protect=old, resume=resume)
        if entry is None:
            return None
        if entry.get("worse"):
            res["seconds"] = round(time.time() - t0, 1)
            res.update(status="fail", route={"reached": old.get("reached"), "frames": old["frames"]},
                       reasons=[reason("route_regressed", f"the game used to reach {old.get('reached')} and now only reaches "
                                       f"{entry['worse']}: it no longer gets as far (the old route is kept)", "fail")])
            return res
        if "failed" not in entry:
            res["healed"] = f"continued from frame {drift['last_ok']}" if resume else "re-recorded"
            index = load_json(os.path.join(ROUTES, "index.json"), {})
            index[e["crc"]] = {**index.get(e["crc"], {}), **entry}
            save_json(os.path.join(ROUTES, "index.json"), index)
            stop = entry.get("stop", "")
            data, rc, log, timed_out = qa_once(e, tier, args)
    res["seconds"] = round(time.time() - t0, 1)
    if data is None:
        why = "timed out" if timed_out else f"game_qa exit code {rc}"
        res.update(status="fail", reasons=[reason("crash", f"the emulator did not finish: {why}", "fail")], log=log[-1500:])
        return res
    screen = read_final_screen(run_dir(e), data)
    judge(res, data, stop, screen)
    finalize(res, load_known())
    return res


KNOWN_FILE = os.path.join(ROOT, "tools", "qa_known.json")


def load_known():
    """tools/qa_known.json: {"<CRC>": {"name": ..., "ignore": ["slow", ...], "note": "why"}} - findings that are
    understood and accepted for a game, so the report only shows what is new."""
    return {k: v for k, v in (load_json(KNOWN_FILE, {}) or {}).items() if isinstance(v, dict)}


def finalize(g, known):
    """Applies the known-issues file and works out the status from the reasons."""
    k = known.get(g["crc"], {})
    ignore = set(k.get("ignore", []))
    for r in g.get("reasons", []):
        if r["level"] in ("fail", "warn") and r["code"] in ignore:
            r["level"] = "known"
            r["text"] += " (known" + (": " + k["note"] if k.get("note") else "") + ")"
    if g.get("status") == "none":
        return
    rs = g.get("reasons", [])
    g["status"] = "fail" if any(r["level"] == "fail" for r in rs) else "warn" if any(r["level"] == "warn" for r in rs) else "ok"


def reason(code, text, level):
    return {"code": code, "text": text, "level": level}


def pct(x):
    return f"{100 * x:.0f}%"


ERROR_WORDS = ["ERROR", "FATAL", "EXCEPTION", "CRASH", "PANIC", "ASSERT", "FAULT", "TLB", "OUTOFMEMORY", "CORRUPT", "NOTFOUND"]


def read_final_screen(rdir, data):
    """What the last picture of the run says (OCR + tools/screen_reader.py): the kind of screen, the text, and any
    warning the game itself puts up. None when the reader is not installed or the picture is missing."""
    try:
        import screen_reader as sr
        from PIL import Image
    except ImportError:
        return None
    n = len(data["hle"]["pictures"])
    path = os.path.join(rdir, f"hle_{n - 1:02d}.png")
    if n == 0 or not os.path.exists(path):
        return None
    try:
        adv = sr.advise(Image.open(path))
    except Exception:
        return None
    return {"kind": adv.kind, "lines": adv.lines[:24], "notes": adv.notes[:12], "options": [o.text for o in adv.options][:8]}


def judge(res, d, stop="", screen=None):
    """Turns one qa.json into metrics, reasons and a status."""
    hle, lle = d["hle"], d.get("lle")
    pics = hle["pictures"]
    reasons = []
    reached = d["route"]["reached"]
    wants_play = reached in ("ingame", "big", "scene")  # a moving 3D scene is expected to keep drawing
    m = {
        "reached": reached, "route_frames": d["route"]["frames"], "replay_s": d["route"]["replay_seconds"],
        "playable_diff": d["route"]["playable_diff"], "fps": hle["fps"],
        "triangles": hle["triangles"], "cull": (hle["cull_back"] + hle["cull_front"]) / max(1, hle["triangles"]),
        "audio_db": hle.get("audio_rms_db"), "moving": hle["moving"], "gfx_tasks": hle["gfx_tasks"],
        "noise": max((p["noise"] for p in pics), default=0), "lit": max((p["lit"] for p in pics), default=0),
        "colors": max((p["colors"] for p in pics), default=0),
    }
    res["route"] = {"reached": reached, "frames": d["route"]["frames"], "play": d["route"]["play"]}
    dr = d["route"].get("drift") or {}
    if dr.get("first", -1) >= 0 and d["route"]["route_ok"]:
        lvl = "warn" if dr["max"] > 0.25 else "info"
        reasons.append(reason("drift", f"the picture at frame {dr['first']} of the route no longer matches the recording "
                              f"(up to {pct(dr['max'])} of the grey cells): the game looks different from when the route was "
                              f"recorded; `qa.py rebase` accepts the new look", lvl))
    if not d["route"]["route_ok"]:
        reasons.append(reason("route_lost", f"the route no longer reaches gameplay (the stick moves {pct(d['route']['playable_diff'])} of the picture)", "fail"))
    alive = hle["gfx_tasks"] > 0 or hle["vi_swaps"] > 0 or hle["moving"] > 0
    if stop == "dead" and not wants_play:
        if not alive:
            reasons.append(reason("stalled", f"the game stalls: after {d['route']['frames']} frames of the route nothing happens any more "
                                  f"(no graphics task, no frame flip, no change on screen) - the route reproduces it", "fail"))
        else:
            reasons.append(reason("unstalled", "the route used to end in a stall and the game now keeps running: "
                                  "continue it with `qa.py record --improve`", "info"))
    if hle["cpu_lost"]:
        reasons.append(reason("cpu_lost", f"the CPU is at {hle['pc']} (exception vector / address 0): the game crashed", "fail"))
    if wants_play and hle["gfx_tasks"] == 0 and hle["vi_swaps"] == 0:
        reasons.append(reason("no_graphics", "no graphics task and no frame buffer flip while playing: the game does not draw", "fail"))
    elif wants_play and m["lit"] < 0.02:
        reasons.append(reason("black", "the picture is black all the time while playing", "fail"))
    elif wants_play and hle["moving"] == 0 and len(pics) > 2:
        reasons.append(reason("frozen", "the picture never changes while the stick is held", "warn"))
    if hle["triangles"] > 200 and m["cull"] > 0.9:
        reasons.append(reason("culled", f"{pct(m['cull'])} of the triangles are culled by HLE", "warn"))
    if hle.get("audio_rms_db") is not None and hle["audio_tasks"] > 20 and hle["audio_rms_db"] < -70 and wants_play:
        reasons.append(reason("silent", f"audio tasks run but the output is silent ({hle['audio_rms_db']:.0f} dB)", "info"))
    if wants_play and m["fps"] < 25:
        reasons.append(reason("slow", f"only {m['fps']:.0f} frames per second in headless play (with several tests running at once, "
                                      f"this is a lower bound)", "warn"))
    # HLE against the real microcode, from the same machine state.
    cmp_ = d.get("compare") or []
    if lle and "error" in lle:
        reasons.append(reason("lle_error", f"the low-level run could not start: {lle['error']}", "info"))
    elif lle and cmp_:
        # the coarse measure (see game_qa.cpp): edges and anti-aliasing do not count, missing things do
        cells = [c.get("coarse", c["cells"]) for c in cmp_]
        m["cells_mean"] = statistics.mean(cells)
        m["cells_max"] = max(cells)
        m["pixels_mean"] = statistics.mean(c["pixels"] for c in cmp_)
        m["worst"] = max(range(len(cells)), key=lambda i: cells[i])
        m["noise_lle"] = max((p["noise"] for p in lle["pictures"]), default=0)
        if lle.get("cpu_lost"):
            reasons.append(reason("lle_cpu_lost", "the CPU is lost in the LLE run (the HLE run is not)", "info"))
        if m["noise"] > 2.5 * m["noise_lle"] + 0.03:
            reasons.append(reason("snow", f"HLE picture is much noisier than LLE ({m['noise']:.3f} vs {m['noise_lle']:.3f}): wrong textures?", "warn"))
    res["ucode"] = d.get("ucode")
    if screen:
        res["screen"] = screen
        text = " ".join(screen["lines"]).upper()
        norm_text = "".join(ch for ch in text if ch.isalnum())
        if screen["kind"] == "warning" and screen["notes"]:
            shown = " / ".join(screen["notes"])[:160]
            hint = " (an accessory the game does not accept: how the emulated pak answers?)" if any(
                w in norm_text for w in ("RUMBLEPAK", "CONTROLLERPAK", "TRANSFERPAK", "PAK")) else ""
            reasons.append(reason("screen_warning", f"the game shows a warning: {shown}{hint}", "warn"))
        elif any(w in norm_text for w in ERROR_WORDS):
            reasons.append(reason("screen_error", f"the screen reads like an error message: {' / '.join(screen['lines'])[:160]}", "warn"))
    if hle.get("lle_tasks", 0) > 0:
        res["hle_fallback"] = True
        reasons.append(reason("hle_fallback", f"{hle['lle_tasks']} graphics tasks ran on the low-level RSP: HLE does not know this "
                              f"microcode ({d.get('ucode')})", "warn"))
    for entry in d.get("log", []):
        line = entry["line"]
        if line.startswith("[UCODE-DETECT-FAIL]"):
            continue  # the same finding as hle_fallback
        reasons.append(reason("log", f"emulator log: {line[:140]} (x{entry['count']})", "info"))
    res["metrics"] = m
    res["reasons"] = reasons
    res["samples"] = len(pics)
    res["has_lle"] = bool(lle and cmp_)


def post_process(results):
    """Judgements that need the whole corpus: how far HLE and LLE usually differ."""
    cells = [r["metrics"]["cells_mean"] for r in results if "cells_mean" in r.get("metrics", {})]
    if len(cells) < 8:
        thr = 0.20
    else:
        med = statistics.median(cells)
        mad = statistics.median(abs(c - med) for c in cells) or 0.01
        thr = max(0.12, med + 4 * 1.4826 * mad)
    for r in results:
        m = r.get("metrics", {})
        if "cells_mean" in m and m["cells_mean"] > thr and not r.get("hle_fallback"):
            r["reasons"].append(reason("diverges", f"HLE differs from LLE in {pct(m['cells_mean'])} of the picture on average "
                                       f"(typical: {pct(statistics.median(cells)) if cells else '?'}) - see the pictures", "warn"))
    return thr


def mark_change(g):
    """Compares a game with the result it had before it was last run (g['prev_status'] / g['prev_codes'])."""
    g.pop("change", None)
    g.pop("was", None)
    order = {"ok": 0, "warn": 1, "fail": 2}
    was = g.get("prev_status")
    if was not in order or g["status"] not in order:
        return
    if order[g["status"]] > order[was]:
        g["change"], g["was"] = "regression", was
    elif order[g["status"]] < order[was]:
        g["change"], g["was"] = "fixed", was
    else:
        now = {x["code"] for x in g["reasons"] if x["level"] != "info"}
        if now - set(g.get("prev_codes", [])):
            g["change"], g["was"] = "new_issue", was


def stale(e, tier, args):
    """True if this game's last result was made by another emulator build or route."""
    if args.fresh:
        return True
    meta = load_json(os.path.join(run_dir(e), "meta.json"))
    if not meta or meta.get("tier") != tier or meta.get("exe") != exe_stamp() or meta.get("lle") != (not args.no_lle):
        return True
    try:
        if meta.get("route") != int(os.path.getmtime(route_path(e))):
            return True
    except OSError:
        return True
    return False


def cmd_run(args):
    tier = args.tier
    if not os.path.exists(QA) or not os.path.exists(PROBE):
        print("build the tools first: make game_probe game_qa", file=sys.stderr)
        return 2
    roms = select_roms(args)
    results_path = os.path.join(OUT, "results.json")
    prev_doc = load_json(results_path, {})
    prev = {r["crc"]: r for r in prev_doc.get("games", [])}
    # --baseline: judge changes against another results.json (a release, an older build)
    base = {r["crc"]: r for r in (load_json(args.baseline, {}) or {}).get("games", [])} if args.baseline else prev
    # What the last run knew about the games not selected this time stays in the results.
    results = {k: v for k, v in prev.items() if k not in {e["crc"] for e in roms}}
    todo = []
    for e in roms:
        old = prev.get(e["crc"])
        if old and old.get("tier") == tier and not stale(e, tier, args) and old.get("status") != "none":
            results[e["crc"]] = old
        else:
            todo.append(e)
    print(f"{len(roms)} games, {len(todo)} to run ({tier}), {args.jobs} at a time"
          + ("" if not results else f", {len(results)} up to date"))
    t0 = time.time()
    done = 0
    lock = threading.Lock()

    def finish(e, r):
        nonlocal done
        with lock:
            done += 1
            old = base.get(e["crc"])
            if old and old.get("status") != "none":
                r["prev_status"] = old["status"]
                r["prev_codes"] = [x["code"] for x in old.get("reasons", []) if x["level"] != "info"]
            results[e["crc"]] = r
            mark = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL", "none": "none"}.get(r["status"], r["status"])
            why = "; ".join(x["text"] for x in r["reasons"] if x["level"] != "info")[:110]
            extra = f" [{r['healed']}]" if r.get("healed") else ""
            print(f"[{done}/{len(todo)}] {mark} {e['name']}{extra} {why}", flush=True)
            prog = load_json(os.path.join(OUT, "run_progress.json"), None)
            if not prog or prog.get("t0") != t0:
                prog = {"t0": t0, "tier": tier, "total": len(todo), "counts": {}, "last": []}
            prog["done"] = done
            prog["updated"] = time.time()
            prog["counts"][r["status"]] = prog["counts"].get(r["status"], 0) + 1
            prog["last"] = (prog["last"] + [f"{mark} {e['name']}{extra} {why}"])[-12:]
            prog["finished"] = done >= len(todo)
            save_json(os.path.join(OUT, "run_progress.json"), prog)
            if r["status"] != "none":
                save_json(os.path.join(run_dir(e), "meta.json"), {
                    "tier": tier, "exe": exe_stamp(), "lle": not args.no_lle, "route": int(os.path.getmtime(route_path(e)))
                    if os.path.exists(route_path(e)) else 0})
            if done % 10 == 0:
                write_results(results, roms, prev_doc, tier, args, t0, partial=True)

    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, e, tier, args): e for e in todo}
        try:
            for fut in gs.as_done(futs):
                e = futs[fut]
                r = fut.result()
                if r is not None:
                    finish(e, r)
        except KeyboardInterrupt:
            gs.STOP.set()
            gs.kill_children()
            print("interrupted - the finished games are saved; run again to continue")
            write_results(results, roms, prev_doc, tier, args, t0, partial=True)
            return 1
    write_results(results, roms, prev_doc, tier, args, t0)
    doc = load_json(results_path)
    s = doc["summary"]
    print(f"\n{s['ok']} ok, {s['warn']} warnings, {s['fail']} failures, {s['none']} without a route;"
          f" {s['regression']} regressions, {s['fixed']} fixed ({time.time() - t0:.0f} s)")
    print("report:", os.path.join(OUT, "report.html"))
    if args.strict and (s["fail"] or s["regression"]):
        return 3
    return 0


def write_results(results, roms, prev_doc, tier, args, t0, partial=False):
    # every game that has a result, in the order of the whole ROM collection (a filtered run keeps the rest)
    games = [results[e["crc"]] for e in gs.find_roms(None, None) if e["crc"] in results]
    known = load_known()
    for g in games:
        # the corpus-relative judgement is redone every time, and so is what is known
        g["reasons"] = [x for x in g.get("reasons", []) if x["code"] != "diverges"]
        for x in g["reasons"]:
            if x["level"] == "known":
                x["level"] = "warn"
                x["text"] = x["text"].split(" (known")[0]
        finalize(g, {})
    thr = post_process(games)
    for g in games:
        finalize(g, known)
    for g in games:
        if "prev_status" in g:
            mark_change(g)
    summary = {k: sum(1 for g in games if g["status"] == k) for k in ("ok", "warn", "fail", "none")}
    summary["regression"] = sum(1 for g in games if g.get("change") == "regression")
    summary["fixed"] = sum(1 for g in games if g.get("change") == "fixed")
    summary["new_issue"] = sum(1 for g in games if g.get("change") == "new_issue")
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M"), "build": git_head(), "tier": tier, "partial": partial,
           "seconds": round(time.time() - t0), "diverge_threshold": thr, "summary": summary, "games": games}
    save_json(os.path.join(OUT, "results.json"), doc)
    if not partial:
        build_report(doc)


# ---------------------------------------------------------------------------
# Report

def build_report(doc):
    """One self-contained report.html: the pictures are embedded, so the file can be mailed as it is."""
    import io
    from PIL import Image

    def thumb(src, width=300):
        try:
            im = Image.open(src).convert("RGB")
            if im.width > width:
                im = im.resize((width, max(1, im.height * width // im.width)))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=72)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except OSError:
            return None

    rows = []
    for g in doc["games"]:
        rdir = os.path.join(OUT, "runs", g["crc"])
        m = g.get("metrics", {})
        imgs = {}
        if g["status"] != "none" and os.path.isdir(rdir):
            k = m.get("worst", max(0, g.get("samples", 1) - 1))
            for kind in ("hle", "lle", "diff"):
                p = os.path.join(rdir, f"{kind}_{k:02d}.png")
                if os.path.exists(p):
                    imgs[kind] = thumb(p)
            if "hle" not in imgs:
                p = os.path.join(rdir, "route_end.png")
                if os.path.exists(p):
                    imgs["hle"] = thumb(p)
            # every sample of the HLE run as a small strip: the whole stretch at a glance
            strip = [thumb(os.path.join(rdir, f"hle_{i:02d}.png"), 88) for i in range(g.get("samples", 0))]
            imgs["strip"] = [x for x in strip if x]
        rows.append({**g, "imgs": imgs})
    page = PAGE.replace("__DATA__", json.dumps(rows).replace("</", "<\\/"))
    page = page.replace("__META__", html.escape(f"{doc['generated']} - build {doc['build']} - tier {doc['tier']} - {doc['seconds']} s"))
    with open(os.path.join(OUT, "report.html"), "w", encoding="utf-8") as f:
        f.write(page)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orbit64 game QA</title><style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6b7280;--line:#e5e7eb;--card:#f6f7f9;--ok:#2fa84f;--warn:#d98a00;--fail:#d6453d;--none:#8b93a1;--acc:#4f46e5}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaed;--mut:#9aa3af;--line:#2a2e35;--card:#1c1f25}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{padding:20px 24px 8px}h1{margin:0 0 4px;font-size:22px}.meta{color:var(--mut)}
.sum{display:flex;flex-wrap:wrap;gap:10px;padding:8px 24px}.chip{border:1px solid var(--line);background:var(--card);border-radius:10px;padding:8px 14px;cursor:pointer;user-select:none}
.chip b{font-size:20px;display:block}.chip.on{outline:2px solid var(--acc)}
.bar{display:flex;gap:12px;padding:8px 24px;align-items:center;flex-wrap:wrap}input[type=search]{padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);min-width:240px}
select{padding:7px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
.list{padding:0 24px 40px;display:grid;gap:10px}
.g{display:grid;grid-template-columns:minmax(220px,1fr) 3fr;gap:14px;border:1px solid var(--line);background:var(--card);border-radius:12px;padding:12px}
.g h3{margin:0 0 4px;font-size:15px}.badge{display:inline-block;padding:1px 8px;border-radius:99px;color:#fff;font-size:12px;margin-right:6px}
.ok{background:var(--ok)}.warn{background:var(--warn)}.fail{background:var(--fail)}.none{background:var(--none)}.chg{background:var(--acc)}
.rs{margin:6px 0 0;padding:0;list-style:none}.rs li{margin:2px 0}.rs .fail{background:none;color:var(--fail)}.rs .warn{background:none;color:var(--warn)}.rs .info,.rs .known{color:var(--mut)}.rs .known{text-decoration:line-through}
.kv{color:var(--mut);font-size:12px;margin-top:6px}.imgs{display:flex;gap:8px;flex-wrap:wrap}.imgs figure{margin:0}.imgs img{width:240px;max-width:100%;border-radius:6px;display:block;cursor:zoom-in;background:#000}
  .strip{display:flex;gap:3px;margin-top:8px;overflow-x:auto}.strip img{height:50px;border-radius:3px}
figcaption{font-size:11px;color:var(--mut);text-align:center}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9}#modal img{max-width:96vw;max-height:94vh}
@media (max-width:800px){.g{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Orbit64 game QA</h1><div class="meta">__META__</div></header>
<div class="sum" id="sum"></div>
<div class="bar"><input id="q" type="search" placeholder="Filter by name..."><select id="sort"><option value="sev">Worst first</option><option value="name">Name</option><option value="div">HLE/LLE difference</option><option value="fps">Slowest</option></select></div>
<div class="list" id="list"></div><div id="modal"><img id="big"></div>
<script>
const D=__DATA__;let filt="all";
const sev=g=>({fail:3,warn:2,ok:1,none:0})[g.status]*1000+(g.change==="regression"?500:0)+((g.metrics||{}).cells_mean||0)*100;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[c]);
function chips(){const c={all:D.length,fail:0,warn:0,ok:0,none:0,regression:0,fixed:0};D.forEach(g=>{c[g.status]++;if(g.change==="regression")c.regression++;if(g.change==="fixed")c.fixed++});
document.getElementById("sum").innerHTML=Object.entries(c).map(([k,v])=>`<div class="chip${filt===k?" on":""}" data-k="${k}"><b>${v}</b>${({all:"All games",fail:"Failures",warn:"Warnings",ok:"OK",none:"No route",regression:"Regressions",fixed:"Fixed"})[k]}</div>`).join("");
document.querySelectorAll(".chip").forEach(e=>e.onclick=()=>{filt=e.dataset.k;render()})}
function pass(g){if(filt==="all")return true;if(filt==="regression"||filt==="fixed")return g.change===filt;return g.status===filt}
function render(){chips();const q=document.getElementById("q").value.toLowerCase(),s=document.getElementById("sort").value;
let a=D.filter(g=>pass(g)&&g.name.toLowerCase().includes(q));
a.sort(s==="name"?(x,y)=>x.name.localeCompare(y.name):s==="div"?(x,y)=>((y.metrics||{}).cells_mean||0)-((x.metrics||{}).cells_mean||0):s==="fps"?(x,y)=>((x.metrics||{}).fps||1e9)-((y.metrics||{}).fps||1e9):(x,y)=>sev(y)-sev(x));
document.getElementById("list").innerHTML=a.map(g=>{const m=g.metrics||{};const kv=[];
if(g.ucode)kv.push("ucode "+g.ucode);if(m.reached)kv.push("route: "+m.reached+", "+m.route_frames+" frames, replay "+(+m.replay_s).toFixed(1)+" s");
if(m.fps)kv.push((+m.fps).toFixed(0)+" fps");if(m.cells_mean!==undefined)kv.push("HLE/LLE diff "+(100*m.cells_mean).toFixed(0)+"% (worst "+(100*m.cells_max).toFixed(0)+"%)");
if(m.triangles)kv.push(m.triangles+" triangles, "+(100*m.cull).toFixed(0)+"% culled");if(m.audio_db!=null)kv.push("audio "+(+m.audio_db).toFixed(0)+" dB");
const im=g.imgs||{};const fig=(k,l)=>im[k]?`<figure><img src="${im[k]}" data-full="${im[k]}"><figcaption>${l}</figcaption></figure>`:"";
if(g.screen&&g.screen.lines&&g.screen.lines.length)kv.push("reads: "+esc(g.screen.lines.slice(0,8).join(" / ")).slice(0,140)+" ["+g.screen.kind+"]");
const chg=g.change?`<span class="badge chg">${g.change==="new_issue"?"new issue":g.change}${g.was?" (was "+g.was+")":""}</span>`:"";
return `<div class="g"><div><h3>${esc(g.name)} <small style="color:var(--mut)">${esc(g.region||"")} ${esc(g.version||"")}</small></h3><span class="badge ${g.status}">${({ok:"OK",warn:"WARN",fail:"FAIL",none:"NO ROUTE"})[g.status]}</span>${chg}${g.healed?`<span class="badge none">${g.healed}</span>`:""}
<ul class="rs">${(g.reasons||[]).map(r=>`<li class="${r.level}">${esc(r.text)}</li>`).join("")}</ul><div class="kv">${kv.join(" &middot; ")}</div></div>
<div><div class="imgs">${fig("hle","HLE")}${fig("lle","LLE (real microcode)")}${fig("diff","difference")}</div><div class="strip">${(im.strip||[]).map(u=>`<img src="${u}">`).join("")}</div></div></div>`}).join("")||"<p style='padding:12px'>Nothing matches.</p>";
document.querySelectorAll(".imgs img").forEach(i=>i.onclick=()=>{big.src=i.dataset.full;modal.style.display="flex"})}
modal.onclick=()=>modal.style.display="none";q.oninput=render;sort.onchange=render;render();
</script></body></html>"""


def cmd_report(args):
    doc = load_json(os.path.join(OUT, "results.json"))
    if not doc:
        print("no results yet: python tools/qa.py run", file=sys.stderr)
        return 2
    build_report(doc)
    print("report:", os.path.join(OUT, "report.html"))
    return 0


def rebase_one(e, args):
    rdir = os.path.join(OUT, "rebase", e["crc"])
    shutil.rmtree(rdir, ignore_errors=True)
    os.makedirs(rdir, exist_ok=True)
    rc, out, err, to = spawn([QA, os.path.join(ROOT, e["rom"]), "--route", route_path(e), "--out", rdir, "--frames", "30",
                              "--sample", "30", "--max-seconds", "120", "--update-sigs"], 300)
    data = load_json(os.path.join(rdir, "qa.json")) if rc == 0 else None
    if STOP_REQUESTED():
        return None
    if not data:
        return e["name"], "the replay failed"
    return e["name"], ("pictures updated" if data["route"]["route_ok"] else "route no longer works: not touched")


def cmd_rebase(args):
    """Takes the pictures of the current emulator as the recorded ones, for every route (inputs stay)."""
    if not os.path.exists(QA):
        print("build the tools first: make game_qa", file=sys.stderr)
        return 2
    roms = [e for e in select_roms(args) if read_route_head(route_path(e))]
    print(f"{len(roms)} routes, {args.jobs} at a time")
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(rebase_one, e, args): e for e in roms}
        try:
            for fut in gs.as_done(futs):
                r = fut.result()
                done += 1
                if r:
                    print(f"[{done}/{len(roms)}] {r[0]}: {r[1]}", flush=True)
        except KeyboardInterrupt:
            gs.STOP.set()
            gs.kill_children()
            return 1
    return 0


def watch_snapshot(index):
    return {k: {"name": v.get("name"), "reached": v.get("reached"), "attempts": v.get("attempts", 1),
                "stop": v.get("stop", ""), "failed": "failed" in v} for k, v in index.items()}


def watch_view(index, base):
    """What changed since the baseline: how many games were tried again, which ones got further."""
    cand = [k for k, b in base.items() if b["reached"] != "ingame" and b["stop"] != "dead" and not b["failed"] and b["attempts"] < 4]
    # tried: counted again, or kept its old route (the improve run says so), or got further
    tried = [k for k in cand if index.get(k, {}).get("attempts", 1) > base[k]["attempts"] or index.get(k, {}).get("kept")
             or RANK.get(index.get(k, {}).get("reached"), 0) > RANK.get(base[k]["reached"], 0)]
    better = []
    for k in tried:
        now = index[k].get("reached")
        if RANK.get(now, 0) > RANK.get(base[k]["reached"], 0):
            better.append((base[k]["name"], base[k]["reached"], now, index[k].get("frames")))
    counts = {}
    for v in index.values():
        r = v.get("reached") or ("failed" if "failed" in v else "none")
        counts[r] = counts.get(r, 0) + 1
    return cand, tried, better, counts


def cmd_watch(args):
    """Live view of a `record --improve` round: python tools/qa.py watch (Ctrl+C to leave)."""
    index_path = os.path.join(ROUTES, "index.json")
    base_path = os.path.join(OUT, "improve_baseline.json")
    base = load_json(base_path)
    if base is None or args.reset:
        base = watch_snapshot(load_json(index_path, {}))
        save_json(base_path, base)
    page = os.path.join(OUT, "status.html")
    try:
        while True:
            index = load_json(index_path, {})
            cand, tried, better, counts = watch_view(index, base)
            lines = [f"routes now: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
                     f"improve round: {len(tried)} of {len(cand)} games tried, {len(better)} got further", ""]
            for name, was, now, frames in sorted(better):
                lines.append(f"  {name}: {was} -> {now} ({frames} frames)")
            if not better:
                lines.append("  (nothing improved yet)")
            prog = load_json(os.path.join(OUT, "run_progress.json"), None)
            if prog and not prog.get("finished") and time.time() - prog.get("updated", 0) < 600:
                el = prog["updated"] - prog["t0"]
                rate = prog["done"] / el if el > 0 else 0
                eta = (prog["total"] - prog["done"]) / rate / 60 if rate else 0
                lines = [f"{prog['tier']} run: {prog['done']} of {prog['total']} games, "
                         + ", ".join(f"{k} {v}" for k, v in sorted(prog["counts"].items()))
                         + (f", about {eta:.0f} min left" if rate else ""), ""] + ["  " + x for x in prog["last"]]
            text = "\n".join(lines)
            # the Windows console does not always understand ANSI codes: `cls` there
            if os.name == "nt":
                os.system("cls")
                print(time.strftime("%H:%M:%S") + "  (Ctrl+C leaves)\n" + text, flush=True)
            else:
                print("\033[2J\033[H" + time.strftime("%H:%M:%S") + "  (Ctrl+C leaves)\n" + text, flush=True)
            os.makedirs(OUT, exist_ok=True)
            with open(page, "w", encoding="utf-8") as f:
                f.write('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>QA status</title>'
                        '<body style="font:15px/1.5 system-ui;padding:20px;background:#14161a;color:#e8eaed"><h2>Improve round</h2><pre>'
                        + html.escape(text) + "</pre>")
            if args.once:
                return 0
            time.sleep(args.every)
    except KeyboardInterrupt:
        return 0


def cmd_selftest(args):
    """The tools' own tests. A replay must be bit-exact: the same route gives the same machine and the same
    pictures every time - otherwise nothing a run says could be trusted."""
    import hashlib
    if not os.path.exists(QA):
        print("build the tools first: make game_probe game_qa", file=sys.stderr)
        return 2
    roms = [e for e in gs.find_roms(None, None) if read_route_head(route_path(e))]
    if not roms:
        print("no routes yet: python tools/qa.py record --filter Mario", file=sys.stderr)
        return 2
    # a short route keeps the test quick
    roms.sort(key=lambda e: read_route_head(route_path(e))["frames"])
    e = roms[0]
    print(f"game: {e['name']} (route of {read_route_head(route_path(e))['frames']} frames)")
    failures = []

    def once(tag, extra):
        d = os.path.join(OUT, "selftest", tag)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        rc, out, err, to = spawn([QA, os.path.join(ROOT, e["rom"]), "--route", route_path(e), "--out", d, "--frames", "90",
                                  "--sample", "30", "--max-seconds", "120"] + extra, 300)
        data = load_json(os.path.join(d, "qa.json"))
        if rc != 0 or not data:
            failures.append(f"{tag}: game_qa failed (exit {rc})")
            return None, None
        pics = hashlib.sha1(b"".join(open(os.path.join(d, f), "rb").read() for f in sorted(os.listdir(d))
                                     if f.startswith("hle_") and f.endswith(".png"))).hexdigest()
        return data, pics

    for name, extra in (("plain", []), ("mash", ["--mash"])):
        a, pa = once(name + "1", extra)
        b, pb = once(name + "2", extra)
        if a is None or b is None:
            continue
        ok_hash = a["route"]["end_hash"] == b["route"]["end_hash"]
        ok_pics = pa == pb
        ok_cnt = all(a["hle"][k] == b["hle"][k] for k in ("triangles", "gfx_tasks", "audio_tasks", "vi_swaps", "pixels_drawn"))
        print(f"{name:6} replay end state {'same' if ok_hash else 'DIFFERENT'}, pictures {'same' if ok_pics else 'DIFFERENT'}, "
              f"counters {'same' if ok_cnt else 'DIFFERENT'}")
        if not (ok_hash and ok_pics and ok_cnt):
            failures.append(f"{name}: two replays of the same route differ - the emulator is not deterministic here")
    # HLE and LLE start from the same machine: the comparison of a game with itself must find nothing
    data, _ = once("lle", ["--lle", "--lle-frames", "60"])
    if data and data.get("lle") and "error" not in data["lle"]:
        print(f"lle    started from the HLE state, {len(data['compare'])} pictures compared")
    elif data:
        failures.append("lle: the low-level run could not start from the high-level state")
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nselftest passed")
    return 0


def cmd_routes(args):
    roms = select_roms(args)
    index = load_json(os.path.join(ROUTES, "index.json"), {})
    counts = {}
    for e in roms:
        head = read_route_head(route_path(e))
        ent = index.get(e["crc"], {})
        state = head["reached"] if head else ("failed: " + ent["failed"] if "failed" in ent else "none")
        counts[state.split(":")[0]] = counts.get(state.split(":")[0], 0) + 1
        if args.verbose or not head:
            print(f"{state:22} {e['name']}")
    print("\n" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--list", help="a file with ROM file names (tools/test_set.txt)")
        p.add_argument("--filter", help="only ROMs whose file name contains this")
        p.add_argument("--jobs", type=int, default=4)
        p.add_argument("--max-seconds", type=float, default=240, help="wall time of one probe (recording)")

    r = sub.add_parser("record", help="find a route for every game")
    common(r)
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--force", action="store_true", help="record again even where a route exists")
    r.add_argument("--no-advice", dest="advice", action="store_false",
                   help="do not read the screen (OCR) to decide what to press: the probe goes by pictures alone")
    r.add_argument("--max-attempts", type=int, default=4, help="with --improve: give up on a game after this many attempts in all")
    r.add_argument("--improve", action="store_true", help="go on from the end of routes that do not reach gameplay (3 tries per game)")  # 4 attempts in all
    u = sub.add_parser("run", help="replay routes, judge the pictures")
    u.add_argument("tier", nargs="?", default="render", choices=list(TIERS))
    common(u)
    u.add_argument("--no-lle", action="store_true", help="skip the HLE/LLE comparison")
    u.add_argument("--fresh", action="store_true", help="run everything again, even what is up to date")
    u.add_argument("--no-heal", action="store_true", help="do not re-record stale routes")
    u.add_argument("--record-missing", action="store_true", help="record a route where there is none")
    u.add_argument("--checkpoint", action="store_true", help="start from the saved checkpoints (quick look; skips the boot)")
    u.add_argument("--baseline", help="compare with this results.json instead of the last run")
    u.add_argument("--strict", action="store_true", help="exit with code 3 if there are failures or regressions (for scripts / CI)")
    d = sub.add_parser("seed", help="routes from the old probe runs in test_output/game_status (a replay each, no search)")
    common(d)
    d.add_argument("--verbose", "-v", action="store_true")
    b = sub.add_parser("rebase", help="take the pictures of the current emulator as the recorded ones")
    common(b)
    w = sub.add_parser("watch", help="live view of a record --improve round (also writes test_output/qa/status.html)")
    w.add_argument("--every", type=float, default=5, help="seconds between updates")
    w.add_argument("--once", action="store_true", help="print once and leave")
    w.add_argument("--reset", action="store_true", help="take the current routes as the starting point")
    sub.add_parser("report", help="rebuild report.html")
    sub.add_parser("selftest", help="check that replays are reproducible")
    t = sub.add_parser("routes", help="list the games without a route")
    common(t)
    t.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    return {"record": cmd_record, "run": cmd_run, "report": cmd_report, "routes": cmd_routes,
            "selftest": cmd_selftest, "seed": cmd_seed, "rebase": cmd_rebase, "watch": cmd_watch}[args.cmd](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        gs.STOP.set()
        gs.kill_children()
        sys.exit(130)
