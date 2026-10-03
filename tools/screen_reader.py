#!/usr/bin/env python3
"""What is on an emulator screen, and which button to press: OCR + rules.

    python tools/screen_reader.py screen.png            # kind, options, the highlighted one, what to press
    python tools/screen_reader.py screen.png --json     # the same as data (what qa.py's advice loop answers with)

`advise(image)` reads the text (tools/ocr_engines.py), works out what kind of screen it is - "PRESS START", a
menu, a yes/no question, a pause screen, a warning, a text box, a game's HUD - finds the options of a menu and the
one the cursor is on, and returns candidates: short controller macros, best first, each with the reason. They are
*hypotheses*: the prober (tools/game_probe.cpp) tries them and keeps the one that changes the picture.

A macro is a list of [buttons, stick_x, stick_y, frames] segments (buttons are the N64 mask, 0x8000 = A, 0x1000 =
START, 0x4000 = B, 0x0800/0x0400/0x0200/0x0100 = D-pad up/down/left/right; the stick is -80..80).
"""
import argparse
import dataclasses
import json
import os
import re
import sys
from typing import List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocr_engines as oe  # noqa: E402
from ocr_bench import approx_in  # noqa: E402

A, B, Z, START = 0x8000, 0x4000, 0x2000, 0x1000
DUP, DDOWN, DLEFT, DRIGHT = 0x0800, 0x0400, 0x0200, 0x0100

# ---------------------------------------------------------------------------
# Vocabulary (letters and digits only, upper case: see ocr_engines.norm)

HUD = ["LAP", "LAPTIME", "TIME", "SCORE", "LIVES", "AMMO", "HEALTH", "SPEED", "MPH", "KMH", "POSITION", "POS", "ROUND", "LEVEL",
       "ENEMIES", "WEAPON", "SHIELD", "FUEL", "DAMAGE", "GEAR", "RANK", "ARMOR", "STAMINA", "BEST", "TOTALTIME", "BOOST", "TURBO",
       "HITPOINTS", "GOLD", "KILLED", "COINS", "STARS", "ENERGY", "POWER", "RECORD", "PLAYER", "TARGET", "RADAR", "PERIOD", "QTR",
       "DOWN", "YARDS", "BALL", "STRIKE", "OUT", "INNING"]

# options worth choosing to get into a game: (word, weight)
GOOD = [("STARTGAME", 3), ("NEWGAME", 3), ("1PLAYER", 3), ("ONEPLAYER", 3), ("SINGLEPLAYER", 3), ("QUICKRACE", 3), ("QUICKPLAY", 3),
        ("QUICKSTART", 3), ("ARCADE", 3), ("PLAYGAME", 3), ("PLAY", 2), ("START", 2), ("EXHIBITION", 2), ("SEASON", 2), ("STORY", 2),
        ("ADVENTURE", 2), ("CAMPAIGN", 2), ("CHAMPIONSHIP", 2), ("TOURNAMENT", 2), ("MISSION", 2), ("RACE", 2), ("FREEPLAY", 2),
        ("SINGLE", 2), ("PRACTICE", 1), ("TIMEATTACK", 1), ("TRAINING", 1), ("VERSUS", 1), ("BEGIN", 2), ("GO", 1), ("ENGLISH", 2),
        ("NORMAL", 1), ("CONTINUE", 1), ("YES", 1), ("OK", 1), ("SELECT", 1), ("RESUME", 2), ("TRYAGAIN", 2), ("RETRY", 2)]
# options that lead away from the game
BAD = [("OPTIONS", -3), ("OPTION", -3), ("SETTINGS", -3), ("CONFIG", -3), ("CONTROLLER", -3), ("CONTROLS", -3), ("SOUND", -3),
       ("MUSIC", -3), ("VOLUME", -3), ("AUDIO", -3), ("CREDITS", -3), ("EXIT", -3), ("QUIT", -3), ("EXTRAS", -2), ("LOAD", -2),
       ("SAVE", -2), ("MEMORY", -3), ("PAK", -3), ("ERASE", -3), ("DELETE", -3), ("COPY", -3), ("PASSWORD", -3), ("CODES", -2),
       ("CHEATS", -2), ("RECORDS", -2), ("HIGHSCORE", -2), ("STATS", -2), ("HELP", -2), ("INSTRUCTIONS", -3), ("TUTORIAL", -3),
       ("DEMO", -2), ("MOVIES", -2), ("GALLERY", -2), ("2PLAYER", -2), ("TWOPLAYER", -2), ("MULTIPLAYER", -2), ("3PLAYER", -2),
       ("4PLAYER", -2), ("NO", -2), ("CANCEL", -3), ("BACK", -2), ("RETURN", -2), ("GOBACK", -2), ("NOTHANKS", -1), ("DONTEXIT", -3),
       ("DEFAULT", -2), ("RESET", -3)]

QUESTION_YES = ["WITHOUTSAVING", "NOTBESAVED", "NOTSAVED", "NOCONTROLLERPAK", "WITHOUTPAK", "CONTINUEWITHOUT", "PLAYAGAIN", "TRYAGAIN",
                "CONTINUE", "RETRY"]
QUESTION_NO = ["QUIT", "EXIT", "ERASE", "DELETE", "OVERWRITE", "RESET", "ENDGAME", "ARCHIVE", "REPLACE", "INSTRUCTIONS", "TUTORIAL",
               "HOWTOPLAY", "SEEAGAIN", "VIEW", "REPEAT", "EXPLAIN", "TELLME", "MOREINFO", "HEARAGAIN"]
# a dialog that says something went wrong: the way on is BACK (B), not RETRY
ERR_DIALOG = ["NOTFOUND", "DOESNOTCONTAIN", "NOTCONTAIN", "NOSAVED", "COULDNOT", "CANNOTFIND", "NOTENOUGH", "ISFULL", "FAILEDTO", "NOCAMPAIGN",
              "NOGAMESAVED", "NODATA", "CORRUPTED", "CANNOTBE"]
CREDITS_WORDS = ["EXECUTIVEPRODUCER", "LEADPROGRAMMER", "PROGRAMMERS", "LEADARTIST", "LEADDESIGNER", "SPECIALTHANKS", "MUSICBY", "SOUNDBY",
                 "PRODUCEDBY", "DIRECTEDBY", "QUALITYASSURANCE", "ARTISTS", "PRODUCER", "DESIGNER", "PROGRAMMER", "ANIMATION"]
ANSWER_YES = ["YES", "SURE", "OK", "CONTINUE", "RETRY", "TRYAGAIN", "AGAIN", "ACCEPT", "PLAY", "START"]
ANSWER_NO = ["NO", "NOTHANKS", "CANCEL", "NOPE", "DONTTRYAGAIN", "DONT", "BACK"]

WARN = ["WARNING", "MALFUNCTION", "NOTINSERTED", "NOTCONNECTED", "NOTDESIGNED", "ERROR", "CORRUPT", "DAMAGED", "CHECK", "PLEASEINSERT",
        "RUMBLEPAK", "CONTROLLERPAK", "TRANSFERPAK", "WILLNOTBESAVED", "NOTBESAVED", "FAILED", "UNABLE"]
LEGAL = ["TRADEMARK", "ALLRIGHTSRESERVED", "COPYRIGHT", "LICENSED", "NINTENDO", "RESERVED", "PRODUCTIONS", "ENTERTAINMENT", "DEVELOPED",
         "PRESENTS", "WWW", "COM"]
PAUSE = ["PAUSE", "PAUSED", "GAMEPAUSED", "RESUME"]
OPTIONS_TITLE = ["OPTIONS", "OPTION", "SETTINGS", "GAMEOPTIONS", "CONFIGURE", "CONFIGURATION", "CONTROLLERSETUP", "SOUNDOPTIONS"]
GAMEOVER = ["GAMEOVER", "INSERTCOIN", "CONTINUE", "TRYAGAIN", "PLAYAGAIN", "RETRY", "YOULOSE", "YOUDIED"]


def has(norm_text: str, words, err_div=5) -> bool:
    for w in words:
        if not w:
            continue
        # short words must match exactly (as whole words if possible), longer ones with a tolerance
        if len(w) <= 4:
            if re.search(r"(?<![A-Z0-9])" + w + r"(?![A-Z0-9])", norm_text):
                return True
        elif approx_in(w, norm_text, len(w) // err_div):
            return True
    return False


def word_hit(text: str, vocab, whole=True):
    """Weight of the best vocabulary entry the text matches (0 if none): exact whole-word match for short entries,
    tolerant for long ones. `text` is a raw line."""
    n = oe.norm(text)
    best = 0
    for w, wt in vocab:
        if len(w) <= 4:
            ok = re.search(r"(?<![A-Z0-9])" + w + r"(?![A-Z0-9])", re.sub(r"[^A-Za-z0-9]+", " ", text).upper().replace(" ", "")) is not None \
                if False else (n == w or n.startswith(w) and len(n) <= len(w) + 3)
        else:
            ok = approx_in(w, n, len(w) // 6) and len(n) <= len(w) + 8
        if ok and abs(wt) > abs(best):
            best = wt
    return best


# ---------------------------------------------------------------------------
# Result types

@dataclasses.dataclass
class Option:
    text: str
    box: tuple
    score: int = 0
    alts: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Candidate:
    name: str
    macro: list
    score: float
    why: str


@dataclasses.dataclass
class Advice:
    kind: str = ""          # "" while undecided; "unknown" is reported when nothing fitted
    confidence: float = 0.0
    lines: list = dataclasses.field(default_factory=list)
    options: list = dataclasses.field(default_factory=list)
    layout: str = ""            # "column" | "row" | ""
    selected: Optional[int] = None
    candidates: list = dataclasses.field(default_factory=list)
    notes: list = dataclasses.field(default_factory=list)   # things worth reporting (warnings read from the screen)

    def to_json(self):
        d = dataclasses.asdict(self)
        return d


# ---------------------------------------------------------------------------
# Macros

def tap(buttons, frames=6):
    return [[buttons, 0, 0, frames]]


def wait(frames):
    return [[0, 0, 0, frames]]


def move(direction, times, dpad=False):
    """Cursor moves: `times` presses of up/down/left/right, each held 8 frames and released for 14."""
    stick = {"up": (0, 80), "down": (0, -80), "left": (-80, 0), "right": (80, 0)}[direction]
    pad = {"up": DUP, "down": DDOWN, "left": DLEFT, "right": DRIGHT}[direction]
    seg = []
    for _ in range(times):
        seg += [[pad, 0, 0, 8] if dpad else [0, stick[0], stick[1], 8], [0, 0, 0, 14]]
    return seg


# ---------------------------------------------------------------------------
# Menu geometry

def find_stacks(boxes):
    """Every group of short text boxes laid out in a column (same left edge or the same centre) or in a row:
    a list of (layout, [boxes in order]). A screen has several (the HUD's labels are a column too)."""
    def label(b):
        """A menu entry is a short label: a few letters, up to three words, not mostly digits."""
        n = oe.norm(b.text)
        words = len(re.findall(r"[A-Za-z0-9']+", b.text))
        digits = sum(c.isdigit() for c in n)
        return 2 <= len(n) <= 24 and words <= 3 and digits <= len(n) * 0.4 and b.h >= 5

    cand = [b for b in boxes if label(b)]
    groups, seen = [], set()
    for layout in ("column", "row"):
        key = (lambda b: b.cy) if layout == "column" else (lambda b: b.cx)
        for i, a in enumerate(cand):
            group = [a]
            for j, b in enumerate(cand):
                if j == i:
                    continue
                if layout == "column":
                    aligned = abs(a.cx - b.cx) <= 14 or abs(a.x0 - b.x0) <= 8
                    near = abs(a.cy - b.cy) > a.h * 0.8
                else:
                    aligned = abs(a.cy - b.cy) <= max(6, a.h * 0.6)
                    near = abs(a.cx - b.cx) > a.w * 0.4
                if aligned and near and abs(a.h - b.h) <= max(5, a.h * 0.5):
                    group.append(b)
            group.sort(key=key)
            out = []
            for b in group:            # one entry per position (the same line read twice)
                if out and abs(key(b) - key(out[-1])) < 4:
                    continue
                out.append(b)
            if len(out) >= 2:
                sig = (layout, tuple(id(b) for b in out))
                if sig not in seen:
                    seen.add(sig)
                    groups.append((layout, out))
    return groups


def pick_stack(groups):
    """The group that is a menu: the one whose entries are words a menu has (start, options, yes, no, continue...),
    then the larger one."""
    def vocab(b):
        return any(word_hit(t, GOOD) or word_hit(t, BAD) or oe.norm(t) in ("YES", "NO", "OK") for t in [b.text] + b.alts)
    best, best_key = ("", []), (-1, -1)
    for layout, g in groups:
        k = (sum(1 for b in g if vocab(b)), len(g))
        if k > best_key:
            best, best_key = (layout, g), k
    return best


def highlighted(img: Image.Image, opts, layout):
    """Index of the option the cursor is on, or None. Three signs, strongest first: a small marker (arrow, hand,
    bullet) just left of the text that the others do not have; a highlight bar that makes one row's background
    different; the text itself brighter / more saturated than the rest."""
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    H, W = lum.shape

    def clip(x0, y0, x1, y1):
        x0, x1 = int(max(0, np.floor(x0))), int(min(W, np.ceil(x1)))
        y0, y1 = int(max(0, np.floor(y0))), int(min(H, np.ceil(y1)))
        return x0, y0, x1, y1

    n = len(opts)
    marker, text_l, text_s, band_l, band_s = [], [], [], [], []
    for b in opts:
        # marker: pixels in the strip left of the text that differ from the strip's own median
        x0, y0, x1, y1 = clip(b.x0 - 22, b.y0, b.x0 - 2, b.y1)
        if x1 > x0 + 3 and y1 > y0 + 2:
            s = lum[y0:y1, x0:x1]
            marker.append(float((np.abs(s - np.median(s)) > 55).mean()))
        else:
            marker.append(0.0)
        x0, y0, x1, y1 = clip(b.x0, b.y0, b.x1, b.y1)
        s = lum[y0:y1, x0:x1]
        t = sat[y0:y1, x0:x1]
        text_l.append(float(np.percentile(s, 90)) if s.size else 0.0)
        text_s.append(float(np.percentile(t, 90)) if t.size else 0.0)
        # band: the whole row, across the widest option, above and below the text a little
        if layout == "column":
            bx0, bx1 = min(o.x0 for o in opts) - 6, max(o.x1 for o in opts) + 6
            x0, y0, x1, y1 = clip(bx0, b.y0 - 1, bx1, b.y1 + 1)
        else:
            x0, y0, x1, y1 = clip(b.x0 - 4, b.y0 - 3, b.x1 + 4, b.y1 + 3)
        s = lum[y0:y1, x0:x1]
        t = sat[y0:y1, x0:x1]
        band_l.append(float(np.median(s)) if s.size else 0.0)
        band_s.append(float(np.median(t)) if t.size else 0.0)

    def outlier(vals, z=1.7, floor=12.0):
        v = np.array(vals)
        if n < 3:
            # two options: the difference itself is the sign
            d = abs(v[0] - v[1])
            return (int(v.argmax()) if d >= floor * 1.5 else None)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
        dev = (v - med) / max(mad, floor / 2)
        i = int(np.abs(dev).argmax())
        return i if abs(dev[i]) >= z and abs(v[i] - med) >= floor else None

    m = np.array(marker)
    if m.max() >= 0.10 and (n < 2 or np.sort(m)[-1] >= 2.0 * max(np.sort(m)[-2], 0.03)):
        return int(m.argmax()), "marker"
    i = outlier(band_l) if outlier(band_l) is not None else outlier(band_s, floor=18)
    if i is not None:
        return i, "bar"
    i = outlier(text_l) if outlier(text_l) is not None else outlier(text_s, floor=25)
    if i is not None:
        return i, "text"
    return None, ""


# ---------------------------------------------------------------------------
# Deciding

def answer_kind(texts):
    """'no' / 'yes' / None for the text of an answer ("Don't try Again?" is a no, "Try Again?" a yes)."""
    for t in texts:
        n = oe.norm(t)
        if re.match(r"^(NO|NOT|DONT|DON|CANCEL|NEVER|NOPE|BACK|DISCARD)", n) or "DONT" in n:
            return "no"
    for t in texts:
        n = oe.norm(t)
        if any(w in n for w in ("YES", "SURE", "OK", "CONTINUE", "RETRY", "TRYAGAIN", "AGAIN", "ACCEPT", "PLAY", "START", "PROCEED")):
            return "yes"
    return None


def _cursor_macro(layout, si, ti, dpad):
    """Macro that moves the cursor from option si to option ti and confirms with A."""
    if ti == si:
        return tap(A)
    if layout == "row":
        d = "right" if ti > si else "left"
    else:
        d = "down" if ti > si else "up"
    return move(d, abs(ti - si), dpad) + wait(6) + tap(A)


def _option_candidates(adv, layout, opts, sel, target, base_score, why, out):
    """Candidates that select option `target` (an index): from the highlighted one if known, else from the top."""
    dpad_variants = (False, True)
    for dp in dpad_variants:
        if sel is not None:
            out.append(Candidate(f"{'DPAD' if dp else 'STICK'} to '{opts[target].text}' + A", _cursor_macro(layout, sel, target, dp),
                                 base_score - (0.04 if dp else 0.0), why))
        else:
            out.append(Candidate(f"{'DPAD' if dp else 'STICK'} from top to '{opts[target].text}' + A",
                                 _cursor_macro(layout, 0, target, dp), base_score - 0.15 - (0.04 if dp else 0.0), why + " (cursor not seen: counted from the top)"))
            if target > 0:   # in case the cursor was not at the top: go up first (menus that do not wrap stop at the top)
                out.append(Candidate(f"{'DPAD' if dp else 'STICK'} to top then '{opts[target].text}' + A",
                                     move("up" if layout != "row" else "left", len(opts), dp) + wait(6) + _cursor_macro(layout, 0, target, dp),
                                     base_score - 0.2 - (0.04 if dp else 0.0), why + " (after going to the top)"))


def join_boxes(boxes):
    """Words the OCR returned one by one ("NEW", "GAME") become one line when they sit in the same row, close together."""
    boxes = sorted(boxes, key=lambda b: (round(b.cy / max(b.h, 1) * 0.5), b.x0))
    out = []
    for b in boxes:
        prev = out[-1] if out else None
        if (prev is not None and abs(prev.cy - b.cy) < 0.45 * max(prev.h, b.h) and -2 <= b.x0 - prev.x1 <= 1.0 * max(prev.h, b.h)
                and abs(prev.h - b.h) <= 0.5 * max(prev.h, b.h)):
            joined = oe.Box(prev.text + " " + b.text, min(prev.conf, b.conf), prev.x0, min(prev.y0, b.y0), b.x1, max(prev.y1, b.y1), prev.engine)
            joined.alts = []
            out[-1] = joined
        else:
            out.append(b)
    return out


def advise(img: Image.Image, engines=None, boxes=None) -> Advice:
    adv = Advice()
    if boxes is None:
        engines = engines or [e for e in ("rapid", "win") if e in oe.available_engines()]
        boxes = oe.read_screen(img, tuple(engines), (2,))
    boxes = join_boxes(boxes)
    adv.lines = [b.text for b in boxes if oe.norm(b.text)]
    # every reading counts for finding a word, the chosen one is what is shown
    alltext = " ".join(adv.lines + [a for b in boxes for a in b.alts])
    nall = oe.norm(alltext)
    cand: List[Candidate] = []

    layout, stack = pick_stack(find_stacks(boxes))
    def opt_score(b):
        texts = [b.text] + b.alts
        return max((word_hit(t, GOOD) for t in texts), default=0) + min((word_hit(t, BAD) for t in texts), default=0)

    opts = [Option(b.text, (b.x0, b.y0, b.x1, b.y1), opt_score(b), list(b.alts)) for b in stack]
    adv.options, adv.layout = opts, layout
    sel = None
    if opts:
        sel, how = highlighted(img, stack, layout)
        adv.selected = sel
    nlines = [oe.norm(t) for t in adv.lines]
    nlines_all = [oe.norm(" ".join([b.text] + b.alts)) for b in boxes if oe.norm(b.text)]

    # --- 1. press start / press a
    ps = has(nall, ["PRESSSTART", "PUSHSTART", "PRESSSTARTBUTTON", "STARTBUTTON", "PRESSENTER"]) or any(n in ("START", "PRESSSTART") for n in nlines)
    pa = has(nall, ["PRESSA", "PUSHA", "PRESSANYBUTTON", "PRESSABUTTON", "PRESSANYKEY"])
    if ps:
        adv.kind, adv.confidence = "press_start", 0.9
        cand.append(Candidate("START", tap(START), 0.95, "the screen says PRESS START"))
        cand.append(Candidate("A", tap(A), 0.5, "PRESS START: A as well"))
    elif pa:
        adv.kind, adv.confidence = "press_a", 0.85
        cand.append(Candidate("A", tap(A), 0.9, "the screen says PRESS A / ANY BUTTON"))
        cand.append(Candidate("START", tap(START), 0.5, "any button"))

    # --- 2. warnings (also worth reporting: they are often what an emulator gets wrong)
    warn = [t for t, n in zip(adv.lines, nlines) if has(n, ["WARNING", "MALFUNCTION", "NOTINSERTED", "NOTCONNECTED", "NOTDESIGNED", "ERROR", "FAILED", "UNABLE"])]
    if warn and not adv.kind:
        pass
    if has(nall, WARN) and (has(nall, ["WARNING", "MALFUNCTION", "NOTINSERTED", "NOTDESIGNED", "ERROR", "CORRUPT", "FAILED", "UNABLE", "PLEASEINSERT", "WILLNOTBESAVED", "NOTBESAVED"])):
        adv.notes = [t for t in adv.lines if oe.norm(t)]
        if not adv.kind or adv.kind in ("press_a",):
            adv.kind, adv.confidence = "warning", 0.8
        if not has(nall, ["WITHOUTSAVING", "CONTINUEWITHOUT"]) or not opts:
            cand.append(Candidate("A", tap(A), 0.8, "a warning: A dismisses it"))
            cand.append(Candidate("START", tap(START), 0.7, "a warning: START dismisses it"))

    # --- 2b. a dialog about something that is not there / failed: BACK, not RETRY
    if has(nall, ERR_DIALOG) and not adv.kind:
        adv.kind, adv.confidence = "error_dialog", 0.85
        adv.notes = [t for t in adv.lines if oe.norm(t)]
        cand.append(Candidate("B", tap(B), 0.92, "an error dialog (not found / failed): B goes back; RETRY would only repeat it"))
        cand.append(Candidate("START", tap(START), 0.4, "an error dialog: START"))

    # --- 2c. credits roll: leave them
    if not adv.kind and sum(1 for w in CREDITS_WORDS if has(nall, [w])) >= 2:
        adv.kind, adv.confidence = "credits", 0.8
        cand.append(Candidate("B", tap(B), 0.85, "the credits: B leaves them"))
        cand.append(Candidate("START", tap(START), 0.8, "the credits: START skips them"))

    # --- 3. a question with answers
    qline = next((t for t, n in zip(adv.lines, nlines) if "?" in t and len(n) >= 6), None)
    if qline and not adv.kind:
        # The answers are the lines under the question that read as yes / no (they need not form a column with
        # anything else on the screen).
        qb = next((b for b in boxes if b.text == qline), None)
        if qb is not None:
            below = [b for b in boxes if b is not qb and qb.y1 - 3 <= b.y0 <= qb.y1 + 110 and qb.x0 - 60 <= b.x0 <= qb.x1 + 60
                     and answer_kind([b.text] + b.alts)]
            below.sort(key=lambda b: (b.cy, b.x0))
            if len(below) >= 2 or (below and has(oe.norm(qline), ["WITHOUTSAVING", "AREYOUSURE"])):
                layout = "column"
                opts = [Option(b.text, (b.x0, b.y0, b.x1, b.y1), 0, list(b.alts)) for b in below]
                adv.options, adv.layout = opts, layout
                sel, _ = highlighted(img, below, layout) if len(below) >= 2 else (None, "")
                adv.selected = sel
    answers = [o for o in opts if answer_kind([o.text] + o.alts)]
    if (qline or has(nall, ["WITHOUTSAVING", "AREYOUSURE", "DOYOUWANT"])) and len(opts) >= 2 and len(answers) >= 1 and not adv.kind:
        adv.kind, adv.confidence = "prompt", 0.85
        q = oe.norm(qline or alltext)
        want_yes = None
        if has(q, QUESTION_YES):
            want_yes = True
        if has(q, QUESTION_NO) and not has(q, ["WITHOUTSAVING", "CONTINUEWITHOUT"]):
            want_yes = False
        if has(q, ["WITHOUTSAVING", "NOTBESAVED", "CONTINUEWITHOUT"]):
            want_yes = True
        t_idx = None
        for i, o in enumerate(opts):
            k = answer_kind([o.text] + o.alts)
            if (want_yes is True and k == "yes") or (want_yes is False and k == "no"):
                t_idx = i
                break
        if t_idx is None:
            t_idx = 0
        why = f"question: {qline or '?'} -> {'yes' if want_yes else 'no' if want_yes is False else 'the first answer'}"
        # Many games put the answers on buttons ("A Yes, please / B No, thank you"): the button itself first.
        if want_yes is False:
            cand.append(Candidate("B", tap(B), 0.93, why + " (B is usually the no / cancel button)"))
        elif want_yes is True:
            cand.append(Candidate("A", tap(A), 0.91, why + " (A is usually the yes button)"))
        _option_candidates(adv, layout, opts, sel, t_idx, 0.9, why, cand)

    # --- 4. pause
    if has(nall, PAUSE) and adv.kind not in ("prompt",):
        adv.kind, adv.confidence = "pause", 0.8
        cand.append(Candidate("START", tap(START), 0.85, "a pause screen: START resumes"))
        res = next((i for i, o in enumerate(opts) if word_hit(o.text, [("RESUME", 1), ("CONTINUE", 1)])), None)
        if res is not None:
            _option_candidates(adv, layout, opts, sel, res, 0.8, "resume the game", cand)
        else:
            cand.append(Candidate("A", tap(A), 0.5, "a pause screen: A confirms the highlighted entry"))

    # --- 5. options / settings screens (leave them)
    if (adv.kind not in ("prompt", "pause", "press_start", "error_dialog", "credits") and has(nall, OPTIONS_TITLE) and len(opts) >= 3
            and not any(o.score >= 2 for o in opts)):
        adv.kind, adv.confidence = "options", 0.7
        cand.append(Candidate("B", tap(B), 0.85, "an options screen: B goes back"))
        ret = next((i for i, o in enumerate(opts) if word_hit(o.text, [("RETURN", 1), ("BACK", 1), ("GOBACK", 1), ("EXIT", 1), ("DONE", 1)])), None)
        if ret is not None:
            _option_candidates(adv, layout, opts, sel, ret, 0.8, "leave the options screen", cand)
        cand.append(Candidate("START", tap(START), 0.5, "an options screen: START may leave it"))

    # --- 6. game over / continue
    if adv.kind not in ("prompt", "pause", "options", "press_start") and has(nall, ["GAMEOVER", "INSERTCOIN", "TRYAGAIN", "PLAYAGAIN", "CONTINUE?"]):
        adv.kind, adv.confidence = "game_over", 0.7
        ta = next((i for i, o in enumerate(opts) if word_hit(o.text, [("TRYAGAIN", 1), ("CONTINUE", 1), ("RETRY", 1), ("YES", 1), ("PLAYAGAIN", 1)])), None)
        if ta is not None:
            _option_candidates(adv, layout, opts, sel, ta, 0.85, "play again", cand)
        cand.append(Candidate("START", tap(START), 0.7, "game over: START continues"))
        cand.append(Candidate("A", tap(A), 0.65, "game over: A continues"))

    # --- 7. a menu: choose the option that starts a game
    # (Two short words and a bare number - a round timer, a score - are a game's HUD, the names of two fighters
    # above their health bars, not a menu.)
    hud_numbers = sum(1 for t in adv.lines if oe.norm(t) and oe.norm(t).isdigit())
    if not adv.kind and len(opts) >= 2 and not (len(opts) < 3 and hud_numbers >= 1):
        scored = [(o.score, -i) for i, o in enumerate(opts)]
        best = max(range(len(opts)), key=lambda i: (opts[i].score, -i))
        if opts[best].score > 0:
            adv.kind, adv.confidence = "menu", 0.7 + 0.05 * min(opts[best].score, 3)
            _option_candidates(adv, layout, opts, sel, best, 0.8 + 0.03 * min(opts[best].score, 3), f"'{opts[best].text}' leads into a game", cand)
            if sel is None or sel == best or opts[sel].score >= 0:
                cand.append(Candidate("A", tap(A), 0.72 if sel is None else 0.95, "confirm what the cursor is on"))
        elif all(o.score <= 0 for o in opts) and sel is not None and not has(nall, LEGAL):
            # nothing recognisable but a cursor seen: the highlighted entry (usually the default) or the first
            adv.kind, adv.confidence = "menu", 0.45
            cand.append(Candidate("A", tap(A), 0.6, "a menu with nothing recognisable: confirm the highlighted entry"))
            for i, o in enumerate(opts[:3]):
                if o.score >= 0 and i != sel:
                    _option_candidates(adv, layout, opts, sel, i, 0.45, f"try '{o.text}'", cand)

    # --- 8. legal text / dialog / HUD
    if not adv.kind:
        if has(nall, LEGAL) and not has(nall, HUD):
            adv.kind, adv.confidence = "legal", 0.6
            cand.append(Candidate("START", tap(START), 0.6, "legal / logo screen: skip with START"))
            cand.append(Candidate("A", tap(A), 0.55, "legal / logo screen: skip with A"))
        else:
            hud = [w for w in HUD if has(nall, [w])]
            # a clock, a fraction (lap 1/4) and the like are HUD too
            hud += ["clock"] * bool(re.search(r"\d{1,2}[:.]\d{2}", alltext)) + ["fraction"] * bool(re.search(r"\d+\s*/\s*\d+", alltext))
            sentence = any(len(t) > 28 and re.search(r"[a-z]{3,}", t) for t in adv.lines)
            if len(hud) >= 2:
                adv.kind, adv.confidence = "gameplay", min(0.95, 0.5 + 0.12 * len(hud))
            elif sentence:
                adv.kind, adv.confidence = "dialog", 0.55
                cand.append(Candidate("A", tap(A), 0.7, "a text box: A reads on"))
            elif not adv.lines:
                adv.kind, adv.confidence = "no_text", 0.3
    if not adv.kind:
        adv.kind = "unknown"
    adv.candidates = sorted(cand, key=lambda c: -c.score)
    # one of each
    seen, uniq = set(), []
    for c in adv.candidates:
        k = json.dumps(c.macro)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    adv.candidates = uniq
    return adv


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # the Windows console is not UTF-8
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--engines", nargs="*")
    args = ap.parse_args()
    adv = advise(Image.open(args.image), args.engines)
    if args.json:
        print(json.dumps(adv.to_json(), indent=1))
        return 0
    print(f"kind: {adv.kind} ({adv.confidence:.2f})")
    print("text:", " | ".join(adv.lines))
    if adv.options:
        print(f"options ({adv.layout}):", ", ".join(("*" if i == adv.selected else "") + o.text + f"[{o.score}]" for i, o in enumerate(adv.options)))
    for c in adv.candidates[:6]:
        print(f"  {c.score:.2f} {c.name}: {c.why}")
    if adv.notes:
        print("notes:", adv.notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
