# Game QA

Does every game get to its gameplay, and is it drawn right there? One command answers it for all
290 ROMs, in minutes, and says *what* is wrong instead of leaving you to look at 290 screenshots.

```
make game_probe game_qa                       # once (and after changing the emulator)
python tools/qa.py record                     # once: find a route into every game (about an hour)
python tools/qa.py run                        # every day: replay, judge, report (minutes)
start test_output/qa/report.html
```

## The idea

* **Routes.** `game_probe` plays a game on its own: from a save state it tries "nothing", A, START and
  the stick, and sees which one the game reacts to (the emulator is deterministic, so the difference between the
  runs *is* the reaction). The winning inputs, frame by frame, are written to `tools/routes/<CRC>.route`
  - a text file of a few hundred bytes. Commit them: they are the "walkthrough" of every game.
* **Mash.** Before the careful search (and now and then during it) the probe does what a person does: it taps
  START and A in turn for a few hundred frames and checks whether the stick then moves the picture. Many games
  are in within seconds this way; the experiments are for the rest.
* **Replay.** `game_qa` plays a route back from power-on. A game that took the probe a minute of experiments
  is at its gameplay in 5-15 seconds, and always at the same spot.
* **Judging.** From there the game is played for a while (steering + now and then a tap of the other buttons;
  never START). Per picture and per run it measures what is drawn, and - the important part - plays the same
  machine again on the **real microcode** (`lle-gfx`): HLE is an emulation of the RSP, LLE *is* the RSP's code.
  Wherever HLE draws something else, one of them is wrong, and it is almost always HLE. Culled geometry,
  textures decoded wrongly, missing sprites, wrong colours - all of them show up as a difference between two
  pictures of the same instant, with a difference picture next to them.
* **Healing.** If a fix changes the timing so that a route no longer reaches the game, the tool notices (the
  stick no longer moves the picture), records a new route and tries again. You do not maintain routes.
* **Pictures along the route.** A route also remembers how the screen looked after every step (8x6 grey
  cells, `sig` lines). A replay compares them: if the game has parted from the recording, the report says *at
  which frame* (`drift`), which is a regression signal of its own - the same inputs now lead somewhere else.
  And a lost route is not recorded again from power-on: the search continues from the last picture that still
  matched.
* **Stalls are kept as bug reports.** When the probe finds a game that stops doing anything (no graphics task,
  no frame flip, no change on screen for 12 steps), its route ends right there and is marked as a stall. Replaying
  it reproduces the hang every time; the verdict is FAIL `stalled` until the game gets past it - then the route
  is simply continued (`record --improve`).
* **No silent downgrade.** A re-recorded route that gets less far than the one it replaces is *not* installed:
  the old route stays and the game is FAIL `route_regressed` ("it used to reach gameplay").
* **Memory.** Every result is compared with the previous run of that game: a game that got worse is a
  REGRESSION, one that got better is FIXED, a new finding on an otherwise unchanged game is a NEW ISSUE.

## Reading the screen (OCR)

The probe alone sees only *that* the picture changed. With `--advice` (the default in `qa.py record`) it also
asks what the screen *says*: at every eighth step and whenever the picture is stuck it saves a screenshot and
waits for tools/advice.py, which reads it (OCR + rules, tools/screen_reader.py) and answers with the kind of screen
and a few controller macros to try, best first:

| the screen shows | the advice |
|---|---|
| PRESS START / PRESS A | START / A |
| a menu (START GAME, 1 PLAYER, ARCADE, QUICK RACE ...) | move the cursor to the entry that starts a game, A (the highlighted entry is found by its marker or highlight bar, so the number of presses is right) |
| a question (YES / NO ...) | the right answer for the question ("continue without saving?" yes, "view instructions again?" no, "quit?" no) |
| PAUSED / RESUME | START (resume) |
| an options screen | B (back), or the RETURN entry |
| a warning (a Pak that "malfunctions") | A, and the text is kept: it is often what an emulator gets wrong |
| a text box | A |
| a game's HUD (LAP, TIME, SCORE ...) | nothing: it is gameplay |

The advice is only a hypothesis: the probe runs each candidate from the same state and keeps one only if it
changes the picture (and gives up on advice that was followed twice and left the screen the same kind). So a wrong
reading costs a little time, never a wrong route. Routes record the buttons, not the advice: **replaying never
needs OCR**, and stays exact.

`qa.py run` also reads the final screen of every game: a game that puts up a warning ("Rumble Pak is
malfunctioning") or something that reads like an error is flagged (`screen_warning`, `screen_error`) and the report
shows what the screen says.

OCR quality was measured, not assumed: `python tools/ocr_bench.py` scores engines on 32 real screenshots with
145 hand-written phrases (tools/ocr_bench_labels.json). RapidOCR at 2x with three versions of the picture (as it is,
contrast stretched, inverted) reads 95.2% of them and 100% of the words a decision rests on (Windows' own OCR:
58%; EasyOCR: 81%). `python tools/screen_reader_test.py` checks the decisions on 16 screens
(tools/ocr_bench_decisions.json). Needs: `pip install rapidocr-onnxruntime` (optional: `easyocr`,
`winrt-Windows.Media.Ocr` and friends for Windows' engine).

## Commands

| | |
|---|---|
| `qa.py record [--list F] [--filter T] [--jobs 4]` | finds routes for the games that have none (`--force`: all again, `--retry-failed`). `--improve` goes on from the end of routes that stop short of gameplay (3 more tries per game). The route that tests use is only replaced by one that reaches a better class, but the *search* always moves on: every run saves how far it got (`test_output/qa/frontier/`) and the next `--improve` starts there, so nothing explored is explored twice |
| `qa.py run [smoke\|render\|deep]` | replays and judges. `smoke`: route + short look, ~15 s per game. `render` (default): 300 frames, 10 pictures, HLE vs LLE. `deep`: 900 frames |
| `qa.py seed` | rebuilds routes from the old probe runs in `test_output/game_status` (only games that run took to gameplay) and installs the ones that still work: a replay each, no search |
| `qa.py rebase` | takes the pictures the current emulator draws along every route as the recorded ones (the inputs stay). Use it after a change that legitimately alters how games look |
| `qa.py report` | rebuilds `test_output/qa/report.html` from the last run |
| `qa.py routes [-v]` | which games have a route, how far it gets |
| `qa.py selftest` | checks the tools themselves: a replay must reproduce the same machine and pictures, bit for bit |

Useful options of `run`: `--list tools/test_set.txt` (73 representative games), `--filter Mario`, `--jobs N`,
`--no-lle` (skip the comparison, twice as fast), `--fresh` (ignore what is up to date), `--checkpoint` (start from
the saved state at the end of the route - quick, but it skips the boot), `--baseline old/results.json` (compare with
another run, e.g. a release), `--strict` (exit code 3 on failures or regressions: for scripts and CI).
A result is reused while neither the emulator build (`bin/game_qa`) nor the game's route has changed, so a second
`run` after changing one file only re-tests what that change could have touched - which is every game, but at
replay speed.

## What the verdicts mean

**FAIL** - the emulator crashed or hung; the route no longer reaches the game even after re-recording; the CPU
is at an exception vector or address 0 (the game jumped into nothing); no graphics task and no frame flip while
the game should be playing; the picture is black all the time.

**WARN** - HLE differs from LLE much more than games usually do (the threshold is taken from the games run
together: median + 4 MAD, at least 12% of the picture, on a coarse 32x24 grid so that edges and anti-aliasing do not count); most of the triangles are culled; the picture is
much noisier than LLE's (snow where a texture should be); the stick changes nothing on screen; fewer than 25
frames per second; HLE does not know the microcode and ran the game on the low-level RSP.

Info lines: silent audio, lines from the emulator's log, an LLE run that could not start.

`tools/qa_known.json` accepts findings you have looked at, so the report only shows what is new:

```json
{ "635A2BFF-8B022326": { "name": "Super Mario 64", "ignore": ["slow"], "note": "headless speed on this machine" } }
```

The codes: `route_lost route_regressed stalled cpu_lost no_graphics black frozen culled slow snow diverges hle_fallback drift silent screen_warning screen_error`.

## Reading the report

`report.html` is one self-contained file (pictures embedded): mail it as it is. Chips at the top filter by
verdict or change; the list sorts by severity, name, HLE/LLE difference or speed. Each game shows the HLE
picture, the LLE picture and the difference (grey, amplified) of its **worst** moment. A difference along
the edges only (anti-aliasing, 1-pixel offsets) is normal; whole areas are not.

## Layout

```
tools/game_probe.cpp     finds a route (experiments from save states), --route / --checkpoint write it
tools/probe_common.hpp   pictures, comparison, scripted input, the Prober that runs a game headless
tools/probe_route.hpp    the route file format
tools/game_qa.cpp        replays a route, plays, measures, compares with LLE      -> qa.json + PNGs
tools/qa.py              record / run / report / routes / selftest; judging, report
tools/routes/            <CRC>.route (commit these) and index.json (how each was recorded)
tools/qa_known.json      accepted findings
test_output/qa/          runs/<CRC>/ (qa.json, PNGs), checkpoints/, results.json, report.html
```

`qa.json` per game holds everything the verdicts are made from: route data, per-run counters (graphics
tasks, triangles, culled, pixels, z/alpha fails, frame flips, audio level, CPU state, speed), per-picture
measures (brightness, colours, edges, noise) and the HLE/LLE differences per picture.

## Limits worth knowing

* Routes get to "a game that reacts to the stick" - for most games that is real gameplay, for some a menu
  that a stick moves (`reached` says which: `ingame`, `big`, `furthest`). The tests then judge what is drawn
  there. A game whose route is stuck in a menu needs a hand-written route (a text file, see
  `probe_route.hpp`) or a better probe.
* LLE is a reference, not the truth: a difference shows that the two disagree. Looking at the pictures tells
  which one is right (almost always LLE: it runs the real microcode).
* The speed in the report is the speed of a headless run with several games at once.
