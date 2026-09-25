#!/usr/bin/env python3
"""Quality report for audio captured with `n64 <rom> --headless N --wav out.wav`.

    python tools/audio_report.py out.wav [--png spectrogram.png] [--ref other.wav]

Prints level, clipping, DC and a click estimate (samples whose second
difference is far above the local signal's), and optionally draws a
spectrogram (clicks show up as vertical lines, aliasing as mirrored
partials) or compares two captures sample by sample.
"""
import sys
import wave
import numpy as np


def load(path):
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    return rate, data.reshape(-1, ch) / 32768.0


def clicks(x, rate):
    """Samples where |x[n] - 2x[n-1] + x[n-2]| exceeds 12x the local median of it."""
    d2 = np.abs(np.diff(x, 2))
    win = max(64, rate // 100)
    pad = np.pad(d2, (win // 2, win // 2), mode="edge")
    # Rolling median via strided windows on a decimated grid (cheap and good enough).
    step = win // 4
    centers = np.arange(0, len(d2), step)
    med = np.array([np.median(pad[c:c + win]) for c in centers])
    local = np.interp(np.arange(len(d2)), centers, med)
    thresh = np.maximum(local * 12.0, 0.02)
    hits = np.nonzero(d2 > thresh)[0]
    # Merge hits closer than 1 ms into one event.
    events = []
    for h in hits:
        if not events or h - events[-1] > rate // 1000:
            events.append(h)
    return events


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    path = args[0]
    png = args[args.index("--png") + 1] if "--png" in args else None
    ref = args[args.index("--ref") + 1] if "--ref" in args else None
    rate, x = load(path)
    secs = len(x) / rate
    print(f"{path}: {len(x)} frames @ {rate} Hz = {secs:.2f} s")
    for c, name in enumerate(["L", "R"][: x.shape[1]]):
        s = x[:, c]
        rms = np.sqrt(np.mean(s * s))
        clip = np.count_nonzero(np.abs(s) >= 32767 / 32768)
        ev = clicks(s, rate)
        print(f"  {name}: rms={20*np.log10(rms+1e-12):6.1f} dBFS peak={np.max(np.abs(s)):.3f} dc={np.mean(s):+.4f} "
              f"clipped={clip} clicks={len(ev)} ({len(ev)/secs:.1f}/s)")
        if ev and "--verbose" in args:
            print("     first clicks at s:", ", ".join(f"{e/rate:.3f}" for e in ev[:12]))
    if ref:
        rrate, y = load(ref)
        n = min(len(x), len(y))
        diff = x[:n] - y[:n]
        first = np.nonzero(np.any(np.abs(diff) > 0, axis=1))[0]
        print(f"  vs {ref}: rate {rrate}, {n} common frames, identical={len(first)==0}"
              + (f", first difference at frame {first[0]} ({first[0]/rate:.3f} s), "
                 f"diff rms={20*np.log10(np.sqrt(np.mean(diff*diff))+1e-12):.1f} dBFS" if len(first) else ""))
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mono = x.mean(axis=1)
        fig, ax = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 3]})
        t = np.arange(len(mono)) / rate
        ax[0].plot(t, mono, lw=0.3)
        ax[0].set_xlim(0, secs)
        ax[0].set_title(path.replace("\\", "/").split("/")[-1])
        ax[1].specgram(mono, NFFT=1024, Fs=rate, noverlap=512, cmap="magma", vmin=-140)
        ax[1].set_xlim(0, secs)
        fig.tight_layout()
        fig.savefig(png, dpi=90)
        print(f"  spectrogram -> {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
