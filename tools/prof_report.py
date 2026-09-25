#!/usr/bin/env python3
"""Turns a `--profile` samples file (src/profiler.cpp) into per-function tables.

    python tools/prof_report.py bin/n64.exe samples.txt [--top N]

Self = samples whose innermost frame is in the function; inclusive = samples
with the function anywhere on the (frame-pointer) call chain. Addresses
outside the executable (JIT-compiled code, system DLLs) are grouped as
"[outside exe]". Needs nm/objdump on PATH (w64devkit/binutils).
"""
import bisect
import collections
import subprocess
import sys


def image_base(exe):
    out = subprocess.run(["objdump", "-p", exe], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("ImageBase"):
            return int(line.split()[1], 16)
    return 0x140000000


def load_symbols(exe):
    out = subprocess.run(["nm", "-C", "-n", "--defined-only", exe], capture_output=True, text=True).stdout
    addrs, names = [], []
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) < 3 or parts[1] not in "Tt":
            continue
        addrs.append(int(parts[0], 16))
        names.append(parts[2])
    return addrs, names


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    exe, samples_path = sys.argv[1], sys.argv[2]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
    pref_base = image_base(exe)
    addrs, names = load_symbols(exe)
    text_end = addrs[-1] + 0x10000 if addrs else 0

    def resolve(a, load_base):
        va = a - load_base + pref_base
        if not addrs or va < addrs[0] or va > text_end:
            return "[outside exe]"
        return names[bisect.bisect_right(addrs, va) - 1]

    self_c, incl_c = collections.Counter(), collections.Counter()
    total = 0
    with open(samples_path) as f:
        load_base = int(f.readline().split()[1], 16)
        for line in f:
            chain = [int(x, 16) for x in line.split()]
            if not chain:
                continue
            total += 1
            funcs = [resolve(a, load_base) for a in chain]
            self_c[funcs[0]] += 1
            for fn in set(funcs):
                incl_c[fn] += 1

    def short(n):
        return n if len(n) <= 90 else n[:87] + "..."

    if "--lines" in sys.argv:
        # Self samples per source line of the functions whose name contains
        # the given text (needs a -g build).
        needle = sys.argv[sys.argv.index("--lines") + 1]
        hits = collections.Counter()
        with open(samples_path) as f:
            f.readline()
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                a = int(parts[0], 16)
                if needle in resolve(a, load_base):
                    hits[a - load_base + pref_base] += 1
        addrs_list = sorted(hits)
        out = subprocess.run(["addr2line", "-e", exe, "-i"] + [hex(a) for a in addrs_list],
                             capture_output=True, text=True).stdout.splitlines()
        # With -i every address can expand to several lines (inlining); the
        # first one is the innermost location.
        per_line = collections.Counter()
        res = subprocess.run(["addr2line", "-e", exe] + [hex(a) for a in addrs_list],
                             capture_output=True, text=True).stdout.splitlines()
        for a, loc in zip(addrs_list, res):
            per_line[loc.split("/")[-1].split("\\")[-1]] += hits[a]
        n = sum(hits.values())
        print(f"{n} self samples in functions matching '{needle}'")
        for loc, c in per_line.most_common(top):
            print(f"  {100.0 * c / max(n, 1):6.2f}%  {loc}")
        del out
        return 0

    print(f"{total} samples\n")
    print("SELF")
    for fn, c in self_c.most_common(top):
        print(f"  {100.0 * c / total:6.2f}%  {short(fn)}")
    print("\nINCLUSIVE")
    for fn, c in incl_c.most_common(top):
        print(f"  {100.0 * c / total:6.2f}%  {short(fn)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
