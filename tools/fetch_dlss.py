#!/usr/bin/env python3
"""Fetches the NVIDIA DLSS SDK headers and Windows libraries into
third_party/dlss (a sparse, shallow clone of github.com/NVIDIA/DLSS, ~480 MB),
which turns on DLSS in the next CMake configure. The SDK comes under NVIDIA's
license (third_party/dlss/LICENSE.txt), which applies to its use; it is not
part of this repository.

  python3 tools/fetch_dlss.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "third_party", "dlss")


def run(*cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    if os.path.exists(os.path.join(DEST, "include", "nvsdk_ngx.h")):
        print("DLSS SDK already in", DEST)
        return
    run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "https://github.com/NVIDIA/DLSS.git", DEST)
    run("git", "sparse-checkout", "set", "include", "lib/Windows_x86_64", cwd=DEST)
    print("DLSS SDK in", DEST, "- reconfigure CMake to build with DLSS")


if __name__ == "__main__":
    sys.exit(main())
