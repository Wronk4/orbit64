#!/bin/sh
# Prints the version the app shows: major.minor from project(VERSION) in CMakeLists.txt, then the number of commits (the
# number the release workflow gives its releases, v<major>.<minor>.<commits on main>), and "-<branch>" for a branch
# other than main. The commit count and the branch come from a .build_info file (the CI writes it before it builds),
# else from git when the clone is complete; with neither the count is 0.
cd "$(dirname "$0")/.." || exit 1
base=$(sed -nE 's/^project\(orbit64 VERSION ([0-9]+)\.([0-9]+).*/\1.\2/p' CMakeLists.txt)
[ -n "$base" ] || base=1.0
commits=; branch=
if [ -f .build_info ]; then
    commits=$(sed -n 1p .build_info)
    branch=$(sed -n 2p .build_info)
elif [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = false ]; then
    commits=$(git rev-list --count HEAD 2>/dev/null)
    branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
fi
version="$base.${commits:-0}"
case "$branch" in
    ""|main|HEAD) ;;
    *) version="$version-$(printf '%s' "$branch" | tr -c 'A-Za-z0-9._-' '-')" ;;
esac
printf '%s\n' "$version"
