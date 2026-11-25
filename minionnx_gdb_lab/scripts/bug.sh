#!/usr/bin/env bash
# Lab helper (Linux / WSL / MSYS2 bash / Git Bash).  Run from the lab root folder.
#   bash scripts/bug.sh apply 7     introduce bug 7 into code/
#   bash scripts/bug.sh reset 7     remove bug 7 again
#   bash scripts/bug.sh check       which bugs are currently applied?
set -e
cd "$(dirname "$0")/../code"
cmd=$1; n=$(printf "%02d" "${2:-0}")
case "$cmd" in
  apply) p=$(ls ../bugs/B${n}_*.patch); git apply --verbose "$p" && echo "bug $n applied: rebuild with  cmake --build build -j  then run  ./build/lab_runner $((10#$n))" ;;
  reset) p=$(ls ../bugs/B${n}_*.patch); git apply -R --verbose "$p" && echo "bug $n removed" ;;
  check) for p in ../bugs/B*.patch; do if git apply -R --check "$p" 2>/dev/null; then echo "APPLIED: $(basename $p)"; fi; done; echo "(nothing listed = clean code)" ;;
  *) echo "usage: bash scripts/bug.sh apply|reset N   or   bash scripts/bug.sh check"; exit 2 ;;
esac
