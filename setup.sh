#!/bin/sh
# Set up the MaXXXLab Jr XRD Work-up App: finds Python 3.11+ (newest first), creates .venv and installs everything.
#   ./setup.sh                 core + Materials Project/CIF + Le Bail
#   ./setup.sh --core-only     core only        (other options: --no-mp --no-lebail --dev --recreate)
# Set PYTHON=/path/to/python3.12 to choose the interpreter yourself.
cd "$(dirname "$0")" || exit 1
ok() { "$1" -c 'import sys, venv; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; }
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && ok "$c"; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "[error] No Python 3.11 or newer found."
  echo "        macOS: install Python 3.12 or 3.13 from https://www.python.org/downloads/ (or: brew install python@3.12)"
  echo "        Linux: sudo apt install python3.12 python3.12-venv   (or your distribution's equivalent)"
  echo "        Then run ./setup.sh again, or PYTHON=/path/to/python3.12 ./setup.sh"
  exit 1
fi
exec "$PY" install.py "$@"
