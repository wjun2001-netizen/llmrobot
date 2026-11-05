#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the validated environment"'
if [ ! -x .venv/bin/python ]; then
    "$python_bin" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12)'
export PIP_CACHE_DIR="$PWD/.cache/pip"
export ALFWORLD_DATA="$PWD/data/alfworld"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/download_data.py
