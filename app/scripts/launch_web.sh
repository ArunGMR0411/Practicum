#!/usr/bin/env bash
set -euo pipefail

# Launch from repository root. Research backends are configured inside run_web.py.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r app/requirements.txt
.venv/bin/python app/run_web.py
