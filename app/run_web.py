#!/usr/bin/env python3
"""Entry point for the privacy pipeline Web demonstrator."""

from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
SRC = APP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Configure research backend defaults (RiDDLE/FALCO/CUDA) automatically.
from privacy_pipeline_app.runtime_env import configure_app_runtime  # noqa: E402

configure_app_runtime()

from privacy_pipeline_app.production_app import main  # noqa: E402


if __name__ == "__main__":
    main()
