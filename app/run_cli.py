#!/usr/bin/env python3
"""Entry point for the privacy pipeline CLI."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
SRC = APP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from privacy_pipeline_app.runtime_env import configure_app_runtime  # noqa: E402

configure_app_runtime()

runpy.run_module("privacy_pipeline_app.pipeline_demo", run_name="__main__")
