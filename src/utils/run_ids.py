"""Run identifier helpers shared by thesis logging scripts."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone


def build_run_id(run_group: str) -> str:
    """Generate a traceable UTC run identifier for filesystem or W&B logging."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(4)
    return f"{run_group}-{stamp}-{suffix}"
