#!/usr/bin/env python3

"""Report whether the Reverse Personalization adapter is runnable."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.reverse_personalization_anonymiser import ReversePersonalizationAnonymiser


def main() -> None:
    anonymiser = ReversePersonalizationAnonymiser()
    payload = {
        "method": anonymiser.method_name,
        "backend_root": str(anonymiser.backend_root),
        "runner_path": str(anonymiser.runner_path),
        "python_executable": anonymiser.python_executable,
        "ready": not bool(anonymiser.reason),
        "reason": anonymiser.reason or "ready",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
