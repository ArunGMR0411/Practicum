#!/usr/bin/env python3

"""Report whether the NullFace anonymiser adapter is runnable."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.nullface_anonymiser import NullFaceAnonymiser


def _check_runtime_versions(python_executable: str) -> tuple[list[str], dict[str, str]]:
    details: dict[str, str] = {}
    warnings: list[str] = []
    probe = (
        "import json\n"
        "import diffusers, insightface, torch, transformers\n"
        "print(json.dumps({"
        "'diffusers': diffusers.__version__, "
        "'insightface': insightface.__version__, "
        "'transformers': transformers.__version__, "
        "'torch': torch.__version__"
        "}))\n"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        details = json.loads(result.stdout.strip())
    except Exception as exc:
        return [f"runtime import failure via {python_executable}: {exc}"], details

    if Version(details["diffusers"]) < Version("0.30.0"):
        warnings.append("diffusers is older than the upstream NullFace environment recommendation")
    if Version(details["transformers"]) < Version("4.44.0"):
        warnings.append("transformers is older than the upstream NullFace environment recommendation")
    return warnings, details


def _latest_smoke_summary() -> dict[str, object] | None:
    summary_path = PROJECT_ROOT / "outputs" / "nullface_probe" / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    anonymiser = NullFaceAnonymiser()
    runtime_warnings, runtime_versions = _check_runtime_versions(anonymiser.python_executable)
    smoke_summary = _latest_smoke_summary()
    smoke_ok = bool(smoke_summary and smoke_summary.get("returncode") == 0 and smoke_summary.get("output_exists"))
    ready = not bool(anonymiser.reason) and (smoke_ok or not runtime_warnings)
    reasons: list[str] = []
    if anonymiser.reason:
        reasons.append(anonymiser.reason)
    if not smoke_ok:
        reasons.append("no successful NullFace probe artifact found")
    payload = {
        "method": anonymiser.method_name,
        "backend_root": str(anonymiser.backend_root),
        "runner_path": str(anonymiser.runner_path),
        "model_id": anonymiser.model_id,
        "python_executable": anonymiser.python_executable,
        "ready": ready,
        "reason": "ready" if ready else "; ".join(reasons),
        "runtime_versions": runtime_versions,
        "compatibility_warnings": runtime_warnings,
        "latest_smoke_summary": smoke_summary,
        "note": "The stronger readiness signal is a passing CASTLE smoke because NullFace depends on the surrounding diffusion and InsightFace stack, not just import success.",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
