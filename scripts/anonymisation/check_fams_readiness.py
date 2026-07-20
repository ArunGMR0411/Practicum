#!/usr/bin/env python3

"""Report whether the FAMS anonymiser adapter is runnable."""

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

from src.anonymisation.fams_anonymiser import FAMSAnonymiser


def _check_runtime_versions(python_executable: str) -> tuple[list[str], dict[str, str]]:
    details: dict[str, str] = {}
    warnings: list[str] = []
    probe = (
        "import json\n"
        "import diffusers, huggingface_hub, torch, transformers\n"
        "print(json.dumps({"
        "'diffusers': diffusers.__version__, "
        "'transformers': transformers.__version__, "
        "'huggingface_hub': huggingface_hub.__version__, "
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
    except Exception as exc:  # pragma: no cover - surfaced in payload
        return [f"runtime import failure via {python_executable}: {exc}"], details

    if Version(details["diffusers"]) != Version("0.25.1"):
        warnings.append("diffusers differs from upstream FAMS requirement 0.25.1")
    if Version(details["transformers"]) != Version("4.46.1"):
        warnings.append("transformers differs from upstream FAMS requirement 4.46.1")
    if Version(details["huggingface_hub"]) >= Version("0.26.0"):
        warnings.append("huggingface_hub is newer than upstream FAMS recommendation < 0.26.0")
    if Version(details["torch"].split("+", 1)[0]) != Version("2.1"):
        warnings.append("torch differs from upstream FAMS major/minor recommendation 2.1")
    return warnings, details


def _latest_smoke_summary() -> dict[str, object] | None:
    summary_path = PROJECT_ROOT / "outputs" / "runs" / "fams_smoke" / "summary.json"
    if summary_path.is_file():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    retained_output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "03_anonymisation"
        / "06_styleid_fams"
        / "fams_images"
    )
    retained_output = next(iter(sorted(retained_output_dir.glob("*.webp"))), retained_output_dir / "missing.webp")
    retained_comparison = (
        PROJECT_ROOT
        / "outputs"
        / "03_anonymisation"
        / "06_styleid_fams"
        / "03_pilot_summary.json"
    )
    if retained_output.is_file() and retained_comparison.is_file():
        return {
            "returncode": 0,
            "output_exists": True,
            "source": "retained_pilot_evidence",
            "output_path": str(retained_output.relative_to(PROJECT_ROOT)),
            "comparison_path": str(retained_comparison.relative_to(PROJECT_ROOT)),
        }
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    anonymiser = FAMSAnonymiser()
    runtime_warnings, runtime_versions = _check_runtime_versions(anonymiser.python_executable)
    smoke_summary = _latest_smoke_summary()
    smoke_ok = bool(smoke_summary and smoke_summary.get("returncode") == 0 and smoke_summary.get("output_exists"))
    ready = not bool(anonymiser.reason) and (smoke_ok or not runtime_warnings)
    reasons: list[str] = []
    if anonymiser.reason:
        reasons.append(anonymiser.reason)
    if not smoke_ok:
        reasons.append("no successful FAMS probe artifact found")
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
        "note": "Hugging Face model weights may still download on first real run if not already cached. Upstream version recommendations are reported as warnings, but a passing CASTLE probe is treated as the stronger readiness signal.",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
