#!/usr/bin/env python3
"""Production-style runner for the privacy pipeline app.

This module wraps the stable ``app/src/privacy_pipeline_app/pipeline_demo.py`` CLI so the Web UI and
validation scripts use the same deterministic anonymisation path as the
report evidence. It deliberately keeps all processing within the configured environment.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / ".venv" / "bin" / "python"
PIPELINE_CLI = ROOT / "app" / "src" / "privacy_pipeline_app" / "pipeline_demo.py"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FIXED_MODES = ["blur", "pixelate", "solid_mask", "layered"]
SELECTOR_OBJECTIVES = [
    "privacy_first",
    "utility_priority",
    "utility_under_privacy_floor",
    "runtime_aware",
    "compute_profile_adaptive",
    "failure_avoidance",
    "multimodal_risk",
]
ALL_MODES = FIXED_MODES + ["objective_profile"]


@dataclass(frozen=True)
class ProductionRunResult:
    ok: bool
    message: str
    run_dir: Path
    frames_requested: int
    frames_ok: int
    frames_failed: int
    runtime_total_seconds: float
    runtime_mean_seconds: float
    method_counts: dict[str, int]
    routing_log: Path
    runtime_summary: Path
    failure_log: Path
    manifest: Path
    config: Path
    readme: Path


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def probe_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "cwd": str(ROOT),
        "cpu_count": os.cpu_count(),
        "cuda_available": False,
        "gpu_name": "not_available",
        "vram_total_mb": "not_available",
        "device": "cpu",
    }
    try:
        import torch

        payload["torch_version"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["device"] = "cuda"
            index = torch.cuda.current_device()
            payload["gpu_name"] = torch.cuda.get_device_name(index)
            props = torch.cuda.get_device_properties(index)
            payload["vram_total_mb"] = int(props.total_memory / (1024 * 1024))
    except Exception as exc:  # pragma: no cover - depends on runtime stack
        payload["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from src.utils.compute_policy import build_compute_policy

        policy = build_compute_policy()
        payload["compute_policy"] = {
            "device": policy.device,
            "detection_num_workers": policy.detection_num_workers,
            "generative_control_max_workers": policy.generative_control_max_workers,
            "accelerator_total_gb": policy.accelerator_total_gb,
            "accelerator_available_gb": policy.accelerator_available_gb,
            "host_ram_total_gb": policy.host_ram_total_gb,
            "host_ram_available_gb": policy.host_ram_available_gb,
            "resource_concurrency_policy": "resource-derived concurrency rather than a fixed cap",
        }
    except Exception as exc:  # pragma: no cover - config-dependent
        payload["compute_policy_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def supported_files(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def create_manifest_from_folder(input_dir: Path, run_dir: Path, recursive: bool, limit: int | None) -> Path:
    files = supported_files(input_dir, recursive)
    if limit and limit > 0:
        files = files[:limit]
    manifest = run_dir / "input_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "local_path"])
        writer.writeheader()
        for path in files:
            writer.writerow({"image_id": path.name, "local_path": str(path)})
    return manifest


def count_manifest_rows(path: Path, limit: int | None = None) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    return min(count, limit) if limit and limit > 0 else count


def timestamped_run_dir(output_root: Path, mode: str, objective: str, run_label: str | None = None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = run_label or f"{utc_stamp()}_{mode}_{objective}"
    candidate = output_root / base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_readme(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Privacy Pipeline App Run",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Mode: `{result.get('mode')}`",
        f"- Objective: `{result.get('objective')}`",
        f"- Frames requested: `{result.get('frames_requested')}`",
        f"- Frames OK: `{result.get('frames_ok')}`",
        f"- Frames failed: `{result.get('frames_failed')}`",
        f"- Runtime total seconds: `{result.get('runtime_total_seconds')}`",
        f"- Output directory: `{result.get('run_dir')}`",
        "",
        "## Output Contract",
        "",
        "- `images/`: anonymised images",
        "- `side_by_side/`: before/after preview images where generated",
        "- `routing_log.csv`: selected method and explanation per image",
        "- `per_image_manifest.csv`: input/output mapping per image",
        "- `failure_log.csv`: recoverable failures",
        "- `runtime_summary.json`: runtime and method counts",
        "- `app_run_manifest.json`: execution manifest",
        "- `config.json`: app/runtime configuration",
        "",
        "Privacy note: processing uses the configured environment. Do not publish raw CASTLE frames or private images.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def empty_csv(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def run_pipeline(
    *,
    input_folder: Path | None = None,
    manifest_path: Path | None = None,
    output_root: Path,
    mode: str,
    objective: str,
    recursive: bool = True,
    dry_run: bool = False,
    limit: int | None = None,
    resolution: str = "native",
    workers: int | None = None,
    run_label: str | None = None,
) -> ProductionRunResult:
    requested_mode = mode
    if mode.lower() == "oapr":
        mode = "objective_profile"
        legacy_alias_note = "Strategy alias resolved to objective_profile."
    else:
        legacy_alias_note = None
    if mode not in ALL_MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    if objective not in SELECTOR_OBJECTIVES:
        raise ValueError(f"Unsupported selector objective: {objective}")

    run_dir = timestamped_run_dir(output_root, mode, objective, run_label)
    config_path = run_dir / "config.json"
    app_manifest_path = run_dir / "app_run_manifest.json"
    readme_path = run_dir / "README_SUMMARY.md"
    failure_log = run_dir / "failure_log.csv"
    routing_log = run_dir / "routing_log.csv"
    per_image_manifest = run_dir / "per_image_manifest.csv"
    runtime_summary = run_dir / "runtime_summary.json"

    env_payload = probe_environment()
    config = {
        "mode": mode,
        "requested_mode": requested_mode,
        "legacy_alias_note": legacy_alias_note,
        "objective": objective,
        "recursive": recursive,
        "dry_run": dry_run,
        "limit": limit,
        "resolution": resolution,
        "workers": workers,
        "input_folder": str(input_folder) if input_folder else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "output_root": str(output_root),
        "environment": env_payload,
        "privacy_boundary": "configured-environment processing only; privacy-risk reduction, not full anonymisation",
    }
    write_json(config_path, config)

    status = "ok"
    message = "completed"
    started = time.perf_counter()
    try:
        if manifest_path is None:
            if input_folder is None:
                raise ValueError("Provide either input_folder or manifest_path.")
            if not input_folder.exists() or not input_folder.is_dir():
                raise FileNotFoundError(f"Input folder not found: {input_folder}")
            manifest_path = create_manifest_from_folder(input_folder, run_dir, recursive, limit)
            frames_requested = count_manifest_rows(manifest_path)
        else:
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")
            frames_requested = count_manifest_rows(manifest_path, limit)

        if frames_requested == 0:
            message = "no supported input images found"
            status = "empty_input"
            empty_csv(routing_log, ["case_id", "image_id", "selected_method", "runtime_seconds", "status", "error"])
            empty_csv(per_image_manifest, ["image_id", "input_path", "output_path", "side_by_side_path", "selected_method", "objective_mode", "status"])
            empty_csv(failure_log, ["image_id", "status", "error"])
            summary = {
                "frames_requested": 0,
                "frames_ok": 0,
                "frames_failed": 0,
                "mode": mode,
                "objective": objective,
                "runtime_total_seconds": 0.0,
                "runtime_mean_seconds": 0.0,
                "method_counts": {},
                "output_dir": str(run_dir),
                "status": status,
            }
            write_json(runtime_summary, summary)
        elif dry_run:
            rows: list[dict[str, str]] = []
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for index, row in enumerate(reader, start=1):
                    if limit and index > limit:
                        break
                    image_id = row.get("image_id") or row.get("relative_path") or row.get("local_path") or f"image_{index:04d}"
                    rows.append(
                        {
                            "case_id": f"dry_run_{index:03d}",
                            "image_id": image_id,
                            "selected_method": "not_executed_dry_run",
                            "runtime_seconds": "0.0",
                            "status": "dry_run",
                            "error": "",
                        }
                    )
            with routing_log.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case_id", "image_id", "selected_method", "runtime_seconds", "status", "error"])
                writer.writeheader()
                writer.writerows(rows)
            empty_csv(per_image_manifest, ["image_id", "input_path", "output_path", "side_by_side_path", "selected_method", "objective_mode", "status"])
            empty_csv(failure_log, ["image_id", "status", "error"])
            summary = {
                "frames_requested": len(rows),
                "frames_ok": len(rows),
                "frames_failed": 0,
                "mode": mode,
                "objective": objective,
                "runtime_total_seconds": 0.0,
                "runtime_mean_seconds": 0.0,
                "method_counts": {"dry_run": len(rows)},
                "output_dir": str(run_dir),
                "status": "dry_run",
            }
            write_json(runtime_summary, summary)
        else:
            command = [
                str(PYTHON if PYTHON.exists() else sys.executable),
                str(PIPELINE_CLI),
                "--manifest",
                str(manifest_path),
                "--mode",
                mode,
                "--objective",
                objective,
                "--resolution",
                resolution,
                "--output-dir",
                str(run_dir),
            ]
            if limit:
                command.extend(["--limit", str(limit)])
            if workers:
                command.extend(["--workers", str(max(1, workers))])
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                status = "failed"
                message = f"pipeline_demo failed with return code {completed.returncode}"
            if runtime_summary.exists():
                summary = json.loads(runtime_summary.read_text(encoding="utf-8"))
            else:
                summary = {
                    "frames_requested": frames_requested,
                    "frames_ok": 0,
                    "frames_failed": frames_requested,
                    "mode": mode,
                    "objective": objective,
                    "runtime_total_seconds": round(time.perf_counter() - started, 6),
                    "runtime_mean_seconds": 0.0,
                    "method_counts": {},
                    "output_dir": str(run_dir),
                    "status": status,
                }
    except Exception as exc:
        status = "failed"
        message = f"{type(exc).__name__}: {exc}"
        frames_requested = 0
        empty_csv(routing_log, ["case_id", "image_id", "selected_method", "runtime_seconds", "status", "error"])
        empty_csv(per_image_manifest, ["image_id", "input_path", "output_path", "side_by_side_path", "selected_method", "objective_mode", "status"])
        with failure_log.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "status", "error"])
            writer.writeheader()
            writer.writerow({"image_id": "run_start", "status": "error", "error": message})
        summary = {
            "frames_requested": 0,
            "frames_ok": 0,
            "frames_failed": 1,
            "mode": mode,
            "objective": objective,
            "runtime_total_seconds": round(time.perf_counter() - started, 6),
            "runtime_mean_seconds": 0.0,
            "method_counts": {},
            "output_dir": str(run_dir),
            "status": status,
        }
        write_json(runtime_summary, summary)

    manifest_payload = {
        "run_dir": str(run_dir),
        "status": status,
        "message": message,
        "config": str(config_path),
        "routing_log": str(routing_log),
        "runtime_summary": str(runtime_summary),
        "failure_log": str(failure_log),
        "readme": str(readme_path),
    }
    write_json(app_manifest_path, manifest_payload)
    summary = json.loads(runtime_summary.read_text(encoding="utf-8"))
    summary["status"] = status
    summary["message"] = message
    summary["run_dir"] = str(run_dir)
    write_json(runtime_summary, summary)
    write_readme(readme_path, summary)

    return ProductionRunResult(
        ok=status in {"ok", "empty_input"} or summary.get("status") in {"dry_run"},
        message=message,
        run_dir=run_dir,
        frames_requested=int(summary.get("frames_requested", 0)),
        frames_ok=int(summary.get("frames_ok", 0)),
        frames_failed=int(summary.get("frames_failed", 0)),
        runtime_total_seconds=float(summary.get("runtime_total_seconds", 0.0)),
        runtime_mean_seconds=float(summary.get("runtime_mean_seconds", 0.0)),
        method_counts={str(k): int(v) for k, v in dict(summary.get("method_counts", {})).items()},
        routing_log=routing_log,
        runtime_summary=runtime_summary,
        failure_log=failure_log,
        manifest=per_image_manifest,
        config=config_path,
        readme=readme_path,
    )


def aggregate_runtime(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(row["runtime_total_seconds"]) for row in rows if row.get("runtime_total_seconds") not in {"", None}]
    return {
        "mean_runtime_seconds": round(sum(values) / len(values), 6) if values else 0.0,
        "median_runtime_seconds": round(median(values), 6) if values else 0.0,
    }
