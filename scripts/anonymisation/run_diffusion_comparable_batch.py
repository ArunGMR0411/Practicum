#!/usr/bin/env python3
"""Run the low-step diffusion anonymiser over a comparable JSONL manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.diffusion_anonymiser import DiffusionAnonymiser
from src.utils.runtime_tuning import configure_torch_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-id", default="data/models/stable-diffusion-inpainting")
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--strength-padding-ratio", type=float, default=0.75)
    parser.add_argument("--mask-expansion-ratio", type=float, default=0.0)
    parser.add_argument("--mask-feather-px", type=int, default=0)
    parser.add_argument("--target-resolution", type=int, default=512)
    parser.add_argument("--eval-method-name", default="diffusion_low_step")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--process-share",
        type=float,
        default=None,
        help="Fraction of the host CPU budget this process should use when multiple GPU workers run concurrently.",
    )
    return parser.parse_args()


def load_jobs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_job_shard(jobs: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    if shard_count <= 1:
        return jobs
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard selection: index={shard_index}, count={shard_count}")
    return [job for idx, job in enumerate(jobs) if idx % shard_count == shard_index]


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "relative_path",
        "method",
        "output_path",
        "boxes_processed",
        "tiling_required",
        "runtime_seconds",
        "status",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    jobs = select_job_shard(load_jobs(args.manifest_path), args.shard_index, args.shard_count)
    process_share = args.process_share
    if process_share is None:
        process_share = float(os.environ.get("DIFFUSION_PROCESS_SHARE", "1.0"))
    tuning = configure_torch_runtime(args.device or "cuda", process_share=process_share)
    anonymiser = DiffusionAnonymiser(
        model_id=args.model_id,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        strength_padding_ratio=args.strength_padding_ratio,
        mask_expansion_ratio=args.mask_expansion_ratio,
        mask_feather_px=args.mask_feather_px,
        target_resolution=args.target_resolution,
        device=args.device,
    )

    results: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for job in jobs:
        input_path = resolve_project_path(job["input_path"])
        output_path = resolve_project_path(job["output_path"])
        relative_path = job.get("relative_path", "")
        frame_start = time.perf_counter()
        payload: dict[str, Any] = {
            "relative_path": relative_path,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "box_count": len(job.get("boxes", [])),
        }
        manifest_row: dict[str, Any] = {
            "relative_path": relative_path,
            "method": args.eval_method_name,
            "output_path": str(output_path.relative_to(PROJECT_ROOT)) if output_path.is_relative_to(PROJECT_ROOT) else str(output_path),
            "boxes_processed": 0,
            "tiling_required": False,
            "runtime_seconds": "not_available",
            "status": "error",
            "error": "",
        }
        try:
            with Image.open(input_path) as image:
                result = anonymiser.anonymise(image.convert("RGB"), [tuple(box) for box in job.get("boxes", [])])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.image.save(output_path)
            boxes_processed = int(result.metadata.get("boxes_processed", 0))
            tiling_required = bool(result.metadata.get("tiling_required", False))
            payload.update({"status": "ok", "boxes_processed": boxes_processed, "tiling_required": tiling_required})
            manifest_row.update(
                {
                    "boxes_processed": boxes_processed,
                    "tiling_required": tiling_required,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            payload.update({"status": "error", "error": str(exc), "boxes_processed": 0, "tiling_required": False})
            manifest_row.update({"status": "error", "error": str(exc)})
        runtime_seconds = round(time.perf_counter() - frame_start, 6)
        payload["runtime_seconds"] = runtime_seconds
        manifest_row["runtime_seconds"] = runtime_seconds
        results.append(payload)
        manifest_rows.append(manifest_row)

    summary = {
        "jobs_requested": len(jobs),
        "jobs_completed": len(results),
        "ok_count": sum(1 for row in results if row.get("status") == "ok"),
        "error_count": sum(1 for row in results if row.get("status") == "error"),
        "total_runtime_seconds": round(time.perf_counter() - started, 6),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "method": args.eval_method_name,
        "runtime_tuning": {
            "cpu_threads": tuning.cpu_threads,
            "interop_threads": tuning.interop_threads,
            "tf32_enabled": tuning.tf32_enabled,
            "cudnn_benchmark": tuning.cudnn_benchmark,
            "float32_matmul_precision": tuning.float32_matmul_precision,
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_manifest(args.output_manifest, manifest_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
