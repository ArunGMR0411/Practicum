#!/usr/bin/env python3

"""Run the official StyleID backend over a manifest while keeping the model hot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.stylegan_backend_utils import StyleGANComposeConfig, anonymise_styleid_faces
from src.utils.runtime_tuning import configure_torch_runtime

BACKEND_ROOT = PROJECT_ROOT / "third_party" / "styleid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
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


def main() -> None:
    args = parse_args()
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    os.chdir(BACKEND_ROOT)

    required = [
        args.model_path,
        BACKEND_ROOT / "pretrained_models" / "psp_celebs_seg_to_face.pt",
        BACKEND_ROOT / "pretrained_models" / "CurricularFace_Backbone.pth",
        BACKEND_ROOT / "pretrained_models" / "mobilenet_celeba.pth",
        BACKEND_ROOT / "pretrained_models" / "unet_model.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"StyleID assets missing: {missing}")

    if not torch.cuda.is_available():
        raise RuntimeError("StyleID batch backend requires CUDA")

    from styleid import StyleID

    process_share = 1.0 / max(1, int(args.shard_count))
    tuning = configure_torch_runtime("cuda", process_share=process_share)
    torch.manual_seed(args.seed)

    model = StyleID(checkpoint=str(args.model_path))
    jobs = select_job_shard(load_jobs(args.manifest_path), args.shard_index, args.shard_count)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for job in jobs:
        input_path = Path(job["input_path"]).resolve()
        output_path = Path(job["output_path"]).resolve()
        boxes = [tuple(map(int, box)) for box in job.get("boxes", [])]
        frame_start = time.perf_counter()

        source = Image.open(input_path).convert("RGB")
        if not boxes:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            source.save(output_path)
            status = "copied"
        else:
            output = anonymise_styleid_faces(model, source, boxes, StyleGANComposeConfig())
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output.save(output_path)
            status = "ok"

        results.append(
            {
                "relative_path": job.get("relative_path", ""),
                "input_path": str(input_path),
                "output_path": str(output_path),
                "status": status,
                "runtime_seconds": round(time.perf_counter() - frame_start, 6),
                "box_count": len(boxes),
            }
        )

    summary = {
        "jobs_requested": len(jobs),
        "jobs_completed": len(results),
        "total_runtime_seconds": round(time.perf_counter() - started, 6),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "runtime_tuning": {
            "cpu_threads": tuning.cpu_threads,
            "interop_threads": tuning.interop_threads,
            "tf32_enabled": tuning.tf32_enabled,
            "cudnn_benchmark": tuning.cudnn_benchmark,
            "float32_matmul_precision": tuning.float32_matmul_precision,
        },
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
