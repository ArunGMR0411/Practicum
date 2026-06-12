#!/usr/bin/env python3

"""Benchmark per-frame routing decision time versus fixed strategies."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.routing import LearnedRouter, QualityAssessment, QualitySignals, RuleBasedRouter


def build_assessments(quality_df: pd.DataFrame) -> list[QualityAssessment]:
    assessments: list[QualityAssessment] = []
    for row in quality_df.itertuples(index=False):
        assessments.append(
            QualityAssessment(
                signals=QualitySignals(
                    blur_score=float(row.blur_score),
                    face_size_px=float(row.face_size_px),
                    occlusion_ratio=float(row.occlusion_ratio),
                    webp_artifact_score=float(row.webp_artifact_score),
                ),
                metadata={"face_box_count": int(row.face_box_count)},
            )
        )
    return assessments


def benchmark_callable(fn, repeats: int) -> dict[str, float]:
    per_repeat_ns: list[int] = []
    per_call_ns: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        calls = fn()
        elapsed = time.perf_counter_ns() - start
        per_repeat_ns.append(elapsed)
        per_call_ns.append(elapsed / max(calls, 1))
    return {
        "repeats": repeats,
        "mean_per_repeat_ms": float(np.mean(per_repeat_ns) / 1_000_000.0),
        "median_per_repeat_ms": float(np.median(per_repeat_ns) / 1_000_000.0),
        "mean_per_frame_ms": float(np.mean(per_call_ns) / 1_000_000.0),
        "median_per_frame_ms": float(np.median(per_call_ns) / 1_000_000.0),
        "min_per_frame_ms": float(np.min(per_call_ns) / 1_000_000.0),
        "max_per_frame_ms": float(np.max(per_call_ns) / 1_000_000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-csv", default="outputs/01_protocol/05_calibration_quality_signals.csv")
    parser.add_argument("--learned-model", default="outputs/runs/routing_baseline/learned_router.joblib")
    parser.add_argument("--output-json", default="outputs/runs/routing_baseline/routing_runtime_benchmark.json")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    quality_df = pd.read_csv(PROJECT_ROOT / args.quality_csv)
    assessments = build_assessments(quality_df)

    rule_router = RuleBasedRouter()
    learned_router = LearnedRouter(PROJECT_ROOT / args.learned_model)

    def run_fixed_blur() -> int:
        for _assessment in assessments:
            _ = "blur"
        return len(assessments)

    def run_fixed_pixelate() -> int:
        for _assessment in assessments:
            _ = "pixelate"
        return len(assessments)

    def run_rule_router() -> int:
        for assessment in assessments:
            rule_router.decide(assessment)
        return len(assessments)

    def run_learned_router() -> int:
        for assessment in assessments:
            learned_router.decide(assessment)
        return len(assessments)

    payload = {
        "quality_csv": args.quality_csv,
        "n_frames": len(assessments),
        "repeats": args.repeats,
        "strategies": {
            "fixed_blur": benchmark_callable(run_fixed_blur, args.repeats),
            "fixed_pixelate": benchmark_callable(run_fixed_pixelate, args.repeats),
            "rule_based_router": benchmark_callable(run_rule_router, args.repeats),
            "learned_router": benchmark_callable(run_learned_router, args.repeats),
        },
    }

    fixed_baseline = payload["strategies"]["fixed_blur"]["mean_per_frame_ms"]
    payload["overhead_vs_fixed_blur_ms"] = {
        "rule_based_router": float(payload["strategies"]["rule_based_router"]["mean_per_frame_ms"] - fixed_baseline),
        "learned_router": float(payload["strategies"]["learned_router"]["mean_per_frame_ms"] - fixed_baseline),
    }

    output_path = PROJECT_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
