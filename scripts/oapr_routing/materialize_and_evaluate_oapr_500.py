#!/usr/bin/env python3
"""Materialise and evaluate OAPR routed outputs on the locked 500-frame protocol."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_SRC = PROJECT_ROOT / "app" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from privacy_pipeline_app.pipeline_demo import apply_blur, apply_layered, apply_solid
from src.evaluation.perceptual_metrics import evaluate_manifest
from src.evaluation.reid_evaluator import ReIDEvaluator
from src.utils.compute_policy import build_compute_policy


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "runs" / "oapr_evaluation"
DEFAULT_ROUTING_LOG = DEFAULT_OUTPUT_DIR / "oapr_500_routing_log.csv"
DEFAULT_DETECTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "submission_evidence"
    / "classical_baselines"
    / "anonymisation_eval_subset_yolo_scrfd_fallback.csv"
)
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"
METHOD_FUNCS = {
    "blur": apply_blur,
    "solid_mask": apply_solid,
    "layered": apply_layered,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows available.")
    else:
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_boxes(path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[row["image_id"]].append((int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])))
    return dict(grouped)


def objective_method_name(objective: str) -> str:
    return f"oapr_{objective}"


def materialize_outputs(
    routing_rows: list[dict[str, str]],
    boxes_by_image: dict[str, list[tuple[int, int, int, int]]],
    raw_root: Path,
    output_dir: Path,
    image_quality: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in routing_rows:
        grouped[row["objective_mode"]].append(row)

    for objective, rows in sorted(grouped.items()):
        counts: Counter[str] = Counter()
        runtimes: list[float] = []
        objective_failures = 0
        for row in rows:
            started = time.perf_counter()
            relative_path = row["relative_path"]
            selected = row["selected_method"]
            method = objective_method_name(objective)
            raw_path = raw_root / relative_path
            output_path = output_dir / "materialized_outputs" / method / relative_path
            boxes = boxes_by_image.get(relative_path, [])
            try:
                if selected == "copy":
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    # Copy-through is intentionally materialised as a symlink when possible.
                    if output_path.exists() or output_path.is_symlink():
                        output_path.unlink()
                    output_path.symlink_to(raw_path.resolve())
                else:
                    with Image.open(raw_path).convert("RGB") as image:
                        func = METHOD_FUNCS[selected]
                        anonymised = func(image, boxes)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        anonymised.save(output_path, quality=image_quality, method=6)
                elapsed = round(time.perf_counter() - started, 6)
                counts[selected] += 1
                runtimes.append(elapsed)
                manifest_rows.append(
                    {
                        "relative_path": relative_path,
                        "method": method,
                        "objective_mode": objective,
                        "selected_method": selected,
                        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
                        "boxes_processed": len(boxes),
                        "runtime_seconds": elapsed,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                elapsed = round(time.perf_counter() - started, 6)
                objective_failures += 1
                failure_rows.append(
                    {
                        "relative_path": relative_path,
                        "objective_mode": objective,
                        "selected_method": selected,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "runtime_seconds": elapsed,
                    }
                )
        runtime_rows.append(
            {
                "objective_mode": objective,
                "frames_requested": len(rows),
                "frames_ok": len(rows) - objective_failures,
                "frames_failed": objective_failures,
                "method_counts": json.dumps(dict(counts), sort_keys=True),
                "runtime_total_seconds": round(sum(runtimes), 6),
                "runtime_mean_seconds": round(statistics.mean(runtimes), 6) if runtimes else "not_available",
                "runtime_median_seconds": round(statistics.median(runtimes), 6) if runtimes else "not_available",
            }
        )
    return manifest_rows, runtime_rows, failure_rows


def compute_reid_from_oapr_manifest(
    manifest_rows: list[dict[str, Any]],
    boxes_by_image: dict[str, list[tuple[int, int, int, int]]],
    raw_root: Path,
    output_details: Path,
    output_summary: Path,
    device: str,
    batch_size: int,
    allow_arcface_cpu_fallback: bool = False,
) -> dict[str, dict[str, Any]]:
    by_method: dict[str, dict[str, Path]] = defaultdict(dict)
    for row in manifest_rows:
        if row.get("status") == "ok":
            by_method[row["method"]][row["relative_path"]] = PROJECT_ROOT / row["output_path"]

    all_relative_paths = sorted({row["relative_path"] for row in manifest_rows})
    gallery_crops: list[Image.Image] = []
    crop_meta: list[dict[str, Any]] = []
    for relative_path in all_relative_paths:
        raw_path = raw_root / relative_path
        boxes = boxes_by_image.get(relative_path, [])
        if not boxes or not raw_path.exists():
            continue
        with Image.open(raw_path).convert("RGB") as original:
            for box_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
                gallery_crops.append(original.crop(box))
                crop_meta.append({"image_id": relative_path, "box_idx": box_idx, "box": list(box)})

    if not gallery_crops:
        output_details.write_text("[]\n", encoding="utf-8")
        output_summary.write_text("{}\n", encoding="utf-8")
        return {}

    evaluator = ReIDEvaluator(
        adaface_ckpt_path=str(PROJECT_ROOT / "data/models/adaface_ir50_ms1mv2.ckpt"),
        arcface_onnx_path=str(Path("~/.insightface/models/buffalo_l/w600k_r50.onnx").expanduser()),
        device=device,
        require_arcface_gpu=(device != "cpu" and not allow_arcface_cpu_fallback),
    )
    gallery_ada = evaluator.extract_embeddings_adaface(gallery_crops, batch_size=batch_size)
    gallery_arc = evaluator.extract_embeddings_arcface(gallery_crops, batch_size=batch_size)

    details: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for method, output_by_rel in sorted(by_method.items()):
        query_crops: list[Image.Image] = []
        query_meta: list[dict[str, Any]] = []
        for meta in crop_meta:
            relative_path = meta["image_id"]
            output_path = output_by_rel.get(relative_path)
            if output_path is None or not output_path.exists():
                continue
            with Image.open(output_path).convert("RGB") as anonymised:
                query_crops.append(anonymised.crop(tuple(meta["box"])))
                query_meta.append(meta)
        if len(query_crops) != len(gallery_crops):
            # This should not happen for a complete OAPR materialisation; keep it explicit.
            summary[method] = {
                "face_crop_count": len(query_crops),
                "expected_face_crop_count": len(gallery_crops),
                "adaface_cosine_sim_mean": "not_available",
                "adaface_reid_rate": "not_available",
                "arcface_cosine_sim_mean": "not_available",
                "arcface_reid_rate": "not_available",
                "status": "incomplete_query_crops",
            }
            continue
        query_ada = evaluator.extract_embeddings_adaface(query_crops, batch_size=batch_size)
        query_arc = evaluator.extract_embeddings_arcface(query_crops, batch_size=batch_size)
        ada_metrics = evaluator.compute_reid_metrics(gallery_ada, query_ada)
        arc_metrics = evaluator.compute_reid_metrics(gallery_arc, query_arc)
        sims_ada = np.diag(np.dot(query_ada, gallery_ada.T))
        sims_arc = np.diag(np.dot(query_arc, gallery_arc.T))
        hits_ada = np.dot(query_ada, gallery_ada.T).argmax(axis=1) == np.arange(len(gallery_crops))
        hits_arc = np.dot(query_arc, gallery_arc.T).argmax(axis=1) == np.arange(len(gallery_crops))
        summary[method] = {
            "face_crop_count": len(query_crops),
            "adaface_cosine_sim_mean": ada_metrics["cosine_similarity"],
            "adaface_reid_rate": ada_metrics["reid_rate"],
            "arcface_cosine_sim_mean": arc_metrics["cosine_similarity"],
            "arcface_reid_rate": arc_metrics["reid_rate"],
            "arcface_available": evaluator.arcface_model is not None,
            "status": "ok",
        }
        for idx, meta in enumerate(query_meta):
            details.append(
                {
                    **meta,
                    "method": method,
                    "adaface_cosine_sim": float(sims_ada[idx]),
                    "adaface_hit": bool(hits_ada[idx]),
                    "arcface_cosine_sim": float(sims_arc[idx]),
                    "arcface_hit": bool(hits_arc[idx]),
                }
            )
    output_details.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_metric_summary(
    manifest_rows: list[dict[str, Any]],
    perceptual: dict[str, Any],
    reid: dict[str, dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    p_summary = perceptual.get("summary", {})
    runtime_by_method = {objective_method_name(row["objective_mode"]): row for row in runtime_rows}
    rows: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in manifest_rows}):
        method_rows = [row for row in manifest_rows if row["method"] == method]
        p = p_summary.get(method, {})
        r = reid.get(method, {})
        rt = runtime_by_method.get(method, {})
        rows.append(
            {
                "method": method,
                "objective_mode": method.removeprefix("oapr_"),
                "n_input_frames": len(method_rows),
                "n_success": sum(1 for row in method_rows if row.get("status") == "ok"),
                "n_failure": sum(1 for row in method_rows if row.get("status") != "ok"),
                "method_counts": rt.get("method_counts", ""),
                "SSIM_mean": p.get("ssim_mean", "not_available"),
                "SSIM_std": p.get("ssim_std", "not_available"),
                "LPIPS_mean": p.get("lpips_mean", "not_available"),
                "LPIPS_std": p.get("lpips_std", "not_available"),
                "AdaFace_cosine_mean": r.get("adaface_cosine_sim_mean", "not_available"),
                "AdaFace_reid_rate": r.get("adaface_reid_rate", "not_available"),
                "ArcFace_cosine_mean": r.get("arcface_cosine_sim_mean", "not_available"),
                "ArcFace_reid_rate": r.get("arcface_reid_rate", "not_available"),
                "runtime_mean_seconds": rt.get("runtime_mean_seconds", "not_available"),
                "runtime_total_seconds": rt.get("runtime_total_seconds", "not_available"),
                "evidence_level": "actual_routed_output_metrics",
                "report_safe_claim": "OAPR objective mode was materialised as real routed images and evaluated; this is objective-specific evidence, not global dominance over fixed blur.",
            }
        )
    return rows


def read_fixed_method_metrics() -> dict[str, dict[str, Any]]:
    fixed: dict[str, dict[str, Any]] = {}
    comparison_path = PROJECT_ROOT / "outputs/03_anonymisation/01_all_methods_comparison.csv"
    if comparison_path.exists():
        for row in read_csv(comparison_path):
            if row.get("n_input_frames") != "500":
                continue
            fixed[row["method"]] = {
                "SSIM_mean": _to_float(row.get("SSIM_mean")),
                "LPIPS_mean": _to_float(row.get("LPIPS_mean")),
                "AdaFace_reid_rate": _to_float(row.get("AdaFace_reid_rate")),
                "ArcFace_reid_rate": _to_float(row.get("ArcFace_reid_rate")),
            }
    for path in [
        PROJECT_ROOT / "outputs/03_anonymisation/01_all_methods_comparison.csv",
        PROJECT_ROOT / "outputs/adaptive_upgrade/stronger_baseline_full_metric_results.csv",
    ]:
        if path.exists():
            for row in read_csv(path):
                fixed[row["method"]] = {
                    "SSIM_mean": _to_float(row.get("SSIM_mean")),
                    "LPIPS_mean": _to_float(row.get("LPIPS_mean")),
                    "AdaFace_reid_rate": _to_float(row.get("AdaFace_metric") or row.get("AdaFace_reid_rate")),
                    "ArcFace_reid_rate": _to_float(row.get("ArcFace_metric") or row.get("ArcFace_reid_rate")),
                }
    for path in [
        PROJECT_ROOT / "outputs/03_anonymisation/03_nullface/01_nullface_full_500_manifest.csv",
        PROJECT_ROOT / "outputs/03_anonymisation/04_diffusion/07_diffusion_full_500_metric_summary.csv",
        PROJECT_ROOT / "outputs/03_anonymisation/05_reverse_personalization/09_rp_final_metric_summary.csv",
    ]:
        if path.exists():
            for row in read_csv(path):
                method = row.get("method", path.parent.name)
                fixed[method] = {
                    "SSIM_mean": _to_float(row.get("SSIM_mean")),
                    "LPIPS_mean": _to_float(row.get("LPIPS_mean")),
                    "AdaFace_reid_rate": _to_float(row.get("AdaFace_metric") or row.get("AdaFace_reid_rate")),
                    "ArcFace_reid_rate": _to_float(row.get("ArcFace_metric") or row.get("ArcFace_reid_rate")),
                }
    return fixed


def _to_float(value: Any) -> float | None:
    try:
        if value in {None, "", "not_available"}:
            return None
        return float(value)
    except Exception:
        return None


def build_comparisons(summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixed = read_fixed_method_metrics()
    comparisons: list[dict[str, Any]] = []
    wins: list[dict[str, Any]] = []
    for oapr in summary_rows:
        oapr_ssim = _to_float(oapr.get("SSIM_mean"))
        oapr_lpips = _to_float(oapr.get("LPIPS_mean"))
        oapr_ada = _to_float(oapr.get("AdaFace_reid_rate"))
        oapr_arc = _to_float(oapr.get("ArcFace_reid_rate"))
        objective = oapr["objective_mode"]
        win_counter = Counter()
        for fixed_method, metrics in sorted(fixed.items()):
            fixed_ssim = _to_float(metrics.get("SSIM_mean"))
            fixed_lpips = _to_float(metrics.get("LPIPS_mean"))
            fixed_ada = _to_float(metrics.get("AdaFace_reid_rate"))
            fixed_arc = _to_float(metrics.get("ArcFace_reid_rate"))
            privacy_delta = None if oapr_ada is None or fixed_ada is None else fixed_ada - oapr_ada
            utility_delta = None if oapr_ssim is None or fixed_ssim is None else oapr_ssim - fixed_ssim
            if privacy_delta is not None and privacy_delta > 0:
                win_counter["privacy_wins"] += 1
            elif privacy_delta is not None and privacy_delta < 0:
                win_counter["privacy_losses"] += 1
            if utility_delta is not None and utility_delta > 0:
                win_counter["utility_wins"] += 1
            elif utility_delta is not None and utility_delta < 0:
                win_counter["utility_losses"] += 1
            comparisons.append(
                {
                    "objective_mode": objective,
                    "fixed_method": fixed_method,
                    "OAPR_SSIM_mean": oapr_ssim if oapr_ssim is not None else "not_available",
                    "fixed_SSIM_mean": fixed_ssim if fixed_ssim is not None else "not_available",
                    "SSIM_delta_OAPR_minus_fixed": utility_delta if utility_delta is not None else "not_available",
                    "OAPR_LPIPS_mean": oapr_lpips if oapr_lpips is not None else "not_available",
                    "fixed_LPIPS_mean": fixed_lpips if fixed_lpips is not None else "not_available",
                    "OAPR_AdaFace_reid_rate": oapr_ada if oapr_ada is not None else "not_available",
                    "fixed_AdaFace_reid_rate": fixed_ada if fixed_ada is not None else "not_available",
                    "privacy_delta_fixed_minus_OAPR_positive_means_OAPR_better": privacy_delta
                    if privacy_delta is not None
                    else "not_available",
                    "OAPR_ArcFace_reid_rate": oapr_arc if oapr_arc is not None else "not_available",
                    "fixed_ArcFace_reid_rate": fixed_arc if fixed_arc is not None else "not_available",
                    "comparison_basis": "actual_routed_output_metrics",
                    "report_safe_claim": "Objective-specific comparison only; OAPR is not claimed to dominate fixed blur globally.",
                }
            )
        wins.append(
            {
                "objective_mode": objective,
                "privacy_wins_vs_fixed_methods": win_counter["privacy_wins"],
                "privacy_losses_vs_fixed_methods": win_counter["privacy_losses"],
                "utility_wins_vs_fixed_methods": win_counter["utility_wins"],
                "utility_losses_vs_fixed_methods": win_counter["utility_losses"],
                "interpretation": "Objective-specific routed output evidence; use to support bounded routing claims only.",
            }
        )
    return comparisons, wins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-log", type=Path, default=DEFAULT_ROUTING_LOG)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-quality", type=int, default=85)
    parser.add_argument("--eval-scale", type=float, default=0.25)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--no-gpu-lpips", action="store_true")
    parser.add_argument(
        "--reuse-materialized",
        action="store_true",
        help="Reuse an existing oapr_materialized_manifest.csv and runtime/failure logs.",
    )
    parser.add_argument(
        "--reuse-perceptual",
        action="store_true",
        help="Reuse an existing oapr_perceptual_summary.json.",
    )
    parser.add_argument(
        "--allow-arcface-cpu-fallback",
        action="store_true",
        help="Explicitly permit ArcFace CPU fallback. Do not use for final GPU-required metrics.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    routing_rows = read_csv(args.routing_log)
    boxes_by_image = load_boxes(args.detections)
    manifest_fields = [
        "relative_path",
        "method",
        "objective_mode",
        "selected_method",
        "output_path",
        "boxes_processed",
        "runtime_seconds",
        "status",
    ]
    manifest_path = args.output_dir / "oapr_materialized_manifest.csv"
    runtime_path = args.output_dir / "oapr_runtime_summary.csv"
    failure_path = args.output_dir / "oapr_failure_log.csv"
    if args.reuse_materialized and manifest_path.exists() and runtime_path.exists() and failure_path.exists():
        manifest_rows = read_csv(manifest_path)
        runtime_rows = read_csv(runtime_path)
        failure_rows = read_csv(failure_path)
    else:
        manifest_rows, runtime_rows, failure_rows = materialize_outputs(
            routing_rows,
            boxes_by_image,
            args.raw_root,
            args.output_dir,
            args.image_quality,
        )
        write_csv(manifest_path, manifest_rows, manifest_fields)
        write_csv(args.output_dir / "oapr_runtime_summary.csv", runtime_rows, list(runtime_rows[0]))
        write_csv(
            args.output_dir / "oapr_failure_log.csv",
            failure_rows,
            ["relative_path", "objective_mode", "selected_method", "status", "error", "runtime_seconds"],
        )

    perceptual_json = args.output_dir / "oapr_perceptual_summary.json"
    if args.reuse_perceptual and perceptual_json.exists():
        perceptual = json.loads(perceptual_json.read_text(encoding="utf-8"))
    else:
        perceptual = evaluate_manifest(
            manifest_path,
            args.raw_root,
            perceptual_json,
            use_gpu=not args.no_gpu_lpips,
            eval_scale=args.eval_scale,
        )

    policy = build_compute_policy()
    device = args.device or policy.device
    batch_size = args.batch_size or policy.reid_batch_size
    reid = compute_reid_from_oapr_manifest(
        manifest_rows,
        boxes_by_image,
        args.raw_root,
        args.output_dir / "oapr_reid_details.json",
        args.output_dir / "oapr_reid_summary.json",
        device=device,
        batch_size=batch_size,
        allow_arcface_cpu_fallback=args.allow_arcface_cpu_fallback,
    )

    summary_rows = build_metric_summary(manifest_rows, perceptual, reid, runtime_rows)
    summary_fields = list(summary_rows[0])
    write_csv(args.output_dir / "oapr_full_metric_summary.csv", summary_rows, summary_fields)

    per_frame_rows = []
    for detail in perceptual.get("detailed", []):
        per_frame_rows.append(
            {
                "relative_path": detail["relative_path"],
                "method": detail["method"],
                "objective_mode": detail["method"].removeprefix("oapr_"),
                "SSIM": detail["ssim"],
                "LPIPS": detail["lpips"],
            }
        )
    if per_frame_rows:
        write_csv(args.output_dir / "oapr_per_frame_metrics.csv", per_frame_rows, list(per_frame_rows[0]))

    comparisons, wins = build_comparisons(summary_rows)
    comparison_fields = list(comparisons[0])
    win_fields = list(wins[0])
    write_csv(args.output_dir / "oapr_vs_fixed_methods_metric_comparison.csv", comparisons, comparison_fields)
    write_csv(args.output_dir / "oapr_objective_specific_wins.csv", wins, win_fields)

    final_decision = f"""# OAPR Final Decision

## Status

`BOUNDED_ACTUAL_ROUTED_OUTPUT_EVIDENCE`

## Evidence Added

- Routed decisions: `{len(routing_rows)}`
- Materialized output rows: `{len(manifest_rows)}`
- Materialization failures: `{len(failure_rows)}`
- Objective modes: `{len(summary_rows)}`
- Metric basis: actual routed outputs, not weighted method-count estimates.

## Interpretation

OAPR now has metadata routing evidence and actual routed-output metric evidence over the locked 500-frame protocol. The evidence supports objective-aware method selection, privacy-first fallback, runtime-aware deterministic routing, and failure avoidance. It does not support a global claim that OAPR beats fixed blur across every objective.

## Canonical Claim Boundary

Use OAPR as a bounded, auditable policy layer: it selects among evidence-supported deterministic methods, avoids quality-limited advanced methods, and preserves fixed blur as the practical default when objective evidence does not justify a stronger action.
"""


if __name__ == "__main__":
    main()
