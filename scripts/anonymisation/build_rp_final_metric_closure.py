#!/usr/bin/env python3
"""Close Reverse Personalization metrics after failed-frame retry recovery."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.perceptual_metrics import evaluate_manifest
from src.evaluation.reid_evaluator import ReIDEvaluator
from src.utils.compute_policy import build_compute_policy


DEFAULT_RP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "submission_evidence"
    / "03_anonymisation"
    / "05_reverse_personalization"
)
DEFAULT_ORIGINAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "submission_evidence"
    / "03_anonymisation"
    / "05_reverse_personalization"
    / "01_rp_final_manifest.csv"
)
DEFAULT_LOCKED_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "submission_evidence"
    / "01_protocol"
    / "01_locked_500_input_manifest.csv"
)
DEFAULT_RETRY_RESULTS = DEFAULT_RP_DIR / "02_rp_retry_results.csv"
DEFAULT_DETECTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "submission_evidence"
    / "02_face_detection"
    / "13_anonymisation_protocol_face_boxes.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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
    grouped: dict[str, list[tuple[int, int, int, int]]] = {}
    for row in read_csv(path):
        grouped.setdefault(row["image_id"], []).append(
            (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
        )
    return grouped


def retry_output_map(retry_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    mapped: dict[str, dict[str, str]] = {}
    for row in retry_rows:
        if row.get("retry_status") in {"RECOVERED", "RECOVERED_EXISTING_OUTPUT"}:
            out = PROJECT_ROOT / row.get("output_path", "")
            if out.exists():
                mapped[row["relative_path"]] = row
    return mapped


def build_final_manifest(
    locked_rows: list[dict[str, str]],
    original_rows: list[dict[str, str]],
    retry_rows: list[dict[str, str]],
    output_path: Path,
) -> list[dict[str, Any]]:
    original_by_rel = {row["relative_path"]: row for row in original_rows}
    recovered = retry_output_map(retry_rows)
    final_rows: list[dict[str, Any]] = []
    for locked in locked_rows:
        row = original_by_rel.get(locked["relative_path"], locked)
        relative_path = row["relative_path"]
        original_output = PROJECT_ROOT / row.get(
            "output_path",
            str(Path("outputs/03_anonymisation/05_reverse_personalization/images") / relative_path),
        )
        status = row.get("status", "")
        output = original_output
        source = "original_comparable_run"
        if not output.exists() and relative_path in recovered:
            output = PROJECT_ROOT / recovered[relative_path]["output_path"]
            status = "recovered_after_retry"
            source = recovered[relative_path].get("retry_variant", "retry")
        elif output.exists() and relative_path in recovered:
            # The retry pass copied some already-existing outputs into the retry tree.
            # Preserve the original comparable output path to avoid duplicate evidence.
            status = "ok" if status in {"ok", "copied", "existing"} else status
        elif not output.exists():
            status = "failed_after_retry"
            source = "persistent_failure"

        final_rows.append(
            {
                "relative_path": relative_path,
                "method": "reverse_personalization",
                "output_path": str(output.relative_to(PROJECT_ROOT)) if output.exists() else row.get("output_path", ""),
                "status": status,
                "box_count": row.get("box_count", ""),
                "runtime_seconds": row.get("runtime_seconds", ""),
                "source": source,
            }
        )
    write_csv(
        output_path,
        final_rows,
        ["relative_path", "method", "output_path", "status", "box_count", "runtime_seconds", "source"],
    )
    return final_rows


def compute_reid_from_manifest(
    manifest_rows: list[dict[str, Any]],
    raw_root: Path,
    detections_path: Path,
    output_details: Path,
    output_summary: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    boxes_by_image = load_boxes(detections_path)
    gallery: list[Image.Image] = []
    query: list[Image.Image] = []
    metadata: list[dict[str, Any]] = []

    output_by_rel = {
        row["relative_path"]: PROJECT_ROOT / str(row.get("output_path", ""))
        for row in manifest_rows
        if row.get("status") != "failed_after_retry"
    }
    for relative_path, boxes in boxes_by_image.items():
        output_path = output_by_rel.get(relative_path)
        raw_path = raw_root / relative_path
        if not output_path or not output_path.exists() or not raw_path.exists():
            continue
        with Image.open(raw_path).convert("RGB") as original, Image.open(output_path).convert("RGB") as anonymised:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
                gallery.append(original.crop(box))
                query.append(anonymised.crop(box))
                metadata.append({"image_id": relative_path, "box_idx": idx, "box": list(box)})

    if not gallery:
        summary = {
            "method": "reverse_personalization",
            "face_crop_count": 0,
            "adaface_cosine_sim_mean": "not_available",
            "adaface_reid_rate": "not_available",
            "arcface_cosine_sim_mean": "not_available",
            "arcface_reid_rate": "not_available",
        }
        output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        output_details.write_text("[]\n", encoding="utf-8")
        return summary

    evaluator = ReIDEvaluator(
        adaface_ckpt_path=str(PROJECT_ROOT / "data/models/adaface_ir50_ms1mv2.ckpt"),
        arcface_onnx_path=str(Path("~/.insightface/models/buffalo_l/w600k_r50.onnx").expanduser()),
        device=device,
    )
    gallery_ada = evaluator.extract_embeddings_adaface(gallery, batch_size=batch_size)
    query_ada = evaluator.extract_embeddings_adaface(query, batch_size=batch_size)
    gallery_arc = evaluator.extract_embeddings_arcface(gallery, batch_size=batch_size)
    query_arc = evaluator.extract_embeddings_arcface(query, batch_size=batch_size)

    ada_metrics = evaluator.compute_reid_metrics(gallery_ada, query_ada)
    arc_metrics = evaluator.compute_reid_metrics(gallery_arc, query_arc)
    sims_ada = np.diag(np.dot(query_ada, gallery_ada.T))
    sims_arc = np.diag(np.dot(query_arc, gallery_arc.T))
    hits_ada = np.dot(query_ada, gallery_ada.T).argmax(axis=1) == np.arange(len(gallery))
    hits_arc = np.dot(query_arc, gallery_arc.T).argmax(axis=1) == np.arange(len(gallery))

    details: list[dict[str, Any]] = []
    for idx, meta in enumerate(metadata):
        details.append(
            {
                **meta,
                "method": "reverse_personalization",
                "adaface_cosine_sim": float(sims_ada[idx]),
                "adaface_hit": bool(hits_ada[idx]),
                "arcface_cosine_sim": float(sims_arc[idx]),
                "arcface_hit": bool(hits_arc[idx]),
            }
        )
    summary = {
        "method": "reverse_personalization",
        "face_crop_count": len(gallery),
        "adaface_cosine_sim_mean": ada_metrics["cosine_similarity"],
        "adaface_reid_rate": ada_metrics["reid_rate"],
        "arcface_cosine_sim_mean": arc_metrics["cosine_similarity"],
        "arcface_reid_rate": arc_metrics["reid_rate"],
        "arcface_available": evaluator.arcface_model is not None,
        "device": device,
        "batch_size": batch_size,
    }
    output_details.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-manifest", type=Path, default=DEFAULT_ORIGINAL_MANIFEST)
    parser.add_argument("--locked-manifest", type=Path, default=DEFAULT_LOCKED_MANIFEST)
    parser.add_argument("--retry-results", type=Path, default=DEFAULT_RETRY_RESULTS)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "castle2024" / "raw")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RP_DIR)
    parser.add_argument("--eval-scale", type=float, default=0.25)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--no-gpu-lpips", action="store_true")
    args = parser.parse_args()

    locked_rows = read_csv(args.locked_manifest)
    original_rows = read_csv(args.original_manifest)
    retry_rows = read_csv(args.retry_results)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = output_dir / "rp_final_482_manifest.csv"
    final_rows = build_final_manifest(locked_rows, original_rows, retry_rows, final_manifest_path)

    perceptual_json = output_dir / "rp_final_482_perceptual.json"
    perceptual = evaluate_manifest(
        final_manifest_path,
        args.raw_root,
        perceptual_json,
        use_gpu=not args.no_gpu_lpips,
        eval_scale=args.eval_scale,
    )

    policy = build_compute_policy()
    device = args.device or policy.device
    batch_size = args.batch_size or policy.reid_batch_size
    reid_summary = compute_reid_from_manifest(
        final_rows,
        args.raw_root,
        args.detections,
        output_dir / "rp_final_482_reid_details.json",
        output_dir / "rp_final_482_reid_summary.json",
        device=device,
        batch_size=batch_size,
    )

    success_rows = [
        row
        for row in final_rows
        if row.get("output_path") and (PROJECT_ROOT / str(row.get("output_path", ""))).is_file()
    ]
    failure_rows = [row for row in final_rows if row.get("status") == "failed_after_retry"]
    p_summary = perceptual.get("summary", {}).get("reverse_personalization", {})
    runtimes = []
    for row in final_rows:
        try:
            value = float(row.get("runtime_seconds") or 0.0)
        except Exception:
            continue
        if value > 0:
            runtimes.append(value)

    metric_row = {
        "method": "reverse_personalization",
        "subset": "locked_500_comparable",
        "n_input_frames": len(final_rows),
        "n_success": len(success_rows),
        "n_failure": len(failure_rows),
        "face_rows": sum(1 for row in final_rows if str(row.get("box_count", "0")) not in {"", "0"}),
        "SSIM_mean": p_summary.get("ssim_mean", "not_available"),
        "SSIM_std": p_summary.get("ssim_std", "not_available"),
        "LPIPS_mean": p_summary.get("lpips_mean", "not_available"),
        "LPIPS_std": p_summary.get("lpips_std", "not_available"),
        "AdaFace_cosine_mean": reid_summary.get("adaface_cosine_sim_mean", "not_available"),
        "AdaFace_reid_rate": reid_summary.get("adaface_reid_rate", "not_available"),
        "ArcFace_cosine_mean": reid_summary.get("arcface_cosine_sim_mean", "not_available"),
        "ArcFace_reid_rate": reid_summary.get("arcface_reid_rate", "not_available"),
        "runtime_mean_seconds": round(statistics.mean(runtimes), 6) if runtimes else "not_available",
        "runtime_total_wall_seconds": round(sum(runtimes), 6) if runtimes else "not_available",
        "evidence_level": "improved_partial_comparable_with_final_metrics",
        "limitation": "18/500 frames still failed after targeted retry; RP remains runtime/failure-limited and non-default.",
        "report_safe_claim": "Reverse Personalization improved from 444 to 482 successful outputs after failed-frame retry, and final metrics were recomputed on the available outputs; it remains partial comparable evidence, not a deployment fallback.",
    }
    fields = list(metric_row)
    write_csv(output_dir / "09_rp_final_metric_summary.csv", [metric_row], fields)

    before_after_fields = [
        "stage",
        "n_input_frames",
        "n_success",
        "n_failure",
        "AdaFace_reid_rate",
        "ArcFace_reid_rate",
        "SSIM_mean",
        "LPIPS_mean",
        "runtime_total_wall_seconds",
        "evidence_level",
    ]
    before_after = [
        {
            "stage": "before_retry",
            "n_input_frames": 500,
            "n_success": 444,
            "n_failure": 56,
            "AdaFace_reid_rate": "0.13570741097208855",
            "ArcFace_reid_rate": "0.1530317613089509",
            "SSIM_mean": "0.9914419491692018",
            "LPIPS_mean": "0.005044553427839534",
            "runtime_total_wall_seconds": "70918.933639",
            "evidence_level": "partial_comparable_with_method_failures",
        },
        {
            "stage": "after_failed_frame_retry",
            "n_input_frames": metric_row["n_input_frames"],
            "n_success": metric_row["n_success"],
            "n_failure": metric_row["n_failure"],
            "AdaFace_reid_rate": metric_row["AdaFace_reid_rate"],
            "ArcFace_reid_rate": metric_row["ArcFace_reid_rate"],
            "SSIM_mean": metric_row["SSIM_mean"],
            "LPIPS_mean": metric_row["LPIPS_mean"],
            "runtime_total_wall_seconds": metric_row["runtime_total_wall_seconds"],
            "evidence_level": metric_row["evidence_level"],
        },
    ]
    write_csv(output_dir / "03_rp_before_after_fix_comparison.csv", before_after, before_after_fields)


if __name__ == "__main__":
    main()
