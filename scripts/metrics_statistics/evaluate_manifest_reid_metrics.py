#!/usr/bin/env python3
"""Memory-safe Re-ID evaluation for manifest-backed anonymised outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reid_evaluator import ReIDEvaluator


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_boxes(path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int, int]]] = {}
    for row in read_csv(path):
        grouped.setdefault(row["image_id"], []).append(
            (int(float(row["x1"])), int(float(row["y1"])), int(float(row["x2"])), int(float(row["y2"])))
        )
    return grouped


def rows_for_method(manifest_path: Path, method: str, output_root: Path | None) -> list[dict[str, str]]:
    rows = read_csv(manifest_path)
    selected = [row for row in rows if row.get("method") == method]
    if selected:
        return selected
    if output_root is None:
        return []
    out_rows = []
    for row in rows:
        relative_path = row.get("relative_path", "")
        out_rows.append(
            {
                "relative_path": relative_path,
                "method": method,
                "output_path": str((output_root / method / relative_path).relative_to(PROJECT_ROOT)),
            }
        )
    return out_rows


def valid_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    left = max(0, min(x1, width))
    top = max(0, min(y1, height))
    right = max(0, min(x2, width))
    bottom = max(0, min(y2, height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def flush_batch(
    evaluator: ReIDEvaluator,
    gallery_crops: list[Image.Image],
    query_crops: list[Image.Image],
    batch_size: int,
    gallery_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
) -> None:
    if not gallery_crops:
        return
    gallery_embeddings.append(evaluator.extract_embeddings_adaface(gallery_crops, batch_size=batch_size))
    query_embeddings.append(evaluator.extract_embeddings_adaface(query_crops, batch_size=batch_size))
    for crop in gallery_crops + query_crops:
        crop.close()
    gallery_crops.clear()
    query_crops.clear()


def flush_arc_batch(
    evaluator: ReIDEvaluator,
    gallery_crops: list[Image.Image],
    query_crops: list[Image.Image],
    batch_size: int,
    gallery_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
) -> None:
    if not gallery_crops:
        return
    gallery_embeddings.append(evaluator.extract_embeddings_arcface(gallery_crops, batch_size=batch_size))
    query_embeddings.append(evaluator.extract_embeddings_arcface(query_crops, batch_size=batch_size))
    for crop in gallery_crops + query_crops:
        crop.close()
    gallery_crops.clear()
    query_crops.clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--raw-root", default="data/castle2024/raw")
    parser.add_argument("--adaface-ckpt", default="data/models/adaface_ir50_ms1mv2.ckpt")
    parser.add_argument("--arcface-onnx", default="~/.insightface/models/buffalo_l/w600k_r50.onnx")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--details-json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-arcface", action="store_true")
    parser.add_argument(
        "--require-arcface-gpu",
        action="store_true",
        help="Fail if ArcFace cannot initialize CUDAExecutionProvider when --device is cuda.",
    )
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    detections_path = PROJECT_ROOT / args.detections
    raw_root = PROJECT_ROOT / args.raw_root
    output_root = PROJECT_ROOT / args.output_root if args.output_root else None
    summary_path = PROJECT_ROOT / args.summary_json
    details_path = PROJECT_ROOT / args.details_json
    boxes_by_image = load_boxes(detections_path)
    rows = rows_for_method(manifest_path, args.method, output_root)

    evaluator = ReIDEvaluator(
        adaface_ckpt_path=str(PROJECT_ROOT / args.adaface_ckpt),
        arcface_onnx_path=str(Path(os.path.expanduser(args.arcface_onnx))),
        device=args.device,
        require_arcface_gpu=args.require_arcface_gpu,
    )

    gallery_ada_parts: list[np.ndarray] = []
    query_ada_parts: list[np.ndarray] = []
    gallery_arc_parts: list[np.ndarray] = []
    query_arc_parts: list[np.ndarray] = []
    gallery_batch: list[Image.Image] = []
    query_batch: list[Image.Image] = []
    arc_gallery_batch: list[Image.Image] = []
    arc_query_batch: list[Image.Image] = []
    crop_metadata: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []

    for row in rows:
        relative_path = row.get("relative_path", "")
        output_path = PROJECT_ROOT / row.get("output_path", "")
        raw_path = raw_root / relative_path
        if not raw_path.exists() or not output_path.exists():
            skipped_rows.append(
                {
                    "relative_path": relative_path,
                    "reason": f"raw_exists={raw_path.exists()} output_exists={output_path.exists()}",
                }
            )
            continue
        boxes = boxes_by_image.get(relative_path, [])
        if not boxes:
            continue
        with Image.open(raw_path) as raw_img, Image.open(output_path) as out_img:
            raw_img = raw_img.convert("RGB")
            out_img = out_img.convert("RGB")
            for box_idx, box in enumerate(boxes):
                valid = valid_box(box, raw_img.width, raw_img.height)
                if valid is None:
                    continue
                gallery_crop = raw_img.crop(valid)
                query_crop = out_img.crop(valid)
                gallery_batch.append(gallery_crop.copy())
                query_batch.append(query_crop.copy())
                if not args.skip_arcface:
                    arc_gallery_batch.append(gallery_crop.copy())
                    arc_query_batch.append(query_crop.copy())
                gallery_crop.close()
                query_crop.close()
                crop_metadata.append({"image_id": relative_path, "box": list(valid), "box_idx": box_idx})
                if len(gallery_batch) >= args.batch_size:
                    flush_batch(
                        evaluator,
                        gallery_batch,
                        query_batch,
                        args.batch_size,
                        gallery_ada_parts,
                        query_ada_parts,
                    )
                if not args.skip_arcface and len(arc_gallery_batch) >= args.batch_size:
                    flush_arc_batch(
                        evaluator,
                        arc_gallery_batch,
                        arc_query_batch,
                        args.batch_size,
                        gallery_arc_parts,
                        query_arc_parts,
                    )

    flush_batch(evaluator, gallery_batch, query_batch, args.batch_size, gallery_ada_parts, query_ada_parts)
    if not args.skip_arcface:
        flush_arc_batch(evaluator, arc_gallery_batch, arc_query_batch, args.batch_size, gallery_arc_parts, query_arc_parts)

    gallery_ada = np.concatenate(gallery_ada_parts, axis=0) if gallery_ada_parts else np.zeros((0, 512), dtype=np.float32)
    query_ada = np.concatenate(query_ada_parts, axis=0) if query_ada_parts else np.zeros((0, 512), dtype=np.float32)
    gallery_arc = np.concatenate(gallery_arc_parts, axis=0) if gallery_arc_parts else np.zeros((len(gallery_ada), 512), dtype=np.float32)
    query_arc = np.concatenate(query_arc_parts, axis=0) if query_arc_parts else np.zeros((len(query_ada), 512), dtype=np.float32)

    ada_metrics = evaluator.compute_reid_metrics(gallery_ada, query_ada)
    arc_metrics = evaluator.compute_reid_metrics(gallery_arc, query_arc)
    details: list[dict[str, Any]] = []
    if len(gallery_ada) and len(query_ada):
        ada_sim = np.dot(query_ada, gallery_ada.T)
        arc_sim = np.dot(query_arc, gallery_arc.T)
        ada_diag = np.diag(ada_sim)
        arc_diag = np.diag(arc_sim)
        ada_hits = ada_sim.argmax(axis=1) == np.arange(len(query_ada))
        arc_hits = arc_sim.argmax(axis=1) == np.arange(len(query_arc))
        for i, metadata in enumerate(crop_metadata):
            details.append(
                {
                    **metadata,
                    "method": args.method,
                    "adaface_cosine_sim": float(ada_diag[i]),
                    "adaface_hit": bool(ada_hits[i]),
                    "arcface_cosine_sim": float(arc_diag[i]),
                    "arcface_hit": bool(arc_hits[i]),
                }
            )

    summary = {
        "method": args.method,
        "face_crop_count": int(len(crop_metadata)),
        "adaface_cosine_sim_mean": ada_metrics["cosine_similarity"],
        "adaface_reid_rate": ada_metrics["reid_rate"],
        "arcface_cosine_sim_mean": arc_metrics["cosine_similarity"],
        "arcface_reid_rate": arc_metrics["reid_rate"],
        "device": args.device,
        "batch_size": args.batch_size,
        "arcface_available": evaluator.arcface_model is not None and not args.skip_arcface,
        "skipped_rows": skipped_rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    details_path.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
