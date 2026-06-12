#!/usr/bin/env python3

"""Run face re-identification evaluation using AdaFace and ArcFace attackers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.castle_loader import CASTLEDataset
from src.evaluation.reid_evaluator import ReIDEvaluator
from src.utils.compute_policy import build_compute_policy


def load_boxes_by_image(path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """Load saved detector boxes grouped by image relative path."""
    grouped: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["image_id"]].append(
                (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--methods", nargs="+", default=["blur", "pixelate"])
    parser.add_argument("--output-root", default="outputs/dev_anonymised")
    parser.add_argument("--adaface-ckpt", default="data/models/adaface_ir50_ms1mv2.ckpt")
    parser.add_argument("--arcface-onnx", default="~/.insightface/models/buffalo_l/w600k_r50.onnx")
    parser.add_argument("--results-json", default="outputs/experimental_runs/classical_baselines/anonymisation_dev_results.json")
    parser.add_argument("--reid-results-json", default="outputs/experimental_runs/classical_baselines/reid_results.json")
    parser.add_argument("--device", default="", help="Empty means auto from compute policy")
    parser.add_argument("--batch-size", type=int, default=0, help="0 means auto from compute policy")
    args = parser.parse_args()
    policy = build_compute_policy()

    # Setup paths
    manifest_path = Path(args.manifest)
    detections_path = Path(args.detections)
    output_root = PROJECT_ROOT / args.output_root
    adaface_ckpt_path = PROJECT_ROOT / args.adaface_ckpt
    arcface_onnx_path = Path(os.path.expanduser(args.arcface_onnx))
    results_json_path = PROJECT_ROOT / args.results_json
    reid_results_path = PROJECT_ROOT / args.reid_results_json

    # Load dataset & detections
    dataset = CASTLEDataset(str(manifest_path), return_format="pil", filters={})
    boxes_by_image = load_boxes_by_image(detections_path)

    print("Extracting face crops...")
    gallery_crops = []
    # Maintain crop metadata to associate crops back to frames/boxes
    crop_metadata = []
    
    query_crops_by_method = defaultdict(list)

    for item in dataset:
        image = item["image"]
        relative_path = item["metadata"]["relative_path"]
        boxes = boxes_by_image.get(relative_path, [])
        
        # Load anonymised images
        anon_imgs = {}
        for method in args.methods:
            anon_path = output_root / method / relative_path
            if anon_path.exists():
                anon_imgs[method] = Image.open(anon_path)

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Safeguard against degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Original crop (gallery)
            gal_crop = image.crop((x1, y1, x2, y2))
            gallery_crops.append(gal_crop)
            crop_metadata.append({
                "image_id": relative_path,
                "box": [x1, y1, x2, y2],
                "box_idx": idx
            })

            # Anonymised crops (query)
            for method, anon_img in anon_imgs.items():
                q_crop = anon_img.crop((x1, y1, x2, y2))
                query_crops_by_method[method].append(q_crop)

    n_crops = len(gallery_crops)
    print(f"Extracted {n_crops} face crops from the dev set.")
    if n_crops == 0:
        print("No face crops found for evaluation.")
        return

    # Initialize evaluator
    print("Initializing Re-ID Evaluator...")
    device = args.device or policy.device
    batch_size = args.batch_size or policy.reid_batch_size
    evaluator = ReIDEvaluator(
        adaface_ckpt_path=str(adaface_ckpt_path),
        arcface_onnx_path=str(arcface_onnx_path),
        device=device
    )
    arcface_available = evaluator.arcface_model is not None
    if not arcface_available:
        print("ArcFace unavailable in this environment; proceeding with AdaFace-only and zeroed ArcFace fields.")

    print("Extracting gallery embeddings...")
    gallery_feats_ada = evaluator.extract_embeddings_adaface(gallery_crops, batch_size=batch_size)
    gallery_feats_arc = evaluator.extract_embeddings_arcface(gallery_crops, batch_size=batch_size)

    method_results = {}
    detailed_results = []

    for method in args.methods:
        query_crops = query_crops_by_method.get(method, [])
        if len(query_crops) != n_crops:
            print(f"Warning: Method '{method}' has {len(query_crops)} crops instead of expected {n_crops}. Skipping.")
            continue

        print(f"Extracting query embeddings for method: {method}...")
        query_feats_ada = evaluator.extract_embeddings_adaface(query_crops, batch_size=batch_size)
        query_feats_arc = evaluator.extract_embeddings_arcface(query_crops, batch_size=batch_size)

        # Compute summary metrics
        metrics_ada = evaluator.compute_reid_metrics(gallery_feats_ada, query_feats_ada)
        metrics_arc = evaluator.compute_reid_metrics(gallery_feats_arc, query_feats_arc)

        method_results[method] = {
            "adaface_cosine_sim_mean": metrics_ada["cosine_similarity"],
            "adaface_reid_rate": metrics_ada["reid_rate"],
            "arcface_cosine_sim_mean": metrics_arc["cosine_similarity"],
            "arcface_reid_rate": metrics_arc["reid_rate"],
            "face_crop_count": n_crops
        }

        # Calculate individual crop similarities
        sims_ada = np.diag(np.dot(query_feats_ada, gallery_feats_ada.T))
        sims_arc = np.diag(np.dot(query_feats_arc, gallery_feats_arc.T))

        # Check Rank-1 hit per query
        sim_matrix_ada = np.dot(query_feats_ada, gallery_feats_ada.T)
        predicted_idx_ada = sim_matrix_ada.argmax(axis=1)
        hits_ada = (predicted_idx_ada == np.arange(n_crops))

        sim_matrix_arc = np.dot(query_feats_arc, gallery_feats_arc.T)
        predicted_idx_arc = sim_matrix_arc.argmax(axis=1)
        hits_arc = (predicted_idx_arc == np.arange(n_crops))

        for i in range(n_crops):
            detailed_results.append({
                "image_id": crop_metadata[i]["image_id"],
                "box": crop_metadata[i]["box"],
                "box_idx": crop_metadata[i]["box_idx"],
                "method": method,
                "adaface_cosine_sim": float(sims_ada[i]),
                "adaface_hit": bool(hits_ada[i]),
                "arcface_cosine_sim": float(sims_arc[i]),
                "arcface_hit": bool(hits_arc[i])
            })

    # Save detailed results
    reid_results_path.parent.mkdir(parents=True, exist_ok=True)
    with reid_results_path.open("w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Detailed Re-ID metrics saved to {reid_results_path}")

    # Load and update consolidated results
    results_data = {}
    if results_json_path.exists():
        with results_json_path.open("r", encoding="utf-8") as f:
            results_data = json.load(f)

    # Record the metric protocol with the results.
    results_data["note"] = "All SSIM and LPIPS computed against WebP-compressed originals at eval_scale=0.25. Re-ID and cosine similarity evaluated using AdaFace (primary) and ArcFace (secondary) models."
    results_data["reid_summary"] = method_results
    results_data["reid_compute_policy"] = {
        "device": device,
        "batch_size": int(batch_size),
        "accelerator_total_gb": policy.accelerator_total_gb,
        "arcface_available": arcface_available,
    }

    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Consolidated results updated in {results_json_path}")

    # Print clean table
    print("\n" + "="*80)
    print("Re-ID Privacy Evaluation Results Summary".center(80))
    print("="*80)
    print(f"{'Method':<12} | {'AdaFace CosSim':<15} | {'AdaFace Re-ID':<15} | {'ArcFace CosSim':<15} | {'ArcFace Re-ID':<15}")
    print("-"*80)
    for method, metrics in method_results.items():
        print(f"{method:<12} | "
              f"{metrics['adaface_cosine_sim_mean']:<15.4f} | "
              f"{metrics['adaface_reid_rate']:<15.4f} | "
              f"{metrics['arcface_cosine_sim_mean']:<15.4f} | "
              f"{metrics['arcface_reid_rate']:<15.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import numpy as np
    main()
