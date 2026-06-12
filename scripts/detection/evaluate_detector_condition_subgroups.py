#!/usr/bin/env python3

"""Evaluate detector predictions across reviewed/candidate condition subgroups."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.detection_metrics import GroundTruthBox, ScoredBox, compute_average_precision


def load_ground_truth(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def to_gt_rows(df: pd.DataFrame) -> list[GroundTruthBox]:
    return [
        GroundTruthBox(
            image_id=str(row.image_id),
            box=(int(row.x1), int(row.y1), int(row.x2), int(row.y2)),
        )
        for row in df.itertuples(index=False)
    ]


def to_pred_rows(df: pd.DataFrame) -> list[ScoredBox]:
    return [
        ScoredBox(
            image_id=str(row.image_id),
            box=(int(row.x1), int(row.y1), int(row.x2), int(row.y2)),
            score=float(row.score),
        )
        for row in df.itertuples(index=False)
    ]


def zero_face_stats(pred_subset: pd.DataFrame, image_ids: set[str]) -> dict[str, float | int]:
    pred_images = set(pred_subset["image_id"].astype(str)) if not pred_subset.empty else set()
    false_positive_images = len(pred_images & image_ids)
    image_count = len(image_ids)
    return {
        "zero_face_false_positive_images": false_positive_images,
        "zero_face_specificity": 1.0 - (false_positive_images / max(1, image_count)),
    }


def score_for_oapr(metric: dict[str, float | int], gt_count: int, image_count: int) -> float:
    """Return a privacy-weighted detector score for routing analysis."""
    if gt_count == 0:
        return float(metric.get("zero_face_specificity", 0.0))
    recall = float(metric["recall"])
    f1 = float(metric["f1"])
    precision = float(metric["precision"])
    return 0.65 * recall + 0.25 * f1 + 0.10 * precision


def write_markdown(rows: list[dict[str, object]], output_path: Path, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| Subgroup | Images | GT boxes | AP | Precision | Recall | F1 | TP | FP | FN | OAPR score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subgroup']} | {row['image_count']} | {row['ground_truth_boxes']} | "
            f"{float(row['ap']):.4f} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{float(row['f1']):.4f} | {row['true_positives']} | {row['false_positives']} | "
            f"{row['false_negatives']} | {float(row['oapr_detector_score']):.4f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--category-review", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    protocol_name = args.protocol_name
    category_df = pd.read_csv(args.category_review)
    gt_df = load_ground_truth(Path(args.ground_truth))
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    membership_dir = output_dir / "subgroup_membership"
    models_dir.mkdir(parents=True, exist_ok=True)
    membership_dir.mkdir(parents=True, exist_ok=True)

    subgroup_masks = {
        "all_images": category_df["image_id"].notna(),
        "multi_face": category_df["face_count_category"].eq("multi_face"),
        "single_face": category_df["face_count_category"].eq("single_face"),
        "no_face": category_df["face_count_category"].eq("no_face"),
        "very_small_or_distant_face": category_df["face_scale_category"].eq("very_small_or_distant"),
        "small_face": category_df["face_scale_category"].eq("small"),
        "medium_face": category_df["face_scale_category"].eq("medium"),
        "large_face": category_df["face_scale_category"].eq("large"),
        "mixed_scale_face": category_df["face_scale_category"].eq("mixed_scale"),
        "edge_or_partial_face": category_df["edge_partial_face"].eq("yes"),
        "profile_or_occluded_face": category_df["profile_occluded_face"].eq("yes"),
        "downward_egocentric_view": category_df["downward_egocentric_view"].eq("yes"),
        "motion_blur_or_low_sharpness": category_df["blur_low_sharpness"].eq("yes"),
        "low_light_or_dim": category_df["low_light_dim"].eq("yes"),
        "high_clutter": category_df["clutter_level"].eq("high"),
        "outdoor_or_vehicle_scene": category_df["outdoor_vehicle_scene"].eq("yes"),
    }

    prediction_files = sorted(predictions_dir.glob("*_predictions.csv"))
    all_rows: list[dict[str, object]] = []
    for pred_file in prediction_files:
        model = pred_file.name.replace("_predictions.csv", "")
        pred_df = load_predictions(pred_file)
        model_rows: list[dict[str, object]] = []
        for subgroup, mask in subgroup_masks.items():
            members = category_df.loc[mask].copy()
            image_ids = set(members["image_id"].astype(str))
            if not image_ids:
                continue
            members.to_csv(membership_dir / f"{subgroup}.csv", index=False)
            gt_subset = gt_df[gt_df["image_id"].astype(str).isin(image_ids)]
            pred_subset = pred_df[pred_df["image_id"].astype(str).isin(image_ids)]
            metric = compute_average_precision(to_pred_rows(pred_subset), to_gt_rows(gt_subset), iou_threshold=0.5)
            metric.update(zero_face_stats(pred_subset, image_ids) if len(gt_subset) == 0 else {
                "zero_face_false_positive_images": 0,
                "zero_face_specificity": "",
            })
            row = {
                "protocol": protocol_name,
                "model": model,
                "subgroup": subgroup,
                "image_count": len(image_ids),
                "ground_truth_boxes": len(gt_subset),
                "prediction_boxes": len(pred_subset),
                **metric,
            }
            row["oapr_detector_score"] = score_for_oapr(row, len(gt_subset), len(image_ids))
            model_rows.append(row)
            all_rows.append(row)
        model_dir = models_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "subgroup_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(model_rows[0]))
            writer.writeheader()
            writer.writerows(model_rows)
        (model_dir / "subgroup_scores.json").write_text(json.dumps(model_rows, indent=2), encoding="utf-8")
        write_markdown(model_rows, model_dir / "subgroup_scores.md", f"{protocol_name} {model} subgroup scores")

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(output_dir / "all_models_subgroup_scores.csv", index=False)
    (output_dir / "all_models_subgroup_scores.json").write_text(
        json.dumps(all_rows, indent=2),
        encoding="utf-8",
    )
    write_markdown(all_rows, output_dir / "all_models_subgroup_scores.md", f"{protocol_name} all detector subgroup scores")

    best_rows: list[dict[str, object]] = []
    for subgroup, group in all_df.groupby("subgroup"):
        best_f1 = group.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
        best_oapr = group.sort_values(["oapr_detector_score", "recall", "f1"], ascending=False).iloc[0]
        best_rows.append(
            {
                "protocol": protocol_name,
                "subgroup": subgroup,
                "image_count": int(best_oapr["image_count"]),
                "ground_truth_boxes": int(best_oapr["ground_truth_boxes"]),
                "best_by_f1": best_f1["model"],
                "best_f1": float(best_f1["f1"]),
                "best_by_oapr_detector_score": best_oapr["model"],
                "best_oapr_detector_score": float(best_oapr["oapr_detector_score"]),
                "best_oapr_precision": float(best_oapr["precision"]),
                "best_oapr_recall": float(best_oapr["recall"]),
                "best_oapr_false_positives": int(best_oapr["false_positives"]),
                "best_oapr_false_negatives": int(best_oapr["false_negatives"]),
                "selection_note": "F1 is reported, but OAPR detector choice should be recall-weighted for face-positive privacy-risk subgroups and specificity-weighted for no-face subgroups.",
            }
        )
    best_df = pd.DataFrame(best_rows).sort_values("subgroup")
    best_df.to_csv(output_dir / "best_detector_by_subgroup.csv", index=False)
    (output_dir / "best_detector_by_subgroup.json").write_text(
        json.dumps(best_rows, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# {protocol_name} best detector by subgroup",
        "",
        "| Subgroup | Images | GT boxes | Best by F1 | F1 | Best by OAPR score | OAPR score | Recall | FP | FN |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in best_df.itertuples(index=False):
        lines.append(
            f"| {row.subgroup} | {row.image_count} | {row.ground_truth_boxes} | {row.best_by_f1} | "
            f"{row.best_f1:.4f} | {row.best_by_oapr_detector_score} | {row.best_oapr_detector_score:.4f} | "
            f"{row.best_oapr_recall:.4f} | {row.best_oapr_false_positives} | {row.best_oapr_false_negatives} |"
        )
    (output_dir / "best_detector_by_subgroup.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote subgroup analysis to {output_dir}")


if __name__ == "__main__":
    main()
