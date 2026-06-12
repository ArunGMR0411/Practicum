#!/usr/bin/env python3
"""Evaluate selective abstaining scene-condition routing from retained evidence.

This is a retained-evidence simulation, not a box-level detector rerun. The
compact repository currently retains subgroup detector metrics, but not all
per-image detector prediction boxes. The script therefore estimates routing
value by applying subgroup detector scores to cross-validated raw-router
predictions and abstaining to the global detector when conditions are weak.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_raw_scene_models import EXCLUDED_LABELS, SUPPORTED_MIN_SUPPORT, handcrafted_features


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path("/tmp/practicum_router_runs/selective_routing")
SUBGROUP_ROOT = ROOT / "outputs" / "02_face_detection" / "05_condition_subgroup_analysis"

PROTOCOL_MAP = {
    "baseline_500": "01_baseline_500",
    "egocentric_stress_500": "02_egocentric_stress_500",
}

FIXED_DETECTORS = ["yolo_face_s", "scrfd", "yolo_face_scrfd_fallback"]


def train_predict_best_raw_router(df: pd.DataFrame, folds: int, random_state: int) -> pd.DataFrame:
    label_cols = [c for c in df.columns if c.startswith("label_")]
    label_support = df[label_cols].sum()
    labels_for_routing = [
        c
        for c in label_cols
        if c not in EXCLUDED_LABELS and label_support[c] >= SUPPORTED_MIN_SUPPORT
    ]
    hand = handcrafted_features(df)
    feature_cols = [c for c in hand.columns if c != "relative_path"]
    overlap = [c for c in feature_cols if c in df.columns]
    merged = df.drop(columns=overlap).merge(hand, on="relative_path", how="left", validate="one_to_one")
    x = merged[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    y = merged[label_cols].astype(int).to_numpy()
    groups = merged["source_group"].fillna("unknown").astype(str).to_numpy()
    model = OneVsRestClassifier(
        Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    )
    pred = np.zeros_like(y)
    gkf = GroupKFold(n_splits=folds)
    for train_idx, test_idx in gkf.split(x, y, groups=groups):
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx])
    rows: list[dict[str, Any]] = []
    for i, row in merged.iterrows():
        rec: dict[str, Any] = {
            "protocol": row["protocol"],
            "relative_path": row["relative_path"],
        }
        for j, label_col in enumerate(label_cols):
            label = label_col.removeprefix("label_")
            rec[f"true_{label}"] = int(y[i, j])
            rec[f"pred_{label}"] = int(pred[i, j])
            rec[f"routing_supported_{label}"] = label_col in labels_for_routing
        rows.append(rec)
    return pd.DataFrame(rows)


def load_scores() -> pd.DataFrame:
    frames = []
    for protocol in ["01_baseline_500", "02_egocentric_stress_500"]:
        df = pd.read_csv(SUBGROUP_ROOT / protocol / "all_models_subgroup_scores.csv")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def best_detector_rows() -> pd.DataFrame:
    return pd.read_csv(SUBGROUP_ROOT / "combined_best_detector_by_subgroup.csv")


def fixed_detector_metric(scores: pd.DataFrame, protocol: str, detector: str) -> dict[str, Any]:
    row = scores[(scores["protocol"] == protocol) & (scores["subgroup"] == "all_images") & (scores["model"] == detector)]
    if row.empty:
        raise RuntimeError(f"missing fixed detector metric: {protocol} {detector}")
    item = row.iloc[0]
    return {
        "selected_subgroup": "all_images",
        "selected_detector": detector,
        "expected_oapr_detector_score": float(item["oapr_detector_score"]),
        "expected_precision": float(item["precision"]),
        "expected_recall": float(item["recall"]),
        "expected_f1": float(item["f1"]),
        "expected_false_positives": float(item["false_positives"]),
        "expected_false_negatives": float(item["false_negatives"]),
    }


def global_best_metric(best: pd.DataFrame, protocol: str) -> dict[str, Any]:
    row = best[(best["protocol"] == protocol) & (best["subgroup"] == "all_images")]
    if row.empty:
        raise RuntimeError(f"missing global best row: {protocol}")
    item = row.iloc[0]
    return {
        "selected_subgroup": "all_images",
        "selected_detector": item["best_by_oapr_detector_score"],
        "expected_oapr_detector_score": float(item["best_oapr_detector_score"]),
        "expected_precision": float(item["best_oapr_precision"]),
        "expected_recall": float(item["best_oapr_recall"]),
        "expected_f1": np.nan,
        "expected_false_positives": float(item["best_oapr_false_positives"]),
        "expected_false_negatives": float(item["best_oapr_false_negatives"]),
    }


def choose_router_metric(
    best: pd.DataFrame,
    protocol: str,
    labels: list[str],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    rows = best[(best["protocol"] == protocol) & (best["subgroup"].isin(labels))].copy()
    if rows.empty:
        return {**fallback, "selected_subgroup": "all_images", "abstained": True}
    rows["advantage"] = rows["best_oapr_detector_score"] - fallback["expected_oapr_detector_score"]
    rows = rows.sort_values(["advantage", "best_oapr_detector_score"], ascending=False)
    item = rows.iloc[0]
    if float(item["advantage"]) <= 0:
        return {**fallback, "selected_subgroup": "all_images", "abstained": True}
    return {
        "selected_subgroup": item["subgroup"],
        "selected_detector": item["best_by_oapr_detector_score"],
        "expected_oapr_detector_score": float(item["best_oapr_detector_score"]),
        "expected_precision": float(item["best_oapr_precision"]),
        "expected_recall": float(item["best_oapr_recall"]),
        "expected_f1": np.nan,
        "expected_false_positives": float(item["best_oapr_false_positives"]),
        "expected_false_negatives": float(item["best_oapr_false_negatives"]),
        "abstained": False,
    }


def supported_labels(row: pd.Series, prefix: str, route_eligible: set[str]) -> list[str]:
    labels = []
    for label in route_eligible:
        if int(row.get(f"{prefix}_{label}", 0)) == 1:
            labels.append(label)
    return sorted(labels)


def build_routing_tables(predictions: pd.DataFrame, scores: pd.DataFrame, best: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_eligible = {"high_clutter", "low_light_or_dim", "motion_blur_or_low_sharpness", "multi_face"}
    details: list[dict[str, Any]] = []
    for _, row in predictions.iterrows():
        protocol = PROTOCOL_MAP[row["protocol"]]
        fallback = global_best_metric(best, protocol)
        true_labels = supported_labels(row, "true", route_eligible)
        pred_labels = supported_labels(row, "pred", route_eligible)
        oracle = choose_router_metric(best, protocol, true_labels, fallback)
        predicted = choose_router_metric(best, protocol, pred_labels, fallback)
        for mode, labels, metric in [
            ("oracle_selective_router", true_labels, oracle),
            ("predicted_selective_router", pred_labels, predicted),
        ]:
            details.append(
                {
                    "protocol": protocol,
                    "relative_path": row["relative_path"],
                    "routing_mode": mode,
                    "labels_used": "|".join(labels) if labels else "none",
                    **metric,
                    "evidence_boundary": "retained_subgroup_metric_simulation_not_box_level_detector_rerun",
                }
            )
        for detector in FIXED_DETECTORS:
            details.append(
                {
                    "protocol": protocol,
                    "relative_path": row["relative_path"],
                    "routing_mode": f"fixed_{detector}",
                    "labels_used": "not_applicable",
                    **fixed_detector_metric(scores, protocol, detector),
                    "abstained": False,
                    "evidence_boundary": "fixed_all_images_detector_metric_applied_to_same_image_rows",
                }
            )
    details_df = pd.DataFrame(details)
    summary = (
        details_df.groupby(["protocol", "routing_mode"], as_index=False)
        .agg(
            image_rows=("relative_path", "count"),
            routed_rows=("abstained", lambda s: int((~s.fillna(False).astype(bool)).sum())),
            abstained_rows=("abstained", lambda s: int(s.fillna(False).astype(bool).sum())),
            mean_oapr_detector_score=("expected_oapr_detector_score", "mean"),
            mean_precision=("expected_precision", "mean"),
            mean_recall=("expected_recall", "mean"),
            mean_f1=("expected_f1", "mean"),
        )
        .sort_values(["protocol", "mean_oapr_detector_score"], ascending=[True, False])
    )
    return details_df, summary


def add_best_fixed_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol, group in summary.groupby("protocol", sort=False):
        fixed = group[group["routing_mode"].str.startswith("fixed_")].copy()
        if fixed.empty:
            raise RuntimeError(f"missing fixed detector rows for {protocol}")
        best_fixed = fixed.sort_values("mean_oapr_detector_score", ascending=False).iloc[0]
        for _, row in group.iterrows():
            rec = row.to_dict()
            rec["best_fixed_routing_mode"] = best_fixed["routing_mode"]
            rec["best_fixed_oapr_detector_score"] = best_fixed["mean_oapr_detector_score"]
            rec["delta_vs_best_fixed_score"] = (
                row["mean_oapr_detector_score"] - best_fixed["mean_oapr_detector_score"]
            )
            rec["delta_vs_best_fixed_recall"] = row["mean_recall"] - best_fixed["mean_recall"]
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(
        ["protocol", "mean_oapr_detector_score"], ascending=[True, False]
    )


def write_report(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Selective Abstaining Router Evaluation With Deltas",
        "",
        "This retained-evidence simulation compares each routing policy against the best fixed detector in the same protocol.",
        "",
        "## Compared Policies",
        "",
        "- fixed YOLO-Face",
        "- fixed SCRFD",
        "- fixed YOLO-Face+SCRFD fallback",
        "- oracle selective router using true reviewed broad labels",
        "- predicted selective router using cross-validated raw-router labels",
        "",
        "## Result",
        "",
        summary.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- Positive deltas are useful evidence, but still simulation-only.",
        "- The compact evidence has subgroup detector metrics, not per-image routed detector boxes.",
        "- Predicted routing can be overestimated if a wrong predicted label maps to a high-scoring subgroup.",
        "- This must be followed by a true box-level routed detector run before thesis promotion.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/02_face_detection/04_scene_condition_router/01_condition_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset)
    predictions = train_predict_best_raw_router(df, args.folds, args.random_state)
    scores = load_scores()
    best = best_detector_rows()
    details, summary = build_routing_tables(predictions, scores, best)
    summary_with_deltas = add_best_fixed_deltas(summary)
    predictions.to_csv(args.output_dir / "selective_router_condition_predictions.csv", index=False)
    details.to_csv(args.output_dir / "selective_router_policy_details.csv", index=False)
    summary_with_deltas.to_csv(
        args.output_dir / "selective_router_policy_comparison_with_deltas.csv", index=False
    )
    write_report(
        summary_with_deltas,
        args.output_dir / "selective_router_policy_comparison_with_deltas.md",
    )
    print(f"wrote {args.output_dir / 'selective_router_policy_comparison_with_deltas.csv'}")
    print(f"wrote {args.output_dir / 'selective_router_policy_comparison_with_deltas.md'}")


if __name__ == "__main__":
    main()
