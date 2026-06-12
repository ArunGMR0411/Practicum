#!/usr/bin/env python3
"""Export a deployable RF filter for error_hardened_all_raw_rf_iou0_45.

Trains on retained multi-detector candidate boxes (offline telemetry).
Uses geometry + source-agreement features only so the App can apply the same
filter at runtime without scene-condition telemetry.

The thesis exploratory score (0.9172) remains the fold-safe CV evaluation from
run_detector_error_hardening.py; this export is the runtime artefact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detection.run_detector_error_hardening import (  # noqa: E402
    CANDIDATE_BOXES,
    assign_cluster_labels,
    classifier_factory,
    cluster_candidate_boxes,
    load_protocol_records,
)
from src.detection.error_hardened_rf_policy import (  # noqa: E402
    ADOPTED_FACE_DETECTOR_EXPLORATORY_SCORE,
    ADOPTED_FACE_DETECTOR_POLICY_ID,
    ALL_RAW_SOURCE_NAMES,
    CLUSTER_IOU,
    DEFAULT_META_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    cluster_feature_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_protocol_records()
    records_by_id = {record.scoped_id: record for record in records}
    candidate_df = pd.read_csv(CANDIDATE_BOXES)
    sources = list(ALL_RAW_SOURCE_NAMES)
    clusters = cluster_candidate_boxes(candidate_df, sources, CLUSTER_IOU)
    assign_cluster_labels(clusters, records_by_id)

    rows: list[list[float]] = []
    labels: list[int] = []
    for cluster in clusters:
        record = records_by_id[cluster["image_id"]]
        width = float(record.attributes.get("image_width") or 3840)
        height = float(record.attributes.get("image_height") or 2160)
        rows.append(
            cluster_feature_row(
                cluster,
                image_width=width,
                image_height=height,
                source_names=sources,
            )
        )
        labels.append(int(cluster["label"]))

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    model = classifier_factory("rf")
    model.fit(X, y)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "threshold": float(args.threshold),
        "source_names": sources,
        "cluster_iou": CLUSTER_IOU,
        "policy_id": ADOPTED_FACE_DETECTOR_POLICY_ID,
        "feature_recipe": "geometry_plus_source_scores",
        "offline_cv_exploratory_score": ADOPTED_FACE_DETECTOR_EXPLORATORY_SCORE,
        "n_clusters": int(len(clusters)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "note": (
            "Runtime deploy model (geometry+source features). "
            "Thesis score 0.9172 is fold-safe CV OOF from the full hardening experiment."
        ),
    }
    joblib.dump(payload, args.model_path)
    meta = {k: v for k, v in payload.items() if k != "model"}
    meta_path = args.model_path.with_suffix(".json") if args.model_path == DEFAULT_MODEL_PATH else DEFAULT_META_PATH
    if args.model_path != DEFAULT_MODEL_PATH:
        meta_path = args.model_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.model_path}")
    print(f"Wrote {meta_path}")
    print(
        f"policy={ADOPTED_FACE_DETECTOR_POLICY_ID} clusters={len(clusters)} "
        f"threshold={args.threshold} offline_cv_score={ADOPTED_FACE_DETECTOR_EXPLORATORY_SCORE:.6f}"
    )


if __name__ == "__main__":
    main()
