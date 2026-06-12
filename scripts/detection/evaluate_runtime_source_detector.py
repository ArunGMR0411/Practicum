#!/usr/bin/env python3
"""Evaluate the live-App 3-source detector bank separately from the 7-source 0.917165 policy.

Uses retained candidate boxes (no re-inference). Writes:
  outputs/02_face_detection/16_runtime_source_validation/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detection.run_detector_error_hardening import (  # noqa: E402
    CANDIDATE_BOXES,
    SOURCE_SETS,
    all_images_score,
    assign_cluster_labels,
    cluster_candidate_boxes,
    cross_validated_variant,
    load_image_features,
    load_protocol_records,
    load_scene_predictions,
    SCENE_PREDICTIONS,
)
from src.detection.error_hardened_rf_policy import (  # noqa: E402
    ADOPTED_FACE_DETECTOR_EXPLORATORY_SCORE,
    ADOPTED_FACE_DETECTOR_POLICY_ID,
    RUNTIME_SOURCE_NAMES,
)

OUT = PROJECT_ROOT / "outputs/02_face_detection/16_runtime_source_validation"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_protocol_records()
    records_by_id = {record.scoped_id: record for record in records}
    candidate_df = pd.read_csv(CANDIDATE_BOXES)
    scene_predictions = load_scene_predictions(SCENE_PREDICTIONS)
    image_features = load_image_features()

    sources = list(SOURCE_SETS.get("runtime_3") or RUNTIME_SOURCE_NAMES)
    cluster_iou = 0.45
    clusters = cluster_candidate_boxes(candidate_df, sources, cluster_iou)
    assign_cluster_labels(clusters, records_by_id)

    rows = []
    for classifier_name in ["rf", "logreg"]:
        method_name = f"error_hardened_runtime_3_{classifier_name}_iou0_45"
        predictions, folds = cross_validated_variant(
            clusters=clusters,
            records=records,
            sources=sources,
            scene_predictions=scene_predictions,
            image_features=image_features,
            classifier_name=classifier_name,
            method_name=method_name,
        )
        score = all_images_score(method_name, predictions, records)
        rows.append(
            {
                "model": method_name,
                "source_bank": "runtime_3",
                "n_sources": len(sources),
                "sources": "|".join(sources),
                "classifier": classifier_name,
                "cluster_iou": cluster_iou,
                "cluster_count": len(clusters),
                "prediction_count": len(predictions),
                **score,
                "scientific_all_raw_rf_score": ADOPTED_FACE_DETECTOR_EXPLORATORY_SCORE,
                "scientific_policy_id": ADOPTED_FACE_DETECTOR_POLICY_ID,
                "delta_vs_scientific_exploratory": float(score["oapr_detector_score"])
                - float(ADOPTED_FACE_DETECTOR_EXPLORATORY_SCORE),
            }
        )
        print(
            f"{method_name}: score={float(score['oapr_detector_score']):.6f} "
            f"P={float(score['precision']):.4f} R={float(score['recall']):.4f}",
            flush=True,
        )
        pd.DataFrame(folds).to_csv(OUT / f"folds_{classifier_name}.csv", index=False)

    table = pd.DataFrame(rows).sort_values("oapr_detector_score", ascending=False)
    table.to_csv(OUT / "01_runtime_3_source_scores.csv", index=False)
    best = table.iloc[0].to_dict()
    summary = {
        "live_app_source_bank": sources,
        "n_sources": len(sources),
        "best_model": best["model"],
        "best_oapr_detector_score": float(best["oapr_detector_score"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_f1": float(best["f1"]),
        "scientific_all_raw_7_source_score": ADOPTED_FACE_DETECTOR_EXPLORATORY_SCORE,
        "scientific_policy_id": ADOPTED_FACE_DETECTOR_POLICY_ID,
        "delta_vs_scientific": float(best["delta_vs_scientific_exploratory"]),
        "note": (
            "This is the honest live-App detector score using only RF-DETR + YOLO11-1280 + SCRFD "
            "candidates. The thesis 0.917165 figure is offline CV on the seven-source all_raw bank."
        ),
    }
    (OUT / "02_runtime_vs_scientific_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Runtime 3-source detector validation (live App bank)",
        "",
        f"- Live sources: `{', '.join(sources)}`",
        f"- Best runtime score: **{summary['best_oapr_detector_score']:.6f}** "
        f"({summary['best_model']})",
        f"- Precision / Recall / F1: "
        f"{summary['best_precision']:.4f} / {summary['best_recall']:.4f} / {summary['best_f1']:.4f}",
        f"- Scientific 7-source score (`{ADOPTED_FACE_DETECTOR_POLICY_ID}`): "
        f"**{ADOPTED_FACE_DETECTOR_EXPLORATORY_SCORE:.6f}**",
        f"- Delta (runtime − scientific): **{summary['delta_vs_scientific']:+.6f}**",
        "",
        summary["note"],
        "",
    ]
    (OUT / "03_runtime_vs_scientific_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
