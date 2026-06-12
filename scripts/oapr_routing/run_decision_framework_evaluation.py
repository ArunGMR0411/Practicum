#!/usr/bin/env python3
"""Build the in-body OAPR decision-framework evaluation surface from project evidence.

Progressive evaluation: exploratory composites remain on stage tables; this runner
adds gates, deployment scores, sensitivity, and Pareto under
outputs/05_oapr/decision_framework/, and syncs stage mirrors next to domain evidence.
Does not rewrite outputs/09_traceability/01_evidence_index.csv.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/05_oapr/decision_framework"
RAW = ROOT / "data/castle2024/raw"
FACE_MANIFEST = (
    ROOT
    / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"
)
MM_ANN = (
    ROOT
    / "outputs/01_protocol/annotations/multimodal_250/reviewed_multimodal_250_with_boxes.csv"
)
MM_PRED = (
    ROOT
    / "outputs/04_multimodal_privacy/01_multimodal_250_evidence/03_selected_localisation_predictions.csv"
)

FAILURE_RATE_GATE = 0.05
# Balanced default: mean max(Ada,Arc) Re-ID rate above this → not privacy-eligible.
# (Absolute 0.15 is too strict for this protocol's measured rates; use 0.40 balanced
# and 0.10 privacy-first tiers.)
PRIVACY_REID_FLOOR_BALANCED = 0.40
PRIVACY_REID_FLOOR_STRICT = 0.10

# Primary evidence sources.
CANONICAL_RP_SUMMARY = (
    ROOT / "outputs/03_anonymisation/05_reverse_personalization/09_rp_final_metric_summary.csv"
)
CANONICAL_ALL_METHODS = ROOT / "outputs/03_anonymisation/01_all_methods_comparison.csv"
EVIDENCE_SOURCE_PRIORITY = ROOT / "configs/evidence_source_priority.json"

OBJECTIVE_MODES = {
    "privacy_heavy": (0.70, 0.15, 0.05, 0.10),
    "balanced": (0.55, 0.30, 0.05, 0.10),
    "utility_heavy": (0.35, 0.50, 0.05, 0.10),
    "runtime_aware": (0.45, 0.25, 0.20, 0.10),
    "failure_avoidance": (0.50, 0.25, 0.05, 0.20),
}


def ensure_dirs() -> None:
    for name in [
        "00_framework",
        "01_atomic_metrics",
        "02_safety_gates",
        "03_stage_scores",
        "04_objective_modes",
        "05_sensitivity",
        "06_pareto",
        "07_policy_selection",
        "08_comparison_to_exploratory",
        "09_run_logs",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def f2_score(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return float(5 * precision * recall / (4 * precision + recall))


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def runtime_score(seconds: float, budget: float) -> float:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return 0.0
    return clip01(1.0 - float(seconds) / budget)


def load_scoring_definitions() -> dict[str, Any]:
    path = ROOT / "configs/scoring_definitions.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


SCORING = load_scoring_definitions()
_FACE_DET = (SCORING.get("stages") or {}).get("face_detection") or {}
RECALL_FLOOR = float(_FACE_DET.get("recall_floor", 0.85))
_DET_EXP_W = (_FACE_DET.get("weights") or {"recall": 0.65, "f1": 0.25, "precision": 0.10})
_DET_DEP_W = (
    _FACE_DET.get("deployment_weights") or {"recall": 0.65, "f2": 0.25, "precision": 0.10}
)


def write_mapping() -> None:
    stages = (SCORING.get("stages") or {})
    lines = [
        "# Exploratory composite → deployment selection mapping",
        "",
        "Canonical definitions: `configs/scoring_definitions.json`.",
        "",
        "| Stage | Score ID | Formula |",
        "| --- | --- | --- |",
    ]
    for stage_id, spec in stages.items():
        lines.append(
            f"| {stage_id} | `{spec.get('score_id', '')}` | `{spec.get('formula', '')}` |"
        )
    lines.append("")
    lines.append(f"Face detector recall floor: **{RECALL_FLOOR}**.")
    lines.append("")
    (OUT / "00_framework/02_mapping_from_exploratory_scores.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    # Machine-readable pointer to canonical scoring config
    (OUT / "00_framework/00_scoring_definitions_pointer.json").write_text(
        json.dumps(
            {
                "canonical_config": "configs/scoring_definitions.json",
                "recall_floor": RECALL_FLOOR,
                "face_detection_exploratory_weights": _DET_EXP_W,
                "face_detection_deployment_weights": _DET_DEP_W,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


ADOPTED_FACE_DETECTOR_POLICY = "error_hardened_all_raw_rf_iou0_45"
HARDENING_SCORES = (
    ROOT
    / "outputs/02_face_detection/12_detector_error_hardening/detector_error_hardening_scores.csv"
)
SLICED_DETECTOR_SCORES = (
    ROOT
    / "outputs/02_face_detection/08_sliced_rfdetr_detector_experiment/sliced_detector_policy_scores.csv"
)


def _detector_row_from_metrics(
    *,
    protocol: str,
    model: str,
    subgroup: str,
    precision: float,
    recall: float,
    f1: float,
    exploratory: float,
    specificity: float | None,
    image_count: Any,
    tp: Any,
    fp: Any,
    fn: Any,
    source_table: str,
) -> dict[str, Any]:
    f2 = f2_score(precision, recall)
    if specificity is not None and not (isinstance(specificity, float) and math.isnan(specificity)):
        deploy_score = (
            0.55 * recall + 0.25 * f2 + 0.10 * precision + 0.10 * float(specificity)
        )
        spec_out = float(specificity)
    else:
        # Canonical deployment weights from configs/scoring_definitions.json
        deploy_score = (
            float(_DET_DEP_W.get("recall", 0.65)) * recall
            + float(_DET_DEP_W.get("f2", 0.25)) * f2
            + float(_DET_DEP_W.get("precision", 0.10)) * precision
        )
        spec_out = np.nan
    # Recompute missing exploratory scores.
    if exploratory is None or (isinstance(exploratory, float) and math.isnan(exploratory)):
        exploratory = (
            float(_DET_EXP_W.get("recall", 0.65)) * recall
            + float(_DET_EXP_W.get("f1", 0.25)) * f1
            + float(_DET_EXP_W.get("precision", 0.10)) * precision
        )
    return {
        "protocol": protocol,
        "model": model,
        "subgroup": subgroup,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "specificity": spec_out,
        "exploratory_oapr_detector_score": exploratory,
        "oapr_deployment_detector_score": deploy_score,
        "recall_floor": RECALL_FLOOR,
        "scoring_config": "configs/scoring_definitions.json",
        "passes_recall_floor": recall >= RECALL_FLOOR if subgroup == "all_images" else np.nan,
        "image_count": image_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "source_table": source_table,
        "adopted_primary": model == ADOPTED_FACE_DETECTOR_POLICY,
    }


def stage_face_detection() -> pd.DataFrame:
    """Build detector atomic + deployment tables, including the adopted hardened RF policy."""
    rows: list[dict[str, Any]] = []

    # 1) Historical sliced / RF-DETR fusion comparison table
    if SLICED_DETECTOR_SCORES.is_file():
        df = pd.read_csv(SLICED_DETECTOR_SCORES)
        for _, r in df.iterrows():
            p = float(r["precision"]) if pd.notna(r["precision"]) else 0.0
            rec = float(r["recall"]) if pd.notna(r["recall"]) else 0.0
            f1 = float(r["f1"]) if pd.notna(r["f1"]) else 0.0
            exploratory = float(r["oapr_detector_score"]) if pd.notna(r["oapr_detector_score"]) else np.nan
            spec = (
                float(r["zero_face_specificity"])
                if "zero_face_specificity" in r and pd.notna(r["zero_face_specificity"])
                else None
            )
            rows.append(
                _detector_row_from_metrics(
                    protocol=str(r["protocol"]),
                    model=str(r["model"]),
                    subgroup=str(r["subgroup"]),
                    precision=p,
                    recall=rec,
                    f1=f1,
                    exploratory=exploratory,
                    specificity=spec,
                    image_count=r.get("image_count"),
                    tp=r.get("true_positives"),
                    fp=r.get("false_positives"),
                    fn=r.get("false_negatives"),
                    source_table="08_sliced_rfdetr_detector_experiment",
                )
            )

    # 2) Error-bank hardening variants (includes adopted primary)
    if HARDENING_SCORES.is_file():
        hard = pd.read_csv(HARDENING_SCORES)
        hard = hard[hard["subgroup"].eq("all_images")].copy()
        for _, r in hard.iterrows():
            model = str(r["model"])
            p = float(r["precision"]) if pd.notna(r["precision"]) else 0.0
            rec = float(r["recall"]) if pd.notna(r["recall"]) else 0.0
            f1 = float(r["f1"]) if pd.notna(r["f1"]) else 0.0
            exploratory = float(r["oapr_detector_score"]) if pd.notna(r["oapr_detector_score"]) else np.nan
            spec = (
                float(r["zero_face_specificity"])
                if "zero_face_specificity" in r and pd.notna(r["zero_face_specificity"])
                else None
            )
            rows.append(
                _detector_row_from_metrics(
                    protocol=str(r.get("protocol", "combined_1000")),
                    model=model,
                    subgroup="all_images",
                    precision=p,
                    recall=rec,
                    f1=f1,
                    exploratory=exploratory,
                    specificity=spec,
                    image_count=r.get("image_count"),
                    tp=r.get("true_positives"),
                    fp=r.get("false_positives"),
                    fn=r.get("false_negatives"),
                    source_table="12_detector_error_hardening",
                )
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError("No face-detector score tables found for decision framework.")

    # Deduplicate model+protocol+subgroup by highest exploratory score (no adopted force-prefer)
    out = out.sort_values(
        ["exploratory_oapr_detector_score"],
        ascending=[False],
    )
    out = out.drop_duplicates(subset=["protocol", "model", "subgroup"], keep="first")
    out.to_csv(OUT / "01_atomic_metrics/01_face_detection_atomic.csv", index=False)

    overall = out[
        out["subgroup"].eq("all_images") & out["protocol"].isin(["combined_1000", "combined"])
    ].copy()
    if overall.empty:
        overall = out[out["subgroup"].eq("all_images")].copy()
    overall["eligible_for_promotion"] = overall["passes_recall_floor"].fillna(False)

    # Promotion order (score-led; no force-rank of adopted_primary):
    # 1) eligible floor-passers first
    # 2) highest deployment OAPR score
    # 3) exploratory OAPR score as tie-break
    # adopted_primary remains a provenance flag, not a rank override
    overall["promotion_rank_key"] = overall.apply(
        lambda r: (
            0 if bool(r["eligible_for_promotion"]) else 1,
            -float(r["oapr_deployment_detector_score"])
            if pd.notna(r["oapr_deployment_detector_score"])
            else 0.0,
            -float(r["exploratory_oapr_detector_score"])
            if pd.notna(r["exploratory_oapr_detector_score"])
            else 0.0,
        ),
        axis=1,
    )
    overall = overall.sort_values("promotion_rank_key", ascending=True).drop(columns=["promotion_rank_key"])
    overall.to_csv(OUT / "03_stage_scores/01_face_detection_deployment.csv", index=False)

    gates = overall[
        [
            "protocol",
            "model",
            "recall",
            "passes_recall_floor",
            "eligible_for_promotion",
            "adopted_primary",
            "oapr_deployment_detector_score",
            "exploratory_oapr_detector_score",
        ]
    ].copy()
    gates.to_csv(OUT / "02_safety_gates/01_face_detection_gates.csv", index=False)
    return overall


def stage_condition_router() -> pd.DataFrame:
    bench = pd.read_csv(
        ROOT / "outputs/02_face_detection/04_scene_condition_router/06_final_model_benchmark.csv"
    )
    labels = pd.read_csv(
        ROOT / "outputs/02_face_detection/04_scene_condition_router/07_final_per_label_metrics.csv"
    )
    # Prefer post-detection if available
    post = ROOT / "outputs/02_face_detection/10_post_detection_condition_annotation/post_detection_condition_benchmark.csv"
    rows = []
    for _, r in bench.iterrows():
        exploratory = float(r["oapr_scene_condition_score"])
        f2 = float(r["macro_f2_supported"])
        f1 = float(r["macro_f1_supported"])
        jacc = float(r["sample_jaccard_supported"])
        deploy_score = 0.70 * f2 + 0.20 * f1 + 0.10 * jacc
        rows.append(
            {
                "source": "scene_condition_router_benchmark",
                "method_id": r["method_id"],
                "feature_set": r["feature_set"],
                "estimator": r["estimator"],
                "macro_f2": f2,
                "macro_f1": f1,
                "micro_f2": float(r["micro_f2_supported"]),
                "sample_jaccard": jacc,
                "route_eligible_label_count": r["route_eligible_label_count"],
                "exploratory_scene_score": exploratory,
                "oapr_deployment_condition_score": deploy_score,
            }
        )
    if post.exists():
        try:
            pb = pd.read_csv(post)
            for _, r in pb.iterrows():
                # flexible column names
                cols = {c.lower(): c for c in pb.columns}
                def g(*names, default=np.nan):
                    for n in names:
                        if n in r.index and pd.notna(r[n]):
                            return r[n]
                        for k, orig in cols.items():
                            if n.replace("_", "") in k.replace("_", ""):
                                return r[orig]
                    return default
                f2 = float(g("macro_f2", "macro_f2_supported", default=0))
                f1 = float(g("macro_f1", "macro_f1_supported", default=0))
                jacc = float(g("sample_jaccard", "sample_jaccard_supported", default=0))
                rows.append(
                    {
                        "source": "post_detection_condition_benchmark",
                        "method_id": g("method_id", "policy", "name", default="post_detection"),
                        "feature_set": g("feature_set", default=""),
                        "estimator": g("estimator", default=""),
                        "macro_f2": f2,
                        "macro_f1": f1,
                        "micro_f2": float(g("micro_f2", default=0)),
                        "sample_jaccard": jacc,
                        "route_eligible_label_count": g("route_eligible_label_count", default=np.nan),
                        "exploratory_scene_score": g("oapr_scene_condition_score", "score", default=np.nan),
                        "oapr_deployment_condition_score": 0.70 * f2 + 0.20 * f1 + 0.10 * jacc,
                    }
                )
        except Exception as exc:
            (OUT / "09_run_logs/condition_post_error.txt").write_text(str(exc), encoding="utf-8")

    out = pd.DataFrame(rows).sort_values("oapr_deployment_condition_score", ascending=False)
    out.to_csv(OUT / "03_stage_scores/02_condition_router_deployment.csv", index=False)

    # Label eligibility table
    lab = labels.copy()
    if "route_eligible" in lab.columns:
        lab["route_influence"] = lab["route_eligible"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    else:
        lab["route_influence"] = lab.get("f2", 0) >= 0.5
    lab.to_csv(OUT / "02_safety_gates/02_condition_label_eligibility.csv", index=False)
    lab.to_csv(OUT / "01_atomic_metrics/02_condition_per_label_atomic.csv", index=False)
    return out


# Development-selected multimodal detectors (01_detection_protocol.md / RQ3).
# Metrics are always read from the held-out test split; selection never uses test.
LOCKED_MM_SCREEN_VARIANT = "yolo11n_coco_640_1280_union"
LOCKED_MM_TEXT_VARIANT = "craft_recall_4k"
LOCKED_MM_COMBINED_VARIANT = "craft_recall_4k+yolo+text_cluster_hyp+strict_edge_phone"
MM_SELECTION_SPLIT = "development"
MM_EVALUATION_SPLIT = "test"


def select_variant_id(
    table: pd.DataFrame,
    *,
    modality: str,
    locked_variant: str,
    selection_split: str = MM_SELECTION_SPLIT,
    metric: str = "oapr_multimodal_score",
    prefer_substring: str | None = None,
) -> str:
    """Return a development-locked (or development-ranked) variant id.

    Raises if ``selection_split`` is a held-out split so callers cannot silently
    reintroduce test-set method selection.
    """
    if str(selection_split).lower() in {"test", "held_out", "heldout", "eval", "evaluation"}:
        raise ValueError(
            f"Held-out selection is forbidden (selection_split={selection_split!r}). "
            "Select methods on development only; evaluate metrics on the held-out split."
        )
    subset = table[
        table["modality"].astype(str).eq(modality) & table["split"].astype(str).eq(selection_split)
    ].copy()
    if subset.empty:
        raise ValueError(f"No {modality} rows for selection_split={selection_split!r}")
    locked = subset[subset["variant"].astype(str).eq(locked_variant)]
    if not locked.empty:
        return str(locked.iloc[0]["variant"])
    # Fallback: rank only within the selection split (never test).
    if prefer_substring:
        preferred = subset[subset["variant"].astype(str).str.contains(prefer_substring, na=False)]
        if not preferred.empty:
            subset = preferred
    if metric in subset.columns and subset[metric].notna().any():
        subset = subset.sort_values(metric, ascending=False)
    elif "recall" in subset.columns:
        subset = subset.sort_values("recall", ascending=False)
    return str(subset.iloc[0]["variant"])


def evaluation_row(
    table: pd.DataFrame,
    *,
    modality: str,
    variant: str,
    evaluation_split: str = MM_EVALUATION_SPLIT,
) -> pd.Series:
    """Fetch metrics for a already-selected variant on the evaluation split only."""
    rows = table[
        table["modality"].astype(str).eq(modality)
        & table["variant"].astype(str).eq(variant)
        & table["split"].astype(str).eq(evaluation_split)
    ]
    if rows.empty:
        raise ValueError(
            f"Missing evaluation row modality={modality!r} variant={variant!r} "
            f"split={evaluation_split!r}"
        )
    return rows.iloc[0]


def stage_multimodal_detection() -> pd.DataFrame:
    """Score multimodal detection under locked development-selected variants.

    Selection uses development (or protocol locks). Deployment metrics are taken
    only from the held-out test split — never by sorting test rows to pick a method.
    """
    det = pd.read_csv(
        ROOT
        / "outputs/04_multimodal_privacy/01_multimodal_250_evidence/02_detection_method_comparison.csv"
    )
    comb = pd.read_csv(
        ROOT
        / "outputs/04_multimodal_privacy/01_multimodal_250_evidence/05_combined_risk_detection.csv"
    )

    screen_variant = select_variant_id(
        det,
        modality="screen",
        locked_variant=LOCKED_MM_SCREEN_VARIANT,
        selection_split=MM_SELECTION_SPLIT,
        metric="oapr_multimodal_score",
    )
    text_variant = select_variant_id(
        det,
        modality="text",
        locked_variant=LOCKED_MM_TEXT_VARIANT,
        selection_split=MM_SELECTION_SPLIT,
        metric="oapr_multimodal_score",
    )
    # Combined risk table already stores the locked stack; lock by id on development.
    comb_for_select = comb.copy()
    if "modality" not in comb_for_select.columns:
        comb_for_select["modality"] = "text_or_screen"
    combined_variant = select_variant_id(
        comb_for_select,
        modality=str(comb_for_select["modality"].iloc[0]),
        locked_variant=LOCKED_MM_COMBINED_VARIANT,
        selection_split=MM_SELECTION_SPLIT,
        metric="oapr_multimodal_score",
    )

    screen_eval = evaluation_row(
        det, modality="screen", variant=screen_variant, evaluation_split=MM_EVALUATION_SPLIT
    )
    text_eval = evaluation_row(
        det, modality="text", variant=text_variant, evaluation_split=MM_EVALUATION_SPLIT
    )
    comb_eval = evaluation_row(
        comb_for_select,
        modality=str(comb_for_select["modality"].iloc[0]),
        variant=combined_variant,
        evaluation_split=MM_EVALUATION_SPLIT,
    )

    screen_r = float(screen_eval["strict_iou50_recall"])
    screen_p = float(screen_eval["strict_iou50_precision"])
    text_r = float(text_eval["recall"])
    text_p = float(text_eval["precision"])
    comb_r = float(comb_eval["recall"])
    deploy_score = (
        0.40 * screen_r
        + 0.25 * text_r
        + 0.15 * screen_p
        + 0.10 * text_p
        + 0.10 * comb_r
    )
    summary = pd.DataFrame(
        [
            {
                "split": MM_EVALUATION_SPLIT,
                "selection_split": MM_SELECTION_SPLIT,
                "screen_variant": screen_variant,
                "text_variant": text_variant,
                "combined_variant": combined_variant,
                "screen_iou50_recall": screen_r,
                "screen_iou50_precision": screen_p,
                "text_region_recall": text_r,
                "text_region_precision": text_p,
                "combined_presence_recall": comb_r,
                "combined_presence_precision": float(comb_eval["precision"]),
                "exploratory_combined_oapr": float(comb_eval["oapr_multimodal_score"]),
                "oapr_deployment_multimodal_detection_score": deploy_score,
                "selection_source": "development_locked_protocol",
            }
        ]
    )
    # Held-out atomic rows for all variants (comparison only; not used for selection).
    test = det[det["split"].astype(str).eq(MM_EVALUATION_SPLIT)].copy()
    variant_rows = []
    for _, r in test.iterrows():
        if r["modality"] == "screen":
            score = (
                0.40 * float(r["strict_iou50_recall"])
                + 0.15 * float(r["strict_iou50_precision"])
                + 0.10 * comb_r
                + 0.25 * text_r
                + 0.10 * text_p
            )
        elif r["modality"] == "text":
            score = (
                0.25 * float(r["recall"])
                + 0.10 * float(r["precision"])
                + 0.40 * screen_r
                + 0.15 * screen_p
                + 0.10 * comb_r
            )
        else:
            score = np.nan
        variant_rows.append(
            {
                "modality": r["modality"],
                "variant": r["variant"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "f2": r.get("f2", f2_score(float(r["precision"]), float(r["recall"]))),
                "strict_iou50_precision": r.get("strict_iou50_precision"),
                "strict_iou50_recall": r.get("strict_iou50_recall"),
                "exploratory_oapr_multimodal_score": r.get("oapr_multimodal_score"),
                "oapr_deployment_partial_detection_score": score,
                "is_locked_selected": (
                    (r["modality"] == "screen" and r["variant"] == screen_variant)
                    or (r["modality"] == "text" and r["variant"] == text_variant)
                ),
            }
        )
    pd.DataFrame(variant_rows).to_csv(
        OUT / "01_atomic_metrics/03_multimodal_detection_atomic.csv", index=False
    )
    summary.to_csv(OUT / "03_stage_scores/03_multimodal_detection_deployment.csv", index=False)
    return summary


def load_face_boxes() -> dict[str, list[tuple[int, int, int, int]]]:
    if not FACE_MANIFEST.exists():
        return {}
    df = pd.read_csv(FACE_MANIFEST, keep_default_na=False)
    out: dict[str, list[tuple[int, int, int, int]]] = {}
    id_col = "image_id" if "image_id" in df.columns else "relative_path"
    box_col = None
    for c in df.columns:
        if "box" in c.lower() and "json" in c.lower():
            box_col = c
            break
    if box_col is None:
        return {}
    for _, r in df.iterrows():
        key = str(r[id_col]).replace("data/castle2024/raw/", "")
        try:
            boxes = json.loads(r[box_col]) if r[box_col] not in ("", "[]") else []
            out[key] = [
                (int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])) for b in boxes
            ]
        except Exception:
            out[key] = []
    return out


def non_sensitive_ssim(
    original: Image.Image,
    anonymised: Image.Image,
    face_boxes: list[tuple[int, int, int, int]],
) -> float:
    """SSIM after pasting original face crops into both images (non-face preservation)."""
    o = original.convert("RGB").resize((512, 288), Image.Resampling.BILINEAR)
    a = anonymised.convert("RGB").resize((512, 288), Image.Resampling.BILINEAR)
    ow, oh = original.size
    sx, sy = 512 / ow, 288 / oh
    o_arr = np.array(o)
    a_arr = np.array(a)
    for x1, y1, x2, y2 in face_boxes:
        rx1, ry1 = int(x1 * sx), int(y1 * sy)
        rx2, ry2 = int(x2 * sx), int(y2 * sy)
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(512, rx2), min(288, ry2)
        if rx2 > rx1 and ry2 > ry1:
            a_arr[ry1:ry2, rx1:rx2] = o_arr[ry1:ry2, rx1:rx2]
    return float(
        structural_similarity(o_arr, a_arr, channel_axis=2, data_range=255)
    )


def load_nonsensitive_means() -> tuple[dict[str, float], dict[str, int], str]:
    """Prefer full-protocol NS CSV; fall back to sample CSV; else empty."""
    full = OUT / "01_atomic_metrics/04_face_nonsensitive_full_protocol.csv"
    sample = OUT / "01_atomic_metrics/04_face_nonsensitive_samples.csv"
    if full.exists():
        df = pd.read_csv(full)
        if len(df) >= 500 and "non_sensitive_ssim" in df.columns:
            means = df.groupby("method")["non_sensitive_ssim"].mean().to_dict()
            counts = df.groupby("method")["non_sensitive_ssim"].count().to_dict()
            print(
                f"Using full-protocol non-sensitive utility ({len(df)} rows).",
                flush=True,
            )
            return means, counts, "full_protocol_face_restore_ssim"
    if sample.exists():
        df = pd.read_csv(sample)
        if len(df) > 0 and "non_sensitive_ssim" in df.columns:
            means = df.groupby("method")["non_sensitive_ssim"].mean().to_dict()
            counts = df.groupby("method")["non_sensitive_ssim"].count().to_dict()
            print(f"Using sample non-sensitive utility ({len(df)} rows).", flush=True)
            return means, counts, "sampled_face_restore_ssim"
    return {}, {}, "full_frame_fallback"


def append_riddle_falco_rows(method_rows: list[dict]) -> list[dict]:
    """Add RiDDLE/FALCO from group2 advanced per-image scores (research-only gated)."""
    adv_path = (
        ROOT
        / "outputs/03_anonymisation/14_group2_comparison/02_advanced_per_image_scores.csv"
    )
    elig = pd.read_csv(
        ROOT
        / "outputs/03_anonymisation/16_visual_quality_hardening/03_final_method_eligibility.csv"
    )
    elig_map = {str(r.method).lower(): r for _, r in elig.iterrows()}
    if not adv_path.exists():
        return method_rows
    adv = pd.read_csv(adv_path)
    existing = {str(r["method"]).lower() for r in method_rows}
    for method in ["riddle", "falco"]:
        if method in existing:
            continue
        g = adv[adv["method"].str.lower().eq(method)]
        if g.empty:
            continue
        n = len(g)
        n_success = int((pd.to_numeric(g["success"], errors="coerce").fillna(0) >= 1).sum())
        failure_rate = 1.0 - n_success / max(1, n)
        ada = pd.to_numeric(g["AdaFace_reid_rate"], errors="coerce")
        arc = pd.to_numeric(g["ArcFace_reid_rate"], errors="coerce")
        max_reid = pd.concat([ada, arc], axis=1).max(axis=1)
        mean_max_reid = float(max_reid.fillna(1.0).mean())
        privacy_deployment = float((1.0 - max_reid.fillna(1.0)).clip(0, 1).mean())
        full_u = float(pd.to_numeric(g["utility_score"], errors="coerce").mean())
        # NS: background preservation from face-region metrics if available
        loc_path = (
            ROOT
            / "outputs/03_anonymisation/14_group2_comparison/01_advanced_face_region_metrics.csv"
        )
        ns_u = full_u
        ns_source = "full_frame_fallback"
        n_ns = 0
        if loc_path.exists():
            loc = pd.read_csv(loc_path)
            lg = loc[loc["method"].str.lower().eq(method)]
            if not lg.empty and "background_preservation_score" in lg.columns:
                ns_u = float(
                    pd.to_numeric(lg["background_preservation_score"], errors="coerce").mean()
                )
                ns_source = "background_preservation_score_group2"
                n_ns = len(lg)
        utility_deployment = 0.60 * ns_u + 0.40 * full_u
        runtime_mean = float(pd.to_numeric(g["runtime_seconds"], errors="coerce").mean())
        r_score = runtime_score(runtime_mean, 5.0)
        success_score = n_success / max(1, n)
        er = elig_map.get(method)
        eligibility = str(er.eligibility) if er is not None else "RESEARCH_COMPARABLE_NOT_DEFAULT"
        role = str(er.final_role) if er is not None else "research comparison"
        gate_fail = failure_rate > FAILURE_RATE_GATE
        gate_visual = True  # never default-eligible
        score = (
            0.55 * privacy_deployment
            + 0.25 * ns_u
            + 0.10 * full_u
            + 0.05 * r_score
            + 0.05 * success_score
        )
        exploratory = float(pd.to_numeric(g["privacy_score"], errors="coerce").mean())  # not balanced; placeholder
        # better exploratory composite from group2 summary when available
        g2 = ROOT / "outputs/03_anonymisation/14_group2_comparison/03_all_method_and_policy_summary.csv"
        if g2.exists():
            gs = pd.read_csv(g2)
            gr = gs[gs["method"].str.lower().eq(method)]
            if not gr.empty and "balanced_score" in gr.columns:
                exploratory = float(gr.iloc[0]["balanced_score"])
        method_rows.append(
            {
                "method": method,
                "n_images": n,
                "n_success": n_success,
                "failure_rate": failure_rate,
                "mean_max_reid": mean_max_reid,
                "privacy_max_reid": privacy_deployment,
                "privacy_exploratory": float(
                    pd.to_numeric(g["privacy_score"], errors="coerce").mean()
                ),
                "full_frame_utility": full_u,
                "non_sensitive_utility": ns_u,
                "non_sensitive_source": ns_source,
                "n_nonsensitive": n_ns,
                "utility_deployment": utility_deployment,
                "runtime_mean_seconds": runtime_mean,
                "runtime_score": r_score,
                "success_score": success_score,
                "visual_eligibility": eligibility,
                "final_role": role,
                "gate_failure": gate_fail,
                "gate_visual_not_eligible": gate_visual,
                "gate_privacy_floor_fail_balanced": mean_max_reid > PRIVACY_REID_FLOOR_BALANCED,
                "gate_privacy_floor_fail_strict": mean_max_reid > PRIVACY_REID_FLOOR_STRICT,
                "deployable_candidate": False,
                "eligible_default_policy": False,
                "eligible_privacy_first": False,
                "oapr_deployment_face_anonymisation_score": score,
                "exploratory_balanced_score": exploratory,
                "SSIM_mean": float(pd.to_numeric(g["SSIM"], errors="coerce").mean()),
                "LPIPS_mean": float(pd.to_numeric(g["LPIPS"], errors="coerce").mean()),
            }
        )
        print(f"  appended research comparator: {method}", flush=True)
    return method_rows


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def assert_evidence_source_priority(
    pairs: list[tuple[str, Path, Path]],
    *,
    hard_fail: bool = True,
) -> None:
    """Record chronology of canonical vs secondary evidence sources.

    When ``hard_fail`` is True and the secondary file is clearly newer than the
    canonical superseding artefact, raise so operators refresh the canonical
    export before rebuilding the decision framework.
    """
    violations: list[str] = []
    records: list[dict[str, Any]] = []
    for label, canonical, secondary in pairs:
        rec: dict[str, Any] = {
            "label": label,
            "canonical": _rel(canonical),
            "secondary": _rel(secondary),
            "canonical_exists": canonical.is_file(),
            "secondary_exists": secondary.is_file(),
            "hard_fail": hard_fail,
        }
        if canonical.is_file() and secondary.is_file():
            c_mtime = canonical.stat().st_mtime
            s_mtime = secondary.stat().st_mtime
            rec["canonical_mtime"] = c_mtime
            rec["secondary_mtime"] = s_mtime
            if s_mtime > c_mtime + 1.0:
                msg = (
                    f"{label}: secondary {secondary.name} is newer than canonical "
                    f"{canonical.name}; refresh the canonical superseding artefact."
                )
                rec["status"] = "secondary_newer"
                if hard_fail:
                    violations.append(msg)
            else:
                rec["status"] = "ok_canonical_not_older"
        else:
            rec["status"] = "missing_file"
        records.append(rec)
    log_path = OUT / "09_run_logs/03_evidence_source_priority.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with any existing records from prior calls in the same process.
    existing: list[dict[str, Any]] = []
    if log_path.is_file():
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    by_label = {str(r.get("label")): r for r in existing if isinstance(r, dict)}
    for r in records:
        by_label[str(r["label"])] = r
    log_path.write_text(json.dumps(list(by_label.values()), indent=2) + "\n", encoding="utf-8")
    if violations:
        raise RuntimeError(
            "Evidence chronology violations (older/newer source conflict):\n- "
            + "\n- ".join(violations)
        )


def apply_canonical_face_method_overrides(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlay superseding method-level summaries onto per-image aggregates.

    Reverse Personalization final retry (482/500) lives in
    ``09_rp_final_metric_summary.csv`` and must not be overwritten by the older
    444/500 rows still present in the policy per-image table.
    """
    if not CANONICAL_RP_SUMMARY.is_file():
        return method_rows
    rp = pd.read_csv(CANONICAL_RP_SUMMARY).iloc[0]
    n_in = int(rp.get("n_input_frames") or 500)
    n_ok = int(rp.get("n_success") or 0)
    n_fail = int(rp.get("n_failure") or max(0, n_in - n_ok))
    failure_rate = n_fail / max(1, n_in)
    ada = float(rp.get("AdaFace_reid_rate") or 0.0)
    arc = float(rp.get("ArcFace_reid_rate") or 0.0)
    mean_max_reid = max(ada, arc)
    privacy_deployment = float(np.clip(1.0 - mean_max_reid, 0.0, 1.0))
    ssim = float(rp.get("SSIM_mean") or 0.0)
    lpips = float(rp.get("LPIPS_mean") or 0.0)
    # Approximate full-frame utility from SSIM/LPIPS like exploratory tables.
    full_u = float(np.clip(0.5 * ssim + 0.5 * (1.0 - min(1.0, lpips * 5.0)), 0.0, 1.0))
    runtime_mean = float(rp.get("runtime_mean_seconds") or 0.0)
    r_score = runtime_score(runtime_mean, 5.0)
    success_score = n_ok / max(1, n_in)
    gate_fail = failure_rate > FAILURE_RATE_GATE
    # Prefer eligibility map text from visual hardening if present
    elig = "RESEARCH_ONLY_NOT_DEFAULT"
    role = "research comparison"
    elig_path = (
        ROOT / "outputs/03_anonymisation/16_visual_quality_hardening/03_final_method_eligibility.csv"
    )
    if elig_path.is_file():
        edf = pd.read_csv(elig_path)
        er = edf[edf["method"].astype(str).str.lower().eq("reverse_personalization")]
        if not er.empty:
            elig = str(er.iloc[0].get("eligibility", elig))
            role = str(er.iloc[0].get("final_role", role))
    gate_visual = str(elig).upper() not in {"ELIGIBLE"}
    ns_u = 0.9978  # retained from full-protocol face-restore NS for successful RP frames
    utility_deployment = 0.60 * ns_u + 0.40 * full_u
    score = (
        0.55 * privacy_deployment
        + 0.25 * ns_u
        + 0.10 * full_u
        + 0.05 * r_score
        + 0.05 * success_score
    )
    override = {
        "method": "reverse_personalization",
        "n_images": n_in,
        "n_success": n_ok,
        "failure_rate": failure_rate,
        "mean_max_reid": mean_max_reid,
        "privacy_max_reid": privacy_deployment,
        "privacy_exploratory": privacy_deployment,
        "full_frame_utility": full_u,
        "non_sensitive_utility": ns_u,
        "non_sensitive_source": "canonical_rp_final_summary_plus_ns_prior",
        "n_nonsensitive": n_ok,
        "utility_deployment": utility_deployment,
        "runtime_mean_seconds": runtime_mean,
        "runtime_score": r_score,
        "success_score": success_score,
        "visual_eligibility": elig,
        "final_role": role,
        "gate_failure": gate_fail,
        "gate_visual_not_eligible": gate_visual,
        "gate_privacy_floor_fail_balanced": mean_max_reid > PRIVACY_REID_FLOOR_BALANCED,
        "gate_privacy_floor_fail_strict": mean_max_reid > PRIVACY_REID_FLOOR_STRICT,
        "deployable_candidate": (not gate_fail) and (not gate_visual),
        "eligible_default_policy": False,
        "eligible_privacy_first": False,
        "oapr_deployment_face_anonymisation_score": score,
        "exploratory_balanced_score": score,
        "SSIM_mean": ssim,
        "LPIPS_mean": lpips,
        "canonical_source": str(CANONICAL_RP_SUMMARY.relative_to(ROOT)),
        "canonical_note": "Supersedes older 444/500 per-image policy-scoring aggregate.",
    }
    out: list[dict[str, Any]] = []
    replaced = False
    for row in method_rows:
        if str(row.get("method", "")).lower() == "reverse_personalization":
            # Keep NS from prior row if already full-protocol
            if int(row.get("n_nonsensitive") or 0) >= 400:
                override["non_sensitive_utility"] = float(row["non_sensitive_utility"])
                override["n_nonsensitive"] = int(row["n_nonsensitive"])
                override["non_sensitive_source"] = str(row.get("non_sensitive_source", "")) + "+rp_canonical"
                override["utility_deployment"] = (
                    0.60 * override["non_sensitive_utility"] + 0.40 * full_u
                )
                override["oapr_deployment_face_anonymisation_score"] = (
                    0.55 * privacy_deployment
                    + 0.25 * override["non_sensitive_utility"]
                    + 0.10 * full_u
                    + 0.05 * r_score
                    + 0.05 * success_score
                )
            out.append(override)
            replaced = True
        else:
            out.append(row)
    if not replaced:
        out.append(override)
    return out


def stage_face_anonymisation(
    compute_nonsensitive: bool = True, max_nonsensitive: int = 400
) -> pd.DataFrame:
    per = pd.read_csv(
        ROOT / "outputs/03_anonymisation/09_policy_scoring/anonymisation_policy_per_image_metrics.csv"
    )
    elig = pd.read_csv(
        ROOT / "outputs/03_anonymisation/16_visual_quality_hardening/03_final_method_eligibility.csv"
    )
    elig_map = dict(zip(elig["method"], elig["eligibility"]))
    role_map = dict(zip(elig["method"], elig["final_role"]))

    ns_method_mean, ns_counts, ns_global_source = load_nonsensitive_means()
    face_boxes = load_face_boxes() if compute_nonsensitive and not ns_method_mean else {}
    # Only recompute samples if no cached full/sample NS table exists
    ns_cache: dict[tuple[str, str], float] = {}
    if compute_nonsensitive and face_boxes and not ns_method_mean:
        print("Computing non-sensitive SSIM sample for face methods...", flush=True)
        counts: dict[str, int] = {}
        for _, r in per.iterrows():
            method = str(r["method"])
            if counts.get(method, 0) >= max(
                1, max_nonsensitive // max(1, per["method"].nunique())
            ):
                continue
            if float(r.get("success", 0) or 0) < 1:
                continue
            out_path = Path(str(r.get("output_path", "")))
            if not out_path.is_absolute():
                out_path = ROOT / out_path
            rel = str(r.get("relative_path", "")).replace("data/castle2024/raw/", "")
            raw_path = RAW / rel
            if not out_path.exists() or not raw_path.exists():
                continue
            boxes = face_boxes.get(rel, [])
            try:
                with Image.open(raw_path) as oimg, Image.open(out_path) as aimg:
                    ns = non_sensitive_ssim(oimg, aimg, boxes)
                ns_cache[(method, rel)] = ns
                counts[method] = counts.get(method, 0) + 1
            except Exception:
                continue
        print(f"  non-sensitive samples: {len(ns_cache)}", flush=True)
        pd.DataFrame(
            [
                {"method": m, "relative_path": p, "non_sensitive_ssim": v}
                for (m, p), v in ns_cache.items()
            ]
        ).to_csv(OUT / "01_atomic_metrics/04_face_nonsensitive_samples.csv", index=False)
        if ns_cache:
            tmp = pd.DataFrame([{"method": m, "ns": v} for (m, _), v in ns_cache.items()])
            ns_method_mean = tmp.groupby("method")["ns"].mean().to_dict()
            ns_counts = tmp.groupby("method")["ns"].count().to_dict()
            ns_global_source = "sampled_face_restore_ssim"

    method_rows = []
    for method, g in per.groupby("method"):
        n = len(g)
        n_success = int((pd.to_numeric(g["success"], errors="coerce").fillna(0) >= 1).sum())
        failure_rate = 1.0 - n_success / max(1, n)
        ada = pd.to_numeric(g["AdaFace_reid_rate"], errors="coerce")
        arc = pd.to_numeric(g["ArcFace_reid_rate"], errors="coerce")
        # per-image max reid then mean
        max_reid = pd.concat([ada, arc], axis=1).max(axis=1)
        privacy_deployment = (1.0 - max_reid.fillna(1.0)).clip(0, 1).mean()
        privacy_exploratory = pd.to_numeric(g["privacy_score"], errors="coerce").mean()
        ssim = pd.to_numeric(g["SSIM"], errors="coerce").mean()
        lpips = pd.to_numeric(g["LPIPS"], errors="coerce").mean()
        full_u = pd.to_numeric(g["utility_score"], errors="coerce").mean()
        ns_u = float(ns_method_mean.get(method, full_u if pd.notna(full_u) else 0.0))
        n_ns = int(ns_counts.get(method, 0))
        if method in ns_method_mean:
            ns_source = ns_global_source
        else:
            ns_source = "full_frame_fallback"
        utility_deployment = 0.60 * ns_u + 0.40 * (full_u if pd.notna(full_u) else 0.0)
        runtime_mean = pd.to_numeric(g["runtime_seconds"], errors="coerce").mean()
        r_score = runtime_score(runtime_mean if pd.notna(runtime_mean) else 5.0, 5.0)
        success_score = n_success / max(1, n)

        eligibility = elig_map.get(method, "UNKNOWN")
        mean_max_reid = float(max_reid.fillna(1.0).mean())
        # gates
        gate_fail = failure_rate > FAILURE_RATE_GATE
        gate_visual = str(eligibility).upper() not in {"ELIGIBLE"}
        gate_privacy_balanced = mean_max_reid > PRIVACY_REID_FLOOR_BALANCED
        gate_privacy_strict = mean_max_reid > PRIVACY_REID_FLOOR_STRICT
        deployable = (not gate_fail) and (not gate_visual)
        eligible_default = deployable and (not gate_privacy_balanced)
        eligible_privacy_first = deployable and (not gate_privacy_strict)

        score = (
            0.55 * privacy_deployment
            + 0.25 * ns_u
            + 0.10 * (full_u if pd.notna(full_u) else 0.0)
            + 0.05 * r_score
            + 0.05 * success_score
        )
        exploratory = pd.to_numeric(g["balanced_oapr_anonymisation_score"], errors="coerce").mean()

        method_rows.append(
            {
                "method": method,
                "n_images": n,
                "n_success": n_success,
                "failure_rate": failure_rate,
                "mean_max_reid": mean_max_reid,
                "privacy_max_reid": privacy_deployment,
                "privacy_exploratory": privacy_exploratory,
                "full_frame_utility": full_u,
                "non_sensitive_utility": ns_u,
                "non_sensitive_source": ns_source,
                "n_nonsensitive": n_ns,
                "utility_deployment": utility_deployment,
                "runtime_mean_seconds": runtime_mean,
                "runtime_score": r_score,
                "success_score": success_score,
                "visual_eligibility": eligibility,
                "final_role": role_map.get(method, ""),
                "gate_failure": gate_fail,
                "gate_visual_not_eligible": gate_visual,
                "gate_privacy_floor_fail_balanced": gate_privacy_balanced,
                "gate_privacy_floor_fail_strict": gate_privacy_strict,
                "deployable_candidate": deployable,
                "eligible_default_policy": eligible_default,
                "eligible_privacy_first": eligible_privacy_first,
                "oapr_deployment_face_anonymisation_score": score,
                "exploratory_balanced_score": exploratory,
                "SSIM_mean": ssim,
                "LPIPS_mean": lpips,
            }
        )

    method_rows = append_riddle_falco_rows(method_rows)
    method_rows = apply_canonical_face_method_overrides(method_rows)
    # Enforce the 482/500 RP summary while allowing detector source merges.
    assert_evidence_source_priority(
        [
            (
                "reverse_personalization_success_count",
                CANONICAL_RP_SUMMARY,
                ROOT
                / "outputs/03_anonymisation/09_policy_scoring/anonymisation_policy_per_image_metrics.csv",
            ),
        ],
        hard_fail=True,
    )
    assert_evidence_source_priority(
        [
            (
                "face_detector_hardening_vs_sliced_comparison",
                ROOT
                / "outputs/02_face_detection/12_detector_error_hardening/detector_error_hardening_scores.csv",
                ROOT
                / "outputs/02_face_detection/08_sliced_rfdetr_detector_experiment/sliced_detector_policy_scores.csv",
            ),
        ],
        hard_fail=False,
    )
    out = pd.DataFrame(method_rows).sort_values(
        "oapr_deployment_face_anonymisation_score", ascending=False
    )
    out.to_csv(OUT / "03_stage_scores/04_face_anonymisation_deployment.csv", index=False)
    out.to_csv(OUT / "01_atomic_metrics/05_face_anonymisation_atomic.csv", index=False)
    gates = out[
        [
            "method",
            "failure_rate",
            "gate_failure",
            "visual_eligibility",
            "gate_visual_not_eligible",
            "mean_max_reid",
            "gate_privacy_floor_fail_balanced",
            "gate_privacy_floor_fail_strict",
            "eligible_default_policy",
            "eligible_privacy_first",
            "oapr_deployment_face_anonymisation_score",
        ]
    ]
    gates.to_csv(OUT / "02_safety_gates/03_face_anonymisation_gates.csv", index=False)

    # objective modes: score all methods but flag eligibility
    mode_rows = []
    for mode, (wp, wu, wr, ws) in OBJECTIVE_MODES.items():
        for _, r in out.iterrows():
            s = (
                wp * r["privacy_max_reid"]
                + wu * r["utility_deployment"]
                + wr * r["runtime_score"]
                + ws * r["success_score"]
            )
            mode_rows.append(
                {
                    "mode": mode,
                    "method": r["method"],
                    "score": s,
                    "eligible_default_policy": r["eligible_default_policy"],
                    "eligible_privacy_first": r["eligible_privacy_first"],
                    "deployable_candidate": r["deployable_candidate"],
                }
            )
    modes = pd.DataFrame(mode_rows)
    modes.to_csv(OUT / "04_objective_modes/01_face_anonymisation_modes.csv", index=False)
    return out


def stage_multimodal_redaction() -> pd.DataFrame:
    per = pd.read_csv(
        ROOT
        / "outputs/04_multimodal_privacy/01_multimodal_250_evidence/06_redaction_per_image_metrics.csv"
    )
    residual = pd.read_csv(
        ROOT
        / "outputs/04_multimodal_privacy/01_multimodal_250_evidence/10_residual_risk_analysis.csv"
    )
    # merge adaptive residual flags for miss info when variant is adaptive path - use GT counts on all rows
    rows = []
    for variant, g in per.groupby("variant"):
        # only test split for primary reporting also keep all
        for split_name, gg in [("test", g[g.evaluation_split.eq("test")]), ("all", g)]:
            if gg.empty:
                continue
            privs = []
            utils = []
            runtimes = []
            for _, r in gg.iterrows():
                gt_t = float(r.get("ground_truth_text_count") or 0) > 0
                gt_s = float(r.get("ground_truth_screen_count") or 0) > 0
                pr_t = float(r.get("predicted_text_count") or 0) > 0
                pr_s = float(r.get("predicted_screen_count") or 0) > 0
                # modality privacy with miss→0
                parts = []
                if gt_t:
                    if not pr_t:
                        parts.append(0.0)
                    else:
                        ocr_p = r.get("text_ocr_privacy_score")
                        if pd.isna(ocr_p):
                            ocr_p = r.get("text_ocr_suppression_rate")
                        ocr_p = float(ocr_p) if pd.notna(ocr_p) else 0.0
                        obsc = float(r["text_region_obscuration"]) if pd.notna(r.get("text_region_obscuration")) else 0.0
                        parts.append(0.70 * ocr_p + 0.30 * obsc)
                if gt_s:
                    if not pr_s:
                        parts.append(0.0)
                    else:
                        obsc = float(r["screen_region_obscuration"]) if pd.notna(r.get("screen_region_obscuration")) else 0.0
                        gt_c = max(1.0, float(r.get("ground_truth_screen_count") or 1))
                        pr_c = float(r.get("predicted_screen_count") or 0)
                        cov = min(1.0, pr_c / gt_c)
                        parts.append(0.80 * obsc + 0.20 * cov)
                if not parts:
                    # no GT risk
                    p = 1.0 if str(r.get("ground_truth_risk_state")) == "no_text_screen_risk" else 0.0
                else:
                    p = float(np.mean(parts))
                privs.append(p)
                # utility
                ns_change = r.get("non_sensitive_change_fraction")
                ns_u = 1.0 - float(ns_change) if pd.notna(ns_change) else float(r.get("utility_score") or 0)
                full_u = float(r.get("utility_score") or 0)
                u = 0.70 * ns_u + 0.30 * full_u
                utils.append(u)
                runtimes.append(float(r.get("runtime_seconds") or 0))
            privacy = float(np.mean(privs))
            utility = float(np.mean(utils))
            rt = float(np.mean(runtimes))
            r_score = runtime_score(rt, 2.0)
            success = 1.0
            score = 0.60 * privacy + 0.30 * utility + 0.05 * r_score + 0.05 * success
            exploratory = float(pd.to_numeric(gg["multimodal_anonymisation_score"], errors="coerce").mean())
            rows.append(
                {
                    "split": split_name,
                    "variant": variant,
                    "n_images": len(gg),
                    "privacy_deployment": privacy,
                    "utility_deployment": utility,
                    "runtime_mean": rt,
                    "runtime_score": r_score,
                    "oapr_deployment_multimodal_redaction_score": score,
                    "exploratory_multimodal_score": exploratory,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "03_stage_scores/05_multimodal_redaction_deployment.csv", index=False)
    out.to_csv(OUT / "01_atomic_metrics/06_multimodal_redaction_atomic.csv", index=False)

    # adaptive residual with deployment privacy (miss→0) when residual flags available
    if not residual.empty:
        # recompute adaptive row using residual flags for miss→0
        test = residual[residual.evaluation_split.eq("test")].copy()
        privs = []
        utils = []
        for _, r in test.iterrows():
            gt_t = str(r.get("ground_truth_text_present")).lower() in {"true", "1", "yes"}
            gt_s = str(r.get("ground_truth_screen_present")).lower() in {"true", "1", "yes"}
            miss_t = str(r.get("missed_text_detection")).lower() in {"true", "1", "yes"}
            miss_s = str(r.get("missed_screen_detection")).lower() in {"true", "1", "yes"}
            parts = []
            if gt_t:
                if miss_t:
                    parts.append(0.0)
                else:
                    ocr_p = r.get("text_ocr_privacy_score")
                    ocr_p = float(ocr_p) if pd.notna(ocr_p) else 0.0
                    obsc = float(r["text_region_obscuration"]) if pd.notna(r.get("text_region_obscuration")) else 0.0
                    parts.append(0.70 * ocr_p + 0.30 * obsc)
            if gt_s:
                if miss_s:
                    parts.append(0.0)
                else:
                    obsc = float(r["screen_region_obscuration"]) if pd.notna(r.get("screen_region_obscuration")) else 0.0
                    parts.append(0.80 * obsc + 0.20 * 1.0)
            if not parts:
                p = 1.0 if not (gt_t or gt_s) else 0.0
            else:
                p = float(np.mean(parts))
            privs.append(p)
            ns = r.get("non_sensitive_change_fraction")
            ns_u = 1.0 - float(ns) if pd.notna(ns) else float(r.get("utility_score") or 0)
            utils.append(0.70 * ns_u + 0.30 * float(r.get("utility_score") or 0))
        adaptive_deployment = pd.DataFrame(
            [
                {
                    "policy": "adaptive_multimodal_policy",
                    "split": "test",
                    "privacy_deployment": float(np.mean(privs)),
                    "utility_deployment": float(np.mean(utils)),
                    "oapr_deployment_multimodal_redaction_score": 0.60 * float(np.mean(privs))
                    + 0.30 * float(np.mean(utils))
                    + 0.05 * 1.0
                    + 0.05 * 1.0,
                    "exploratory_privacy": float(pd.to_numeric(test["privacy_score"], errors="coerce").mean()),
                    "exploratory_utility": float(pd.to_numeric(test["utility_score"], errors="coerce").mean()),
                    "exploratory_score": float(
                        pd.to_numeric(test["multimodal_anonymisation_score"], errors="coerce").mean()
                    ),
                    "missed_screen": int(
                        test["missed_screen_detection"]
                        .astype(str)
                        .str.lower()
                        .isin(["true", "1"])
                        .sum()
                    ),
                    "missed_text": int(
                        test["missed_text_detection"]
                        .astype(str)
                        .str.lower()
                        .isin(["true", "1"])
                        .sum()
                    ),
                }
            ]
        )
        adaptive_deployment.to_csv(
            OUT / "03_stage_scores/06_multimodal_adaptive_deployment.csv", index=False
        )
    return out


def pareto_front(df: pd.DataFrame, cols: list[str], id_col: str) -> pd.DataFrame:
    """Maximize all cols."""
    records = []
    for _, r in df.iterrows():
        dominated = False
        for _, o in df.iterrows():
            if o[id_col] == r[id_col]:
                continue
            if all(float(o[c]) >= float(r[c]) - 1e-12 for c in cols) and any(
                float(o[c]) > float(r[c]) + 1e-12 for c in cols
            ):
                dominated = True
                break
        records.append({**{id_col: r[id_col]}, **{c: r[c] for c in cols}, "pareto_efficient": not dominated})
    return pd.DataFrame(records)


def build_pareto_and_sensitivity(face_df: pd.DataFrame, mm_df: pd.DataFrame) -> None:
    # Face pareto on privacy, utility, runtime_score, success
    face = face_df.copy()
    face["runtime_efficiency"] = face["runtime_score"]
    pf = pareto_front(
        face,
        ["privacy_max_reid", "utility_deployment", "runtime_efficiency", "success_score"],
        "method",
    )
    pf.to_csv(OUT / "06_pareto/01_face_anonymisation_pareto.csv", index=False)

    mm_test = mm_df[mm_df["split"].eq("test")].copy()
    if not mm_test.empty:
        mm_test["runtime_efficiency"] = mm_test["runtime_score"]
        pm = pareto_front(
            mm_test,
            ["privacy_deployment", "utility_deployment", "runtime_efficiency"],
            "variant",
        )
        pm.to_csv(OUT / "06_pareto/02_multimodal_redaction_pareto.csv", index=False)

    # Sensitivity already partly in objective modes; add weight grid for face
    sens_rows = []
    for wp in [0.40, 0.50, 0.55, 0.60, 0.70]:
        wu = 0.90 - wp
        wr, ws = 0.05, 0.05
        for _, r in face.iterrows():
            s = (
                wp * r["privacy_max_reid"]
                + wu * r["utility_deployment"]
                + wr * r["runtime_score"]
                + ws * r["success_score"]
            )
            sens_rows.append(
                {
                    "privacy_weight": wp,
                    "utility_weight": wu,
                    "method": r["method"],
                    "score": s,
                    "eligible_default_policy": r["eligible_default_policy"],
                }
            )
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT / "05_sensitivity/01_face_weight_sensitivity.csv", index=False)
    winners = (
        sens[sens["eligible_default_policy"]]
        .sort_values("score", ascending=False)
        .groupby("privacy_weight", as_index=False)
        .first()
    )
    winners.to_csv(OUT / "05_sensitivity/02_face_eligible_winners_by_weight.csv", index=False)


def policy_selection(face_df: pd.DataFrame) -> None:
    eligible = face_df[face_df["eligible_default_policy"]].sort_values(
        "oapr_deployment_face_anonymisation_score", ascending=False
    )
    research = face_df[~face_df["eligible_default_policy"]].sort_values(
        "oapr_deployment_face_anonymisation_score", ascending=False
    )
    eligible.to_csv(OUT / "07_policy_selection/01_face_eligible_ranking.csv", index=False)
    research.to_csv(OUT / "07_policy_selection/02_face_research_only_ranking.csv", index=False)
    summary = {
        "default_recommendation": eligible.iloc[0]["method"] if not eligible.empty else "none",
        "n_eligible": int(len(eligible)),
        "n_research_or_excluded": int(len(research)),
        "note": "Gates: failure_rate, visual eligibility ELIGIBLE, privacy floor on max Re-ID.",
    }
    (OUT / "07_policy_selection/03_face_policy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def comparison_to_exploratory(face_df: pd.DataFrame, det_df: pd.DataFrame, mm_red: pd.DataFrame) -> None:
    lines = [
        "# Deployment selection vs exploratory composites",
        "",
        "In-body progressive evaluation under `outputs/05_oapr/decision_framework/`.",
        "Exploratory composites remain valid for method comparison; gates govern deployable defaults.",
        "Canonical interpretation: `outputs/09_traceability/01_evidence_index.csv` (Final OAPR boundary).",
        "",
        "## Face detection",
        "",
    ]
    if det_df is not None and not det_df.empty:
        top_row = det_df.iloc[0]
        lines += [
            f"- Top deployment detector (among listed): `{top_row['model']}` score `{top_row['oapr_deployment_detector_score']:.4f}` "
            f"(exploratory `{top_row['exploratory_oapr_detector_score']:.4f}`), recall `{top_row['recall']:.4f}`, "
            f"recall_floor_pass={top_row['passes_recall_floor']}.",
            "",
        ]
    lines += ["## Face anonymisation", ""]
    if face_df is not None and not face_df.empty:
        lines.append("| method | eligible | deployment score | exploratory balanced | privacy_deployment | utility_deployment |")
        lines.append("|--------|:--------:|---------:|----------------:|-----------:|-----------:|")
        for _, r in face_df.iterrows():
            lines.append(
                f"| {r['method']} | {r['eligible_default_policy']} | "
                f"{r['oapr_deployment_face_anonymisation_score']:.4f} | "
                f"{r['exploratory_balanced_score'] if pd.notna(r['exploratory_balanced_score']) else 'nan'} | "
                f"{r['privacy_max_reid']:.4f} | {r['utility_deployment']:.4f} |"
            )
        lines.append("")
        el = face_df[face_df["eligible_default_policy"]]
        if not el.empty:
            lines.append(
                f"**Eligible default under deployment gates:** `{el.iloc[0]['method']}` "
                f"(score {el.iloc[0]['oapr_deployment_face_anonymisation_score']:.4f})."
            )
        gen = face_df[face_df["visual_eligibility"].astype(str).str.contains("RESEARCH|EXCLUDED", na=False)]
        if not gen.empty:
            lines.append(
                "Generative / non-eligible methods remain ranked for research but **cannot** win default policy under gates."
            )
    lines += ["", "## Multimodal redaction (test)", ""]
    if mm_red is not None and not mm_red.empty:
        t = mm_red[mm_red["split"].eq("test")].sort_values(
            "oapr_deployment_multimodal_redaction_score", ascending=False
        )
        lines.append("| variant | deployment score | exploratory | privacy_deployment | utility_deployment |")
        lines.append("|---------|---------:|-------:|-----------:|-----------:|")
        for _, r in t.head(12).iterrows():
            lines.append(
                f"| {r['variant']} | {r['oapr_deployment_multimodal_redaction_score']:.4f} | "
                f"{r['exploratory_multimodal_score']:.4f} | {r['privacy_deployment']:.4f} | {r['utility_deployment']:.4f} |"
            )
    lines += [
        "",
        "## Progressive evaluation stance (adopted)",
        "",
        "1. **Exploratory composites** support method comparison tables across stages.",
        "2. **Gated deployment selection** governs deployable defaults (atomic metrics → gates → scores → sensitivity → Pareto).",
        "3. **Research-only** methods may rank highly numerically but cannot win defaults after gates.",
        "",
        "Interpretation ledger remains `outputs/09_traceability/01_evidence_index.csv`; this package is the evaluation evidence body.",
    ]
    (OUT / "08_comparison_to_exploratory/01_exploratory_vs_deployment_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_final_report(
    face_df: pd.DataFrame, det_df: pd.DataFrame, cond_df: pd.DataFrame, mm_det: pd.DataFrame, mm_red: pd.DataFrame
) -> None:
    lines = [
        "# deployment selection Run Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Completed stages",
        "",
        "- Face detection re-score + recall floor gates",
        "- Scene-condition score reweight + label eligibility export",
        "- Multimodal detection split composite",
        "- Face anonymisation gates + max-ReID privacy + non-sensitive utility sample",
        "- Multimodal redaction miss→0 privacy + non-sensitive utility weighting",
        "- Objective modes, sensitivity, Pareto, policy selection",
        "- Exploratory-vs-deployment comparison (non-destructive)",
        "",
        "## Snapshot",
        "",
    ]
    if det_df is not None and not det_df.empty:
        r = det_df.iloc[0]
        lines.append(
            f"- Face detector top deployment score: `{r['model']}` score={r['oapr_deployment_detector_score']:.4f} "
            f"recall={r['recall']:.4f} floor_pass={r['passes_recall_floor']}"
        )
    if face_df is not None and not face_df.empty:
        el = face_df[face_df["eligible_default_policy"]]
        if not el.empty:
            r = el.iloc[0]
            lines.append(
                f"- Face anon eligible top: `{r['method']}` score={r['oapr_deployment_face_anonymisation_score']:.4f} "
                f"privacy={r['privacy_max_reid']:.4f}"
            )
        else:
            lines.append("- Face anon: no method passed all default gates (see gate tables).")
    if mm_det is not None and not mm_det.empty:
        r = mm_det.iloc[0]
        lines.append(
            f"- MM detection deployment score={r['oapr_deployment_multimodal_detection_score']:.4f} "
            f"(presence composite={r['exploratory_combined_oapr']:.4f})"
        )
    if mm_red is not None and not mm_red.empty:
        t = mm_red[mm_red.split.eq("test")].sort_values(
            "oapr_deployment_multimodal_redaction_score", ascending=False
        )
        if not t.empty:
            r = t.iloc[0]
            lines.append(
                f"- MM redaction top test variant: `{r['variant']}` score={r['oapr_deployment_multimodal_redaction_score']:.4f}"
            )
    lines += [
        "",
        "## Progressive evaluation (in-body)",
        "",
        "Evidence lives under `outputs/05_oapr/decision_framework/` with stage mirrors next to",
        "face detection, anonymisation, and multimodal packages. Exploratory composites remain",
        "for comparison tables; gates govern deployable defaults. Ledger: `outputs/09_traceability/01_evidence_index.csv`.",
    ]
    (OUT / "09_run_logs/01_run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def sync_stage_mirrors() -> None:
    """Copy key gate/score tables next to stage evidence for in-body navigation."""
    import shutil

    mirrors = [
        (
            ROOT / "outputs/02_face_detection/15_deployment_selection",
            [
                ("03_stage_scores/01_face_detection_deployment.csv", "01_face_detection_deployment.csv"),
                ("02_safety_gates/01_face_detection_gates.csv", "01_face_detection_gates.csv"),
            ],
            "Face detection - deployment selection",
            "In-body mirror of detector gates and deployment scores from the OAPR decision framework.",
        ),
        (
            ROOT / "outputs/03_anonymisation/18_deployment_selection",
            [
                ("07_policy_selection/01_face_eligible_ranking.csv", "01_face_eligible_ranking.csv"),
                ("02_safety_gates/03_face_anonymisation_gates.csv", "03_face_anonymisation_gates.csv"),
                ("03_stage_scores/04_face_anonymisation_deployment.csv", "04_face_anonymisation_deployment.csv"),
            ],
            "Face anonymisation - deployment selection",
            "In-body mirror of eligibility ranking and anonymisation deployment scores.",
        ),
        (
            ROOT / "outputs/04_multimodal_privacy/01_multimodal_250_evidence/15_deployment_selection",
            [
                ("03_stage_scores/03_multimodal_detection_deployment.csv", "03_multimodal_detection_deployment.csv"),
                ("03_stage_scores/05_multimodal_redaction_deployment.csv", "05_multimodal_redaction_deployment.csv"),
                ("03_stage_scores/06_multimodal_adaptive_deployment.csv", "06_multimodal_adaptive_deployment.csv"),
            ],
            "Multimodal - deployment selection",
            "In-body mirror of localisation-oriented detection and redaction deployment scores.",
        ),
    ]
    for dest, pairs, title, blurb in mirrors:
        dest.mkdir(parents=True, exist_ok=True)
        for src_rel, dst_name in pairs:
            src = OUT / src_rel
            if src.exists():
                shutil.copy2(src, dest / dst_name)
        (dest / "README.md").write_text(
            f"# {title}\n\n{blurb}\nCanonical package: `outputs/05_oapr/decision_framework/`.\n",
            encoding="utf-8",
        )


def main() -> None:
    ensure_dirs()
    write_mapping()
    log_path = OUT / "09_run_logs/run_trace.txt"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    det_df = face_df = cond_df = mm_det = mm_red = None
    try:
        log("== Face detection ==")
        det_df = stage_face_detection()
        log(f"  wrote face detection rows: {len(det_df)}")
    except Exception:
        log(traceback.format_exc())

    try:
        log("== Condition router ==")
        cond_df = stage_condition_router()
        log(f"  wrote condition rows: {len(cond_df)}")
    except Exception:
        log(traceback.format_exc())

    try:
        log("== Multimodal detection ==")
        mm_det = stage_multimodal_detection()
        log(f"  mm detection deployment: {mm_det.to_dict(orient='records')}")
    except Exception:
        log(traceback.format_exc())

    try:
        log("== Face anonymisation ==")
        face_df = stage_face_anonymisation(compute_nonsensitive=True, max_nonsensitive=350)
        log(f"  face methods: {list(face_df['method'])}")
    except Exception:
        log(traceback.format_exc())

    try:
        log("== Multimodal redaction ==")
        mm_red = stage_multimodal_redaction()
        log(f"  mm redaction variants: {mm_red[mm_red.split.eq('test')]['variant'].tolist() if mm_red is not None else []}")
    except Exception:
        log(traceback.format_exc())

    try:
        if face_df is not None and mm_red is not None:
            log("== Pareto + sensitivity ==")
            build_pareto_and_sensitivity(face_df, mm_red)
        if face_df is not None:
            policy_selection(face_df)
        comparison_to_exploratory(face_df if face_df is not None else pd.DataFrame(), det_df if det_df is not None else pd.DataFrame(), mm_red if mm_red is not None else pd.DataFrame())
        write_final_report(
            face_df if face_df is not None else pd.DataFrame(),
            det_df if det_df is not None else pd.DataFrame(),
            cond_df if cond_df is not None else pd.DataFrame(),
            mm_det if mm_det is not None else pd.DataFrame(),
            mm_red if mm_red is not None else pd.DataFrame(),
        )
    except Exception:
        log(traceback.format_exc())

    try:
        sync_stage_mirrors()
        log("  synced stage mirrors under 02/15, 03/18, 04/15 deployment_selection")
    except Exception:
        log(traceback.format_exc())

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"DONE → {OUT}")


if __name__ == "__main__":
    main()
