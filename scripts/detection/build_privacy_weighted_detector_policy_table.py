#!/usr/bin/env python3

"""Build the final privacy-weighted face-detector policy evidence table."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs/02_face_detection/09_privacy_weighted_detector_policy"

DETECTOR_SCORE_SOURCES = [
    PROJECT_ROOT / "outputs/02_face_detection/06_detector_hardening_experiment/detector_hardening_subgroup_scores.csv",
    PROJECT_ROOT / "outputs/02_face_detection/08_sliced_rfdetr_detector_experiment/sliced_detector_policy_scores.csv",
]
SCR_METRICS = PROJECT_ROOT / "outputs/02_face_detection/04_scene_condition_router/07_final_per_label_metrics.csv"
SCR_METHOD = "handcrafted_yolo_multiscale__logistic_regression"
DEFAULT_POLICY = "cv_box_reranker_with_rfdetr_predicted_conditions"
MEANINGFUL_MARGIN = 0.005


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def as_int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_row(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    return max(rows, key=lambda row: as_float(row.get(metric)))


def ranked_rows(rows: list[dict[str, str]], metric: str) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: as_float(row.get(metric)), reverse=True)


def load_detector_scores() -> tuple[list[dict[str, str]], dict[tuple[str, str], str]]:
    rows: list[dict[str, str]] = []
    source_by_key: dict[tuple[str, str], str] = {}
    for source in DETECTOR_SCORE_SOURCES:
        for row in read_csv(source):
            if row["protocol"] != "combined_1000":
                continue
            key = (row["model"], row["subgroup"])
            # Keep the richer/latest source if a model/subgroup appears twice.
            source_by_key[key] = str(source.relative_to(PROJECT_ROOT))
            rows = [existing for existing in rows if (existing["model"], existing["subgroup"]) != key]
            rows.append(row)
    return rows, source_by_key


def load_scr_metrics() -> dict[str, dict[str, str]]:
    rows = read_csv(SCR_METRICS)
    return {
        row["label"]: row
        for row in rows
        if row.get("method_id") == SCR_METHOD
    }


def decision_for_category(
    subgroup: str,
    route_eligible: bool,
    best_oapr: dict[str, str],
    default_row: dict[str, str] | None,
) -> tuple[str, str, str, float | str]:
    if subgroup == "all_images":
        return (
            "global_default",
            DEFAULT_POLICY,
            "Use the privacy-weighted RF-DETR-aware box reranker as the single default detector policy.",
            "",
        )

    if not route_eligible:
        return (
            "fallback_to_privacy_weighted_reranker",
            DEFAULT_POLICY,
            "SCR does not reliably predict this manual category, so runtime routing must use the fallback detector policy.",
            "",
        )

    if default_row is None:
        return (
            "route_to_category_winner",
            best_oapr["model"],
            "No default-policy row exists for this subgroup; use the category OAPR winner if the category is confidently predicted.",
            "",
        )

    margin = as_float(best_oapr["oapr_detector_score"]) - as_float(default_row["oapr_detector_score"])
    if best_oapr["model"] == DEFAULT_POLICY:
        return (
            "route_to_privacy_weighted_reranker",
            DEFAULT_POLICY,
            "The global privacy-weighted reranker is also the category OAPR winner.",
            round(margin, 6),
        )
    if margin >= MEANINGFUL_MARGIN:
        return (
            "route_to_category_winner",
            best_oapr["model"],
            "SCR can predict this category and the category OAPR winner has a meaningful margin over the default reranker.",
            round(margin, 6),
        )
    return (
        "fallback_to_privacy_weighted_reranker",
        DEFAULT_POLICY,
        "The category is predictable, but the category winner does not beat the default reranker by a meaningful margin.",
        round(margin, 6),
    )


def build_policy_rows() -> list[dict[str, Any]]:
    scores, source_by_key = load_detector_scores()
    scr = load_scr_metrics()
    subgroups = sorted({row["subgroup"] for row in scores})
    ordered = ["all_images"] + [name for name in subgroups if name != "all_images"]
    rows: list[dict[str, Any]] = []

    for subgroup in ordered:
        subgroup_rows = [row for row in scores if row["subgroup"] == subgroup]
        if not subgroup_rows:
            continue
        f1_ranked = ranked_rows(subgroup_rows, "f1")
        recall_ranked = ranked_rows(subgroup_rows, "recall")
        oapr_ranked = ranked_rows(subgroup_rows, "oapr_detector_score")
        by_f1 = f1_ranked[0]
        by_recall = recall_ranked[0]
        by_oapr = oapr_ranked[0]
        second_oapr = oapr_ranked[1] if len(oapr_ranked) > 1 else None
        default_row = next((row for row in subgroup_rows if row["model"] == DEFAULT_POLICY), None)
        scr_row = scr.get(subgroup, {})
        route_eligible = str(scr_row.get("route_eligible", "False")).lower() == "true"
        support = as_int(by_oapr.get("image_count"))
        gt_boxes = as_int(by_oapr.get("ground_truth_boxes"))
        scr_support = as_int(scr_row.get("support")) if scr_row else ""
        scr_predictability = (
            "route_eligible"
            if route_eligible
            else ("not_applicable_global" if subgroup == "all_images" else "fallback_required")
        )
        action, final_policy, reason, margin = decision_for_category(subgroup, route_eligible, by_oapr, default_row)

        rows.append(
            {
                "manual_category": subgroup,
                "support_images": support,
                "ground_truth_boxes": gt_boxes,
                "best_detector_by_f1": "not_applicable_no_face" if subgroup == "no_face" else by_f1["model"],
                "best_f1": "" if subgroup == "no_face" else round(as_float(by_f1["f1"]), 6),
                "best_detector_by_recall": "not_applicable_no_face" if subgroup == "no_face" else by_recall["model"],
                "best_recall": "" if subgroup == "no_face" else round(as_float(by_recall["recall"]), 6),
                "best_detector_by_oapr_score": by_oapr["model"],
                "best_oapr_detector_score": round(as_float(by_oapr["oapr_detector_score"]), 6),
                "second_best_detector_by_oapr_score": second_oapr["model"] if second_oapr else "",
                "second_best_oapr_detector_score": (
                    round(as_float(second_oapr["oapr_detector_score"]), 6) if second_oapr else ""
                ),
                "best_oapr_margin_over_second": (
                    round(
                        as_float(by_oapr["oapr_detector_score"]) - as_float(second_oapr["oapr_detector_score"]),
                        6,
                    )
                    if second_oapr
                    else ""
                ),
                "best_oapr_precision": round(as_float(by_oapr["precision"]), 6),
                "best_oapr_recall": round(as_float(by_oapr["recall"]), 6),
                "best_oapr_f1": round(as_float(by_oapr["f1"]), 6),
                "best_oapr_tp": as_int(by_oapr.get("true_positives")),
                "best_oapr_fp": as_int(by_oapr.get("false_positives")),
                "best_oapr_fn": as_int(by_oapr.get("false_negatives")),
                "scr_predictability_status": scr_predictability,
                "scr_support": scr_support,
                "scr_precision": round(as_float(scr_row.get("precision")), 6) if scr_row else "",
                "scr_recall": round(as_float(scr_row.get("recall")), 6) if scr_row else "",
                "scr_f1": round(as_float(scr_row.get("f1")), 6) if scr_row else "",
                "scr_f2": round(as_float(scr_row.get("f2")), 6) if scr_row else "",
                "scr_predicted_category_or_fallback": subgroup if route_eligible else "fallback_uncertain_or_unsupported",
                "runtime_action": action,
                "final_detector_policy_decision": final_policy,
                "margin_vs_default_policy": margin,
                "decision_reason": reason,
                "default_policy_oapr_score_for_category": (
                    round(as_float(default_row["oapr_detector_score"]), 6) if default_row else ""
                ),
                "default_policy_recall_for_category": round(as_float(default_row["recall"]), 6) if default_row else "",
                "score_source": source_by_key.get((by_oapr["model"], subgroup), ""),
                "scr_source": str(SCR_METRICS.relative_to(PROJECT_ROOT)) if scr_row else "",
            }
        )
    return rows


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Privacy-Weighted Face Detector Policy",
        "",
        "This table connects manual condition evidence to deployable detector-policy decisions.",
        "",
        "Detector objective:",
        "",
        "`OAPR detector score = 0.65 * recall + 0.25 * F1 + 0.10 * precision` for face-positive categories.",
        "",
        "For no-face categories, the score is specificity because false positives damage utility.",
        "",
        "Policy rule:",
        "",
        "- Use all manual categories for thesis/oracle analysis.",
        "- Use only SCR route-eligible categories for runtime category routing.",
        "- Use `cv_box_reranker_with_rfdetr_predicted_conditions` as the privacy-weighted fallback/default policy.",
        "- Do not route runtime images by weak categories unless a later SCR model proves them reliable.",
        "",
        "| Manual category | Support | Best by F1 | Best by recall | Best by OAPR score | OAPR score | OAPR margin vs 2nd | SCR can predict | Runtime action | Final policy |",
        "|---|---:|---|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        margin = row["best_oapr_margin_over_second"]
        margin_text = "" if margin == "" else f"{float(margin):.4f}"
        f1_text = (
            "not applicable"
            if row["best_f1"] == ""
            else f"{row['best_detector_by_f1']} ({float(row['best_f1']):.4f})"
        )
        recall_text = (
            "not applicable"
            if row["best_recall"] == ""
            else f"{row['best_detector_by_recall']} ({float(row['best_recall']):.4f})"
        )
        lines.append(
            f"| {row['manual_category']} | {row['support_images']} | "
            f"{f1_text} | {recall_text} | "
            f"{row['best_detector_by_oapr_score']} | "
            f"{float(row['best_oapr_detector_score']):.4f} | {margin_text} | "
            f"{row['scr_predictability_status']} | {row['runtime_action']} | {row['final_detector_policy_decision']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    route_rows = [row for row in rows if row["scr_predictability_status"] == "route_eligible"]
    fallback_rows = [row for row in rows if row["scr_predictability_status"] == "fallback_required"]
    default = next(row for row in rows if row["manual_category"] == "all_images")
    lines = [
        "# Face Detector Policy Summary",
        "",
        "Final thesis detector-policy position:",
        "",
        "- Privacy requires high recall because a missed face is a privacy failure.",
        "- Uncontrolled false positives damage utility and downstream anonymisation quality.",
        "- The detector stage therefore uses a privacy-weighted detector score rather than pure recall.",
        f"- The final single default detector policy is `{DEFAULT_POLICY}`.",
        f"- Combined 1,000-image default score: `{float(default['best_oapr_detector_score']):.4f}`; table rows provide category-level overrides and fallback decisions.",
        "",
        "Route eligibility:",
        "",
        f"- Route-eligible manual categories from SCR evidence: `{len(route_rows)}`.",
        f"- Manual categories requiring fallback because SCR is not reliable enough: `{len(fallback_rows)}`.",
        "",
        "Evidence boundary:",
        "",
        "- Manual categories are used for scientific analysis and oracle interpretation.",
        "- Runtime routing uses only SCR-reliable categories.",
        "- Unsupported or uncertain categories use the privacy-weighted RF-DETR-aware reranker fallback.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_policy_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "privacy_weighted_detector_policy_table.csv", rows)
    write_markdown(OUTPUT_DIR / "privacy_weighted_detector_policy_table.md", rows)
    write_summary(OUTPUT_DIR / "privacy_weighted_detector_policy_summary.md", rows)
    print(f"Wrote {len(rows)} policy rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
