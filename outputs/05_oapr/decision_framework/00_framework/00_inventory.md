# Decision-framework source inventory

In-body progressive evaluation under `outputs/05_oapr/decision_framework/`.
Re-scores existing stage evidence; does not regenerate 500-frame anonymisation images.

| Stage | Source path | Operation |
|-------|-------------|-----------|
| Face detection | `outputs/02_face_detection/08_sliced_rfdetr_detector_experiment/sliced_detector_policy_scores.csv` | re-score + recall floor |
| Condition | `outputs/02_face_detection/04_scene_condition_router/06_final_model_benchmark.csv` + per-label | re-score + eligibility |
| MM detection | multimodal `02_detection_method_comparison.csv` + `05_combined_risk_detection.csv` | presence + localisation scores |
| Face anon | `09_policy_scoring/anonymisation_policy_per_image_metrics.csv` + visual eligibility + NS SSIM | gates + deployment scores |
| MM redaction | multimodal `06_redaction_per_image_metrics.csv` + residual adaptive | miss→0 privacy + deployment scores |

Hardware: compute-constrained environment. No full 500-frame regeneration required for this package.

Canonical interpretation: `outputs/09_traceability/01_evidence_index.csv` (Final OAPR boundary - progressive evaluation).
Runner: `scripts/oapr_routing/run_decision_framework_evaluation.py`.
