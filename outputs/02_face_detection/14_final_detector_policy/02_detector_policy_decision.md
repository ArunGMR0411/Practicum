# Final Detector Policy Decision

**Date:** 2026-07-14

## Options Considered (on combined 1,000-image reviewed protocol)

- Current default: `cv_box_reranker_with_rfdetr_predicted_conditions` - OAPR 0.912891
- Best hardened: `error_hardened_all_raw_rf_iou0_45` - OAPR **0.917165** (highest)
- Strong recall alternative: `error_hardened_rfdetr_clean_logreg_iou0_45` - OAPR 0.915408
- Low-compute baseline: `cv_box_reranker_predicted_conditions` - OAPR 0.9110

## Decision

**Adopt `error_hardened_all_raw_rf_iou0_45` as the primary detector policy for the balanced/standard OAPR objective.**

**Rationale:**
- Highest measured OAPR detector score under the privacy-weighted metric.
- Better precision (0.9319) while maintaining strong recall (0.9129).
- The previous default reranker is retained as the "recall-first" option for scenarios where missing faces must be minimized at all costs.

**Runtime routing recommendation:**
- Use the hardened RF policy as default for most conditions.
- Fall back to high-recall fusion or the original reranker only for specific high-miss-risk categories if needed.
- Keep SCRFD for confident no-face cases.

This improvement strengthens the upstream detector quality feeding into the condition profiler and OAPR, directly strengthening RQ1 and the overall adaptive pipeline.

**Impact on downstream:**
- The integrated end-to-end policy numbers may improve slightly with the better detector (re-evaluation recommended in future if full re-run is done).
- No change to the visual-safe anonymisation policy itself.

**Propagation status:**
- Decision-framework detector **ranking is score-led** (no force-rank of `adopted_primary`). Deployment table is sorted by eligible floor-passers then highest **deployment** OAPR score, with exploratory score as tie-break. `adopted_primary` remains a provenance flag for the scientific default `error_hardened_all_raw_rf_iou0_45` (exploratory **0.9172**).
- Live App 3-source bank (RF-DETR + YOLO11-1280 + SCRFD) is validated separately under `outputs/02_face_detection/16_runtime_source_validation/` (best runtime CV ≈ **0.9184** logreg / **0.9148** RF) and must not be conflated with the 7-source offline 0.917165 figure without that table.
- App / runtime tier `accelerated_full` defaults to the adopted primary when the deploy model is present.
- Reusable library: `src/detection/error_hardened_rf_policy.py`; App runtime: `app/src/privacy_pipeline_app/thesis_face_detector.py`.
- Deploy model: `outputs/02_face_detection/12_detector_error_hardening/deploy_error_hardened_all_raw_rf_iou0_45.joblib` (export via `scripts/detection/export_error_hardened_rf_deploy_model.py`).
- Triple fusion remains a supporting / fallback tier only.

**Evidence sources:**
- `outputs/02_face_detection/12_detector_error_hardening/detector_error_hardening_scores.csv`
- `outputs/02_face_detection/09_privacy_weighted_detector_policy/`
- `outputs/05_oapr/decision_framework/03_stage_scores/01_face_detection_deployment.csv`
