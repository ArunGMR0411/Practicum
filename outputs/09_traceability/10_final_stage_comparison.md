# Final Stage-wise Comparison Table

**Purpose**: Canonical summary of the chosen component, fallback, and limitation for each major pipeline stage.

**Last updated**: 2026-07-15

| Stage | Final Winner / Policy | Fallback | Key Limitation |
|-------|-----------------------|----------|----------------|
| **Face Detection** | Balanced: `error_hardened_all_raw_rf_iou0_45` (composite **0.9172**). Recall-first: RF-DETR-aware reranker (**0.9129**). | SCRFD (no_face); high-recall fusions | Recall floor for deployment selection; some FN not in candidate pool. |
| **Scene-Condition Router** | Hybrid SCR + detector telemetry (router **0.7015** → post-detection profile **0.9255**) | Rule / no route for weak labels | Only route-eligible labels influence routing. |
| **Multimodal Detection** | CRAFT 4K + YOLO11 640/1280 + text-cluster screen completion + strict edge-phone residual. Presence composite **0.9458**, recall **0.9804**, screen presence FN **0**. | YOLO-only screen | Localisation-oriented score ~**0.62**; low-IoU presence and detector-incapable text remain. |
| **Face Anonymisation** | Visual-safe deterministic: **layered** (balanced) / **solid_mask** (privacy-first). Deployable under gates: solid_mask / layered / blur. | Reviewed no-face copy-through | Generative methods research-only under visual gates. |
| **Multimodal Anonymisation** | Adaptive predicted-box policy: privacy **0.8680**, score **0.8320** (held-out). | Fixed text-blur/screen-fill (**0.8319**) | Not statistically superior to best fixed; residual text/low-IoU; utility cost of fills. |
| **End-to-End Policy** | OAPR visual-safe deterministic with safety-gated eligibility | Fixed layered obfuscation | Objective-specific; no universal adaptive dominance. |
| **Evaluation design** | Atomic metrics → safety gates → deployment scores → sensitivity → Pareto | Exploratory composites alone | Exploratory tables remain valid for comparison; gates govern deployable defaults. |

**No-action copy safety:** 133 final copies audited; all 0 safety candidates after RF-DETR gate (13 overrides).

**Sources:** `outputs/02_face_detection/14_final_detector_policy/`, `outputs/04_multimodal_privacy/01_multimodal_250_evidence/`, `outputs/05_oapr/decision_framework/`, `outputs/03_anonymisation/16_visual_quality_hardening/`.
