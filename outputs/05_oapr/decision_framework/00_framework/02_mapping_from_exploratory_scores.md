# Exploratory composite → deployment selection mapping

Canonical definitions: `configs/scoring_definitions.json`.

| Stage | Score ID | Formula |
| --- | --- | --- |
| face_detection | `oapr_detector_exploratory` | `0.65 * recall + 0.25 * f1 + 0.10 * precision` |
| face_anonymisation | `balanced_oapr_anonymisation` | `0.40 * privacy_max_reid + 0.30 * utility + 0.20 * runtime_score + 0.10 * success_score` |
| multimodal_detection_presence | `oapr_multimodal_presence` | `0.65 * combined_presence_recall + 0.25 * combined_presence_f1 + 0.10 * combined_presence_precision` |
| multimodal_detection_localisation | `oapr_multimodal_localisation_deploy` | `0.40 * screen_iou50_recall + 0.25 * text_region_recall + 0.15 * screen_iou50_precision + 0.10 * text_region_precision + 0.10 * combined_presence_recall` |
| multimodal_redaction | `multimodal_anonymisation_score` | `0.50 * privacy_score + 0.30 * utility_score + 0.10 * runtime_score + 0.10 * success_score` |

Face detector recall floor: **0.85**.

