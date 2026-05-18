# Detector Error Hardening

Purpose: test whether the final face detector policy can be pushed toward `0.95` by mining false positives/false negatives from retained detector candidate boxes.

Result:

- Current final policy `cv_box_reranker_with_rfdetr_predicted_conditions`: OAPR detector score `0.9129`, precision `0.9075`, recall `0.9145`, F1 `0.9110`, TP `2384`, FP `243`, FN `223`.
- Best hardening variant `error_hardened_all_raw_rf_iou0_45`: OAPR detector score `0.9172`, precision `0.9319`, recall `0.9129`, F1 `0.9223`, TP `2380`, FP `174`, FN `227`.
- The retained candidate pool does not support a defensible `0.95` deployable score by reranking alone.

Candidate-pool recoverability:

| Pool | Current FNs | Recoverable @ IoU 0.5 | Recoverable @ IoU 0.4 | Recoverable @ IoU 0.3 |
|---|---:|---:|---:|---:|
| raw_pool | 223 | 105 | 129 | 142 |
| raw_plus_fusions_plus_current | 223 | 105 | 129 | 142 |

Interpretation:

- Some current missed faces are recoverable from retained candidates, but accepting them without new evidence increases false positives.
- The best fold-safe reranker improves precision and the OAPR detector score modestly, but does not recover enough faces to reach `0.95`.
- Reaching `0.95` likely requires new candidate generation for the remaining non-recoverable false negatives or stronger crop-level face/non-face evidence, not just threshold tuning.
