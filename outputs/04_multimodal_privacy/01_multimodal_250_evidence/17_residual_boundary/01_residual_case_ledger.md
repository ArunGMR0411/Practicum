# Multimodal residual boundary ledger

**Status:** residual hard-miss boundary classified (see `outputs/09_traceability/01_evidence_index.csv`).

This package classifies held-out residual flags after the promoted stack
(CRAFT 4K + YOLO11 union + text-cluster screen completion + strict edge-phone residual).
Remaining cases are residual limitations, not unfinished default-policy work.

## Held-out residual flag counts

- Missed text presence: `2`
- Missed screen presence: `0` (closed cascade)
- Residual text readability: `6`
- Insufficient screen obscuration: `1`
- Utility < 0.50: `37`
- Any residual flag: `42`

## Privacy-critical residual cases (privacy ≈ 0)

| protocol_id | image_id | classes | disposition |
|-------------|----------|---------|-------------|
| MM2_0148 | `day3/members/florian/20_0568.webp` | missed_text_presence|residual_text_readability | frozen_detector_incapable_text |
| MM2_0023 | `day1/members/onanong/15_0421.webp` | residual_text_readability | frozen_localisation_or_ocr_residual |
| MM2_0030 | `day1/members/cathal/17_0482.webp` | residual_text_readability | frozen_localisation_or_ocr_residual |
| MM2_0128 | `day3/members/cathal/15_0532.webp` | residual_text_readability | frozen_localisation_or_ocr_residual |
| MM2_0180 | `day4/members/luca/15_0400.webp` | insufficient_screen_obscuration | frozen_low_iou_screen_hyp |

## Residual dispositions

- `frozen_detector_incapable_text` - no recoverable detector family on current stack.
- `frozen_text_miss_screen_protected` - text miss while screen path already redacts.
- `frozen_low_iou_screen_hyp` - presence recovered with wrong box; obscuration residual.
- `frozen_localisation_or_ocr_residual` - text path fires but residual readability flag.
- `frozen_screen_fill_utility_cost` - privacy-first fill utility cost (expected).
- All ledger rows mark residual classes as closed for default-stack promotion.

## Why these residuals remain

1. Screen-presence cascade is closed (FN 5→1→0) with gated residual, not loose high-FP stacks.
2. Remaining privacy-zero text case fails CRAFT/EAST/docTR/MSER probes.
3. Operator ablations for utility/readability were negative under the declared score.
4. Further gains need a new protocol or new detector family with development-gated evidence.

## Machine-readable ledger

- `02_heldout_residual_case_ledger.csv`
- `03_residual_flag_counts.json`

Canonical interpretation: `outputs/09_traceability/01_evidence_index.csv` (Residual hard-miss boundary).
