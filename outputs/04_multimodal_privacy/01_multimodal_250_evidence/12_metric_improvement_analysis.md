# Multimodal Metric Improvement Analysis (250-image protocol)

**Status (2026-07-15):** Historical diagnosis that motivated operator ablation
(negative), **text-cluster screen completion** (promoted), and **strict
edge-phone residual** (promoted; `16_residual_hard_miss_campaign/`). Current
canonical RQ3 numbers are in `11_rq3_final_summary.md` and `outputs/09_traceability/01_evidence_index.csv`
(adaptive privacy ~0.8680 / score ~0.8320; screen presence misses **0** held-out).
Pre-completion adaptive baselines below are retained for chronology only.

This note analyses held-out residual and method tables from the canonical
250-image package. It identifies where metrics can still improve without
reintroducing a discarded protocol. Numbers below for “baseline” are
**pre-hypothesis baseline** unless noted. Sources:
`10_residual_risk_analysis.csv`, `02_detection_method_comparison.csv`,
`05_combined_risk_detection.csv`, and `06_redaction_per_image_metrics.csv`.

## Pre–screen-completion hypothesis held-out baseline (75 images) - superseded for adaptive redaction

| Stage | Metric | Value |
|-------|--------|------:|
| Combined risk detection | Precision / Recall / F1 | 0.8333 / 0.9804 / 0.9009 |
| Combined risk detection | OAPR multimodal score | 0.9458 |
| Adaptive redaction | Privacy / Utility / Score | 0.8199 / 0.7005 / 0.8195 |
| Strongest fixed | text_blur_screen_fill score | 0.8187 |
| Adaptive − fixed | Mean (95% CI) | +0.0008 [−0.0058, 0.0065] |

Residual flags on the held-out split:

- Missed text: 2/75
- Missed screen: 5/75
- Residual text readability: 5/75
- Insufficient screen obscuration: 5/75
- Utility below 0.50: **34/75 (45%)**
- Any residual flag: 43/75

## Where the score is lost

### 1. Utility collapse is screen-fill dominated (largest practical lever)

Adaptive utility is strong on no-risk copy-through (1.0) and text-only
pixelation (~0.91), but collapses when screens are filled:

| Predicted risk state | n | Mean utility | Utility &lt; 0.50 | Mean privacy | Mean score |
|----------------------|--:|-------------:|------------------:|-------------:|-----------:|
| no_text_screen_risk | 15 | 1.00 | 0% | 0.93 | 0.97 |
| text_present | 21 | 0.91 | 10% | 0.54 | 0.74 |
| screen_present | 12 | 0.51 | 75% | 0.92 | 0.81 |
| text_and_screen_present | 27 | 0.46 | 85% | 0.93 | 0.80 |

Screen-box count correlates with utility at about **−0.62**. Fill on large
egocentric displays removes too much of the frame under the SSIM/LPIPS utility
proxy.

**Improvement direction:** keep privacy-first fill for small/high-confidence
screens; use strong blur or layered downscale for large screen areas; or
score fill vs blur by predicted screen area fraction. This targets the 34
utility failures without lowering combined-risk recall.

### 2. Text-only privacy is weak under pixelation

Text-only adaptive privacy averages **0.54** because development selected
`text_pixelate_screen_pixelate` for text-only risk. Pixelation preserves
utility but fails OCR suppression on several residual-readability cases.

**Improvement direction:** re-select text-only action on development with a
privacy floor (for example require `privacy ≥ 0.80`) or switch text-only to
`text_blur` / `text_fill`. Expected effect: higher privacy and residual-text
reduction; some utility cost on text-only frames (currently only 2 utility
failures in that bucket).

### 3. Environmental text false positives (detection precision)

Held-out region-level text precision for CRAFT 4K-recall is **~0.018**
(image-level text precision ~0.26–0.31). Mean predicted text boxes on test is
~14.5 versus ~0.4 ground-truth boxes. False text *presence* rate is ~47%.

Screen-priority filtering already improves precision slightly
(`craft_recall_4k_with_screen_priority` precision 0.068 vs 0.018) but still
leaves massive FP mass.

**Improvement direction (ordered):**

1. Geometry filters: drop tiny boxes, extreme aspect ratios, edge-strip
   watermark candidates.
2. OCR-confidence / alphanumeric density gates before redaction.
3. Keep CRAFT proposals only when they intersect a document/screen-like
   region or a high-contrast rectangular support.
4. Optional: document detector or layout prior before OCR.

Goal is not perfect text mAP; it is fewer harmless redactions and cleaner
privacy-region proposals while protecting the current **0.98** combined-risk
recall.

### 4. Screen misses route as text-only

All five held-out missed-screen images were predicted as `text_present`
(text found, screen not). That both under-protects the display and selects the
wrong redaction state.

Held-out pure screen localisation for the selected union is IoU50 F1 ~0.53
with mean GT area coverage ~0.85. YOLO11 1280 conf0.25 alone has higher pure
screen OAPR (0.636) than the 640/1280 union (0.597), but the union was chosen
for combined-risk recall on development.

**Improvement direction:**

- Confidence/NMS retune on development only.
- Prefer 1280 for precision, keep a low-conf second pass only for high-risk
  strata.
- Fine-tune YOLO on the 139 reviewed screen boxes (175-image train split).
- Multi-label risk: do not let text detections suppress a possible screen.

### 5. Adaptive state routing is already near-oracle among fixed combos

Per-image best fixed variant mean score on held-out: **0.8320** vs adaptive
**0.8195** (+0.012). Oracle still mostly picks blur/fill or pixelate.

So the adaptive *state table* is not the main bottleneck. Bigger gains come
from better boxes and better operators (fill vs blur), not from more risk
states alone.

## Feasible experiment plan (recommended order)

| Priority | Experiment | Expected metric impact | Effort | Risk to recall |
|---------:|------------|------------------------|--------|----------------|
| 1 | Area-aware screen operator (fill if area &lt; τ else strong blur) | Utility ↑, util&lt;0.5 ↓, score ↑ | Low | Low if blur is strong |
| 2 | Text-only privacy floor (blur/fill instead of weak pixelate) | Privacy ↑, residual text ↓ | Low | None on detection |
| 3 | Environmental text FP filters on proposals | Utility ↑, text precision ↑, runtime ↓ | Low–medium | Medium - must guard recall on dev |
| 4 | Screen multi-pass / retune without union FP growth | Screen miss ↓ (5→fewer) | Medium | Medium |
| 5 | Fine-tune screen detector on reviewed boxes | Screen IoU/recall ↑ | Higher | Low if eval stays held-out |

Do **not** reintroduce any discarded multimodal image set. All ablations must
use the locked 175/75 split and report held-out only after development
selection.

## What not to chase

- Chasing adaptive − fixed statistical significance with the current operators:
  the margin is &lt;0.001 and the CI already covers zero.
- Claiming complete semantic leakage removal after OCR filters.
- Replacing CRAFT with docTR alone (held-out text recall 0.20).

## Success criteria for a worthwhile improvement

A promotion-worthy update should show, on the same held-out 75 images:

1. Combined-risk recall ≥ 0.96 (do not sacrifice the privacy-first detection story), and
2. Adaptive score ≥ 0.83 **or** utility_below_0.50 ≤ 25/75 **or** residual text+screen flags ≤ 6 total, and
3. No silent protocol change (same 250 annotations and split).

## Canonical inputs for follow-up runs

- Annotations: `outputs/01_protocol/annotations/multimodal_250/reviewed_multimodal_250_with_boxes.csv`
- Detection runner: `scripts/multimodal/run_multimodal_region_evaluation.py`
- Redaction runner: `scripts/multimodal/run_multimodal_region_redaction_evaluation.py`
- Evidence directory: `outputs/04_multimodal_privacy/01_multimodal_250_evidence/`
