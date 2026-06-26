# OAPR Decision Framework Specification

**Status:** in-body progressive evaluation (canonical under RQ4 / OAPR).  
**Location:** `outputs/05_oapr/decision_framework/`  
**Role:** gates and deployment scores for deployable defaults; exploratory composites remain on stage comparison tables.

## Design principles

1. **Two layers:** hard safety / eligibility gates first; weighted scores only among eligible candidates.
2. **Atomic metrics first;** composites are selection tools, not the only claim.
3. **Privacy failures must not be hidden** by averages (missed face/screen/text → hard penalty or ineligible).
4. **Sensitivity + Pareto** accompany every stage ranking.
5. **Exploratory formulas preserved** under stage packages (`outputs/02..04`) and `outputs/09_traceability/01_evidence_index.csv`.

## Stage formulas (deployment selection)

### Face detection

- Atomic: Precision, Recall, F1, F2, Specificity (zero-face).
- \(F_2 = 5PR/(4P+R)\).
- **Deployment detector score** (face-positive / overall with specificity term when available):

```text
0.55 * Recall + 0.25 * F2 + 0.10 * Precision + 0.10 * Specificity
```

If specificity is unavailable, use `0.65*Recall + 0.25*F2 + 0.10*Precision`.

- **Gate:** cannot promote if recall < recall_floor (default 0.85 on combined surface).

### Scene-condition router

```text
0.70 * macro_F2 + 0.20 * macro_F1 + 0.10 * sample_Jaccard
```

- **Gate:** labels influence routing only if `route_eligible` and support ≥ floor (from existing eligibility).

### Multimodal detection (split then composite)

Report separately: screen (IoU50), text (region-hit), combined presence.

```text
Deployment multimodal detection score =
  0.40 * screen_recall_IoU50
+ 0.25 * text_privacy_region_recall
+ 0.15 * screen_precision_IoU50
+ 0.10 * text_precision_region
+ 0.10 * combined_presence_recall
```

- **Rule:** if GT screen present and missed at image level → screen privacy path for redaction uses 0.

### Face anonymisation

**Gates (in order):**

1. failure_rate > 0.05 → not deployable  
2. visual eligibility not ELIGIBLE → research-only / excluded  
3. privacy floor (balanced default): mean max(AdaFace, ArcFace) Re-ID rate > 0.40 → not eligible for balanced default  
   privacy floor (privacy-first): mean max Re-ID > 0.10 → not eligible for privacy-first default  

**Privacy:** `1 - max(AdaFace_reid_rate, ArcFace_reid_rate)` (stricter of the two recognisers).

**Utility:**

```text
full_frame_utility = 0.5 * SSIM + 0.5 * clip(1 - LPIPS/0.05, 0, 1)
non_sensitive_utility = SSIM after restoring original face regions in both images
                      (measures preservation outside protected face boxes)
Utility = 0.60 * non_sensitive_utility + 0.40 * full_frame_utility
```

If non-sensitive cannot be computed (missing image): fall back to full_frame with `utility_source=full_frame_fallback`.

**Score (eligible only):**

```text
0.55 * Privacy + 0.25 * NonSensitiveUtility + 0.10 * FullFrameUtility
+ 0.05 * RuntimeScore + 0.05 * Success
```

RuntimeScore = clip(1 - runtime_seconds / 5.0, 0, 1).

### Multimodal redaction

**Per modality privacy:**

```text
Text Privacy = 0 if GT text present and predicted text missing
else 0.70 * OCR_suppression_or_privacy + 0.30 * text_region_obscuration

Screen Privacy = 0 if GT screen present and predicted screen missing
else 0.80 * screen_obscuration + 0.20 * min(1, predicted_screen_count/max(1,gt_screen_count))  # coverage proxy

Multimodal Privacy = mean of active GT modality privacies
```

**Utility:**

```text
0.70 * (1 - non_sensitive_change_fraction) + 0.30 * full_frame_utility
```

**Score:**

```text
0.60 * MultimodalPrivacy + 0.30 * MultimodalUtility
+ 0.05 * RuntimeScore + 0.05 * Success
```

RuntimeScore = clip(1 - runtime_seconds / 2.0, 0, 1).

## Objective modes (sensitivity)

| Mode | Privacy | Utility | Runtime | Success |
|------|--------:|--------:|--------:|--------:|
| privacy_heavy | 0.70 | 0.15 | 0.05 | 0.10 |
| balanced | 0.55 | 0.30 | 0.05 | 0.10 |
| utility_heavy | 0.35 | 0.50 | 0.05 | 0.10 |
| runtime_aware | 0.45 | 0.25 | 0.20 | 0.10 |
| failure_avoidance | 0.50 | 0.25 | 0.05 | 0.20 |

(For detector/condition stages, sensitivity reweights recall/F2/precision terms analogously.)

## Pareto

Per stage: methods with no other method strictly better on all of {privacy, utility, runtime_or_recall, success_or_f1}.

## Mapping from exploratory scores

See `02_mapping_from_exploratory_scores.md`.

## Stage mirrors

For navigation next to domain evidence:

- `outputs/02_face_detection/15_deployment_selection/`
- `outputs/03_anonymisation/18_deployment_selection/`
- `outputs/04_multimodal_privacy/01_multimodal_250_evidence/15_deployment_selection/`
