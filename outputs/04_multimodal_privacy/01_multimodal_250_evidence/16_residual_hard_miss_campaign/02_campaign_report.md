# Residual Hard-Miss Campaign

Targets residual detection failures after text-cluster screen completion.

## Held-out ranking

| ID | Name | screen FN | low-IoU presence | combined OAPR | screen misses | text FN |
|---:|------|----------:|-----------------:|--------------:|---------------|--------:|
| 0 | R0_promoted_text_cluster_hyp | 1 | 2 | 0.9458 | `MM2_0216` | 2 |
| 1 | R1_edge_phone_loose | 0 | 1 | 0.9255 | `` | 2 |
| 2 | R2_yolo_conf005_union | 2 | 2 | 0.9359 | `MM2_0015|MM2_0216` | 2 |
| 3 | R3_yolo005_plus_hyp | 1 | 2 | 0.9359 | `MM2_0216` | 2 |
| 4 | R4_hyp_plus_edge_loose | 0 | 2 | 0.9255 | `` | 2 |
| 5 | R5_residual_stack_loose | 0 | 2 | 0.9255 | `` | 2 |
| **8** | **R8_hyp_plus_strict_edge_phone** | **0** | **2** | **0.9458** | `` | **2** |
| 9 | R9_strict_edge_only_if_empty | 4 | 0 | 0.9458 | `MM2_0015|…` | 2 |

## Promotion gate

Control R0: screen FN=1, low-IoU presence=2, OAPR=0.9458.

**Promoted residual config:** `R8_hyp_plus_strict_edge_phone`

- Strict landscape bottom-phone edge residual after text-cluster hyp still empty.
- Gates: area 1.5–5%, AR 1.3–1.9, center_y ≥0.82, edge density ≥0.08, intensity std ≥60, score ≥0.15.
- Held-out: screen FN **0** (recovers `MM2_0216` klaus/11_0708), combined OAPR **unchanged** 0.9458, zero added false screens on protocol.

Loose edge stacks reduce FN but add many false screens and drop OAPR (~0.925) → **not promoted**.

## End-to-end after promotion (held-out adaptive)

| Metric | Pre-residual (text-cluster only) | + strict edge-phone | Δ |
|--------|---------------------------------:|--------------------:|--:|
| Privacy | 0.8546 | **0.8680** | **+0.0134** |
| Utility | 0.6689 | 0.6634 | −0.0055 |
| Score | 0.8273 | **0.8320** | **+0.0048** |
| Screen presence FN | 1 | **0** | **−1** |
| Insufficient screen obscuration | 2 | **1** | **−1** |
| Missed text | 2 | 2 | 0 |
| Utility &lt; 0.50 | 37 | 37 | 0 |

Strongest fixed remains text-blur/screen-fill (~0.8319); adaptive is competitive, not statistically superior.

## Residual taxonomy (after promotion)

| Class | IDs / note | Status |
|-------|------------|--------|
| Screen presence miss | none on held-out | **closed** by R8 |
| Low-IoU presence | `MM2_0180` luca/15_0400 (wrong text-cluster hyp) | residual localisation |
| Text presence miss | `MM2_0014` allie (text next to covered screen; privacy high); `MM2_0148` florian | florian **detector-incapable** under CRAFT/EAST/docTR/MSER |
| Residual text readability | 6 images | localisation mismatch / weak OCR suppression |
| Utility collapse | 37 images | screen-fill cost |

### Probe evidence (not promoted)

- YOLO conf 0.05 recovers luca at IoU ~0.43 but does not beat R0 held-out screen FN without extra FPs in isolation.
- Loose edge-phone: klaus recovered but ~24 held-out false screens → rejected.
- Florian GT text region has near-zero edge density; no detector stack recovered it.

## Artifacts

- `01_campaign_summary.csv`, `pred_*.csv`, `edge_phone_proposals.csv`, `extra_yolo_conf005.csv`
- `03_best_residual_predictions.csv` (= R8)
- `pre_residual_promotion_snapshot/` - canonical tables before residual promotion
- Canonical materialization under parent `01_multimodal_250_evidence/`
