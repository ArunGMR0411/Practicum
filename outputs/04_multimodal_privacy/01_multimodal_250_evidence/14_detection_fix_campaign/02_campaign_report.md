# Detection Fix Campaign Report

Configs evaluated: **11** (baseline + **10** fixes). Hard end-gate: 10 enhancements.

## Held-out detection ranking (promotion gate)

Gate: screen image FN reduced by ≥1, combined recall ≥ 0.96, combined OAPR within 0.015 of baseline, text FN ≤ baseline+2.

| Fix | Name | test screen FN | combined OAPR | combined recall | Promote? |
|----:|------|---------------:|--------------:|----------------:|:--------:|
| 0 | baseline_locked | 5 | 0.9458 | 0.9804 | - |
| 1 | S_text_cluster_screen_hyp_strict | 2 | 0.9458 | 0.9804 | yes |
| **2** | **S_text_cluster_screen_hyp_recall** | **1** | **0.9458** | **0.9804** | **yes (best)** |
| 3 | S_hyp_always_dense_cluster | 2 | 0.9458 | 0.9804 | yes |
| 4 | S_yolo_conf010_union | 3 | 0.9391 | 0.9804 | yes (weaker) |
| 5 | S_bottom_half_tta | 4 | 0.9424 | 0.9804 | yes (weaker) |
| 6–8 | YOLO/TTA stacks | 2–2 | 0.936–0.942 | 0.9804 | yes (not better than 2) |
| 9 | T_text_presence_gate_count2 | 5 | 0.9176 | 0.9216 | **no** (recall drop) |
| 10 | ST stack + text gate | 2 | 0.9026 | 0.9216 | **no** (recall drop) |

## Selected configuration: text-cluster screen completion

**S_text_cluster_screen_hyp_recall** - when YOLO predicts no screen, form a hypothesized screen from dense CRAFT clusters (≥5 linked boxes, 18% margin, min area 0.6% of frame); strip text inside; allow screen routing.

### End-to-end held-out adaptive (base 6 redaction variants)

| Metric | Baseline | Screen completion | Δ |
|--------|---------:|------:|--:|
| Privacy | 0.8199 | **0.8546** | **+0.0348** |
| Utility | 0.7005 | 0.6689 | −0.0316 |
| Score | 0.8192 | **0.8273** | **+0.0081** |
| Missed screen images | 5 | **1** | **−4** |
| Insufficient screen obscuration | 5 | **2** | **−3** |
| Utility < 0.50 | 34 | 37 | +3 |
| Missed text | 2 | 2 | 0 |

### Why this is valid

1. Directly attacks the measured cascade (screen miss → text-only pixelate → privacy ~0).
2. Uses only signals already available (CRAFT boxes); no protocol change; locked 175/75.
3. Combined-risk OAPR/recall unchanged (0.9458 / 0.9804).
4. Privacy and objective score improve; residual screen flags drop.
5. Utility cost is expected and bounded (more legitimate screen fills).

### Residual after promotion

- 1 hard miss remains (`klaus/11_0708`: only 1 text box - below cluster threshold).
- Occasional low-IoU hypotheses (e.g. wrong cluster on `luca/15_0400`) can claim presence without covering GT.
- Pure environmental text FP mass is only partially reduced (not the primary lever).

## Non-promoted paths (accepted after gate)

- Extra YOLO conf/TTA alone: smaller screen-FN gains than text-cluster hyp.
- Text presence count≥2: hurts combined recall below 0.96.
- Full stack + text gate: best on **development** selection score but **fails held-out** recall gate - correctly rejected.

## Artifacts

- Campaign metrics: `01_campaign_summary.csv`
- Screen-completion predictions: `pred_02_S_text_cluster_screen_hyp_recall.csv`
- E2E redaction: `e2e_fix02/`
- Canonical evidence updated under `outputs/04_multimodal_privacy/01_multimodal_250_evidence/`
- Pre-promotion snapshot: `pre_promotion_snapshot/` (within this campaign package)
