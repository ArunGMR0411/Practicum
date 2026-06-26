# Exploratory composites vs gated deployment selection

**Promotion stance (progressive evaluation design):** exploratory = historical/exploratory weighted scores used for method comparison; deployment = final policy-selection framework under gates. Adopted as in-body OAPR evaluation - not a separate evidence tail.

| Stage | Exploratory takeaway | Deployable selection | Why same / changed |
|-------|----------------------|----------------------|--------------------|
| Face detection | Privacy-weighted `0.65R+0.25F1+0.10P`; hardened RF primary (~0.917), RF-DETR reranker recall-first | Same privacy/recall-first family + **recall floor ≥ 0.85**; F2 + specificity in composite | Ranking family unchanged; floor blocks “balanced but missy” detectors |
| Scene-condition | Custom 4-term F2-heavy score; hybrid/post-detection strong | Simpler `0.70 F2 + 0.20 F1 + 0.10 Jaccard`; weak labels analysis-only | Clearer formula; same eligibility spirit |
| Multimodal detection | Combined presence OAPR **~0.95** | Split composite **~0.62** (screen IoU + text + presence) | **Metric stricter, not system collapse** - report both |
| Face anonymisation | Layered often top **exploratory balanced** score | **Gates** then eligible set: solid_mask / layered / blur; generative research-only | Winner is **objective-dependent**; no universal best |
| Multimodal redaction | Adaptive ≈ strongest fixed blur/fill | miss→0 privacy; fill/fill & blur/fill lead | Same safety direction; misses punished harder |
| Generative methods | Strong metrics, visual audit blocks default | **Hard visual/failure gates** block default even if score high | Formalises visual-safe policy |
| OAPR end-to-end | Visual-safe deterministic routes | Gated selection among eligible methods; exploratory scores remain comparison evidence | Progressive evaluation |

## Sensitivity (face, eligible only, after full-protocol NS utility)

| Privacy weight | Utility weight | Eligible winner |
|---------------:|---------------:|-----------------|
| 0.40 | 0.50 | layered_blur_downscale_noise |
| ≥ 0.50 | ≤ 0.40 | solid_mask_black |

## Non-sensitive utility status

- Expanded to **full successful frames** per method (~500 each; RP 444): `01_atomic_metrics/04_face_nonsensitive_full_protocol.csv`.
- Source: face-restore SSIM (non-face preservation). Deterministic methods score ~0.996–0.999 (faces are the main change region).
- Comparisons among methods with full NS coverage are **even**; no eligible method uses full-frame fallback.

## How to write this in the thesis

> Exploratory OAPR weighted composites supported broad method exploration and baseline comparison. After observing that composites can hide privacy misses, visual artifacts, and multimodal localisation failures, the gated decision framework was integrated into the OAPR evidence body: atomic metrics, hard safety gates, stage scores, sensitivity, and Pareto analysis. Deployable defaults are selected only after gates; research methods may still score highly as comparators.
