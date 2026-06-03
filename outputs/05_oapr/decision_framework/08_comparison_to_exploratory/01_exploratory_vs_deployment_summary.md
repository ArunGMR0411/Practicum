# Deployment selection vs exploratory composites

In-body progressive evaluation under `outputs/05_oapr/decision_framework/`.
Exploratory composites remain valid for method comparison; gates govern deployable defaults.
Canonical interpretation: `outputs/09_traceability/01_evidence_index.csv` (Final OAPR boundary).

## Face detection

- Top deployment detector (among listed): `fusion_rfdetr_yolo11s_scrfd10g` score `0.9165` (exploratory `0.9087`), recall `0.9321`, recall_floor_pass=True.

## Face anonymisation

| method | eligible | deployment score | exploratory balanced | privacy_deployment | utility_deployment |
|--------|:--------:|---------:|----------------:|-----------:|-----------:|
| reverse_personalization | False | 0.8487 | 0.8486543421058005 | 0.8232 | 0.9917 |
| solid_mask_black | True | 0.8142 | 0.9324310352859834 | 0.7073 | 0.9194 |
| riddle | False | 0.8130 | 0.9030561813295034 | 0.7033 | 0.9201 |
| layered_blur_downscale_noise | True | 0.8047 | 0.9520045539444723 | 0.6725 | 0.9579 |
| diffusion_low_step | False | 0.7829 | 0.9127037184993366 | 0.6680 | 0.9548 |
| blur | True | 0.7708 | 0.9359917948835658 | 0.6078 | 0.9662 |
| falco | False | 0.7661 | 0.8106769135141331 | 0.7044 | 0.9156 |
| nullface | False | 0.7218 | 0.8179418872861857 | 0.6259 | 0.9174 |
| pixelate | False | 0.6540 | 0.8439795621180424 | 0.3919 | 0.9736 |

**Eligible default under deployment gates:** `solid_mask_black` (score 0.8142).
Generative / non-eligible methods remain ranked for research but **cannot** win default policy under gates.

## Multimodal redaction (test)

| variant | deployment score | exploratory | privacy_deployment | utility_deployment |
|---------|---------:|-------:|-----------:|-----------:|
| text_fill_screen_fill | 0.8889 | 0.8298 | 0.8867 | 0.8572 |
| text_blur_screen_fill | 0.8821 | 0.8319 | 0.8687 | 0.8704 |
| text_fill_screen_blur | 0.7593 | 0.6989 | 0.6586 | 0.8825 |
| text_blur_screen_blur | 0.7530 | 0.7028 | 0.6405 | 0.8975 |
| text_pixelate_screen_blur | 0.7515 | 0.7035 | 0.6364 | 0.9005 |
| text_pixelate_screen_pixelate | 0.7396 | 0.7164 | 0.5947 | 0.9438 |

## Progressive evaluation stance (adopted)

1. **Exploratory composites** support method comparison tables across stages.
2. **Gated deployment selection** governs deployable defaults (atomic metrics → gates → scores → sensitivity → Pareto).
3. **Research-only** methods may rank highly numerically but cannot win defaults after gates.

Interpretation ledger remains `outputs/09_traceability/01_evidence_index.csv`; this package is the evaluation evidence body.
