# OAPR decision-framework validation pass

Date: 2026-07-15

## Checks

- All required stage/gate/sensitivity/Pareto files present under `outputs/05_oapr/decision_framework/`.
- Face scores finite; no NaN in primary deployment scores.
- Research methods (nullface, diffusion, RP) **not** default-eligible.
- Eligible methods share **full_protocol** non-sensitive source after expansion.
- Multimodal deployment composite **below** exploratory combined presence OAPR (stricter metric).
- Redaction leaders are fill/fill or blur/fill class.
- Sensitivity winners restricted to eligible methods.
- Stage mirrors present under `02/15`, `03/18`, and multimodal `15_deployment_selection`.

**Issues found on first pass:** none blocking.

## Non-sensitive utility expansion

- Before: 50 frames/method (350 total).
- After: full success set - blur/layered/solid/pixelate/diffusion/nullface **500**; RP **482** (canonical `09_rp_final_metric_summary.csv` supersedes older 444 per-image aggregate).
- File: `01_atomic_metrics/04_face_nonsensitive_full_protocol.csv` (3444 rows).
- Face stage scores and sensitivity re-written from full means.

## Residual caveats

1. Multimodal detection deployment score (~0.62) must be narrated as **stricter localisation**, not performance regression vs presence OAPR (~0.95).
2. Face-restore NS SSIM is near-ceiling for methods that only edit face boxes; it mainly flags bleed outside faces (e.g. NullFace slightly lower). Full-frame utility still carries most scene-preservation discrimination.
3. RiDDLE/FALCO may lack rows in the policy_scoring per-image CSV used here; they remain gated via visual eligibility tables and all-methods comparison as research comparators.
