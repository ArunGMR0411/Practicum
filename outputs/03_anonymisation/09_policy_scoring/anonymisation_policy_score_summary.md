# Category-Aware Anonymisation Policy Scores

Purpose: derive evidence-based anonymisation policy choices from per-image metrics joined with the reviewed egocentric-stress 500 condition profile and the 250-frame multimodal risk policy.

Score definition:

- `privacy_score = 1 - mean(AdaFace_reid_rate, ArcFace_reid_rate)`; no-face frames get privacy residual `0`, while failed method outputs get residual `1`.
- `utility_score = 0.5 * SSIM + 0.5 * (1 - LPIPS/0.05)`, clipped to `[0, 1]`.
- `runtime_score = 1 - runtime_seconds/5`, clipped to `[0, 1]`; methods without full per-frame runtime use the documented aggregate runtime boundary.
- `success_score = 1` for successful outputs and `0` for failures.
- `balanced_oapr_anonymisation_score = 0.50*privacy + 0.30*utility + 0.10*runtime + 0.10*success`.

Additional objective scores:

- `privacy_first_score = 0.70*privacy + 0.15*utility + 0.05*runtime + 0.10*success`.
- `utility_preserving_score = 0.30*privacy + 0.50*utility + 0.10*runtime + 0.10*success`.
- `runtime_practical_score = 0.40*privacy + 0.20*utility + 0.30*runtime + 0.10*success`.

Policy findings:

- `layered_blur_downscale_noise` is the balanced default decision for 12 category rows.
- `skip_face_anonymisation_then_apply_multimodal_policy_if_needed` is the balanced default decision for 1 category rows.
- `solid_mask_black` is the balanced default decision for 1 category rows.

Boundary:

- This is a category-aware scoring table, not a claim that one method globally dominates.
- `no_face` routes to no face anonymisation; multimodal policy still applies if text/screen risk exists.
- Reverse Personalization is included with its 18-frame failure penalty and high runtime cost; it is not a default deployment candidate.
- StyleID/FAMS are excluded because the final decision is quality-limited after systematic tuning.

Canonical outputs:

- Per-image metric table: `outputs/03_anonymisation/09_policy_scoring/anonymisation_policy_per_image_metrics.csv`.
- Category score table: `outputs/03_anonymisation/09_policy_scoring/anonymisation_policy_category_scores.csv`.
- Final category decision table: `outputs/03_anonymisation/09_policy_scoring/anonymisation_policy_decision_table.csv`.
