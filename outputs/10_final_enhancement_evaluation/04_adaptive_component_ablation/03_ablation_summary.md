# Adaptive component ablation

Goal: show each major adaptive/gated component contributes measurable safety
or selection structure — not only a higher composite score.

## Face policy selection

| ablation_id | component_removed | selected_method | interpretation |
| --- | --- | --- | --- |
| A0_full_gated_framework | none | solid_mask_black | Baseline gated framework |
| A1_exploratory_score_only | all gates | layered_blur_downscale_noise | Exploratory composite alone can change selection |
| A2_no_visual_safety_filter | visual eligibility | solid_mask_black | Gate removal expands eligible set / unsafe exposure; **aggregate winner unchanged** under this config |
| A3_no_privacy_floor | privacy floor | solid_mask_black | Floor constrains weaker-privacy methods; winner unchanged here |
| A9_fixed_layered_only | OAPR routing | layered_blur_downscale_noise | Fixed method loses objective-specific solid_mask routes |

## Multimodal

| ablation_id | interpretation |
| --- | --- |
| A6_mm_pre_screen_completion | Historical adaptive before residual screen completion (higher screen FN) |
| A7_mm_full_residual_stack | Current held-out adaptive after residual stack (privacy improved; screen FN 0) |
| A8_mm_fixed_blur_fill_only | Strongest fixed comparator; adaptive competitive, not statistically dominant |

## Correct wording

- Where removing a gate **did not change the selected winner**, state that removing the gate **expanded the eligible set and increased exposure** to unsafe or weaker methods, **although the aggregate winner remained unchanged** under the evaluated configuration.
- Do **not** claim a generative method became the winner when the table shows solid_mask/layered.
- Do **not** claim every component produced a statistically significant improvement or that OAPR universally outperforms every fixed method.

## Valid retained findings

- Gates prevent visually unsafe methods from being default-eligible.
- Privacy floors constrain utility-oriented selections.
- Residual screen-completion stages reduced multimodal misses.
- Adaptive multimodal redaction was competitive with the strongest fixed method.
- Objective-specific scientific OAPR routing provides different operational behaviour (286 layered / 81 solid_mask / 133 copy) even when aggregate scores are close to fixed layered.

Source table: `01_ablation_comparison_table.csv`
