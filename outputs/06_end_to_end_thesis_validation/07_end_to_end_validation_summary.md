# End-to-End Thesis Pipeline Validation

## Live hardened RF propagation (canonical)

The adopted detector `error_hardened_all_raw_rf_iou0_45` is live-materialised with the scientific visual-safe policy (286 layered / 81 solid_mask / 133 copy) under:

- `outputs/10_final_enhancement_evaluation/06_frozen_scientific_oapr_route_replay/metadata/summary.json`
- Propagation note: `10_hardened_rf_live_propagation.md`

## Result (historical table materialisation in this folder)

The integrated policy completed 500/500 frames with 0 failures. It routed {"layered_blur_downscale_noise": 286, "no_action_copy": 133, "solid_mask_black": 81}.

Its balanced OAPR anonymisation score was 0.9562. The strongest fixed comparator was `fixed_layered_blur_downscale_noise` at 0.9520.

## Interpretation

This experiment tests whether the completed condition profiler changes the anonymisation decision usefully. It does not retrain or regenerate any anonymisation method. Existing outputs are selected through the runtime-safe category policy and validated for presence, success, privacy, utility, and runtime.

The integrated result is objective-specific. It does not establish global dominance over every fixed method. No-face copying is permitted only after the evidence-supported condition profiler predicts no face; residual detector risk therefore remains part of the claim boundary.

After the RF-DETR candidate safety gate, the final policy retained 0 unsafe false `no_face` skip(s). The gate overrides a no-action decision whenever a high-confidence face candidate remains; it fired on 13 frame(s).

## Multimodal Boundary

The face-anonymisation 500 and reviewed multimodal 250 overlap on 0 frames. Multimodal routing is validated separately and is not imputed onto these 500 frames.

## Previous OAPR Comparison

The previous OAPR modes remain valid objective-specific baselines. Their metrics were measured from materialised routed outputs; the current row uses the same 500-frame face protocol but adds the final condition-aware anonymisation policy.

| policy_generation   | method                            | objective_mode              |   n_input_frames |   n_success |   n_failure | method_distribution                                                                  |   SSIM_mean |   LPIPS_mean |   AdaFace_reid_rate |   ArcFace_reid_rate |   runtime_mean_seconds |   runtime_total_seconds |
|:--------------------|:----------------------------------|:----------------------------|-----------------:|------------:|------------:|:-------------------------------------------------------------------------------------|------------:|-------------:|--------------------:|--------------------:|-----------------------:|------------------------:|
| previous_oapr       | oapr_compute_profile_adaptive     | compute_profile_adaptive    |              500 |         500 |           0 | {"blur": 354, "copy": 146}                                                           |    0.991464 |   0.0087908  |         0.0258014   |         0.0555121   |               0.409516 |                 204.758 |
| previous_oapr       | oapr_failure_avoidance            | failure_avoidance           |              500 |         500 |           0 | {"blur": 21, "copy": 146, "layered": 333}                                            |    0.99107  |   0.00899199 |         0.0101642   |         0.028147    |               0.431233 |                 215.616 |
| previous_oapr       | oapr_multimodal_risk              | multimodal_risk             |              500 |         500 |           0 | {"blur": 354, "copy": 146}                                                           |    0.991464 |   0.0087908  |         0.0258014   |         0.0555121   |               0.4038   |                 201.9   |
| previous_oapr       | oapr_privacy_first                | privacy_first               |              500 |         500 |           0 | {"copy": 146, "solid_mask": 354}                                                     |    0.979281 |   0.0186164  |         0.000781861 |         0.000781861 |               0.400564 |                 200.282 |
| previous_oapr       | oapr_runtime_aware                | runtime_aware               |              500 |         500 |           0 | {"blur": 79, "copy": 146, "layered": 275}                                            |    0.991245 |   0.00879141 |         0.0164191   |         0.0344019   |               0.421331 |                 210.666 |
| previous_oapr       | oapr_utility_priority             | utility_priority            |              500 |         500 |           0 | {"blur": 354, "copy": 146}                                                           |    0.991464 |   0.0087908  |         0.0258014   |         0.0555121   |               0.396663 |                 198.331 |
| previous_oapr       | oapr_utility_under_privacy_floor  | utility_under_privacy_floor |              500 |         500 |           0 | {"blur": 354, "copy": 146}                                                           |    0.991464 |   0.0087908  |         0.0258014   |         0.0555121   |               0.395342 |                 197.671 |
| integrated_current  | integrated_condition_aware_policy | balanced_condition_aware    |              500 |         500 |           0 | {"layered_blur_downscale_noise": 286, "no_action_copy": 133, "solid_mask_black": 81} |    0.988642 |   0.0105976  |         0.0079679   |         0.0188817   |               0.325053 |                 162.526 |
