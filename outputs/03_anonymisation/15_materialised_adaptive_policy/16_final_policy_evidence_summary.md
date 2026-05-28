# Final Materialised Adaptive Policy

The final policy was materialised as 500 output images. RiDDLE candidates pass through a participant-validated SigLIP2 artifact gate; predicted artifacts fall back to layered obfuscation.

## Direct output metrics

| method                         |   n_input_frames |   n_success |   n_failure |   face_crop_count |   SSIM_mean |   LPIPS_mean |   AdaFace_cosine_mean |   AdaFace_reid_rate |   ArcFace_cosine_mean |   ArcFace_reid_rate |   FaceNet_cosine_mean |   FaceNet_reid_rate_060 |   three_attacker_privacy_score |   hardened_utility_score |   hardened_balanced_score |   component_runtime_mean_seconds |   artifact_gate_recall |   artifact_gate_precision |   artifact_gate_f2 |   artifact_gate_fallbacks |
|:-------------------------------|-----------------:|------------:|------------:|------------------:|------------:|-------------:|----------------------:|--------------------:|----------------------:|--------------------:|----------------------:|------------------------:|-------------------------------:|-------------------------:|--------------------------:|---------------------------------:|-----------------------:|--------------------------:|-------------------:|--------------------------:|
| adaptive_artifact_gated_policy |              500 |         500 |           0 |              1279 |    0.985611 |     0.017364 |              0.116001 |           0.0039093 |              0.130645 |          0.00625489 |              0.106748 |              0.00781861 |                       0.994006 |                 0.728239 |                  0.906991 |                         0.465907 |               0.785714 |                      0.44 |           0.679012 |                        75 |

## Route distribution

| attempted_method   | artifact_gate_action   | final_method                 |   image_count |
|:-------------------|:-----------------------|:-----------------------------|--------------:|
| no_action_copy     | not_required           | no_action_copy               |           133 |
| pixelate           | not_required           | pixelate                     |             6 |
| riddle             | fallback_layered       | layered_blur_downscale_noise |            75 |
| riddle             | retain_riddle          | riddle                       |           286 |

## Quality-gate effect

|   review_sample_size |   initial_riddle_artifacts |   grouped_validation_artifacts_detected |   grouped_validation_artifacts_missed |   full_fit_known_artifacts_flagged |   known_grouped_misses_flagged_by_final_gate |   grouped_validation_plausible_outputs_flagged |   estimated_unseen_participant_miss_rate |   validation_fallback_rate | interpretation                                                                                                                                          |
|---------------------:|---------------------------:|----------------------------------------:|--------------------------------------:|-----------------------------------:|---------------------------------------------:|-----------------------------------------------:|-----------------------------------------:|---------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
|                  100 |                         14 |                                      11 |                                     3 |                                 14 |                                            3 |                                             14 |                                     0.03 |                       0.25 | participant-grouped generalisation estimate; all three grouped misses were flagged by the final full-fit gate, but independent review is still required |

## Paired comparison

| comparator                     |   policy_mean |   comparator_mean |   mean_difference |       ci_low |     ci_high |   policy_wins |   comparator_wins |   ties |
|:-------------------------------|--------------:|------------------:|------------------:|-------------:|------------:|--------------:|------------------:|-------:|
| blur                           |      0.906991 |          0.883374 |        0.0236163  |  0.0183478   |  0.0293708  |           365 |               135 |      0 |
| diffusion_low_step             |      0.906991 |          0.873475 |        0.0335152  |  0.0302964   |  0.0368491  |           453 |                47 |      0 |
| falco                          |      0.906991 |          0.810677 |        0.0963136  |  0.0926747   |  0.0994219  |           494 |                 6 |      0 |
| layered_blur_downscale_noise   |      0.906991 |          0.889789 |        0.0172016  |  0.0149918   |  0.0194943  |           387 |                38 |     75 |
| nullface                       |      0.906991 |          0.804239 |        0.102751   |  0.0969073   |  0.108901   |           496 |                 4 |      0 |
| pixelate                       |      0.906991 |          0.796057 |        0.110934   |  0.0987318   |  0.1231     |           462 |                32 |      6 |
| reverse_personalization        |      0.906991 |          0.770952 |        0.136039   |  0.122083    |  0.150988   |           495 |                 5 |      0 |
| riddle                         |      0.906991 |          0.903056 |        0.00393432 |  0.000666953 |  0.00652181 |           159 |                55 |    286 |
| solid_mask_black               |      0.906991 |          0.858155 |        0.0488358  |  0.045653    |  0.0519113  |           490 |                10 |      0 |
| ungated_grouped_heldout_policy |      0.906991 |          0.909798 |       -0.00280745 | -0.00433647  | -0.00145526 |            22 |                53 |    425 |

Against fixed layered obfuscation, the materialised policy gain is `0.017202`, 95% CI `[0.014992, 0.019494]`.
The quality gate changes the balanced score by `-0.002807` relative to the ungated held-out policy; this is the measured cost of conservative artifact fallback.
All three grouped-validation misses were flagged by the final full-fit gate and use layered fallback in the materialised outputs. This does not prove that every artifact in the unreviewed outputs was detected.
