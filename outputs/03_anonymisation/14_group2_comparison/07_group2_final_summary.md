# Group 2 Comparable Evaluation

RiDDLE and FALCO were evaluated on the same reviewed 500-frame protocol and under the same three-attacker, face-region utility, runtime, success, held-out grouping, and privacy-floor rules used for the established methods.

## Method and policy summary

| method                             |   n_images |   success_rate |   privacy_score |   utility_score |   mean_runtime_seconds |   balanced_score |   high_compute_score |
|:-----------------------------------|-----------:|---------------:|----------------:|----------------:|-----------------------:|-----------------:|---------------------:|
| blur                               |        500 |          1     |        0.941926 |        0.739307 |               0.492744 |         0.883374 |             0.875733 |
| diffusion_low_step                 |        500 |          1     |        0.970108 |        0.738973 |               2.29643  |         0.873475 |             0.883092 |
| falco                              |        500 |          1     |        0.996983 |        0.695289 |              16.6231   |         0.810677 |             0.863249 |
| layered_blur_downscale_noise       |        500 |          1     |        0.98081  |        0.692738 |               0.440964 |         0.889789 |             0.879228 |
| nullface                           |        500 |          1     |        0.942067 |        0.771448 |              20.1675   |         0.804239 |             0.860621 |
| pixelate                           |        500 |          1     |        0.758069 |        0.754253 |               0.485754 |         0.796057 |             0.789083 |
| reverse_personalization            |        500 |          0.964 |        0.870416 |        0.797814 |             139.44     |         0.770952 |             0.811565 |
| riddle                             |        500 |          1     |        0.995883 |        0.706685 |               0.364974 |         0.903056 |             0.892378 |
| solid_mask_black                   |        500 |          1     |        0.992073 |        0.567685 |               0.427301 |         0.858155 |             0.841185 |
| grouped_heldout_policy_with_group2 |        500 |          1     |        0.993883 |        0.730576 |             nan        |         0.909798 |           nan        |

## Paired policy comparisons

| fixed_method                 |   policy_mean |   fixed_mean |   mean_difference |     ci_low |    ci_high |   policy_wins |   fixed_wins |   ties |
|:-----------------------------|--------------:|-------------:|------------------:|-----------:|-----------:|--------------:|-------------:|-------:|
| blur                         |      0.909798 |     0.883374 |        0.0264238  | 0.0211761  | 0.0321335  |           393 |          107 |      0 |
| diffusion_low_step           |      0.909798 |     0.873475 |        0.0363226  | 0.0330686  | 0.0397249  |           456 |           44 |      0 |
| falco                        |      0.909798 |     0.810677 |        0.099121   | 0.0959458  | 0.101657   |           497 |            3 |      0 |
| layered_blur_downscale_noise |      0.909798 |     0.889789 |        0.020009   | 0.0175946  | 0.0226154  |           440 |           60 |      0 |
| nullface                     |      0.909798 |     0.804239 |        0.105559   | 0.0996374  | 0.111923   |           497 |            3 |      0 |
| pixelate                     |      0.909798 |     0.796057 |        0.113741   | 0.101734   | 0.126091   |           470 |           24 |      6 |
| reverse_personalization      |      0.909798 |     0.770952 |        0.138846   | 0.124631   | 0.154017   |           498 |            2 |      0 |
| riddle                       |      0.909798 |     0.903056 |        0.00674177 | 0.00387983 | 0.00875242 |           137 |            2 |    361 |
| solid_mask_black             |      0.909798 |     0.858155 |        0.0516433  | 0.0486435  | 0.054681   |           488 |           12 |      0 |

## Policy distribution

| selected_method   |   image_count |
|:------------------|--------------:|
| no_action_copy    |           133 |
| pixelate          |             6 |
| riddle            |           361 |

## Visual-quality boundary

A deterministic 100-crop inspection found 86 visually plausible RiDDLE crops
and 78 visually plausible FALCO crops. Obvious failures included implausible
eyes or expressions, geometry/compositing artifacts, and two operational
detector false positives that caused synthetic face insertion. These results
support RiDDLE as a strong measured policy candidate, but not unconditional
deployment: detector validation, artifact checks, and deterministic fallback
remain required. The review does not establish demographic preservation or
universal visual realism.
