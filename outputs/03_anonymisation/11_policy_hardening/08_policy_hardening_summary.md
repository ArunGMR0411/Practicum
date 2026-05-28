# Anonymisation Policy Hardening

All model inference used CUDA. Policy evaluation uses the manually reviewed 500-image egocentric-stress protocol.

## Enhanced method metrics

| method                       |   n_images |   n_success |   privacy_score_three_attacker |   enhanced_utility_score |   face_region_utility_score |   landmark_pair_rate |   background_preservation_score |   runtime_seconds |   constrained_score |   high_compute_score |
|:-----------------------------|-----------:|------------:|-------------------------------:|-------------------------:|---------------------------:|---------------------:|--------------------------------:|------------------:|--------------------:|---------------------:|
| blur                         |        500 |         500 |                       0.941926 |                 0.739307 |                  0.360057  |           0.183215   |                        0.999182 |          0.492744 |            0.883374 |             0.875733 |
| pixelate                     |        500 |         500 |                       0.758069 |                 0.754253 |                  0.380994  |           0.13351    |                        0.999185 |          0.485754 |            0.796057 |             0.789083 |
| solid_mask_black             |        500 |         500 |                       0.992073 |                 0.567685 |                  0.0117965 |           0.00127479 |                        0.998263 |          0.427301 |            0.858155 |             0.841185 |
| layered_blur_downscale_noise |        500 |         500 |                       0.98081  |                 0.692738 |                  0.258751  |           0.00325779 |                        0.998323 |          0.440964 |            0.889789 |             0.879228 |
| nullface                     |        500 |         500 |                       0.942067 |                 0.771448 |                  0.572628  |           0.71625    |                        0.994503 |         20.1675   |            0.804239 |             0.860621 |
| diffusion_low_step           |        500 |         500 |                       0.970108 |                 0.738973 |                  0.398795  |           0.531364   |                        0.998305 |          2.29643  |            0.873475 |             0.883092 |
| reverse_personalization      |        500 |         482 |                       0.870416 |                 0.797814 |                  0.668434  |           0.804043   |                        0.998249 |        139.44     |            0.770952 |             0.811565 |

## Policy comparison

| policy                          |   n_images |   mean_score |   privacy_score |   utility_score |   n_methods | method_distribution                                                                                                                                                                      |
|:--------------------------------|-----------:|-------------:|----------------:|----------------:|------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| current_condition_policy        |        500 |     0.888033 |        0.983943 |        0.674268 |           3 | {"layered_blur_downscale_noise": 286, "no_action_copy": 133, "solid_mask_black": 81}                                                                                                     |
| grouped_heldout_category_policy |        500 |     0.894409 |        0.973352 |        0.714427 |           4 | {"blur": 175, "layered_blur_downscale_noise": 186, "no_action_copy": 133, "pixelate": 6}                                                                                                 |
| privacy_constrained_pareto      |        500 |     0.88362  |        0.994971 |        0.798385 |           8 | {"blur": 46, "diffusion_low_step": 50, "layered_blur_downscale_noise": 15, "no_action_copy": 133, "nullface": 90, "pixelate": 9, "reverse_personalization": 146, "solid_mask_black": 11} |
| privacy_verifying_cascade       |        500 |     0.900863 |        0.995111 |        0.715002 |           7 | {"blur": 159, "diffusion_low_step": 35, "layered_blur_downscale_noise": 42, "no_action_copy": 133, "nullface": 11, "pixelate": 84, "solid_mask_black": 36}                               |

## Statistical comparison

| policy                          | fixed_method                 |   policy_mean |   fixed_mean |   mean_difference |       ci_low |      ci_high |   policy_win_count |   fixed_win_count |   tie_count |
|:--------------------------------|:-----------------------------|--------------:|-------------:|------------------:|-------------:|-------------:|-------------------:|------------------:|------------:|
| current_condition_policy        | blur                         |      0.888033 |     0.883374 |       0.00465838  | -0.000430982 |  0.0100686   |                232 |               268 |           0 |
| current_condition_policy        | pixelate                     |      0.888033 |     0.796057 |       0.0919759   |  0.0813071   |  0.102879    |                413 |                87 |           0 |
| current_condition_policy        | solid_mask_black             |      0.888033 |     0.858155 |       0.0298779   |  0.0276226   |  0.0321562   |                401 |                18 |          81 |
| current_condition_policy        | layered_blur_downscale_noise |      0.888033 |     0.889789 |      -0.00175636  | -0.00423372  |  0.000728571 |                140 |                74 |         286 |
| current_condition_policy        | nullface                     |      0.888033 |     0.804239 |       0.0837935   |  0.0775605   |  0.09018     |                458 |                42 |           0 |
| current_condition_policy        | diffusion_low_step           |      0.888033 |     0.873475 |       0.0145572   |  0.0105445   |  0.0184181   |                376 |               124 |           0 |
| current_condition_policy        | reverse_personalization      |      0.888033 |     0.770952 |       0.117081    |  0.103339    |  0.131946    |                455 |                45 |           0 |
| grouped_heldout_category_policy | blur                         |      0.894409 |     0.883374 |       0.0110349   |  0.00664183  |  0.0160376   |                200 |               125 |         175 |
| grouped_heldout_category_policy | pixelate                     |      0.894409 |     0.796057 |       0.0983524   |  0.0870874   |  0.109679    |                434 |                60 |           6 |
| grouped_heldout_category_policy | solid_mask_black             |      0.894409 |     0.858155 |       0.0362543   |  0.0329026   |  0.039435    |                473 |                27 |           0 |
| grouped_heldout_category_policy | layered_blur_downscale_noise |      0.894409 |     0.889789 |       0.00462011  |  0.00267314  |  0.00626096  |                280 |                34 |         186 |
| grouped_heldout_category_policy | nullface                     |      0.894409 |     0.804239 |       0.09017     |  0.0837315   |  0.096975    |                485 |                15 |           0 |
| grouped_heldout_category_policy | diffusion_low_step           |      0.894409 |     0.873475 |       0.0209337   |  0.0171151   |  0.0246627   |                407 |                93 |           0 |
| grouped_heldout_category_policy | reverse_personalization      |      0.894409 |     0.770952 |       0.123457    |  0.109863    |  0.138098    |                483 |                17 |           0 |
| privacy_constrained_pareto      | blur                         |      0.88362  |     0.883374 |       0.000246151 | -0.00539391  |  0.00633415  |                218 |               236 |          46 |
| privacy_constrained_pareto      | pixelate                     |      0.88362  |     0.796057 |       0.0875637   |  0.0758349   |  0.099462    |                406 |                85 |           9 |
| privacy_constrained_pareto      | solid_mask_black             |      0.88362  |     0.858155 |       0.0254656   |  0.0229772   |  0.0281181   |                428 |                61 |          11 |
| privacy_constrained_pareto      | layered_blur_downscale_noise |      0.88362  |     0.889789 |      -0.00616859  | -0.00940717  | -0.00287155  |                231 |               254 |          15 |
| privacy_constrained_pareto      | nullface                     |      0.88362  |     0.804239 |       0.0793813   |  0.0723261   |  0.0867464   |                408 |                 2 |          90 |
| privacy_constrained_pareto      | diffusion_low_step           |      0.88362  |     0.873475 |       0.010145    |  0.00662706  |  0.0138358   |                290 |               160 |          50 |
| privacy_constrained_pareto      | reverse_personalization      |      0.88362  |     0.770952 |       0.112668    |  0.0983359   |  0.128258    |                348 |                 6 |         146 |
| privacy_verifying_cascade       | blur                         |      0.900863 |     0.883374 |       0.0174886   |  0.0123205   |  0.0232892   |                258 |                83 |         159 |
| privacy_verifying_cascade       | pixelate                     |      0.900863 |     0.796057 |       0.104806    |  0.0931609   |  0.116689    |                410 |                 6 |          84 |
| privacy_verifying_cascade       | solid_mask_black             |      0.900863 |     0.858155 |       0.0427081   |  0.0399617   |  0.0455942   |                456 |                 8 |          36 |
| privacy_verifying_cascade       | layered_blur_downscale_noise |      0.900863 |     0.889789 |       0.0110738   |  0.00860864  |  0.0138192   |                402 |                56 |          42 |
| privacy_verifying_cascade       | nullface                     |      0.900863 |     0.804239 |       0.0966237   |  0.0901566   |  0.103572    |                489 |                 0 |          11 |
| privacy_verifying_cascade       | diffusion_low_step           |      0.900863 |     0.873475 |       0.0273874   |  0.0241298   |  0.0307861   |                397 |                68 |          35 |
| privacy_verifying_cascade       | reverse_personalization      |      0.900863 |     0.770952 |       0.129911    |  0.115701    |  0.145063    |                486 |                14 |           0 |

face-region utility combines crop SSIM, crop LPIPS, landmark geometry, landmark detectability, and background preservation. The independent privacy attacker is FaceNet/VGGFace2 at cosine threshold 0.60. Results apply to the retained output set, declared thresholds, and reviewed 500-image protocol.
