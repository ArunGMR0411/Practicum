# Full Anonymisation Policy

## Method distributions

| objective_mode        | selected_method              |   image_count |
|:----------------------|:-----------------------------|--------------:|
| balanced_high_compute | layered_blur_downscale_noise |           286 |
| balanced_high_compute | no_action_copy               |           133 |
| balanced_high_compute | solid_mask_black             |            81 |
| balanced_standard     | layered_blur_downscale_noise |           286 |
| balanced_standard     | no_action_copy               |           133 |
| balanced_standard     | solid_mask_black             |            81 |
| failure_avoidance     | layered_blur_downscale_noise |           286 |
| failure_avoidance     | no_action_copy               |           133 |
| failure_avoidance     | solid_mask_black             |            81 |
| privacy_first         | layered_blur_downscale_noise |           244 |
| privacy_first         | no_action_copy               |           133 |
| privacy_first         | solid_mask_black             |           123 |
| runtime_practical     | layered_blur_downscale_noise |           286 |
| runtime_practical     | no_action_copy               |           133 |
| runtime_practical     | solid_mask_black             |            81 |
| utility_priority      | layered_blur_downscale_noise |           367 |
| utility_priority      | no_action_copy               |           133 |

## Mode summary

| objective_mode        |   n_input_frames |   n_success |   n_failure |   category_policy_score_mean |   actual_privacy_score_mean |   actual_utility_score_mean |   actual_runtime_score_mean |   actual_objective_score_mean |
|:----------------------|-----------------:|------------:|------------:|-----------------------------:|----------------------------:|----------------------------:|----------------------------:|------------------------------:|
| balanced_high_compute |              500 |         500 |           0 |                     0.956278 |                    0.990495 |                    0.891399 |                    0.934989 |                      0.956166 |
| balanced_standard     |              500 |         500 |           0 |                     0.956278 |                    0.990495 |                    0.891399 |                    0.934989 |                      0.956166 |
| failure_avoidance     |              500 |         500 |           0 |                     0.956278 |                    0.990495 |                    0.891399 |                    0.934989 |                      0.956166 |
| privacy_first         |              500 |         500 |           0 |                     0.972916 |                    0.992812 |                    0.880225 |                    0.935365 |                      0.97377  |
| runtime_practical     |              500 |         500 |           0 |                     0.95504  |                    0.990495 |                    0.891399 |                    0.934989 |                      0.954975 |
| utility_priority      |              500 |         500 |           0 |                     0.9417   |                    0.982995 |                    0.903614 |                    0.934509 |                      0.940156 |

Every eligible method is scored before selection. A method can receive zero selections when another eligible method has a higher category/objective score.
