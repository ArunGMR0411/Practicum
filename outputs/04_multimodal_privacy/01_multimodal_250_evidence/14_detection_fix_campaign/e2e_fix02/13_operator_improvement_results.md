# Operator Improvement Results (area-aware screen + text privacy floor)

Prior adaptive (pre-operator update): privacy 0.8199, utility 0.7005, score 0.8195, util<0.50 = 34/75.

## Development-selected adaptive policy

| predicted_risk_state    | selected_variant      |   development_score |   development_privacy |   development_utility | reason                                                               |
|:------------------------|:----------------------|--------------------:|----------------------:|----------------------:|:---------------------------------------------------------------------|
| no_text_screen_risk     | no_action_copy        |          nan        |            nan        |            nan        | No predicted text/screen boxes; preserve the image.                  |
| screen_present          | text_fill_screen_fill |            0.798778 |              0.869745 |              0.548996 | Highest development-split measured score; privacy breaks score ties. |
| text_and_screen_present | text_blur_screen_fill |            0.788922 |              0.876779 |              0.5054   | Highest development-split measured score; privacy breaks score ties. |
| text_present            | text_blur_screen_blur |            0.88489  |              0.811748 |              0.932961 | Highest development-split measured score; privacy breaks score ties. |

## Held-out adaptive vs fixed (top rows)

| split   | policy                              |   n_images |   privacy_score |   utility_score |   runtime_score |   success_score |   multimodal_anonymisation_score |     SSIM |     LPIPS |   non_sensitive_change_fraction |   runtime_seconds |   adaptive_minus_fixed_mean |   difference_ci_low |   difference_ci_high |   adaptive_win_count |   fixed_win_count |   tie_count |
|:--------|:------------------------------------|-----------:|----------------:|----------------:|----------------:|----------------:|---------------------------------:|---------:|----------:|--------------------------------:|------------------:|----------------------------:|--------------------:|---------------------:|---------------------:|------------------:|------------:|
| test    | adaptive_multimodal_policy          |         75 |        0.854637 |        0.668944 |        0.992854 |               1 |                         0.827287 | 0.91273  | 0.079178  |                       0.0407034 |          0.535974 |                 0           |         0           |          0           |                    0 |                 0 |          75 |
| test    | fixed_text_blur_screen_fill         |         75 |        0.854637 |        0.668944 |        0.991432 |               1 |                         0.827145 | 0.91273  | 0.079178  |                       0.0407034 |          0.64257  |                 0.000142128 |         8.27009e-05 |          0.000205968 |                   31 |                10 |          34 |
| test    | fixed_text_fill_screen_fill         |         75 |        0.8728   |        0.6316   |        0.99223  |               1 |                         0.825103 | 0.906766 | 0.0862574 |                       0.0435377 |          0.582713 |                 0.00218383  |        -0.00518677  |          0.00846459  |                   55 |                 7 |          13 |
| test    | fixed_text_pixelate_screen_pixelate |         75 |        0.52011  |        0.851632 |        0.990728 |               1 |                         0.714617 | 0.986962 | 0.014916  |                       0.0161286 |          0.69543  |                 0.11267     |         0.0798765   |          0.146366    |                   52 |                23 |           0 |
| test    | fixed_text_pixelate_screen_blur     |         75 |        0.570848 |        0.727831 |        0.98091  |               1 |                         0.701865 | 0.971838 | 0.0460646 |                       0.0236774 |          1.43173  |                 0.125422    |         0.0935319   |          0.15842     |                   53 |                22 |           0 |
| test    | fixed_text_blur_screen_blur         |         75 |        0.574942 |        0.718927 |        0.980334 |               1 |                         0.701183 | 0.971409 | 0.047481  |                       0.0242508 |          1.47492  |                 0.126104    |         0.0945052   |          0.15885     |                   54 |                 8 |          13 |
| test    | fixed_text_fill_screen_blur         |         75 |        0.593105 |        0.675596 |        0.980969 |               1 |                         0.697328 | 0.965442 | 0.054647  |                       0.0270851 |          1.4273   |                 0.129959    |         0.0971659   |          0.163674    |                   65 |                10 |           0 |

## Held-out residual flags

- utility_below_050: 37/75
- residual_text_readability: 6/75
- insufficient_screen_obscuration: 2/75
- missed_text: 2/75
- missed_screen: 1/75

Text privacy floor used: 0.70
Area-aware thresholds grid: [0.05, 0.08, 0.1, 0.15]
