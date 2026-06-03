# Operator Improvement Results (area-aware screen + text privacy floor)

Prior adaptive (pre-operator update): privacy 0.8199, utility 0.7005, score 0.8195, util<0.50 = 34/75.

## Development-selected adaptive policy

| predicted_risk_state    | selected_variant                |   development_score |   development_privacy |   development_utility | reason                                                               |
|:------------------------|:--------------------------------|--------------------:|----------------------:|----------------------:|:---------------------------------------------------------------------|
| no_text_screen_risk     | no_action_copy                  |          nan        |            nan        |            nan        | No predicted text/screen boxes; preserve the image.                  |
| screen_present          | text_blur_screen_fill           |            0.798554 |              0.869745 |              0.548996 | Highest development-split measured score; privacy breaks score ties. |
| text_and_screen_present | text_blur_screen_fill           |            0.788615 |              0.876779 |              0.5054   | Highest development-split measured score; privacy breaks score ties. |
| text_present            | text_blur_screen_area_aware_t10 |            0.884864 |              0.811748 |              0.932961 | Highest development-split measured score; privacy breaks score ties. |

## Held-out adaptive vs fixed (top rows)

| split   | policy                                |   n_images |   privacy_score |   utility_score |   runtime_score |   success_score |   multimodal_anonymisation_score |     SSIM |     LPIPS |   non_sensitive_change_fraction |   runtime_seconds |   adaptive_minus_fixed_mean |   difference_ci_low |   difference_ci_high |   adaptive_win_count |   fixed_win_count |   tie_count |
|:--------|:--------------------------------------|-----------:|----------------:|----------------:|----------------:|----------------:|---------------------------------:|---------:|----------:|--------------------------------:|------------------:|----------------------------:|--------------------:|---------------------:|---------------------:|------------------:|------------:|
| test    | adaptive_multimodal_policy            |         75 |        0.867955 |        0.663351 |        0.991106 |               1 |                         0.832093 | 0.912218 | 0.0797116 |                       0.0409181 |          0.667014 |                 0           |         0           |          0           |                    0 |                 0 |          75 |
| test    | fixed_text_blur_screen_fill           |         75 |        0.867955 |        0.663351 |        0.988849 |               1 |                         0.831868 | 0.912218 | 0.0797116 |                       0.0409181 |          0.83634  |                 0.000225768 |         0.000145676 |          0.000310163 |                   25 |                 2 |          48 |
| test    | fixed_text_fill_screen_fill           |         75 |        0.886118 |        0.626006 |        0.989488 |               1 |                         0.82981  | 0.906254 | 0.0867913 |                       0.0437524 |          0.788372 |                 0.00228377  |        -0.00486612  |          0.00860088  |                   59 |                16 |           0 |
| test    | fixed_text_blur_screen_area_aware_t15 |         75 |        0.819685 |        0.672356 |        0.983159 |               1 |                         0.809865 | 0.930228 | 0.0679849 |                       0.0367184 |          1.26304  |                 0.022228    |         0.0116224   |          0.0339177   |                   43 |                32 |           0 |
| test    | fixed_text_fill_screen_area_aware_t15 |         75 |        0.837848 |        0.63501  |        0.983544 |               1 |                         0.807782 | 0.924262 | 0.0751068 |                       0.0395527 |          1.23419  |                 0.0243118   |         0.0116428   |          0.0372198   |                   61 |                14 |           0 |
| test    | fixed_text_blur_screen_area_aware_t10 |         75 |        0.810572 |        0.673821 |        0.981725 |               1 |                         0.805605 | 0.933158 | 0.0661097 |                       0.0357966 |          1.37062  |                 0.0264889   |         0.0144853   |          0.0394752   |                   40 |                23 |          12 |
| test    | fixed_text_blur_screen_area_aware_t08 |         75 |        0.806465 |        0.674604 |        0.980685 |               1 |                         0.803682 | 0.933792 | 0.065714  |                       0.0357296 |          1.44861  |                 0.0284116   |         0.0162593   |          0.0414019   |                   50 |                25 |           0 |
| test    | fixed_text_fill_screen_area_aware_t10 |         75 |        0.828735 |        0.636475 |        0.982284 |               1 |                         0.803538 | 0.927191 | 0.0732339 |                       0.038631  |          1.32869  |                 0.0285553   |         0.0146053   |          0.042758    |                   60 |                15 |           0 |
| test    | fixed_text_fill_screen_area_aware_t08 |         75 |        0.824628 |        0.637257 |        0.981744 |               1 |                         0.801665 | 0.927825 | 0.0728381 |                       0.038564  |          1.36918  |                 0.030428    |         0.0163726   |          0.0447597   |                   61 |                14 |           0 |
| test    | fixed_text_blur_screen_area_aware_t05 |         75 |        0.786857 |        0.678006 |        0.979783 |               1 |                         0.794809 | 0.936775 | 0.0640221 |                       0.0346184 |          1.51625  |                 0.0372847   |         0.0244844   |          0.0508599   |                   55 |                20 |           0 |
| test    | fixed_text_fill_screen_area_aware_t05 |         75 |        0.805021 |        0.639228 |        0.980596 |               1 |                         0.792338 | 0.930809 | 0.0711426 |                       0.0374527 |          1.45528  |                 0.0397552   |         0.024813    |          0.0544264   |                   61 |                14 |           0 |
| test    | fixed_text_pixelate_screen_pixelate   |         75 |        0.524904 |        0.850449 |        0.987649 |               1 |                         0.716351 | 0.986905 | 0.0150314 |                       0.0162057 |          0.926301 |                 0.115742    |         0.0827807   |          0.149178    |                   53 |                22 |           0 |

## Held-out residual flags

- utility_below_050: 37/75
- residual_text_readability: 6/75
- insufficient_screen_obscuration: 1/75
- missed_text: 2/75
- missed_screen: 0/75

Text privacy floor used: 0.70
Area-aware thresholds grid: [0.05, 0.08, 0.1, 0.15]
