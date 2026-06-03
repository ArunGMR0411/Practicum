# OAPR progressive evaluation figure

Canonical path: `outputs/05_oapr/decision_framework/08_comparison_to_exploratory/`.

![Sensitivity and presence vs localisation multimodal scores](03_sensitivity_and_localisation_scores_figure.png)

## Panel A - Sensitivity (eligible face methods only)

|   privacy_weight |   utility_weight | method                       |    score | eligible_default_policy   |
|-----------------:|-----------------:|:-----------------------------|---------:|:--------------------------|
|             0.4  |             0.5  | layered_blur_downscale_noise | 0.843529 | True                      |
|             0.5  |             0.4  | solid_mask_black             | 0.817124 | True                      |
|             0.55 |             0.35 | solid_mask_black             | 0.806517 | True                      |
|             0.6  |             0.3  | solid_mask_black             | 0.79591  | True                      |
|             0.7  |             0.2  | solid_mask_black             | 0.774696 | True                      |

## Panel B - Presence vs localisation multimodal detection scores

- Presence composite (exploratory): `0.9458`
- Localisation-oriented deployment score: `0.6333` (development-locked screen/text variants; held-out metrics only)
- Screen IoU50 recall: `0.6905`
- Text region recall: `0.7600`
- Combined presence recall: `0.9804`

Interpretation: the drop from ~0.95 to 0.633348 is a **stricter metric**, not a system regression.

