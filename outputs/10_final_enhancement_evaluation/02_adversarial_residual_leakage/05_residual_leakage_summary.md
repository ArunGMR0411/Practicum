# Adversarial residual-leakage evaluation

Privacy is checked **after** anonymisation/redaction beyond headline composites.

## Attack surfaces

1. **Face re-identification residual** - max(AdaFace, ArcFace) Re-ID rates by method
   (`04_face_reid_residual_by_method.csv`).
2. **OCR / text recovery** - residual readability + missed text detection on held-out
   multimodal adaptive outputs.
3. **Screen residual risk** - missed screen presence + insufficient obscuration.
4. **VLM-style privacy questions (proxy)** - structured yes/no inference risks for
   identity, text/document, screen UI, and activity context from residual flags
   (no external VLM API dependency).

## Held-out multimodal adaptive summary

| split   | policy                     |   n_images | face_reid_attack_surface             |   ocr_text_recovery_flag_rate |   missed_text_detection_rate |   missed_screen_detection_rate |   insufficient_screen_obscuration_rate |   utility_below_050_rate |   any_residual_risk_flag_rate |   mean_privacy_score |   mean_utility_score |   mean_multimodal_score |   privacy_near_zero_rate |
|:--------|:---------------------------|-----------:|:-------------------------------------|------------------------------:|-----------------------------:|-------------------------------:|---------------------------------------:|-------------------------:|------------------------------:|---------------------:|---------------------:|------------------------:|-------------------------:|
| test    | adaptive_multimodal_policy |         75 | see face tables (500-frame protocol) |                          0.08 |                    0.0266667 |                              0 |                              0.0133333 |                 0.493333 |                          0.56 |             0.867955 |             0.663351 |                0.832047 |                0.0666667 |

## VLM-proxy summary

| split   |   n_images |   vlm_proxy_any_sensitive_inference_rate |   vlm_proxy_identity_rate |   vlm_proxy_text_document_rate |   vlm_proxy_screen_rate |   vlm_proxy_activity_context_rate | note                                                                                                                                                      |
|:--------|-----------:|-----------------------------------------:|--------------------------:|-------------------------------:|------------------------:|----------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------|
| test    |         75 |                                 0.106667 |                 0.0533333 |                      0.0933333 |               0.0133333 |                         0.0666667 | Rule-based adversarial privacy-question proxy over residual flags and privacy scores; not a commercial VLM API evaluation. Complements OCR/Re-ID metrics. |

## Interpretation

> Residual leakage is measurable after processing. Perfect removal is not claimed.
> Adaptive multimodal privacy remains bounded by residual text readability,
> rare detector-incapable text, and low-IoU screen cases. Face defaults that pass
> gates show low residual Re-ID under the protocol attackers.
