# Adaptive Full Pipeline Comparison

This pack compares adaptive and fixed methods separately for face detection, multimodal detection, face anonymisation, and multimodal anonymisation.

## Face Detection

The adaptive RF-DETR-aware reranker scored `0.9129` on the combined 1,000-image reviewed protocol. The highest score in the table is `cv_box_reranker_with_rfdetr_predicted_conditions` at `0.9129`. The adaptive method is therefore the strongest measured policy by a very small margin only if it is the highest row; otherwise the table is the authority and no dominance claim is made.

## Multimodal Detection

Canonical multimodal evidence is the reviewed 250-image region-level protocol (`outputs/04_multimodal_privacy/01_multimodal_250_evidence/`). On the held-out 75-image split, the selected combined text/screen policy (`craft_recall_4k+yolo11_union+text_cluster_screen_hyp`) reaches precision `0.8333`, recall `0.9804`, F1 `0.9009`, and OAPR multimodal score `0.9458`. Region-level text precision remains low because environmental text is proposed; image-level presence is not treated as perfect localisation.

## Face Anonymisation

The materialised RiDDLE-heavy policy ablation scored `0.9070` versus `0.8898` for fixed layered obfuscation. It produced all 500 outputs and used deterministic fallback for predicted RiDDLE artifacts. A subsequent uniform visual-quality investigation found that the grouped gate recalled only `11/14` reviewed artifacts and did not reliably cover pose/gaze failures. The current visual-safe runtime policy therefore uses 286 layered routes, 81 solid-mask routes, and 133 reviewed no-face copy-through routes; it completes `500/500` routes with zero generative selections and zero failures.

## Multimodal Anonymisation

Adaptive risk-state routing is measured end-to-end on predicted boxes over the held-out 75-image split of the reviewed 250-image protocol. Adaptive scored privacy `0.8680`, utility `0.6634`, and objective `0.8320`. The strongest fixed policy was `fixed_text_blur_screen_fill` at `0.8319` (privacy `0.8680`, utility `0.6634`). Adaptive-minus-fixed mean difference was `+0.0000` with 95% bootstrap CI `[0.0000, 0.0000]`, so superiority over that strongest fixed policy is not statistically established. Screen privacy is measured as region obscuration, not semantic leakage elimination.

## Claim Boundary

The adaptive pipeline is evidence-supported and objective-aware. It must not be described as globally superior to every fixed method. A method-by-method and category-by-category claim is required.

Canonical multimodal sources: `outputs/04_multimodal_privacy/01_multimodal_250_evidence/11_rq3_final_summary.md` and the machine-readable tables in the same directory.

Canonical source for the current 500-frame face route: `outputs/03_anonymisation/16_visual_quality_hardening/04_final_visual_safe_policy.csv`.

The RiDDLE-heavy materialised comparison remains available at `outputs/03_anonymisation/15_materialised_adaptive_policy/04_final_policy_routing_log.csv` as quantitative ablation evidence.
