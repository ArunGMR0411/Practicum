# Low-Compute Detector Policy Experiment

This experiment tests low-compute refinements after detector hardening. It does not use RF-DETR, full-resolution slicing, or unavailable specialised models.

Compared methods:

- `fixed_fusion_yolo11s1280_scrfd10g`: current best detector-hardening fusion.
- `deployable_category_policy`: category-specific detector selection using predicted Scene-Condition Router labels.
- `oracle_category_policy`: upper-bound category selection using reviewed condition labels.
- `cv_box_reranker_predicted_conditions`: five-fold image-level cross-validated box reranker using detector features and predicted condition labels.
- `cv_box_reranker_oracle_conditions`: same reranker with reviewed condition labels as an upper bound.

Use predicted-condition results for deployable claims. Use oracle-condition rows only as upper-bound analysis.
