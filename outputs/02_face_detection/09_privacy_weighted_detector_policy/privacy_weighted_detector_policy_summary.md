# Face Detector Policy Summary

Final thesis detector-policy position:

- Privacy requires high recall because a missed face is a privacy failure.
- Uncontrolled false positives damage utility and downstream anonymisation quality.
- The detector stage therefore uses a privacy-weighted detector score rather than pure recall.
- The final single default detector policy is `cv_box_reranker_with_rfdetr_predicted_conditions`.
- Combined 1,000-image default score: `0.9129`; table rows provide category-level overrides and fallback decisions.

Route eligibility:

- Route-eligible manual categories from SCR evidence: `6`.
- Manual categories requiring fallback because SCR is not reliable enough: `9`.

Evidence boundary:

- Manual categories are used for scientific analysis and oracle interpretation.
- Runtime routing uses only SCR-reliable categories.
- Unsupported or uncertain categories use the privacy-weighted RF-DETR-aware reranker fallback.
