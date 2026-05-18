# Post-Detection Condition Annotation Evaluation

Hypothesis:

> SCR context cues plus post-detection face-box geometry produce stronger condition profiles for anonymisation routing than either SCR-only or box-only rules.

Compared methods:

- `handcrafted_yolo_multiscale__logistic_regression`: existing best pre-detection SCR prediction from raw/image-quality cues plus detector telemetry.
- `post_detection_oracle_reviewed_boxes`: upper bound using reviewed face boxes plus image-quality cues.
- `post_detection_available_handoff_boxes`: practical check on the egocentric stress 500 using the retained anonymisation handoff boxes.
- `hybrid_scr_plus_reviewed_boxes_upper_bound`: SCR context/semantic cues plus reviewed-box geometry.
- `hybrid_scr_plus_available_handoff_boxes`: SCR context/semantic cues plus retained detected-box geometry on the egocentric stress 500.
- `cv_crop_conflict_available_handoff_boxes`: out-of-fold crop/box/conflict telemetry model on the egocentric stress 500.
- `selective_cv_hybrid_available_handoff_boxes`: fold-safe selective hybrid that keeps rule labels unless telemetry improves training-fold F2.
- `final_hybrid_condition_profile_available_handoff_boxes`: earlier label-source policy using the strongest measured source per label family.
- `fixed_policy_detector_telemetry_hybrid_available_handoff_boxes`: faster evidence-derived per-label policy using detector telemetry without exhaustive model search.
- `fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes`: final Step 4 policy that adds a fold-safe multiclass scale layer so mutually exclusive scale labels cannot conflict.

Important boundary:

- The retained handoff boxes are available only for the egocentric stress/anonymisation 500 surface.
- Detector candidate boxes are now retained in `outputs/02_face_detection/11_detector_candidate_box_telemetry/`, enabling detector-disagreement and source-consensus features.
- Detector-disagreement telemetry alone did not beat the earlier final policy on the egocentric stress 500; the winning improvement came from combining detector telemetry with a structured multiclass scale layer.
- The final policy keeps base/rule labels where they remain strongest, uses detector telemetry for selected context labels, and models scale labels as one mutually exclusive class.

## Benchmark

| Method | Protocol | Images | OAPR condition score | Macro F2 | Macro F1 | Micro F2 | Jaccard | Route-eligible labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| post_detection_oracle_reviewed_boxes | baseline_500 | 500 | 0.9953 | 0.9959 | 0.9959 | 0.9955 | 0.9907 | 12 |
| hybrid_scr_plus_reviewed_boxes_upper_bound | baseline_500 | 500 | 0.9253 | 0.9391 | 0.9266 | 0.9324 | 0.8425 | 11 |
| selective_cv_hybrid_reviewed_boxes_upper_bound | baseline_500 | 500 | 0.9070 | 0.9231 | 0.8949 | 0.9351 | 0.8148 | 10 |
| cv_crop_conflict_reviewed_boxes_upper_bound | baseline_500 | 500 | 0.9061 | 0.9225 | 0.8933 | 0.9346 | 0.8132 | 10 |
| selective_cv_detector_disagreement_reviewed_boxes_upper_bound | baseline_500 | 500 | 0.9028 | 0.9163 | 0.8948 | 0.9257 | 0.8209 | 10 |
| cv_detector_disagreement_reviewed_boxes_upper_bound | baseline_500 | 500 | 0.9025 | 0.9161 | 0.8944 | 0.9256 | 0.8209 | 10 |
| handcrafted_yolo_multiscale__logistic_regression | baseline_500 | 500 | 0.7617 | 0.7857 | 0.7320 | 0.8125 | 0.6399 | 7 |
| fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes | egocentric_stress_500 | 500 | 0.9255 | 0.9336 | 0.9268 | 0.9328 | 0.8706 | 12 |
| fixed_policy_detector_telemetry_hybrid_available_handoff_boxes | egocentric_stress_500 | 500 | 0.8821 | 0.8920 | 0.8771 | 0.9003 | 0.8179 | 12 |
| final_hybrid_condition_profile_available_handoff_boxes | egocentric_stress_500 | 500 | 0.8742 | 0.8887 | 0.8626 | 0.8968 | 0.7970 | 12 |
| selective_cv_hybrid_available_handoff_boxes | egocentric_stress_500 | 500 | 0.8652 | 0.8772 | 0.8569 | 0.8883 | 0.7918 | 12 |
| cv_crop_conflict_available_handoff_boxes | egocentric_stress_500 | 500 | 0.8638 | 0.8763 | 0.8548 | 0.8874 | 0.7888 | 12 |
| selective_cv_detector_disagreement_available_handoff_boxes | egocentric_stress_500 | 500 | 0.8584 | 0.8690 | 0.8538 | 0.8792 | 0.7856 | 12 |
| cv_detector_disagreement_available_handoff_boxes | egocentric_stress_500 | 500 | 0.8570 | 0.8679 | 0.8519 | 0.8780 | 0.7835 | 12 |
| selective_cv_hybrid_reviewed_boxes_upper_bound | egocentric_stress_500 | 500 | 0.8353 | 0.8532 | 0.8192 | 0.8645 | 0.7416 | 9 |
| cv_crop_conflict_reviewed_boxes_upper_bound | egocentric_stress_500 | 500 | 0.8344 | 0.8526 | 0.8177 | 0.8641 | 0.7406 | 9 |
| selective_cv_detector_disagreement_reviewed_boxes_upper_bound | egocentric_stress_500 | 500 | 0.8155 | 0.8314 | 0.8051 | 0.8449 | 0.7182 | 9 |
| cv_detector_disagreement_reviewed_boxes_upper_bound | egocentric_stress_500 | 500 | 0.8142 | 0.8303 | 0.8034 | 0.8439 | 0.7157 | 9 |
| hybrid_scr_plus_reviewed_boxes_upper_bound | egocentric_stress_500 | 500 | 0.8029 | 0.8187 | 0.7946 | 0.8295 | 0.7046 | 8 |
| hybrid_scr_plus_available_handoff_boxes | egocentric_stress_500 | 500 | 0.7999 | 0.8154 | 0.7916 | 0.8264 | 0.7030 | 8 |
| handcrafted_yolo_multiscale__logistic_regression | egocentric_stress_500 | 500 | 0.7375 | 0.7640 | 0.7079 | 0.7842 | 0.6085 | 6 |
| post_detection_oracle_reviewed_boxes | egocentric_stress_500 | 500 | 0.6915 | 0.6961 | 0.6932 | 0.7171 | 0.6262 | 7 |
| post_detection_available_handoff_boxes | egocentric_stress_500 | 500 | 0.6884 | 0.6929 | 0.6902 | 0.7138 | 0.6233 | 7 |

## Label-Family Benchmark

| Method | Protocol | Label family | Supported labels | Mean F1 | Mean F2 | Route-eligible labels |
|---|---|---|---:|---:|---:|---:|
| hybrid_scr_plus_reviewed_boxes_upper_bound | baseline_500 | box_geometry | 8 | 1.0000 | 1.0000 | 8 |
| post_detection_oracle_reviewed_boxes | baseline_500 | box_geometry | 8 | 1.0000 | 1.0000 | 8 |
| selective_cv_detector_disagreement_reviewed_boxes_upper_bound | baseline_500 | box_geometry | 8 | 0.9120 | 0.9307 | 7 |
| cv_detector_disagreement_reviewed_boxes_upper_bound | baseline_500 | box_geometry | 8 | 0.9113 | 0.9304 | 7 |
| selective_cv_hybrid_reviewed_boxes_upper_bound | baseline_500 | box_geometry | 8 | 0.9062 | 0.9273 | 7 |
| cv_crop_conflict_reviewed_boxes_upper_bound | baseline_500 | box_geometry | 8 | 0.9038 | 0.9263 | 7 |
| handcrafted_yolo_multiscale__logistic_regression | baseline_500 | box_geometry | 8 | 0.7081 | 0.7700 | 4 |
| post_detection_oracle_reviewed_boxes | baseline_500 | image_quality | 3 | 0.9835 | 0.9835 | 3 |
| cv_crop_conflict_reviewed_boxes_upper_bound | baseline_500 | image_quality | 3 | 0.9298 | 0.9536 | 3 |
| selective_cv_hybrid_reviewed_boxes_upper_bound | baseline_500 | image_quality | 3 | 0.9298 | 0.9536 | 3 |
| cv_detector_disagreement_reviewed_boxes_upper_bound | baseline_500 | image_quality | 3 | 0.9227 | 0.9377 | 3 |
| selective_cv_detector_disagreement_reviewed_boxes_upper_bound | baseline_500 | image_quality | 3 | 0.9227 | 0.9377 | 3 |
| handcrafted_yolo_multiscale__logistic_regression | baseline_500 | image_quality | 3 | 0.8773 | 0.9168 | 3 |
| hybrid_scr_plus_reviewed_boxes_upper_bound | baseline_500 | image_quality | 3 | 0.8773 | 0.9168 | 3 |
| post_detection_oracle_reviewed_boxes | baseline_500 | other | 1 | 1.0000 | 1.0000 | 1 |
| cv_crop_conflict_reviewed_boxes_upper_bound | baseline_500 | other | 1 | 0.6995 | 0.7988 | 0 |
| selective_cv_hybrid_reviewed_boxes_upper_bound | baseline_500 | other | 1 | 0.6995 | 0.7988 | 0 |
| cv_detector_disagreement_reviewed_boxes_upper_bound | baseline_500 | other | 1 | 0.6742 | 0.7362 | 0 |
| selective_cv_detector_disagreement_reviewed_boxes_upper_bound | baseline_500 | other | 1 | 0.6742 | 0.7362 | 0 |
| handcrafted_yolo_multiscale__logistic_regression | baseline_500 | other | 1 | 0.4868 | 0.5188 | 0 |
| hybrid_scr_plus_reviewed_boxes_upper_bound | baseline_500 | other | 1 | 0.4868 | 0.5188 | 0 |
| fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes | egocentric_stress_500 | box_geometry | 7 | 0.9576 | 0.9630 | 7 |
| final_hybrid_condition_profile_available_handoff_boxes | egocentric_stress_500 | box_geometry | 7 | 0.8736 | 0.8919 | 7 |
| fixed_policy_detector_telemetry_hybrid_available_handoff_boxes | egocentric_stress_500 | box_geometry | 7 | 0.8723 | 0.8918 | 7 |
| selective_cv_detector_disagreement_available_handoff_boxes | egocentric_stress_500 | box_geometry | 7 | 0.8707 | 0.8844 | 7 |
| cv_detector_disagreement_available_handoff_boxes | egocentric_stress_500 | box_geometry | 7 | 0.8675 | 0.8825 | 7 |
| selective_cv_hybrid_available_handoff_boxes | egocentric_stress_500 | box_geometry | 7 | 0.8669 | 0.8870 | 7 |
| cv_crop_conflict_available_handoff_boxes | egocentric_stress_500 | box_geometry | 7 | 0.8633 | 0.8855 | 7 |
| selective_cv_hybrid_reviewed_boxes_upper_bound | egocentric_stress_500 | box_geometry | 7 | 0.8630 | 0.8876 | 6 |
| cv_crop_conflict_reviewed_boxes_upper_bound | egocentric_stress_500 | box_geometry | 7 | 0.8604 | 0.8865 | 6 |
| selective_cv_detector_disagreement_reviewed_boxes_upper_bound | egocentric_stress_500 | box_geometry | 7 | 0.8576 | 0.8796 | 6 |
| cv_detector_disagreement_reviewed_boxes_upper_bound | egocentric_stress_500 | box_geometry | 7 | 0.8547 | 0.8778 | 6 |
| hybrid_scr_plus_reviewed_boxes_upper_bound | egocentric_stress_500 | box_geometry | 7 | 0.8457 | 0.8527 | 5 |
| post_detection_oracle_reviewed_boxes | egocentric_stress_500 | box_geometry | 7 | 0.8457 | 0.8527 | 5 |
| hybrid_scr_plus_available_handoff_boxes | egocentric_stress_500 | box_geometry | 7 | 0.8405 | 0.8471 | 5 |
| post_detection_available_handoff_boxes | egocentric_stress_500 | box_geometry | 7 | 0.8405 | 0.8471 | 5 |
| handcrafted_yolo_multiscale__logistic_regression | egocentric_stress_500 | box_geometry | 7 | 0.6971 | 0.7590 | 3 |
| fixed_policy_detector_telemetry_hybrid_available_handoff_boxes | egocentric_stress_500 | image_quality | 3 | 0.8811 | 0.8852 | 3 |
| fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes | egocentric_stress_500 | image_quality | 3 | 0.8811 | 0.8852 | 3 |
| final_hybrid_condition_profile_available_handoff_boxes | egocentric_stress_500 | image_quality | 3 | 0.8398 | 0.8821 | 3 |
| handcrafted_yolo_multiscale__logistic_regression | egocentric_stress_500 | image_quality | 3 | 0.8398 | 0.8821 | 3 |
| hybrid_scr_plus_available_handoff_boxes | egocentric_stress_500 | image_quality | 3 | 0.8398 | 0.8821 | 3 |
| hybrid_scr_plus_reviewed_boxes_upper_bound | egocentric_stress_500 | image_quality | 3 | 0.8398 | 0.8821 | 3 |
| cv_crop_conflict_available_handoff_boxes | egocentric_stress_500 | image_quality | 3 | 0.8326 | 0.8476 | 3 |
| selective_cv_hybrid_available_handoff_boxes | egocentric_stress_500 | image_quality | 3 | 0.8326 | 0.8476 | 3 |
| cv_crop_conflict_reviewed_boxes_upper_bound | egocentric_stress_500 | image_quality | 3 | 0.8310 | 0.8504 | 2 |
| selective_cv_hybrid_reviewed_boxes_upper_bound | egocentric_stress_500 | image_quality | 3 | 0.8310 | 0.8504 | 2 |
| cv_detector_disagreement_reviewed_boxes_upper_bound | egocentric_stress_500 | image_quality | 3 | 0.8211 | 0.8332 | 2 |
| selective_cv_detector_disagreement_reviewed_boxes_upper_bound | egocentric_stress_500 | image_quality | 3 | 0.8211 | 0.8332 | 2 |
| cv_detector_disagreement_available_handoff_boxes | egocentric_stress_500 | image_quality | 3 | 0.8205 | 0.8293 | 3 |
| selective_cv_detector_disagreement_available_handoff_boxes | egocentric_stress_500 | image_quality | 3 | 0.8205 | 0.8293 | 3 |
| post_detection_available_handoff_boxes | egocentric_stress_500 | image_quality | 3 | 0.7996 | 0.7949 | 2 |
| post_detection_oracle_reviewed_boxes | egocentric_stress_500 | image_quality | 3 | 0.7996 | 0.7949 | 2 |
| fixed_policy_detector_telemetry_hybrid_available_handoff_boxes | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.8626 | 0.8750 | 1 |
| fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.8626 | 0.8750 | 1 |
| cv_crop_conflict_available_handoff_boxes | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.8426 | 0.8667 | 1 |
| final_hybrid_condition_profile_available_handoff_boxes | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.8426 | 0.8667 | 1 |
| selective_cv_hybrid_available_handoff_boxes | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.8426 | 0.8667 | 1 |
| cv_detector_disagreement_available_handoff_boxes | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.8290 | 0.8469 | 1 |
| selective_cv_detector_disagreement_available_handoff_boxes | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.8290 | 0.8469 | 1 |
| cv_crop_conflict_reviewed_boxes_upper_bound | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.7914 | 0.7971 | 1 |
| selective_cv_hybrid_reviewed_boxes_upper_bound | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.7914 | 0.7971 | 1 |
| cv_detector_disagreement_reviewed_boxes_upper_bound | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.7469 | 0.7325 | 1 |
| selective_cv_detector_disagreement_reviewed_boxes_upper_bound | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.7469 | 0.7325 | 1 |
| handcrafted_yolo_multiscale__logistic_regression | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.6915 | 0.6814 | 0 |
| hybrid_scr_plus_available_handoff_boxes | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.6915 | 0.6814 | 0 |
| hybrid_scr_plus_reviewed_boxes_upper_bound | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.6915 | 0.6814 | 0 |
| post_detection_available_handoff_boxes | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.0000 | 0.0000 | 0 |
| post_detection_oracle_reviewed_boxes | egocentric_stress_500 | not_safely_inferable_from_boxes | 1 | 0.0000 | 0.0000 | 0 |
| fixed_policy_detector_telemetry_hybrid_available_handoff_boxes | egocentric_stress_500 | other | 1 | 0.9130 | 0.9313 | 1 |
| fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes | egocentric_stress_500 | other | 1 | 0.9130 | 0.9313 | 1 |
| cv_crop_conflict_available_handoff_boxes | egocentric_stress_500 | other | 1 | 0.8737 | 0.9081 | 1 |
| final_hybrid_condition_profile_available_handoff_boxes | egocentric_stress_500 | other | 1 | 0.8737 | 0.9081 | 1 |
| selective_cv_hybrid_available_handoff_boxes | egocentric_stress_500 | other | 1 | 0.8737 | 0.9081 | 1 |
| cv_detector_disagreement_available_handoff_boxes | egocentric_stress_500 | other | 1 | 0.8601 | 0.9022 | 1 |
| selective_cv_detector_disagreement_available_handoff_boxes | egocentric_stress_500 | other | 1 | 0.8601 | 0.9022 | 1 |
| cv_crop_conflict_reviewed_boxes_upper_bound | egocentric_stress_500 | other | 1 | 0.5049 | 0.6771 | 0 |
| selective_cv_hybrid_reviewed_boxes_upper_bound | egocentric_stress_500 | other | 1 | 0.5049 | 0.6771 | 0 |
| cv_detector_disagreement_reviewed_boxes_upper_bound | egocentric_stress_500 | other | 1 | 0.4475 | 0.5872 | 0 |
| selective_cv_detector_disagreement_reviewed_boxes_upper_bound | egocentric_stress_500 | other | 1 | 0.4475 | 0.5872 | 0 |
| handcrafted_yolo_multiscale__logistic_regression | egocentric_stress_500 | other | 1 | 0.4041 | 0.5277 | 0 |
| hybrid_scr_plus_available_handoff_boxes | egocentric_stress_500 | other | 1 | 0.4041 | 0.5277 | 0 |
| hybrid_scr_plus_reviewed_boxes_upper_bound | egocentric_stress_500 | other | 1 | 0.4041 | 0.5277 | 0 |
| post_detection_available_handoff_boxes | egocentric_stress_500 | other | 1 | 0.0000 | 0.0000 | 0 |
| post_detection_oracle_reviewed_boxes | egocentric_stress_500 | other | 1 | 0.0000 | 0.0000 | 0 |

## Verdict

- The enhanced Step 4 hypothesis is **supported**.
- SCR-only reached `0.7375` on the egocentric stress 500; box-only retained handoff labels reached `0.6884`.
- The first hybrid improved the score to `0.7999`.
- Crop-level and conflict telemetry improved the practical score to `0.8652` with the selective CV hybrid.
- The earlier label-source policy improved the practical score to `0.8742`, with all 12 supported labels route-eligible.
- Persisted detector candidate boxes plus the fixed detector-telemetry policy improved the practical score further.
- The final multiclass scale layer crossed the 0.9 target by preventing contradictory scale-label predictions.
- The next anonymisation-routing stage should use `fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes` as the Step 4 condition profile.
