# Privacy-Weighted Face Detector Policy

This table connects manual condition evidence to deployable detector-policy decisions.

Detector objective:

`OAPR detector score = 0.65 * recall + 0.25 * F1 + 0.10 * precision` for face-positive categories.

For no-face categories, the score is specificity because false positives damage utility.

Policy rule:

- Use all manual categories for thesis/oracle analysis.
- Use only SCR route-eligible categories for runtime category routing.
- Use `cv_box_reranker_with_rfdetr_predicted_conditions` as the privacy-weighted fallback/default policy.
- Do not route runtime images by weak categories unless a later SCR model proves them reliable.

| Manual category | Support | Best by F1 | Best by recall | Best by OAPR score | OAPR score | OAPR margin vs 2nd | SCR can predict | Runtime action | Final policy |
|---|---:|---|---|---|---:|---:|---|---|---|
| all_images | 1000 | scrfd_10g_current_640 (0.9152) | fusion_rfdetr_sliced_yolo11s_scrfd10g (0.9348) | cv_box_reranker_with_rfdetr_predicted_conditions | 0.9129 | 0.0001 | not_applicable_global | global_default | cv_box_reranker_with_rfdetr_predicted_conditions |
| downward_egocentric_view | 242 | cv_box_reranker_with_sliced_yolo_predicted_conditions (0.9060) | fusion_rfdetr_sliced_yolo11s_scrfd10g (0.9169) | fusion_rfdetr_yolo11s_scrfd10g | 0.8941 | 0.0001 | fallback_required | fallback_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| edge_or_partial_face | 376 | cv_box_reranker_with_rfdetr_sliced_predicted_conditions (0.9163) | fusion_rfdetr_sliced_yolo11s_scrfd10g (0.9331) | fusion_rfdetr_scrfd10g | 0.9145 | 0.0003 | route_eligible | fallback_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| high_clutter | 360 | cv_box_reranker_with_sliced_yolo_predicted_conditions (0.9345) | fusion_rfdetr_yolo11s_scrfd10g (0.9523) | cv_box_reranker_with_rfdetr_predicted_conditions | 0.9365 | 0.0001 | route_eligible | route_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| large_face | 42 | yolo11n_pose_widerface_640 (0.8777) | fusion_rfdetr_scrfd10g (0.8667) | fusion_rfdetr_scrfd10g | 0.8693 | 0.0077 | fallback_required | fallback_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| low_light_or_dim | 199 | cv_box_reranker_with_rfdetr_sliced_predicted_conditions (0.8927) | fusion_rfdetr_sliced_yolo11s_scrfd10g (0.9083) | cv_box_reranker_with_rfdetr_predicted_conditions | 0.8883 | 0.0002 | route_eligible | route_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| medium_face | 223 | scrfd_10g_current_640 (0.9353) | fusion_rfdetr_yolo11s_scrfd10g (0.9485) | cv_box_reranker_with_rfdetr_predicted_conditions | 0.9333 | 0.0025 | fallback_required | fallback_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| mixed_scale_face | 171 | cv_box_reranker_with_rfdetr_sliced_predicted_conditions (0.9071) | fusion_rfdetr_sliced_yolo11s_scrfd10g (0.9198) | fusion_rfdetr_yolo11s_scrfd10g | 0.9025 | 0.0051 | fallback_required | fallback_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| motion_blur_or_low_sharpness | 194 | cv_box_reranker_with_rfdetr_predicted_conditions (0.8471) | fusion_rfdetr_sliced_yolo11s_scrfd10g (0.8776) | fusion_rfdetr_scrfd10g | 0.8502 | 0.0026 | route_eligible | route_to_category_winner | fusion_rfdetr_scrfd10g |
| multi_face | 556 | scrfd_10g_current_640 (0.9257) | fusion_rfdetr_sliced_yolo11s_scrfd10g (0.9411) | fusion_rfdetr_scrfd10g | 0.9228 | 0.0001 | route_eligible | fallback_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| no_face | 283 | not applicable | not applicable | scrfd_10g_current_640 | 0.9682 | 0.0071 | route_eligible | route_to_category_winner | scrfd_10g_current_640 |
| outdoor_or_vehicle_scene | 23 | yolo8s_lindevs_640 (0.9630) | yolo8s_lindevs_640 (1.0000) | yolo8s_lindevs_640 | 0.9836 | 0.0000 | fallback_required | fallback_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| profile_or_occluded_face | 206 | scrfd_10g_current_640 (0.9403) | fusion_rfdetr_yolo11s_scrfd10g (0.9559) | cv_box_reranker_with_rfdetr_predicted_conditions | 0.9375 | 0.0007 | fallback_required | fallback_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| single_face | 161 | cv_box_reranker_with_sliced_yolo_predicted_conditions (0.7923) | fusion_rfdetr_yolo11s_scrfd10g (0.8571) | fusion_rfdetr_yolo11s_scrfd10g | 0.8113 | 0.0093 | fallback_required | fallback_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| small_face | 229 | scrfd_10g_current_640 (0.9317) | fusion_rfdetr_yolo11s_scrfd10g (0.9533) | cv_box_reranker_with_rfdetr_sliced_predicted_conditions | 0.9363 | 0.0016 | fallback_required | fallback_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
| very_small_or_distant_face | 52 | cv_box_reranker_with_rfdetr_predicted_conditions (0.8718) | fusion_rfdetr_yolo11s_scrfd10g (0.9175) | fusion_rfdetr_yolo11s_scrfd10g | 0.8854 | 0.0111 | fallback_required | fallback_to_privacy_weighted_reranker | cv_box_reranker_with_rfdetr_predicted_conditions |
