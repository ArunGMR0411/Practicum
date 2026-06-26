# Face Detection Category Comparison

| category                     | adaptive_detector                                |   adaptive_oapr_detector_score | best_competitor                                         |   best_competitor_score |   adaptive_margin |
|:-----------------------------|:-------------------------------------------------|-------------------------------:|:--------------------------------------------------------|------------------------:|------------------:|
| downward_egocentric_view     | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.889279 | fusion_rfdetr_yolo11s_scrfd10g                          |                0.894121 |      -0.0048418   |
| edge_or_partial_face         | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.911708 | fusion_rfdetr_scrfd10g                                  |                0.914495 |      -0.00278621  |
| high_clutter                 | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.936518 | cv_box_reranker_with_rfdetr_sliced_predicted_conditions |                0.9364   |       0.000118088 |
| large_face                   | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.848181 | fusion_rfdetr_scrfd10g                                  |                0.869292 |      -0.0211115   |
| low_light_or_dim             | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.8883   | fusion_rfdetr_scrfd10g                                  |                0.888087 |       0.000212637 |
| medium_face                  | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.933299 | cv_box_reranker_with_rfdetr_sliced_predicted_conditions |                0.930818 |       0.00248121  |
| mixed_scale_face             | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.892399 | fusion_rfdetr_yolo11s_scrfd10g                          |                0.902482 |      -0.0100833   |
| motion_blur_or_low_sharpness | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.83607  | fusion_rfdetr_scrfd10g                                  |                0.850231 |      -0.0141609   |
| multi_face                   | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.922747 | fusion_rfdetr_scrfd10g                                  |                0.922848 |      -0.000100928 |
| no_face                      | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.936396 | cv_box_reranker_with_sliced_yolo_predicted_conditions   |                0.961131 |      -0.024735    |
| outdoor_or_vehicle_scene     | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.955388 | cv_box_reranker_with_rfdetr_sliced_predicted_conditions |                0.96881  |      -0.0134216   |
| profile_or_occluded_face     | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.93755  | fusion_rfdetr_scrfd10g                                  |                0.936847 |       0.00070277  |
| single_face                  | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.791742 | fusion_rfdetr_yolo11s_scrfd10g                          |                0.81131  |      -0.0195675   |
| small_face                   | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.934694 | cv_box_reranker_with_rfdetr_sliced_predicted_conditions |                0.936263 |      -0.00156856  |
| very_small_or_distant_face   | cv_box_reranker_with_rfdetr_predicted_conditions |                       0.874271 | fusion_rfdetr_yolo11s_scrfd10g                          |                0.885362 |      -0.0110914   |
