# Sliced Detector Policy Scores

| Protocol | Method | OAPR score | F1 | Precision | Recall | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01_baseline_500 | fusion_rfdetr_yolo11s_scrfd10g | 0.8996 | 0.8857 | 0.8619 | 0.9108 | 1205 | 193 | 118 |
| 01_baseline_500 | cv_box_reranker_with_rfdetr_sliced_predicted_conditions | 0.8959 | 0.9088 | 0.9338 | 0.8851 | 1171 | 83 | 152 |
| 01_baseline_500 | cv_box_reranker_with_rfdetr_predicted_conditions | 0.8955 | 0.9061 | 0.9265 | 0.8866 | 1173 | 93 | 150 |
| 01_baseline_500 | fusion_rfdetr_scrfd10g | 0.8948 | 0.8946 | 0.8943 | 0.8949 | 1184 | 140 | 139 |
| 01_baseline_500 | cv_box_reranker_with_sliced_yolo_predicted_conditions | 0.8943 | 0.9079 | 0.9344 | 0.8828 | 1168 | 82 | 155 |
| 01_baseline_500 | fixed_fusion_yolo11s1280_scrfd10g | 0.8926 | 0.8989 | 0.9108 | 0.8874 | 1174 | 115 | 149 |
| 01_baseline_500 | rfdetr_medium_face_030 | 0.8876 | 0.8933 | 0.9040 | 0.8828 | 1168 | 124 | 155 |
| 01_baseline_500 | fusion_rfdetr_sliced_yolo11s_scrfd10g | 0.8466 | 0.7492 | 0.6334 | 0.9169 | 1213 | 702 | 110 |
| 01_baseline_500 | fusion_sliced_yolo11s_yolo11s_scrfd10g | 0.8411 | 0.7627 | 0.6624 | 0.8987 | 1189 | 606 | 134 |
| 01_baseline_500 | fusion_sliced_yolo11s_scrfd10g | 0.8266 | 0.7602 | 0.6715 | 0.8760 | 1159 | 567 | 164 |
| 01_baseline_500 | sliced_yolo11s_widerface_1280 | 0.7499 | 0.6924 | 0.6145 | 0.7929 | 1049 | 658 | 274 |
| 02_egocentric_stress_500 | cv_box_reranker_with_rfdetr_predicted_conditions | 0.9309 | 0.9157 | 0.8898 | 0.9431 | 1211 | 150 | 73 |
| 02_egocentric_stress_500 | cv_box_reranker_with_rfdetr_sliced_predicted_conditions | 0.9305 | 0.9156 | 0.8904 | 0.9424 | 1210 | 149 | 74 |
| 02_egocentric_stress_500 | fusion_rfdetr_scrfd10g | 0.9276 | 0.9000 | 0.8554 | 0.9494 | 1219 | 206 | 65 |
| 02_egocentric_stress_500 | cv_box_reranker_with_sliced_yolo_predicted_conditions | 0.9252 | 0.9156 | 0.8987 | 0.9330 | 1198 | 135 | 86 |
| 02_egocentric_stress_500 | fixed_fusion_yolo11s1280_scrfd10g | 0.9225 | 0.9013 | 0.8664 | 0.9393 | 1206 | 186 | 78 |
| 02_egocentric_stress_500 | fusion_rfdetr_yolo11s_scrfd10g | 0.9185 | 0.8722 | 0.8033 | 0.9540 | 1225 | 300 | 59 |
| 02_egocentric_stress_500 | rfdetr_medium_face_030 | 0.8993 | 0.8804 | 0.8489 | 0.9143 | 1174 | 209 | 110 |
| 02_egocentric_stress_500 | fusion_rfdetr_sliced_yolo11s_scrfd10g | 0.8647 | 0.7389 | 0.6033 | 0.9533 | 1224 | 805 | 60 |
| 02_egocentric_stress_500 | fusion_sliced_yolo11s_yolo11s_scrfd10g | 0.8626 | 0.7544 | 0.6300 | 0.9400 | 1207 | 709 | 77 |
| 02_egocentric_stress_500 | fusion_sliced_yolo11s_scrfd10g | 0.8588 | 0.7637 | 0.6490 | 0.9276 | 1191 | 644 | 93 |
| 02_egocentric_stress_500 | sliced_yolo11s_widerface_1280 | 0.7679 | 0.6899 | 0.5929 | 0.8248 | 1059 | 727 | 225 |
| combined_1000 | cv_box_reranker_with_rfdetr_predicted_conditions | 0.9129 | 0.9110 | 0.9075 | 0.9145 | 2384 | 243 | 223 |
| combined_1000 | cv_box_reranker_with_rfdetr_sliced_predicted_conditions | 0.9128 | 0.9123 | 0.9112 | 0.9133 | 2381 | 232 | 226 |
| combined_1000 | fusion_rfdetr_scrfd10g | 0.9109 | 0.8973 | 0.8741 | 0.9217 | 2403 | 346 | 204 |
| combined_1000 | cv_box_reranker_with_sliced_yolo_predicted_conditions | 0.9094 | 0.9118 | 0.9160 | 0.9076 | 2366 | 217 | 241 |
| combined_1000 | fusion_rfdetr_yolo11s_scrfd10g | 0.9087 | 0.8788 | 0.8313 | 0.9321 | 2430 | 493 | 177 |
| combined_1000 | fixed_fusion_yolo11s1280_scrfd10g | 0.9072 | 0.9002 | 0.8877 | 0.9129 | 2380 | 301 | 227 |
| combined_1000 | rfdetr_medium_face_030 | 0.8932 | 0.8868 | 0.8755 | 0.8984 | 2342 | 333 | 265 |
| combined_1000 | fusion_rfdetr_sliced_yolo11s_scrfd10g | 0.8554 | 0.7440 | 0.6179 | 0.9348 | 2437 | 1507 | 170 |
| combined_1000 | fusion_sliced_yolo11s_yolo11s_scrfd10g | 0.8516 | 0.7585 | 0.6456 | 0.9191 | 2396 | 1315 | 211 |
| combined_1000 | fusion_sliced_yolo11s_scrfd10g | 0.8424 | 0.7620 | 0.6599 | 0.9014 | 2350 | 1211 | 257 |
| combined_1000 | sliced_yolo11s_widerface_1280 | 0.7587 | 0.6911 | 0.6035 | 0.8086 | 2108 | 1385 | 499 |
