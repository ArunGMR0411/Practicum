# Low-Compute Detector Policy Scores

| Protocol | Method | OAPR score | F1 | Precision | Recall | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01_baseline_500 | cv_box_reranker_oracle_conditions | 0.8966 | 0.9031 | 0.9154 | 0.8912 | 1179 | 109 | 144 |
| 01_baseline_500 | oracle_category_policy | 0.8957 | 0.9049 | 0.9223 | 0.8881 | 1175 | 99 | 148 |
| 01_baseline_500 | fixed_fusion_yolo11s1280_scrfd10g | 0.8926 | 0.8989 | 0.9108 | 0.8874 | 1174 | 115 | 149 |
| 01_baseline_500 | cv_box_reranker_predicted_conditions | 0.8911 | 0.9090 | 0.9446 | 0.8760 | 1159 | 68 | 164 |
| 01_baseline_500 | deployable_category_policy | 0.8907 | 0.9037 | 0.9290 | 0.8798 | 1164 | 89 | 159 |
| 02_egocentric_stress_500 | cv_box_reranker_oracle_conditions | 0.9338 | 0.9290 | 0.9205 | 0.9377 | 1204 | 104 | 80 |
| 02_egocentric_stress_500 | cv_box_reranker_predicted_conditions | 0.9315 | 0.9267 | 0.9182 | 0.9354 | 1201 | 107 | 83 |
| 02_egocentric_stress_500 | oracle_category_policy | 0.9252 | 0.9136 | 0.8935 | 0.9346 | 1200 | 143 | 84 |
| 02_egocentric_stress_500 | deployable_category_policy | 0.9240 | 0.9128 | 0.8934 | 0.9330 | 1198 | 143 | 86 |
| 02_egocentric_stress_500 | fixed_fusion_yolo11s1280_scrfd10g | 0.9225 | 0.9013 | 0.8664 | 0.9393 | 1206 | 186 | 78 |
| combined_1000 | cv_box_reranker_oracle_conditions | 0.9149 | 0.9160 | 0.9180 | 0.9141 | 2383 | 213 | 224 |
| combined_1000 | cv_box_reranker_predicted_conditions | 0.9110 | 0.9179 | 0.9310 | 0.9053 | 2360 | 175 | 247 |
| combined_1000 | oracle_category_policy | 0.9102 | 0.9093 | 0.9075 | 0.9110 | 2375 | 242 | 232 |
| combined_1000 | fixed_fusion_yolo11s1280_scrfd10g | 0.9072 | 0.9002 | 0.8877 | 0.9129 | 2380 | 301 | 227 |
| combined_1000 | deployable_category_policy | 0.9070 | 0.9083 | 0.9106 | 0.9060 | 2362 | 232 | 245 |
