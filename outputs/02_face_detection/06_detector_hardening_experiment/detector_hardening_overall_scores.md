# Face Detector Hardening Overall Scores

| Protocol | Model | OAPR score | F1 | Precision | Recall | TP | FP | FN | Predictions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_baseline_500 | fusion_yolo11s1280_scrfd10g_agreement | 0.8926 | 0.8989 | 0.9108 | 0.8874 | 1174 | 115 | 149 | 1289 |
| 01_baseline_500 | yolo11s_widerface_640 | 0.8893 | 0.9041 | 0.9332 | 0.8768 | 1160 | 83 | 163 | 1243 |
| 01_baseline_500 | fusion_yolo11s_scrfd10g_agreement | 0.8887 | 0.8976 | 0.9145 | 0.8813 | 1166 | 109 | 157 | 1275 |
| 01_baseline_500 | yolo11s_widerface_1280 | 0.8851 | 0.8995 | 0.9277 | 0.8730 | 1155 | 90 | 168 | 1245 |
| 01_baseline_500 | fusion_yolo8s_yolo11s_scrfd10g_agreement | 0.8842 | 0.8804 | 0.8735 | 0.8874 | 1174 | 170 | 149 | 1344 |
| 01_baseline_500 | fusion_yolo8s_scrfd10g_agreement | 0.8833 | 0.8866 | 0.8927 | 0.8806 | 1165 | 140 | 158 | 1305 |
| 01_baseline_500 | dsfd_1920 | 0.8818 | 0.8951 | 0.9209 | 0.8707 | 1152 | 99 | 171 | 1251 |
| 01_baseline_500 | yolo8s_widerface_repo_640 | 0.8812 | 0.8847 | 0.8911 | 0.8783 | 1162 | 142 | 161 | 1304 |
| 01_baseline_500 | yolo8s_lindevs_640 | 0.8805 | 0.8895 | 0.9066 | 0.8730 | 1155 | 119 | 168 | 1274 |
| 01_baseline_500 | yolo8s_lindevs_1280 | 0.8797 | 0.8841 | 0.8922 | 0.8760 | 1159 | 140 | 164 | 1299 |
| 01_baseline_500 | yolo11n_pose_widerface_640 | 0.8756 | 0.8904 | 0.9195 | 0.8632 | 1142 | 100 | 181 | 1242 |
| 01_baseline_500 | scrfd_10g_current_640 | 0.8627 | 0.8944 | 0.9625 | 0.8352 | 1105 | 43 | 218 | 1148 |
| 01_baseline_500 | retinaface_mobilenet_1920 | 0.7917 | 0.7710 | 0.7372 | 0.8080 | 1069 | 381 | 254 | 1450 |
| 01_baseline_500 | yunet_2026_1280 | 0.7549 | 0.5878 | 0.4446 | 0.8670 | 1147 | 1433 | 176 | 2580 |
| 02_egocentric_stress_500 | scrfd_10g_current_640 | 0.9262 | 0.9357 | 0.9539 | 0.9182 | 1179 | 57 | 105 | 1236 |
| 02_egocentric_stress_500 | fusion_yolo11s_scrfd10g_agreement | 0.9230 | 0.9065 | 0.8787 | 0.9361 | 1202 | 166 | 82 | 1368 |
| 02_egocentric_stress_500 | fusion_yolo11s1280_scrfd10g_agreement | 0.9225 | 0.9013 | 0.8664 | 0.9393 | 1206 | 186 | 78 | 1392 |
| 02_egocentric_stress_500 | fusion_yolo8s_scrfd10g_agreement | 0.9191 | 0.8986 | 0.8647 | 0.9354 | 1201 | 188 | 83 | 1389 |
| 02_egocentric_stress_500 | fusion_yolo8s_yolo11s_scrfd10g_agreement | 0.9145 | 0.8838 | 0.8351 | 0.9385 | 1205 | 238 | 79 | 1443 |
| 02_egocentric_stress_500 | yolo8s_widerface_repo_640 | 0.9110 | 0.8932 | 0.8634 | 0.9252 | 1188 | 188 | 96 | 1376 |
| 02_egocentric_stress_500 | yolo11s_widerface_640 | 0.9106 | 0.9010 | 0.8845 | 0.9182 | 1179 | 154 | 105 | 1333 |
| 02_egocentric_stress_500 | yolo11n_pose_widerface_640 | 0.9091 | 0.9018 | 0.8888 | 0.9151 | 1175 | 147 | 109 | 1322 |
| 02_egocentric_stress_500 | yolo8s_lindevs_640 | 0.9086 | 0.8947 | 0.8709 | 0.9198 | 1181 | 175 | 103 | 1356 |
| 02_egocentric_stress_500 | yolo11s_widerface_1280 | 0.8996 | 0.8881 | 0.8683 | 0.9089 | 1167 | 177 | 117 | 1344 |
| 02_egocentric_stress_500 | dsfd_1920 | 0.8872 | 0.8787 | 0.8638 | 0.8941 | 1148 | 181 | 136 | 1329 |
| 02_egocentric_stress_500 | yolo8s_lindevs_1280 | 0.8837 | 0.8606 | 0.8230 | 0.9019 | 1158 | 249 | 126 | 1407 |
| 02_egocentric_stress_500 | retinaface_mobilenet_1920 | 0.8229 | 0.7788 | 0.7138 | 0.8567 | 1100 | 441 | 184 | 1541 |
| 02_egocentric_stress_500 | yunet_2026_1280 | 0.7853 | 0.5747 | 0.4173 | 0.9229 | 1185 | 1655 | 99 | 2840 |
| combined_1000 | fusion_yolo11s1280_scrfd10g_agreement | 0.9072 | 0.9002 | 0.8877 | 0.9129 | 2380 | 301 | 227 | 2681 |
| combined_1000 | fusion_yolo11s_scrfd10g_agreement | 0.9055 | 0.9021 | 0.8960 | 0.9083 | 2368 | 275 | 239 | 2643 |
| combined_1000 | fusion_yolo8s_scrfd10g_agreement | 0.9009 | 0.8927 | 0.8782 | 0.9076 | 2366 | 328 | 241 | 2694 |
| combined_1000 | yolo11s_widerface_640 | 0.8996 | 0.9026 | 0.9080 | 0.8972 | 2339 | 237 | 268 | 2576 |
| combined_1000 | fusion_yolo8s_yolo11s_scrfd10g_agreement | 0.8990 | 0.8821 | 0.8536 | 0.9125 | 2379 | 408 | 228 | 2787 |
| combined_1000 | yolo8s_widerface_repo_640 | 0.8959 | 0.8890 | 0.8769 | 0.9014 | 2350 | 330 | 257 | 2680 |
| combined_1000 | yolo8s_lindevs_640 | 0.8943 | 0.8921 | 0.8882 | 0.8960 | 2336 | 294 | 271 | 2630 |
| combined_1000 | scrfd_10g_current_640 | 0.8941 | 0.9152 | 0.9581 | 0.8761 | 2284 | 100 | 323 | 2384 |
| combined_1000 | yolo11n_pose_widerface_640 | 0.8921 | 0.8962 | 0.9037 | 0.8888 | 2317 | 247 | 290 | 2564 |
| combined_1000 | yolo11s_widerface_1280 | 0.8921 | 0.8938 | 0.8969 | 0.8907 | 2322 | 267 | 285 | 2589 |
| combined_1000 | dsfd_1920 | 0.8843 | 0.8868 | 0.8915 | 0.8822 | 2300 | 280 | 307 | 2580 |
| combined_1000 | yolo8s_lindevs_1280 | 0.8814 | 0.8722 | 0.8562 | 0.8888 | 2317 | 389 | 290 | 2706 |
| combined_1000 | retinaface_mobilenet_1920 | 0.8070 | 0.7749 | 0.7252 | 0.8320 | 2169 | 822 | 438 | 2991 |
| combined_1000 | yunet_2026_1280 | 0.7697 | 0.5810 | 0.4303 | 0.8945 | 2332 | 3088 | 275 | 5420 |
