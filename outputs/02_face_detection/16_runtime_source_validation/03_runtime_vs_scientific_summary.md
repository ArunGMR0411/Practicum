# Runtime three-source detector evidence boundary

- Live sources: `rfdetr_medium_face_030, yolo11s_widerface_1280, scrfd_10g_current_640`
- Best separately trained three-source candidate score: **0.918389** (`error_hardened_runtime_3_logreg_iou0_45`)
- Precision / Recall / F1: 0.9200 / 0.9179 / 0.9190
- Scientific 7-source score (`error_hardened_all_raw_rf_iou0_45`): **0.917165**
- Delta (runtime − scientific): **+0.001224**

The scored three-source candidate is logistic regression; it is not the RF model currently loaded by the App. The App runtime id is `runtime_3_source_all_raw_rf_approximation`, using the retained all-raw RF filter with only the three live candidate sources. No numeric score is assigned to that exact runtime implementation. The thesis **0.917165** figure remains offline CV on the seven-source `all_raw` bank and is not transferred to the App tier.
