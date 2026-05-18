# Sliced Detector Policy Experiment

This low-compute experiment tests whether overlapping 1280px YOLO11Face tiles and RF-DETR Medium face detection improve the detector policy on the two reviewed 500-image protocols.

RF-DETR status: run on CUDA and included in the score table.

Main result:

- Best combined 1,000-image method in this experiment: `cv_box_reranker_with_rfdetr_predicted_conditions`.
- Score: OAPR detector score `0.9129`, F1 `0.9110`, precision `0.9075`, recall `0.9145`, TP `2384`, FP `243`, FN `223`.
- Fixed YOLO11/SCRFD fusion comparator: score `0.9072`, F1 `0.9002`, precision `0.8877`, recall `0.9129`.

Sliced inference result:

- Sliced YOLO11Face alone: score `0.7587`, F1 `0.6911`, precision `0.6035`, recall `0.8086`.
- Sliced inference is feasible, but it is not promoted when false-positive control is part of the objective.

RF-DETR result:

- RF-DETR alone: score `0.8932`, F1 `0.8868`, precision `0.8755`, recall `0.8984`.
- RF-DETR is promoted only through the RF-DETR-aware box reranker, not as a standalone detector replacement.
