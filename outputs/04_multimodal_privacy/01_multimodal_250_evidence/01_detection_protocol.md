# Multimodal Region-Level Detection Protocol

- Human-reviewed images: `250` (`175` development; `75` held-out test).
- Text boxes: `116` across `49` images.
- Screen boxes: `139` across `127` images.
- Annotation SHA-256: `86941cf2f892064dc4fe3332809c195e81fb891d092d1bc781f8f3d2179c3ea9`.

## Methods and author-recipe settings

- CRAFT through EasyOCR uses the documented default thresholds (`text_threshold=0.7`, `low_text=0.4`, `link_threshold=0.4`, `canvas_size=2560`, `mag_ratio=1.0`) as the precision reference.
- The 4K recall variant increases the canvas to 3840 and lowers region/affinity thresholds; it is selected only if development evidence improves the privacy-weighted score.
- docTR uses its documented pretrained `db_resnet50` detector with `crnn_vgg16_bn` recognition and aspect-preserving inference.
- Ultralytics models use explicit COCO screen-like classes (`tv`, `laptop`, `cell phone`), FP16 GPU inference, documented confidence/NMS controls, and 640/1280-pixel inference comparisons.
- Screen boxes take priority: a text proposal is removed when any text-box corner lies inside a selected screen box.

## Selection and metrics

- Variants are selected only on the development split; test rows are held out until final scoring.
- Screen localisation uses one-to-one IoU >= 0.50 matching.
- Text reports strict IoU secondarily; its primary region-hit metric accommodates the reviewed mixture of line-level and whole-document boxes.
- `OAPR multimodal score = 0.65 * recall + 0.25 * F1 + 0.10 * precision`.

Selected text method: `craft_recall_4k`.
Selected screen method: `yolo11n_coco_640_1280_union`.

Primary sources:

- CRAFT paper: https://openaccess.thecvf.com/content_CVPR_2019/html/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.html
- EasyOCR API defaults: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/easyocr.py
- docTR model documentation: https://mindee.github.io/doctr/latest/modules/models.html
- Ultralytics prediction settings: https://docs.ultralytics.com/modes/predict/
- Ultralytics COCO classes: https://docs.ultralytics.com/datasets/detect/coco/

## Text-cluster screen completion

- When the selected YOLO screen pass returns no boxes, dense CRAFT text clusters (≥5 linked proposals) form a hypothesized screen box with 18% margin.
- Text proposals overlapping any screen (YOLO or hypothesis) are removed before risk routing.
- Campaign evidence: `14_detection_fix_campaign/02_campaign_report.md`.
