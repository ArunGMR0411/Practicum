# Multimodal Protocol (Canonical)

## Canonical 250-image study

RQ3 uses a single canonical multimodal protocol with independent human
localisation ground truth and held-out evaluation:

1. The 250 images are egocentric and privacy-risk enriched, with a retained
   hard-negative stratum.
2. Project reviewers manually verified every frame and corrected text and
   screen boxes in the reviewer application.
3. The final ground truth contains 116 text boxes in 49 images and 139 screen
   boxes in 127 images. Screen regions take priority over overlapping text.
4. A stratified 70/30 development/test split (175 / 75) prevents parameter
   selection on the held-out result.
5. CRAFT/EasyOCR, docTR DBNet, and multiple Ultralytics COCO screen models are
   evaluated using author-documented inference controls.
6. Text reports privacy-region hits and strict IoU because reviewed text boxes
   intentionally mix line-level and whole-document regions. Screens use strict
   one-to-one IoU >= 0.50 matching.
7. The selected screen policy fuses YOLO11 640- and 1280-pixel passes; text
   proposals overlapping a screen are removed before routing.
8. Six fixed redaction combinations and one adaptive policy are evaluated
   end-to-end using predicted boxes, not oracle boxes.
9. Privacy, OCR suppression, screen obscuration, SSIM, LPIPS, non-sensitive
   change, runtime, success, paired differences, and residual failures are
   retained in machine-readable files.

This protocol is the sole RQ3 evidence surface. It does not claim that every
text or screen region is found or that all semantic leakage is removed.

Primary summary: `11_rq3_final_summary.md`.
