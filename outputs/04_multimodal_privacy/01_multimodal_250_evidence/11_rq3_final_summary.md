# RQ3 Multimodal Privacy Evidence

## Reviewed protocol

- `250` egocentric images were manually reviewed with `116` text boxes and `139` screen boxes.
- Screen-priority annotation and routing remove text boxes that overlap a reviewed/predicted screen.
- Method selection uses the development split; the primary result below is held-out test evidence.
- Adaptive operators include area-aware screen redaction (fill small regions, strong-blur large ones) and a text-only privacy floor on development selection.

## Detection

- Combined risk precision: `0.8333`.
- Combined risk recall: `0.9804`.
- Combined risk F1: `0.9009`.
- OAPR multimodal score: `0.9458`.
- Box/region-level results are reported separately in `02_detection_method_comparison.csv`; image-level presence is not presented as perfect localisation.

## End-to-end redaction

- Adaptive privacy score: `0.8680`.
- Adaptive utility score: `0.6634`.
- Adaptive multimodal anonymisation score: `0.8321`.
- Strongest fixed policy: `fixed_text_blur_screen_fill` with score `0.8319`.
- Adaptive minus strongest fixed: `0.0002`.

## Adaptive policy (development-selected)

- `no_text_screen_risk` → `no_action_copy` (No predicted text/screen boxes; preserve the image.)
- `screen_present` → `text_blur_screen_fill` (Highest development-split measured score; privacy breaks score ties.)
- `text_and_screen_present` → `text_blur_screen_fill` (Highest development-split measured score; privacy breaks score ties.)
- `text_present` → `text_blur_screen_area_aware_t10` (Highest development-split measured score; privacy breaks score ties.)

## Interpretation

- The protocol uses independent human boxes and held-out localization evaluation.
- The combined detector is privacy-oriented: missed-risk recall is weighted more strongly than harmless extra redaction.
- End-to-end privacy includes localization failures; it is not an oracle-box redaction result.
- Area-aware screen operators target utility collapse on large displays while retaining hard fill on small screens.
- The text-only privacy floor prevents weak pixelation when a stronger privacy operator is available on development evidence.
- Residual false positives and missed regions remain measurable limitations and are not converted into a full-anonymisation claim.

## Held-out residual-risk flags

- Missed text-risk images: `2`.
- Missed screen-risk images: `0`.
- Text readability below the privacy threshold: `6` flagged images.
- Screen obscuration below the privacy threshold: `1` flagged images.
- Utility below 0.50: `37` images.
