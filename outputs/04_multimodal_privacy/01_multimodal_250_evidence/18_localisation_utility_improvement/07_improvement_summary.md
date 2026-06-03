# Multimodal localisation + utility improvement

## Localisation
- Published CRAFT region precision: **0.0177** (localisation deploy **0.633348**).
- Improved filter `improved_minarea_4k_aspect8_max12`: text region P=**0.1068**.
- Dual-report localisation deploy score: **0.642255**.

## Utility
- Baseline adaptive util&lt;0.50: **37/75**.
- Promoted per-image soft utility (dev-chosen privacy floor 0.8): util&lt;0.50 = **27/75** (Δ -10).
- Mean privacy/utility/score: **0.8243** / **0.7169** / **0.8259**.
