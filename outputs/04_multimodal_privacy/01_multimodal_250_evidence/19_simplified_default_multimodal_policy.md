# Simplified default multimodal redaction policy

## Default (deployable)

| Risk state | Operator | Rationale |
| --- | --- | --- |
| no_text_screen_risk | no_action_copy | Preserve image |
| screen_present | **text_blur_screen_fill** | Privacy-weighted, competitive score |
| text_and_screen_present | **text_blur_screen_fill** | Same defensible default |
| text_present | **text_blur_screen_fill** | Text blur; screen fill is no-op without screens |

Machine-readable: `19_simplified_default_multimodal_policy.csv`.

## Demoted from default consideration

Retained as **research / ablation only** (not App or scientific default routing):

- All `*_area_aware_*` variants (utility experiments; see `18_localisation_utility_improvement/`)
- `text_fill_screen_*` except when equal to the default fill pattern above
- `text_pixelate_screen_*` (higher utility, weaker privacy)

Canonical adaptive comparison tables remain under `07–09` for scientific honesty; **defaults** follow this simplified privacy-weighted set.
