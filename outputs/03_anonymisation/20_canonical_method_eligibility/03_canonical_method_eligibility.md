# Canonical method eligibility

Single artefact for scientific routing gates and App method catalog defaults.

- CSV: `outputs/03_anonymisation/20_canonical_method_eligibility/01_canonical_method_eligibility.csv`
- JSON: `outputs/03_anonymisation/20_canonical_method_eligibility/02_canonical_method_eligibility.json`

| Method | Eligibility | Default route | App selectable | Role |
| --- | --- | --- | --- | --- |
| no_action_copy | ELIGIBLE | yes | yes | reviewed no-face frames only |
| blur | ELIGIBLE | yes | yes | utility-oriented deterministic candidate |
| pixelate | ELIGIBLE | yes | yes | utility-oriented deterministic candidate |
| solid_mask_black | ELIGIBLE | yes | yes | privacy-first terminal action |
| layered_blur_downscale_noise | ELIGIBLE | yes | yes | default balanced action |
| nullface | RESEARCH_ONLY_NOT_DEFAULT | no | yes | research comparison |
| diffusion_low_step | RESEARCH_ONLY_NOT_DEFAULT | no | yes | negative/conditional evidence |
| reverse_personalization | RESEARCH_ONLY_NOT_DEFAULT | no | yes | research comparison |
| styleid_stylegan | EXCLUDED | no | no | quality-limited evidence |
| fams | EXCLUDED | no | no | quality-limited evidence |
| riddle | RESEARCH_COMPARABLE_NOT_DEFAULT | no | yes | full comparable evidence |
| falco | RESEARCH_COMPARABLE_NOT_DEFAULT | no | yes | full comparable evidence |
