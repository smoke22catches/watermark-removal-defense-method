# Evaluation protocol

This document describes the **evaluation-only** transform registry (`src/eval/registry.py`).
Training uses differentiable approximations in `src/attacks/distortion.py`; evaluation
**never** imports that module.

Official reference benchmark: **WAVES** (An et al., ICML 2024),
code at https://github.com/umd-huang-lab/WAVES (README states MIT License; GitHub
`license` field is unset — we treat reuse of attack *definitions/strengths* as allowed
and reimplement thin wrappers rather than vendoring the full WAVES package).

## Preprocessing

Before every evaluation transform, the watermarked image is quantized with an 8-bit
uint8 round-trip (`src/eval/quantize.py`), matching published 8-bit storage.

## Registry entries

| id | source | strengths | seen_in_training | notes / deviations |
|----|--------|-----------|------------------|--------------------|
| clean | own | 0 | yes | Identity after 8-bit quantize |
| rotate | WAVES | 5, 15, 30, 45, 90° | yes (train has geometric rotate) | `affine_grid` + border pad; WAVES Dist-Rotation |
| crop | WAVES | 0.9, 0.75, 0.6, 0.5 keep | yes | Centre resized-crop (WAVES Dist-RCrop) |
| erase | WAVES | 0.05, 0.10, 0.20 area | no | Mid-grey rectangle erase |
| brightness | WAVES | 0.5, 0.75, 1.25, 1.5 | no | Multiplicative in [-1,1] |
| contrast | WAVES | 0.5, 0.75, 1.25, 1.5 | no | Mean-centred scale |
| noise | WAVES | 0.02, 0.05, 0.10 std | yes | Additive Gaussian |
| unseen_blur | WAVES | 0.8, 1.5, 3.0 σ | no | Separable Gaussian blur |
| jpeg | WAVES | 90, 70, 50, 30, 10 | yes | **PIL** encode/decode (not training DiffJPEG) |
| downsample | own | 2×, 4× | yes | Avg-pool + bilinear upsample |
| regen_proxy | own | t* 0.2, 0.3, 0.4 | yes | Same few-step DDIM reconstruction proxy as training (`regen.sd_model_id` / revision); `k=black`; `torch.no_grad()` at eval. Alias: `regen`. |
| guided_regen | own | ε_g 0.01, 0.02, 0.04 | yes | regen_proxy then guided PGD on the evaluated decoder (`k=white`). Baselines use their own decoder. |
| diffpure | diffusers | t_start 0.2, 0.4, 0.6 | no | Full-step purification with **`eval.diffpure_model_id`** (default `CompVis/stable-diffusion-v1-4`), asserted ≠ `regen.sd_model_id` (default `runwayml/stable-diffusion-v1-5`). `k=black`. Offline stand-in: blur+noise stack when weights missing |
| rinse_2 | WAVES | 2 | no | DiffPure ×2 (WAVES Regen-2xDiff style); `k=black` |
| rinse_4 | WAVES | 4 | no | DiffPure ×4; `k=black` |
| bmshj2018 | compressai | quality 1,3,5,8 | no | compressai `bmshj2018_*`; `k=black`; offline spatial bottleneck if package missing |
| cheng2020 | compressai | quality 1,3,5,6 | no | compressai `cheng2020_*`; `k=black`; offline stand-in if missing |
| adv | WAVES | 2/255 … 8/255 | yes | PGD on decoder (WAVES AdvEmbed ε grid); uses `src.attacks.adversarial` (not DistortionBank) |
| semantic_regen | semanticregen | — | no | **not_evaluated**: official repo https://github.com/KrtiT/semanticRegen is a Jupyter notebook pipeline (MIT), not an importable API. No invented reimplementation |

## Perturbation metrics

For every transformed sample the suite records:

- \(\|T(x_w) - x_w\|_2\)
- \(\|T(x_w) - x_w\|_\infty\)
- LPIPS\((x_w, T(x_w))\)

## Protocol table

`src.eval.registry.protocol_table_rows()` and figure generator `generate_c3` emit exactly
one row per registry entry (including `not_evaluated` rows).
