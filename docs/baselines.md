# Baseline watermark schemes (comparative evaluation)

Post-hoc image watermarking baselines evaluated under the **same** robustness
protocol as the proposed method (`scripts/infer.py --scheme <id>`).
Generation-time methods (Stable Signature, Tree-Ring) are **out of scope**.

Availability is determined at runtime by whether the official wrapper and
weights load successfully (`src/baselines/factory.py`). There is no hard-coded
`available = False` for StegaStamp / TrustMark / VINE.

| Scheme | `scheme` id | Post-hoc? | Native res | Payload \(k\) | Status |
|--------|-------------|-----------|------------|---------------|--------|
| StegaStamp (Tancik et al., CVPR 2020) | `stegastamp` | yes | 400 | 100 | wrapper + official TF SavedModel |
| TrustMark (Bui et al., arXiv:2311.18297 / ICCV 2025) | `trustmark` | yes | 256 | 100 | wrapper + Adobe official package |
| VINE (Lu et al., ICLR 2025) | `vine` | yes | 256 | 100 | wrapper + authors' HF weights |
| Stable Signature | `stable_signature` | **no** | 512 | 48 | `not_applicable` (generation-time) |
| Tree-Ring | `tree_ring` | **no** | 512 | — | `not_applicable` (generation-time) |

## Shared evaluation policy

- **Images:** common test split at `image_size` from config (NCHW, float, \([-1,1]\)).
  Use **`image_size ≥ 256`** for comparative runs so TrustMark / VINE residual
  transfer stays near the authors' operating point (sanity check uses 256²).
- **Messages:** Bernoulli bits from a seeded `torch.Generator` (`seed + 17389`),
  identical procedure for every scheme (`src.baselines.common.sample_bits`).
- **Resolution:** embed at the scheme's native resolution, then **upsample the
  residual** \(x_w^\mathrm{native} - x^\mathrm{native}\) back to the evaluation
  size (authors' recipe in TrustMark / VINE / StegaStamp encode scripts).
- **Metrics path:** same registry transforms, 8-bit quantization, bit accuracy,
  TPR@FPR with each scheme's own \(k\), PSNR / SSIM / LPIPS / FID → `results.csv`.
- **Metadata:** `results_meta.json` records `scheme`, `msg_len`,
  `native_resolution`, and `weight_source` (URL, license, version, checksums).

## StegaStamp

| Field | Value |
|-------|--------|
| Paper | Tancik, Mildenhall, Ng — *StegaStamp: Invisible Hyperlinks in Physical Photographs*, CVPR 2020 |
| Code | https://github.com/tancik/StegaStamp (MIT) |
| Weights | Official Berkeley archive linked from the authors' README: `http://people.eecs.berkeley.edu/~tancik/stegastamp/saved_models.tar.xz` |
| Weight file | `saved_models/stegastamp_pretrained` (TensorFlow SavedModel) |
| Checksum | SHA-256 of the `.tar.xz` is computed on first download and stored under `weights/stegastamp/` / `results_meta.json`. Re-run `python scripts/download_baselines.py --scheme stegastamp` to print the digest. **Do not substitute third-party PyTorch re-trains** (e.g. community HF ports) without documenting them here as unofficial. |
| License | MIT (repository `LICENSE`) |
| Runtime | Requires `tensorflow>=2.12` (TF1 SavedModel via `tf.compat.v1`). |
| Resolution | Native **400×400**; residual upsampled to eval size. |

Install:

```bash
pip install "tensorflow>=2.12"
python scripts/download_baselines.py --scheme stegastamp
```

## TrustMark

| Field | Value |
|-------|--------|
| Paper | Bui, Agarwal, Collomosse — *TrustMark*, arXiv:2311.18297 / ICCV 2025 |
| Code | https://github.com/adobe/trustmark (MIT) |
| Package | Official `trustmark` (PyPI or `pip install git+https://github.com/adobe/trustmark.git#subdirectory=python`) |
| Variant used | **Q** (default; ResNet-50 backbone, ~43 dB PSNR in the paper) |
| Weights host | `https://cai-watermark.adobe.net/watermarking/trustmark-models/` (auto-download on first use) |
| Checksums (MD5, published in `python/trustmark/trustmark.py`) | `encoder_Q.ckpt` = `700328b8754db934b2f6cb5e5185d81f`; `decoder_Q.ckpt` = `4ced90e9cfe13e3295ad082887fe9187`; `trustmark_Q.yaml` = `fe40df84a7feeebfceb7a7678d7e6ec6` |
| License | MIT (code **and** model files, per Adobe README) |
| Resolution | Encode at **256×256**, residual merged back (authors' `TrustMark.encode`). |

Install:

```bash
pip install trustmark
# Windows encoding issues: pip install "git+https://github.com/adobe/trustmark.git#subdirectory=python"
```

Evaluation uses **raw 100-bit** payloads (`use_ECC=False`) so random messages match the shared bit sampler.

## VINE

| Field | Value |
|-------|--------|
| Paper | Lu et al. — *Robust Watermarking Using Generative Priors Against Image Editing*, ICLR 2025 / arXiv:2410.18775 |
| Code | https://github.com/Shilin-LU/VINE (authors' official implementation) |
| Weights | Hugging Face (authors): encoder `Shilin-LU/VINE-R-Enc`, decoder `Shilin-LU/VINE-R-Dec` |
| Checksums | Hugging Face revision pins are recorded in `results_meta.json` at load time (`weight_files.encoder` / `.decoder` IDs). Prefer the Hub file hashes shown on those model cards when auditing. |
| License | See repository `LICENSE` on Shilin-LU/VINE |
| Resolution | Encode at **256×256**, residual bicubic-upsampled to original / eval size (authors' `watermark_encoding.py`). |

Install:

```bash
pip install "git+https://github.com/Shilin-LU/VINE"
```

VINE pulls a large SDXL-Turbo-based encoder; first run downloads several GB from Hugging Face.

## Not applicable (generation-time)

### Stable Signature

Watermark is baked into a fine-tuned generative model; there is no post-hoc
`embed(images, bits)` for arbitrary covers. Kept as `not_applicable` with that
reason (not silently approximated).

### Tree-Ring

Fourier-domain rings injected during diffusion sampling; not a post-hoc
embedder. Kept as `not_applicable`.

## Running comparative eval

```bash
# TrustMark example
python scripts/infer.py --scheme trustmark --dataset coco --device cuda \
  --sweep-strengths --run-name eval_trustmark

# StegaStamp (after TF + weight download)
python scripts/infer.py --scheme stegastamp --dataset coco --device cpu \
  --sweep-strengths --run-name eval_stega

# VINE
python scripts/infer.py --scheme vine --dataset coco --device cuda \
  --sweep-strengths --run-name eval_vine
```

Sanity check (clean bit accuracy on 50 images):

```bash
python tests/test_baselines_sanity.py
```
