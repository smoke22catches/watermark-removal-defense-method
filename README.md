# Regeneration-robust watermark defense

PyTorch project for adversarial, regeneration-robust invisible watermark embedding and
extraction (Encoder / Decoder + DiffJPEG distortions + diffusion RegenerationProxy +
PGD + Stackelberg min-max training). Refactored from the single-file prototype
`start.py` (kept as a reference; the runnable code lives under `src/` and `scripts/`).

## Setup (conda + CUDA 12.x)

Target: NVIDIA RTX 4000-series (Ada, sm_89). PyTorch is installed via official CUDA 12.4
wheels from the pip section of `environment.yml` (not the conda `pytorch` channel).

```bash
conda env create -f environment.yml
conda activate watermark-removal
```

Verify the stack:

```bash
python scripts/check_env.py
```

You should see `cuda available: True` and your GPU name on the training server.

## Download datasets

```bash
# All supported sets (COCO, DIV2K, Imagenette, CLIC; ImageNet prints manual instructions)
python scripts/download_data.py --dataset all --data-root ./data

# Single dataset, capped for a quick smoke test
python scripts/download_data.py --dataset coco --split val2017 --num-samples 100
python scripts/download_data.py --dataset imagenette --num-samples 200
```

Writes `data/manifest.json` with paths and image counts. Training falls back to a
synthetic dataset if the named set is missing.

## Train

```bash
# Minimal CPU dry-run (synthetic data, 2 epochs, placeholder regen proxy)
python scripts/train.py --dataset synthetic --device cpu --epochs 2 --batch-size 2 \
  --num-samples 8 --image-size 64 --run-name dryrun \
  --set regen.use_placeholder=true

# GPU training on COCO
python scripts/train.py --dataset coco --device cuda --epochs 100 --batch-size 8 \
  --run-name coco_v1
```

Common overrides: `--lr`, `--msg-len`, `--lambda-perc`, `--regen-steps`, `--resume`,
`--set key=value` (dotted keys into `configs/default.yaml`).

## Evaluate / infer

```bash
# Attack sweep (includes UNSEEN blur held out from training)
python scripts/infer.py --checkpoint runs/<run>/checkpoints/best.pt \
  --dataset coco --device cuda --attacks clean,jpeg,regen,guided_regen,unseen_blur \
  --run-name eval_coco

# Single image
python scripts/infer.py --checkpoint runs/<run>/checkpoints/best.pt \
  --image path/to/cover.png --attack jpeg --device cuda --run-name single
```

## Where results land

Every train / eval run creates a unique directory:

```
runs/<YYYYmmdd-HHMMSS>_<run-name>/
├── config.yaml
├── metrics.csv
├── checkpoints/     # best.pt, last.pt (train)
├── plots/
└── log.txt
```

See `runs/README.md`. `runs/`, `data/`, and `*.pt` are gitignored.

## Project layout

```
configs/default.yaml   # all hyperparameters
src/models/            # Encoder, Decoder, ConvBNReLU
src/attacks/           # DiffJPEG, DistortionBank, RegenerationProxy, PGD, AttackSampler
src/engine/            # train_step, curriculum loop, evaluate
src/data/              # datasets + dataloaders
src/losses.py          # BCEWithLogits + LPIPS/MSE
scripts/               # download_data, train, infer, check_env
start.py               # original prototype (reference only)
```

## Placeholders / real backends

Defaults stay offline-friendly. Flip flags for production paths:

```bash
# Real block-DCT + STE DiffJPEG
python scripts/train.py ... --set diffjpeg.use_real_diffjpeg=true

# Real SD RegenerationProxy + CLIP text embeds (auto-loads CLIP when placeholder=false)
# Requires ~4–6 GB VRAM (SD1.5 VAE+UNet fp16 @ 128², batch 1–2). image_size % 8 == 0.
# Run `huggingface-cli login` if the model hub requires auth.
python scripts/train.py ... --device cuda \
  --set regen.use_placeholder=false \
  --set regen.use_real_text_embeds=true \
  --set regen.sd_model_id=runwayml/stable-diffusion-v1-5
```

Validate implementations:

```bash
python scripts/check_placeholders.py --skip-sd   # DiffJPEG STE + placeholder regen
python scripts/check_placeholders.py             # also tries real SD/CLIP if CUDA+weights
```

| Flag | Default | Effect |
|------|---------|--------|
| `diffjpeg.use_real_diffjpeg` | `false` | Noise approx → block-DCT + STE JPEG |
| `regen.use_placeholder` | `true` | Blur/noise proxy → SD VAE/UNet/DDIM |
| `regen.use_real_text_embeds` | `false` | Zero embeds → CLIP (auto-on when real regen) |
| `regen.prompt` | `""` | Unconditional null-text conditioning |
