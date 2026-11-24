# Epoch-30 Global-Statistics + Pooled Factor-Token Model

This document records the model used for the **Ours (Joint 1-Pass)** column in
the selected BSDS500 two-factor comparison.

## Model identity

- Checkpoint release name: `epoch-epoch=000029.ckpt`
- One-based epoch: **30**
- Global step: **12,060**
- Resolution: **512 x 512**
- Initialization: official GenDeg `gendeg.ckpt`
- Diffusion target: full degraded-image latent (not residual diffusion)
- Spatial conditioning: disabled
- Factor-contrast and physical-prior losses: disabled

The checkpoint is intentionally not committed to Git because it is about
14 GB. Download the split assets from the GitHub Release, concatenate them in
lexicographic order, and verify the supplied SHA-256 checksum.

```bash
gh release download epoch30-global-pooled \
  --repo greatdsyeomye/GenDeg-epoch30 \
  --pattern 'epoch-epoch.000029.ckpt.part-*' \
  --pattern SHA256SUMS

cat epoch-epoch.000029.ckpt.part-* > 'epoch-epoch=000029.ckpt'
sha256sum -c SHA256SUMS
```

Expected SHA-256:

```text
23514f3b178a2b501e3c505c05a3889b1a38e5c8fb51bf6fc09e38a7f6272d38
```

## Conditioning pipeline

The model retains the original GenDeg global-statistics path and adds a
factorized composite-degradation branch:

1. CLIP encodes the atomic degradation prompt into 77 text tokens.
2. Global residual statistics `(mu, sigma)` are represented as two 129-bin
   vectors, projected to the 77 token positions, fused with the text features,
   and mapped back to the 768-dimensional cross-attention width.
3. A factor encoder produces four 768-dimensional tokens for `low_light`,
   `haze`, `rain`, and `snow`. Each token uses factor presence and aligned
   per-factor `(mu, sigma)` statistics.
4. The four factor tokens are mean-pooled and added to the cross-attention
   context (`composite_condition_injection: additive`).
5. Every active degradation is sampled jointly in one diffusion pass. No
   degradation-order condition is supplied.

In compact form:

```text
CLIP(prompt) + fused global statistics
                  +
mean(factor tokens: low-light, haze, rain, snow)
                  |
          cross-attention context
                  |
       joint one-pass degradation generation
```

## Training

The original launch merged these configurations in order:

```bash
python stable_diffusion/main.py \
  --name gendeg_init_no_llr_512_ft5_gpu0 \
  --logdir logs \
  --base \
    configs/train.yaml \
    configs/train_full_latent_50k_60ep.yaml \
    configs/train_factorwise_stats.yaml \
    configs/finetune_gendeg_no_llr_512_5ep.yaml \
  --scale_lr false --train --no-test
```

The run used batch size 1 with 128-step gradient accumulation. It was first
run for five epochs and then continued to epoch 30 with:

```bash
python stable_diffusion/main.py \
  --resume /path/to/last.ckpt \
  --base configs/continue_gendeg_no_llr_512_to30.yaml \
  --scale_lr false --train --no-test
```

Training sources were CDD-11 composites plus aligned AllWeatherNight, LCDD,
and WeatherBench pairs. Dataset images are not redistributed in this
repository; the dataset loader, manifest format, and configuration are kept
for reproducibility.

## Inference used for the qualitative comparison

```bash
python scripts/infer_bsds500_simultaneous_combinations.py \
  --ckpt /path/to/epoch-epoch=000029.ckpt \
  --project-config /path/to/epoch30-project.yaml \
  --clean-root /path/to/BSDS500/images/test \
  --output-dir results/epoch30_joint \
  --device cuda:0 \
  --resolution 512 \
  --steps 100 \
  --cfg-text 1.0 \
  --cfg-image 1.0 \
  --seed 2031 \
  --fixed-noise-seed
```

The sampler is Euler ancestral. The published comparison used simultaneous,
order-free factor activation and one diffusion pass per output.

## Relevant implementation files

- `composite_condition.py`: factor encoder and null factor condition
- `composite_dataset.py`: aligned composite dataset and factor statistics
- `stable_diffusion/ldm/models/diffusion/ddpm_edit.py`: global/factor
  conditioning, training loss, and optimizer wiring
- `scripts/infer_bsds500_simultaneous_combinations.py`: joint one-pass
  inference and metadata export
- `configs/train_factorwise_stats.yaml`: four-token encoder with additive
  mean-pooled injection
- `configs/finetune_gendeg_no_llr_512_5ep.yaml`: 512-pixel data mixture and
  official-GenDeg initialization
- `configs/continue_gendeg_no_llr_512_to30.yaml`: continuation to epoch 30
