# CompositeGenDeg experiment order

## Gate 0: environment and weights

- [x] Create a project-local Python environment.
- [x] Verify CUDA and PyTorch imports.
- [x] Download and checksum the Stable Diffusion 1.5 EMA checkpoint.
- [x] Instantiate the unmodified GenDeg model from the SD 1.5 checkpoint.

Exit criterion: model construction succeeds and the expanded 8-channel U-Net
loads without unexpected trainable-weight failures.

## Gate 1: structured-condition unit test

- [x] Define four factors: low-light, haze, rain, and snow.
- [x] Encode active flag, component mu/sigma, two observable factors,
  factor-validity mask, and optional order.
- [x] Project each factor to one 768-dimensional cross-attention token.
- [x] Test mixed precision and CUDA execution.

Exit criterion: `[B, 4, 768]` tokens are produced and gradients reach every new
trainable projection.

## Gate 2: dataset contract

- [x] Download the CDD-11 test archive and inspect filename alignment.
- [ ] Define a JSONL manifest with clean/target paths, active factors, global
  mu/sigma, component labels, and known/unknown parameter masks.
- [ ] Split by clean-image identity, not by degraded-image filename.
- [x] Implement a loader that returns both original GenDeg conditions and new
  factor conditions.

Exit criterion: one batch containing single and composite samples is
deterministic, visualized, and shape-checked.

## Gate 3: baseline GenDeg smoke test

- [x] Run 1 forward/backward step with the original global mu/sigma condition.
- [x] Run a 20-step tiny overfit check (loss 0.07075 -> 0.00555).
- [x] Run and checkpoint the reproducible 100-step overfit experiment.
- [x] Save, reload, and infer from the produced checkpoint.

Exit criterion: loss decreases on the overfit set and reload reproduces the
same conditioning behavior.

## Gate 4: CompositeGenDeg integration

- [x] Append four factor tokens to the frozen CLIP token sequence.
- [x] Add independent classifier-free dropout for text, clean image, and
  structured factors.
- [x] Register the factor encoder in the optimizer and checkpoint state.
- [x] Add checkpoint-reload composite inference with factor tokens and seeds.
- [ ] Expose manual per-factor controls after train-split normalization.

Exit criterion: the full U-Net completes forward/backward and inactive factors
map to the null condition.

## Gate 5: minimum pilot

- [x] Train a composite-only pilot covering all seven CDD-11 composite classes.
- [x] Add a 5k-step two-GPU pilot config restricted to physical GPUs 0 and 1.
- [ ] Compare global mu/sigma baseline against factorized conditioning.
- [ ] Sweep rain intensity while holding haze fixed, then reverse.
- [ ] Repeat each condition with multiple diffusion/style seeds.

Exit criterion: requested factors appear, clean content is preserved, and one
factor can be varied without a large unintended change in the other.

## Gate 6: full four-factor experiment

- [ ] Train with all CDD-11 single, two-factor, and three-factor categories.
- [ ] Mix single-degradation real paired datasets for disentanglement.
- [ ] Reserve unseen combinations and CCDD-11 for external evaluation.
- [ ] Add depth/illumination spatial adapters only after the token baseline.
- [ ] Run ablations: text only, global statistics, factor tokens, spatial
  factors, and style latent.

Exit criterion: quantitative controllability, realism, content preservation,
and unseen-combination results are reported against a matched GenDeg baseline.

## Smoke-test measurements

- Hardware: one RTX 6000 Ada 48 GB.
- SD 1.5 -> original GenDeg, random 64x64 batch: forward/backward passed.
- Composite factor encoder: CPU and CUDA FP16 unit tests passed.
- SD 1.5 -> CompositeGenDeg, CDD-11 haze+rain at 256x256: loss 0.07488,
  factor gradient present, peak allocated memory 10.58 GiB.
- Full optimizer, one CDD-11 haze+rain pair at 128x128 for 20 fixed-noise
  steps: loss 0.07075 -> 0.00555, peak allocated memory 20.12 GiB.
- CDD-11 physical parameter labels are unavailable; their parameter masks are
  not fabricated. The current loader measures observable factors from aligned
  single variants and masks unverified application order.
- The one-pair 100-step checkpoint reloaded and sampled successfully, but its
  output was structurally invalid. It is a pipeline gate, not a quality result.
- A 16-scene, seven-composite, 256px, 1k-step gate completed but overfit: the
  held-out scene geometry changed substantially during inference.
- Continued from that checkpoint on 256 clean scenes x seven composite classes
  (1,792 additional steps, physical GPU 0 only).  The saved checkpoint reloads
  with zero missing/unexpected keys.  On held-out CDD-11 haze+rain image 00002,
  CFG 1.0 preserves the main geometry and produces PSNR 19.619 / SSIM 0.6343
  against the paired reference.  This is a pilot result, not a final benchmark.
