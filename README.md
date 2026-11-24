# [CVPR-2025] GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration
[Sudarshan Rajagopalan](https://sudraj2002.github.io/) | [Nithin Gopalakrishnan Nair](http://nithin-gk.github.io/) | [Jay N. Paranjape](https://jayparanjape.github.io/website/)| [Vishal M. Patel](https://scholar.google.com/citations?user=AkEXTbIAAAAJ&hl=en)

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://sudraj2002.github.io/gendegpage/) [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.17687) [![GenDS Dataset](https://img.shields.io/badge/HuggingFace-Data-orange?logo=huggingface)](https://huggingface.co/datasets/Sudarshan2002/GenDS)

## Epoch-30 joint one-pass research extension

This fork adds a 512px composite-degradation pipeline that keeps GenDeg's
global statistics and injects a mean-pooled set of four factor tokens
(`low-light`, `haze`, `rain`, and `snow`) for order-free, joint one-pass
generation.

- [Method, training, and inference details](GenDeg/EPOCH30_GLOBAL_POOLED.md)
- [Epoch-30 checkpoint release](https://github.com/greatdsyeomye/GenDeg-epoch30/releases/tag/epoch30-global-pooled)
- Checkpoint: `epoch-epoch=000029.ckpt` (epoch 30, global step 12,060)

Training datasets and generated results are intentionally not redistributed.
This repository contains the implementation, experiment configuration, tests,
and checkpoint needed to reproduce the pipeline with separately obtained data.

## GenDeg
For instructions regarding training and inference of the GenDeg model refer to this [file](https://github.com/sudraj2002/GenDeg/blob/main/GenDeg/README.md).

## Restoration models
To use the restoration models trained on the GenDS dataset, refer to this [file](https://github.com/sudraj2002/GenDeg/blob/main/Restoration/README.md).


## GenDS Dataset
To download the GenDS dataset please refer to the [dataset page](https://huggingface.co/datasets/Sudarshan2002/GenDS).

## Citation

If you found our work useful, please consider citing:

```bibtex
@inproceedings{rajagopalan2025gendeg,
  title={Gendeg: Diffusion-based degradation synthesis for generalizable all-in-one image restoration},
  author={Rajagopalan, Sudarshan and Nair, Nithin Gopalakrishnan and Paranjape, Jay N and Patel, Vishal M},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={28144--28154},
  year={2025}
}
```
