# SWMVR: Spectral-Wavelet Masked Video Representation Learning for Endoscopic Polyp Detection

Official PyTorch implementation of the SWMVR framework.

## Overview

SWMVR is a self-supervised video representation learning framework designed for endoscopic polyp detection. It leverages **spectral-domain** (STFT / wavelet) and **spatial-domain** (pixel-level) masked modeling to learn robust spatio-temporal representations from unannotated endoscopic videos.

### Key Components

- **Spectral Branch**: Short-Time Fourier Transform (STFT) and wavelet masking capture frequency-domain patterns in endoscopic sequences.
- **Spatial Branch**: Transformer-based spatio-temporal modeling (TimeSformer / ViT backbone) learns pixel-level video representations.
- **Multi-view Fusion**: Cross-view contrastive learning aligns spectral and spatial embeddings for downstream polyp segmentation and diagnosis.

## Repository Structure

```
SWMVR/
├── models/                  # Model backbones and architectures
│   └── timesformer.py        # TimeSformer-based spatio-temporal encoder
├── TransUNet/               # TransUNet segmentation head
├── STFT/                    # Spectral-domain (STFT / wavelet) modules
├── utils/
│   └── logging.py            # Logging utilities
├── vision_transformer.py     # Vision Transformer backbone
├── environment.yaml          # Conda environment configuration
└── README.MD
```

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.10+
- CUDA 11.0+

### Setup

```bash
conda env create -f environment.yaml
conda activate swmvr
```

## Model Zoo

We provide the model architecture implementations. Pre-trained weights will be released upon paper acceptance.

| Model | Backbone | Pretrain Data | Downstream | Config |
|-------|----------|---------------|------------|--------|
| SWMVR-Base | TimeSformer-B/16 | Endoscopic Videos | Polyp Segmentation | - |

## Datasets

This repo does not include datasets. We evaluate on:

- **PolypDiag** — endoscopic video polyp diagnosis
- **ASU-Video** — colonoscopy video dataset

Please refer to the respective dataset websites for access.

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{swmvr2026,
  title={SWMVR: Spectral-Wavelet Masked Video Representation Learning for Endoscopic Polyp Detection},
  author={Meng, Qiuhan and et al.},
  year={2026}
}
```

## License

This project is for academic research purposes only.

---

> **Note**: This repository provides the model architecture implementation only. Complete training scripts, data loaders, and evaluation pipelines are not released.
