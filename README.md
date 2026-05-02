# HiFi-Mamba: Dual-Stream W-Laplacian Enhanced Mamba for High-Fidelity MRI Reconstruction (AAAI 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2508.09179-b31b1b.svg)](https://arxiv.org/abs/2508.09179)
[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/)

Official implementation of **HiFi-Mamba: Dual-Stream W-Laplacian Enhanced Mamba for High-Fidelity MRI Reconstruction**, accepted by **AAAI 2026**.

Hongli Chen*, Pengcheng Fang*, Yuxia Chen, Yingxuan Ren, Jing Hao, Fangfang Tang, Xiaohao Cai, Shanshan Shan, Feng Liu

Paper: [arXiv:2508.09179](https://arxiv.org/abs/2508.09179) | [PDF](https://arxiv.org/pdf/2508.09179)

This repository contains the public implementation of HiFi-Mamba for MRI reconstruction. The release keeps the core model and training code only; checkpoints, logs, masks, datasets, and experiment outputs are intentionally excluded.

## Contents

- `code/networks/mamba_sys_final.py`: core HiFi-Mamba / VSSM model implementation.
- `code/networks/vision_mamba.py`: model wrappers used by the training scripts.
- `code/train_fastmri.py`: distributed training entry for fastMRI.
- `code/train_cc359.py`: distributed training entry for CC359.
- `code/dataloaders/`: fastMRI and CC359 dataloaders plus mask generation utilities.
- `mamba/` and `causal-conv1d/`: local source dependencies used by the selective-scan backend.

## Installation

```bash
conda env create -f environment.yml
conda activate hifi_mamba

pip install -e causal-conv1d
pip install -e mamba
```

Install a PyTorch/CUDA build that matches your machine if the pinned CUDA version in `environment.yml` is not suitable for your system.

## Training

Run from the repository root. Data paths are required arguments; the repository does not include datasets or masks.

fastMRI:

```bash
torchrun --nproc_per_node=4 code/train_fastmri.py \
  --train_data_dir /path/to/fastMRI/singlecoil_train \
  --val_data_dir /path/to/fastMRI/singlecoil_val \
  --output_dir outputs/fastmri
```

CC359:

```bash
torchrun --nproc_per_node=4 code/train_cc359.py \
  --train_data_dir /path/to/CC359/Train \
  --val_data_dir /path/to/CC359/Val \
  --output_dir outputs/cc359
```

Useful options:

```bash
--resume /path/to/checkpoint.pth
--batch_size 4
--max_epoch 100
--acceleration 8
--mask_type equispaced
```

## Notes

The public dataloaders generate undersampling masks through `code/dataloaders/subsample.py`. No fixed private mask files are required or included.

By default, training artifacts are written under `outputs/`, which is ignored by git.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{chen2026hifimamba,
  title={HiFi-Mamba: Dual-Stream W-Laplacian Enhanced Mamba for High-Fidelity MRI Reconstruction},
  author={Chen, Hongli and Fang, Pengcheng and Chen, Yuxia and Ren, Yingxuan and Hao, Jing and Tang, Fangfang and Cai, Xiaohao and Shan, Shanshan and Liu, Feng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
