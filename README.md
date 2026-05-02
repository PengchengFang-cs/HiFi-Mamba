# HiFi-Mamba

This repository contains the public V2 implementation of HiFi-Mamba for MRI reconstruction. The release keeps the core model and training code only; checkpoints, logs, masks, datasets, and experiment outputs are intentionally excluded.

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
