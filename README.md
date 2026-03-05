# HiFi-Mamba

Unified refactor of `MambaRecon`, `MambaReconV2`, and `MambaReconV3` into one repository with three explicit versions.

## Project Layout

- `versions/v1`: code lineage from `MambaRecon/code`.
- `versions/v2`: code lineage from `MambaReconV2/code`.
- `versions/v3`: code lineage from `MambaReconV3/code`.
- `third_party/mamba`: shared Mamba-SSM source dependency.
- `third_party/causal-conv1d`: shared causal-conv1d source dependency.
- `scripts/`: unified launch and upload-preflight utilities.

Only training/testing/inference-related code and configs are kept. Logs, model weights, figures, png outputs, and large artifacts are excluded.

## Version Summary

- `v1` (baseline): early MambaRecon training variants.
  - recommended train entry: `versions/v1/code/train_scan_out_8fast_mri_best_ddp.py`
- `v2` (split-conv/gate family): stronger modularized `mamba_sys_*` variants.
  - recommended train entry: `versions/v2/code/train_scan_out_8_fastmri_best_ddp.py`
- `v3` (multi-task + moe): includes MoE and additional ACDC/prostate/multi-brain experiments.
  - recommended train entry: `versions/v3/code/train_scan_out_8_fastmri_best_ddp.py`
  - recommended moe entry: `versions/v3/code/train_scan_out_8_mul_moe.py`

## Quick Start

1. Install shared third-party dependencies:

```bash
bash scripts/setup_third_party.sh
```

2. Run a training script (example: v3):

```bash
bash scripts/run_train.sh v3 train_scan_out_8_fastmri_best_ddp.py
```

3. Run an inference script (example: v2):

```bash
bash scripts/run_infer.sh v2 infer_single_mambarecon.py
```

## Upload Flow (training/test code only)

```bash
bash scripts/apply_training_upload_gitignore.sh --repo .
bash scripts/preflight_training_upload.sh --repo . --max-mb 20 --manifest .training_upload_manifest.txt
git reset
git add --pathspec-from-file=.training_upload_manifest.txt
git diff --cached --name-only
```

## Notes

- Some dataloaders still use absolute local data/mask paths from prior experiments.
- Before running experiments on a new machine, update dataset and mask paths in each version's dataloader files.
