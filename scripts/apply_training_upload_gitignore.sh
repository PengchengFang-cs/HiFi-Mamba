#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  apply_training_upload_gitignore.sh --repo <repo-path>

Description:
  Append a managed .gitignore block for ML training/test-only GitHub uploads.
EOF
}

repo=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      repo="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$repo" ]]; then
  echo "Missing required argument: --repo" >&2
  usage
  exit 1
fi

if [[ ! -d "$repo" ]]; then
  echo "Repository path does not exist: $repo" >&2
  exit 1
fi

if ! git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository: $repo" >&2
  exit 1
fi

gitignore_path="$repo/.gitignore"
start_marker="# codex:training-upload-filter start"
end_marker="# codex:training-upload-filter end"

if [[ -f "$gitignore_path" ]] && grep -q "$start_marker" "$gitignore_path"; then
  echo "Managed block already exists in $gitignore_path"
  exit 0
fi

cat >> "$gitignore_path" <<'EOF'

# codex:training-upload-filter start
# Model checkpoints and binaries
*.pt
*.pth
*.ckpt
*.safetensors
*.onnx
*.bin
*.h5
*.hdf5
*.npy
*.npz
*.pkl
*.joblib

# Data and archives
data/
dataset/
datasets/
*.zip
*.tar
*.tar.gz
*.tgz
*.7z
*.rar

# Generated outputs and logs
output/
outputs/
result/
results/
run/
runs/
log/
logs/
wandb/
checkpoints/
weights/
artifacts/

# Plotting code and generated figures
*plot*.py
*viz*.py
*visual*.py
*figure*.py
*draw*.py
figures/
figure/
plots/
plot/
visualization/
visualisation/
*.png
*.jpg
*.jpeg
*.gif
*.bmp
*.tif
*.tiff
*.svg
*.pdf

# Notebook/cache/temporary
*.ipynb
.ipynb_checkpoints/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
tmp/
temp/
cache/

# Secrets and local environment
.env
.env.*
*.key
*.pem
*.secret

# OS/editor noise
.DS_Store
Thumbs.db
# codex:training-upload-filter end
EOF

echo "Appended managed training-upload filter block to $gitignore_path"
echo "Markers: '$start_marker' ... '$end_marker'"
