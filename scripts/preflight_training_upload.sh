#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  preflight_training_upload.sh --repo <repo-path> [--max-mb 20] [--manifest <path>] [--fail-on-untracked]

Description:
  Audit a git repo before GitHub upload and generate a manifest that keeps only
  training/testing-related source/config/docs files.

Exit codes:
  0  success
  2  blocked files found
  3  no files selected for upload
EOF
}

repo=""
max_mb=20
manifest=""
fail_on_untracked=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      repo="${2:-}"
      shift 2
      ;;
    --max-mb)
      max_mb="${2:-}"
      shift 2
      ;;
    --manifest)
      manifest="${2:-}"
      shift 2
      ;;
    --fail-on-untracked)
      fail_on_untracked=1
      shift
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

if ! [[ "$max_mb" =~ ^[0-9]+$ ]] || [[ "$max_mb" -le 0 ]]; then
  echo "--max-mb must be a positive integer, got: $max_mb" >&2
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

repo="$(cd "$repo" && pwd)"
max_bytes=$((max_mb * 1024 * 1024))
tmp_manifest="$(mktemp)"
trap 'rm -f "$tmp_manifest"' EXIT

if [[ -z "$manifest" ]]; then
  manifest="$repo/.training_upload_manifest.txt"
fi

violations=()
untracked_hits=()
included_count=0
excluded_count=0

is_root_keep_file() {
  local file="$1"
  case "$file" in
    README|README.md|LICENSE|LICENCE|Makefile|Dockerfile|requirements.txt|requirements-dev.txt|pyproject.toml|setup.py|setup.cfg|environment.yml|environment.yaml|.gitignore|.gitattributes)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_allowed_ext() {
  local path="$1"
  [[ "$path" =~ \.(py|sh|yaml|yml|json|toml|ini|cfg|md|txt|rst|c|cc|cpp|cu|cuh|h|hpp)$ ]]
}

is_blocked_ext() {
  local path="$1"
  [[ "$path" =~ \.(pt|pth|ckpt|safetensors|onnx|bin|h5|hdf5|npy|npz|pkl|joblib|zip|tar|gz|bz2|xz|7z|rar|png|jpg|jpeg|gif|bmp|tif|tiff|svg|pdf|mp4|mov|avi|mkv)$ ]]
}

is_blocked_dir() {
  local path="$1"
  [[ "$path" =~ (^|/)(data|dataset|datasets|checkpoint|checkpoints|weight|weights|artifact|artifacts|output|outputs|result|results|log|logs|wandb|run|runs|figure|figures|plot|plots|visualization|visualisation|notebook|notebooks|tmp|temp|cache|__pycache__|\.ipynb_checkpoints)(/|$) ]]
}

is_plot_code() {
  local path="$1"
  [[ "$path" =~ (^|/).*?(plot|viz|visual|figure|draw).*\.py$ ]]
}

is_training_related() {
  local path="$1"
  [[ "$path" =~ (^|/)(train|training|test|testing|eval|evaluation|infer|inference|benchmark|model|models|config|configs|dataset|dataloader|loader|loss|metric|engine|runner|main|script|scripts|version|versions|environment|readme|license)(_|-|/|\.) ]] || \
  [[ "$path" =~ (^|/)(src|tests|models|configs|train|training|eval|evaluation|inference|utils|network|networks|dataloader|dataloaders|data_loading|third_party|mamba|mamba_ssm|causal-conv1d|causal_conv1d|csrc|scripts|docs|versions)(/|$) ]]
}

while IFS= read -r -d '' file; do
  abs="$repo/$file"
  [[ -e "$abs" ]] || continue

  base="${file##*/}"
  path_lc="$(printf '%s' "$file" | tr '[:upper:]' '[:lower:]')"
  size_bytes="$(wc -c < "$abs")"

  if (( size_bytes > max_bytes )); then
    violations+=("large file > ${max_mb}MB: $file")
    continue
  fi
  if is_blocked_dir "$path_lc"; then
    violations+=("blocked directory pattern: $file")
    continue
  fi
  if is_blocked_ext "$path_lc"; then
    violations+=("blocked extension: $file")
    continue
  fi
  if is_plot_code "$path_lc"; then
    violations+=("plot/visualization code: $file")
    continue
  fi

  if is_root_keep_file "$base"; then
    printf '%s\n' "$file" >> "$tmp_manifest"
    included_count=$((included_count + 1))
    continue
  fi

  if is_allowed_ext "$path_lc" && is_training_related "$path_lc"; then
    printf '%s\n' "$file" >> "$tmp_manifest"
    included_count=$((included_count + 1))
  else
    excluded_count=$((excluded_count + 1))
  fi
done < <(git -C "$repo" ls-files -z)

while IFS= read -r -d '' file; do
  path_lc="$(printf '%s' "$file" | tr '[:upper:]' '[:lower:]')"
  if is_blocked_ext "$path_lc" || is_blocked_dir "$path_lc" || is_plot_code "$path_lc"; then
    untracked_hits+=("$file")
  fi
done < <(git -C "$repo" ls-files --others --exclude-standard -z)

if (( ${#untracked_hits[@]} > 0 )) && (( fail_on_untracked == 1 )); then
  for file in "${untracked_hits[@]}"; do
    violations+=("untracked blocked candidate: $file")
  done
fi

if (( ${#violations[@]} > 0 )); then
  echo "Preflight failed: blocked files detected." >&2
  printf ' - %s\n' "${violations[@]}" >&2
  if (( ${#untracked_hits[@]} > 0 )) && (( fail_on_untracked == 0 )); then
    echo "Untracked blocked candidates (warning only):" >&2
    printf ' - %s\n' "${untracked_hits[@]}" >&2
  fi
  exit 2
fi

sort -u "$tmp_manifest" > "$manifest"

if [[ ! -s "$manifest" ]]; then
  echo "Preflight failed: no files selected for upload. Check filtering rules." >&2
  exit 3
fi

echo "Preflight passed."
echo "Included files: $included_count"
echo "Excluded files: $excluded_count"
echo "Manifest: $manifest"

if (( ${#untracked_hits[@]} > 0 )) && (( fail_on_untracked == 0 )); then
  echo "Untracked blocked candidates (warning only):"
  printf ' - %s\n' "${untracked_hits[@]}"
fi
