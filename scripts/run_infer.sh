#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_infer.sh <v1|v2|v3> <infer_script.py> [args...]

Example:
  run_infer.sh v2 infer_single_mambarecon.py --help
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

version="$1"
script="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${REPO_ROOT}/versions/${version}/code"

if [[ ! -d "${CODE_DIR}" ]]; then
  echo "Invalid version: ${version}" >&2
  exit 1
fi

if [[ ! -f "${CODE_DIR}/${script}" ]]; then
  echo "Script not found: ${CODE_DIR}/${script}" >&2
  exit 1
fi

cd "${CODE_DIR}"
exec python3 "${script}" "$@"
