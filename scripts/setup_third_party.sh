#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m pip install -e "${REPO_ROOT}/third_party/causal-conv1d"
python3 -m pip install -e "${REPO_ROOT}/third_party/mamba"

echo "Installed third-party dependencies from ${REPO_ROOT}/third_party"
