#!/usr/bin/env bash
set -euo pipefail

# One-time environment setup for AMD HPC Fund cluster.
# Run from the login node after cloning the repo to $WORK.
#
# Usage:
#   cd "$WORK/madm"
#   bash cluster/setup.sh

PROJECT_DIR="${WORK}/madm"
export UV_CACHE_DIR="${WORK}/.cache/uv"

echo "==> Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> Installing Python and dependencies"
cd "${PROJECT_DIR}"
uv sync

echo "==> Installing PyTorch with ROCm support"
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1

echo "==> Creating log directory"
mkdir -p "${PROJECT_DIR}/logs"

echo "==> Done"
echo "    Activate with:  source ${PROJECT_DIR}/.venv/bin/activate"
