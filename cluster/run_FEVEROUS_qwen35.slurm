#\!/bin/bash
#SBATCH -J madm-run_FEVEROUS
#SBATCH -o logs/run_FEVEROUS.%j.out
#SBATCH -e logs/run_FEVEROUS.%j.err
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

uv pip install --upgrade transformers
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1

python scripts/run_FEVEROUS.py
