#!/bin/bash
#SBATCH -J madm-grpo-hotel-prep
#SBATCH -o logs/grpo_prepare_hotel_%j.out
#SBATCH -e logs/grpo_prepare_hotel_%j.err
#SBATCH -p mi2101x
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
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR" logs/

uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --upgrade transformers typing_extensions datasets tqdm openai

echo "=== GRPO Data Preparation (HotelBookings) ==="
python scripts/grpo_prepare_hotel.py
echo "=== Done ==="
