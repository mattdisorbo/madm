#!/bin/bash
#SBATCH -J gpu-test
#SBATCH -o logs/gpu_test_%j.out
#SBATCH -e logs/gpu_test_%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00

set -e
cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export TORCH_BLAS_PREFER_HIPBLASLT=0
export UV_HTTP_TIMEOUT=600

mkdir -p logs/

# Current torch is broken (2.10.0+rocm7.1). Try rocm6.2.4 which is stable on MI210.
echo "Before:"
python -c "import torch; print(f'torch={torch.__version__}, hip={torch.version.hip}')"

echo "Installing torch for rocm6.2.4..."
uv pip install torch==2.6.0+rocm6.2.4 --index-url https://download.pytorch.org/whl/rocm6.2.4
uv pip install --upgrade transformers typing_extensions

echo "After:"
python -c "import torch; print(f'torch={torch.__version__}, hip={torch.version.hip}')"

# Simple test
echo "=== TEST: Simple matmul ==="
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
a = torch.randn(4, 64, 1, device='cuda')
b = torch.randn(4, 1, 128, device='cuda')
c = a @ b
print(f'OK: matmul {c.shape}')
"
echo "=== Matmul passed ==="

# Model test with Qwen3-8B
echo "=== TEST: Qwen3-8B forward ==="
python scripts/gpu_test.py Qwen/Qwen3-8B || echo "FAILED: Qwen3-8B"

echo "=== DONE ==="
