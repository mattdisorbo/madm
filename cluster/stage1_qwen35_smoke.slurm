#!/bin/bash
#SBATCH -J madm-qwen35-smoke
#SBATCH -o logs/stage1_qwen35_smoke.%j.out
#SBATCH -e logs/stage1_qwen35_smoke.%j.err
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

set -x
set -e

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

# Upgrade transformers to support qwen3_5_moe architecture
uv pip install --upgrade transformers

# Reinstall ROCm torch (in case transformers upgrade pulled in CPU torch)
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1

python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen3.5-35B-A3B'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

print(f'Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print(f'Loading model...')
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
print(f'Model loaded! Layers: {model.config.num_hidden_layers}')
print(f'Model type: {type(model).__name__}')

# Quick generation test
messages = [{'role': 'system', 'content': '/no_think'}, {'role': 'user', 'content': 'Say hello in one word.'}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
toks = tokenizer.encode(prompt, return_tensors='pt').to(device)
with torch.no_grad():
    out = model.generate(toks, max_new_tokens=10, do_sample=False)
print(f'Output: {tokenizer.decode(out[0][toks.shape[1]:], skip_special_tokens=True)}')
print('SMOKE TEST PASSED')
"
