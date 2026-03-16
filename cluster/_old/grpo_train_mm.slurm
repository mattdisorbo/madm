#!/bin/bash
#SBATCH -J madm-grpo-mm
#SBATCH -o logs/grpo_mm_%j.out
#SBATCH -e logs/grpo_mm_%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

set -e
cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --upgrade "transformers @ git+https://github.com/huggingface/transformers.git@main" typing_extensions datasets trl peft accelerate scikit-learn
uv pip install --reinstall torchvision --index-url https://download.pytorch.org/whl/rocm7.1

MODEL="Qwen/Qwen3.5-9B"
DATASET="data/grpo_dataset_mm"
OUTPUT="outputs/grpo_lora_mm"

python -u << PYEOF
import os, re, sys
from datasets import load_from_disk
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = "$MODEL"
DATASET_PATH = "$DATASET"
OUTPUT_DIR = "$OUTPUT"

def parse_decision(text):
    match = re.search(r'[01]', text.strip())
    if match: return int(match.group())
    if 'implement' in text.lower(): return 0
    if 'escalat' in text.lower(): return 1
    return None

def reward_fn(completions, ground_truth, prediction, **kwargs):
    rewards = []
    for completion, gt, pred in zip(completions, ground_truth, prediction):
        text = completion[-1]["content"] if isinstance(completion, list) else str(completion)
        escalate = parse_decision(text)
        pred_correct = (str(pred) == str(gt))
        if escalate is None: rewards.append(-1.0)
        elif pred_correct and escalate == 0: rewards.append(1.0)
        elif not pred_correct and escalate == 1: rewards.append(1.0)
        elif not pred_correct and escalate == 0: rewards.append(-1.0)
        elif pred_correct and escalate == 1: rewards.append(-0.5)
        else: rewards.append(0.0)
    return rewards

print(f"Loading dataset from {DATASET_PATH}...")
dataset = load_from_disk(DATASET_PATH)
print(f"Dataset size: {len(dataset)}")
dataset = dataset.map(lambda x: {"prompt": [{"role": "user", "content": x["prompt"]}]})

print(f"Loading model {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

try:
    from transformers import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL)
except ImportError:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL)

peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR, num_train_epochs=1,
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=5e-6, max_completion_length=8, num_generations=4,
    beta=0.05, logging_steps=10, save_steps=100, save_total_limit=2,
    bf16=True, gradient_checkpointing=True, max_grad_norm=1.0,
    warmup_ratio=0.1, report_to="none",
)

trainer = GRPOTrainer(model=model, processing_class=tokenizer, args=training_args,
    train_dataset=dataset, reward_funcs=reward_fn, peft_config=peft_config)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete.")
PYEOF

echo "=== Training done ==="
