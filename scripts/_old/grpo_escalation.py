"""
GRPO Training for Escalation Calibration.

Fine-tunes a LoRA adapter on Qwen3.5-9B using GRPO so the model learns
when to escalate vs implement. The prediction step stays frozen; only
the escalation decision is trained.

Usage:
    python scripts/grpo_escalation.py [MODEL] [--dataset-path PATH] [--output-dir PATH]
"""

import os, re, sys
from datasets import load_from_disk
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../data/grpo_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "../outputs/grpo_lora")

for i, arg in enumerate(sys.argv):
    if arg == "--dataset-path" and i + 1 < len(sys.argv):
        DATASET_PATH = sys.argv[i + 1]
    if arg == "--output-dir" and i + 1 < len(sys.argv):
        OUTPUT_DIR = sys.argv[i + 1]


# --- Reward function ---
def parse_decision(text):
    match = re.search(r'[01]', text.strip())
    if match:
        return int(match.group())
    low = text.lower()
    if 'implement' in low:
        return 0
    if 'escalat' in low:
        return 1
    return None


def reward_fn(completions, ground_truth, prediction, **kwargs):
    """
    Custom reward for GRPO: scores escalation decisions based on
    whether the frozen prediction was correct.

    Args:
        completions: list of model outputs (strings)
        ground_truth: list of ground truth labels (strings)
        prediction: list of frozen-model predictions (strings)

    Returns:
        list of float rewards
    """
    rewards = []
    for completion, gt, pred in zip(completions, ground_truth, prediction):
        # Extract the completion text
        if isinstance(completion, list):
            # Chat format: extract content from last message
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)

        escalate = parse_decision(text)
        pred_correct = (str(pred) == str(gt))

        if escalate is None:
            rewards.append(-1.0)   # unparseable
        elif pred_correct and escalate == 0:
            rewards.append(1.0)    # correct + implement = good
        elif not pred_correct and escalate == 1:
            rewards.append(1.0)    # wrong + escalate = good
        elif not pred_correct and escalate == 0:
            rewards.append(-1.0)   # wrong + implement = bad
        elif pred_correct and escalate == 1:
            rewards.append(-0.5)   # correct + escalate = wasteful
        else:
            rewards.append(0.0)

    return rewards


# --- Main ---
if __name__ == "__main__":
    print(f"Loading dataset from {DATASET_PATH}...", flush=True)
    dataset = load_from_disk(DATASET_PATH)
    print(f"Dataset size: {len(dataset)}", flush=True)

    # Convert prompt strings to chat format for GRPOTrainer
    def format_prompt(example):
        example["prompt"] = [{"role": "user", "content": example["prompt"]}]
        return example

    dataset = dataset.map(format_prompt)

    print(f"Loading model {MODEL}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load text-only backbone to avoid multimodal 3D position encoding issues
    # Qwen3.5 is natively multimodal; AutoModelForCausalLM loads the full vision
    # model which crashes with TRL's generation pipeline.
    try:
        from transformers import Qwen3_5ForCausalLM
        print("Using Qwen3_5ForCausalLM (text-only backbone)...", flush=True)
        model = Qwen3_5ForCausalLM.from_pretrained(MODEL)
    except ImportError:
        print("Qwen3_5ForCausalLM not available, using AutoModelForCausalLM...", flush=True)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL)

    # LoRA config: only train attention projections
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # GRPO training config
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_completion_length=8,   # just "0" or "1"
        num_generations=4,         # GRPO compares within group
        beta=0.05,                 # KL coefficient
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        report_to="none",
    )

    print("Initializing GRPOTrainer...", flush=True)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
    )

    print("Starting GRPO training...", flush=True)
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}...", flush=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete.", flush=True)
