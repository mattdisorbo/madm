"""
GRPO training for cost-sensitive escalation using real hotel booking data.

Uses actual hotel booking scenarios with real LLM predictions and ground truths.
The model learns when to escalate vs implement based on cost tradeoffs.

Reward structure:
  - Escalate: -1 (pay labor cost)
  - Implement correct prediction: 0 (free)
  - Implement wrong prediction: -R (pay wrong-answer cost)
  - Unparseable output: -1

Usage:
  # Quick test (~5 min on 1 GPU):
  python scripts/grpo_escalation.py --quick

  # Full run:
  python scripts/grpo_escalation.py
"""

import argparse
import os
import re
import sys
import types
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

# Import dataset helpers from study3
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from study3 import load_hotel, hotel_prompt, hotel_conditions, HOTEL


def patch_rotary_for_rocm(model):
    """Patch rotary embeddings to compute on CPU (avoids ROCm HIP crash)."""
    patched = 0
    for name, module in model.named_modules():
        if "rotary" in name.lower() or type(module).__name__.endswith("RotaryEmbedding"):
            @torch.no_grad()
            def _safe_forward(self, x, position_ids):
                inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
                    position_ids.shape[0], -1, 1)
                position_ids_expanded = position_ids[:, None, :].float()
                device = x.device
                freqs = (inv_freq_expanded.cpu() @ position_ids_expanded.cpu()
                         ).transpose(1, 2).to(device)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
                return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
            module.forward = types.MethodType(_safe_forward, module)
            patched += 1
    if patched:
        print(f"Patched {patched} rotary embedding instances for ROCm", flush=True)
    return model


# ---- Dataset construction ----

COST_RATIOS = [2, 4, 8, 10, 20, 50]
PREDICT_PROMPT = HOTEL["predict_prompt"]
ESCALATE_PROMPT = HOTEL["escalate_prompt"]


def build_dataset(n_per_condition=50, seed=42):
    """Build training dataset from real hotel booking data.

    For each condition, samples n_per_condition bookings, creates the scenario
    prompt with hint, and records the ground truth. Each sample is expanded
    across all cost ratios. The LLM prediction is simulated by using the
    ground truth with probability = base_rate (matching the hint's stated
    accuracy), so the model sees realistic correct/wrong predictions.
    """
    import random
    random.seed(seed)

    df = load_hotel()
    conditions = hotel_conditions(df)
    gt_col = HOTEL["gt_col"]

    rows = []
    for cond in conditions:
        name = cond["name"]
        mask = cond["mask"]
        hint = cond["hint"]
        base_rate = cond["base_rate"]

        subset = df[mask]
        sample = subset.sample(n=min(n_per_condition, len(subset)), random_state=seed)

        for _, row in sample.iterrows():
            scenario = hotel_prompt(row)
            gt = int(row[gt_col])

            # Simulate a prediction: correct with probability = base_rate
            pred_correct = random.random() < base_rate
            prediction = gt if pred_correct else (1 - gt)

            pred_text = "kept booking" if prediction == 1 else "cancelled"
            prompt_with_hint = f"{scenario}\n\n{hint}\n\n{PREDICT_PROMPT}"

            for R in COST_RATIOS:
                esc_prompt = (
                    f"Implementing a wrong answer costs {R}x more than escalating.\n\n"
                    f"{ESCALATE_PROMPT}"
                )
                # The full prompt is the escalation decision prompt
                # (scenario + hint + predict prompt is context,
                #  the model's "prediction" is given, then ask to escalate or not)
                full_prompt = (
                    f"{scenario}\n\n{hint}\n\n"
                    f"An LLM predicted: {pred_text}.\n\n"
                    f"{esc_prompt}"
                )
                rows.append({
                    "prompt": full_prompt,
                    "condition": name,
                    "base_rate": base_rate,
                    "cost_ratio": R,
                    "ground_truth": gt,
                    "prediction": prediction,
                    "pred_correct": int(pred_correct),
                })

    print(f"Built {len(rows)} examples "
          f"({len(conditions)} conditions x {n_per_condition} samples x {len(COST_RATIOS)} cost ratios)")
    correct = sum(r["pred_correct"] for r in rows) / len(rows)
    print(f"Prediction accuracy: {correct:.1%}")

    return Dataset.from_list(rows)


# ---- Reward function ----

_reward_log_count = 0


def parse_decision(text):
    """Parse escalation decision from model output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    match = re.search(r'[01]', text)
    if match is not None:
        return int(match.group())
    low = text.lower()
    if 'implement' in low and 'escalat' not in low:
        return 0
    if 'escalat' in low and 'implement' not in low:
        return 1
    return None


def reward_fn(completions, pred_correct=None, cost_ratio=None, **kwargs):
    """Cost-based reward using actual prediction correctness.

    - Escalate: -1 (labor cost)
    - Implement + prediction correct: 0 (no cost)
    - Implement + prediction wrong: -R (wrong-answer cost)
    - Unparseable: -1
    """
    global _reward_log_count
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion

        if _reward_log_count < 5:
            decision = parse_decision(text)
            print(f"  [DEBUG] text=[{text[:80]}] decision={decision}", flush=True)
            _reward_log_count += 1

        decision = parse_decision(text)
        if decision is None:
            rewards.append(-1.0)
            continue

        correct = pred_correct[i] if isinstance(pred_correct, list) else pred_correct
        R = cost_ratio[i] if isinstance(cost_ratio, list) else cost_ratio
        R = float(R)

        if decision == 1:  # escalate
            rewards.append(-1.0)
        elif correct:  # implement correct prediction
            rewards.append(0.0)
        else:  # implement wrong prediction
            rewards.append(-R)
    return rewards


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--quick", action="store_true", help="Tiny run for testing")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--output-dir", default="outputs/grpo_escalation")
    args = parser.parse_args()

    if args.quick:
        n_per_condition = 10
        num_generations = 16
        epochs = 3
        batch_size = 4
        max_steps = args.max_steps or 100
    else:
        n_per_condition = 50   # 50 x 10 x 6 = 3000 examples
        num_generations = 16
        epochs = 1
        batch_size = 8
        max_steps = -1

    print(f"Model: {args.model}")
    print(f"Quick mode: {args.quick}", flush=True)

    dataset = build_dataset(n_per_condition=n_per_condition)
    print(f"Dataset: {len(dataset)} examples", flush=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=num_generations,
        max_completion_length=32,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=5e-6,
        logging_steps=1,
        save_steps=500,
        bf16=True,
        gradient_accumulation_steps=2,
        report_to="none",
        generation_batch_size=16,
        generation_kwargs={"do_sample": True, "temperature": 1.2},
    )

    # Load model manually so we can patch rotary embeddings for ROCm
    print(f"Loading model {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = patch_rotary_for_rocm(model)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting training...", flush=True)
    trainer.train()

    trainer.save_model(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
