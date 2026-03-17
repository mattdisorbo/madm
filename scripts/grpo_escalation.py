"""
GRPO training for cost-sensitive escalation.

Usage:
  # Quick test (tiny, ~5 min on 1 GPU):
  python scripts/grpo_escalation.py --quick

  # Full run:
  python scripts/grpo_escalation.py
"""

import argparse
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

# ---- Dataset construction ----

# Hotel booking hints: (name, base_rate, hint_text)
HINTS = [
    ("no_special_requests", 0.523, "when the guest made no special requests, 52% of bookings were kept"),
    ("lead_90_180", 0.554, "when the booking was made 90 to 180 days in advance, 55% of bookings were kept"),
    ("lead_30_90", 0.622, "when the booking was made 30 to 90 days in advance, 62% of bookings were kept"),
    ("no_prev_cancel", 0.661, "when the guest had no previous cancellations, 66% of bookings were kept"),
    ("no_deposit", 0.716, "when no deposit was required, 72% of bookings were kept"),
    ("has_special_requests", 0.783, "when the guest made special requests, 78% of bookings were kept"),
    ("lead_under_30", 0.818, "when the booking was made less than 30 days in advance, 82% of bookings were kept"),
    ("lead_under_30_special", 0.866, "when the booking was made less than 30 days in advance and the guest made special requests, 87% of bookings were kept"),
    ("lead_under_7", 0.906, "when the booking was made less than 7 days in advance, 91% of bookings were kept"),
    ("repeated_special", 0.950, "when the guest was a repeated guest who made special requests, 95% of bookings were kept"),
]

COST_RATIOS = [2, 4, 8, 10, 20, 50]


def oracle_label(base_rate, R):
    """Optimal escalation: escalate if P(wrong) > 1/(1+R)."""
    return 1 if (1 - base_rate) > 1 / (1 + R) else 0


def make_prompt(hint_text, R):
    return (
        f"A decision tree trained on this dataset finds that {hint_text}.\n\n"
        f"Implementing a wrong answer costs {R}x more than escalating.\n\n"
        f"Should we implement the prediction or escalate to a human?\n"
        f"Output 0 to implement or 1 to escalate."
    )


def build_dataset(n_repeats=1):
    """Build training dataset. Each (hint, cost_ratio) combo repeated n_repeats times."""
    rows = []
    for name, base_rate, hint_text in HINTS:
        for R in COST_RATIOS:
            prompt = make_prompt(hint_text, R)
            optimal = oracle_label(base_rate, R)
            for _ in range(n_repeats):
                rows.append({
                    "prompt": prompt,
                    "hint": name,
                    "base_rate": base_rate,
                    "cost_ratio": R,
                    "optimal": optimal,
                })
    return Dataset.from_list(rows)


# ---- Reward function ----

def reward_fn(completions, prompts=None, hint=None, base_rate=None, cost_ratio=None, optimal=None, **kwargs):
    """Reward: +1 for matching optimal, -1 for not."""
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        match = re.search(r'[01]', text)
        if match is None:
            rewards.append(-1.0)
            continue
        pred = int(match.group())
        opt = optimal[i] if isinstance(optimal, list) else optimal
        rewards.append(1.0 if pred == opt else -1.0)
    return rewards


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--quick", action="store_true", help="Tiny run for testing")
    parser.add_argument("--output-dir", default="outputs/grpo_escalation")
    args = parser.parse_args()

    if args.quick:
        n_repeats = 2       # 2 repeats x 10 hints x 6 ratios = 120 examples
        num_generations = 4
        epochs = 1
        batch_size = 4
        max_steps = 20
    else:
        n_repeats = 50      # 50 x 10 x 6 = 3000 examples
        num_generations = 8
        epochs = 3
        batch_size = 8
        max_steps = -1

    print(f"Model: {args.model}")
    print(f"Quick mode: {args.quick}")

    dataset = build_dataset(n_repeats=n_repeats)
    print(f"Dataset: {len(dataset)} examples")

    # Print oracle label distribution
    labels = [oracle_label(br, R) for _, br, _ in HINTS for R in COST_RATIOS]
    print(f"Oracle: {sum(labels)}/{len(labels)} escalate ({sum(labels)/len(labels):.0%})")

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
        max_completion_length=64,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=5e-6,
        logging_steps=1,
        save_steps=500,
        bf16=True,
        gradient_accumulation_steps=2,
        report_to="none",
        generation_kwargs={"do_sample": True, "temperature": 0.7},
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
