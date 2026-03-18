"""
Quick eval of GRPO-trained model vs baseline on escalation decisions.

Usage:
  python scripts/grpo_eval.py --adapter outputs/grpo_quick
"""

import argparse
import os
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from study3 import load_hotel, hotel_prompt, hotel_conditions, HOTEL

COST_RATIOS = [2, 4, 8, 10, 20, 50]
ESCALATE_PROMPT = HOTEL["escalate_prompt"]


def parse_decision(text):
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


def oracle_label(base_rate, R):
    return 1 if (1 - base_rate) > 1 / (1 + R) else 0


def eval_model(model, tokenizer, device, n_per_condition=20, seed=42):
    """Evaluate model on held-out hotel booking samples."""
    import random
    random.seed(seed + 999)  # different seed from training

    df = load_hotel()
    conditions = hotel_conditions(df)
    gt_col = HOTEL["gt_col"]

    results = []
    for cond in conditions:
        name = cond["name"]
        mask = cond["mask"]
        hint = cond["hint"]
        base_rate = cond["base_rate"]

        subset = df[mask]
        sample = subset.sample(n=min(n_per_condition, len(subset)), random_state=seed + 999)

        for _, row in sample.iterrows():
            scenario = hotel_prompt(row)
            gt = int(row[gt_col])

            # Simulate prediction
            pred_correct = random.random() < base_rate
            prediction = gt if pred_correct else (1 - gt)
            pred_text = "kept booking" if prediction == 1 else "cancelled"

            for R in COST_RATIOS:
                full_prompt = (
                    f"{hint}\n"
                    f"An LLM predicted: {pred_text}.\n"
                    f"Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.\n"
                    f"Output 0 to implement or 1 to escalate."
                )

                messages = [{"role": "user", "content": full_prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(device)

                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                decision = parse_decision(gen)

                optimal = oracle_label(base_rate, R)

                # Compute cost
                if decision is None:
                    cost = 1.0  # unparseable treated as escalate
                elif decision == 1:
                    cost = 1.0
                elif pred_correct:
                    cost = 0.0
                else:
                    cost = float(R)

                results.append({
                    "condition": name,
                    "base_rate": base_rate,
                    "cost_ratio": R,
                    "pred_correct": int(pred_correct),
                    "decision": decision,
                    "optimal": optimal,
                    "match_optimal": int(decision == optimal) if decision is not None else 0,
                    "cost": cost,
                })

    return results


def print_results(results, label):
    import pandas as pd
    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Overall
    match = df["match_optimal"].mean()
    avg_cost = df["cost"].mean()
    esc_rate = df[df["decision"] == 1].shape[0] / len(df) if len(df) > 0 else 0
    print(f"  Overall match with optimal: {match:.1%}")
    print(f"  Average cost: {avg_cost:.2f}")
    print(f"  Escalation rate: {esc_rate:.1%}")

    # By cost ratio
    print(f"\n  {'R':>3} {'Esc%':>6} {'Match':>6} {'Cost':>6}")
    print(f"  {'-'*25}")
    for R in COST_RATIOS:
        sub = df[df["cost_ratio"] == R]
        if len(sub) == 0:
            continue
        esc = (sub["decision"] == 1).mean()
        m = sub["match_optimal"].mean()
        c = sub["cost"].mean()
        print(f"  {R:>3} {esc:>6.1%} {m:>6.1%} {c:>6.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--n", type=int, default=20, help="Samples per condition")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Eval baseline
    print("\nLoading baseline model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    baseline_results = eval_model(model, tokenizer, device, n_per_condition=args.n)
    print_results(baseline_results, f"BASELINE ({args.model})")

    # Eval fine-tuned
    if args.adapter:
        print(f"\nLoading adapter from {args.adapter}...", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        ft_results = eval_model(model, tokenizer, device, n_per_condition=args.n)
        print_results(ft_results, f"GRPO-TRAINED ({args.adapter})")


if __name__ == "__main__":
    main()
