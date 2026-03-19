"""
Robustness eval: novel base rates + repeated trials.

Tests:
1. Seen base rates with 10 repeated trials each (noise check)
2. Novel base rates never in training data
"""

import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

COST_RATIOS = [2, 4, 8, 10, 20, 50]

# Novel base rates not in any training data
NOVEL_BASE_RATES = [
    (0.55, "A decision tree trained on this dataset finds that 55% of predictions were correct."),
    (0.60, "A decision tree trained on this dataset finds that 60% of predictions were correct."),
    (0.65, "A decision tree trained on this dataset finds that 65% of predictions were correct."),
    (0.70, "A decision tree trained on this dataset finds that 70% of predictions were correct."),
    (0.75, "A decision tree trained on this dataset finds that 75% of predictions were correct."),
    (0.80, "A decision tree trained on this dataset finds that 80% of predictions were correct."),
    (0.85, "A decision tree trained on this dataset finds that 85% of predictions were correct."),
    (0.90, "A decision tree trained on this dataset finds that 90% of predictions were correct."),
    (0.95, "A decision tree trained on this dataset finds that 95% of predictions were correct."),
]

# Seen base rates from hotel bookings training
SEEN_BASE_RATES = [
    (0.523, "A decision tree trained on this dataset finds that when the guest made no special requests, 52% of bookings were kept."),
    (0.661, "A decision tree trained on this dataset finds that when the guest had no previous cancellations, 66% of bookings were kept."),
    (0.818, "A decision tree trained on this dataset finds that when the booking was made less than 30 days in advance, 82% of bookings were kept."),
    (0.950, "A decision tree trained on this dataset finds that when the guest was a repeated guest who made special requests, 95% of bookings were kept."),
]


def parse_decision(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    match = re.search(r'[01]', text)
    if match:
        return int(match.group())
    return None


def eval_set(model, tokenizer, device, base_rates, n_repeats=1, label=""):
    results = {}

    for base_rate, hint in base_rates:
        for R in COST_RATIOS:
            optimal = 1 if R * (1 - base_rate) > 1 else 0
            correct = 0
            total = 0
            decisions = []

            for _ in range(n_repeats):
                prompt = (
                    f"{hint}\n"
                    f"An LLM predicted: positive outcome.\n"
                    f"Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.\n"
                    f"Output 0 to implement or 1 to escalate."
                )
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                decision = parse_decision(gen)
                decisions.append(decision)
                is_correct = decision == optimal
                correct += int(is_correct)
                total += 1

            results[(base_rate, R)] = {
                "correct": correct,
                "total": total,
                "optimal": optimal,
                "decisions": decisions,
            }

    # Print results
    print(f"\n{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*60}", flush=True)

    total_correct = sum(v["correct"] for v in results.values())
    total_n = sum(v["total"] for v in results.values())
    print(f"  Overall: {total_correct}/{total_n} ({total_correct/total_n:.1%})", flush=True)

    # By R
    print(f"\n  {'R':>3}  {'Acc':>6}  {'N':>4}", flush=True)
    for R in COST_RATIOS:
        at_r = {k: v for k, v in results.items() if k[1] == R}
        c = sum(v["correct"] for v in at_r.values())
        n = sum(v["total"] for v in at_r.values())
        if n > 0:
            print(f"  {R:>3}  {c/n:>6.0%}  {n:>4}", flush=True)

    # By base rate
    print(f"\n  {'BR':>5}  {'R':>3}  {'Optimal':>7}  {'Acc':>6}  {'Decisions':>20}", flush=True)
    for (br, R), v in sorted(results.items()):
        dec_str = str(v["decisions"][:5])
        opt_str = "esc" if v["optimal"] == 1 else "impl"
        acc = v["correct"] / v["total"]
        print(f"  {br:>5.2f}  {R:>3}  {opt_str:>7}  {acc:>6.0%}  {dec_str}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print("Model loaded.", flush=True)

    # 1. Seen base rates with 10 repeats
    eval_set(model, tokenizer, device, SEEN_BASE_RATES, n_repeats=10,
             label="SEEN BASE RATES (10 repeats each)")

    # 2. Novel base rates
    eval_set(model, tokenizer, device, NOVEL_BASE_RATES, n_repeats=1,
             label="NOVEL BASE RATES (never in training)")


if __name__ == "__main__":
    main()
