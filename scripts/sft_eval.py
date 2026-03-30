"""
Evaluate SFT fine-tuned model on all datasets, cost ratios, and framings.
Saves results to results/sft_eval/ for verification.

Usage:
  python scripts/sft_eval.py [--no-signal]
"""

import argparse
import os
import re
import sys
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study3 import (
    load_hotel, hotel_conditions,
    load_lending, lending_conditions,
    load_wiki, wiki_conditions,
    load_movielens, movielens_conditions,
)

ADAPTER_DIR = "data/together_hotel/adapter"
COST_RATIOS = [2, 4, 8, 10, 20, 50]
N_PER_COST = 50
OUTPUT_DIR = "results/sft_eval"

COST_FORMATS = {
    "original": "Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.",
    "dollar": "Escalation costs $1. A wrong implementation costs ${R}.",
    "wording": "The cost of being wrong is {R} times the cost of asking a human.",
}

DATASET_REGISTRY = {
    "HotelBookings": {"load": load_hotel, "conditions": hotel_conditions},
    "LendingClub": {"load": load_lending, "conditions": lending_conditions},
    "WikipediaToxicity": {"load": load_wiki, "conditions": wiki_conditions},
    "MovieLens": {"load": load_movielens, "conditions": movielens_conditions},
}

BASE_ESC = (
    "You predicted the outcome above.\n"
    "TASK: Should we implement your prediction, or escalate to a human?\n"
    "Explain your reasoning in 1-2 sentences, then conclude with exactly "
    "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
)


def load_model():
    print("Loading base model and adapter...")
    config = json.load(open(os.path.join(ADAPTER_DIR, "adapter_config.json")))
    base_model = config["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    print(f"Loaded {base_model} + LoRA adapter")
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=256):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_decision(text):
    decisions = re.findall(r'DECISION:\s*([01])', text)
    if decisions:
        return decisions[-1]
    # Fall back to just 0 or 1
    text = text.strip()
    if text in ("0", "1"):
        return text
    return None


def run_eval(model, tokenizer, no_signal=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for ds_name, ds_info in DATASET_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {ds_name}")
        print(f"{'='*60}")

        df_full = ds_info["load"]()
        conditions = ds_info["conditions"](df_full)
        # Pick conditions that span the base rate range
        # Use all conditions, sample N_PER_COST scenarios per cost ratio
        for cond in conditions:
            name = cond["name"]
            hint = cond["hint"]
            base_rate = cond["base_rate"]
            predict_prompt = DATASET_REGISTRY[ds_name].get("predict_prompt",
                f"TASK: Predict the outcome. Explain in 1 sentence. Conclude with 'PREDICTION: 1' or 'PREDICTION: 0'.")

            for framing_name, framing_template in COST_FORMATS.items():
                for R in COST_RATIOS:
                    # Build prompt
                    if no_signal:
                        scenario_text = "You are making a prediction."
                    else:
                        # Use the hint as the scenario
                        scenario_text = hint

                    # Turn 1: prediction (we use a dummy since we're testing escalation)
                    turn1_user = scenario_text + "\n\n" + predict_prompt if not no_signal else predict_prompt
                    prediction_response = generate(model, tokenizer, [
                        {"role": "user", "content": turn1_user}
                    ], max_new_tokens=128)

                    # Turn 2: escalation
                    cost_line = framing_template.format(R=R)
                    esc_prompt = cost_line + "\n\n" + BASE_ESC

                    esc_response = generate(model, tokenizer, [
                        {"role": "user", "content": turn1_user},
                        {"role": "assistant", "content": prediction_response},
                        {"role": "user", "content": esc_prompt},
                    ], max_new_tokens=256)

                    decision = parse_decision(esc_response)

                    # Optimal decision
                    threshold = (R - 1) / R
                    optimal = "1" if base_rate < threshold else "0"
                    correct = decision == optimal

                    result = {
                        "dataset": ds_name,
                        "condition": name,
                        "base_rate": base_rate,
                        "cost_ratio": R,
                        "framing": framing_name,
                        "threshold": threshold,
                        "optimal": optimal,
                        "decision": decision,
                        "correct": correct,
                        "no_signal": no_signal,
                        "prediction_response": prediction_response,
                        "esc_response": esc_response,
                    }
                    all_results.append(result)

                    status = "✓" if correct else "✗"
                    print(f"  {name} R={R} {framing_name}: {status} (decision={decision}, optimal={optimal}, base_rate={base_rate:.2f})")

    # Save results
    df = pd.DataFrame(all_results)
    signal_tag = "nosignal" if no_signal else "signal"
    out_path = os.path.join(OUTPUT_DIR, f"sft_eval_{signal_tag}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} results to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary = df.groupby(["dataset", "framing"]).agg(
        n=("correct", "count"),
        accuracy=("correct", "mean"),
    ).reset_index()
    summary["accuracy"] = (summary["accuracy"] * 100).round(1).astype(str) + "%"
    print(summary.pivot(index="dataset", columns="framing", values="accuracy").to_string())
    print(f"\nOverall: {df.correct.mean()*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-signal", action="store_true", help="Run without signal (ablation)")
    args = parser.parse_args()

    model, tokenizer = load_model()
    run_eval(model, tokenizer, no_signal=args.no_signal)
