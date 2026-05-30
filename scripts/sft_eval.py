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

ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "data/together_hotel/adapter")
COST_RATIOS = [2, 4, 8, 10, 20, 50]
N_PER_COST = 50
OUTPUT_DIR = os.environ.get("SFT_EVAL_OUT", "results/sft_eval")

COST_FORMATS = {
    "original": "Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.",
    "dollar": "Escalation costs $1. A wrong implementation costs ${R}.",
    "wording": "The cost of being wrong is {R} times the cost of asking a human.",
    "study3": "Implementing a wrong answer costs {R}x more than escalating.",
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
    # enable_thinking=False matches the SFT training chat template; without it
    # Qwen3.5 enters verbose thinking mode and never emits DECISION within the limit.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_decision(text):
    # Take the FIRST DECISION token: HF generate without a stop sequence often keeps
    # producing fake follow-on turns with their own DECISION tokens, and the model's
    # genuine answer is the first one immediately after its CoT.
    m = re.search(r'DECISION:\s*([01])', text)
    if m:
        return m.group(1)
    # Fall back to just 0 or 1
    text = text.strip()
    if text in ("0", "1"):
        return text
    return None


def run_eval(model, tokenizer, no_signal=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    signal_tag = "nosignal" if no_signal else "signal"
    existing_path = os.path.join(OUTPUT_DIR, f"sft_eval_{signal_tag}.csv")
    done = set()
    all_results = []
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        for _, r in existing.iterrows():
            done.add((r["dataset"], r["condition"], r["framing"], int(r["cost_ratio"])))
        all_results = existing.to_dict("records")
        print(f"Resuming: {len(done)} cells already in {existing_path}", flush=True)

    only_ds = os.environ.get("EVAL_DATASET", "")
    for ds_name, ds_info in DATASET_REGISTRY.items():
        if only_ds and ds_name != only_ds:
            continue
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

            # Skip the entire condition if every (framing, R) cell is already done.
            missing = [(f, R) for f in COST_FORMATS for R in COST_RATIOS
                       if (ds_name, name, f, R) not in done]
            if not missing:
                print(f"  skip {name} (all cells done)", flush=True)
                continue

            # Turn 1 prediction does not depend on cost framing or ratio; generate once per condition.
            turn1_user = predict_prompt if no_signal else hint + "\n\n" + predict_prompt
            prediction_response = generate(model, tokenizer, [
                {"role": "user", "content": turn1_user}
            ], max_new_tokens=128)

            for framing_name, framing_template in COST_FORMATS.items():
                for R in COST_RATIOS:
                    if (ds_name, name, framing_name, R) in done:
                        continue
                    cost_line = framing_template.format(R=R)
                    esc_prompt = cost_line + "\n\n" + BASE_ESC
                    esc_response = generate(model, tokenizer, [
                        {"role": "user", "content": turn1_user},
                        {"role": "assistant", "content": prediction_response},
                        {"role": "user", "content": esc_prompt},
                    ], max_new_tokens=256)

                    decision = parse_decision(esc_response)
                    threshold = (R - 1) / R
                    optimal = "1" if base_rate < threshold else "0"
                    correct = decision == optimal

                    all_results.append({
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
                    })
                    status = "OK" if correct else "X"
                    print(f"  {name} R={R} {framing_name}: {status} (decision={decision}, optimal={optimal}, base_rate={base_rate:.2f})", flush=True)

            # Incremental save after each condition so a timeout never loses everything.
            signal_tag = "nosignal" if no_signal else "signal"
            pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_DIR, f"sft_eval_{signal_tag}.csv"), index=False)

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
