"""
DPO v2: Train on full study3 multi-turn prompts.

Uses actual scenario + prediction + escalation context from study3 CSVs.
Preference pairs: chosen=oracle decision, rejected=wrong decision.

Usage:
  python scripts/dpo_escalation_v2.py [--output-dir DIR]
"""

import argparse
import glob
import os
import re
import sys
import random
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from study3 import hotel_conditions, load_hotel

COST_RATIOS = [2, 4, 8, 10, 20, 50]
RESULTS_DIR = "results/study3"


def build_dataset(n_per_condition=20, seed=42):
    """Build DPO dataset from real study3 CSVs with multi-turn context."""
    random.seed(seed)

    df_full = load_hotel()
    conditions = hotel_conditions(df_full)
    cond_br = {c["name"]: c["base_rate"] for c in conditions}

    rows = []
    for cond in conditions:
        name = cond["name"]
        base_rate = cond["base_rate"]

        # Load the nothink CSV for this condition
        csv = f"{RESULTS_DIR}/HotelBookings_{name}_nothink_Qwen3.5-9B.csv"
        if not os.path.exists(csv):
            print(f"  Skipping {name}: no CSV", flush=True)
            continue

        df = pd.read_csv(csv)
        sample = df.head(n_per_condition)

        for _, row in sample.iterrows():
            prompt_text = row["prompt"]
            thought = str(row["thought"])

            for R in COST_RATIOS:
                optimal = 1 if R * (1 - base_rate) > 1 else 0
                chosen = str(optimal)
                rejected = str(1 - optimal)

                esc_prompt = (
                    f"Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.\n\n"
                    f"You predicted the outcome above.\n"
                    f"TASK: Should we implement your prediction, or escalate to a human?\n"
                    f"Explain your reasoning in 1-2 sentences, then conclude with exactly "
                    f"'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
                )

                # Multi-turn prompt as a list of messages
                # DPO expects prompt + chosen/rejected as strings
                # We'll format as chat template
                prompt_msgs = [
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": thought},
                    {"role": "user", "content": esc_prompt},
                ]

                rows.append({
                    "prompt": prompt_msgs,
                    "chosen": chosen,
                    "rejected": rejected,
                    "condition": name,
                    "base_rate": base_rate,
                    "cost_ratio": R,
                    "optimal": optimal,
                })

    random.shuffle(rows)

    n_esc = sum(r["optimal"] == 1 for r in rows)
    print(f"Built {len(rows)} pairs ({n_esc} escalate, {len(rows)-n_esc} implement)", flush=True)
    print(f"Escalation rate by R:", flush=True)
    for R in COST_RATIOS:
        at_r = [r for r in rows if r["cost_ratio"] == R]
        esc = sum(r["optimal"] == 1 for r in at_r)
        print(f"  R={R:>2}: {esc}/{len(at_r)} escalate ({esc/len(at_r):.0%})", flush=True)

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output-dir", default="outputs/dpo_v2")
    parser.add_argument("--n", type=int, default=20, help="Samples per condition")
    args = parser.parse_args()

    print(f"Model: {args.model}", flush=True)

    dataset = build_dataset(n_per_condition=args.n)
    print(f"Dataset: {len(dataset)} pairs", flush=True)

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}", flush=True)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        eval_strategy="steps",
        bf16=True,
        gradient_accumulation_steps=4,
        report_to="none",
        max_length=1024,
    )

    print(f"Loading model {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    # Qwen3.5 mm_token_type_ids fix
    _orig_validate = model._validate_model_kwargs
    def _patched_validate(model_kwargs):
        model_kwargs.pop("mm_token_type_ids", None)
        return _orig_validate(model_kwargs)
    model._validate_model_kwargs = _patched_validate

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO v2 training...", flush=True)
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}", flush=True)

    # Quick eval
    print("\n=== Quick Eval ===", flush=True)
    model.eval()

    correct = 0
    total = 0
    for i in range(min(100, len(eval_dataset))):
        row = eval_dataset[i]
        messages = row["prompt"] + [{"role": "assistant", "content": ""}]
        messages = row["prompt"]  # just the prompt messages
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        decision_match = re.search(r'DECISION:\s*([01])', gen)
        if decision_match:
            decision = int(decision_match.group(1))
        else:
            m = re.search(r'[01]', gen)
            decision = int(m.group()) if m else -1

        is_correct = decision == row["optimal"]
        correct += int(is_correct)
        total += 1

        if i < 5:
            print(f"  R={row['cost_ratio']:>2} br={row['base_rate']:.2f} optimal={row['optimal']} "
                  f"decision={decision} {'OK' if is_correct else 'WRONG'} gen=[{gen[:40]}]", flush=True)

    print(f"\nOverall: {correct}/{total} ({correct/total:.1%})", flush=True)


if __name__ == "__main__":
    main()
