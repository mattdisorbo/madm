"""
DPO training for cost-sensitive escalation using Qwen3.5-9B.

For each (base_rate, R) prompt, creates a preference pair:
  chosen = oracle-optimal decision ("0" or "1")
  rejected = wrong decision

Usage:
  python scripts/dpo_escalation.py [--quick] [--output-dir DIR]
"""

import argparse
import os
import re
import sys
import types
import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from study3 import load_hotel, hotel_prompt, hotel_conditions, HOTEL

COST_RATIOS = [2, 4, 8, 10, 20, 50]


def build_dataset(n_per_condition=50, seed=42):
    """Build DPO dataset with preference pairs."""
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
            gt = int(row[gt_col])
            pred_correct = random.random() < base_rate
            prediction = gt if pred_correct else (1 - gt)
            pred_text = "kept booking" if prediction == 1 else "cancelled"

            for R in COST_RATIOS:
                prompt = (
                    f"{hint}\n"
                    f"An LLM predicted: {pred_text}.\n"
                    f"Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.\n"
                    f"Output 0 to implement or 1 to escalate."
                )

                optimal = 1 if R * (1 - base_rate) > 1 else 0
                chosen = str(optimal)
                rejected = str(1 - optimal)

                rows.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "condition": name,
                    "base_rate": base_rate,
                    "cost_ratio": R,
                    "optimal": optimal,
                })

    random.shuffle(rows)

    # Print summary
    n_esc = sum(r["optimal"] == 1 for r in rows)
    print(f"Built {len(rows)} pairs ({n_esc} escalate, {len(rows)-n_esc} implement)")
    print(f"Escalation rate by R:")
    for R in COST_RATIOS:
        at_r = [r for r in rows if r["cost_ratio"] == R]
        esc = sum(r["optimal"] == 1 for r in at_r)
        print(f"  R={R:>2}: {esc}/{len(at_r)} escalate ({esc/len(at_r):.0%})")

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="outputs/dpo_escalation")
    args = parser.parse_args()

    if args.quick:
        n_per_condition = 10
        epochs = 1
        batch_size = 4
        max_steps = 50
    else:
        n_per_condition = 50
        epochs = 3
        batch_size = 4
        max_steps = -1

    print(f"Model: {args.model}", flush=True)
    print(f"Quick: {args.quick}", flush=True)

    dataset = build_dataset(n_per_condition=n_per_condition)
    print(f"Dataset: {len(dataset)} pairs", flush=True)

    # Split train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}", flush=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        eval_strategy="steps",
        bf16=True,
        gradient_accumulation_steps=2,
        report_to="none",
        max_length=512,
        max_prompt_length=480,
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

    print("Starting DPO training...", flush=True)
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}", flush=True)

    # Quick inline eval
    print("\n=== Quick Eval ===", flush=True)
    model.eval()
    from peft import PeftModel
    # Model is already the trained one

    correct = 0
    total = 0
    by_r = {R: {"correct": 0, "total": 0} for R in COST_RATIOS}

    for i in range(min(200, len(eval_dataset))):
        row = eval_dataset[i]
        messages = [{"role": "user", "content": row["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse decision
        gen_clean = re.sub(r'<think>.*?</think>', '', gen, flags=re.DOTALL).strip()
        match = re.search(r'[01]', gen_clean)
        if match:
            decision = int(match.group())
        else:
            decision = -1

        optimal = row["optimal"]
        is_correct = decision == optimal
        correct += int(is_correct)
        total += 1

        R = row["cost_ratio"]
        by_r[R]["total"] += 1
        by_r[R]["correct"] += int(is_correct)

        if i < 10:
            print(f"  R={R:>2} br={row['base_rate']:.2f} optimal={optimal} decision={decision} "
                  f"{'OK' if is_correct else 'WRONG'} gen=[{gen_clean[:30]}]", flush=True)

    print(f"\nOverall accuracy: {correct}/{total} ({correct/total:.1%})", flush=True)
    print(f"{'R':>3}  {'Acc':>6}  {'N':>4}", flush=True)
    for R in COST_RATIOS:
        d = by_r[R]
        if d["total"] > 0:
            print(f"{R:>3}  {d['correct']/d['total']:>6.0%}  {d['total']:>4}", flush=True)


if __name__ == "__main__":
    main()
