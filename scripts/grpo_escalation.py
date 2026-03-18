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
                full_prompt = (
                    f"{hint}\n"
                    f"An LLM predicted: {pred_text}.\n"
                    f"Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.\n"
                    f"Output 0 to implement or 1 to escalate."
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


def format_reward_fn(completions, **kwargs):
    """Reward for clean output format: +1 if output is just '0' or '1', 0 otherwise."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if text in ("0", "1"):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


_step_count = 0
_decision_log = []  # collect (decision, R, pred_correct) for periodic reporting


def cost_reward_fn(completions, pred_correct=None, cost_ratio=None, **kwargs):
    """Cost-based reward using actual prediction correctness.

    - Escalate: -1 (labor cost)
    - Implement + prediction correct: 0 (no cost)
    - Implement + prediction wrong: -R (wrong-answer cost)
    - Unparseable: -1
    """
    global _reward_log_count, _step_count, _decision_log
    _step_count += 1
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion

        if _reward_log_count < 10:
            decision = parse_decision(text)
            print(f"  [DEBUG] text=[{text[:60]}] decision={decision}", flush=True)
            _reward_log_count += 1

        decision = parse_decision(text)
        if decision is None:
            rewards.append(-1.0)
            _decision_log.append((None, None, None))
            continue

        correct = pred_correct[i] if isinstance(pred_correct, list) else pred_correct
        R = cost_ratio[i] if isinstance(cost_ratio, list) else cost_ratio
        R = float(R)

        _decision_log.append((decision, R, correct))

        if decision == 1:  # escalate
            rewards.append(-1.0)
        elif correct:  # implement correct prediction
            rewards.append(0.0)
        else:  # implement wrong prediction
            rewards.append(-R)

    # Periodic report every 50 steps
    if _step_count % 50 == 0 and _decision_log:
        recent = _decision_log[-500:]  # last ~500 decisions
        parseable = [(d, r, c) for d, r, c in recent if d is not None]
        unparseable = len(recent) - len(parseable)
        print(f"\n  === Step {_step_count} Report ===", flush=True)
        print(f"  Parseable: {len(parseable)}/{len(recent)} ({len(parseable)/len(recent):.0%})", flush=True)
        if parseable:
            esc_total = sum(1 for d, _, _ in parseable if d == 1)
            print(f"  Overall esc rate: {esc_total/len(parseable):.0%}", flush=True)
            print(f"  {'R':>3}  {'Esc%':>6}  {'N':>4}", flush=True)
            for R in [2, 4, 8, 10, 20, 50]:
                at_r = [(d, c) for d, r, c in parseable if r == R]
                if at_r:
                    esc = sum(1 for d, _ in at_r if d == 1)
                    print(f"  {R:>3}  {esc/len(at_r):>6.0%}  {len(at_r):>4}", flush=True)
            # Sample outputs
            sample_texts = []
            for d, r, c in parseable[-5:]:
                sample_texts.append(f"d={d} R={r} correct={c}")
            print(f"  Recent: {sample_texts}", flush=True)
        print(flush=True)

    return rewards


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
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
        epochs = 3
        batch_size = 4
        max_steps = 500

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
        max_completion_length=8,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=2e-5,
        beta=0.1,  # KL penalty to prevent entropy collapse
        logging_steps=1,
        save_steps=500,
        bf16=True,
        gradient_accumulation_steps=2,
        report_to="none",
        generation_batch_size=16,
        chat_template_kwargs={"enable_thinking": False},
        generation_kwargs={"do_sample": True, "temperature": 0.8},
    )

    # Load model manually so we can patch for ROCm and fix multimodal issues
    print(f"Loading model {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    # Qwen3.5 tokenizer produces mm_token_type_ids which breaks generation
    # Monkey-patch the model's prepare_inputs_for_generation to drop it
    _orig_prepare = model.prepare_inputs_for_generation
    def _patched_prepare(*args, **kwargs):
        kwargs.pop("mm_token_type_ids", None)
        return _orig_prepare(*args, **kwargs)
    model.prepare_inputs_for_generation = _patched_prepare
    if torch.cuda.is_available():
        model = patch_rotary_for_rocm(model)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=[format_reward_fn, cost_reward_fn],
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting training...", flush=True)
    trainer.train()

    trainer.save_model(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
