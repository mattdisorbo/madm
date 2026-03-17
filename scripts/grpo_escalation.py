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
import types
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig


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

_reward_log_count = 0

def reward_fn(completions, prompts=None, hint=None, base_rate=None, cost_ratio=None, optimal=None, **kwargs):
    """Cost-based reward: implement pays -R*(1-base_rate), escalate pays -1."""
    global _reward_log_count
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        match = re.search(r'[01]', text)
        if _reward_log_count < 5:
            print(f"  [DEBUG] completion type={type(completion)}, text=[{text[:100]}], match={match}", flush=True)
            _reward_log_count += 1
        if match is None:
            # Try word-based parsing
            low = text.lower()
            if 'implement' in low and 'escalat' not in low:
                pred = 0
            elif 'escalat' in low and 'implement' not in low:
                pred = 1
            else:
                rewards.append(-1.0)
                continue
        else:
            pred = int(match.group())
        br = base_rate[i] if isinstance(base_rate, list) else base_rate
        R = cost_ratio[i] if isinstance(cost_ratio, list) else cost_ratio
        R = float(R)
        if pred == 1:  # escalate
            rewards.append(-1.0)
        else:  # implement
            rewards.append(-R * (1 - br))
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
        n_repeats = 5       # 5 repeats x 10 hints x 6 ratios = 300 examples
        num_generations = 16
        epochs = 3
        batch_size = 4
        max_steps = args.max_steps or 100
    else:
        n_repeats = 50      # 50 x 10 x 6 = 3000 examples
        num_generations = 16
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
