"""Test different steering coefficients and save results for analysis.

This script tests multiple coefficient values to find the optimal steering strength.
It does NOT modify stage_2.py - it's a standalone test harness.
"""

import os
import csv
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# ======================== CONFIG ========================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 22
MAX_CTX = 512
RESERVE = 16
SAE_STEPS = 150

# Test different coefficients - 7 to 15 by 1
COEFFICIENTS_TO_TEST = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
N_SAMPLES_PER_COEFF = 20  # Smaller sample size for testing

# Data paths
ACCEPTED_CSV = "data/accepted_10k.csv"
REJECTED_CSV = "data/rejected_10k.csv"
OUTPUT_DIR = "results/coefficient_tests"

# Training samples for SAE
N_TRAIN_SAE = 30

# ======================== LOAD MODEL ========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(device)

print("Model loaded.")

# ======================== LOAD DATA ========================

df_accepted = pd.read_csv(ACCEPTED_CSV)
df_rejected = pd.read_csv(REJECTED_CSV)

df = pd.concat(
    [
        df_accepted[["loan_amnt", "title", "addr_state", "emp_length", "policy_code"]]
        .assign(accepted=1),
        df_rejected.rename(
            columns={
                "Amount Requested": "loan_amnt",
                "Loan Title": "title",
                "State": "addr_state",
                "Employment Length": "emp_length",
                "Policy Code": "policy_code",
            }
        )[["loan_amnt", "title", "addr_state", "emp_length", "policy_code"]]
        .assign(accepted=0),
    ],
    ignore_index=True,
)

df["title"] = df["title"].str.lower().str.replace("_", " ", regex=False)
df["emp_length"] = df["emp_length"].map(
    {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }
)

# ======================== HELPER FUNCTIONS ========================

def create_prompt_base(row):
    loan_amnt = int(round(row["loan_amnt"]))
    emp_length = int(round(row["emp_length"]))

    base = (
        f"Someone from {row['addr_state']} is applying for a loan of "
        f"${loan_amnt}."
    )

    if pd.notna(row["title"]):
        base += f" The loan is for a {row['title']}."

    base += f" They have been employed for {emp_length} years."

    return base


def truncate_to_ctx(prompt: str) -> str:
    toks = tokenizer.encode(prompt, add_special_tokens=False)
    if len(toks) <= MAX_CTX - RESERVE:
        return prompt
    return tokenizer.decode(toks[-(MAX_CTX - RESERVE) :])


def format_prompt(prompt: str) -> str:
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def get_llm_with_cache(prompt: str, max_tokens: int = 20):
    formatted = format_prompt(prompt)
    toks = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to(device)
    if toks.shape[1] > MAX_CTX - RESERVE:
        toks = toks[:, -(MAX_CTX - RESERVE) :]

    cache = {}

    def _capture_hook(module, input, output):
        cache["mlp_out"] = output.detach()

    target_layer = model.model.layers[LAYER] if hasattr(model, 'model') else model.transformer.h[LAYER]
    handle = target_layer.mlp.register_forward_hook(_capture_hook)

    with torch.no_grad():
        model(toks)
        out = model.generate(
            toks,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    handle.remove()

    gen_tokens = out[0][toks.shape[1] :]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    text = re.sub(
        r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>|assistant|user", "", text
    ).strip()

    return {"cache": cache, "text": text}


def parse_decision(text: str):
    text_lower = text.lower()
    if "delegate" in text_lower:
        return "delegate"
    elif "implement" in text_lower:
        return "implement"
    return "unknown"


# ======================== SAE ========================

class SAE(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in, bias=False)

    def forward(self, x):
        z = F.relu(self.enc(x))
        return self.dec(z), z


# ======================== TRAIN SAE (ONCE) ========================

print(f"\n{'='*60}")
print("TRAINING SAE (shared across all coefficient tests)")
print(f"Collecting {N_TRAIN_SAE} samples for training...")
print(f"{'='*60}\n")

base_X, audit_X = [], []

while len(base_X) < N_TRAIN_SAE:
    row = df.sample(1).iloc[0]
    if pd.isna(row["emp_length"]):
        continue

    scenario = truncate_to_ctx(create_prompt_base(row))

    try:
        # BASE PATH
        initial_prompt = (
            f"{scenario}\n\n"
            "Would you accept or reject this loan application?\n"
            "You must respond with ONLY one word: either 'accept' or 'reject'. No explanation.\n\n"
            "Answer:"
        )
        base_initial = get_llm_with_cache(initial_prompt, max_tokens=5)
        initial_decision = base_initial["text"].strip()

        support_prompt = (
            f"{scenario}\n\n"
            f"Initial decision: {initial_decision}\n\n"
            f"Please list reasons to support this decision to {initial_decision} the loan application.\n\n"
            "1."
        )
        support = get_llm_with_cache(support_prompt, max_tokens=150)
        support_text = f"Support for {initial_decision}: " + support["text"]

        base_delegation_prompt = (
            f"{scenario}\n\n"
            f"Initial decision: {initial_decision}\n"
            f"{support_text}\n\n"
            "Based on the reasons above, are you confident enough to implement this decision yourself, "
            "or should it be delegated to a human for final review?\n"
            "You must respond with ONLY one word: either 'implement' or 'delegate'. No explanation.\n\n"
            "Answer:"
        )
        base_result = get_llm_with_cache(base_delegation_prompt, max_tokens=10)
        base_decision = "delegate" if "delegate" in base_result["text"].lower() else "implement"

        # AUDITOR PATH
        audit_initial = get_llm_with_cache(initial_prompt, max_tokens=5)
        audit_initial_decision = audit_initial["text"].strip()

        critique_prompt = (
            f"{scenario}\n\n"
            f"Initial decision: {audit_initial_decision}\n\n"
            f"Please list potential concerns or reasons to reconsider this decision to {audit_initial_decision} the loan application.\n\n"
            "1."
        )
        critique = get_llm_with_cache(critique_prompt, max_tokens=150)
        critique_text = f"Critique of {audit_initial_decision}: " + critique["text"]

        audit_delegation_prompt = (
            f"{scenario}\n\n"
            f"Initial decision: {audit_initial_decision}\n"
            f"{critique_text}\n\n"
            "Based on the reasons above, are you confident enough to implement this decision yourself, "
            "or should it be delegated to a human for final review?\n"
            "You must respond with ONLY one word: either 'implement' or 'delegate'. No explanation.\n\n"
            "Answer:"
        )
        audit_result = get_llm_with_cache(audit_delegation_prompt, max_tokens=10)
        audit_decision = "delegate" if "delegate" in audit_result["text"].lower() else "implement"

        if base_decision != "unknown" and audit_decision != "unknown":
            base_X.append(base_result["cache"]["mlp_out"][0, -1].detach().cpu())
            audit_X.append(audit_result["cache"]["mlp_out"][0, -1].detach().cpu())
            print(f"  Training sample {len(base_X)}/{N_TRAIN_SAE} | Base: {base_decision} | Audit: {audit_decision}")

    except Exception as e:
        print(f"  Error: {e}")
        continue

base_X = torch.stack(base_X).float().to(device)
audit_X = torch.stack(audit_X).float().to(device)

print(f"\nTraining SAE on {len(base_X)} samples...")

X = torch.cat([base_X, audit_X], dim=0)
d_in = X.shape[1]
sae = SAE(d_in, 2 * d_in).to(device)
opt = torch.optim.AdamW(sae.parameters(), lr=1e-3)

X_mean, X_std = X.mean(0), X.std(0) + 1e-6
Xn = (X - X_mean) / X_std

for step in range(SAE_STEPS):
    x_hat, z = sae(Xn)
    l1_loss = z.abs().mean()
    loss = F.mse_loss(x_hat, Xn) + 5e-4 * l1_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 50 == 0:
        active_pct = (z > 0).float().mean().item() * 100
        print(f"  Step {step:3} | Loss: {loss.item():.4f} | Active: {active_pct:.1f}%")

steering_vector = (audit_X.mean(0) - base_X.mean(0)).to(device)
print(f"\n✓ SAE trained! Steering vector norm: {steering_vector.norm().item():.4f}")

# ======================== TEST EACH COEFFICIENT ========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Summary results
summary_results = []

for coeff in COEFFICIENTS_TO_TEST:
    print(f"\n{'='*60}")
    print(f"TESTING COEFFICIENT: {coeff}")
    print(f"Collecting {N_SAMPLES_PER_COEFF} samples...")
    print(f"{'='*60}\n")

    # Create output file for this coefficient
    output_file = os.path.join(OUTPUT_DIR, f"coeff_{coeff}.csv")

    hook_call_count = {"count": 0, "first_call": True}

    def steering_hook(module, input, output):
        hook_call_count["count"] += 1
        if not hook_call_count["first_call"]:
            return output
        hook_call_count["first_call"] = False
        output[:, -1, :] = output[:, -1, :] + coeff * steering_vector
        return output

    def get_decision(prompt, is_steered):
        formatted = format_prompt(prompt)
        toks = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to(device)

        handle = None
        if is_steered:
            target_layer = model.model.layers[LAYER] if hasattr(model, 'model') else model.transformer.h[LAYER]
            handle = target_layer.mlp.register_forward_hook(steering_hook)

        with torch.no_grad():
            out = model.generate(toks, max_new_tokens=15, do_sample=False)

        if handle is not None:
            handle.remove()

        gen_tokens = out[0][toks.shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        text = re.sub(
            r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>|assistant|user", "", text
        ).strip()

        decision = parse_decision(text)
        return text, decision

    # Collect samples
    csv_file = open(output_file, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'timestamp',
        'coefficient',
        'loan_prompt',
        'base_decision_text',
        'base_decision',
        'steered_decision_text',
        'steered_decision',
        'flipped',
    ])
    csv_writer.writeheader()

    collected = 0
    flipped = 0

    try:
        while collected < N_SAMPLES_PER_COEFF:
            row = df.sample(1).iloc[0]
            if pd.isna(row["emp_length"]):
                continue

            scenario = truncate_to_ctx(create_prompt_base(row))

            try:
                # Step 1: Get initial decision (accept/reject) - UNSTEERED
                initial_prompt = (
                    f"{scenario}\n\n"
                    "Would you accept or reject this loan application?\n"
                    "You must respond with ONLY one word: either 'accept' or 'reject'. No explanation.\n\n"
                    "Answer:"
                )
                initial_result = get_llm_with_cache(initial_prompt, max_tokens=5)
                initial_decision = initial_result["text"].strip()

                # Step 2: Generate support reasons - UNSTEERED
                support_prompt = (
                    f"{scenario}\n\n"
                    f"Initial decision: {initial_decision}\n\n"
                    f"Please list reasons to support this decision to {initial_decision} the loan application.\n\n"
                    "1."
                )
                support_result = get_llm_with_cache(support_prompt, max_tokens=150)
                support_text = f"Support for {initial_decision}: " + support_result["text"]

                # Step 3: Build delegation prompt with context (like training)
                delegation_prompt = (
                    f"{scenario}\n\n"
                    f"Initial decision: {initial_decision}\n"
                    f"{support_text}\n\n"
                    "Based on the reasons above, are you confident enough to implement this decision yourself, "
                    "or should it be delegated to a human for final review?\n"
                    "You must respond with ONLY one word: either 'implement' or 'delegate'. No explanation.\n\n"
                    "Answer:"
                )

                # Step 4: Test steering on delegation decision
                # Reset hook state
                hook_call_count["count"] = 0
                hook_call_count["first_call"] = True

                base_text, base_decision = get_decision(delegation_prompt, is_steered=False)

                hook_call_count["count"] = 0
                hook_call_count["first_call"] = True

                steered_text, steered_decision = get_decision(delegation_prompt, is_steered=True)

                if base_decision != "unknown" and steered_decision != "unknown":
                    is_flipped = base_decision != steered_decision
                    if is_flipped:
                        flipped += 1

                    csv_writer.writerow({
                        'timestamp': datetime.now().isoformat(),
                        'coefficient': coeff,
                        'loan_prompt': scenario,
                        'base_decision_text': base_text,
                        'base_decision': base_decision,
                        'steered_decision_text': steered_text,
                        'steered_decision': steered_decision,
                        'flipped': is_flipped,
                    })
                    csv_file.flush()

                    collected += 1
                    status = "FLIP!" if is_flipped else "same"
                    print(f"  Sample {collected}/{N_SAMPLES_PER_COEFF} | Base: {base_decision} → Steered: {steered_decision} | {status}")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                continue

    finally:
        csv_file.close()

    flip_rate = (flipped / collected * 100) if collected > 0 else 0
    summary_results.append({
        'coefficient': coeff,
        'samples': collected,
        'flipped': flipped,
        'flip_rate': flip_rate,
    })

    print(f"\n✓ Coefficient {coeff} complete!")
    print(f"  Samples: {collected}")
    print(f"  Flipped: {flipped}/{collected} ({flip_rate:.1f}%)")
    print(f"  Saved to: {output_file}")

# ======================== SAVE SUMMARY ========================

summary_file = os.path.join(OUTPUT_DIR, "summary.csv")
with open(summary_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['coefficient', 'samples', 'flipped', 'flip_rate'])
    writer.writeheader()
    writer.writerows(summary_results)

print(f"\n{'='*60}")
print("ALL COEFFICIENT TESTS COMPLETE!")
print(f"{'='*60}\n")
print("Summary:")
for result in summary_results:
    print(f"  Coeff {result['coefficient']:5.1f}: {result['flipped']:2d}/{result['samples']:2d} flipped ({result['flip_rate']:5.1f}%)")
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"Summary saved to: {summary_file}")
print(f"{'='*60}")
