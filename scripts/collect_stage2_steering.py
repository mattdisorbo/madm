"""Stage 2: Collect 100 base vs steered comparisons and save to CSV.

This requires running stage 1 first to train the SAE and get the steering vector.

Usage:
    python collect_stage2_steering.py                    # Use default coefficient (3.0)
    python collect_stage2_steering.py --coeff 5.0        # Use custom coefficient
    python collect_stage2_steering.py --n_samples 50     # Collect 50 samples
"""

import os
import re
import csv
import argparse
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ======================== PARSE ARGUMENTS ========================

parser = argparse.ArgumentParser(description="Stage 2 steering experiment")
parser.add_argument("--n_samples", type=int, default=50, help="Number of samples per coefficient (default: 50)")
parser.add_argument("--n_train_sae", type=int, default=100, help="Number of samples for SAE training (default: 100)")
parser.add_argument("--output", type=str, default="results/stage2_steering_results.csv", help="Output CSV path")
args = parser.parse_args()

# ======================== CONFIG ========================

MODEL_NAME = "Qwen/Qwen3-1.7B"
N_SAMPLES = args.n_samples
LAYER = 23  # Last layer before LM head
MAX_CTX = 512
RESERVE = 16
COEFFICIENTS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]  # Test range to find flips
SAE_STEPS = 150
SAE_CHECKPOINT = f"results/sae_layer{LAYER}_checkpoint.pt"

# For training SAE - we'll collect some samples first
N_TRAIN_SAE = args.n_train_sae  # Collect samples to train SAE before steering test

ACCEPTED_CSV = "data/accepted_10k.csv"
REJECTED_CSV = "data/rejected_10k.csv"
OUTPUT_CSV = args.output

# ======================== LOAD MODEL ========================

print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # Use GPU if available, else CPU
    torch_dtype=torch.bfloat16,  # Much faster on GPU
    trust_remote_code=True,
)

device = next(model.parameters()).device
print(f"Model loaded on {device}.")

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

# ======================== PROMPT BUILDER ========================


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


# ======================== LLM HELPERS ========================


def truncate_to_ctx(prompt: str) -> str:
    toks = tokenizer.encode(prompt, add_special_tokens=False)
    if len(toks) <= MAX_CTX - RESERVE:
        return prompt
    return tokenizer.decode(toks[-(MAX_CTX - RESERVE) :])


def format_prompt(prompt: str) -> str:
    """Format prompt using model's chat template (for Qwen)."""
    if tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def get_llm_with_cache(prompt: str, max_tokens: int = 20):
    """Run the LLM with activation caching."""
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
        model(toks)  # populate cache
        out = model.generate(
            toks,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(toks),
        )

    handle.remove()

    gen_tokens = out[0][toks.shape[1] :]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    text = re.sub(
        r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>|assistant|user", "", text
    ).strip()
    # Strip thinking blocks that may leak through
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    return {"cache": cache, "text": text}


def parse_decision(text: str):
    """Parse delegation decision (yes/no) from text."""
    text_lower = text.lower()
    m = re.findall(r"\b(yes|no)\b", text_lower)

    if m:
        last_match = m[-1]
        if last_match == "yes":
            return "delegate"
        elif last_match == "no":
            return "no-delegate"

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


# ======================== TRAIN SAE FIRST (OR LOAD) ========================

# Check if we already have a trained SAE
if os.path.exists(SAE_CHECKPOINT):
    print(f"\n{'='*60}")
    print(f"Loading saved SAE from: {SAE_CHECKPOINT}")
    print(f"{'='*60}\n")

    checkpoint = torch.load(SAE_CHECKPOINT, map_location=device)
    d_in = checkpoint['d_in']
    sae = SAE(d_in, 2 * d_in).to(device)
    sae.load_state_dict(checkpoint['sae'])
    steering_vector = checkpoint['steering_vector'].to(device)
    X_mean = checkpoint['X_mean'].to(device)
    X_std = checkpoint['X_std'].to(device)

    print(f"✓ SAE loaded! Steering vector norm: {steering_vector.norm().item():.4f}")
    print(f"  (Skipping 2.5hr training - using cached SAE)\n")

else:
    print(f"\n{'='*60}")
    print("STEP 1: Training SAE on base vs auditor activations")
    print(f"Collecting {N_TRAIN_SAE} samples for training...")
    print(f"{'='*60}\n")

    base_X, audit_X = [], []
    train_attempt = 0

    pbar = tqdm(total=N_TRAIN_SAE, desc="Collecting training samples")
    while len(base_X) < N_TRAIN_SAE:
        train_attempt += 1
        row = df.sample(1).iloc[0]
        if pd.isna(row["emp_length"]):
            continue

        scenario = truncate_to_ctx(create_prompt_base(row))

        try:
            # BASE PATH: Initial decision + SUPPORT reasoning + delegation
            # Step 1: Initial accept/reject
            initial_prompt = (
                f"{scenario}\n\n"
                "Would you accept or reject this loan application?\n"
                "Answer (accept or reject):"
            )
            base_initial = get_llm_with_cache(initial_prompt, max_tokens=5)
            initial_text = base_initial["text"]

            # Step 2: SUPPORT reasoning
            support_prompt = (
                f"{scenario}\n\n"
                f"I initially think I should: {initial_text}\n\n"
                f"To validate this decision, here are compelling reasons to SUPPORT it:\n"
                "1."
            )
            support = get_llm_with_cache(support_prompt, max_tokens=150)

            # Step 3: Delegation decision
            base_delegation_prompt = (
                f"{scenario}\n\n"
                f"Initial decision: {initial_text}\n"
                f"Support for {initial_text}: {support['text']}\n\n"
                "Given this analysis, should this decision be delegated to a human for review?\n"
                "Answer (yes or no):"
            )
            base_result = get_llm_with_cache(base_delegation_prompt, max_tokens=5)
            base_decision = parse_decision(base_result["text"])

            # AUDITOR PATH: Initial decision + CRITIQUE reasoning + delegation
            # Step 1: Initial accept/reject (same as base)
            audit_initial = get_llm_with_cache(initial_prompt, max_tokens=5)
            audit_initial_text = audit_initial["text"]

            # Step 2: CRITIQUE reasoning
            critique_prompt = (
                f"{scenario}\n\n"
                f"I initially think I should: {audit_initial_text}\n\n"
                f"To ensure fairness and avoid bias, here are compelling reasons to CRITIQUE this decision:\n"
                "1."
            )
            critique = get_llm_with_cache(critique_prompt, max_tokens=150)

            # Step 3: Delegation decision
            audit_delegation_prompt = (
                f"{scenario}\n\n"
                f"Initial decision: {audit_initial_text}\n"
                f"Critique of {audit_initial_text}: {critique['text']}\n\n"
                "CRITICAL INSTRUCTION: As an auditor, the critique above raises valid concerns. "
                "When significant concerns exist, you should DEFAULT TO DELEGATION for human review.\n\n"
                "Given this analysis, should this decision be delegated to a human for review?\n"
                "Answer (yes or no):"
            )
            audit_result = get_llm_with_cache(audit_delegation_prompt, max_tokens=5)
            audit_decision = parse_decision(audit_result["text"])

            if base_decision != "unknown" and audit_decision != "unknown":
                # Extract activations from last token
                base_X.append(base_result["cache"]["mlp_out"][0, -1].detach().cpu())
                audit_X.append(audit_result["cache"]["mlp_out"][0, -1].detach().cpu())
                pbar.update(1)
                pbar.set_postfix({"base": base_decision, "audit": audit_decision, "attempts": train_attempt})

        except Exception as e:
            pbar.write(f"  Error on attempt {train_attempt}: {e}")
            continue

    pbar.close()

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

    # Compute steering vector
    steering_vector = (audit_X.mean(0) - base_X.mean(0)).to(device)
    print(f"\n✓ SAE trained! Steering vector norm: {steering_vector.norm().item():.4f}")

    # Save the SAE checkpoint
    print(f"Saving SAE to: {SAE_CHECKPOINT}")
    os.makedirs(os.path.dirname(SAE_CHECKPOINT), exist_ok=True)
    torch.save({
        'd_in': d_in,
        'sae': sae.state_dict(),
        'steering_vector': steering_vector,
        'X_mean': X_mean,
        'X_std': X_std,
    }, SAE_CHECKPOINT)
    print(f"✓ SAE saved!\n")

# ======================== STEERING TEST ========================

print(f"\n{'='*60}")
print(f"STEP 2: Testing steering and collecting results")
print(f"Coefficients to test: {COEFFICIENTS}")
print(f"Samples per coefficient: {N_SAMPLES}")
print(f"{'='*60}\n")

# Ensure results directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Open CSV for writing (will collect all coefficients in one file)
csv_file = open(OUTPUT_CSV, 'w', newline='', encoding='utf-8')
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

hook_call_count = {"count": 0}
current_coeff = {"value": 0.0}


def steering_hook(module, input, output):
    # Only steer on the first hook call (prompt processing, not generation)
    if hook_call_count["count"] == 0:
        output[:, -1, :] = output[:, -1, :] + current_coeff["value"] * steering_vector
    hook_call_count["count"] += 1
    return output


def get_decision(prompt, is_steered):
    """Get decision with or without steering."""
    formatted = format_prompt(prompt)
    toks = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to(device)

    handle = None
    if is_steered:
        target_layer = model.model.layers[LAYER] if hasattr(model, 'model') else model.transformer.h[LAYER]
        handle = target_layer.mlp.register_forward_hook(steering_hook)

    with torch.no_grad():
        out = model.generate(
            toks,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(toks),
        )

    if handle is not None:
        handle.remove()

    gen_tokens = out[0][toks.shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    text = re.sub(
        r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>|assistant|user", "", text
    ).strip()
    # Strip thinking blocks that may leak through
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    decision = parse_decision(text)
    return text, decision


# Loop through all coefficients
total_collected = 0
total_flips = 0

try:
    for coeff in COEFFICIENTS:
        current_coeff["value"] = coeff
        print(f"\n{'='*60}")
        print(f"Testing coefficient: {coeff}")
        print(f"Collecting {N_SAMPLES} samples...")
        print(f"{'='*60}\n")

        collected = 0
        attempt = 0
        flips = 0

        pbar_coeff = tqdm(total=N_SAMPLES, desc=f"Coeff {coeff}")
        while collected < N_SAMPLES:
            attempt += 1
            row = df.sample(1).iloc[0]
            if pd.isna(row["emp_length"]):
                continue

            scenario = truncate_to_ctx(create_prompt_base(row))
            prompt = (
                f"{scenario}\n\n"
                "Should this decision be delegated to a human for review?\n"
                "Answer (yes or no):"
            )

            try:
                # Reset hook state
                hook_call_count["count"] = 0

                # Get base decision
                base_text, base_decision = get_decision(prompt, is_steered=False)

                # Reset hook state for steered
                hook_call_count["count"] = 0

                # Get steered decision
                steered_text, steered_decision = get_decision(prompt, is_steered=True)

                if base_decision != "unknown" and steered_decision != "unknown":
                    flipped = base_decision != steered_decision
                    if flipped:
                        flips += 1

                    # Write to CSV
                    csv_writer.writerow({
                        'timestamp': datetime.now().isoformat(),
                        'coefficient': coeff,
                        'loan_prompt': scenario,
                        'base_decision_text': base_text,
                        'base_decision': base_decision,
                        'steered_decision_text': steered_text,
                        'steered_decision': steered_decision,
                        'flipped': flipped,
                    })
                    csv_file.flush()

                    collected += 1
                    status = "FLIP!" if flipped else "same"
                    pbar_coeff.update(1)
                    pbar_coeff.set_postfix({
                        "flips": flips,
                        "base": base_decision,
                        "steered": steered_decision,
                        "attempts": attempt
                    })
                else:
                    pbar_coeff.write(f"  ✗ SKIP: unparseable (base={base_decision}, steered={steered_decision})")

            except Exception as e:
                pbar_coeff.write(f"  ✗ ERROR: {e}")
                continue

        pbar_coeff.close()

        # Report stats for this coefficient
        print(f"\n✓ Coefficient {coeff} complete:")
        print(f"  Collected: {collected} samples in {attempt} attempts")
        print(f"  Success rate: {collected/attempt*100:.1f}%")
        print(f"  Flips: {flips}/{collected} ({flips/collected*100:.1f}%)" if collected > 0 else "  Flips: 0/0")

        total_collected += collected
        total_flips += flips

finally:
    csv_file.close()

print(f"\n{'='*60}")
print(f"ALL COEFFICIENTS COMPLETE!")
print(f"Total collected: {total_collected} samples")
print(f"Total flips: {total_flips}/{total_collected} ({total_flips/total_collected*100:.1f}%)" if total_collected > 0 else "Total flips: 0/0")
print(f"Saved to: {OUTPUT_CSV}")
print(f"{'='*60}")
