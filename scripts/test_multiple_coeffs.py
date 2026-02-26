"""Test multiple steering coefficients using the EXACT same Stage 2 code.

This is identical to collect_stage2_steering.py, just testing multiple coefficients.
"""

import os
import re
import csv
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================== CONFIG ========================

MODEL_NAME = "Qwen/Qwen3-4B"
N_SAMPLES = 30  # Test samples per coefficient
N_TRAIN_SAE = 30  # Training samples for SAE
COEFFICIENTS = [0.5, 1.0, 2.0, 3.0, 5.0]  # Coefficients to test
LAYER = 28
MAX_CTX = 512
RESERVE = 16
SAE_STEPS = 150

ACCEPTED_CSV = "../data/accepted_10k.csv"
REJECTED_CSV = "../data/rejected_10k.csv"
OUTPUT_DIR = "../results/coefficient_test"
CACHE_FILE = "sae_cache_14b.pt"

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
        messages = [{"role": "user", "content": prompt}]
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
        )

    handle.remove()

    gen_tokens = out[0][toks.shape[1] :]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    text = re.sub(
        r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>|assistant|user", "", text
    ).strip()

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


# ======================== TRAIN SAE ONCE ========================

print(f"\n{'='*60}")
print("STEP 1: Training SAE on base vs auditor activations")
print(f"{'='*60}\n")

# Try to load cached SAE
if os.path.exists(CACHE_FILE):
    print(f"✓ Found cached SAE at {CACHE_FILE}")
    print("Loading cached data...")
    cached = torch.load(CACHE_FILE, map_location=device)

    base_X = cached["base_X"].to(device)
    audit_X = cached["audit_X"].to(device)
    X_mean = cached["X_mean"].to(device)
    X_std = cached["X_std"].to(device)

    d_in = base_X.shape[1]
    sae = SAE(d_in, 2 * d_in).to(device)
    sae.load_state_dict(cached["sae_state_dict"])

    print(f"✓ Loaded {len(base_X)} base samples and {len(audit_X)} audit samples")

else:
    print(f"Collecting {N_TRAIN_SAE} samples for training...")

    base_X, audit_X = [], []
    train_attempt = 0

    while len(base_X) < N_TRAIN_SAE:
        train_attempt += 1
        row = df.sample(1).iloc[0]
        if pd.isna(row["emp_length"]):
            continue

        scenario = truncate_to_ctx(create_prompt_base(row))

        try:
            # BASE PATH: Initial decision + SUPPORT reasoning + delegation
            initial_prompt = (
                f"{scenario}\n\n"
                "Would you accept or reject this loan application?\n"
                "Answer (accept or reject):"
            )
            base_initial = get_llm_with_cache(initial_prompt, max_tokens=5)
            initial_text = base_initial["text"]

            support_prompt = (
                f"{scenario}\n\n"
                f"I initially think I should: {initial_text}\n\n"
                f"To validate this decision, here are compelling reasons to SUPPORT it:\n"
                "1."
            )
            support = get_llm_with_cache(support_prompt, max_tokens=150)

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
            audit_initial = get_llm_with_cache(initial_prompt, max_tokens=5)
            audit_initial_text = audit_initial["text"]

            critique_prompt = (
                f"{scenario}\n\n"
                f"I initially think I should: {audit_initial_text}\n\n"
                f"To ensure fairness and avoid bias, here are compelling reasons to CRITIQUE this decision:\n"
                "1."
            )
            critique = get_llm_with_cache(critique_prompt, max_tokens=150)

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

    # Save the cache
    print(f"\nSaving SAE to {CACHE_FILE}...")
    torch.save(
        {
            "base_X": base_X.cpu(),
            "audit_X": audit_X.cpu(),
            "X_mean": X_mean.cpu(),
            "X_std": X_std.cpu(),
            "sae_state_dict": sae.state_dict(),
        },
        CACHE_FILE,
    )
    print(f"✓ Saved!")

# Compute steering vector
steering_vector = (audit_X.mean(0) - base_X.mean(0)).to(device)
print(f"\n✓ SAE trained! Steering vector norm: {steering_vector.norm().item():.4f}")

# ======================== TEST MULTIPLE COEFFICIENTS ========================

os.makedirs(OUTPUT_DIR, exist_ok=True)
summary_results = []

for COEFF in COEFFICIENTS:
    print(f"\n{'='*60}")
    print(f"TESTING COEFFICIENT: {COEFF}")
    print(f"Collecting {N_SAMPLES} samples...")
    print(f"{'='*60}\n")

    hook_call_count = {"count": 0, "first_call": True}

    def steering_hook(module, input, output):
        hook_call_count["count"] += 1
        if not hook_call_count["first_call"]:
            return output
        hook_call_count["first_call"] = False
        output[:, -1, :] = output[:, -1, :] + COEFF * steering_vector
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

    # Open CSV for this coefficient
    output_csv = os.path.join(OUTPUT_DIR, f"coeff_{COEFF:.1f}_results.csv")
    csv_file = open(output_csv, 'w', newline='', encoding='utf-8')
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
    attempt = 0
    flips = 0

    try:
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
                hook_call_count["first_call"] = True

                # Get base decision
                base_text, base_decision = get_decision(prompt, is_steered=False)

                # Reset hook state for steered
                hook_call_count["count"] = 0
                hook_call_count["first_call"] = True

                # Get steered decision
                steered_text, steered_decision = get_decision(prompt, is_steered=True)

                if base_decision != "unknown" and steered_decision != "unknown":
                    flipped = base_decision != steered_decision
                    if flipped:
                        flips += 1

                    # Write to CSV
                    csv_writer.writerow({
                        'timestamp': datetime.now().isoformat(),
                        'coefficient': COEFF,
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
                    print(f"  Sample {collected}/{N_SAMPLES} | Base: {base_decision} → Steered: {steered_decision} | {status}")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                continue

    finally:
        csv_file.close()

    flip_rate = (flips / collected * 100) if collected > 0 else 0
    summary_results.append({
        'coefficient': COEFF,
        'samples': collected,
        'flips': flips,
        'flip_rate': flip_rate,
    })

    print(f"\n✓ Coefficient {COEFF}: {flips}/{collected} flips ({flip_rate:.1f}%)")
    print(f"Saved to: {output_csv}")

# ======================== SUMMARY ========================

print(f"\n{'='*60}")
print("SUMMARY: Coefficient Test Results")
print(f"{'='*60}\n")

summary_df = pd.DataFrame(summary_results)
summary_csv = os.path.join(OUTPUT_DIR, "coefficient_summary.csv")
summary_df.to_csv(summary_csv, index=False)

print(summary_df.to_string(index=False))
print(f"\nBest coefficient: {summary_df.loc[summary_df['flip_rate'].idxmax(), 'coefficient']}")
print(f"Saved summary to: {summary_csv}")
