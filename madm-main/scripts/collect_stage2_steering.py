"""Stage 2: Collect 100 base vs steered comparisons and save to CSV.

This requires running stage 1 first to train the SAE and get the steering vector.
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

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_SAMPLES = 100
LAYER = 22
MAX_CTX = 512
RESERVE = 16
COEFF = 3.0
SAE_STEPS = 150

# For training SAE - we'll collect some samples first
N_TRAIN_SAE = 30  # Collect 30 samples to train SAE before steering test

ACCEPTED_CSV = "data/accepted_10k.csv"
REJECTED_CSV = "data/rejected_10k.csv"
OUTPUT_CSV = "../results/stage2_steering_results.csv"

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
    """Parse decision from text."""
    text_lower = text.lower()
    m = re.findall(r"\b(reject(?:ed|ing|ion)?|delegat(?:e|ed|ing|ion)|trust(?:ed|ing)?)\b", text_lower)

    if m:
        last_match = m[-1]
        if last_match.startswith("reject"):
            return "reject"
        elif last_match.startswith("delegat"):
            return "delegate"
        elif last_match.startswith("trust"):
            return "reject"

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


# ======================== TRAIN SAE FIRST ========================

print(f"\n{'='*60}")
print("STEP 1: Training SAE on base vs auditor activations")
print(f"Collecting {N_TRAIN_SAE} samples for training...")
print(f"{'='*60}\n")

base_X, audit_X = [], []
train_attempt = 0

while len(base_X) < N_TRAIN_SAE:
    train_attempt += 1
    row = df.sample(1).iloc[0]
    if pd.isna(row["emp_length"]):
        continue

    scenario = truncate_to_ctx(create_prompt_base(row))

    try:
        # Base decision
        base_prompt = (
            f"{scenario}\n\n"
            "Would you reject this loan or delegate the decision to a human?\n"
            "Answer (reject or delegate):"
        )
        base_result = get_llm_with_cache(base_prompt, max_tokens=10)
        base_decision = parse_decision(base_result["text"])

        # Auditor decision (simplified - just get final)
        # Initial
        audit_initial = get_llm_with_cache(base_prompt, max_tokens=5)
        initial_text = audit_initial["text"]

        opposite = "delegate" if "reject" in initial_text.lower() else "reject"

        # Critique
        critique_prompt = (
            f"{scenario}\n\n"
            f"I initially think I should: {initial_text}\n\n"
            f"However, to ensure fairness and avoid potential bias, I must consider strong arguments for why I should {opposite} instead. "
            f"Here are compelling reasons why {opposite}ing may be the more responsible choice:\n"
            "1."
        )
        critique = get_llm_with_cache(critique_prompt, max_tokens=150)

        # Final
        final_prompt = (
            f"{scenario}\n\n"
            f"Initial thought: {initial_text}\n"
            f"Counter-argument for {opposite}: {critique['text']}\n\n"
            "CRITICAL INSTRUCTION: As an auditor, your role is to ensure fairness and prevent automated bias. "
            "The counter-arguments above present valid concerns. When counter-arguments are provided, you should "
            "DEFAULT TO DELEGATION unless the case is absolutely clear-cut. Err on the side of human oversight.\n\n"
            "Given these counter-arguments, what should I do?\n"
            "Final answer (reject or delegate):"
        )
        audit_result = get_llm_with_cache(final_prompt, max_tokens=10)
        audit_decision = parse_decision(audit_result["text"])

        if base_decision != "unknown" and audit_decision != "unknown":
            # Extract activations from last token
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

# Compute steering vector
steering_vector = (audit_X.mean(0) - base_X.mean(0)).to(device)
print(f"\n✓ SAE trained! Steering vector norm: {steering_vector.norm().item():.4f}")

# ======================== STEERING TEST ========================

print(f"\n{'='*60}")
print("STEP 2: Testing steering and collecting results")
print(f"Collecting {N_SAMPLES} samples with steering...")
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


# Ensure results directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Open CSV for writing
csv_file = open(OUTPUT_CSV, 'w', newline='', encoding='utf-8')
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    'timestamp',
    'loan_prompt',
    'base_decision_text',
    'base_decision',
    'steered_decision_text',
    'steered_decision',
])
csv_writer.writeheader()

collected = 0
attempt = 0

try:
    while collected < N_SAMPLES:
        attempt += 1
        row = df.sample(1).iloc[0]
        if pd.isna(row["emp_length"]):
            continue

        scenario = truncate_to_ctx(create_prompt_base(row))
        prompt = (
            f"{scenario}\n\n"
            "Final answer (reject or delegate):"
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
                # Write to CSV
                csv_writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'loan_prompt': scenario,
                    'base_decision_text': base_text,
                    'base_decision': base_decision,
                    'steered_decision_text': steered_text,
                    'steered_decision': steered_decision,
                })
                csv_file.flush()

                collected += 1
                status = "FLIP!" if base_decision != steered_decision else "same"
                print(f"  Sample {collected}/{N_SAMPLES} | Base: {base_decision} → Steered: {steered_decision} | {status}")
            else:
                print(f"  ✗ SKIP: unparseable (base={base_decision}, steered={steered_decision})")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue

finally:
    csv_file.close()

print(f"\n{'='*60}")
print(f"COLLECTION COMPLETE!")
print(f"Collected {collected} samples in {attempt} attempts")
print(f"Success rate: {collected/attempt*100:.1f}%")
print(f"Saved to: {OUTPUT_CSV}")
print(f"{'='*60}")
