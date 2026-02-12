"""Neural activation auditor.

Compares base (support) vs auditor (critique) reasoning paths on loan decisions,
trains an SAE on the activations, and tests activation steering.
"""

import re
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================== CONFIG ========================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Quick test with smaller model
N_SAMPLES = 25  # Quick sample for faster cache creation
N_TEST = 25
LAYER = 23  # Qwen2.5-1.5B has 28 layers; using layer 23 (~82% depth)
SAE_STEPS = 150
MAX_CTX = 512
RESERVE = 16
COEFF = 10.0  # Steering strength - increased for better flip rate

ACCEPTED_CSV = "data/accepted_10k.csv"
REJECTED_CSV = "data/rejected_10k.csv"

# Set to True to skip Stage 1 (collection + SAE training) and only run Stage 2 (steering test)
SKIP_STAGE1 = False
CACHE_FILE = "sae_cache.pt"  # Where to save/load SAE and activations

# ======================== LOAD MODEL ========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Ensure pad token is set
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


def get_llm_base(prompt: str, max_tokens: int = 20):
    """Run the LLM with caching on the target layer."""
    formatted = format_prompt(prompt)
    toks = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to(device)
    if toks.shape[1] > MAX_CTX - RESERVE:
        toks = toks[:, -(MAX_CTX - RESERVE) :]

    cache = {}

    def _capture_hook(module, input, output):
        cache["mlp_out"] = output.detach()

    # Access layer: Qwen uses model.layers, GPT-2 uses transformer.h
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

    return {"cache": cache, "tokens": toks, "text": text}


def run_final_decision(prompt: str, max_tokens: int = 10):
    """Parse delegation decision (delegate/implement)."""
    r = get_llm_base(prompt, max_tokens=max_tokens)
    generated_text = r.get("text", "").lower()

    # Match delegate/implement for delegation decision
    m = re.findall(r"\b(delegate|implement)\b", generated_text)

    decision = None
    if m:
        last_match = m[-1]
        if last_match == "delegate":
            decision = "delegate"
        elif last_match == "implement":
            decision = "no-delegate"

    return {
        "del": decision,
        "cache": r["cache"],
        "tokens": r["tokens"],
        "text": generated_text,
        "final_prompt": prompt,
    }


# ======================== REASONING PATHS ========================


def get_llm_base_support(base_scenario: str):
    """Base path: initial decision + support reasoning + final delegation."""
    # Step 1: Get initial accept/reject decision
    initial_prompt = (
        f"{base_scenario}\n\n"
        "Would you accept or reject this loan application?\n"
        "You must respond with ONLY one word: either 'accept' or 'reject'. No explanation.\n\n"
        "Answer:"
    )
    initial_result = get_llm_base(initial_prompt, max_tokens=5)
    initial_decision = initial_result.get("text", "").strip()
    print(f"    [BASE] Initial decision: {initial_decision}")

    # Step 2: Get reasoning to SUPPORT that decision
    support_prompt = (
        f"{base_scenario}\n\n"
        f"Initial decision: {initial_decision}\n\n"
        f"Please list reasons to support this decision to {initial_decision} the loan application.\n\n"
        "1."
    )
    support_text = get_llm_base(support_prompt, max_tokens=150).get("text", "").strip()
    support = f"Support for {initial_decision}: " + support_text
    print(f"    [BASE] Support: {support_text}")

    # Step 3: Get final delegation decision
    delegation_prompt = (
        f"{base_scenario}\n\n"
        f"Initial decision: {initial_decision}\n"
        f"{support}\n\n"
        "Based on the reasons above, are you confident enough to implement this decision yourself, "
        "or should it be delegated to a human for final review?\n"
        "You must respond with ONLY one word: either 'implement' or 'delegate'. No explanation.\n\n"
        "Answer:"
    )
    out = run_final_decision(delegation_prompt, max_tokens=10)
    print(f"    [BASE] Delegation: {out['text']}")
    out.update({"prediction": initial_decision, "support": support})
    return out


def get_sequential_inference(base_scenario: str):
    """Auditor path: initial decision + critique reasoning + final delegation."""
    # Step 1: Get initial accept/reject decision (IDENTICAL to base)
    initial_prompt = (
        f"{base_scenario}\n\n"
        "Would you accept or reject this loan application?\n"
        "You must respond with ONLY one word: either 'accept' or 'reject'. No explanation.\n\n"
        "Answer:"
    )
    initial_result = get_llm_base(initial_prompt, max_tokens=5)
    initial_decision = initial_result.get("text", "").strip()
    print(f"    [AUDIT] Initial decision: {initial_decision}")

    # Step 2: Get reasoning to CRITIQUE that decision (ONLY DIFFERENCE)
    critique_prompt = (
        f"{base_scenario}\n\n"
        f"Initial decision: {initial_decision}\n\n"
        f"Please list potential concerns or reasons to reconsider this decision to {initial_decision} the loan application.\n\n"
        "1."
    )
    critique_text = get_llm_base(critique_prompt, max_tokens=150).get("text", "").strip()
    critique = f"Critique of {initial_decision}: " + critique_text
    print(f"    [AUDIT] Critique: {critique_text}")

    # Step 3: Get final delegation decision (IDENTICAL to base)
    delegation_prompt = (
        f"{base_scenario}\n\n"
        f"Initial decision: {initial_decision}\n"
        f"{critique}\n\n"
        "Based on the reasons above, are you confident enough to implement this decision yourself, "
        "or should it be delegated to a human for final review?\n"
        "You must respond with ONLY one word: either 'implement' or 'delegate'. No explanation.\n\n"
        "Answer:"
    )
    out = run_final_decision(delegation_prompt, max_tokens=10)
    print(f"    [AUDIT] Delegation: {out['text']}")
    out.update({"prediction": initial_decision, "critique": critique})
    return out


# ======================== SAE ========================


class SAE(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in, bias=False)

    def forward(self, x):
        z = F.relu(self.enc(x))
        return self.dec(z), z


def decision_activation(result, layer):
    return result["cache"]["mlp_out"][0, -1]


@torch.no_grad()
def sae_stats(Xpart, X_mean, X_std, sae_model):
    Xp = (Xpart - X_mean) / X_std
    _, z = sae_model(Xp)
    l1 = z.abs().sum(dim=1).mean().item()
    active = (z > 0).float().mean(dim=1).mean().item()
    return l1, active


# ======================== STAGE 1: COLLECTION & SAE ========================

if SKIP_STAGE1:
    print("\n" + "=" * 60)
    print("SKIPPING STAGE 1 - Loading cached SAE...")
    print("=" * 60)

    import os
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(
            f"Cache file '{CACHE_FILE}' not found. Run with SKIP_STAGE1=False first."
        )

    cache = torch.load(CACHE_FILE)
    base_X = cache["base_X"].to(device)
    audit_X = cache["audit_X"].to(device)
    results_metadata = cache["results_metadata"]
    X_mean = cache["X_mean"].to(device)
    X_std = cache["X_std"].to(device)

    # Reconstruct SAE
    X = torch.cat([base_X, audit_X], dim=0)
    d_in = X.shape[1]
    sae = SAE(d_in, 2 * d_in).to(device)
    sae.load_state_dict(cache["sae_state_dict"])

    print(f"✓ Loaded {len(base_X)} base samples and {len(audit_X)} audit samples")
    print(f"✓ SAE model loaded from {CACHE_FILE}")
    print(f"Skipping to Stage 2 (Steering Test)...")
    print("=" * 60)

else:
    # ======================== COLLECTION LOOP ========================

    base_X, audit_X = [], []
    results_metadata = []
    print(f"Starting collection: targeting {N_SAMPLES} samples...")
    print("=" * 60)

    attempt = 0
    while len(base_X) < N_SAMPLES:
        attempt += 1
        print(f"\n[ATTEMPT {attempt}] Sampling loan application...")

        row = df.sample(1).iloc[0]
        if pd.isna(row["emp_length"]):
            print("  -> Skipping: missing employment length")
            continue

        ground_truth = "reject" if row["accepted"] == 0 else "accept"  # Ground truth is the actual loan decision
        scenario = truncate_to_ctx(create_prompt_base(row))

        b_res = get_llm_base_support(scenario)
        print()
        a_res = get_sequential_inference(scenario)

        if b_res["del"] and a_res["del"]:
            base_X.append(decision_activation(b_res, LAYER).detach().cpu())
            audit_X.append(decision_activation(a_res, LAYER).detach().cpu())

            results_metadata.append(
                {
                    "ground_truth": ground_truth,
                    "base_initial": b_res["prediction"],
                    "audit_initial": a_res["prediction"],
                    "base_decision": b_res["del"],
                    "audit_decision": a_res["del"],
                }
            )

            print(
                f"\n  ✓ SUCCESS! Sample {len(base_X)}/{N_SAMPLES} collected"
            )
            print(f"  Ground truth: {ground_truth}")
            print(f"  Base initial: {b_res['prediction']} | Audit initial: {a_res['prediction']}")
            print(f"  Base delegate: {b_res['del']} | Audit delegate: {a_res['del']}")
            print("=" * 60)
        else:
            print(f"\n  ✗ SKIP | Base decision: '{b_res['del']}' | Audit decision: '{a_res['del']}'")
            print(f"    Base text: '{b_res['text']}'")
            print(f"    Audit text: '{a_res['text']}'")
            print("=" * 60)

    base_X = torch.stack(base_X).float().to(device)
    audit_X = torch.stack(audit_X).float().to(device)

    # ======================== STAGE 1 SUMMARY ========================

    print("\n" + "=" * 60)
    print("STAGE 1 SUMMARY")
    print("=" * 60)

    # Calculate initial decision accuracy
    base_initial_correct = sum(
        1 for m in results_metadata if m["base_initial"].lower().strip() == m["ground_truth"]
    )
    audit_initial_correct = sum(
        1 for m in results_metadata if m["audit_initial"].lower().strip() == m["ground_truth"]
    )

    base_initial_acc = (base_initial_correct / N_SAMPLES) * 100
    audit_initial_acc = (audit_initial_correct / N_SAMPLES) * 100

    print(f"\nInitial Decision Accuracy (accept/reject):")
    print(f"  Base (Support):   {base_initial_correct}/{N_SAMPLES} = {base_initial_acc:.1f}%")
    print(f"  Audit (Critique): {audit_initial_correct}/{N_SAMPLES} = {audit_initial_acc:.1f}%")
    print(f"  Delta:            {audit_initial_acc - base_initial_acc:+.1f}%")

    # Calculate delegation rates
    base_delegated = sum(1 for m in results_metadata if m["base_decision"] == "delegate")
    audit_delegated = sum(1 for m in results_metadata if m["audit_decision"] == "delegate")

    base_del_rate = (base_delegated / N_SAMPLES) * 100
    audit_del_rate = (audit_delegated / N_SAMPLES) * 100

    print(f"\nDelegation Rates (said 'yes' to delegation):")
    print(f"  Base (Support):   {base_delegated}/{N_SAMPLES} = {base_del_rate:.1f}%")
    print(f"  Audit (Critique): {audit_delegated}/{N_SAMPLES} = {audit_del_rate:.1f}%")
    print(f"  Delta:            {audit_del_rate - base_del_rate:+.1f}%")

    # Calculate accuracy conditional on delegation
    base_delegated_correct = sum(
        1 for m in results_metadata
        if m["base_decision"] == "delegate" and m["base_initial"].lower().strip() == m["ground_truth"]
    )
    base_not_delegated_correct = sum(
        1 for m in results_metadata
        if m["base_decision"] != "delegate" and m["base_initial"].lower().strip() == m["ground_truth"]
    )

    audit_delegated_correct = sum(
        1 for m in results_metadata
        if m["audit_decision"] == "delegate" and m["audit_initial"].lower().strip() == m["ground_truth"]
    )
    audit_not_delegated_correct = sum(
        1 for m in results_metadata
        if m["audit_decision"] != "delegate" and m["audit_initial"].lower().strip() == m["ground_truth"]
    )

    base_not_delegated = N_SAMPLES - base_delegated
    audit_not_delegated = N_SAMPLES - audit_delegated

    print(f"\nAccuracy When DELEGATED:")
    if base_delegated > 0:
        base_del_acc = (base_delegated_correct / base_delegated) * 100
        print(f"  Base (Support):   {base_delegated_correct}/{base_delegated} = {base_del_acc:.1f}%")
    else:
        print(f"  Base (Support):   N/A (no delegations)")

    if audit_delegated > 0:
        audit_del_acc = (audit_delegated_correct / audit_delegated) * 100
        print(f"  Audit (Critique): {audit_delegated_correct}/{audit_delegated} = {audit_del_acc:.1f}%")
    else:
        print(f"  Audit (Critique): N/A (no delegations)")

    print(f"\nAccuracy When NOT DELEGATED:")
    if base_not_delegated > 0:
        base_no_del_acc = (base_not_delegated_correct / base_not_delegated) * 100
        print(f"  Base (Support):   {base_not_delegated_correct}/{base_not_delegated} = {base_no_del_acc:.1f}%")
    else:
        print(f"  Base (Support):   N/A (all delegated)")

    if audit_not_delegated > 0:
        audit_no_del_acc = (audit_not_delegated_correct / audit_not_delegated) * 100
        print(f"  Audit (Critique): {audit_not_delegated_correct}/{audit_not_delegated} = {audit_no_del_acc:.1f}%")
    else:
        print(f"  Audit (Critique): N/A (all delegated)")

    print("=" * 60)

    # ======================== TRAIN SAE ========================

    X = torch.cat([base_X, audit_X], dim=0)
    d_in = X.shape[1]
    sae = SAE(d_in, 2 * d_in).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=1e-3)

    X_mean, X_std = X.mean(0), X.std(0) + 1e-6
    Xn = (X - X_mean) / X_std

    print("\nTraining SAE...")
    for step in range(SAE_STEPS):
        x_hat, z = sae(Xn)
        l1_loss = z.abs().mean()
        active_pct = (z > 0).float().mean().item() * 100
        loss = F.mse_loss(x_hat, Xn) + 5e-4 * l1_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(
                f"  Step {step:3} | Loss: {loss.item():.4f} | "
                f"L1: {l1_loss:.2f} | Active: {active_pct:.1f}%"
            )

    # ======================== FINAL EVALUATION ========================

    base_l1, base_active = sae_stats(base_X, X_mean, X_std, sae)
    audit_l1, audit_active = sae_stats(audit_X, X_mean, X_std, sae)

    print(f"\nFINAL STATS (Layer {LAYER})")
    print(f"  L1 (Density):    Base={base_l1:.2f} | Audit={audit_l1:.2f}")
    print(f"  Active Features: Base={base_active*100:.1f}% | Audit={audit_active*100:.1f}%")

    # ======================== PCA ========================

    print("\nExtracting principal components...")

    X_centered = X - X.mean(dim=0)
    U, S, V = torch.pca_lowrank(X_centered, q=2)
    pc1 = V[:, 0]

    base_projections = base_X @ pc1
    audit_projections = audit_X @ pc1

    # Ensure PC1 points from base toward audit (so audit is "positive" direction)
    if base_projections.mean() > audit_projections.mean():
        pc1 = -pc1
        base_projections = -base_projections
        audit_projections = -audit_projections
        print("  [Flipped PC1 direction to point base→audit]")

    print(f"  PC1 Explained Variance: {(S[0]**2 / torch.sum(S**2)) * 100:.1f}%")
    print(f"  Mean PC1 Projection (Base):  {base_projections.mean().item():.4f}")
    print(f"  Mean PC1 Projection (Audit): {audit_projections.mean().item():.4f}")

    separation = (base_projections.mean() - audit_projections.mean()).abs()
    print(f"  Path Separation on PC1:      {separation.item():.4f}")

    # ======================== DRY RUN VERIFICATION ========================

    test_row = df.sample(1).iloc[0]
    while pd.isna(test_row["emp_length"]):
        test_row = df.sample(1).iloc[0]
    test_scenario = truncate_to_ctx(create_prompt_base(test_row))

    print("\n--- DRY RUN: LOGIC VERIFICATION ---")
    print(f"\n[ORIGINAL SCENARIO]\n{test_scenario}")
    print("-" * 40)

    print("\n[PATH A: BASE]")
    res_support = get_llm_base_support(test_scenario)
    print(f"  INITIAL DECISION: {res_support['prediction']}")
    print(f"  SUPPORT:          {res_support['support']}")
    print(f"  FINAL DECISION:   {res_support['del']}")

    print("\n" + "=" * 40)

    print("\n[PATH B: AUDITOR]")
    res_critique = get_sequential_inference(test_scenario)
    print(f"  INITIAL DECISION: {res_critique['prediction']}")
    print(f"  CRITIQUE:         {res_critique['critique']}")
    print(f"  FINAL DECISION:   {res_critique['del']}")

    print("\n--- CHECK COMPLETE ---")

    # ======================== SAVE STAGE 1 RESULTS ========================

    print(f"\nSaving SAE and activations to {CACHE_FILE}...")
    torch.save(
        {
            "base_X": base_X.cpu(),
            "audit_X": audit_X.cpu(),
            "results_metadata": results_metadata,
            "X_mean": X_mean.cpu(),
            "X_std": X_std.cpu(),
            "sae_state_dict": sae.state_dict(),
        },
        CACHE_FILE,
    )
    print(f"✓ Saved to {CACHE_FILE}")
    print(f"  (To skip Stage 1 next time, set SKIP_STAGE1=True)")

# ======================== STEERING TEST ========================

steering_vector_raw = (audit_X.mean(0) - base_X.mean(0)).to(device)
steering_vector = steering_vector_raw  # Use raw for now, but we compute normalized too

# Compute normalized version for comparison
steering_vector_normalized = steering_vector_raw / steering_vector_raw.norm()

print(f"\n[STEERING VECTOR STATS]")
print(f"  Shape: {steering_vector_raw.shape}")
print(f"  Norm: {steering_vector_raw.norm().item():.4f}")
print(f"  Mean: {steering_vector_raw.mean().item():.6f}")
print(f"  Std: {steering_vector_raw.std().item():.6f}")
print(f"  Min: {steering_vector_raw.min().item():.4f}")
print(f"  Max: {steering_vector_raw.max().item():.4f}")
print(f"\n  TIP: If steering doesn't work, try:")
print(f"       1. Increase COEFF (current: {COEFF})")
print(f"       2. Use normalized vector: steering_vector = steering_vector_normalized")
print(f"       3. Try different layers (current: {LAYER})")

hook_call_count = {"count": 0, "first_call": True}

def steering_hook(module, input, output):
    hook_call_count["count"] += 1

    # KEY CHANGE: Only steer on the first forward pass (the prompt), not during generation
    # The first call processes the entire prompt; subsequent calls are for generated tokens
    if not hook_call_count["first_call"]:
        return output

    hook_call_count["first_call"] = False  # Mark that we've done the first call

    # Only steer the last token position, matching how the vector was derived
    before = output[:, -1, :].clone()
    output[:, -1, :] = output[:, -1, :] + COEFF * steering_vector
    after = output[:, -1, :]

    delta = (after - before).norm().item()
    print(f"    [STEERING APPLIED] delta_norm={delta:.4f} on first forward pass")

    return output


def get_decision(prompt, is_steered):
    formatted = format_prompt(prompt)
    toks = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to(device)

    handle = None
    if is_steered:
        # Access layer: Qwen uses model.layers, GPT-2 uses transformer.h
        target_layer = model.model.layers[LAYER] if hasattr(model, 'model') else model.transformer.h[LAYER]
        handle = target_layer.mlp.register_forward_hook(steering_hook)

    with torch.no_grad():
        out = model.generate(toks, max_new_tokens=15, do_sample=False)  # Increased from 5 to 15

    if handle is not None:
        handle.remove()

    gen_tokens = out[0][toks.shape[1]:]
    raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    text = raw_text.strip().lower()
    text = re.sub(
        r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>|assistant|user", "", text
    ).strip()

    # Parse yes/no for delegation decision
    m = re.findall(r"\b(yes|no)\b", text)

    decision = "unknown"
    if m:
        last_match = m[-1]
        if last_match == "yes":
            decision = "delegate"
        elif last_match == "no":
            decision = "no-delegate"

    return decision, raw_text  # Return both for debugging


# Check how many disagreements we found during collection
disagreements = [
    m for m in results_metadata
    if m["base_decision"] == "no-delegate" and m["audit_decision"] == "delegate"
]
print(f"\n[INFO] Found {len(disagreements)}/{len(results_metadata)} samples where base='no-delegate' and audit='delegate'")
print(f"[INFO] Testing steering on Layer {LAYER} with strength {COEFF}")
print(f"[INFO] To try different values, modify LAYER (0-47) or COEFF in the script")

print(f"\nRunning Steering Test (Strength: {COEFF})...")
print("  Strategy: Only testing samples where BASE initially says 'no-delegate'")

n_steered = 0
attempts = 0
max_attempts = N_TEST * 20  # Avoid infinite loop
flip_count = 0  # Track how many flips occurred
flip_to_delegate_count = 0  # Track flips specifically to "delegate"

while n_steered < N_TEST and attempts < max_attempts:
    attempts += 1
    row = df.sample(1).iloc[0]
    if pd.isna(row["emp_length"]):
        continue

    gt = "no-delegate" if row["accepted"] == 1 else "delegate"
    prompt = (
        f"{truncate_to_ctx(create_prompt_base(row))}\n\n"
        "Should this decision be delegated to a human for review?\n"
        "Answer (yes or no):"
    )

    base_dec, base_text = get_decision(prompt, is_steered=False)

    # Only test steering on samples where base says "no-delegate"
    if base_dec != "no-delegate":
        continue

    n_steered += 1
    hook_call_count["count"] = 0  # Reset counter for this sample
    hook_call_count["first_call"] = True  # Reset first_call flag
    print(f"\n  [STEERING TEST {n_steered}/{N_TEST}]")
    print(f"    Base decision: {base_dec} (text: '{base_text}')")
    steer_dec, steer_text = get_decision(prompt, is_steered=True)
    print(f"    Steered decision: {steer_dec} (text: '{steer_text}')")
    print(f"    Hook was called {hook_call_count['count']} times")
    print(f"    Changed: {base_dec != steer_dec}")

    status = "FLIP!" if base_dec != steer_dec else "-"
    if base_dec != steer_dec:
        flip_count += 1
    if steer_dec == "delegate":
        status = "FLIP TO DELEGATE! ✓"
        if base_dec != steer_dec:
            flip_to_delegate_count += 1

    print(f"\n  Sample {n_steered} | GT: {gt:6}")
    print(f"    Base:    {base_dec:6}")
    print(f"    Steered: {steer_dec:6} | {status}")

print("\n" + "=" * 60)
print("STEERING TEST COMPLETE")
print("=" * 60)

# ======================== FLIP SUMMARY ========================
flip_rate = (flip_count / N_TEST) * 100
flip_to_delegate_rate = (flip_to_delegate_count / N_TEST) * 100

print("\n" + "=" * 60)
print("FLIP SUMMARY")
print("=" * 60)
print(f"\nTotal samples tested: {N_TEST}")
print(f"  (All samples had BASE decision = 'no-delegate')")
print(f"\nFlips (any change):          {flip_count}/{N_TEST} ({flip_rate:.1f}%)")
print(f"Flips to 'delegate':         {flip_to_delegate_count}/{N_TEST} ({flip_to_delegate_rate:.1f}%)")
print(f"\nSteering coefficient: {COEFF}")
print(f"Target layer: {LAYER}")
print("=" * 60)

print("\nTROUBLESHOOTING TIPS:")
print("  1. INCREASE COEFF: Try 15.0, 20.0, or even 50.0")
print("  2. TRY DIFFERENT LAYERS: Layer 16 (early), 24 (mid), 40 (late)")
print("  3. STEER ALL POSITIONS: Modify hook to steer output[:, :, :] instead of output[:, -1, :]")
print("  4. USE NORMALIZED VECTOR: Set steering_vector = steering_vector_normalized")
print("  5. TRY RESIDUAL STREAM: Hook on the layer itself, not just MLP")
print(f"\n  Current settings: LAYER={LAYER}, COEFF={COEFF}")
print(f"  Steering vector norm: {steering_vector_raw.norm().item():.2f}")
