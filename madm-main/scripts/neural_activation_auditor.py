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

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Much more capable than GPT-2
N_SAMPLES = 5
N_TEST = 3
LAYER = 14  # Qwen2.5-1.5B has 28 layers; using middle layer
SAE_STEPS = 150
MAX_CTX = 512
RESERVE = 16
COEFF = 2.0  # Steering strength

ACCEPTED_CSV = "data/accepted_10k.csv"
REJECTED_CSV = "data/rejected_10k.csv"

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


def run_final_decision(prompt: str, max_tokens: int = 5):
    r = get_llm_base(prompt, max_tokens=max_tokens)
    generated_text = r.get("text", "").lower()
    m = re.findall(r"\b(reject|delegate)\b", generated_text)
    decision = m[-1] if m else None

    return {
        "del": decision,
        "cache": r["cache"],
        "tokens": r["tokens"],
        "text": generated_text,
        "final_prompt": prompt,
    }


# ======================== REASONING PATHS ========================


def get_llm_base_support(base_scenario: str):
    """Base path: direct decision without supporting argument."""
    print("    [BASE PATH] Getting decision...")
    decision_prompt = (
        f"{base_scenario}\n\n"
        "Would you reject this loan or delegate the decision to a human?\n"
        "Answer (reject or delegate):"
    )
    out = run_final_decision(decision_prompt, max_tokens=10)
    print(f"    [BASE PATH] Decision: '{out['text']}'")
    out.update({"prediction": out.get("text", ""), "support": ""})
    return out


def get_sequential_inference(base_scenario: str):
    """Auditor path: initial decision + critique + final decision."""
    print("    [AUDIT PATH] Getting initial decision...")
    pred_prompt = (
        f"{base_scenario}\n\n"
        "Would you reject this loan or delegate the decision to a human?\n"
        "Answer (reject or delegate):"
    )
    prediction = get_llm_base(pred_prompt, max_tokens=5).get("text", "").strip()
    print(f"    [AUDIT PATH] Initial: '{prediction}'")

    print("    [AUDIT PATH] Getting critique...")
    critique_prompt = (
        f"{base_scenario}\n\n"
        f"I initially think: {prediction}\n"
        "However, one concern is:"
    )
    critique_text = get_llm_base(critique_prompt, max_tokens=150).get("text", "").strip()
    critique = "However, one concern is: " + critique_text
    print(f"    [AUDIT PATH] Critique generated")

    print("    [AUDIT PATH] Getting final decision...")
    final_prompt = (
        f"{base_scenario}\n\n"
        f"Initial thought: {prediction}\n"
        f"Concern: {critique}\n\n"
        "Given these concerns, should I reject this loan or delegate it to a human for careful review?\n"
        "Final answer (reject or delegate):"
    )
    out = run_final_decision(final_prompt, max_tokens=10)
    print(f"    [AUDIT PATH] Final: '{out['text']}'")
    out.update({"prediction": prediction, "critique": critique})
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

    ground_truth = "reject" if row["accepted"] == 0 else "delegate"
    scenario = truncate_to_ctx(create_prompt_base(row))
    print(f"  Scenario: {scenario[:100]}...")
    print(f"  Ground truth: {ground_truth}")

    b_res = get_llm_base_support(scenario)
    print()
    a_res = get_sequential_inference(scenario)

    if b_res["del"] and a_res["del"]:
        base_X.append(decision_activation(b_res, LAYER).detach().cpu())
        audit_X.append(decision_activation(a_res, LAYER).detach().cpu())

        results_metadata.append(
            {
                "ground_truth": ground_truth,
                "base_decision": b_res["del"],
                "audit_decision": a_res["del"],
            }
        )

        print(
            f"\n  ✓ SUCCESS! Sample {len(base_X)}/{N_SAMPLES} collected | "
            f"Actual: {ground_truth} | Base: {b_res['del']} | Audit: {a_res['del']}"
        )
        print("=" * 60)
    else:
        print(f"\n  ✗ SKIP | Base decision: '{b_res['del']}' | Audit decision: '{a_res['del']}'")
        print(f"    Base text: '{b_res['text']}'")
        print(f"    Audit text: '{a_res['text']}'")
        print("=" * 60)

base_X = torch.stack(base_X).float().to(device)
audit_X = torch.stack(audit_X).float().to(device)

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

# ======================== ACCURACY ========================

base_correct = sum(
    1 for m in results_metadata if m["base_decision"] == m["ground_truth"]
)
audit_correct = sum(
    1 for m in results_metadata if m["audit_decision"] == m["ground_truth"]
)

base_acc = (base_correct / N_SAMPLES) * 100
audit_acc = (audit_correct / N_SAMPLES) * 100

print(f"\nACCURACY REPORT")
print(f"  Base Accuracy (Support):   {base_acc:.1f}%")
print(f"  Audit Accuracy (Critique): {audit_acc:.1f}%")
print(f"  Accuracy Delta:            {audit_acc - base_acc:+.1f}%")

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

# ======================== STEERING TEST ========================

steering_vector = (audit_X.mean(0) - base_X.mean(0)).to(device)


def steering_hook(module, input, output):
    # Only steer the last token position, matching how the vector was derived
    output[:, -1, :] = output[:, -1, :] + COEFF * steering_vector
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
        out = model.generate(toks, max_new_tokens=5, do_sample=False)

    if handle is not None:
        handle.remove()

    gen_tokens = out[0][toks.shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()
    text = re.sub(
        r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>|assistant|user", "", text
    ).strip()

    if "reject" in text:
        return "reject"
    if "delegate" in text:
        return "delegate"
    return "unknown"


print(f"\nRunning Steering Test (Strength: {COEFF})...")

n_steered = 0
while n_steered < N_TEST:
    row = df.sample(1).iloc[0]
    if pd.isna(row["emp_length"]):
        continue
    n_steered += 1
    gt = "delegate" if row["accepted"] == 1 else "reject"
    prompt = (
        f"{truncate_to_ctx(create_prompt_base(row))}\n\n"
        "Would you reject this loan or delegate the decision to a human?\n"
        "Answer (reject or delegate):"
    )

    base_dec = get_decision(prompt, is_steered=False)
    steer_dec = get_decision(prompt, is_steered=True)

    status = "FLIP!" if base_dec != steer_dec else "-"
    if steer_dec == gt and base_dec != gt:
        status = "CORRECTIVE FLIP!"

    print(f"\n  Sample {n_steered} | GT: {gt:6}")
    print(f"    Base:    {base_dec:6}")
    print(f"    Steered: {steer_dec:6} | {status}")
