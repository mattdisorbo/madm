"""Visualize PCA of base vs auditor activations and steering vector.

This script:
1. Collects activations from base and auditor decisions
2. Computes PCA and plots PC1 vs PC2
3. Shows steering vector direction
4. Trains SAE and visualizes learned features
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ======================== CONFIG ========================

MODEL_NAME = "Qwen/Qwen3-4B"
N_SAMPLES = 25  # Number of samples to collect for visualization
LAYER = 28
MAX_CTX = 512
RESERVE = 16
SAE_STEPS = 150

ACCEPTED_CSV = "data/accepted_10k.csv"
REJECTED_CSV = "data/rejected_10k.csv"
OUTPUT_DIR = "../results/visualizations/"

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


# ======================== COLLECT ACTIVATIONS ========================

print(f"\n{'='*60}")
print(f"Collecting {N_SAMPLES} samples for visualization...")
print(f"{'='*60}\n")

base_activations = []
auditor_activations = []
base_decisions = []
auditor_decisions = []
ground_truth = []

attempt = 0

while len(base_activations) < N_SAMPLES:
    attempt += 1
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
            base_activations.append(base_result["cache"]["mlp_out"][0, -1].detach().cpu().numpy())
            auditor_activations.append(audit_result["cache"]["mlp_out"][0, -1].detach().cpu().numpy())
            base_decisions.append(base_decision)
            auditor_decisions.append(audit_decision)
            ground_truth.append(row["accepted"])

            print(f"  Sample {len(base_activations)}/{N_SAMPLES} | Base: {base_decision} | Audit: {audit_decision} | GT: {'accepted' if row['accepted'] else 'rejected'}")

    except Exception as e:
        print(f"  Error: {e}")
        continue

base_activations = np.array(base_activations)
auditor_activations = np.array(auditor_activations)

print(f"\n✓ Collected {len(base_activations)} samples")
print(f"  Base activations shape: {base_activations.shape}")
print(f"  Auditor activations shape: {auditor_activations.shape}")

# ======================== COMPUTE STEERING VECTOR ========================

steering_vector = auditor_activations.mean(axis=0) - base_activations.mean(axis=0)
print(f"\nSteering vector norm: {np.linalg.norm(steering_vector):.4f}")

# ======================== PCA VISUALIZATION ========================

print(f"\n{'='*60}")
print("Computing PCA...")
print(f"{'='*60}\n")

# Combine all activations for PCA
all_activations = np.vstack([base_activations, auditor_activations])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_activations)

# Split back into base and auditor
base_pca = all_pca[:len(base_activations)]
auditor_pca = all_pca[len(base_activations):]

# Transform steering vector to PCA space
steering_pca = pca.transform(steering_vector.reshape(1, -1))[0]

print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

# ======================== CREATE VISUALIZATIONS ========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Figure 1: Base vs Auditor in PCA space
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Colored by model type
ax = axes[0]
ax.scatter(base_pca[:, 0], base_pca[:, 1], alpha=0.5, label='Base Model', c='blue', s=50)
ax.scatter(auditor_pca[:, 0], auditor_pca[:, 1], alpha=0.5, label='Auditor Model', c='red', s=50)

# Add steering vector
origin = np.mean(base_pca, axis=0)
ax.arrow(origin[0], origin[1], steering_pca[0]*2, steering_pca[1]*2,
         head_width=0.3, head_length=0.3, fc='green', ec='green', linewidth=3, label='Steering Vector')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('Base vs Auditor Activations in PCA Space', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Colored by decision
ax = axes[1]
for i, (pca_point, decision) in enumerate(zip(base_pca, base_decisions)):
    color = 'orange' if decision == 'reject' else 'purple'
    marker = 'o'
    ax.scatter(pca_point[0], pca_point[1], alpha=0.5, c=color, s=50, marker=marker)

for i, (pca_point, decision) in enumerate(zip(auditor_pca, auditor_decisions)):
    color = 'orange' if decision == 'reject' else 'purple'
    marker = '^'
    ax.scatter(pca_point[0], pca_point[1], alpha=0.5, c=color, s=50, marker=marker)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Base Model', alpha=0.5),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label='Auditor Model', alpha=0.5),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Reject', alpha=0.5),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Delegate', alpha=0.5),
]
ax.legend(handles=legend_elements, fontsize=10)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('Activations Colored by Decision', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_activations.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}pca_activations.png")

# Figure 2: Decision flip analysis
fig, ax = plt.subplots(figsize=(10, 8))

for i in range(len(base_activations)):
    base_point = base_pca[i]
    audit_point = auditor_pca[i]

    # Color based on whether decision flipped
    flipped = base_decisions[i] != auditor_decisions[i]
    color = 'red' if flipped else 'gray'
    alpha = 0.8 if flipped else 0.2

    # Draw arrow from base to auditor
    ax.annotate('', xy=audit_point, xytext=base_point,
                arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=1.5))

ax.scatter(base_pca[:, 0], base_pca[:, 1], alpha=0.3, c='blue', s=30, label='Base')
ax.scatter(auditor_pca[:, 0], auditor_pca[:, 1], alpha=0.3, c='red', s=30, label='Auditor')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('Decision Changes: Base → Auditor\n(Red arrows = decision flipped)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'decision_flips.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}decision_flips.png")

# Figure 3: 3D visualization with PC1, PC2, PC3
pca_3d = PCA(n_components=3)
all_pca_3d = pca_3d.fit_transform(all_activations)
base_pca_3d = all_pca_3d[:len(base_activations)]
auditor_pca_3d = all_pca_3d[len(base_activations):]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(base_pca_3d[:, 0], base_pca_3d[:, 1], base_pca_3d[:, 2],
           alpha=0.5, c='blue', s=50, label='Base Model')
ax.scatter(auditor_pca_3d[:, 0], auditor_pca_3d[:, 1], auditor_pca_3d[:, 2],
           alpha=0.5, c='red', s=50, label='Auditor Model')

ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=11)
ax.set_title('3D PCA: Base vs Auditor Activations', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_3d.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}pca_3d.png")

# ======================== TRAIN SAE AND VISUALIZE ========================

print(f"\n{'='*60}")
print("Training SAE...")
print(f"{'='*60}\n")

X = torch.tensor(all_activations, dtype=torch.float32).to(device)
d_in = X.shape[1]
sae = SAE(d_in, 2 * d_in).to(device)
opt = torch.optim.AdamW(sae.parameters(), lr=1e-3)

X_mean, X_std = X.mean(0), X.std(0) + 1e-6
Xn = (X - X_mean) / X_std

losses = []
for step in range(SAE_STEPS):
    x_hat, z = sae(Xn)
    l1_loss = z.abs().mean()
    loss = F.mse_loss(x_hat, Xn) + 5e-4 * l1_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(loss.item())

    if step % 50 == 0:
        active_pct = (z > 0).float().mean().item() * 100
        print(f"  Step {step:3} | Loss: {loss:.4f} | Active: {active_pct:.1f}%")

print(f"\n✓ SAE trained!")

# Plot training loss
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(losses, linewidth=2)
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('SAE Training Loss', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sae_training_loss.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}sae_training_loss.png")

# ======================== SUMMARY STATISTICS ========================

print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}\n")

# Decision statistics
base_reject_pct = sum(1 for d in base_decisions if d == 'reject') / len(base_decisions) * 100
base_delegate_pct = sum(1 for d in base_decisions if d == 'delegate') / len(base_decisions) * 100

audit_reject_pct = sum(1 for d in auditor_decisions if d == 'reject') / len(auditor_decisions) * 100
audit_delegate_pct = sum(1 for d in auditor_decisions if d == 'delegate') / len(auditor_decisions) * 100

flip_count = sum(1 for b, a in zip(base_decisions, auditor_decisions) if b != a)
flip_pct = flip_count / len(base_decisions) * 100

print(f"Base Model Decisions:")
print(f"  Reject: {base_reject_pct:.1f}%")
print(f"  Delegate: {base_delegate_pct:.1f}%")

print(f"\nAuditor Model Decisions:")
print(f"  Reject: {audit_reject_pct:.1f}%")
print(f"  Delegate: {audit_delegate_pct:.1f}%")

print(f"\nDecision Changes:")
print(f"  Flipped: {flip_count}/{len(base_decisions)} ({flip_pct:.1f}%)")

print(f"\nPCA Variance Explained:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"  Total (PC1+PC2): {sum(pca.explained_variance_ratio_[:2]):.2%}")

print(f"\n{'='*60}")
print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
print(f"{'='*60}\n")
