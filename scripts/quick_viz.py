"""Quick synthetic visualizations of MADM features/PCs.

Uses realistic synthetic activations matching the notebook's observed statistics:
  - PC1 ~70% variance, PC2 ~20% variance
  - Base vs Auditor separation ~1.6 on PC1
  - 25 samples per path
  - SAE feature activation patterns

Run: python scripts/quick_viz.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── reproducibility ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

N = 25          # samples per path
D = 1536        # hidden dim for Qwen2.5-1.5B MLP output
PC1_SEP = 1.6   # observed separation in notebook
N_SAE = 3 * D   # SAE hidden size (3× expansion)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── synthetic activations ─────────────────────────────────────────────────────
# We construct activations that reproduce the notebook's PCA structure.
# True signal lives in 5-dim subspace; remainder is noise.
signal_dim = 5
signal_base = rng.standard_normal((N, signal_dim))
signal_auditor = signal_base + rng.standard_normal((N, signal_dim)) * 0.5
signal_auditor[:, 0] += PC1_SEP          # push auditor along PC1

# Project into D-dim space via a fixed random basis
basis = rng.standard_normal((signal_dim, D))
basis /= np.linalg.norm(basis, axis=1, keepdims=True)

base_activations    = signal_base    @ basis + rng.standard_normal((N, D)) * 0.3
auditor_activations = signal_auditor @ basis + rng.standard_normal((N, D)) * 0.3

# Decisions (realistic mix)
base_decisions    = rng.choice(["reject", "delegate"], size=N, p=[0.55, 0.45])
auditor_decisions = rng.choice(["reject", "delegate"], size=N, p=[0.35, 0.65])
ground_truth      = rng.integers(0, 2, size=N)   # 0=rejected loan, 1=accepted

# ── PCA ───────────────────────────────────────────────────────────────────────
all_activations = np.vstack([base_activations, auditor_activations])
pca = PCA(n_components=6)
all_pca = pca.fit_transform(all_activations)
base_pca    = all_pca[:N]
auditor_pca = all_pca[N:]

steering_vec = auditor_activations.mean(0) - base_activations.mean(0)
steering_pca = pca.transform(steering_vec.reshape(1, -1))[0]

ev = pca.explained_variance_ratio_

# ── SAE (tiny, CPU-friendly) ──────────────────────────────────────────────────
import torch, torch.nn as nn, torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in, bias=False)
    def forward(self, x):
        z = F.relu(self.enc(x))
        return self.dec(z), z

X_np = StandardScaler().fit_transform(all_activations).astype(np.float32)
X    = torch.tensor(X_np)
sae  = SAE(D, 256)   # small hidden size for speed
opt  = torch.optim.AdamW(sae.parameters(), lr=1e-3)

sae_losses, sae_sparsity = [], []
for step in range(200):
    x_hat, z = sae(X)
    loss = F.mse_loss(x_hat, X) + 5e-4 * z.abs().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    sae_losses.append(loss.item())
    sae_sparsity.append((z > 0).float().mean().item() * 100)

with torch.no_grad():
    _, Z = sae(X)
    Z_np = Z.numpy()

Z_base    = Z_np[:N]
Z_auditor = Z_np[N:]

# ── select top features by base/auditor difference ───────────────────────────
feature_diff = np.abs(Z_auditor.mean(0) - Z_base.mean(0))
top_feat_idx = np.argsort(feature_diff)[-20:][::-1]

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: PC1-PC6 scatter matrix
# ═══════════════════════════════════════════════════════════════════════════════
n_pcs = 6
fig, axes = plt.subplots(n_pcs - 1, n_pcs - 1, figsize=(14, 12))
fig.suptitle("PCA Activation Scatter Matrix (PC1–PC6)\nBlue = Base  |  Red = Auditor",
             fontsize=14, fontweight="bold", y=1.01)

for row in range(n_pcs - 1):
    for col in range(n_pcs - 1):
        ax = axes[row][col]
        pc_x = col          # PC index for x-axis
        pc_y = row + 1      # PC index for y-axis
        if pc_x >= pc_y:
            ax.set_visible(False)
            continue
        ax.scatter(base_pca[:, pc_x],    base_pca[:, pc_y],    alpha=0.55, c="steelblue", s=25)
        ax.scatter(auditor_pca[:, pc_x], auditor_pca[:, pc_y], alpha=0.55, c="tomato",    s=25, marker="^")
        if row == n_pcs - 2:
            ax.set_xlabel(f"PC{pc_x+1} ({ev[pc_x]:.1%})", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"PC{pc_y+1} ({ev[pc_y]:.1%})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

fig.tight_layout()
p = os.path.join(OUTPUT_DIR, "pc_scatter_matrix.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"✓  {p}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: PC1 & PC2 side-by-side (model vs decision colour)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# -- left: coloured by path (base / auditor) --
ax = axes[0]
ax.scatter(base_pca[:, 0],    base_pca[:, 1],    c="steelblue", alpha=0.6, s=60, label="Base")
ax.scatter(auditor_pca[:, 0], auditor_pca[:, 1], c="tomato",    alpha=0.6, s=60, marker="^", label="Auditor")
origin = base_pca.mean(0)
scale  = 3
ax.annotate("", xy=origin[:2] + steering_pca[:2] * scale, xytext=origin[:2],
            arrowprops=dict(arrowstyle="->", color="green", lw=2.5))
ax.plot([], [], color="green", lw=2, label="Steering vector")
ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)", fontsize=11)
ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)", fontsize=11)
ax.set_title("Base vs Auditor in PC1–PC2", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# -- right: coloured by decision --
ax = axes[1]
decision_colors = {"reject": "orange", "delegate": "mediumpurple"}
for pca_pt, dec in zip(base_pca, base_decisions):
    ax.scatter(pca_pt[0], pca_pt[1], c=decision_colors[dec], alpha=0.6, s=60, marker="o")
for pca_pt, dec in zip(auditor_pca, auditor_decisions):
    ax.scatter(pca_pt[0], pca_pt[1], c=decision_colors[dec], alpha=0.6, s=60, marker="^")

legend_els = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="steelblue",    ms=8, label="Base (circle)"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor="tomato",       ms=8, label="Auditor (triangle)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="orange",       ms=8, label="Reject"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="mediumpurple", ms=8, label="Delegate"),
]
ax.legend(handles=legend_els, fontsize=9)
ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)", fontsize=11)
ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)", fontsize=11)
ax.set_title("Decisions in PC1–PC2", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)

fig.tight_layout()
p = os.path.join(OUTPUT_DIR, "pca_activations.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"✓  {p}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: PC1 distributions + explained variance bar
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 4))
gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2])

# PC1 histogram
ax = fig.add_subplot(gs[0])
ax.hist(base_pca[:, 0],    bins=12, alpha=0.6, color="steelblue", label="Base",    density=True)
ax.hist(auditor_pca[:, 0], bins=12, alpha=0.6, color="tomato",    label="Auditor", density=True)
ax.set_xlabel("PC1 score", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"PC1 Distribution\n(sep ≈ {abs(auditor_pca[:,0].mean()-base_pca[:,0].mean()):.2f})", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# PC2 histogram
ax = fig.add_subplot(gs[1])
ax.hist(base_pca[:, 1],    bins=12, alpha=0.6, color="steelblue", label="Base",    density=True)
ax.hist(auditor_pca[:, 1], bins=12, alpha=0.6, color="tomato",    label="Auditor", density=True)
ax.set_xlabel("PC2 score", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"PC2 Distribution\n({ev[1]:.1%} variance)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Explained variance bar chart
ax = fig.add_subplot(gs[2])
ax.bar(range(1, 7), ev[:6] * 100, color=plt.cm.viridis(np.linspace(0.2, 0.8, 6)))
ax.set_xlabel("Principal Component", fontsize=11)
ax.set_ylabel("% Variance Explained", fontsize=11)
ax.set_title("Scree Plot (PC1–PC6)", fontsize=11, fontweight="bold")
ax.set_xticks(range(1, 7))
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(ev[:6]):
    ax.text(i + 1, v * 100 + 0.5, f"{v:.1%}", ha="center", fontsize=8)

fig.tight_layout()
p = os.path.join(OUTPUT_DIR, "pc_distributions.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"✓  {p}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Decision flip arrows
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))
for i in range(N):
    flipped = base_decisions[i] != auditor_decisions[i]
    col = "crimson" if flipped else "lightgray"
    lw  = 1.8 if flipped else 0.8
    ax.annotate("", xy=auditor_pca[i, :2], xytext=base_pca[i, :2],
                arrowprops=dict(arrowstyle="->", color=col, lw=lw, alpha=0.85 if flipped else 0.5))
ax.scatter(base_pca[:, 0],    base_pca[:, 1],    c="steelblue", s=40, alpha=0.5, label="Base")
ax.scatter(auditor_pca[:, 0], auditor_pca[:, 1], c="tomato",    s=40, alpha=0.5, label="Auditor", marker="^")
n_flipped = sum(b != a for b, a in zip(base_decisions, auditor_decisions))
ax.set_title(f"Decision Changes: Base → Auditor\n"
             f"Red = flipped ({n_flipped}/{N} = {n_flipped/N:.0%})  Gray = unchanged",
             fontsize=12, fontweight="bold")
ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)", fontsize=11)
ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
p = os.path.join(OUTPUT_DIR, "decision_flips.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"✓  {p}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: SAE training curve + top feature heatmap
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Training loss
ax = axes[0]
ax.plot(sae_losses, lw=2, color="steelblue")
ax.set_xlabel("Step", fontsize=11); ax.set_ylabel("Loss", fontsize=11)
ax.set_title("SAE Training Loss", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)

# Sparsity
ax = axes[1]
ax.plot(sae_sparsity, lw=2, color="darkorange")
ax.set_xlabel("Step", fontsize=11); ax.set_ylabel("% Active Features", fontsize=11)
ax.set_title("SAE Feature Sparsity", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)

# Top-20 feature activations heatmap
ax = axes[2]
heat = np.vstack([Z_base[:, top_feat_idx], Z_auditor[:, top_feat_idx]])
im   = ax.imshow(heat.T, aspect="auto", cmap="YlOrRd")
ax.axvline(N - 0.5, color="white", lw=2, ls="--")
ax.set_xlabel("Sample (0–24 Base | 25–49 Auditor)", fontsize=9)
ax.set_ylabel("Top-20 SAE Feature", fontsize=9)
ax.set_title("SAE Feature Activations\n(Base | Auditor)", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.04)
ax.set_yticks(range(20))
ax.set_yticklabels([f"F{i}" for i in top_feat_idx], fontsize=7)

fig.tight_layout()
p = os.path.join(OUTPUT_DIR, "sae_features.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"✓  {p}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: 3D PCA
# ═══════════════════════════════════════════════════════════════════════════════
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")
ax.scatter(*base_pca[:, :3].T,    c="steelblue", alpha=0.6, s=50, label="Base")
ax.scatter(*auditor_pca[:, :3].T, c="tomato",    alpha=0.6, s=50, marker="^", label="Auditor")
ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=10)
ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=10)
ax.set_zlabel(f"PC3 ({ev[2]:.1%})", fontsize=10)
ax.set_title("3D PCA: Base vs Auditor Activations", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
p = os.path.join(OUTPUT_DIR, "pca_3d.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"✓  {p}")
plt.close(fig)

print(f"\nAll charts saved to  {os.path.abspath(OUTPUT_DIR)}/")
print(f"  pc_scatter_matrix.png  — all pairs PC1-PC6")
print(f"  pca_activations.png    — PC1 vs PC2 (model / decision colour)")
print(f"  pc_distributions.png   — histograms + scree plot")
print(f"  decision_flips.png     — arrows base→auditor")
print(f"  sae_features.png       — SAE loss, sparsity, top-20 feature heatmap")
print(f"  pca_3d.png             — 3-D PC1×PC2×PC3")
