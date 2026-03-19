"""
Plot escalation rate vs base rate (accuracy from hint) for the DPO model.

Shows the model's escalation decisions across all datasets and cost ratios,
compared to the optimal threshold.
"""

import matplotlib.pyplot as plt
import numpy as np

COST_RATIOS = [2, 4, 8, 10, 20, 50]

# All base rates across datasets
ALL_BASE_RATES = {
    "HotelBookings": [0.523, 0.554, 0.622, 0.661, 0.716, 0.783, 0.818, 0.866, 0.906, 0.950],
    "MoralMachine": [0.50, 0.54, 0.57, 0.64, 0.67, 0.74, 0.79, 0.84, 0.87, 0.91],
    "LendingClub": [0.51, 0.53, 0.64, 0.67, 0.75, 0.83, 0.90, 0.91, 0.92, 0.93],
    "WikipediaToxicity": [0.52, 0.61, 0.67, 0.74, 0.85, 0.87, 0.92, 0.92, 0.94, 0.96],
    "MovieLens": [0.514, 0.545, 0.606, 0.667, 0.732, 0.801, 0.851, 0.888, 0.907, 0.942],
}

# Model accuracy from eval (per dataset)
MODEL_ACC = {
    "HotelBookings": 1.0,
    "MoralMachine": 1.0,
    "LendingClub": 0.95,
    "WikipediaToxicity": 0.933,
    "MovieLens": 0.90,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left panel: Optimal escalation thresholds ---
ax = axes[0]
base_rates = np.linspace(0.5, 1.0, 200)

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(COST_RATIOS)))
for R, color in zip(COST_RATIOS, colors):
    threshold = 1 - 1 / (1 + R)  # base_rate below this → escalate
    escalate = (base_rates < threshold).astype(float)
    ax.plot(base_rates, escalate, color=color, label=f"R={R}", linewidth=2, alpha=0.8)
    # Mark threshold
    ax.axvline(x=threshold, color=color, linestyle=":", alpha=0.3)

ax.set_xlabel("Base Rate (from hint)", fontsize=12)
ax.set_ylabel("Escalation Decision", fontsize=12)
ax.set_title("Optimal Policy: Escalate if R×(1 - base_rate) > 1", fontsize=13)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Implement (0)", "Escalate (1)"])
ax.legend(title="Cost Ratio", fontsize=9, title_fontsize=10)
ax.set_xlim(0.48, 1.0)

# --- Right panel: Model accuracy by dataset ---
ax = axes[1]
datasets = list(MODEL_ACC.keys())
accs = [MODEL_ACC[d] for d in datasets]
colors_bar = ["#2ecc71" if d == "HotelBookings" else "#3498db" for d in datasets]

bars = ax.bar(range(len(datasets)), accs, color=colors_bar, edgecolor="white", linewidth=1.5)
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("DPO Model: Cross-Dataset Generalization", fontsize=13)
ax.set_ylim(0.8, 1.02)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

# Add percentage labels on bars
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{acc:.0%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Add legend for trained vs unseen
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2ecc71", label="Trained"),
                   Patch(facecolor="#3498db", label="Unseen")]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig("paper/figures/dpo_escalation_results.png", dpi=200, bbox_inches="tight")
print("Saved to paper/figures/dpo_escalation_results.png")
plt.close()

# --- Second figure: Escalation heatmap ---
fig, ax = plt.subplots(figsize=(10, 6))

# Combine all base rates
all_brs = sorted(set(br for brs in ALL_BASE_RATES.values() for br in brs))

# Build heatmap: for each (base_rate, R), what's the optimal decision?
heatmap = np.zeros((len(COST_RATIOS), len(all_brs)))
for i, R in enumerate(COST_RATIOS):
    for j, br in enumerate(all_brs):
        optimal = 1 if R * (1 - br) > 1 else 0
        heatmap[i, j] = optimal

im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
               extent=[-0.5, len(all_brs) - 0.5, -0.5, len(COST_RATIOS) - 0.5],
               origin="lower")

ax.set_xticks(range(0, len(all_brs), 3))
ax.set_xticklabels([f"{all_brs[i]:.0%}" for i in range(0, len(all_brs), 3)], fontsize=9)
ax.set_yticks(range(len(COST_RATIOS)))
ax.set_yticklabels([str(R) for R in COST_RATIOS])
ax.set_xlabel("Base Rate (from hint)", fontsize=12)
ax.set_ylabel("Cost Ratio R", fontsize=12)
ax.set_title("Optimal Escalation Policy\n(Red = Escalate, Green = Implement)", fontsize=13)

# Add colorbar
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#2ecc71", "#e74c3c"])
im.set_cmap(cmap)

plt.tight_layout()
plt.savefig("paper/figures/dpo_escalation_heatmap.png", dpi=200, bbox_inches="tight")
print("Saved to paper/figures/dpo_escalation_heatmap.png")
plt.close()
