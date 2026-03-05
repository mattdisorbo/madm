import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

# Parse results from job 276907 output
# Phase 1: patching flips per layer (20 tested per layer)
patching_flips = {
    0: 5, 1: 1, 2: 0, 3: 0, 4: 3, 5: 0, 6: 0, 7: 2, 8: 2, 9: 4,
    10: 1, 11: 3, 12: 0, 13: 2, 14: 3, 15: 1, 16: 3, 17: 0,
    18: 6, 19: 9, 20: 6, 21: 2, 22: 1, 23: 1, 24: 1, 25: 0,
    26: 1, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0,
}
n_tested = 20

# Phase 2: steering flips at top 3 layers (coeff=20, best performing)
# Layer 19: 4/20, Layer 18: 0/20, Layer 20: 4/20
steering_flips = {19: 4, 18: 0, 20: 4}

n_layers = 36
layers = np.arange(n_layers)
patch_rates = np.array([patching_flips[l] / n_tested for l in range(n_layers)])
steer_rates = np.array([steering_flips.get(l, np.nan) / n_tested for l in range(n_layers)])

fig, ax = plt.subplots(figsize=(7, 3))

bar_width = 0.7
ax.bar(layers, patch_rates * 100, bar_width, color="#4878CF", label="Activation patching", zorder=3)

# Overlay steering results where available
steer_layers = [l for l in range(n_layers) if l in steering_flips]
steer_vals = [steering_flips[l] / n_tested * 100 for l in steer_layers]
ax.bar(steer_layers, steer_vals, bar_width * 0.5, color="#D65F5F", label="Steering (coeff=20)", zorder=4)

ax.set_xlabel("Layer")
ax.set_ylabel("Flip rate (%)")
ax.set_xlim(-0.5, n_layers - 0.5)
ax.set_ylim(0, 55)
ax.set_xticks(range(0, n_layers, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", framealpha=0.9)
ax.axhline(0, color="black", linewidth=0.5)

# Annotate peak
ax.annotate("Layer 19\n45%", xy=(19, 45), xytext=(25, 48),
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

fig.tight_layout()

out_pdf = "fig_stage2.pdf"
out_png = "fig_stage2.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight")
print(f"Saved {out_pdf} and {out_png}")
