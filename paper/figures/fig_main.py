"""Generate the main 3-panel figure: prediction accuracy, escalation rate, final accuracy.

Aggregated across all datasets and all Qwen3 models into a single bar per method.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

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

data = json.loads(r'''
[{"dataset": "FEVEROUS", "model": "Qwen3-14B", "method": "adversarial", "n": 100, "pred_acc": 0.52, "esc_rate": 0.96, "final_acc": 0.97, "naive_acc": 0.56}, {"dataset": "FEVEROUS", "model": "Qwen3-1.7B", "method": "base", "n": 100, "pred_acc": 0.46, "esc_rate": 0.14, "final_acc": 0.47, "naive_acc": 0.64}, {"dataset": "FEVEROUS", "model": "Qwen3-8B", "method": "adversarial", "n": 100, "pred_acc": 0.56, "esc_rate": 0.6, "final_acc": 0.74, "naive_acc": 0.67}, {"dataset": "FEVEROUS", "model": "Qwen3-14B", "method": "base", "n": 100, "pred_acc": 0.49, "esc_rate": 0.17, "final_acc": 0.53, "naive_acc": 0.6}, {"dataset": "FEVEROUS", "model": "Qwen3-8B", "method": "base", "n": 100, "pred_acc": 0.56, "esc_rate": 0.01, "final_acc": 0.57, "naive_acc": 0.58}, {"dataset": "FEVEROUS", "model": "Qwen3-1.7B", "method": "adversarial", "n": 100, "pred_acc": 0.39, "esc_rate": 0.12, "final_acc": 0.46, "naive_acc": 0.61}, {"dataset": "HotelBookings", "model": "Qwen3-1.7B", "method": "base", "n": 100, "pred_acc": 0.55, "esc_rate": 0.83, "final_acc": 0.89, "naive_acc": 0.6}, {"dataset": "HotelBookings", "model": "Qwen3-8B", "method": "base", "n": 100, "pred_acc": 0.65, "esc_rate": 0.91, "final_acc": 0.98, "naive_acc": 0.6}, {"dataset": "HotelBookings", "model": "Qwen3-14B", "method": "adversarial", "n": 100, "pred_acc": 0.67, "esc_rate": 0.29, "final_acc": 0.73, "naive_acc": 0.64}, {"dataset": "HotelBookings", "model": "Qwen3-1.7B", "method": "adversarial", "n": 100, "pred_acc": 0.65, "esc_rate": 0.58, "final_acc": 0.88, "naive_acc": 0.65}, {"dataset": "HotelBookings", "model": "Qwen3-8B", "method": "adversarial", "n": 100, "pred_acc": 0.62, "esc_rate": 0.98, "final_acc": 1.0, "naive_acc": 0.6}, {"dataset": "HotelBookings", "model": "Qwen3-4B", "method": "base", "n": 100, "pred_acc": 0.42, "esc_rate": 0.01, "final_acc": 0.43, "naive_acc": 0.57}, {"dataset": "HotelBookings", "model": "Qwen3-14B", "method": "base", "n": 100, "pred_acc": 0.7, "esc_rate": 0.93, "final_acc": 0.98, "naive_acc": 0.67}, {"dataset": "HotelBookings", "model": "Qwen3-4B", "method": "adversarial", "n": 100, "pred_acc": 0.59, "esc_rate": 0.99, "final_acc": 1.0, "naive_acc": 0.56}, {"dataset": "LendingClub", "model": "Qwen3-8B", "method": "base", "n": 100, "pred_acc": 0.93, "esc_rate": 0.74, "final_acc": 0.96, "naive_acc": 0.75}, {"dataset": "LendingClub", "model": "Qwen3-1.7B", "method": "base", "n": 16, "pred_acc": 0.75, "esc_rate": 0.25, "final_acc": 0.8125, "naive_acc": 0.5625}, {"dataset": "LendingClub", "model": "Qwen3-14B", "method": "adversarial", "n": 100, "pred_acc": 0.91, "esc_rate": 0.93, "final_acc": 1.0, "naive_acc": 0.82}, {"dataset": "LendingClub", "model": "Qwen3-8B", "method": "adversarial", "n": 100, "pred_acc": 0.87, "esc_rate": 0.96, "final_acc": 0.99, "naive_acc": 0.76}, {"dataset": "LendingClub", "model": "Qwen3-14B", "method": "base", "n": 100, "pred_acc": 0.9, "esc_rate": 0.66, "final_acc": 0.92, "naive_acc": 0.77}, {"dataset": "LendingClub", "model": "Qwen3-1.7B", "method": "adversarial", "n": 100, "pred_acc": 0.87, "esc_rate": 0.66, "final_acc": 0.94, "naive_acc": 0.76}, {"dataset": "MovieLens", "model": "Qwen3-4B", "method": "adversarial", "n": 100, "pred_acc": 0.51, "esc_rate": 0.85, "final_acc": 0.9, "naive_acc": 0.5}, {"dataset": "MovieLens", "model": "Qwen3-8B", "method": "adversarial", "n": 100, "pred_acc": 0.53, "esc_rate": 0.45, "final_acc": 0.73, "naive_acc": 0.5}, {"dataset": "MovieLens", "model": "Qwen3-14B", "method": "adversarial", "n": 100, "pred_acc": 0.56, "esc_rate": 1.0, "final_acc": 1.0, "naive_acc": 0.5}, {"dataset": "MovieLens", "model": "Qwen3-1.7B", "method": "adversarial", "n": 99, "pred_acc": 0.505, "esc_rate": 0.202, "final_acc": 0.566, "naive_acc": 0.5}, {"dataset": "MovieLens", "model": "Qwen3-4B", "method": "base", "n": 100, "pred_acc": 0.44, "esc_rate": 0.0, "final_acc": 0.44, "naive_acc": 0.5}, {"dataset": "MovieLens", "model": "Qwen3-1.7B", "method": "base", "n": 85, "pred_acc": 0.494, "esc_rate": 0.976, "final_acc": 0.988, "naive_acc": 0.5}, {"dataset": "MovieLens", "model": "Qwen3-8B", "method": "base", "n": 100, "pred_acc": 0.45, "esc_rate": 0.14, "final_acc": 0.52, "naive_acc": 0.5}, {"dataset": "MovieLens", "model": "Qwen3-14B", "method": "base", "n": 100, "pred_acc": 0.53, "esc_rate": 0.73, "final_acc": 0.86, "naive_acc": 0.5}, {"dataset": "WikipediaToxicity", "model": "Qwen3-14B", "method": "base", "n": 100, "pred_acc": 0.83, "esc_rate": 0.11, "final_acc": 0.93, "naive_acc": 0.93}, {"dataset": "WikipediaToxicity", "model": "Qwen3-1.7B", "method": "adversarial", "n": 100, "pred_acc": 0.63, "esc_rate": 0.39, "final_acc": 0.87, "naive_acc": 0.83}, {"dataset": "WikipediaToxicity", "model": "Qwen3-1.7B", "method": "base", "n": 98, "pred_acc": 0.898, "esc_rate": 0.102, "final_acc": 0.918, "naive_acc": 0.867}, {"dataset": "WikipediaToxicity", "model": "Qwen3-4B", "method": "base", "n": 100, "pred_acc": 0.85, "esc_rate": 0.24, "final_acc": 0.94, "naive_acc": 0.79}, {"dataset": "WikipediaToxicity", "model": "Qwen3-14B", "method": "adversarial", "n": 100, "pred_acc": 0.9, "esc_rate": 0.52, "final_acc": 0.98, "naive_acc": 0.84}, {"dataset": "WikipediaToxicity", "model": "Qwen3-8B", "method": "adversarial", "n": 100, "pred_acc": 0.78, "esc_rate": 0.3, "final_acc": 0.9, "naive_acc": 0.85}, {"dataset": "WikipediaToxicity", "model": "Qwen3-4B", "method": "adversarial", "n": 100, "pred_acc": 0.78, "esc_rate": 0.6, "final_acc": 0.97, "naive_acc": 0.78}, {"dataset": "WikipediaToxicity", "model": "Qwen3-8B", "method": "base", "n": 100, "pred_acc": 0.88, "esc_rate": 0.1, "final_acc": 0.89, "naive_acc": 0.8}, {"dataset": "MoralMachine", "model": "Qwen3-8B", "method": "base", "n": 100, "pred_acc": 0.51, "esc_rate": 0.19, "final_acc": 0.59, "naive_acc": 0.51}, {"dataset": "MoralMachine", "model": "Qwen3-4B", "method": "adversarial", "n": 100, "pred_acc": 0.43, "esc_rate": 0.59, "final_acc": 0.76, "naive_acc": 0.51}, {"dataset": "MoralMachine", "model": "Qwen3-14B", "method": "base", "n": 100, "pred_acc": 0.58, "esc_rate": 0.0, "final_acc": 0.58, "naive_acc": 0.58}, {"dataset": "MoralMachine", "model": "Qwen3-4B", "method": "base", "n": 100, "pred_acc": 0.62, "esc_rate": 0.02, "final_acc": 0.64, "naive_acc": 0.64}, {"dataset": "MoralMachine", "model": "Qwen3-8B", "method": "adversarial", "n": 100, "pred_acc": 0.51, "esc_rate": 0.22, "final_acc": 0.6, "naive_acc": 0.54}, {"dataset": "MoralMachine", "model": "Qwen3-14B", "method": "adversarial", "n": 100, "pred_acc": 0.47, "esc_rate": 0.89, "final_acc": 0.96, "naive_acc": 0.55}, {"dataset": "MoralMachine", "model": "Qwen3-1.7B", "method": "adversarial", "n": 100, "pred_acc": 0.56, "esc_rate": 0.41, "final_acc": 0.74, "naive_acc": 0.61}, {"dataset": "MoralMachine", "model": "Qwen3-1.7B", "method": "base", "n": 87, "pred_acc": 0.575, "esc_rate": 0.138, "final_acc": 0.644, "naive_acc": 0.575}, {"dataset": "AIME", "model": "Qwen3-14B", "method": "base", "n": 100, "pred_acc": 0.03, "esc_rate": 0.26, "final_acc": 0.29, "naive_acc": 0.0}, {"dataset": "AIME", "model": "Qwen3-1.7B", "method": "base", "n": 97, "pred_acc": 0.0, "esc_rate": 0.742, "final_acc": 0.742, "naive_acc": 0.0}, {"dataset": "AIME", "model": "Qwen3-14B", "method": "adversarial", "n": 100, "pred_acc": 0.37, "esc_rate": 0.96, "final_acc": 1.0, "naive_acc": 0.0}, {"dataset": "AIME", "model": "Qwen3-4B", "method": "base", "n": 100, "pred_acc": 0.01, "esc_rate": 0.06, "final_acc": 0.06, "naive_acc": 0.0}, {"dataset": "AIME", "model": "Qwen3-1.7B", "method": "adversarial", "n": 100, "pred_acc": 0.16, "esc_rate": 0.63, "final_acc": 0.75, "naive_acc": 0.0}, {"dataset": "AIME", "model": "Qwen3-4B", "method": "adversarial", "n": 100, "pred_acc": 0.34, "esc_rate": 0.32, "final_acc": 0.52, "naive_acc": 0.0}, {"dataset": "AIME", "model": "Qwen3-8B", "method": "adversarial", "n": 100, "pred_acc": 0.28, "esc_rate": 0.49, "final_acc": 0.75, "naive_acc": 0.0}, {"dataset": "AIME", "model": "Qwen3-8B", "method": "base", "n": 100, "pred_acc": 0.01, "esc_rate": 0.04, "final_acc": 0.05, "naive_acc": 0.0}, {"dataset": "Uber", "model": "Qwen3-4B", "method": "adversarial", "n": 100, "pred_acc": 0.56, "esc_rate": 0.99, "final_acc": 0.99, "naive_acc": 0.57}, {"dataset": "Uber", "model": "Qwen3-14B", "method": "adversarial", "n": 100, "pred_acc": 0.68, "esc_rate": 0.83, "final_acc": 0.95, "naive_acc": 0.7}, {"dataset": "Uber", "model": "Qwen3-8B", "method": "base", "n": 100, "pred_acc": 0.62, "esc_rate": 0.74, "final_acc": 0.88, "naive_acc": 0.67}, {"dataset": "Uber", "model": "Qwen3-4B", "method": "base", "n": 100, "pred_acc": 0.3, "esc_rate": 0.0, "final_acc": 0.3, "naive_acc": 0.7}, {"dataset": "Uber", "model": "Qwen3-1.7B", "method": "adversarial", "n": 100, "pred_acc": 0.55, "esc_rate": 0.68, "final_acc": 0.82, "naive_acc": 0.64}, {"dataset": "Uber", "model": "Qwen3-1.7B", "method": "base", "n": 100, "pred_acc": 0.33, "esc_rate": 0.03, "final_acc": 0.33, "naive_acc": 0.7}, {"dataset": "Uber", "model": "Qwen3-8B", "method": "adversarial", "n": 100, "pred_acc": 0.72, "esc_rate": 0.94, "final_acc": 0.98, "naive_acc": 0.72}, {"dataset": "Uber", "model": "Qwen3-14B", "method": "base", "n": 100, "pred_acc": 0.54, "esc_rate": 0.56, "final_acc": 0.68, "naive_acc": 0.74}]
''')

df = pd.DataFrame(data)

# Aggregate across ALL datasets and models: row-weighted grand mean per method
# SE comes from treating each (dataset, model) combo as an observation
metrics = ["pred_acc", "esc_rate", "final_acc"]
methods = ["base", "adversarial"]

agg = {}
for mth in methods:
    g = df[df["method"] == mth]
    agg[mth] = {}
    for metric in metrics:
        wmean = np.average(g[metric], weights=g["n"])
        # SE across (dataset, model) observations, weighted by n
        wvar = np.average((g[metric] - wmean)**2, weights=g["n"])
        k = len(g)
        se = np.sqrt(wvar / k) if k > 1 else 0
        agg[mth][metric] = (wmean, se)

# Naive accuracy (grand mean across all rows, method-independent)
naive_mean = np.average(df["naive_acc"], weights=df["n"])

# Colors
c_base = "#4878CF"
c_adv = "#D65F5F"

fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), sharey=False)
panels = [
    ("pred_acc", "Prediction Accuracy", True),
    ("esc_rate", "Escalation Rate", False),
    ("final_acc", "Final Accuracy", False),
]

x = np.array([0])
w = 0.35

for ax, (metric, title, show_naive) in zip(axes, panels):
    for mth, color, label, offset in [
        ("base", c_base, "Base", -w/2),
        ("adversarial", c_adv, "Adversarial", w/2),
    ]:
        mean, se = agg[mth][metric]
        ax.bar(x + offset, [mean], w, yerr=[se], color=color, label=label,
               capsize=3, edgecolor="white", linewidth=0.5)

    if show_naive:
        ax.axhline(naive_mean, color="black", linestyle="--", linewidth=0.8, zorder=0)
        ax.text(0.5, naive_mean + 0.02, f"Naive ({naive_mean:.2f})", ha="center",
                fontsize=7, color="black", transform=ax.get_yaxis_transform())

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(loc="upper left", framealpha=0.9, fontsize=7)
fig.tight_layout()
fig.savefig("paper/figures/fig_main.pdf", bbox_inches="tight")
print("Saved paper/figures/fig_main.pdf")
