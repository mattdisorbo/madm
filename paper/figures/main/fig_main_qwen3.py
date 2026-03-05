"""Generate the main 3-panel figure: prediction accuracy, escalation rate, final accuracy.

Aggregated across all datasets and all Qwen3 models into a single bar per method.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

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

results_dir = os.path.join(os.environ["WORK"], "madm/results")
datasets = ["AIME","FEVEROUS","HotelBookings","LendingClub",
            "MoralMachine","MovieLens","Uber","WikipediaToxicity"]
methods = ["base", "agreeable", "multiagent", "adversarial"]
models = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]
gt_cols = {"AIME": "Answer"}
default_gt = "human_response"

rows = []
for ds in datasets:
    for method in methods:
        for model in models:
            path = os.path.join(results_dir, ds, f"{method}_{model}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            gt = gt_cols.get(ds, default_gt)
            if gt not in df.columns or "llm_prediction" not in df.columns:
                continue
            n = len(df)
            pred_acc = (df["llm_prediction"] == df[gt]).mean()
            esc_rate = df["llm_escalate"].mean() if "llm_escalate" in df.columns else 0.0
            final_acc = ((df["llm_escalate"] == 1) | (df["llm_prediction"] == df[gt])).mean()
            rows.append({"dataset": ds, "method": method, "model": model, "n": n,
                         "pred_acc": pred_acc, "esc_rate": esc_rate, "final_acc": final_acc})

data = pd.DataFrame(rows)

# Compute row-weighted mean and SE per method
metrics = ["pred_acc", "esc_rate", "final_acc"]
agg = {}
for mth in methods:
    g = data[data["method"] == mth]
    agg[mth] = {}
    for metric in metrics:
        wmean = np.average(g[metric], weights=g["n"])
        wvar = np.average((g[metric] - wmean)**2, weights=g["n"])
        k = len(g)
        se = np.sqrt(wvar / k) if k > 1 else 0
        agg[mth][metric] = (wmean, se)

# Naive accuracy: majority-class baseline per dataset
naive_accs = []
for ds in datasets:
    for model in models:
        path = os.path.join(results_dir, ds, f"base_{model}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        gt = gt_cols.get(ds, default_gt)
        if gt not in df.columns:
            continue
        majority = df[gt].mode()[0]
        naive_accs.append((df[gt] == majority).mean())
        break  # one per dataset is enough
naive_mean = np.mean(naive_accs) if naive_accs else 0.5

# Colors
colors = {
    "base": "#4878CF",
    "agreeable": "#6ACC65",
    "multiagent": "#B47CC7",
    "adversarial": "#D65F5F",
}
labels = {
    "base": "Base",
    "agreeable": "Agreeable",
    "multiagent": "Multiagent",
    "adversarial": "Adversarial",
}

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=False)
panels = [
    ("pred_acc", "Prediction Accuracy", True),
    ("esc_rate", "Escalation Rate", False),
    ("final_acc", "Final Accuracy", False),
]

n_methods = len(methods)
w = 0.18
x = np.array([0])

for ax, (metric, title, show_naive) in zip(axes, panels):
    offsets = np.linspace(-(n_methods-1)*w/2, (n_methods-1)*w/2, n_methods)
    for mth, offset in zip(methods, offsets):
        mean, se = agg[mth][metric]
        ax.bar(x + offset, [mean], w, yerr=[se], color=colors[mth],
               label=labels[mth], capsize=3, edgecolor="white", linewidth=0.5)

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
out = os.path.join(os.environ["WORK"], "madm/paper/figures/fig_main_qwen3.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
