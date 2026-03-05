"""Generate 3-panel figure for GLM-4-9B: prediction accuracy, escalation rate, final accuracy."""
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
models = ["glm-4-9b-chat-hf"]
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

metrics = ["pred_acc", "esc_rate", "final_acc"]
agg = {}
for mth in methods:
    g = data[data["method"] == mth]
    agg[mth] = {}
    for metric in metrics:
        if len(g) == 0:
            agg[mth][metric] = (0, 0)
            continue
        wmean = np.average(g[metric], weights=g["n"])
        wvar = np.average((g[metric] - wmean)**2, weights=g["n"])
        k = len(g)
        se = np.sqrt(wvar / k) if k > 1 else 0
        agg[mth][metric] = (wmean, se)

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
    ("pred_acc", "Prediction Accuracy", False),
    ("esc_rate", "Escalation Rate", False),
    ("final_acc", "Final Accuracy", False),
]

n_methods = len(methods)
w = 0.18
x = np.array([0])

for ax, (metric, title, _) in zip(axes, panels):
    offsets = np.linspace(-(n_methods-1)*w/2, (n_methods-1)*w/2, n_methods)
    for mth, offset in zip(methods, offsets):
        mean, se = agg[mth][metric]
        ax.bar(x + offset, [mean], w, yerr=[se], color=colors[mth],
               label=labels[mth], capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(loc="upper left", framealpha=0.9, fontsize=7)
fig.suptitle("GLM-4-9B", fontsize=11, y=1.02)
fig.tight_layout()
out = os.path.join(os.environ["WORK"], "madm/paper/figures/fig_main_glm.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
out_png = out.replace(".pdf", ".png")
fig.savefig(out_png, bbox_inches="tight")
print(f"Saved {out_png}")
