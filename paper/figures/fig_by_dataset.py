import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, glob

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
datasets = ["AIME", "FEVEROUS", "HotelBookings", "LendingClub",
            "MoralMachine", "MovieLens", "Uber", "WikipediaToxicity"]
methods = ["base", "agreeable", "multiagent", "adversarial"]
gt_cols = {"AIME": "Answer"}

rows = []
for ds in datasets:
    for f in glob.glob(os.path.join(results_dir, ds, "*.csv")):
        basename = os.path.basename(f)
        parts = basename.replace(".csv", "").split("_", 1)
        if len(parts) != 2:
            continue
        method, model = parts
        if method not in methods:
            continue
        df = pd.read_csv(f)
        gt = gt_cols.get(ds, "human_response")
        if gt not in df.columns or "llm_prediction" not in df.columns:
            continue
        final_acc = ((df.get("llm_escalate", 0) == 1) | (df["llm_prediction"] == df[gt])).mean()
        rows.append({"dataset": ds, "method": method, "model": model, "final_acc": final_acc})

data = pd.DataFrame(rows)

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

ds_short = {
    "AIME": "AIME",
    "FEVEROUS": "FEVER.",
    "HotelBookings": "Hotel",
    "LendingClub": "Lending",
    "MoralMachine": "Moral",
    "MovieLens": "Movie",
    "Uber": "Uber",
    "WikipediaToxicity": "Wiki.",
}

fig, ax = plt.subplots(figsize=(8, 3.5))

n_ds = len(datasets)
n_methods = len(methods)
bar_width = 0.18
x = np.arange(n_ds)

for i, method in enumerate(methods):
    means = []
    sems = []
    for ds in datasets:
        subset = data[(data["dataset"] == ds) & (data["method"] == method)]
        if len(subset) > 0:
            means.append(subset["final_acc"].mean())
            sems.append(subset["final_acc"].sem() if len(subset) > 1 else 0)
        else:
            means.append(0)
            sems.append(0)

    offset = (i - (n_methods - 1) / 2) * bar_width
    ax.bar(x + offset, means, bar_width, yerr=sems, capsize=2,
           color=colors[method], label=labels[method], edgecolor="white", linewidth=0.5)

ax.set_xlabel("Dataset")
ax.set_ylabel("Final accuracy")
ax.set_xticks(x)
ax.set_xticklabels([ds_short[ds] for ds in datasets], rotation=30, ha="right")
ax.set_ylim(0, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", framealpha=0.9)
fig.tight_layout()

out = os.path.join(os.environ["WORK"], "madm/paper/figures/fig_by_dataset.pdf")
fig.savefig(out, bbox_inches="tight")
print("Saved %s" % out)

out_png = os.path.join(os.environ["WORK"], "madm/paper/figures/fig_by_dataset.png")
fig.savefig(out_png, bbox_inches="tight")
print("Saved %s" % out_png)
