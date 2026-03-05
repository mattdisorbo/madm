"""Calibration: accuracy of implemented vs escalated decisions per method."""
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
fig_base = os.path.join(os.environ["WORK"], "madm/paper/figures")

datasets = ["AIME", "FEVEROUS", "HotelBookings", "LendingClub",
            "MoralMachine", "MovieLens", "Uber", "WikipediaToxicity"]
methods = ["base", "agreeable", "multiagent", "adversarial"]
gt_cols = {"AIME": "Answer"}
default_gt = "human_response"

# Collect per-file stats
rows = []
for ds in datasets:
    ds_dir = os.path.join(results_dir, ds)
    if not os.path.isdir(ds_dir):
        continue
    for fname in os.listdir(ds_dir):
        if not fname.endswith(".csv"):
            continue
        parts = fname.replace(".csv", "").split("_", 1)
        if len(parts) != 2:
            continue
        method, model = parts
        if method not in methods:
            continue
        path = os.path.join(ds_dir, fname)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        gt = gt_cols.get(ds, default_gt)
        if gt not in df.columns or "llm_prediction" not in df.columns:
            continue
        esc_col = None
        for c in ["llm_escalate", "llm_delegate"]:
            if c in df.columns:
                esc_col = c
                break
        if esc_col is None:
            continue
        valid = df.dropna(subset=["llm_prediction", esc_col, gt])
        if len(valid) == 0:
            continue
        correct = valid["llm_prediction"] == valid[gt]
        esc = valid[esc_col] == 1
        impl = ~esc

        n_impl = impl.sum()
        n_esc = esc.sum()
        acc_impl = correct[impl].mean() if n_impl > 0 else np.nan
        acc_esc = correct[esc].mean() if n_esc > 0 else np.nan

        rows.append({
            "dataset": ds, "method": method, "model": model,
            "acc_impl": acc_impl, "acc_esc": acc_esc,
            "n_impl": n_impl, "n_esc": n_esc,
        })

data = pd.DataFrame(rows)

# Aggregate per method
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

fig, ax = plt.subplots(figsize=(6, 3.5))
n_meth = len(methods)
w = 0.35
x = np.arange(n_meth)

impl_means, impl_ses = [], []
esc_means, esc_ses = [], []

for method in methods:
    g = data[data["method"] == method].dropna(subset=["acc_impl", "acc_esc"])
    impl_means.append(g["acc_impl"].mean())
    impl_ses.append(g["acc_impl"].sem() if len(g) > 1 else 0)
    esc_means.append(g["acc_esc"].mean())
    esc_ses.append(g["acc_esc"].sem() if len(g) > 1 else 0)
    print(f"{method}: impl_acc={g['acc_impl'].mean():.3f} ({len(g)} files), "
          f"esc_acc={g['acc_esc'].mean():.3f}, "
          f"gap={g['acc_impl'].mean() - g['acc_esc'].mean():.3f}")

ax.bar(x - w/2, impl_means, w, yerr=impl_ses, capsize=3,
       color="#4878CF", label="Implemented", edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, esc_means, w, yerr=esc_ses, capsize=3,
       color="#D65F5F", label="Escalated", edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([labels[m] for m in methods])
ax.set_ylabel("Prediction accuracy")
ax.set_ylim(0, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", framealpha=0.9)
fig.tight_layout()

for ext in ["pdf", "png"]:
    out = os.path.join(fig_base, f"fig_calibration.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
plt.close(fig)
