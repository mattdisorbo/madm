"""Final accuracy gap (adversarial - base) per model with SE bars."""
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
gt_cols = {"AIME": "Answer"}
default_gt = "human_response"

# Collect per-dataset final_acc for base and adversarial
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
        if method not in ("base", "adversarial"):
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
        final_acc = ((~esc & correct).sum() + esc.sum()) / len(valid)
        rows.append({"dataset": ds, "method": method, "model": model, "final_acc": final_acc})

data = pd.DataFrame(rows)

# For each model, compute gap per dataset, then mean and SE across datasets
models = sorted(data["model"].unique())
gaps = []
for model in models:
    base = data[(data["model"] == model) & (data["method"] == "base")]
    adv = data[(data["model"] == model) & (data["method"] == "adversarial")]
    if len(base) == 0 or len(adv) == 0:
        continue
    # Match on datasets that have both
    shared_ds = set(base["dataset"]) & set(adv["dataset"])
    if len(shared_ds) == 0:
        continue
    ds_gaps = []
    for ds in shared_ds:
        b = base[base["dataset"] == ds]["final_acc"].values[0]
        a = adv[adv["dataset"] == ds]["final_acc"].values[0]
        ds_gaps.append(a - b)
    mean_gap = np.mean(ds_gaps)
    se_gap = np.std(ds_gaps, ddof=1) / np.sqrt(len(ds_gaps)) if len(ds_gaps) > 1 else 0
    gaps.append({"model": model, "gap": mean_gap, "se": se_gap, "n_ds": len(ds_gaps)})

gap_df = pd.DataFrame(gaps).sort_values("gap", ascending=True)
print(gap_df.to_string(index=False))

# Shorter display names
short = {
    "Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen3-4B": "Qwen3-4B",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-14B": "Qwen3-14B",
    "Qwen3.5-0.8B": "Qwen3.5-0.8B",
    "Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen3.5-9B": "Qwen3.5-9B",
    "glm-4-9b-chat-hf": "GLM-4-9B",
    "gpt-5-mini-2025-08-07": "GPT-5-mini",
    "gpt-5-nano-2025-08-07": "GPT-5-nano",
    "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "DeepSeek-R1-Distill-Qwen-7B": "DS-R1-7B",
    "Qwen3.5-35B-A3B": "Qwen3.5-35B-MoE",
}

fig, ax = plt.subplots(figsize=(6, 4))
y = np.arange(len(gap_df))
colors = ["#D65F5F" if g > 0 else "#4878CF" for g in gap_df["gap"]]
ax.barh(y, gap_df["gap"], xerr=gap_df["se"], capsize=3,
        color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels([short.get(m, m) for m in gap_df["model"]])
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Final accuracy gap (adversarial $-$ base)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

for ext in ["pdf", "png"]:
    out = os.path.join(fig_base, f"fig_gap.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
plt.close(fig)
