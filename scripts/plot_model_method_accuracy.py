"""
Cross-model accuracy heatmap/bar chart: model x method final accuracy,
averaged across datasets where data exists for that (model, method) pair.

Final accuracy definition: delegated rows = correct (human assumed perfect);
implemented rows = correct iff llm_prediction == ground_truth.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path("results")
VISUALS_DIR = Path("visuals")

DATASETS = [
    "FEVEROUS", "HotelBookings", "JFLEG",
    "LendingClub", "MovieLens", "Uber", "WikipediaToxicity",
]
MODELS = [
    "gpt-5-nano-2025-08-07", "Qwen3-1.7B",
    "gpt-5-mini-2025-08-07", "glm-4-9b-chat-hf",
    "Qwen3-4B", "Qwen3-8B", "Qwen3-14B",
]
TOOL_METHODS = {"rf", "ols", "glm"}
TEXT_GEN_DATASETS = {"JFLEG"}
MODE_ORDER = ["base", "auditor", "tool"]
MODE_LABELS = {"base": "Base", "auditor": "Auditor", "tool": "Tool"}
COLORS = {"base": "#5B8DB8", "auditor": "#E07B39", "tool": "#5FAD56"}


def ground_truth_col(df):
    return "Answer" if "Answer" in df.columns else "human_response"


def compute_counts(df, dataset):
    """Return (n_correct, n_rows) for row-weighted aggregation. Returns (nan, 0) for text-gen."""
    if dataset in TEXT_GEN_DATASETS:
        return np.nan, 0
    valid = df[df["llm_delegate"].notna()].copy()
    if valid.empty:
        return np.nan, 0
    gt_col = ground_truth_col(df)
    pred = pd.to_numeric(valid["llm_prediction"], errors="coerce")
    truth = pd.to_numeric(valid[gt_col], errors="coerce")
    correct_implemented = (valid["llm_delegate"] == 0) & (pred == truth)
    delegated = valid["llm_delegate"] == 1
    n_correct = (correct_implemented | delegated).sum()
    return n_correct, len(valid)


# Build records: (model, mode, dataset, n_correct, n_rows)
records = []
for model in MODELS:
    for dataset in DATASETS:
        dataset_dir = RESULTS_DIR / dataset
        if not dataset_dir.exists():
            continue
        for filepath in sorted(dataset_dir.glob(f"*_{model}.csv")):
            df = pd.read_csv(filepath)
            method = df["method"].dropna().iloc[0]
            mode = "tool" if method in TOOL_METHODS else method
            n_correct, n_rows = compute_counts(df, dataset)
            if n_rows > 0 and not np.isnan(n_correct):
                records.append({"model": model, "mode": mode, "dataset": dataset,
                                "n_correct": n_correct, "n_rows": n_rows})

df_all = pd.DataFrame(records)

# Row-weighted accuracy: sum(correct) / sum(rows) per (model, mode)
def weighted_accuracy(grp):
    return grp["n_correct"].sum() / grp["n_rows"].sum()

pivot = df_all.groupby(["model", "mode"]).apply(weighted_accuracy).unstack("mode")
pivot = pivot.reindex(index=MODELS, columns=MODE_ORDER)

n_models = len(pivot)
n_modes = len(MODE_ORDER)
x = np.arange(n_models)
width = 0.22

fig, ax = plt.subplots(figsize=(13, 5))
for i, mode in enumerate(MODE_ORDER):
    if mode not in pivot.columns:
        continue
    vals = pivot[mode].values.astype(float)
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, color=COLORS[mode],
                  label=MODE_LABELS[mode], zorder=3)
    # label bars that actually have data
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels([m.replace("-Instruct", "") for m in MODELS], rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Final Accuracy (row-weighted across available datasets)", fontsize=10)
ax.set_title("Final Accuracy by Model × Method\n(row-weighted, excl. AIME & MoralMachine)", fontsize=11)
ax.legend(fontsize=10, frameon=False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
out1 = VISUALS_DIR / "model_x_method_accuracy.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out1}")

# ── Figure 2: auditor gain = auditor_acc - base_acc, per model ────────────────
gain = pivot["auditor"] - pivot["base"]
gain = gain.dropna()

fig2, ax2 = plt.subplots(figsize=(9, 4))
bar_colors = ["#E07B39" if v >= 0 else "#C0392B" for v in gain.values]
bars = ax2.bar(range(len(gain)), gain.values, color=bar_colors, zorder=3)
for bar, v in zip(bars, gain.values):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + (0.005 if v >= 0 else -0.02),
             f"{v:+.3f}", ha="center", va="bottom", fontsize=8)

ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xticks(range(len(gain)))
ax2.set_xticklabels([m.replace("-Instruct", "") for m in gain.index], rotation=30, ha="right", fontsize=9)
ax2.set_ylabel("Auditor − Base Accuracy", fontsize=10)
ax2.set_title("Auditor Gain over Base (row-weighted, excl. AIME & MoralMachine)", fontsize=11)
ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax2.set_axisbelow(True)
plt.tight_layout()
out2 = VISUALS_DIR / "auditor_gain.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")

# ── Figure 3: per-dataset heatmap of auditor gain ─────────────────────────────
pivot_ds = df_all.groupby(["model", "mode", "dataset"]).apply(weighted_accuracy).unstack("mode")
auditor_gain_ds = (pivot_ds["auditor"] - pivot_ds["base"]).unstack("dataset")
auditor_gain_ds = auditor_gain_ds.reindex(index=MODELS)

fig3, ax3 = plt.subplots(figsize=(12, 5))
cmap = plt.cm.RdYlGn
im = ax3.imshow(auditor_gain_ds.values.astype(float), aspect="auto", cmap=cmap, vmin=-0.3, vmax=0.3)
ax3.set_xticks(range(len(auditor_gain_ds.columns)))
ax3.set_xticklabels(auditor_gain_ds.columns, rotation=30, ha="right", fontsize=9)
ax3.set_yticks(range(len(MODELS)))
ax3.set_yticklabels([m.replace("-Instruct", "") for m in MODELS], fontsize=9)
ax3.set_title("Auditor Gain (Auditor − Base) per Model × Dataset", fontsize=11)
plt.colorbar(im, ax=ax3, label="Accuracy gain")
# annotate cells
for i in range(len(MODELS)):
    for j in range(len(auditor_gain_ds.columns)):
        v = auditor_gain_ds.values[i, j]
        if not np.isnan(v):
            ax3.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                     color="black")
        else:
            ax3.text(j, i, "—", ha="center", va="center", fontsize=8, color="#aaa")
plt.tight_layout()
out3 = VISUALS_DIR / "auditor_gain_heatmap.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out3}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n── Mean accuracy by model × method ──")
print(pivot.round(3).to_string())
print("\n── Auditor gain (auditor − base) ──")
print(gain.round(3).to_string())
