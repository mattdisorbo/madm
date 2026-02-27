"""
Visual 1: CoT vs Base vs Auditor final accuracy for gpt-5-nano across all nine datasets.
Visual 2: Fine-tuned (ft:gpt-4o-mini) vs Base gpt-5-nano prediction accuracy
          for HotelBookings and LendingClub.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path("results")
VISUALS_DIR = Path("visuals")
VISUALS_DIR.mkdir(exist_ok=True)

DATASETS = [
    "AIME", "FEVEROUS", "HotelBookings", "JFLEG",
    "LendingClub", "MoralMachine", "MovieLens", "Uber", "WikipediaToxicity",
]
MODEL = "gpt-5-nano-2025-08-07"
TEXT_GEN_DATASETS = {"JFLEG"}

# Friendly dataset labels
DATASET_LABELS = {
    "AIME": "AIME",
    "FEVEROUS": "FEVEROUS",
    "HotelBookings": "Hotel\nBookings",
    "JFLEG": "JFLEG",
    "LendingClub": "Lending\nClub",
    "MoralMachine": "Moral\nMachine",
    "MovieLens": "Movie\nLens",
    "Uber": "Uber",
    "WikipediaToxicity": "Wikipedia\nToxicity",
}

COLORS = {
    "base":    "#4C72B0",
    "cot":     "#55A868",
    "auditor": "#C44E52",
    "ft":      "#DD8452",
}


def ground_truth_col(df: pd.DataFrame) -> str:
    return "Answer" if "Answer" in df.columns else "human_response"


def compute_accuracy(df: pd.DataFrame, dataset: str) -> float:
    if dataset in TEXT_GEN_DATASETS:
        return np.nan
    valid = df[df["llm_delegate"].notna()].copy()
    if valid.empty:
        return np.nan
    gt_col = ground_truth_col(df)
    pred = pd.to_numeric(valid["llm_prediction"], errors="coerce")
    truth = pd.to_numeric(valid[gt_col], errors="coerce")
    correct_implemented = (valid["llm_delegate"] == 0) & (pred == truth)
    delegated = valid["llm_delegate"] == 1
    return (correct_implemented | delegated).mean()


def try_load(path: Path, dataset: str):
    try:
        df = pd.read_csv(path)
        return compute_accuracy(df, dataset)
    except FileNotFoundError:
        return None


# ── Visual 1: CoT vs Base vs Auditor ─────────────────────────────────────────

bar_labels = ["Base", "CoT", "Auditor"]
bar_keys   = ["base", "cot", "auditor"]
x = np.arange(len(DATASETS))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 5))

for j, (key, label) in enumerate(zip(bar_keys, bar_labels)):
    accs = []
    for d in DATASETS:
        path = RESULTS_DIR / d / f"{key}_{MODEL}.csv"
        accs.append(try_load(path, d))

    for i, acc in enumerate(accs):
        xpos = i + (j - 1) * width
        if acc is None:
            # Dataset missing for this method — skip bar
            continue
        if np.isnan(acc):
            # Text-gen dataset — draw a hatched placeholder
            ax.bar(xpos, 0.05, width * 0.9, color=COLORS[key], alpha=0.25,
                   hatch="////", edgecolor=COLORS[key])
            continue
        bar = ax.bar(xpos, acc, width * 0.9, color=COLORS[key], label=label)
        ax.text(xpos, acc + 0.012, f"{acc:.2f}", ha="center", va="bottom",
                fontsize=7, color=COLORS[key])

ax.set_xticks(x)
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Final Accuracy", fontsize=11)
ax.set_title("CoT vs Base vs Auditor — gpt-5-nano Final Accuracy Across Datasets", fontsize=12)
ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
ax.tick_params(axis="x", length=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_handles = [
    mpatches.Patch(color=COLORS[k], label=l)
    for k, l in zip(bar_keys, bar_labels)
]
legend_handles.append(
    mpatches.Patch(color="grey", alpha=0.4, hatch="////", label="N/A (text-gen)")
)
ax.legend(handles=legend_handles, frameon=False, fontsize=9, loc="upper right")

# Annotate missing bars
for i, d in enumerate(DATASETS):
    has_base = (RESULTS_DIR / d / f"base_{MODEL}.csv").exists()
    has_auditor = (RESULTS_DIR / d / f"auditor_{MODEL}.csv").exists()
    if not has_base or not has_auditor:
        ax.text(i, -0.08, "base/auditor\nnot available", ha="center",
                fontsize=6.5, color="grey", style="italic")

plt.tight_layout()
out1 = VISUALS_DIR / "cot_vs_base_vs_auditor_gpt_nano.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out1}")


# ── Visual 2: Fine-tuned vs Base gpt-nano ─────────────────────────────────────

FT_DATASETS = ["HotelBookings", "LendingClub"]
FT_MODEL_IDS = {
    "HotelBookings": "ft:gpt-4o-mini-2024-07-18:mit-ide:hotelbookings:DDBQuoVq",
    "LendingClub":   "ft:gpt-4o-mini-2024-07-18:mit-ide:lendingclub:DDBReFWb",
}

fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

for ax, dataset in zip(axes, FT_DATASETS):
    ft_model    = FT_MODEL_IDS[dataset]
    ft_base_acc = try_load(RESULTS_DIR / dataset / f"base_{ft_model}.csv", dataset)
    base_acc    = try_load(RESULTS_DIR / dataset / f"base_{MODEL}.csv", dataset)
    aud_acc     = try_load(RESULTS_DIR / dataset / f"auditor_{MODEL}.csv", dataset)

    entries = [
        (ft_base_acc, COLORS["ft"],      "Fine-tuned Base\n(gpt-4o-mini)"),
        (base_acc,    COLORS["base"],    "Base\n(gpt-5-nano)"),
        (aud_acc,     COLORS["auditor"], "Auditor\n(gpt-5-nano)"),
    ]

    bar_width = 0.45
    x_pos = np.arange(len(entries))
    for xi, (acc, color, _) in enumerate(entries):
        val = acc if acc is not None else np.nan
        if np.isnan(val):
            ax.bar(xi, 0.05, bar_width, color=color, alpha=0.2, hatch="////",
                   edgecolor=color)
            ax.text(xi, 0.08, "N/A", ha="center", va="bottom",
                    fontsize=9, color="grey", style="italic")
        else:
            ax.bar(xi, val, bar_width, color=color)
            ax.text(xi, val + 0.015, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([e[2] for e in entries], fontsize=9)
    ax.set_title(dataset, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Final Accuracy", fontsize=11)

legend_handles = [
    mpatches.Patch(color=COLORS["ft"],      label="Fine-tuned Base (gpt-4o-mini)"),
    mpatches.Patch(color=COLORS["base"],    label="Base (gpt-5-nano)"),
    mpatches.Patch(color=COLORS["auditor"], label="Auditor (gpt-5-nano)"),
]
fig.legend(handles=legend_handles, frameon=False, fontsize=9,
           loc="upper right", bbox_to_anchor=(1.02, 1.0))

fig.suptitle("Fine-tuned Base vs Base vs Auditor — Final Accuracy\n(HotelBookings & LendingClub)",
             fontsize=12, y=1.02)
plt.tight_layout()
out2 = VISUALS_DIR / "finetuned_vs_base_gpt_nano.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")
