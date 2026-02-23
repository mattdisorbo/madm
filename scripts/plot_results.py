"""
Generate per-model bar charts across datasets.

For each model, produces a figure with one facet per dataset.
X-axis: mode (base, auditor, tool=rf/ols)
Y-axis: 0–1
Bars: LLM accuracy (llm_prediction == ground_truth, all rows) and delegation rate.

Note: JFLEG is a grammar-correction task (text generation); accuracy there
requires GLEU scoring, not simple equality, so the accuracy bar is omitted.
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
    "AIME", "FEVEROUS", "JFLEG",
    "LendingClub", "MoralMachine", "MovieLens", "WikipediaToxicity",
]
MODELS = ["Qwen2.5-1.5B-Instruct", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07"]
TOOL_METHODS = {"rf", "ols"}

MODE_ORDER = ["base", "auditor", "tool"]
MODE_LABELS = {"base": "Base", "auditor": "Auditor", "tool": "Tool"}

COLORS = {"accuracy": "#4C72B0", "delegation": "#DD8452"}

TEXT_GEN_DATASETS = {"JFLEG"}  # accuracy requires GLEU, not equality


def ground_truth_col(df: pd.DataFrame) -> str:
    return "Answer" if "Answer" in df.columns else "human_response"


def compute_metrics(df: pd.DataFrame, dataset: str) -> tuple[float, float]:
    """Return (accuracy, delegation_rate) for a results dataframe.

    Rows where llm_delegate is NaN are excluded (incomplete entries).
    Accuracy is llm_prediction == ground_truth across all valid rows.
    """
    valid = df[df["llm_delegate"].notna()]
    delegation_rate = (valid["llm_delegate"] == 1).mean()

    if dataset in TEXT_GEN_DATASETS:
        accuracy = np.nan
    else:
        gt_col = ground_truth_col(df)
        pred = pd.to_numeric(valid["llm_prediction"], errors="coerce")
        truth = pd.to_numeric(valid[gt_col], errors="coerce")
        accuracy = (pred == truth).mean()

    return accuracy, delegation_rate


def load_dataset_modes(dataset: str, model: str) -> dict[str, tuple[float, float]]:
    modes: dict[str, tuple[float, float]] = {}
    dataset_dir = RESULTS_DIR / dataset
    for filepath in sorted(dataset_dir.glob(f"*_{model}.csv")):
        df = pd.read_csv(filepath)
        method = df["method"].dropna().iloc[0]
        mode = "tool" if method in TOOL_METHODS else method
        modes[mode] = compute_metrics(df, dataset)
    return modes


for model in MODELS:
    n = len(DATASETS)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.5), sharey=True)

    for ax, dataset in zip(axes, DATASETS):
        modes_data = load_dataset_modes(dataset, model)

        x = np.arange(len(MODE_ORDER))
        width = 0.35

        for i, mode in enumerate(MODE_ORDER):
            if mode not in modes_data:
                continue
            accuracy, delegation_rate = modes_data[mode]

            if not np.isnan(accuracy):
                ax.bar(i - width / 2, accuracy, width, color=COLORS["accuracy"])
            ax.bar(i + width / 2, delegation_rate, width, color=COLORS["delegation"])

        if dataset in TEXT_GEN_DATASETS:
            ax.set_title(dataset + "\n(accuracy n/a)", fontsize=9)
        else:
            ax.set_title(dataset, fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODE_LABELS[m] for m in MODE_ORDER], rotation=30, ha="right", fontsize=9
        )
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Rate", fontsize=10)

    legend_handles = [
        mpatches.Patch(color=COLORS["accuracy"], label="LLM Accuracy"),
        mpatches.Patch(color=COLORS["delegation"], label="Delegation Rate"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(model, fontsize=12, y=1.01)

    plt.tight_layout()
    out_path = VISUALS_DIR / f"{model}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
