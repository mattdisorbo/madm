"""
Generate two paper charts:
  Fig 1 - Final accuracy by model × method (base vs auditor), row-weighted
  Fig 2 - Delegation rate by model × method (base vs auditor), row-weighted

Final accuracy: delegated rows counted as correct; non-delegated rows correct
iff llm_prediction == human_response.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "gpt-5-nano-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "glm-4-9b-chat-hf",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
]
MODEL_LABELS = {
    "gpt-5-nano-2025-08-07": "GPT-5 Nano",
    "gpt-5-mini-2025-08-07": "GPT-5 Mini",
    "glm-4-9b-chat-hf":      "GLM-4-9B",
    "Qwen3-1.7B":             "Qwen3-1.7B",
    "Qwen3-4B":               "Qwen3-4B",
    "Qwen3-8B":               "Qwen3-8B",
    "Qwen3-14B":              "Qwen3-14B",
}

# Exclude AIME (math, no human labels) and JFLEG (text generation, no binary GT)
SKIP_DATASETS = {"AIME", "JFLEG", "MoralMachine"}

METHODS = ["base", "auditor"]
COLORS  = {"base": "#5B8DB8", "auditor": "#E07B39"}
LABELS  = {"base": "Base", "auditor": "Auditor"}


def ground_truth_col(df):
    return "Answer" if "Answer" in df.columns else "human_response"


def load_records():
    records = []
    for dataset_dir in sorted(RESULTS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name in SKIP_DATASETS:
            continue
        for model in MODELS:
            for method in METHODS:
                fp = dataset_dir / f"{method}_{model}.csv"
                if not fp.exists():
                    continue
                df = pd.read_csv(fp)
                df = df[df["llm_delegate"].notna()].copy()
                if df.empty:
                    continue
                gt_col = ground_truth_col(df)
                pred  = pd.to_numeric(df["llm_prediction"], errors="coerce")
                truth = pd.to_numeric(df[gt_col], errors="coerce")
                dlg   = pd.to_numeric(df["llm_delegate"], errors="coerce")

                correct   = ((dlg == 0) & (pred == truth)) | (dlg == 1)
                delegated = (dlg == 1)

                records.append({
                    "model":    model,
                    "method":   method,
                    "dataset":  dataset_dir.name,
                    "n_correct":  correct.sum(),
                    "n_delegated": delegated.sum(),
                    "n_rows":   len(df),
                })
    return pd.DataFrame(records)


def weighted_stat(grp, num_col):
    return grp[num_col].sum() / grp["n_rows"].sum()


df_all = load_records()

# ── compute row-weighted stats ─────────────────────────────────────────────
acc_pivot = (
    df_all.groupby(["model", "method"])
    .apply(lambda g: weighted_stat(g, "n_correct"), include_groups=False)
    .unstack("method")
    .reindex(index=MODELS, columns=METHODS)
)

dlg_pivot = (
    df_all.groupby(["model", "method"])
    .apply(lambda g: weighted_stat(g, "n_delegated"), include_groups=False)
    .unstack("method")
    .reindex(index=MODELS, columns=METHODS)
)

print("Final accuracy:")
print(acc_pivot.round(3))
print("\nDelegation rate:")
print(dlg_pivot.round(3))


# ── shared chart helper ────────────────────────────────────────────────────
def make_chart(pivot, ylabel, title, outfile):
    n_models = len(MODELS)
    x = np.arange(n_models)
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, method in enumerate(METHODS):
        vals   = pivot[method].values.astype(float)
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, vals, width,
                        color=COLORS[method], label=LABELS[method],
                        zorder=3, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_ylim(0, min(1.12, pivot.max().max() * 1.25 + 0.05))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {outfile}")


make_chart(
    acc_pivot,
    ylabel="Final accuracy (row-weighted)",
    title="Final Accuracy by Model and Reasoning Method",
    outfile=OUT_DIR / "fig_accuracy.pdf",
)

make_chart(
    dlg_pivot,
    ylabel="Delegation rate (row-weighted)",
    title="Delegation Rate by Model and Reasoning Method",
    outfile=OUT_DIR / "fig_delegation.pdf",
)
