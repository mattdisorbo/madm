"""Generate all figures: fig_main variants, fig_by_dataset, fig_utility variants."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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
fig_base = os.path.join(os.environ["WORK"], "madm/paper/figures")
os.makedirs(os.path.join(fig_base, "main"), exist_ok=True)
os.makedirs(os.path.join(fig_base, "utility"), exist_ok=True)
os.makedirs(os.path.join(fig_base, "by_dataset"), exist_ok=True)

datasets = ["AIME", "FEVEROUS", "HotelBookings", "LendingClub",
            "MoralMachine", "MovieLens", "Uber", "WikipediaToxicity"]
methods = ["base", "agreeable", "multiagent", "adversarial"]
gt_cols = {"AIME": "Answer"}
default_gt = "human_response"

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

# ── Collect all data once ──
all_rows = []
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
        # Find escalation column
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
        pred_acc = (valid["llm_prediction"] == valid[gt]).mean()
        esc_rate = valid[esc_col].mean()
        impl = valid[valid[esc_col] == 0]
        correct = (valid["llm_prediction"] == valid[gt])
        esc = valid[esc_col] == 1
        impl_correct = ((~esc) & correct).sum()
        esc_count = esc.sum()
        final_acc = (impl_correct + esc_count) / len(valid)
        # Utility components
        correct_impl_rate = ((~esc) & correct).mean()
        incorrect_impl_rate = ((~esc) & (~correct)).mean()

        all_rows.append({
            "dataset": ds, "method": method, "model": model,
            "n": len(valid), "pred_acc": pred_acc,
            "esc_rate": esc_rate, "final_acc": final_acc,
            "correct_impl": correct_impl_rate,
            "incorrect_impl": incorrect_impl_rate,
        })

all_data = pd.DataFrame(all_rows)
print(f"Total rows: {len(all_data)}, models: {sorted(all_data['model'].unique())}")

# ── Model groups ──
groups = {
    "fig_main": {
        "models": ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B",
                    "Qwen3.5-0.8B", "Qwen3.5-4B", "Qwen3.5-9B",
                    "glm-4-9b-chat-hf",
                    "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07"],
        "title_suffix": "(all models)",
    },
    "fig_main_qwen3": {
        "models": ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B"],
        "title_suffix": "(Qwen3)",
    },
    "fig_main_qwen3.5": {
        "models": ["Qwen3.5-0.8B", "Qwen3.5-4B", "Qwen3.5-9B"],
        "title_suffix": "(Qwen3.5)",
    },
    "fig_main_glm": {
        "models": ["glm-4-9b-chat-hf"],
        "title_suffix": "(GLM-4-9B)",
    },
    "fig_main_gpt": {
        "models": ["gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07"],
        "title_suffix": "(GPT-5)",
    },
}

# ════════════════════════════════════════════════════════════════
# 1. FIG_MAIN variants
# ════════════════════════════════════════════════════════════════
print("\n=== Generating fig_main variants ===")

for fig_name, cfg in groups.items():
    model_list = cfg["models"]
    data = all_data[all_data["model"].isin(model_list)]
    if len(data) == 0:
        print(f"  {fig_name}: no data, skipping")
        continue

    print(f"  {fig_name}: {len(data)} rows, {data['model'].nunique()} models, {data['dataset'].nunique()} datasets")

    # Aggregate
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
            wvar = np.average((g[metric] - wmean) ** 2, weights=g["n"])
            k = len(g)
            se = np.sqrt(wvar / k) if k > 1 else 0
            agg[mth][metric] = (wmean, se)

    # Naive accuracy
    naive_accs = []
    for ds in datasets:
        for model in model_list:
            path = os.path.join(results_dir, ds, f"base_{model}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            gt = gt_cols.get(ds, default_gt)
            if gt not in df.columns:
                continue
            majority = df[gt].mode()[0]
            naive_accs.append((df[gt] == majority).mean())
            break
    naive_mean = np.mean(naive_accs) if naive_accs else 0.5

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=False)
    panels = [
        ("pred_acc", "Prediction Accuracy", True),
        ("esc_rate", "Escalation Rate", False),
        ("final_acc", "Final Accuracy", False),
    ]

    n_meth = len(methods)
    w = 0.18
    x = np.array([0])

    for ax, (metric, title, show_naive) in zip(axes, panels):
        offsets = np.linspace(-(n_meth - 1) * w / 2, (n_meth - 1) * w / 2, n_meth)
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

    for ext in ["pdf", "png"]:
        out = os.path.join(fig_base, "main", f"{fig_name}.{ext}")
        fig.savefig(out, bbox_inches="tight")
    print(f"    -> {os.path.join(fig_base, 'main', fig_name)}.pdf/.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# 2. FIG_BY_DATASET (all models)
# ════════════════════════════════════════════════════════════════
print("\n=== Generating fig_by_dataset ===")

ds_short = {
    "AIME": "AIME", "FEVEROUS": "FEVER.", "HotelBookings": "Hotel",
    "LendingClub": "Lending", "MoralMachine": "Moral",
    "MovieLens": "Movie", "Uber": "Uber", "WikipediaToxicity": "Wiki.",
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
        subset = all_data[(all_data["dataset"] == ds) & (all_data["method"] == method)]
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

for ext in ["pdf", "png"]:
    out = os.path.join(fig_base, "by_dataset", f"fig_by_dataset.{ext}")
    fig.savefig(out, bbox_inches="tight")
print(f"  -> saved fig_by_dataset.pdf/.png")
# Also save to top-level figures dir for backward compat
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(fig_base, f"fig_by_dataset.{ext}"), bbox_inches="tight")
plt.close(fig)


# ════════════════════════════════════════════════════════════════
# 3. FIG_UTILITY variants (overall + per model group)
# ════════════════════════════════════════════════════════════════
print("\n=== Generating fig_utility variants ===")

utility_groups = {
    "fig_utility": {
        "models": None,  # all
        "title": "All models",
    },
    "fig_utility_qwen3": {
        "models": ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B"],
        "title": "Qwen3",
    },
    "fig_utility_qwen3.5": {
        "models": ["Qwen3.5-0.8B", "Qwen3.5-4B", "Qwen3.5-9B"],
        "title": "Qwen3.5",
    },
    "fig_utility_glm": {
        "models": ["glm-4-9b-chat-hf"],
        "title": "GLM-4-9B",
    },
    "fig_utility_gpt": {
        "models": ["gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07"],
        "title": "GPT-5",
    },
}

c = np.linspace(0, 1, 200)

for fig_name, ucfg in utility_groups.items():
    model_filter = ucfg["models"]
    if model_filter is None:
        data = all_data
    else:
        data = all_data[all_data["model"].isin(model_filter)]

    if len(data) == 0:
        print(f"  {fig_name}: no data, skipping")
        continue

    stats = {}
    for method in methods:
        g = data[data["method"] == method]
        if len(g) == 0:
            continue
        stats[method] = {
            "correct_impl": g["correct_impl"].mean(),
            "incorrect_impl": g["incorrect_impl"].mean(),
            "esc": g["esc_rate"].mean(),
        }

    if not stats:
        print(f"  {fig_name}: no method data, skipping")
        continue

    print(f"  {fig_name}: {len(data)} rows, methods={list(stats.keys())}")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for method in methods:
        if method not in stats:
            continue
        s = stats[method]
        utility = s["correct_impl"] * 1 + s["incorrect_impl"] * (-1) + s["esc"] * (1 - c)
        ax.plot(c, utility, color=colors[method], label=labels[method], linewidth=1.5)

    ax.set_xlabel("Escalation cost $c$")
    ax.set_ylabel("Expected utility")
    ax.set_xlim(0, 1)
    ax.set_title(ucfg["title"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        out = os.path.join(fig_base, "utility", f"{fig_name}.{ext}")
        fig.savefig(out, bbox_inches="tight")
    # Also save to top-level for backward compat
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(fig_base, f"{fig_name}.{ext}"), bbox_inches="tight")
    print(f"    -> {fig_name}.pdf/.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# 4. Check for stage 2 results
# ════════════════════════════════════════════════════════════════
stage2_dir = os.path.join(results_dir, "stage2_patching")
if os.path.isdir(stage2_dir) and os.listdir(stage2_dir):
    print(f"\n=== Stage 2 results found in {stage2_dir} ===")
    for f in sorted(os.listdir(stage2_dir)):
        print(f"  {f}")
else:
    print("\n=== No stage 2 patching results yet ===")

print("\nDone!")
