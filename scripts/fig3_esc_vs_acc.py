"""Figure 3: Escalation rate vs predictive accuracy, 7 models."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

DATA_DIR = '/Users/mdisorbo/madm_colm/results/study3'
OUT_PATH = '/Users/mdisorbo/madm_colm/paper/figures/esc_vs_acc_nothink_5models.png'

MODELS = [
    ('Qwen3.5-9B', 'Qwen3.5-9B'),
    ('GPT-5-nano', 'gpt-5-nano'),
    ('Llama4-Maverick', 'Llama-4-Maverick-17B-128E-Instruct-FP8'),
    ('Mixtral-8x7B', 'Mixtral-8x7B-Instruct-v0.1'),
    ('Gemma3-4B', 'gemma-3-4b-it'),
    ('Claude Sonnet 4.6', 'claude-sonnet-4-6'),
    ('Qwen3.5-397B', 'Qwen3.5-397B-A17B'),
    ('GPT-5-mini', 'gpt-5-mini'),
    ('Llama3.3-70B', 'Llama-3.3-70B-Instruct-Turbo'),
    ('Mistral-Small-24B', 'Mistral-Small-24B-Instruct-2501'),
    ('Gemma3-12B', 'gemma-3-12b-it'),
    ('Claude Opus 4.7', 'claude-opus-4-7'),
]

DATASETS = {
    'HotelBookings': 'orange',
    'LendingClub': 'blue',
    'WikipediaToxicity': 'red',
    'MovieLens': 'purple',
}

# Auto-detect available datasets per model (excluding MoralMachine for main figures)
ALL_DATASETS = ['HotelBookings', 'LendingClub', 'WikipediaToxicity', 'MovieLens']

fig, axes = plt.subplots(2, 6, figsize=(22, 8))
axes_flat = axes.flatten()

# Collect global ranges
all_acc, all_esc = [], []

# Pre-load data
model_data = {}
for short_name, tag in MODELS:
    frames = []
    for ds in ALL_DATASETS:
        fpath = os.path.join(DATA_DIR, f'{ds}_summary_nothink_{tag}.csv')
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            if df.empty:
                continue
            df['dataset'] = ds
            frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        model_data[tag] = combined
        all_acc.extend(combined['pred_acc'].tolist())
        all_esc.extend(combined['esc_rate'].tolist())

xmin, xmax = min(all_acc) - 0.02, max(all_acc) + 0.02
ymin, ymax = min(all_esc) - 0.02, max(all_esc) + 0.02

for i, (short_name, tag) in enumerate(MODELS):
    ax = axes_flat[i]
    df = model_data.get(tag)
    if df is None:
        ax.set_visible(False)
        continue
    for ds, color in DATASETS.items():
        sub = df[df['dataset'] == ds]
        if sub.empty:
            continue
        label = ds if i == 0 else None
        se = np.sqrt(sub['esc_rate'] * (1 - sub['esc_rate']) / sub['n'])
        ax.errorbar(sub['pred_acc'], sub['esc_rate'], yerr=se, fmt='o', c=color, ms=7, alpha=0.7, label=label, elinewidth=1.0, capsize=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(short_name, fontsize=16, fontweight='bold')
    ax.set_xlabel('Predictive accuracy', fontsize=13)
    ax.set_ylabel('Escalation rate', fontsize=13)
    ax.tick_params(labelsize=12)


handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=16, frameon=False)
plt.tight_layout(rect=[0, 0.07, 1, 1], h_pad=4.0)
fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'Saved to {OUT_PATH}')
