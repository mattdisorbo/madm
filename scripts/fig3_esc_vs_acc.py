"""Figure 3: Escalation rate vs predictive accuracy, 7 models."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

DATA_DIR = '/Users/mdisorbo/madm/results/study3'
OUT_PATH = '/Users/mdisorbo/madm/paper/figures/esc_vs_acc_nothink_5models.png'

MODELS = [
    ('Qwen3.5-9B', 'Qwen3.5-9B'),
    ('Qwen3.5-397B', 'Qwen3.5-397B-A17B'),
    ('Qwen3-Next-3B', 'Qwen3-Next-80B-A3B-Instruct'),
    ('Llama4-Maverick', 'Llama-4-Maverick-17B-128E-Instruct-FP8'),
    ('GPT-5-mini', 'gpt-5-mini'),
    ('GPT-5-nano', 'gpt-5-nano'),
    ('Llama3.3-70B', 'Llama-3.3-70B-Instruct-Turbo'),
]

DATASETS = {
    'HotelBookings': 'orange',
    'LendingClub': 'blue',
    'WikipediaToxicity': 'red',
    'MovieLens': 'purple',
}

# Which datasets each model has (hint data)
MODEL_DATASETS = {
    'Qwen3.5-9B': ['HotelBookings', 'LendingClub', 'WikipediaToxicity', 'MovieLens'],
    'Qwen3.5-397B-A17B': ['HotelBookings', 'LendingClub', 'WikipediaToxicity', 'MovieLens'],
    'Qwen3-Next-80B-A3B-Instruct': ['HotelBookings', 'LendingClub', 'WikipediaToxicity', 'MovieLens'],
    'Llama-4-Maverick-17B-128E-Instruct-FP8': ['HotelBookings', 'LendingClub', 'WikipediaToxicity', 'MovieLens'],
    'gpt-5-mini': ['HotelBookings', 'LendingClub', 'WikipediaToxicity'],
    'gpt-5-nano': ['HotelBookings', 'LendingClub', 'WikipediaToxicity', 'MovieLens'],
    'Llama-3.3-70B-Instruct-Turbo': ['HotelBookings', 'LendingClub', 'WikipediaToxicity', 'MovieLens'],
}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes_flat = axes.flatten()

# Collect global ranges
all_acc, all_esc = [], []

# Pre-load data
model_data = {}
for short_name, tag in MODELS:
    frames = []
    for ds in MODEL_DATASETS[tag]:
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
        ax.scatter(sub['pred_acc'], sub['esc_rate'], c=color, s=30, alpha=0.7, label=label, edgecolors='none')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(short_name, fontsize=12)
    ax.set_xlabel('Predictive accuracy', fontsize=10)
    ax.set_ylabel('Escalation rate', fontsize=10)
    if i == 0:
        ax.legend(fontsize=8, loc='best')

# Hide the 8th panel
axes_flat[7].set_visible(False)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'Saved to {OUT_PATH}')
