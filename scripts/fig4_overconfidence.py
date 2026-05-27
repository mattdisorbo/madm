"""Figure 4: Overconfidence scatter – actual vs self-estimated accuracy."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import LinearRegression

DATA_DIR = '/Users/mdisorbo/madm_colm/results/study3'
OUT_PATH = '/Users/mdisorbo/madm_colm/paper/figures/overconfidence_by_condition.png'

MODELS = [
    ('Qwen3.5-9B', 'Qwen3.5-9B'),
    ('Gemma 3 4B', 'gemma-3-4b-it'),
    ('Claude Sonnet 4.6', 'claude-sonnet-4-6'),
    ('Qwen3.5-397B', 'Qwen3.5-397B-A17B'),
    ('Gemma 3 12B', 'gemma-3-12b-it'),
    ('Claude Opus 4.7', 'claude-opus-4-7'),
]

DATASETS = {
    'HotelBookings': 'orange',
    'LendingClub': 'blue',
    'WikipediaToxicity': 'red',
    'MovieLens': 'purple',
}

ALL_DATASETS = ['HotelBookings', 'LendingClub', 'WikipediaToxicity', 'MovieLens']

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes_flat = axes.flatten()

all_vals = []

# For each model: fit linear regression on hint data (pred_acc -> esc_rate),
# then invert to get self_estimated_acc from nohint esc_rate
model_results = {}
for short_name, tag in MODELS:
    # Load all hint data for this model
    hint_frames = []
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
            hint_frames.append(df)
    if not hint_frames:
        continue
    hint_all = pd.concat(hint_frames, ignore_index=True)

    # Fit linear regression: esc_rate = slope * pred_acc + intercept
    X = hint_all['pred_acc'].values.reshape(-1, 1)
    y = hint_all['esc_rate'].values
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    # For each dataset, load nohint data and compute self-estimated accuracy
    results = []
    for ds in ALL_DATASETS:
        nohint_path = os.path.join(DATA_DIR, f'{ds}_summary_nothink_nohint_{tag}.csv')
        hint_path = os.path.join(DATA_DIR, f'{ds}_summary_nothink_{tag}.csv')
        if not os.path.exists(nohint_path) or not os.path.exists(hint_path):
            continue
        try:
            nohint_df = pd.read_csv(nohint_path)
            hint_df = pd.read_csv(hint_path)
        except Exception:
            continue
        if nohint_df.empty or hint_df.empty:
            continue

        # Match conditions
        for _, row in nohint_df.iterrows():
            cond = row['condition']
            nohint_esc = row['esc_rate']
            # Invert: self_estimated_acc = (nohint_esc - intercept) / slope
            if slope != 0:
                self_est = np.clip((nohint_esc - intercept) / slope, 0, 1)
            else:
                self_est = np.nan
            # Get actual accuracy from hint data
            hint_row = hint_df[hint_df['condition'] == cond]
            if hint_row.empty:
                continue
            actual_acc = hint_row['pred_acc'].values[0]
            results.append({
                'dataset': ds,
                'condition': cond,
                'actual_acc': actual_acc,
                'self_est_acc': self_est,
            })
    if results:
        model_results[tag] = pd.DataFrame(results)
        all_vals.extend([r['actual_acc'] for r in results])
        all_vals.extend([r['self_est_acc'] for r in results])

vmin = 0.4
vmax = 1.0

for i, (short_name, tag) in enumerate(MODELS):
    ax = axes_flat[i]
    df = model_results.get(tag)
    if df is None:
        ax.set_visible(False)
        continue
    # Diagonal line
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, linewidth=1)
    for ds, color in DATASETS.items():
        sub = df[df['dataset'] == ds]
        if sub.empty:
            continue
        label = ds if i == 0 else None
        ax.scatter(sub['actual_acc'], sub['self_est_acc'], c=color, s=50, alpha=0.7,
                   label=label, edgecolors='none')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(short_name, fontsize=16, fontweight='bold')
    ax.set_xlabel('Actual accuracy', fontsize=13)
    ax.set_ylabel('Self-estimated accuracy', fontsize=13)
    ax.tick_params(labelsize=12)


handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=16, frameon=False)
plt.tight_layout(rect=[0, 0.07, 1, 1], h_pad=4.0)
fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'Saved to {OUT_PATH}')
