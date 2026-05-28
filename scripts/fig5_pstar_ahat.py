"""Figure 5 (appendix): p* and ahat bar chart for all 8 models."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import LinearRegression

DATA_DIR = 'results/study3'
OUT_PATH = 'paper/figures/pstar_ahat_bars.png'

ALL_DATASETS = ['HotelBookings', 'LendingClub', 'WikipediaToxicity', 'MovieLens']

models = [
    ('Qwen3.5-4B', 'Qwen3.5-4B'),
    ('Qwen3.5-9B', 'Qwen3.5-9B'),
    ('Qwen3.5-397B', 'Qwen3.5-397B-A17B'),
    ('GPT-5-nano', 'gpt-5-nano'),
    ('GPT-5-mini', 'gpt-5-mini'),
    ('Llama4-Maverick', 'Llama-4-Maverick-17B-128E-Instruct-FP8'),
    ('Llama3.3-70B', 'Llama-3.3-70B-Instruct-Turbo'),
    ('Mixtral-8x7B', 'Mixtral-8x7B-Instruct-v0.1'),
    ('Mistral-Small-24B', 'Mistral-Small-24B-Instruct-2501'),
    ('Gemma3-4B', 'gemma-3-4b-it'),
    ('Gemma3-12B', 'gemma-3-12b-it'),
    ('Claude Sonnet 4.6', 'claude-sonnet-4-6'),
    ('Claude Opus 4.7', 'claude-opus-4-7'),
]

results = []
for name, tag in models:
    hint_frames = []
    for ds in ALL_DATASETS:
        try:
            df = pd.read_csv(f'{DATA_DIR}/{ds}_summary_nothink_{tag}.csv')
            if not df.empty:
                df['dataset'] = ds
                hint_frames.append(df)
        except: pass
    if not hint_frames: continue
    hint_all = pd.concat(hint_frames)
    X = hint_all['pred_acc'].values.reshape(-1,1)
    y = hint_all['esc_rate'].values
    reg = LinearRegression().fit(X, y)
    slope, intercept = reg.coef_[0], reg.intercept_
    pstar = (0.5 - intercept) / slope if slope != 0 else None

    nohint_frames = []
    for ds in ALL_DATASETS:
        try:
            df = pd.read_csv(f'{DATA_DIR}/{ds}_summary_nothink_nohint_{tag}.csv')
            if not df.empty: nohint_frames.append(df)
        except: pass
    ahat = None
    if nohint_frames:
        nohint_all = pd.concat(nohint_frames)
        ahat = float(np.clip((nohint_all['esc_rate'].mean() - intercept) / slope, 0, 1))
    actual = hint_all['pred_acc'].mean()
    results.append({'name': name, 'pstar': pstar, 'ahat': ahat, 'actual': actual})

df = pd.DataFrame(results)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

colors = ['#4C72B0', '#4C72B0', '#4C72B0', '#DD8452', '#DD8452', '#55A868', '#55A868', '#C44E52', '#C44E52', '#8172B2', '#8172B2', '#937860', '#937860']
bars1 = ax1.barh(df['name'], df['pstar'].clip(upper=1.2), color=colors, alpha=0.8)
ax1.axvline(x=0.75, color='gray', linestyle=':', linewidth=1, label='$\\tau^*$ at $R=4$')
ax1.set_xlabel('Implicit threshold $p^*$')
ax1.set_xlim(0, 1.2)
ax1.invert_yaxis()
ax1.legend(fontsize=9)

bars2 = ax2.barh(df['name'], df['ahat'], color=colors, alpha=0.8, label='Self-estimated')
ax2.axvline(x=df['actual'].mean(), color='gray', linestyle=':', linewidth=1, label='Avg actual accuracy')
ax2.set_xlabel('Self-estimated accuracy $\\hat{a}$')
ax2.set_xlim(0.5, 1.0)
ax2.invert_yaxis()
ax2.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'Saved to {OUT_PATH}')
