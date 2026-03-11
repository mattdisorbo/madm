"""Study 3 visualizations."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, os

MODEL = "Qwen3.5-9B"
DATA_DIR = "results/study3"
OUTPUT_DIR = "paper/figures"

# ── Load all data ──
datasets = {}
for f in sorted(glob.glob(f'{DATA_DIR}/*_{MODEL}.csv')):
    if '_summary_' in f:
        continue
    basename = os.path.basename(f).replace(f'_{MODEL}.csv', '')
    for ds in ['WikipediaToxicity', 'MoralMachine', 'HotelBookings', 'LendingClub']:
        if basename.startswith(ds):
            subset = basename.replace(ds + '_', '')
            if ds not in datasets:
                datasets[ds] = []
            df = pd.read_csv(f)
            n = len(df)
            gt = df['ground_truth']
            base_rate = max(gt.mean(), 1 - gt.mean())
            pred_acc = (df['prediction'] == gt).mean()
            pred_se = np.sqrt(pred_acc * (1 - pred_acc) / n)
            esc_rate = df['escalate'].mean()
            esc_se = np.sqrt(esc_rate * (1 - esc_rate) / n)
            esc_correct = ((df['escalate'] == 1) & (df['correct'] == 0) |
                           (df['escalate'] == 0) & (df['correct'] == 1)).mean()
            datasets[ds].append({
                'subset': subset, 'n': n, 'base_rate': base_rate,
                'pred_acc': pred_acc, 'pred_se': pred_se,
                'esc_rate': esc_rate, 'esc_se': esc_se,
                'esc_acc': esc_correct,
            })
            break

# Sort each dataset by base_rate
for ds in datasets:
    datasets[ds] = sorted(datasets[ds], key=lambda x: x['base_rate'])

COLORS = {
    'LendingClub': '#1f77b4',
    'HotelBookings': '#ff7f0e',
    'MoralMachine': '#2ca02c',
    'WikipediaToxicity': '#d62728',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Base rate vs Predictive accuracy
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='y = x')
for ds in ['LendingClub', 'HotelBookings', 'MoralMachine', 'WikipediaToxicity']:
    rows = datasets[ds]
    x = [r['base_rate'] for r in rows]
    y = [r['pred_acc'] for r in rows]
    se = [r['pred_se'] for r in rows]
    ax.errorbar(x, y, yerr=se, marker='o', capsize=3, label=ds, color=COLORS[ds])
ax.set_xlabel('Base Rate (Hint Strength)')
ax.set_ylabel('Predictive Accuracy')
ax.set_title(f'Predictive Accuracy vs. Hint Strength ({MODEL})')
ax.legend()
ax.set_xlim(0.45, 1.02)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pred_accuracy_vs_base_rate_{MODEL}.png', dpi=150)
plt.close()
print(f'Saved pred_accuracy_vs_base_rate_{MODEL}.png')

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Base rate vs Escalation rate
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [1, 0], 'k--', alpha=0.4, label='1 - base rate')
for ds in ['LendingClub', 'HotelBookings', 'MoralMachine', 'WikipediaToxicity']:
    rows = datasets[ds]
    x = [r['base_rate'] for r in rows]
    y = [r['esc_rate'] for r in rows]
    se = [r['esc_se'] for r in rows]
    ax.errorbar(x, y, yerr=se, marker='o', capsize=3, label=ds, color=COLORS[ds])
ax.set_xlabel('Base Rate (Hint Strength)')
ax.set_ylabel('Escalation Rate')
ax.set_title(f'Escalation Rate vs. Hint Strength ({MODEL})')
ax.legend()
ax.set_xlim(0.45, 1.02)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/esc_rate_vs_base_rate_{MODEL}.png', dpi=150)
plt.close()
print(f'Saved esc_rate_vs_base_rate_{MODEL}.png')

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Implement preference region (linear fit, shaded below 50%)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

for ds in ['LendingClub', 'HotelBookings', 'MoralMachine', 'WikipediaToxicity']:
    rows = datasets[ds]
    rows_sorted = sorted(rows, key=lambda r: r['pred_acc'])
    accs = np.array([r['pred_acc'] for r in rows_sorted])
    escs = np.array([r['esc_rate'] for r in rows_sorted])

    # Linear fit
    slope, intercept = np.polyfit(accs, escs, 1)
    acc_fine = np.linspace(0.5, 1.0, 200)
    esc_fit = slope * acc_fine + intercept
    esc_fit = np.clip(esc_fit, 0, 1)

    # Plot data points and linear fit
    ax.scatter(accs, escs, color=COLORS[ds], zorder=5, s=30)
    ax.plot(acc_fine, esc_fit, color=COLORS[ds], label=f'{ds} (linear fit)',
            linestyle='--', alpha=0.8)

    # Shade region where fitted esc_rate < 0.5
    below = esc_fit < 0.5
    ax.fill_between(acc_fine, 0, esc_fit, where=below, alpha=0.12, color=COLORS[ds])

ax.axhline(0.5, color='k', linestyle=':', alpha=0.4, label='50% escalation threshold')
ax.set_xlabel('Predictive Accuracy')
ax.set_ylabel('Escalation Rate')
ax.set_title(f'Implement Preference Region ({MODEL})')
ax.legend(loc='upper right', fontsize=8)
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/implement_region_{MODEL}.png', dpi=150)
plt.close()
print(f'Saved implement_region_{MODEL}.png')
