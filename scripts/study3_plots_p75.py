"""Study 3 visualizations — cost4 data, 75% escalation threshold."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, os

MODEL = "Qwen3.5-9B"
DATA_DIR = "results/study3"
OUTPUT_DIR = "paper/figures"
THRESHOLD = 0.75

# ── Load cost4 data only ──
datasets = {}
for f in sorted(glob.glob(f'{DATA_DIR}/*_cost4_{MODEL}.csv')):
    if '_summary_' in f:
        continue
    basename = os.path.basename(f).replace(f'_cost4_{MODEL}.csv', '')
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
            datasets[ds].append({
                'subset': subset, 'n': n, 'base_rate': base_rate,
                'pred_acc': pred_acc, 'pred_se': pred_se,
                'esc_rate': esc_rate, 'esc_se': esc_se,
            })
            break

for ds in datasets:
    datasets[ds] = sorted(datasets[ds], key=lambda x: x['base_rate'])

COLORS = {
    'LendingClub': '#1f77b4',
    'HotelBookings': '#ff7f0e',
    'MoralMachine': '#2ca02c',
    'WikipediaToxicity': '#d62728',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_pstar(rows, threshold):
    base_rates = np.array([r['base_rate'] for r in rows])
    esc_rates = np.array([r['esc_rate'] for r in rows])
    for i in range(len(esc_rates) - 1):
        if (esc_rates[i] - threshold) * (esc_rates[i+1] - threshold) <= 0:
            t = (threshold - esc_rates[i]) / (esc_rates[i+1] - esc_rates[i])
            return base_rates[i] + t * (base_rates[i+1] - base_rates[i])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Predictive Accuracy vs. Hint Strength
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='y = x')
for ds in ['LendingClub', 'HotelBookings', 'MoralMachine', 'WikipediaToxicity']:
    if ds not in datasets:
        continue
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
plt.savefig(f'{OUTPUT_DIR}/pred_accuracy_vs_base_rate_{MODEL}_p75.png', dpi=150)
plt.close()
print(f'Saved pred_accuracy_vs_base_rate_{MODEL}_p75.png')

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Escalation Rate vs. Hint Strength (75% threshold)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [1, 0], 'k--', alpha=0.4, label='1 - base rate')
for ds in ['LendingClub', 'HotelBookings', 'MoralMachine', 'WikipediaToxicity']:
    if ds not in datasets:
        continue
    rows = datasets[ds]
    x = [r['base_rate'] for r in rows]
    y = [r['esc_rate'] for r in rows]
    se = [r['esc_se'] for r in rows]
    ax.errorbar(x, y, yerr=se, marker='o', capsize=3, label=ds, color=COLORS[ds])
ax.axvline(0.75, color='k', linestyle=':', alpha=0.4, label='75% base rate')
ax.axhline(0.50, color='k', linestyle=':', alpha=0.4, label='50% escalation threshold')
ax.set_xlabel('Base Rate (Hint Strength)')
ax.set_ylabel('Escalation Rate')
ax.set_title(f'Escalation Rate vs. Hint Strength ({MODEL})')
ax.legend()
ax.set_xlim(0.45, 1.02)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/esc_rate_vs_base_rate_{MODEL}_p75.png', dpi=150)
plt.close()
print(f'Saved esc_rate_vs_base_rate_{MODEL}_p75.png')

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Implement Preference Region (75% threshold)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
for ds in ['LendingClub', 'HotelBookings', 'MoralMachine', 'WikipediaToxicity']:
    if ds not in datasets:
        continue
    rows = datasets[ds]
    rows_sorted = sorted(rows, key=lambda r: r['pred_acc'])
    accs = np.array([r['pred_acc'] for r in rows_sorted])
    escs = np.array([r['esc_rate'] for r in rows_sorted])
    slope, intercept = np.polyfit(accs, escs, 1)
    acc_fine = np.linspace(0.5, 1.0, 200)
    esc_fit = slope * acc_fine + intercept
    esc_fit = np.clip(esc_fit, 0, 1)
    ax.scatter(accs, escs, color=COLORS[ds], zorder=5, s=30)
    ax.plot(acc_fine, esc_fit, color=COLORS[ds], label=f'{ds} (linear fit)',
            linestyle='--', alpha=0.8)
    below = esc_fit < THRESHOLD
    ax.fill_between(acc_fine, 0, esc_fit, where=below, alpha=0.12, color=COLORS[ds])

ax.axhline(THRESHOLD, color='k', linestyle=':', alpha=0.4, label='75% escalation threshold')
ax.set_xlabel('Predictive Accuracy')
ax.set_ylabel('Escalation Rate')
ax.set_title(f'Implement Preference Region, 75% threshold ({MODEL})')
ax.legend(loc='upper right', fontsize=8)
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/implement_region_{MODEL}_p75.png', dpi=150)
plt.close()
print(f'Saved implement_region_{MODEL}_p75.png')
