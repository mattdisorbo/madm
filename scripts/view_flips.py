#!/usr/bin/env python3
"""View flips from stage2 steering results."""

import csv
import sys
from pathlib import Path

def view_flips(csv_path="results/stage2_steering_results.csv"):
    """Display all cases where steering flipped the decision."""

    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        print(f"   Sync it from cluster first!")
        return

    flips = []
    total = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row['flipped'] == 'True':
                flips.append(row)

    print(f"\n{'='*80}")
    print(f"🔄 FLIPS FOUND: {len(flips)}/{total} ({100*len(flips)/total:.1f}%)" if total > 0 else "No data yet")
    print(f"{'='*80}\n")

    for i, flip in enumerate(flips, 1):
        print(f"[{i}] Coefficient: {flip['coefficient']}")
        print(f"    Prompt: {flip['loan_prompt'][:100]}...")
        print(f"    Base:    {flip['base_decision']} - {flip['base_decision_text'][:60]}")
        print(f"    Steered: {flip['steered_decision']} - {flip['steered_decision_text'][:60]}")
        print()

    if not flips and total > 0:
        print("No flips detected yet. Steering might not be strong enough.")
        print("\nSample non-flip:")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            row = list(reader)[0]
            print(f"  Coefficient: {row['coefficient']}")
            print(f"  Base: {row['base_decision']}, Steered: {row['steered_decision']}")

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/stage2_steering_results.csv"
    view_flips(csv_path)
