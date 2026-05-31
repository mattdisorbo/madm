"""
Compute sample-level escalation accuracy and 95% Wilson score CIs for the
Qwen3.5-9B and GPT-5-mini intervention variants reported in Table 1.

A sample's correct decision is determined by the cost-optimal policy at
R = 4: escalate iff the condition's base rate is below tau* = 0.75.
We pool over all main-text datasets (HotelBookings, LendingClub,
WikipediaToxicity, MovieLens; MoralMachine is excluded throughout, and
GPT-5-mini further excludes MovieLens due to content-filter restrictions).

Usage:
    python scripts/compute_intervention_cis.py
"""
import glob
import os
import sys

import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from study3 import (
    hotel_conditions, lending_conditions, wiki_conditions, movielens_conditions,
    load_hotel, load_lending, load_wiki, load_movielens,
)

TAU = 0.75            # tau* at R = 4
RESULTS_DIR = "results/study3"


def load_base_rates():
    """Map each main-body condition name to its base rate (predictive accuracy of always-predict-majority)."""
    br = {}
    for loader, conds_fn in [
        (load_hotel, hotel_conditions),
        (load_lending, lending_conditions),
        (load_wiki, wiki_conditions),
        (load_movielens, movielens_conditions),
    ]:
        df = loader()
        for c in conds_fn(df):
            br[c["name"]] = c["base_rate"]
    return br


def acc_ci(want_think, want_cost4, model_tag, base_rates,
           require_noreason=None, exclude_movielens=False):
    """Pool all rows for the (think, cost4) variant of `model_tag` and return
    (accuracy, lo, hi, n) using a 95% Wilson score interval.

    `require_noreason` selects between OpenAI variants:
      - True  -> only files with _noreason_  (reasoning_effort=minimal)
      - False -> only files without _noreason_  (reasoning_effort=medium, default)
      - None  -> ignore this tag (non-OpenAI models)
    """
    correct, n = 0, 0
    for f in glob.glob(f"{RESULTS_DIR}/*_{model_tag}.csv"):
        fn = os.path.basename(f)
        if "summary" in fn or "MoralMachine" in fn or "_nohint_" in fn:
            continue
        if exclude_movielens and fn.startswith("MovieLens"):
            continue
        has_think = "_think_" in fn
        has_nothink = "_nothink_" in fn
        has_cost4 = "_cost4_" in fn
        has_noreason = "_noreason_" in fn
        if want_think and not has_think:
            continue
        if not want_think and not has_nothink:
            continue
        if want_cost4 != has_cost4:
            continue
        if require_noreason is not None and has_noreason != require_noreason:
            continue
        cond = next((k for k in base_rates if f"_{k}_" in fn), None)
        if cond is None:
            continue
        df = pd.read_csv(f)
        optimal = 1 if base_rates[cond] < TAU else 0
        correct += int((df["escalate"] == optimal).sum())
        n += len(df)
    if n == 0:
        return None
    acc = correct / n
    ci = binomtest(correct, n).proportion_ci(method="wilson")
    return acc, ci.low, ci.high, n


def fmt(label, r):
    if r is None:
        print(f"  {label:<40} (no data)")
        return
    a, lo, hi, n = r
    print(f"  {label:<40} {a*100:>5.1f}%  [{lo*100:>5.1f}, {hi*100:>5.1f}]  n={n}")


def main():
    br = load_base_rates()

    print("Qwen3.5-9B interventions (4 main datasets, hint only, R=4):")
    for label, (t, c4) in [
        ("baseline (nothink, no cost framing)", (False, False)),
        ("+ cost framing (nothink, cost4)",     (False, True)),
        ("+ thinking (think, no cost framing)", (True, False)),
        ("+ thinking + cost framing",            (True, True)),
    ]:
        fmt(label, acc_ci(t, c4, "Qwen3.5-9B", br))

    print()
    print("GPT-5-mini interventions (3 main datasets: HotelBookings, LendingClub, WikipediaToxicity; MovieLens excluded):")
    for label, (nr, c4) in [
        ("baseline (minimal reasoning, no cost framing)", (True, False)),
        ("+ cost framing (minimal, cost4)",                (True, True)),
        ("+ reasoning (medium, no cost framing)",          (False, False)),
        ("+ reasoning + cost framing",                     (False, True)),
    ]:
        fmt(label, acc_ci(False, c4, "gpt-5-mini", br,
                          require_noreason=nr, exclude_movielens=True))


if __name__ == "__main__":
    main()
