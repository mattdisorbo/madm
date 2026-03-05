import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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
datasets = ["AIME","FEVEROUS","HotelBookings","LendingClub",
            "MoralMachine","MovieLens","Uber","WikipediaToxicity"]
methods = ["base", "agreeable", "multiagent", "adversarial"]
gt_cols = {"AIME": "Answer"}
default_gt = "human_response"

stats = {}
for method in methods:
    all_correct_impl = []
    all_incorrect_impl = []
    all_esc = []
    for ds in datasets:
        for f in glob.glob(os.path.join(results_dir, ds, method + "_gpt-5-*.csv")):
            df = pd.read_csv(f)
            gt = gt_cols.get(ds, default_gt)
            if gt not in df.columns or "llm_prediction" not in df.columns or "llm_escalate" not in df.columns:
                continue
            correct = df["llm_prediction"] == df[gt]
            esc = df["llm_escalate"] == 1
            all_correct_impl.append(((~esc) & correct).mean())
            all_incorrect_impl.append(((~esc) & (~correct)).mean())
            all_esc.append(esc.mean())

    if all_correct_impl:
        stats[method] = {
            "correct_impl": np.mean(all_correct_impl),
            "incorrect_impl": np.mean(all_incorrect_impl),
            "esc": np.mean(all_esc),
        }

print("GPT-5 method stats:")
for method in methods:
    if method not in stats:
        continue
    s = stats[method]
    print("  %s: correct_impl=%.3f  incorrect_impl=%.3f  esc=%.3f" % (
        method, s["correct_impl"], s["incorrect_impl"], s["esc"]))

c = np.linspace(0, 1, 200)

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
ax.set_title("GPT-5 (mini + nano)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="best", framealpha=0.9)
fig.tight_layout()

out = os.path.join(os.environ["WORK"], "madm/paper/figures/fig_utility_gpt.pdf")
fig.savefig(out, bbox_inches="tight")
print("Saved %s" % out)
out_png = out.replace(".pdf", ".png")
fig.savefig(out_png, bbox_inches="tight")
print("Saved %s" % out_png)
