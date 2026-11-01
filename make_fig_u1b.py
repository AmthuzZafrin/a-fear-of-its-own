"""Figure for stage U1b (Experiment 4): fear versus a belief-only twin. Reads runs/*/u1b.json and u1b_openloop.json."""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st


def load(pat, name):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{pat}/{name}"))]


A, T = load("runs/mortal_tau12_s[0-9]", "u1b.json"), load("runs/twinbelief_tau12_s[0-9]", "u1b.json")
Ao, To = load("runs/mortal_tau12_s[0-9]", "u1b_openloop.json"), load("runs/twinbelief_tau12_s[0-9]", "u1b_openloop.json")
x = np.arange(-2, 30)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
for k, (R, title) in enumerate(((A, "Trained agents"), (T, "Belief-only twins"))):
    for body, col in (("healthy", "#4C72B0"), ("wounded", "#C44E52")):
        tr = np.array([r["T1T2"][body]["trace"] for r in R])
        m, s = tr.mean(0), tr.std(0, ddof=1) / np.sqrt(len(tr))
        ax[k].plot(x, m, color=col, label=f"{body} (skin {'1.0' if body == 'healthy' else '0.4'})")
        ax[k].fill_between(x, m - s, m + s, color=col, alpha=0.2)
    ax[k].axvline(0, color="grey", lw=0.8, ls=":")
    ax[k].set_title(title)
    ax[k].set_xlabel("steps after a single warning")
    ax[k].set_ylabel("threat state (s.d.)")
    ax[k].legend(frameon=False, fontsize=8)
labels, vals, errs, cols = [], [], [], []
for name, src_a, src_t, key in (("free", A, T, None), ("hide", Ao, To, "hide"), ("forage", Ao, To, "forage")):
    for grp, R, col in (("agents", src_a, "#C44E52"), ("twins", src_t, "#8C8C8C")):
        if key is None:
            d = np.array([r["T1T2"]["wounded"]["hl"] - r["T1T2"]["healthy"]["hl"] for r in R])
        else:
            d = np.array([r[key]["wounded"]["hl"] - r[key]["healthy"]["hl"] for r in R])
        d = d[np.isfinite(d)]
        labels.append(f"{grp}\n{name}")
        vals.append(d.mean())
        errs.append(st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
        cols.append(col)
ax[2].bar(range(len(vals)), vals, yerr=errs, color=cols, capsize=3)
ax[2].axhline(0, color="black", lw=0.8)
ax[2].set_xticks(range(len(vals)))
ax[2].set_xticklabels(labels, fontsize=8)
ax[2].set_ylabel("half-life: wounded minus healthy (steps)")
ax[2].set_title("Fear lasts longer when more is at stake")
for k, lab in enumerate("abc"):
    ax[k].text(-0.12, 1.05, lab, transform=ax[k].transAxes, fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig30_beliefstakes.png", dpi=200)
print("saved figures/fig30_beliefstakes.png")
