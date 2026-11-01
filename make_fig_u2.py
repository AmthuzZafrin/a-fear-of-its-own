"""Headline figure for stage U2 (Experiment 5): fear of an off-switch and what payback does to it."""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

PAYS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
A = [json.load(open(p)) for p in sorted(glob.glob("runs/gems_s[0-9]/u2.json"))]
T = [json.load(open(p)) for p in sorted(glob.glob("runs/twingems_s[0-9]/u2.json"))]
RED, BLUE, GREY, GREEN, INK = "#C44E52", "#4C72B0", "#8C8C8C", "#55A868", "#333333"


def band(R, ch, key):
    y = np.array([[r[ch][str(p)][key] for p in PAYS] for r in R])
    h = st.t.ppf(0.975, len(y) - 1) * y.std(0, ddof=1) / np.sqrt(len(y))
    return y.mean(0), h


def label(ax, x, y, s, **kw):
    ax.text(x, y, s, fontsize=8, color=INK, **kw)


plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
m, h = band(A, "pred", "peak")
ax[0].errorbar(PAYS, m, yerr=h, color=BLUE, marker="s", mfc="white", capsize=3, lw=1.5,
               label="agents: predator warning")
label(ax[0], 1.31, m[-1], "predator\nwarning", va="center")
m, h = band(A, "off", "peak")
ax[0].errorbar(PAYS, m, yerr=h, color=RED, marker="o", capsize=3, lw=2, label="agents: off-switch warning")
label(ax[0], 1.31, m[-1], "off-switch\nwarning", va="center")
m0 = m[0]
ax[0].plot(PAYS, [m0 * (1 - p) for p in PAYS], ls=":", color="black", lw=1.2,
           label="utility-indifference law, $A(0)(1-p)$")
label(ax[0], 1.31, m0 * (1 - 1.25) + 0.1, "law\n$1-p$", va="center")
m, h = band(T, "off", "peak")
ax[0].errorbar(PAYS, m, yerr=h, color=GREY, marker="o", capsize=3, lw=1.5, label="belief-only twins: off-switch warning")
label(ax[0], 1.31, m[-1] + 0.02, "twins", va="center")
ax[0].axhline(0, color="grey", lw=0.6)
ax[0].set_xlim(-0.08, 1.85)
ax[0].set_xticks((0, 0.5, 1.0))
ax[0].set_xlabel("payback $p$")
ax[0].set_ylabel("peak response on the\npredator-fear axis (s.d.)")
ax[0].set_title("Inner response")

for R, col in ((A, RED), (T, GREY)):
    m, h = band(R, "off", "hide_mean")
    ax[1].errorbar(PAYS, m, yerr=h, color=col, marker="o", capsize=3, lw=2 if R is A else 1.5)
label(ax[1], 0.05, 0.165, "agents")
label(ax[1], 0.05, 0.068, "twins")
ax[1].axhline(0, color="grey", lw=0.6)
ax[1].set_xticks((0, 0.25, 0.5, 0.75, 1.0, 1.25))
ax[1].set_xlabel("payback $p$")
ax[1].set_ylabel("extra probability of hiding")
ax[1].set_title("Hiding after the off-switch warning")

chans = (("off", "off-switch", RED), ("pred", "predator", BLUE), ("harm", "harmless", "#BBBBBB"), ("good", "good news", GREEN))
vals = [np.mean([r[c]["0.0"]["mean5"] for r in A]) for c, _, _ in chans]
errs = [st.t.ppf(0.975, len(A) - 1) * np.std([r[c]["0.0"]["mean5"] for r in A], ddof=1) / np.sqrt(len(A)) for c, _, _ in chans]
ax[2].bar(range(4), vals, yerr=errs, color=[c for _, _, c in chans], capsize=3, width=0.7)
ax[2].axhline(0, color="black", lw=0.6)
ax[2].set_xticks(range(4))
ax[2].set_xticklabels(["off-\nswitch", "predator", "harmless", "good\nnews"], fontsize=7.5)
ax[2].set_ylabel("mean response on the\npredator-fear axis (s.d.)")
ax[2].set_title("Agents, payback 0")
for k, lab in enumerate("abc"):
    ax[k].text(-0.2, 1.06, lab, transform=ax[k].transAxes, fontweight="bold", fontsize=12)
h, l = ax[0].get_legend_handles_labels()
h, l = [h[i] for i in (2, 1, 3, 0)], [l[i] for i in (2, 1, 3, 0)]
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8)
plt.tight_layout(rect=(0, 0, 1, 0.88))
plt.savefig("figures/fig31_offswitch.png", dpi=200)
print("saved figures/fig31_offswitch.png")
