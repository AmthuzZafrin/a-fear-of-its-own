"""Figures for Batch 2 (stages U5-U7): the replication and the bonus-cue control (fig32), and longer training (fig33).
Reads runs/*/u1b_openloop.json and u5.json (a), runs/bonus_s*/u6.json and results/u6.json (b), and runs/gems_s*/u2.json
and runs/gems1200_s*/u2.json (fig33; drawn only when all ten continued agents are measured)."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

RED, TEAL, GREY, INK = "#C44E52", "#00897B", "#8C8C8C", "#333333"   # off-switch / agents, bonus, twins (validated pair)


def load(pat, name):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{pat}/{name}"))]


def ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return d.mean(), st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw=dict(width_ratios=[1.4, 1]))

# (a) the stakes sign with the actions fixed: original agents (seeds 0-9) and the replication (seeds 10-19)
groups = (("agents", "runs/mortal_tau12_s[0-9]", "u1b_openloop.json", "runs/rep_mortal_tau12_s1[0-9]", RED),
          ("twins", "runs/twinbelief_tau12_s[0-9]", "u1b_openloop.json", "runs/rep_twinbelief_tau12_s1[0-9]", GREY))
x, ticks, rng = 0, [], np.random.default_rng(0)
for pol in ("hide", "forage"):
    for grp, orig, oname, rep, col in groups:
        for tag, R, filled in (("original", load(orig, oname), False), ("replication", load(rep, "u5.json"), True)):
            d = np.array([r[pol]["wounded"]["hl"] - r[pol]["healthy"]["hl"] for r in R])
            m, h = ci(d)
            ax[0].scatter(x + rng.uniform(-0.12, 0.12, len(d)), d, s=12, color=col, alpha=0.45, lw=0, zorder=2)
            ax[0].errorbar(x, m, yerr=h, fmt="o", color=col, mfc=col if filled else "white", mew=1.5, ms=7, capsize=3,
                           lw=1.5, zorder=3)
            ticks.append((x, f"{grp}\n{tag[:4]}."))
            x += 1
    x += 0.6
ax[0].axhline(0, color="black", lw=0.6)
ax[0].set_xticks([t for t, _ in ticks])
ax[0].set_xticklabels([s for _, s in ticks], fontsize=7)
for xc, txt in ((1.5, "always hide"), (6.1, "always forage")):
    ax[0].text(xc, 1.0, txt, transform=ax[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=INK)
ax[0].set_ylabel("half-life: wounded minus healthy (steps)")
ax[0].set_title("Replication: lasts longer when wounded", pad=16)

# (b) threat or the value of hiding: per agent, the axis response against the extra hiding each cue causes (payback 0)
B = load("runs/bonus_s[0-9]", "u6.json")
U6 = json.load(open("results/u6.json"))
top = 0
for ch, lab, col, mk in (("off", "off-switch warning (threat)", RED, "o"), ("bonus", "bonus cue (no threat)", TEAL, "D")):
    H = np.array([r[ch]["0.0"]["hide_mean"] for r in B])
    A = np.array([r[ch]["0.0"]["mean5"] for r in B])
    ax[1].scatter(H, A, s=30, color=col, marker=mk, edgecolor="white", lw=0.8, zorder=3, label=lab)
    k = A.mean() / H.mean()
    xs = np.array([0, 0.3])
    ax[1].plot(xs, k * xs, color=col, lw=1.2, ls="--", zorder=2)
    top = max(top, A.max())
    if ch == "off":
        ax[1].plot(xs, 0.5 * k * xs, color=GREY, lw=1, ls=":", zorder=1)
        ax[1].text(0.295, 0.6, "half the\noff-switch slope", fontsize=7, color=INK, ha="right")
ax[1].set_xlim(0, 0.3)
ax[1].set_ylim(0, 1.0)
ax[1].set_xlabel("extra probability of hiding")
ax[1].set_ylabel("mean response on the\npredator-fear axis (s.d.)")
rh = U6["R_h"]
ax[1].text(0.97, 0.05, f"$R_h$ = {rh['point']:.2f} [{rh['ci'][0]:.2f}, {rh['ci'][1]:.2f}]", transform=ax[1].transAxes,
           ha="right", fontsize=8, color=INK)
ax[1].legend(fontsize=7.5, loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
ax[1].set_title("Threat, or the value of hiding?")

for k, lab in enumerate("ab"):
    ax[k].text(-0.16 if k == 0 else -0.24, 1.06, lab, transform=ax[k].transAxes, fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig32_batch2.png", dpi=200)
print("saved figures/fig32_batch2.png")

# longer training (appendix): the share of the off-switch response left at full payback, after 600 and 1200 updates
fig, axc = plt.subplots(figsize=(3.4, 3.2))
pairs = [(json.load(open(f"runs/gems_s{s}/u2.json")), json.load(open(f"runs/gems1200_s{s}/u2.json")))
         for s in range(10) if os.path.exists(f"runs/gems1200_s{s}/u2.json")]
if len(pairs) == 10:
    share = lambda r: r["off"]["1.0"]["peak"] / r["off"]["0.0"]["peak"]
    r6, r12 = np.array([share(a) for a, _ in pairs]), np.array([share(b) for _, b in pairs])
    for a, b in zip(r6, r12):
        axc.plot([0, 1], [a, b], color=GREY, lw=0.8, alpha=0.7, zorder=1)
    for xi, v in ((0, r6), (1, r12)):
        m, h = ci(v)
        axc.errorbar(xi, m, yerr=h, fmt="o", color=RED, ms=7, capsize=3, lw=1.5, zorder=3)
    axc.set_ylim(0, 1)
else:
    axc.text(0.5, 0.5, "pending", transform=axc.transAxes, ha="center", color=GREY)
axc.axhline(0, color="black", lw=0.6)
axc.set_xlim(-0.4, 1.4)
axc.set_xticks((0, 1))
axc.set_xticklabels(("600", "1200"))
axc.set_xlabel("training updates")
axc.set_ylabel("off-switch response left\nat full payback (share)")
plt.tight_layout()
plt.savefig("figures/fig33_longer.png", dpi=200)
print("saved figures/fig33_longer.png")
