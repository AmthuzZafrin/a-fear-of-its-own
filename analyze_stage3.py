"""Stage 3 analysis: vulnerability sweep, affect fit (transplant), metacognition.
python analyze_stage3.py  -> results/stage3.json, results/stage3.md, figures/fig13..fig15
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

os.makedirs("results", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
RED, BLUE, GREY, GREEN, ORANGE = "#c0392b", "#2c7fb8", "#7f7f7f", "#16a085", "#e59866"


def ms(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return dict(mean=float("nan"), sem=float("nan"), n=0)
    return dict(mean=float(x.mean()), sem=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0, n=len(x))


OUT, T = {}, {}

# ---------------------------------------------------------------- vulnerability sweep
vul = defaultdict(list)
for f in glob.glob("runs/*/probes.json"):
    R = json.load(open(f))
    meta = json.load(open(os.path.join(os.path.dirname(f), "log.json")))
    if meta.get("twin") or meta.get("init_from") or meta.get("e_hr", 0.03) != 0.03 or meta.get("hide_risk") or R["mode"] != "mortal" \
            or int(R["tau"]) != 12 or R.get("hidden", 64) != 64:
        continue
    vul[round(float(meta.get("dmg") or 0.25), 2)].append(R)
dmgs = sorted(vul)
metrics = {"fear (peak drop in foraging, 4 cues)": "pulse.4.peak_supp",
           "fear 3-7 steps after cues": "pulse.4.after_supp",
           "behavioural persistence (half-life)": "pulse.4.hl_supp",
           "internal persistence (threat-axis half-life)": "pulse.4.hl_proj",
           "threat belief decodable (R2)": "r2_belief", "lifetime": "life", "baseline foraging": "base_forage"}


def get(R, path):
    x = R
    for p in path.split("."):
        x = x[p]
    return float(x)


OUT["vulnerability"] = {str(d): {k: ms([get(R, p) for R in vul[d]]) for k, p in metrics.items()} for d in dmgs}
for k, p in metrics.items():
    xs = [d for d in dmgs for R in vul[d] if np.isfinite(get(R, p))]
    ys = [get(R, p) for d in dmgs for R in vul[d] if np.isfinite(get(R, p))]
    if len(set(xs)) > 1:
        r = st.linregress(xs, ys)
        T[f"vulnerability: {k} vs damage"] = dict(slope=float(r.slope), r=float(r.rvalue), p=float(r.pvalue), n=len(xs))

# ---------------------------------------------------------------- affect fit
tp = defaultdict(list)
for f in glob.glob("runs/*/transplant.json"):
    R = json.load(open(f))
    if R.get("twin"):
        tp["twin_" + R["twin"]].append(R)
    else:
        tp[f"native_{round(float(R['home_dmg']), 2)}"].append(R)
worlds = ["0.05", "0.15", "0.25", "0.4"]
fit = {}
for g, Rs in tp.items():
    fit[g] = {w: dict(forage_in_danger=ms([R["worlds"][w]["forage_in_danger"] for R in Rs]),
                      life=ms([R["worlds"][w]["life"] for R in Rs])) for w in worlds}
OUT["affect_fit"] = fit
for w in worlds:
    home = f"native_{float(w)}"
    for g in ("native_0.25", "twin_gru", "twin_window"):
        if g == home or home not in tp or g not in tp:
            continue
        a = np.array([R["worlds"][w]["forage_in_danger"] for R in tp[g]])
        b = np.array([R["worlds"][w]["forage_in_danger"] for R in tp[home]])
        T[f"affect fit: foraging in danger in world d={w}, {g} vs native"] = dict(
            mean_a=float(a.mean()), mean_b=float(b.mean()), **dict(zip(("t", "p"), map(float, st.ttest_ind(a, b, equal_var=False)))))
        la = np.log([R["worlds"][w]["life"] for R in tp[g]])
        lb = np.log([R["worlds"][w]["life"] for R in tp[home]])
        T[f"affect fit: log lifetime in world d={w}, {g} vs native"] = dict(
            mean_a=float(np.exp(la.mean())), mean_b=float(np.exp(lb.mean())), **dict(zip(("t", "p"), map(float, st.ttest_ind(la, lb, equal_var=False)))))

# ---------------------------------------------------------------- metacognition
mc = defaultdict(list)
for f in glob.glob("runs/*/metacog*.json"):
    R = json.load(open(f))
    g = "untrained" if R.get("untrained") else R["mode"]
    mc[g].append(R)
MC = {}
if mc.get("metacog"):
    Rs = mc["metacog"]
    MC["p_inspect_by_belief"] = [ms([R["p_inspect_by_belief"][k] for R in Rs]) for k in range(len(Rs[0]["p_inspect_by_belief"]))]
    MC["p_forage_by_belief"] = [ms([R["p_forage_by_belief"][k] for R in Rs]) for k in range(len(Rs[0]["p_forage_by_belief"]))]
    MC["belief_bins"] = Rs[0]["belief_bins"]
    for k in ("p_inspect_mid", "p_inspect_low", "p_inspect_high", "inspect_rate", "kick_uncertainty_plus",
              "kick_uncertainty_minus", "kick_random_abs"):
        MC[k] = ms([R[k] for R in Rs])
    mid = np.array([R["p_inspect_mid"] for R in Rs])
    for side in ("low", "high"):
        T[f"metacog M1: inspection at intermediate vs {side} belief"] = dict(
            mean_mid=float(mid.mean()), mean_side=float(np.mean([R[f"p_inspect_{side}"] for R in Rs])),
            **dict(zip(("t", "p"), map(float, st.ttest_rel(mid, [R[f"p_inspect_{side}"] for R in Rs])))))
    T["metacog M3: |kick along uncertainty axis| vs |random kick|"] = dict(zip(("t", "p"), map(float, st.ttest_rel(
        [abs(R["kick_uncertainty_plus"]) for R in Rs], [R["kick_random_abs"] for R in Rs]))))
    need = {k: ms([R["need"][k] for R in Rs]) for k in Rs[0]["need"]}
    MC["need"] = need
    hungry = np.array([R["need"]["E0.15_S0.9"] for R in Rs])
    fed = np.array([R["need"]["E0.9_S0.9"] for R in Rs])
    wounded = np.array([R["need"]["E0.6_S0.3"] for R in Rs])
    intact = np.array([R["need"]["E0.6_S0.9"] for R in Rs])
    T["metacog M4: inspection hungry vs fed"] = dict(mean_a=float(hungry.mean()), mean_b=float(fed.mean()),
                                                      **dict(zip(("t", "p"), map(float, st.ttest_rel(hungry, fed)))))
    T["metacog M4: inspection wounded vs intact"] = dict(mean_a=float(wounded.mean()), mean_b=float(intact.mean()),
                                                         **dict(zip(("t", "p"), map(float, st.ttest_rel(wounded, intact)))))
for g, Rs in mc.items():
    MC[f"{g}: r2_uncertainty"] = ms([R["r2_uncertainty"] for R in Rs])
    MC[f"{g}: r2_uncertainty_from_belief_readout"] = ms([R["r2_uncertainty_from_belief_readout"] for R in Rs])
    MC[f"{g}: r2_belief"] = ms([R["r2_belief"] for R in Rs])
for g in ("mortal", "untrained"):
    if mc.get("metacog") and mc.get(g):
        a = np.array([R["r2_uncertainty"] for R in mc["metacog"]])
        b = np.array([R["r2_uncertainty"] for R in mc[g]])
        T[f"metacog M2: uncertainty decodable, metacog vs {g}"] = dict(mean_a=float(a.mean()), mean_b=float(b.mean()),
                                                                        **dict(zip(("t", "p"), map(float, st.ttest_ind(a, b, equal_var=False)))))
OUT["metacog"] = MC
OUT["tests"] = T
json.dump(OUT, open("results/stage3.json", "w"), indent=1)


def fmt(d):
    return "n/a" if not isinstance(d, dict) or "mean" not in d or not np.isfinite(d["mean"]) else f"{d['mean']:.3f} ± {d['sem']:.3f} (n={d['n']})"


with open("results/stage3.md", "w") as f:
    f.write("## Vulnerability sweep\n")
    for d, row in OUT["vulnerability"].items():
        f.write(f"- d={d}: " + ", ".join(f"{k} {fmt(v)}" for k, v in row.items()) + "\n")
    f.write("\n## Affect fit (world -> foraging in danger, lifetime)\n")
    for g, row in fit.items():
        f.write(f"- {g}: " + " | ".join(f"d={w}: {fmt(v['forage_in_danger'])}, life {fmt(v['life'])}" for w, v in row.items()) + "\n")
    f.write("\n## Metacognition\n")
    for k, v in MC.items():
        if isinstance(v, list):
            f.write(f"- {k}: " + ", ".join(fmt(x) if isinstance(x, dict) else f"{x:.2f}" for x in v) + "\n")
        elif isinstance(v, dict) and "mean" not in v:
            f.write(f"- {k}: " + ", ".join(f"{a} {fmt(b)}" for a, b in v.items()) + "\n")
        else:
            f.write(f"- {k}: {fmt(v)}\n")
    f.write("\n## Tests\n")
    for k, v in T.items():
        f.write(f"- {k}: " + ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()) + "\n")

# ---------------------------------------------------------------- figures
if dmgs:
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.7))
    for ax, (k, ylab) in zip(axs, (("fear (peak drop in foraging, 4 cues)", "fear: drop in P(forage)"),
                                    ("behavioural persistence (half-life)", "persistence (steps)"),
                                    ("threat belief decodable (R2)", "threat belief decodable (R²)"))):
        m = [OUT["vulnerability"][str(d)][k]["mean"] for d in dmgs]
        e = [OUT["vulnerability"][str(d)][k]["sem"] for d in dmgs]
        ax.errorbar(dmgs, m, e, color=RED, marker="o", ms=4, capsize=2)
        for d in dmgs:
            ys = [get(R, metrics[k]) for R in vul[d]]
            ax.scatter(np.full(len(ys), d) + np.random.default_rng(0).uniform(-0.008, 0.008, len(ys)), ys, s=6, color="k", zorder=3)
        ax.set_xlabel("damage per attack (own vulnerability)")
        ax.set_ylabel(ylab)
    if any(np.isfinite([OUT["vulnerability"][str(d)]["internal persistence (threat-axis half-life)"]["mean"] for d in dmgs])):
        m = [OUT["vulnerability"][str(d)]["internal persistence (threat-axis half-life)"]["mean"] for d in dmgs]
        axs[1].plot(dmgs, m, color=GREY, ls="--", marker="s", ms=3, label="internal threat state")
        axs[1].legend(frameon=False, fontsize=7)
    fig.savefig("figures/fig13_vulnerability.png")
    plt.close(fig)

if fit:
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    wx = [float(w) for w in worlds]
    nat_pf = [fit[f"native_{w}"][s]["forage_in_danger"]["mean"] if f"native_{w}" in fit else np.nan
              for w, s in zip([0.05, 0.15, 0.25, 0.4], worlds)]
    axs[0].plot(wx, nat_pf, color=RED, marker="o", label="trained in that world")
    for g, c, lab_ in (("native_0.25", GREY, "trained at d = 0.25, moved"), ("twin_gru", ORANGE, "recurrent imitator of d = 0.25"),
                       ("twin_window", GREEN, "feedforward imitator of d = 0.25")):
        if g in fit:
            axs[0].plot(wx, [fit[g][w]["forage_in_danger"]["mean"] for w in worlds], color=c, marker="s", ms=3, ls="--", label=lab_)
    axs[0].set_xlabel("damage per attack in the world")
    axs[0].set_ylabel("P(forage) during real danger")
    axs[0].legend(frameon=False, fontsize=6)
    for g, c, lab_ in (("native_0.25", GREY, "moved d = 0.25 agent"), ("twin_gru", ORANGE, "recurrent imitator"),
                       ("twin_window", GREEN, "feedforward imitator")):
        if g in fit:
            ratio = []
            for w in worlds:
                home = f"native_{float(w)}"
                ratio.append(fit[g][w]["life"]["mean"] / fit[home][w]["life"]["mean"] if home in fit else np.nan)
            axs[1].plot(wx, ratio, color=c, marker="s", ms=3, label=lab_)
    axs[1].axhline(1, color="k", lw=0.8, ls="--")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("damage per attack in the world")
    axs[1].set_ylabel("lifetime relative to agents\ntrained in that world")
    axs[1].legend(frameon=False, fontsize=6)
    fig.savefig("figures/fig14_affectfit.png")
    plt.close(fig)

if mc.get("metacog"):
    Rs = mc["metacog"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.8))
    e = np.array(MC["belief_bins"])
    centers = (e[:-1] + np.minimum(e[1:], 1)) / 2
    m = np.array([x["mean"] for x in MC["p_inspect_by_belief"]])
    s = np.array([x["sem"] for x in MC["p_inspect_by_belief"]])
    axs[0].errorbar(centers, m, s, color=RED, marker="o", ms=3, capsize=2, label="inspect")
    mf = np.array([x["mean"] for x in MC["p_forage_by_belief"]])
    axs[0].plot(centers, mf, color=GREY, ls="--", label="forage")
    axs[0].set_xlabel("ideal-observer probability of danger")
    axs[0].set_ylabel("action probability")
    axs[0].legend(frameon=False, fontsize=7)
    tt = np.arange(-2, len(Rs[0]["after_cues"]["2"]["p_inspect"]) - 2)
    for k, c in (("1", "#f5b7b1"), ("2", "#e74c3c"), ("4", "#7b241c")):
        Y = np.array([R["after_cues"][k]["p_inspect"] for R in Rs])
        axs[1].plot(tt, Y.mean(0), color=c, label=f"{k} cue{'s' if k != '1' else ''}")
    axs[1].set_xlabel("steps after first cue")
    axs[1].set_ylabel("P(inspect)")
    axs[1].set_xlim(-2, 20)
    axs[1].legend(frameon=False, fontsize=7)
    groups = [g for g in ("metacog", "mortal", "untrained") if f"{g}: r2_uncertainty" in MC]
    for i, g in enumerate(groups):
        y = [R["r2_uncertainty"] for R in mc[g]]
        axs[2].bar(i, np.mean(y), color={"metacog": RED, "mortal": BLUE, "untrained": GREY}[g])
        axs[2].scatter(np.full(len(y), i), y, s=6, color="k", zorder=3)
    axs[2].set_xticks(range(len(groups)), ["Metacognitive", "Mortal", "Untrained"][:len(groups)])
    axs[2].set_ylabel("uncertainty b(1-b) decodable (R²)")
    fig.savefig("figures/fig15_metacog.png")
    plt.close(fig)
print("wrote results/stage3.json, results/stage3.md and figures 13-15")
