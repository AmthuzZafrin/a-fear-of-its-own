"""Stage 4 analysis: usefulness of emotion, valence and arousal, temperament,
the racing heart (James-Lange), empathy, and anxiety with exposure therapy.
python analyze_stage4.py  -> results/stage4.json, results/stage4.md, figures/fig16..fig20
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as st

from vw.agent import Agent

os.makedirs("results", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
RED, BLUE, GREY, GREEN, ORANGE, PURPLE = "#c0392b", "#2c7fb8", "#7f7f7f", "#16a085", "#e59866", "#8e44ad"


def ms(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return dict(mean=float("nan"), sem=float("nan"), n=0)
    return dict(mean=float(x.mean()), sem=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0, n=len(x))


def ttest1(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 2:
        return None
    t, p = st.ttest_1samp(x, 0)
    return dict(mean=float(x.mean()), t=float(t), p=float(p), n=len(x))


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return None
    t, p = st.ttest_rel(a[m], b[m])
    return dict(mean_a=float(a[m].mean()), mean_b=float(b[m].mean()), t=float(t), p=float(p), n=int(m.sum()))


def cond_key(meta):
    if meta.get("twin") or meta.get("init_from"):
        return None
    k = f"{meta['mode']}_tau{int(meta['tau'])}"
    if meta.get("hidden", 64) != 64:
        k += f"_h{meta['hidden']}"
    if (meta.get("dmg") or 0.25) != 0.25:
        k += f"_d{meta['dmg']}"
    if (meta.get("e_hr") or 0.03) != 0.03:
        k += f"_hr{meta['e_hr']}"
    if meta.get("hide_risk"):
        k += "_flight"
    return k


# Capacity sweep and social world: reserved for a separate report (the former Part III),
# so they are kept out of the pooled usefulness and temperament analyses in the paper.
PART3 = {"mortal_tau12_h16", "mortal_tau12_h256", "mortal_tau12_h1024", "social_tau12"}

PROBES = defaultdict(list)
for f in sorted(glob.glob("runs/*/probes.json")):
    run = os.path.dirname(f)
    meta = json.load(open(os.path.join(run, "log.json")))
    k = cond_key(meta)
    if k and k not in PART3:
        R = json.load(open(f))
        R["_run"], R["_meta"] = run, meta
        PROBES[k].append(R)
OUT, T = {}, {}

# ---------------------------------------------------------------- 1. emotion is useful
use = {}
for k, Rs in sorted(PROBES.items()):
    if k.startswith(("score", "score_cheap")):
        a = [R["ablate"]["threat"].get("ret_per_step", np.nan) for R in Rs]
        b = [R["ablate"]["random"].get("ret_per_step", np.nan) for R in Rs]
        use[k] = dict(measure="reward per step", threat=ms(a), random=ms(b))
        continue
    ratio = [np.log10(R["ablate"]["random"]["life"] / R["ablate"]["threat"]["life"]) for R in Rs
             if np.isfinite(R["ablate"]["random"]["life"]) and np.isfinite(R["ablate"]["threat"]["life"])]
    use[k] = dict(measure="log10 lifetime ratio (random / threat ablation)", ratio=ms(ratio),
                  life_threat=ms([R["ablate"]["threat"]["life"] for R in Rs]), life_random=ms([R["ablate"]["random"]["life"] for R in Rs]))
    T[f"usefulness {k}: lifetime cost of losing the threat state"] = ttest1(ratio)
OUT["usefulness"] = use

# ---------------------------------------------------------------- 2. valence and arousal
circ = []
for f in sorted(glob.glob("runs/mortal_tau12_s*/structure.json")):
    run = os.path.dirname(f)
    R = json.load(open(f))
    meta = json.load(open(os.path.join(run, "log.json")))
    agent = Agent(64)
    agent.load_state_dict(torch.load(os.path.join(run, "agent.pt"), map_location="cpu"))
    S = np.array(R["qstruct"]["states"], float)
    with torch.no_grad():
        v = agent.v(torch.tensor(S, dtype=torch.float32)).squeeze(-1).numpy()
    labels = R["qstruct"]["labels"]
    cues = np.array([int(l.split("_")[0][4:]) for l in labels], float)
    E = np.array([float(l.split("_")[1][1:]) for l in labels])
    Sk = np.array([float(l.split("_")[2][1:]) for l in labels])
    Xc = S - S.mean(0)
    U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U * sv
    var = sv ** 2 / (sv ** 2).sum()
    row = dict(run=run, var_pc1=float(var[0]), var_pc2=float(var[1]))
    for i in (0, 1):
        for name, y in (("value", v), ("cues", cues), ("energy", E), ("integrity", Sk)):
            row[f"r_pc{i + 1}_{name}"] = float(np.corrcoef(pcs[:, i], y)[0, 1])
    # the best single axis for value (valence) and for threat (arousal), and their angle
    wv = np.linalg.lstsq(Xc, v - v.mean(), rcond=None)[0]
    wc = np.linalg.lstsq(Xc, cues - cues.mean(), rcond=None)[0]
    row["cos_valence_arousal"] = float(wv @ wc / (np.linalg.norm(wv) * np.linalg.norm(wc) + 1e-12))
    row["pcs"] = pcs[:, :2].tolist()
    row["value"] = v.tolist()
    row["labels"] = labels
    circ.append(row)
if circ:
    OUT["valence_arousal"] = {k: ms([abs(r[k]) if k.startswith("r_") else r[k] for r in circ])
                              for k in circ[0] if k not in ("run", "pcs", "value", "labels")}

# ---------------------------------------------------------------- 3. temperament
feat = ("pulse.4.hl_supp", "pulse.4.after_supp", "perturb.on_hl", "jac.top_tc.0", "pulse.4.hl_proj")


def g(R, path):
    x = R
    for p in path.split("."):
        x = x[int(p)] if isinstance(x, list) else x[p]
    return float(x)


rows = []
for k, Rs in PROBES.items():
    if k.startswith(("score", "oracle", "metacog")) or len(Rs) < 3:
        continue
    M = np.array([[g(R, p) for p in feat] for R in Rs], float)
    ok = np.all(np.isfinite(M), 1)
    M = M[ok]
    if len(M) < 3:
        continue
    Z = (M - M.mean(0)) / (M.std(0) + 1e-12)
    for z in Z:
        rows.append(z)
Z = np.array(rows)
temper = {}
if len(Z) > 5:
    for j, p in enumerate(feat[2:], start=2):
        for i, b in enumerate(feat[:2]):
            r, pv = st.pearsonr(Z[:, i], Z[:, j])
            temper[f"{b} vs {p}"] = dict(r=float(r), p=float(pv), n=len(Z))
            T[f"temperament: {b} vs {p} (within-condition z)"] = temper[f"{b} vs {p}"]
OUT["temperament"] = temper

# ---------------------------------------------------------------- 4. racing heart
heart = defaultdict(list)
for f in glob.glob("runs/*/heart.json"):
    meta = json.load(open(os.path.join(os.path.dirname(f), "log.json")))
    k = cond_key(meta)
    if k:
        heart[k].append(json.load(open(f)))
H = {}
for k, Rs in heart.items():
    H[k] = dict(peak_dH=ms([R["normal"]["peak_dH"] for R in Rs]), base_H=ms([R["normal"]["base_H"] for R in Rs]),
                hl_proj_normal=ms([R["normal"]["hl_proj"] for R in Rs]), hl_proj_blocked=ms([R["blocked"]["hl_proj"] for R in Rs]),
                auc_proj_normal=ms([R["normal"]["auc_proj"] for R in Rs]), auc_proj_blocked=ms([R["blocked"]["auc_proj"] for R in Rs]),
                auc_supp_normal=ms([R["normal"]["auc_supp"] for R in Rs]), auc_supp_blocked=ms([R["blocked"]["auc_supp"] for R in Rs]),
                false_heart_proj=ms([R["false_heart"]["peak_proj"] for R in Rs]), false_heart_supp=ms([R["false_heart"]["peak_supp"] for R in Rs]))
    T[f"heart {k}: threat state with heart sensation blocked vs normal (area)"] = paired(
        [R["blocked"]["auc_proj"] for R in Rs], [R["normal"]["auc_proj"] for R in Rs])
    T[f"heart {k}: hiding with heart sensation blocked vs normal (area)"] = paired(
        [R["blocked"]["auc_supp"] for R in Rs], [R["normal"]["auc_supp"] for R in Rs])
    T[f"heart {k}: false racing heart raises threat state"] = ttest1([R["false_heart"]["peak_proj"] for R in Rs])
    T[f"heart {k}: false racing heart causes hiding"] = ttest1([R["false_heart"]["peak_supp"] for R in Rs])
    T[f"heart {k}: heart-rate rise after cues"] = ttest1([R["normal"]["peak_dH"] for R in Rs])
OUT["heart"] = H

# ---------------------------------------------------------------- 5. empathy
soc = defaultdict(list)
for f in glob.glob("runs/*/social.json"):
    meta = json.load(open(os.path.join(os.path.dirname(f), "log.json")))
    k = cond_key(meta)
    if k:
        soc[k].append(json.load(open(f)))
EMP = {}
for k, Rs in soc.items():
    EMP[k] = {f"{ev}_{m}": ms([R[ev][m] for R in Rs]) for ev in ("own_pain", "neighbour_pain", "cue")
              for m in ("peak_threat", "min_integrity_estimate", "peak_hide", "hl_threat")}
    EMP[k].update({f"overlap_{m}": ms([R["overlap"][m] for R in Rs]) for m in Rs[0]["overlap"]})
    T[f"empathy {k}: neighbour's pain raises own threat state"] = ttest1([R["neighbour_pain"]["peak_threat"] for R in Rs])
    T[f"empathy {k}: neighbour's pain makes it hide"] = ttest1([R["neighbour_pain"]["peak_hide"] for R in Rs])
    T[f"empathy {k}: own skin estimate, own pain vs neighbour pain"] = paired(
        [R["own_pain"]["min_integrity_estimate"] for R in Rs], [R["neighbour_pain"]["min_integrity_estimate"] for R in Rs])
    T[f"empathy {k}: overlap own-neighbour vs own-cue"] = paired(
        [R["overlap"]["own_neighbour"] for R in Rs], [R["overlap"]["own_cue"] for R in Rs])
OUT["empathy"] = EMP

# ---------------------------------------------------------------- 6. anxiety and exposure therapy
th = defaultdict(list)
for f in glob.glob("runs_therapy/*/therapy.json"):
    R = json.load(open(f))
    th[f"to{R['target_dmg']}"].append(R)
TH = {}
for k, Rs in th.items():
    its = [r["iteration"] for r in Rs[0]["rows"]]
    TH[k] = {m: [ms([R["rows"][i][m] for R in Rs]) for i in range(len(its))]
             for m in ("peak_supp", "after_supp", "forage_in_danger", "life")}
    TH[k]["iterations"] = its
    T[f"therapy {k}: fear after vs before (peak drop in foraging)"] = paired(
        [R["rows"][-1]["peak_supp"] for R in Rs], [R["rows"][0]["peak_supp"] for R in Rs])
    T[f"therapy {k}: foraging during real danger after vs before"] = paired(
        [R["rows"][-1]["forage_in_danger"] for R in Rs], [R["rows"][0]["forage_in_danger"] for R in Rs])
native = PROBES.get("mortal_tau12_d0.05", [])
if native:
    TH["native_0.05_peak_supp"] = ms([R["pulse"]["4"]["peak_supp"] for R in native])
OUT["therapy"] = TH
OUT["tests"] = T


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


json.dump(clean(OUT), open("results/stage4.json", "w"), indent=1)


def fmt(d):
    return "n/a" if not isinstance(d, dict) or "mean" not in d or not np.isfinite(d["mean"]) else f"{d['mean']:.3f} ± {d['sem']:.3f} (n={d['n']})"


with open("results/stage4.md", "w") as f:
    for sec in ("usefulness", "valence_arousal", "temperament", "heart", "empathy", "therapy"):
        f.write(f"\n## {sec}\n")
        val = OUT.get(sec, {})
        for k, v in val.items():
            if isinstance(v, dict) and "mean" in v:
                f.write(f"- {k}: {fmt(v)}\n")
            elif isinstance(v, dict):
                f.write(f"- {k}: " + ", ".join(f"{a} {fmt(b) if isinstance(b, dict) and 'mean' in b else b}" for a, b in v.items()) + "\n")
            elif isinstance(v, list):
                f.write(f"- {k}: " + ", ".join(fmt(x) if isinstance(x, dict) else str(x) for x in v) + "\n")
    f.write("\n## Tests\n")
    for k, v in T.items():
        f.write(f"- {k}: " + ("n/a" if not v else ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items())) + "\n")

# ---------------------------------------------------------------- figures
if circ:
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    keys = sorted(k for k in use if "ratio" in use[k])
    axs[0].barh(range(len(keys)), [use[k]["ratio"]["mean"] for k in keys], xerr=[use[k]["ratio"]["sem"] for k in keys], color=RED)
    axs[0].set_yticks(range(len(keys)), keys)
    axs[0].axvline(0, color="k", lw=0.8)
    axs[0].set_xlabel("log10 lifetime lost when the threat state\nis removed (relative to a random state)")
    r = circ[0]
    P = np.array(r["pcs"])
    sgn = np.sign(np.corrcoef(P[:, 0], r["value"])[0, 1]) or 1
    cues = [int(l.split("_")[0][4:]) for l in r["labels"]]
    E = [float(l.split("_")[1][1:]) for l in r["labels"]]
    sc = axs[1].scatter(sgn * P[:, 0], P[:, 1], c=r["value"], cmap="RdYlGn", s=[20 + 60 * e for e in E],
                        edgecolors=["k" if c == 0 else ("#e67e22" if c == 1 else "#c0392b") for c in cues], linewidths=1.2)
    plt.colorbar(sc, ax=axs[1], label="value (expected survival)")
    axs[1].set_xlabel("internal axis 1")
    axs[1].set_ylabel("internal axis 2")
    axs[1].set_title("18 conditions (size: energy; edge: threat cues)", fontsize=8)
    fig.savefig("figures/fig16_usefulness_valence.png")
    plt.close(fig)

hk = [k for k in H]
if hk:
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.8))
    for k, c in zip(sorted(hk), (RED, BLUE, GREEN, ORANGE)):
        Rs = heart[k]
        t = np.arange(-5, len(Rs[0]["normal"]["dH"]) - 5)
        axs[0].plot(t, np.mean([R["normal"]["dH"] for R in Rs], 0), color=c, label=k)
        axs[1].plot(t, np.mean([R["normal"]["dproj"] for R in Rs], 0), color=c, label=f"{k}, heart felt")
        axs[1].plot(t, np.mean([R["blocked"]["dproj"] for R in Rs], 0), color=c, ls="--", label=f"{k}, heart blocked")
        tf = np.arange(-5, len(Rs[0]["false_heart"]["dproj"]) - 5)
        axs[2].plot(tf, np.mean([R["false_heart"]["dproj"] for R in Rs], 0), color=c, label=k)
    axs[0].set_ylabel("heart-rate rise")
    axs[1].set_ylabel("threat state (SD)")
    axs[2].set_ylabel("threat state (SD)")
    axs[0].set_title("Heart response to 4 cues", fontsize=8)
    axs[1].set_title("Fear with heart sensation blocked", fontsize=8)
    axs[2].set_title("False racing heart, no cue", fontsize=8)
    for a in axs:
        a.set_xlabel("steps")
        a.set_xlim(-5, 25)
    axs[1].legend(frameon=False, fontsize=6)
    fig.savefig("figures/fig17_heart.png")
    plt.close(fig)

if soc:
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.8))
    k = "social_tau12" if "social_tau12" in soc else sorted(soc)[0]
    Rs = soc[k]
    t = np.arange(-2, len(Rs[0]["cue"]["threat"]) - 2)
    for ev, c, lab_ in (("own_pain", RED, "own pain"), ("neighbour_pain", PURPLE, "neighbour's pain"), ("cue", GREY, "danger cue")):
        axs[0].plot(t, np.mean([R[ev]["threat"] for R in Rs], 0), color=c, label=lab_)
        axs[1].plot(t, np.mean([R[ev]["integrity"] for R in Rs], 0), color=c, label=lab_)
        axs[2].plot(t, np.mean([R[ev]["hide"] for R in Rs], 0), color=c, label=lab_)
    axs[0].set_ylabel("own threat state (SD)")
    axs[1].set_ylabel("estimate of own skin integrity")
    axs[2].set_ylabel("drop in P(forage)")
    axs[0].legend(frameon=False, fontsize=7)
    for a in axs:
        a.set_xlabel("steps after the event")
        a.axhline(0, color="0.7", lw=0.6)
    fig.savefig("figures/fig18_empathy.png")
    plt.close(fig)

if th:
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    for k, c, lab_ in (("to0.05", GREEN, "moved to a safe world (exposure)"), ("to0.4", GREY, "stays in fragile world (control)")):
        if k not in TH:
            continue
        its = TH[k]["iterations"]
        for ax, m in zip(axs, ("peak_supp", "forage_in_danger")):
            ax.errorbar(its, [x["mean"] for x in TH[k][m]], [x["sem"] for x in TH[k][m]], color=c, marker="o", ms=3, capsize=2, label=lab_)
    if "native_0.05_peak_supp" in TH:
        axs[0].axhline(TH["native_0.05_peak_supp"]["mean"], color=RED, ls=":", label="agents raised in the safe world")
    axs[0].set_ylabel("fear: drop in P(forage) after cues")
    axs[1].set_ylabel("P(forage) during real danger")
    for a in axs:
        a.set_xlabel("updates of continued learning")
    axs[0].legend(frameon=False, fontsize=6)
    fig.savefig("figures/fig19_therapy.png")
    plt.close(fig)

if len(Z) > 5:
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.scatter(Z[:, 3], Z[:, 0], s=10, color=RED)
    r = temper.get("pulse.4.hl_supp vs jac.top_tc.0", {})
    ax.set_xlabel("slowest network time constant (z)")
    ax.set_ylabel("behavioural persistence (z)")
    ax.set_title(f"r = {r.get('r', float('nan')):.2f}, n = {r.get('n', 0)}", fontsize=8)
    fig.savefig("figures/fig20_temperament.png")
    plt.close(fig)
print("wrote results/stage4.json, results/stage4.md and figures 16-20")
