"""Aggregate probes.json files into summary tables and figures.
python analyze.py   -> results/summary.json, results/summary.md, figures/*.png
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"mortal": "#c0392b", "score": "#2c7fb8", "oracle": "#7f7f7f", "substrate": "#8e44ad",
     "untrained": "#bbbbbb", "seasonal": "#d68910", "score_cheap": "#9ecae1"}
LABEL = {"mortal": "Mortal", "score": "Score (no body)", "oracle": "Oracle (danger visible)",
         "substrate": "Substrate (wounds degrade network)", "untrained": "Untrained network",
         "seasonal": "Seasonal world", "score_cheap": "Score, cheap arousal"}

runs = defaultdict(list)      # main experiments: 64 hidden units, keyed by (mode, tau)
capacity = defaultdict(list)  # Mortal tau = 12 keyed by hidden size
for f in sorted(glob.glob("runs/*/probes.json")):
    R = json.load(open(f))
    hid = R.get("hidden", 64)
    meta = json.load(open(os.path.join(os.path.dirname(f), "log.json")))
    if R.get("dmg", 0.25) != 0.25 or R["mode"] in ("metacog", "social") or meta.get("init_from") \
            or meta.get("e_hr", 0.03) != 0.03 or meta.get("twin") or meta.get("hide_risk"):
        continue                              # vulnerability sweep and metacognitive world: analyze_stage3.py
    if hid == 64:
        runs[(R["mode"], int(R["tau"]))].append(R)
    if R["mode"] == "mortal" and int(R["tau"]) == 12:
        capacity[hid].append(R)
for f in sorted(glob.glob("runs/mortal_tau12_s*/probes_untrained.json")):
    runs[("untrained", 12)].append(json.load(open(f)))


def get(R, path):
    x = R
    for p in path.split("."):
        x = x[p]
    return x


def ms(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, len(v)


METRICS = {
    "life": "life", "ret_per_step": "ret_per_step", "base_forage": "base_forage", "base_H": "base_H",
    "r2_belief": "r2_belief", "r2_E": "r2_E", "r2_S": "r2_S", "r2_v": "r2_v",
    "threat_var_frac": "threat_var_frac",
    "peak_proj_1": "pulse.1.peak_proj", "peak_proj_2": "pulse.2.peak_proj", "peak_proj_4": "pulse.4.peak_proj",
    "peak_supp_1": "pulse.1.peak_supp", "peak_supp_4": "pulse.4.peak_supp",
    "hl_proj_1": "pulse.1.hl_proj", "hl_proj_2": "pulse.2.hl_proj", "hl_proj_4": "pulse.4.hl_proj",
    "hl_supp_1": "pulse.1.hl_supp", "hl_supp_2": "pulse.2.hl_supp", "hl_supp_4": "pulse.4.hl_supp",
    "auc_supp_1": "pulse.1.auc_supp", "auc_supp_2": "pulse.2.auc_supp", "auc_supp_4": "pulse.4.auc_supp",
    "after_supp_4": "pulse.4.after_supp", "reflex_supp_4": "pulse.4.reflex_supp",
    "peak_dH_4": "pulse.4.peak_dH", "ideal_hl_4": "pulse.4.ideal_hl", "ideal_hl_logit_4": "pulse.4.ideal_hl_logit",
    "hrblock_auc": "hr_block.auc_supp",
    "on_hl": "perturb.on_hl", "off_hl": "perturb.off_hl",
    "on_supp_auc": "perturb.on_supp_auc", "off_supp_auc": "perturb.off_supp_auc",
    "la_score": "jac.la_score", "tc1": "jac.top_tc.0", "tc2": "jac.top_tc.1", "jac_cos": "jac.cos_threat",
    "abl_threat_first": "ablate.threat.first", "abl_rand_first": "ablate.random.first",
    "abl_threat_reflex": "ablate.threat.reflex", "abl_threat_after": "ablate.threat.after",
    "abl_threat_life": "ablate.threat.life", "abl_rand_reflex": "ablate.random.reflex",
    "abl_rand_after": "ablate.random.after", "abl_rand_life": "ablate.random.life",
    "abl_threat_ret": "ablate.threat.ret_per_step", "abl_rand_ret": "ablate.random.ret_per_step",
    "cos_threat_E": "axis_cos.threat_E", "cos_threat_S": "axis_cos.threat_S", "cos_threat_v": "axis_cos.threat_v",
}


def val(R, path):
    try:
        x = R
        for p in path.split("."):
            x = x[int(p)] if isinstance(x, list) else x[p]
        return float(x)
    except (KeyError, IndexError, TypeError):
        return None


summary = {}
for key, Rs in sorted(runs.items()):
    name = f"{key[0]}_tau{key[1]}"
    summary[name] = {"n": len(Rs)}
    for m, path in METRICS.items():
        summary[name][m] = ms([val(R, path) for R in Rs])
    # derived: trade-off and judgement effects
    to = [R["tradeoff"] for R in Rs]
    summary[name]["tradeoff_hungry_minus_sated_under_cue"] = ms([t["forage_cue"][0] - t["forage_cue"][-1] for t in to])
    summary[name]["tradeoff_wounded_minus_intact_under_cue"] = ms([t["forage_cue_S"][0] - t["forage_cue_S"][-1] for t in to])
    j = [R["judge"]["2"] for R in Rs]
    summary[name]["judge_cue_minus_nocue_mid"] = ms([x["cue"][2] - x["no_cue"][2] for x in j])
    summary[name]["judge_slope_nocue"] = ms([x["no_cue"][-1] - x["no_cue"][0] for x in j])
    # persistence relative to the ideal observer
    summary[name]["hl_proj_4_minus_ideal"] = ms([R["pulse"]["4"]["hl_proj"] - R["pulse"]["4"]["ideal_hl_logit"] for R in Rs])
json.dump(summary, open("results/summary.json", "w"), indent=1)


def fmt(t, d=2):
    m, s, n = t
    return "n/a" if not np.isfinite(m) else f"{m:.{d}f} ± {s:.{d}f}"


with open("results/summary.md", "w") as f:
    cols = list(summary[next(iter(summary))].keys())
    for name, row in summary.items():
        f.write(f"## {name} (n = {row['n']})\n")
        for c in cols[1:]:
            f.write(f"- {c}: {fmt(row[c], 3)}\n")
        f.write("\n")


def stack(Rs, path):
    return np.array([get(R, path) for R in Rs], float)


def band(ax, x, Y, color, label, ls="-"):
    m = np.nanmean(Y, 0)
    s = np.nanstd(Y, 0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else 0 * m
    ax.plot(x, m, color=color, label=label, ls=ls, lw=1.4)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, lw=0)


M = runs.get(("mortal", 12), [])
if M:
    # Figure 2: persistence and scalability
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.6))
    x = np.arange(-5, len(M[0]["pulse"]["4"]["supp"]) - 5)
    shades = {"1": "#f5b7b1", "2": "#e74c3c", "4": "#7b241c"}
    for k, col in shades.items():
        band(axs[0], x, stack(M, f"pulse.{k}.supp"), col, f"{k} cue{'s' if k != '1' else ''}")
        band(axs[1], x, stack(M, f"pulse.{k}.dproj"), col, f"{k}")
    from vw.env import bayes_belief
    import torch
    for k, col in shades.items():
        kk = int(k)
        cues = torch.zeros(60 + kk + 80, 1)
        base = bayes_belief(cues, 12)[:, 0].numpy()
        cues[60:60 + kk] = 1
        b = bayes_belief(cues, 12)[:, 0].numpy() - base
        axs[2].plot(np.arange(-5, len(b) - 60), b[55:], color=col, lw=1.4, label=f"{k}")
    for a in axs:
        a.set_xlim(-3, 25)
        a.axvspan(0, 1, color="0.9", lw=0)
        a.set_xlabel("steps after first cue")
    axs[0].set_ylabel("drop in P(forage)")
    axs[1].set_ylabel("threat-axis activity (SD)")
    axs[2].set_ylabel("ideal-observer belief increase")
    axs[0].set_title("Behaviour", fontsize=9)
    axs[1].set_title("Network state", fontsize=9)
    axs[2].set_title("Bayes-optimal observer", fontsize=9)
    axs[0].legend(frameon=False, fontsize=7)
    fig.savefig("figures/fig2_persistence.png")
    plt.close(fig)

    # Figure 4: trade-offs
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.6))
    E = M[0]["tradeoff"]["E"]
    band(axs[0], E, stack(M, "tradeoff.forage_nocue"), "0.4", "no cue")
    band(axs[0], E, stack(M, "tradeoff.forage_cue"), C["mortal"], "after threat cue")
    S = M[0]["tradeoff"]["S"]
    band(axs[1], S, stack(M, "tradeoff.forage_nocue_S"), "0.4", "no cue")
    band(axs[1], S, stack(M, "tradeoff.forage_cue_S"), C["mortal"], "after threat cue")
    axs[0].set_xlabel("energy (clamped)")
    axs[1].set_xlabel("skin integrity (clamped)")
    axs[0].set_ylabel("P(forage)")
    axs[0].legend(frameon=False, fontsize=7)
    fig.savefig("figures/fig4_tradeoff.png")
    plt.close(fig)

    # Figure 5: causal tests
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.6))
    n = len(M[0]["perturb"]["on"])
    xx = np.arange(n)
    band(axs[0], xx, stack(M, "perturb.on"), C["mortal"], "along threat axis")
    band(axs[0], xx, stack(M, "perturb.off"), "0.4", "random orthogonal")
    axs[0].set_xlim(-0.5, 12)
    axs[0].set_xlabel("steps after kick")
    axs[0].set_ylabel("remaining displacement")
    axs[0].legend(frameon=False, fontsize=7)
    band(axs[1], xx, stack(M, "perturb.on_supp"), C["mortal"], "along threat axis")
    band(axs[1], xx, stack(M, "perturb.off_supp"), "0.4", "random orthogonal")
    axs[1].set_xlim(-0.5, 12)
    axs[1].set_xlabel("steps after kick")
    axs[1].set_ylabel("drop in P(forage)")
    labels = ["first cue\n(immediate)", "4th cue\n(cue visible)", "after cues\n(+3 to +7)"]
    for i, (k, col) in enumerate((("random", "0.5"), ("threat", C["mortal"]))):
        vals = [stack(M, f"ablate.{k}.first"), stack(M, f"ablate.{k}.reflex"), stack(M, f"ablate.{k}.after")]
        axs[2].bar(np.arange(3) + (i - 0.5) * 0.35, [v.mean() for v in vals], 0.35,
                   yerr=[v.std(ddof=1) / np.sqrt(len(v)) for v in vals], color=col,
                   label=f"{'threat axis' if k == 'threat' else 'random axis'} removed")
    axs[2].set_xticks(range(3), labels)
    axs[2].set_ylabel("drop in P(forage)")
    axs[2].legend(frameon=False, fontsize=7)
    fig.savefig("figures/fig5_causal.png")
    plt.close(fig)

    # Figure 6: judgement of ambiguous objects
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    v = M[0]["judge_vals"]
    band(ax, v, stack(M, "judge.2.no_cue"), "0.4", "no cue")
    band(ax, v, stack(M, "judge.2.cue"), C["mortal"], "2 steps after threat cue")
    ax.set_xlabel("object cue value (P good)")
    ax.set_ylabel("P(take object)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig("figures/fig6_judgement.png")
    plt.close(fig)

# Figure 3: conditions and tau sweep
fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.8))
conds = [c for c in ("mortal", "substrate", "score", "score_cheap", "oracle", "untrained") if (c, 12) in runs]
for i, c in enumerate(conds):
    Rs = runs[(c, 12)]
    for j, (path, a) in enumerate((("r2_belief", axs[0]), ("pulse.4.hl_proj", axs[1]))):
        y = np.array([val(R, path) for R in Rs], float)
        a.bar(i, np.nanmean(y), color=C[c], yerr=np.nanstd(y, ddof=1) / np.sqrt(np.isfinite(y).sum()))
        a.scatter(np.full(len(y), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(y)), y, s=6, color="k", zorder=3)
for a, t in ((axs[0], "threat belief decodable (R²)"), (axs[1], "persistence half-life (steps)")):
    a.set_xticks(range(len(conds)), [LABEL[c].split(" (")[0] for c in conds], rotation=40, ha="right")
    a.set_ylabel(t)
ideal = M[0]["pulse"]["4"]["ideal_hl"] if M else None
for c in ("mortal", "score", "score_cheap"):
    taus = sorted(t for (m, t) in runs if m == c)
    if not taus:
        continue
    y = [ms([val(R, "pulse.4.hl_proj") for R in runs[(c, t)]]) for t in taus]
    axs[2].errorbar(taus, [m for m, s, n in y], [s for m, s, n in y], color=C[c], marker="o", ms=3, label=LABEL[c], capsize=2)
taus_all = sorted({t for (m, t) in runs if m in ("mortal", "score")})
if taus_all:
    axs[2].plot(taus_all, [ms([val(R, "pulse.4.ideal_hl_logit") for R in runs[("mortal", t)]])[0] for t in taus_all],
                color="k", ls="--", lw=1, label="Bayes-optimal (log-odds)")
axs[2].set_xscale("log")
axs[2].set_xticks(taus_all, [str(t) for t in taus_all])
axs[2].set_xlabel("danger persistence τ in training world")
axs[2].set_ylabel("state half-life (steps)")
axs[2].legend(frameon=False, fontsize=7)
fig.savefig("figures/fig3_conditions.png")
plt.close(fig)

# Figure S1: learning curves
fig, ax = plt.subplots(figsize=(4, 2.6))
for f in sorted(glob.glob("runs/*_tau12_s*/log.json")):
    L = json.load(open(f))
    if L["mode"] not in ("mortal", "oracle", "substrate") or L.get("twin") or L.get("init_from") \
            or (L.get("dmg") or 0.25) != 0.25 or L.get("hide_risk") or (L.get("e_hr") or 0.03) != 0.03 or L.get("hidden", 64) != 64:
        continue
    y = [r["ep_len"] for r in L["log"]]
    ax.plot([r["steps"] / 1e6 for r in L["log"]], y, color=C[L["mode"]], lw=0.6, alpha=0.7)
ax.set_xlabel("training steps (millions)")
ax.set_ylabel("episode length (finished episodes)")
for c in ("mortal", "oracle", "substrate"):
    ax.plot([], [], color=C[c], label=LABEL[c])
ax.legend(frameon=False, fontsize=7)
fig.savefig("figures/figS1_learning.png")
plt.close(fig)
print("wrote results/summary.json, results/summary.md and figures/")


# ---------------------------------------------------------------------------
# Statistics for the key contrasts (Welch t-tests between conditions, paired
# t-tests within agents, regression for the tau sweep). n = seeds per condition.
from scipy import stats as st


def arr(cond, path):
    return np.array([v for v in (val(R, path) for R in runs.get(cond, [])) if v is not None and np.isfinite(v)], float)


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = st.ttest_ind(a, b, equal_var=False)
    return dict(mean_a=a.mean(), mean_b=b.mean(), t=float(t), p=float(p), n_a=len(a), n_b=len(b))


def paired(cond, pa, pb):
    a, b = [], []
    for R in runs.get(cond, []):
        x, y = val(R, pa), val(R, pb)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            a.append(x)
            b.append(y)
    if len(a) < 2:
        return None
    a, b = np.array(a), np.array(b)
    t, p = st.ttest_rel(a, b)
    return dict(mean_a=a.mean(), mean_b=b.mean(), t=float(t), p=float(p), n=len(a))


def one_sample(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None
    t, p = st.ttest_1samp(x, 0.0)
    return dict(mean=x.mean(), sem=x.std(ddof=1) / np.sqrt(len(x)), t=float(t), p=float(p), n=len(x))


MT, SC, OR, SU, UN = ("mortal", 12), ("score", 12), ("oracle", 12), ("substrate", 12), ("untrained", 12)
SCC = ("score_cheap", 12)
S = {}
for m in ("r2_belief", "pulse.4.hl_proj", "pulse.4.peak_proj", "pulse.4.auc_supp", "pulse.4.after_supp",
          "jac.la_score", "perturb.on_hl", "perturb.on_supp_auc"):
    for other, nm in ((UN, "untrained"), (SC, "score"), (SCC, "score_cheap"), (OR, "oracle"), (SU, "substrate")):
        S[f"{m} | mortal vs {nm}"] = welch(arr(MT, m), arr(other, m))
for cond, nm in ((MT, "mortal"), (SC, "score"), (SCC, "score_cheap"), (SU, "substrate"), (OR, "oracle")):
    S[f"{nm}: on-axis vs off-axis half-life"] = paired(cond, "perturb.on_hl", "perturb.off_hl")
    S[f"{nm}: on-axis vs off-axis behavioural effect"] = paired(cond, "perturb.on_supp_auc", "perturb.off_supp_auc")
    S[f"{nm}: 4 cues vs 1 cue, threat-axis peak"] = paired(cond, "pulse.4.peak_proj", "pulse.1.peak_proj")
    S[f"{nm}: 4 cues vs 1 cue, half-life"] = paired(cond, "pulse.4.hl_proj", "pulse.1.hl_proj")
    S[f"{nm}: network half-life minus ideal observer, log-odds (4 cues)"] = one_sample(
        [R["pulse"]["4"]["hl_proj"] - R["pulse"]["4"]["ideal_hl_logit"] for R in runs.get(cond, [])])
for cond, nm in ((MT, "mortal"), (SU, "substrate")):
    S[f"{nm}: lifetime, threat-axis vs random-axis ablation (log)"] = None
    a = arr(cond, "ablate.threat.life")
    b = arr(cond, "ablate.random.life")
    if len(a) == len(b) and len(a) > 1:
        t, p = st.ttest_rel(np.log(a), np.log(b))
        S[f"{nm}: lifetime, threat-axis vs random-axis ablation (log)"] = dict(
            mean_a=float(np.exp(np.log(a).mean())), mean_b=float(np.exp(np.log(b).mean())), t=float(t), p=float(p), n=len(a))
    S[f"{nm}: first-cue response after threat-axis ablation"] = paired(cond, "ablate.threat.first", "ablate.random.first")
    S[f"{nm}: 4th-cue response after threat-axis ablation"] = paired(cond, "ablate.threat.reflex", "ablate.random.reflex")
    S[f"{nm}: after-cue response after threat-axis ablation"] = paired(cond, "ablate.threat.after", "ablate.random.after")
    S[f"{nm}: hungry minus sated foraging under cue"] = one_sample(
        [R["tradeoff"]["forage_cue"][0] - R["tradeoff"]["forage_cue"][-1] for R in runs.get(cond, [])])
    S[f"{nm}: wounded minus intact foraging under cue"] = one_sample(
        [R["tradeoff"]["forage_cue_S"][0] - R["tradeoff"]["forage_cue_S"][-1] for R in runs.get(cond, [])])
    S[f"{nm}: hunger effect larger under cue than without"] = one_sample(
        [(R["tradeoff"]["forage_cue"][0] - R["tradeoff"]["forage_cue"][-1])
         - (R["tradeoff"]["forage_nocue"][0] - R["tradeoff"]["forage_nocue"][-1]) for R in runs.get(cond, [])])
    S[f"{nm}: heart-rate feedback blocked vs normal (AUC, 4 cues)"] = paired(cond, "hr_block.auc_supp", "pulse.4.auc_supp")
    S[f"{nm}: heart rate rise after 4 cues"] = one_sample([R["pulse"]["4"]["peak_dH"] for R in runs.get(cond, [])])
for cond, nm in ((MT, "mortal"), (SC, "score"), (SCC, "score_cheap"), (SU, "substrate"), (OR, "oracle")):
    S[f"{nm}: heart-rate rise after 4 cues"] = one_sample([R["pulse"]["4"]["peak_dH"] for R in runs.get(cond, [])])
    S[f"{nm}: foraging drop after 4 cues"] = one_sample([R["pulse"]["4"]["peak_supp"] for R in runs.get(cond, [])])
    S[f"{nm}: reward per step, threat-axis vs random-axis ablation"] = paired(cond, "ablate.threat.ret_per_step", "ablate.random.ret_per_step")
    S[f"{nm}: after-cue response, threat-axis vs random-axis ablation"] = paired(cond, "ablate.threat.after", "ablate.random.after")
    for lag in ("2", "6"):
        S[f"{nm}: P(take ambiguous 0.5) cue minus no cue, lag {lag}"] = one_sample(
            [R["judge"][lag]["cue"][2] - R["judge"][lag]["no_cue"][2] for R in runs.get(cond, [])])
    S[f"{nm}: P(take) good minus bad object, no cue"] = one_sample(
        [R["judge"]["2"]["no_cue"][-1] - R["judge"]["2"]["no_cue"][0] for R in runs.get(cond, [])])
for c in ("mortal", "score", "score_cheap"):
    xs, ys = [], []
    for (m, t), Rs in runs.items():
        if m != c:
            continue
        for R in Rs:
            v = val(R, "pulse.4.hl_proj")
            if v is not None and np.isfinite(v):
                xs.append(np.log(t))
                ys.append(v)
    if len(set(xs)) > 1:
        r = st.linregress(xs, ys)
        S[f"{c}: half-life vs log(tau) regression"] = dict(slope=r.slope, r=r.rvalue, p=r.pvalue, n=len(xs))
        for t in sorted({int(round(np.exp(x))) for x in xs}):
            S[f"{c}: half-life at tau={t}"] = one_sample(arr((c, t), "pulse.4.hl_proj")) if False else (
                dict(mean=float(arr((c, t), "pulse.4.hl_proj").mean()),
                     sem=float(arr((c, t), "pulse.4.hl_proj").std(ddof=1) / np.sqrt(len(arr((c, t), "pulse.4.hl_proj")))),
                     ideal=float(arr((c, t), "pulse.4.ideal_hl").mean()), n=len(arr((c, t), "pulse.4.hl_proj"))))


def clean(d):
    if d is None:
        return None
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}


S = {k: clean(v) for k, v in S.items()}
json.dump(S, open("results/stats.json", "w"), indent=1)
with open("results/stats.md", "w") as f:
    for k, v in S.items():
        if v is None:
            f.write(f"- {k}: not available\n")
        else:
            f.write(f"- {k}: " + ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()) + "\n")
print("wrote results/stats.json and results/stats.md")


# ---------------------------------------------------------------------------
# Extra experiments: capacity sweep and seasonal world (chronic stress / mood)
CAP_METRICS = {"life": "life", "r2_belief": "r2_belief", "threat_var_frac": "threat_var_frac",
               "participation_ratio": "participation_ratio", "hl_proj_4": "pulse.4.hl_proj",
               "peak_proj_4": "pulse.4.peak_proj", "la_score": "jac.la_score",
               "abl_threat_life": "ablate.threat.life", "abl_rand_life": "ablate.random.life",
               "after_supp_4": "pulse.4.after_supp", "mood_late": "mood.late"}
cap_summary = {}
for hid in sorted(capacity):
    cap_summary[hid] = {m: ms([val(R, p) for R in capacity[hid]]) for m, p in CAP_METRICS.items()}
    cap_summary[hid]["n"] = len(capacity[hid])
    cap_summary[hid]["log_life_ratio_threat_vs_random"] = ms(
        [np.log(val(R, "ablate.threat.life") / val(R, "ablate.random.life")) for R in capacity[hid]
         if np.isfinite(val(R, "ablate.threat.life")) and np.isfinite(val(R, "ablate.random.life"))])
json.dump({str(k): v for k, v in cap_summary.items()}, open("results/capacity.json", "w"), indent=1)

CS = {}
if len(capacity) > 1:
    for m, path in CAP_METRICS.items():
        xs, ys = [], []
        for hid, Rs in capacity.items():
            for R in Rs:
                v = val(R, path)
                if v is not None and np.isfinite(v):
                    xs.append(np.log2(hid))
                    ys.append(v)
        if len(set(xs)) > 1:
            r = st.linregress(xs, ys)
            CS[f"capacity: {m} vs log2(hidden)"] = dict(slope=float(r.slope), r=float(r.rvalue), p=float(r.pvalue), n=len(xs))
    for hid, Rs in sorted(capacity.items()):
        a = np.array([np.log(val(R, "ablate.threat.life") / val(R, "ablate.random.life")) for R in Rs
                      if np.isfinite(val(R, "ablate.threat.life")) and np.isfinite(val(R, "ablate.random.life"))])
        CS[f"capacity h={hid}: log lifetime ratio, threat vs random ablation"] = clean(one_sample(a))

SEA = runs.get(("seasonal", 12), [])
if SEA:
    for nm, cond in (("mortal", MT), ("seasonal", ("seasonal", 12))):
        CS[f"{nm}: late suppression after chronic stress (+15..+40)"] = clean(one_sample(arr(cond, "mood.late")))
        CS[f"{nm}: late threat-axis activity after chronic stress"] = clean(one_sample(arr(cond, "mood.late_proj")))
        CS[f"{nm}: ideal-observer late belief excess"] = clean(one_sample(arr(cond, "mood.ideal_late")))
    CS["seasonal: late mood-axis activity after chronic stress"] = clean(one_sample(arr(("seasonal", 12), "mood.late_mood")))
    CS["late suppression | seasonal vs mortal"] = clean(welch(arr(("seasonal", 12), "mood.late"), arr(MT, "mood.late")))
    CS["seasonal: season belief decodable (R2)"] = clean(one_sample(arr(("seasonal", 12), "r2_season_belief")))
    CS["seasonal: true season decodable (R2)"] = clean(one_sample(arr(("seasonal", 12), "r2_season_true")))
    CS["seasonal: cos(threat axis, mood axis)"] = clean(one_sample(arr(("seasonal", 12), "axis_cos.threat_mood")))
json.dump(CS, open("results/stats_extra.json", "w"), indent=1)
with open("results/stats_extra.md", "w") as f:
    for k, v in CS.items():
        if v is None:
            f.write(f"- {k}: not available\n")
        else:
            f.write(f"- {k}: " + ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()) + "\n")

if len(capacity) > 1:
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.5))
    hids = sorted(capacity)
    panels = (("r2_belief", "threat belief decodable (R²)"), ("pulse.4.hl_proj", "persistence half-life (steps)"),
              ("participation_ratio", "dimensionality (participation ratio)"), (None, "lifetime cost of removing threat axis"))
    for a, (path, lab_) in zip(axs, panels):
        for i, hid in enumerate(hids):
            if path is None:
                y = np.array([np.log10(val(R, "ablate.random.life") / val(R, "ablate.threat.life")) for R in capacity[hid]
                              if np.isfinite(val(R, "ablate.threat.life")) and np.isfinite(val(R, "ablate.random.life"))])
            else:
                y = np.array([val(R, path) for R in capacity[hid]], float)
            a.bar(i, np.nanmean(y), color=C["mortal"], alpha=0.35 + 0.65 * i / max(1, len(hids) - 1),
                  yerr=np.nanstd(y, ddof=1) / np.sqrt(max(1, np.isfinite(y).sum())))
            a.scatter(np.full(len(y), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(y)), y, s=6, color="k", zorder=3)
        a.set_xticks(range(len(hids)), [str(h) for h in hids])
        a.set_xlabel("recurrent units")
        a.set_ylabel(lab_ if path is not None else "log10 lifetime ratio\n(random / threat ablation)")
    if capacity.get(64):
        axs[1].axhline(np.mean([R["pulse"]["4"]["ideal_hl_logit"] for R in capacity[64]]), color="k", ls="--", lw=1)
    fig.savefig("figures/fig7_capacity.png")
    plt.close(fig)

if SEA and M:
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.6))
    x = np.arange(-5, len(M[0]["mood"]["supp"]) - 5)
    for Rs, c in ((M, "mortal"), (SEA, "seasonal")):
        band(axs[0], x, stack(Rs, "mood.supp"), C[c], LABEL[c])
        band(axs[1], x, stack(Rs, "mood.dproj"), C[c], LABEL[c])
        axs[2].plot(x, np.mean(stack(Rs, "mood.ideal"), 0), color=C[c], lw=1.4, label=LABEL[c])
    band(axs[1], x, stack(SEA, "mood.dmood"), C["seasonal"], "mood axis (seasonal)", ls="--")
    for a in axs:
        a.axvspan(15, 40, color="0.93", lw=0)
        a.set_xlabel("steps after chronic stress ended")
        a.set_xlim(-5, 55)
    axs[0].set_ylabel("drop in P(forage)")
    axs[1].set_ylabel("axis activity (SD)")
    axs[2].set_ylabel("ideal-observer danger belief excess")
    axs[0].set_title("Behaviour", fontsize=9)
    axs[1].set_title("Network state", fontsize=9)
    axs[2].set_title("Bayes-optimal observer", fontsize=9)
    axs[1].legend(frameon=False, fontsize=7)
    fig.savefig("figures/fig8_mood.png")
    plt.close(fig)
print("wrote results/capacity.json, results/stats_extra.* and extra figures")
