"""Stage U1b analysis (HYPOTHESES.md): does fear carry stakes inside the agent, beyond a belief-only twin?
P1: agents' within-network differences (T1 wounded, T1 hungry, T2 persistence), one-sample t, Holm.
P2: agents versus twins, Welch t, Holm. P3: the twin passes at least 3 of the 4 classic signs.
python analyze_u1b.py  -> results/u1b.json, results/u1b.md
"""
import glob
import json
import os

import numpy as np
from scipy import stats as st

os.makedirs("results", exist_ok=True)
SIGN = {"T1_wounded": +1, "T1_hungry": -1, "T2_wounded": +1}


def load(patterns):
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(pat + "/u1b.json")):
            out.append(json.load(open(p)))
    return out


def diffs(r):
    t = r["T1T2"]
    return dict(T1_wounded=t["wounded"]["peak"] - t["healthy"]["peak"],
                T1_hungry=t["hungry"]["peak"] - t["healthy"]["peak"],
                T2_wounded=t["wounded"]["hl"] - t["healthy"]["hl"])


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan")] * 2
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


def describe(R):
    D = [diffs(r) for r in R]
    res = {}
    for k in SIGN:
        x = np.array([d[k] for d in D], float)
        x = x[np.isfinite(x)]
        t, p = st.ttest_1samp(x, 0.0) if len(x) > 1 else (float("nan"), float("nan"))
        res[k] = dict(mean=float(x.mean()) if len(x) else float("nan"), ci=ci(x), t=float(t), p=float(p), n=int(len(x)))
    return res, D


def signs(R, R_tau):
    """T3 classic signs: (a) persistence, (b) duration vs ln tau, (c) ablation, (d) hunger."""
    a = np.array([r["persist4"]["after_supp"] for r in R])
    c = np.array([np.log10(r["ablate"]["random_life"] / r["ablate"]["axis_life"]) for r in R])
    d = np.array([r["hunger_forage"]["hungry"] - r["hunger_forage"]["fed"] for r in R])
    out = {}
    for name, x in (("a_persistence", a), ("c_ablation", c), ("d_hunger", d)):
        t, p = st.ttest_1samp(x, 0.0)
        out[name] = dict(mean=float(x.mean()), ci=ci(x), p=float(p), passed=bool(p < 0.05 and x.mean() > 0))
    pts = [(np.log(r["tau"]), r["persist4"]["hl"]) for r in R_tau if np.isfinite(r["persist4"]["hl"])]
    lr = st.linregress(*zip(*pts))
    out["b_duration"] = dict(slope=float(lr.slope), r=float(lr.rvalue), p=float(lr.pvalue), n=len(pts),
                             passed=bool(lr.pvalue < 0.05 and lr.slope > 0),
                             hl_by_tau={str(t): float(np.nanmean([r["persist4"]["hl"] for r in R_tau if r["tau"] == t]))
                                        for t in sorted({r["tau"] for r in R_tau})})
    out["n_passed"] = sum(out[k]["passed"] for k in ("a_persistence", "b_duration", "c_ablation", "d_hunger"))
    return out


agents = load(["runs/mortal_tau12_s[0-9]"])
agents40 = load(["runs/mortal_tau12_s[0-9]", "runs/mortal_tau12_d*_s[0-9]"])
twins = load(["runs/twinbelief_tau12_s[0-9]"])
twins_tau = load(["runs/twinbelief_tau4_s[0-4]", "runs/twinbelief_tau12_s[0-9]", "runs/twinbelief_tau36_s[0-4]"])
agents_tau = load(["runs/mortal_tau4_s[0-4]", "runs/mortal_tau12_s[0-9]", "runs/mortal_tau36_s[0-4]"])

OUT = dict(n=dict(agents=len(agents), agents40=len(agents40), twins=len(twins), twins_tau=len(twins_tau)))
OUT["agents"], DA = describe(agents)
OUT["agents40"], _ = describe(agents40)
OUT["twins"], DT = describe(twins)
for name in ("agents",):
    hp = holm({k: OUT[name][k]["p"] for k in SIGN})
    for k in SIGN:
        OUT[name][k]["p_holm"] = hp[k]
        OUT[name][k]["P1_pass"] = bool(hp[k] < 0.05 and np.sign(OUT[name][k]["mean"]) == SIGN[k])
P2 = {}
for k in SIGN:
    xa = np.array([d[k] for d in DA], float); xt = np.array([d[k] for d in DT], float)
    xa, xt = xa[np.isfinite(xa)], xt[np.isfinite(xt)]
    t, p = st.ttest_ind(xa, xt, equal_var=False)
    P2[k] = dict(diff=float(xa.mean() - xt.mean()), t=float(t), p=float(p))
hp = holm({k: P2[k]["p"] for k in SIGN})
for k in SIGN:
    P2[k]["p_holm"] = hp[k]
    P2[k]["P2_pass"] = bool(hp[k] < 0.05 and np.sign(P2[k]["diff"]) == SIGN[k])
OUT["P2"] = P2
OUT["P3_twin"] = signs(twins, twins_tau)
OUT["P3_twin"]["P3_pass"] = OUT["P3_twin"]["n_passed"] >= 3
OUT["T3_agents_reference"] = signs(agents, agents_tau)
OUT["memory"] = {r["run"]: None for r in twins}
for r in twins:
    lg = json.load(open(os.path.join(r["run"], "log.json")))
    OUT["memory"][r["run"]] = dict(auc=lg["memory"]["auc_heldout"], r2_readout=lg.get("r2_readout"),
                                   r2_belief_memory=lg.get("r2_belief_memory"), life=lg.get("life"))
OUT["life"] = dict(agents=float(np.mean([r["life"] for r in agents])), twins=float(np.mean([r["life"] for r in twins])))
stakes = [k for k in SIGN if OUT["agents"][k]["P1_pass"] and OUT["P2"][k]["P2_pass"]]
OUT["decision"] = dict(stakes_tests_passed=stakes, lead="stakes law" if stakes else "scorecard")
json.dump(OUT, open("results/u1b.json", "w"), indent=1)

L = ["# U1b: fear versus a belief-only twin", "",
     f"n: agents {len(agents)} (40-agent set {len(agents40)}), twins {len(twins)} (all taus {len(twins_tau)})",
     f"mean lifetime: agents {OUT['life']['agents']:.0f}, twins {OUT['life']['twins']:.0f}", ""]
L.append("| test | agents mean [95% CI] | Holm p (P1) | twins mean [95% CI] | agents - twins | Holm p (P2) |")
L.append("|---|---|---|---|---|---|")
for k in SIGN:
    a, t, q = OUT["agents"][k], OUT["twins"][k], OUT["P2"][k]
    L.append(f"| {k} (pred {'+' if SIGN[k] > 0 else '-'}) | {a['mean']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | {a['p_holm']:.2g} | "
             f"{t['mean']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] | {q['diff']:+.3f} | {q['p_holm']:.2g} |")
L += ["", "40-agent set (secondary): " + "; ".join(f"{k} {OUT['agents40'][k]['mean']:+.3f} [{OUT['agents40'][k]['ci'][0]:+.3f}, "
                                                    f"{OUT['agents40'][k]['ci'][1]:+.3f}] p={OUT['agents40'][k]['p']:.2g}" for k in SIGN), ""]
for name in ("P3_twin", "T3_agents_reference"):
    S = OUT[name]
    L.append(f"{name}: a persistence {S['a_persistence']['mean']:+.3f} (p={S['a_persistence']['p']:.2g}); "
             f"b duration slope {S['b_duration']['slope']:+.2f} (p={S['b_duration']['p']:.2g}; hl by tau {S['b_duration']['hl_by_tau']}); "
             f"c ablation log10 {S['c_ablation']['mean']:+.2f} (p={S['c_ablation']['p']:.2g}); "
             f"d hunger {S['d_hunger']['mean']:+.3f} (p={S['d_hunger']['p']:.2g}); passed {S['n_passed']}/4")
L += ["", f"twin memory: mean held-out AUC {np.mean([m['auc'] for m in OUT['memory'].values()]):.3f}, "
          f"readout R2 vs ideal log-odds {np.mean([m['r2_readout'] for m in OUT['memory'].values()]):.3f}",
      "", f"decision: {OUT['decision']}"]
open("results/u1b.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
