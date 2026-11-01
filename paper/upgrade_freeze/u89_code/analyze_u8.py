"""Stage U8 analysis (HYPOTHESES.md, 31 Oct 2026): the stakes sign in the decision, and the stakes law.
Reads runs/*/u8.json (vw/u8.py). Run once, after all measures exist.
python analyze_u8.py [root]     (root defaults to runs; a fake tree is used only to test this script)
-> results/u8.json, results/u8.md (only when root is runs)
"""
import glob
import json
import sys

import numpy as np
from scipy import stats as st

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
Q0, Q1, P_ATT = 0.03, 0.35, 0.6          # cue rates in safety and danger; attack probability at heart rate 0
BOUND = 1.3                              # twins' equivalence bound (steps): 25% of the pilot agents' +5.2
SEED, NBOOT = 20261031, 10000


def lam_f(tau):
    """Ideal observer's per-step fall in the log-odds of danger while foraging unharmed with the heart at rest."""
    return -np.log(1 - 1 / tau) + np.log((1 - Q0) / (1 - Q1)) - np.log(1 - P_ATT)


def load(pats):
    return [json.load(open(f)) for p in pats for f in sorted(glob.glob(f"{ROOT}/{p}/u8.json"))]


AG = load(["mortal_tau4_s[0-9]", "mortal_tau12_s[0-9]", "mortal_tau36_s[0-9]"])
TW = load(["twinbelief_tau4_s[0-9]", "twinbelief_tau12_s[0-9]", "twinbelief_tau36_s[0-9]"])
SW = load(["mortal_tau12_d0.05_s[0-9]", "mortal_tau12_d0.15_s[0-9]", "mortal_tau12_d0.4_s[0-9]"])


def diffs(R, clamp="forage", key="D"):
    out, status = [], {"always hides (healthy)": 0, "always hides (wounded)": 0, "capped": 0}
    for r in R:
        h, w = r[clamp]["healthy"], r[clamp]["wounded"]
        for b, x in (("healthy", h), ("wounded", w)):
            if key == "D" and x["D_status"] == "always hides":
                status[f"always hides ({b})"] += 1
            if key == "D" and x["D_status"] == "capped":
                status["capped"] += 1
        a, c = h[key], w[key]
        if a is None or c is None or not (np.isfinite(a) and np.isfinite(c)):
            continue
        out.append(dict(diff=c - a, h=a, w=c, tau=r["tau"], dmg=r["dmg"], L_h=r["L_h"], L_w=r["L_w"]))
    return out, status


def ci(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = st.t.ppf(0.975, len(x) - 1) * se
    return dict(mean=float(m), ci=[float(m - h), float(m + h)], t=float(m / se) if se > 0 else float("nan"),
                p=float(2 * st.t.sf(abs(m / se), len(x) - 1)) if se > 0 else float("nan"), n=len(x))


def tost(x, bound):
    x = np.asarray(x, float)
    se, df = x.std(ddof=1) / np.sqrt(len(x)), len(x) - 1
    p_lo = st.t.sf((x.mean() + bound) / se, df)          # H0: mean <= -bound
    p_hi = st.t.cdf((x.mean() - bound) / se, df)          # H0: mean >= +bound
    return dict(mean=float(x.mean()), bound=bound, p=float(max(p_lo, p_hi)), n=len(x))


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj.tolist()


def law(D):
    rows = [d for d in D if d["L_h"] > 0 and d["L_w"] > 0]
    obs = np.array([d["diff"] for d in rows])
    pred = np.array([np.log(d["L_w"] / d["L_h"]) / lam_f(d["tau"]) for d in rows])
    if len(rows) < 3:
        return dict(n=len(rows), outcome="too few networks")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (NBOOT, len(rows)))
    boots = obs[idx].mean(1) / pred[idx].mean(1)
    R = float(obs.mean() / pred.mean())
    lo, hi = (float(np.quantile(boots, q)) for q in (0.025, 0.975))
    if 0.67 <= lo and hi <= 1.5:
        out = "law fits"
    elif lo > 1.5:
        out = "beyond the law"
    elif hi < 0.67:
        out = "below the law"
    else:
        out = "unclear"
    r = st.pearsonr(pred, obs) if len(rows) > 2 and pred.std() > 0 and obs.std() > 0 else (float("nan"), float("nan"))
    return dict(n=len(rows), excluded_nonpositive_L=len(D) - len(rows), R=R, ci=[lo, hi], outcome=out,
                pred_mean=float(pred.mean()), obs_mean=float(obs.mean()), pearson_r=float(r[0]), pearson_p=float(r[1]),
                loss_ratio_median=float(np.median([d["L_w"] / d["L_h"] for d in rows])))


OUT = {"n": dict(agents=len(AG), twins=len(TW), sweep=len(SW)), "lambda_f": {t: float(lam_f(t)) for t in (4.0, 12.0, 36.0)}}
dA, sA = diffs(AG)
dT, sT = diffs(TW)
xa, xt = [d["diff"] for d in dA], [d["diff"] for d in dT]
p1, p2 = ci(xa), st.ttest_ind(xa, xt, equal_var=False)
hp = holm([p1["p"], float(p2.pvalue)])
P = {"P8.1_agents": dict(**p1, p_holm=hp[0], passed=bool(hp[0] < 0.05 and p1["mean"] > 0)),
     "P8.2_agents_vs_twins": dict(diff=float(np.mean(xa) - np.mean(xt)), t=float(p2.statistic), p=float(p2.pvalue),
                                  p_holm=hp[1], passed=bool(hp[1] < 0.05 and np.mean(xa) > np.mean(xt))),
     "P8.3_twins_equivalent": dict(**tost(xt, BOUND))}
P["P8.3_twins_equivalent"]["passed"] = bool(P["P8.3_twins_equivalent"]["p"] < 0.05)
OUT["primary"] = P
OUT["status"] = dict(agents=sA, twins=sT)
OUT["outcome"] = "CONFIRMED" if all(v["passed"] for v in P.values()) else "NOT CONFIRMED"
OUT["twins_ci"] = ci(xt)
OUT["law_agents"], OUT["law_twins"] = law(dA), law(dT)
OUT["by_tau"] = {}
for tau in (4.0, 12.0, 36.0):
    a = [d["diff"] for d in dA if d["tau"] == tau]
    t = [d["diff"] for d in dT if d["tau"] == tau]
    OUT["by_tau"][str(tau)] = dict(agents=ci(a) if len(a) > 1 else None, twins=ci(t) if len(t) > 1 else None,
                                   agents_D_healthy=float(np.mean([d["h"] for d in dA if d["tau"] == tau])) if a else None,
                                   agents_D_wounded=float(np.mean([d["w"] for d in dA if d["tau"] == tau])) if a else None)
hA, hsA = diffs(AG, "hide")
hT, hsT = diffs(TW, "hide")
OUT["hide_clamp"] = dict(agents=ci([d["diff"] for d in hA]) if len(hA) > 1 else None, agents_status=hsA,
                         twins=ci([d["diff"] for d in hT]) if len(hT) > 1 else None, twins_status=hsT)
OUT["axis_area"] = {}
for clamp in ("forage", "hide"):
    aa, _ = diffs(AG, clamp, "area")
    ta, _ = diffs(TW, clamp, "area")
    OUT["axis_area"][clamp] = dict(agents=ci([d["diff"] for d in aa]), twins=ci([d["diff"] for d in ta]),
                                   agents_vs_twins_p=float(st.ttest_ind([d["diff"] for d in aa], [d["diff"] for d in ta],
                                                                        equal_var=False).pvalue))
OUT["hl_at_window_end"] = sum(int(r[c][b]["hl_at_window_end"]) for r in AG + TW + SW for c in ("forage", "hide")
                              for b in ("healthy", "wounded"))
OUT["hl_total"] = 4 * len(AG + TW + SW)
dS, sS = diffs(SW)
d25 = [d for d in dA if d["tau"] == 12.0]
sweep = sorted(dS + d25, key=lambda d: d["dmg"])
OUT["sweep"] = dict(by_damage={str(g): ci([d["diff"] for d in sweep if d["dmg"] == g]) if
                               sum(d["dmg"] == g for d in sweep) > 1 else None for g in (0.05, 0.15, 0.25, 0.4)},
                    status=sS, spearman=[float(x) for x in st.spearmanr([d["dmg"] for d in sweep], [d["diff"] for d in sweep])],
                    law=law(sweep))

if ROOT == "runs":
    json.dump(OUT, open("results/u8.json", "w"), indent=1)
L = ["# U8: the stakes sign in the decision, and the stakes law", "",
     f"networks: agents {len(AG)}, twins {len(TW)}, sweep {len(SW)}; lambda_f {OUT['lambda_f']}", "",
     "Alarm duration D: steps after one warning until ln P(hide) - ln P(forage) falls below 0 (always foraging).", ""]
for k, v in P.items():
    L.append(f"- {k}: " + ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
L += [f"- status: {OUT['status']}", f"- twins: {OUT['twins_ci']}", f"- OUTCOME: {OUT['outcome']}", "",
      f"- stakes law, agents: {OUT['law_agents']}", f"- stakes law, twins: {OUT['law_twins']}", "",
      f"- by tau: {OUT['by_tau']}", f"- always hiding: {OUT['hide_clamp']}", f"- threat-axis area: {OUT['axis_area']}",
      f"- half-lives at the end of the window: {OUT['hl_at_window_end']} of {OUT['hl_total']}",
      f"- vulnerability sweep: {OUT['sweep']}"]
txt = "\n".join(L) + "\n"
if ROOT == "runs":
    open("results/u8.md", "w").write(txt)
print(txt)
