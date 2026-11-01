"""Upgrade stage U1.1 analysis (HYPOTHESES.md): belief versus fear, pooled over the 40 Mortal agents
(d = 0.05, 0.15, 0.25, 0.40; seeds 0-9), Mortal tau 12 alone as secondary, untrained nulls.
python analyze_u1.py  -> results/u1_beliefstakes.json, results/u1_beliefstakes.md
"""
import glob
import json
import os

import numpy as np
from scipy import stats as st

os.makedirs("results", exist_ok=True)
PRED = {"L": +1, "LxS": -1, "LxE": +1}          # predicted signs (HYPOTHESES.md, U1.1)


def load(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        r = json.load(open(p))
        if r["seed"] <= 9:
            out.append(r)
    return out


def ci(x):
    x = np.asarray(x, float)
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def tests(R, label):
    res = {"n": len(R)}
    for model in ("F", "T"):
        res[model] = {}
        for k in ("L", "E", "S", "LxE", "LxS"):
            x = [r[model][k] for r in R]
            t, p = st.ttest_1samp(x, 0.0)
            res[model][k] = dict(mean=float(np.mean(x)), ci=ci(x), t=float(t), p=float(p))
        # Holm correction over the two interaction tests
        ps = sorted((res[model][k]["p"], k) for k in ("LxE", "LxS"))
        m = len(ps)
        adj, run = {}, 0.0
        for i, (p, k) in enumerate(ps):
            run = max(run, min(1.0, (m - i) * p))
            adj[k] = run
        for k in adj:
            res[model][k]["p_holm"] = adj[k]
            res[model][k]["predicted_sign_ok"] = bool(np.sign(res[model][k]["mean"]) == PRED[k])
    for k in ("binmatched_S", "binmatched_E"):
        x = [r[k] for r in R if np.isfinite(r[k])]
        t, p = st.ttest_1samp(x, 0.0)
        res[k] = dict(mean=float(np.mean(x)), ci=ci(x), t=float(t), p=float(p), n=len(x))
    return res


OUT = {}
main = [r for d in ("runs/mortal_tau12_s*", "runs/mortal_tau12_d*_s*") for r in load(d + "/beliefstakes.json")]
OUT["pooled_40"] = tests(main, "pooled")
OUT["mortal_tau12"] = tests([r for r in main if abs(r["dmg"] - 0.25) < 1e-9], "mortal")
OUT["untrained"] = tests(load("runs/mortal_tau12_s*/beliefstakes_untrained.json"), "untrained")
slopes = np.array([r["F_slope_raw"] for r in main])
dmg = np.array([r["dmg"] for r in main])
r, p = st.pearsonr(dmg, slopes)
z, se = np.arctanh(r), 1 / np.sqrt(len(slopes) - 3)
OUT["slope_vs_vulnerability"] = dict(r=float(r), p=float(p), n=len(slopes), ci=[float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
                                     by_d={str(d): float(slopes[dmg == d].mean()) for d in sorted(set(dmg))})
P = OUT["pooled_40"]["F"]
inter_ok = any(P[k]["p_holm"] < 0.05 and P[k]["predicted_sign_ok"] for k in ("LxE", "LxS"))
OUT["pass"] = dict(interaction=bool(inter_ok), slope_vulnerability=bool(r > 0 and p < 0.05),
                   fear_weighs_stakes=bool(inter_ok and r > 0 and p < 0.05))
json.dump(OUT, open("results/u1_beliefstakes.json", "w"), indent=1)

lines = ["# U1.1 belief versus fear", ""]
for name in ("pooled_40", "mortal_tau12", "untrained"):
    R = OUT[name]
    lines.append(f"## {name} (n = {R['n']})")
    for model in ("F", "T"):
        lines.append(f"- {model}: " + "; ".join(
            f"{k} {R[model][k]['mean']:+.3f} [{R[model][k]['ci'][0]:+.3f}, {R[model][k]['ci'][1]:+.3f}] p={R[model][k]['p']:.2g}"
            + (f" (Holm {R[model][k]['p_holm']:.2g})" if "p_holm" in R[model][k] else "") for k in ("L", "E", "S", "LxE", "LxS")))
    for k in ("binmatched_S", "binmatched_E"):
        lines.append(f"- {k}: {R[k]['mean']:+.4f} [{R[k]['ci'][0]:+.4f}, {R[k]['ci'][1]:+.4f}] p={R[k]['p']:.2g}")
s = OUT["slope_vs_vulnerability"]
lines += ["", f"## slope of F on L versus vulnerability: r = {s['r']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}], p = {s['p']:.2g}, n = {s['n']}",
          "- mean slope by d: " + ", ".join(f"{d}: {v:.3f}" for d, v in s["by_d"].items()),
          "", f"## pass: {OUT['pass']}"]
open("results/u1_beliefstakes.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
