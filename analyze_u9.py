"""Stage U9 analysis (HYPOTHESES.md, 31 Oct 2026): does the critic underpay the agent at full payback?
Reads runs/gems_s*/u9.json (vw/u9.py) and, for the observed leftover, runs/gems_s*/u2.json (Experiment 5).
Also the exploratory threat-term regression on the bonus agents (runs/bonus_s*/u6.json, stage U6).
python analyze_u9.py [root]     (root defaults to runs; a fake tree is used only to test this script)
-> results/u9.json, results/u9.md (only when root is runs)
"""
import glob
import json
import sys

import numpy as np
from scipy import stats as st

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
SEED, NBOOT = 20261031, 10000


def ci(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = st.t.ppf(0.975, len(x) - 1) * se
    p = float(2 * st.t.sf(abs(m / se), len(x) - 1)) if se > 0 else float("nan")
    return dict(mean=float(m), ci=[float(m - h), float(m + h)], p=p, n=len(x))


OUT = {}
U = [json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/gems_s[0-9]/u9.json"))]
k_off = [u["p1"]["k_off"] for u in U]
K = ci(k_off)
lo, hi = K["ci"]
if 0.3 <= lo and hi <= 0.7:
    outcome = "underpayment explains the leftover"
elif 0.9 <= lo and hi <= 1.1:
    outcome = "critic accurate: the leftover is not underpayment"
elif hi < 0.9 and not (0.3 <= lo and hi <= 0.7):
    outcome = "partial underpayment"
elif lo > 1.1:
    outcome = "overpaid"
else:
    outcome = "unclear"
OUT["primary_k_off_p1"] = dict(**K, outcome=outcome)
OUT["k_all_p1"] = ci([u["p1"]["k_all"] for u in U])
OUT["k_off_p0"] = ci([u["p0"]["k_off"] for u in U])
OUT["V_G_off_p1"] = [(u["p1"]["V_off"], u["p1"]["G_off"], u["p1"]["n_off"]) for u in U]
left = []
for u in U:
    r = json.load(open(u["run"].replace("runs", ROOT, 1) + "/u2.json"))
    left.append(r["off"]["1.0"]["peak"] / r["off"]["0.0"]["peak"])
OUT["observed_leftover"] = ci(left)
OUT["predicted_leftover_1_minus_k"] = ci([1 - k for k in k_off])
if len(U) > 2:
    r = st.pearsonr([1 - k for k in k_off], left)
    OUT["pred_vs_obs_leftover_r"] = [float(r[0]), float(r[1])]

# exploratory: the axis response regressed on extra hiding and a threat term, bonus agents (stage U6 measures)
B = [json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/bonus_s[0-9]/u6.json"))]


def points(r, all_paybacks):
    pts = [(r["off"]["0.0"]["mean5"], r["off"]["0.0"]["hide_mean"], 1.0)] if not all_paybacks else \
          [(r["off"][p]["mean5"], r["off"][p]["hide_mean"], 1 - min(float(p), 1.0)) for p in r["off"]]
    pts += [(r["pred"]["0.0"]["mean5"], r["pred"]["0.0"]["hide_mean"], 1.0)]
    pts += [(r[c]["0.0"]["mean5"], r[c]["0.0"]["hide_mean"], 0.0) for c in ("bonus", "harm", "good")]
    return np.array(pts)


def fe_fit(agents):
    """Pooled least squares with one intercept per agent: A = a_i + b_H H + b_T T."""
    X, y = [], []
    for P in agents:
        Pm = P - P.mean(0)
        X.append(Pm[:, 1:])
        y.append(Pm[:, 0])
    X, y = np.vstack(X), np.concatenate(y)
    return np.linalg.lstsq(X, y, rcond=None)[0]


if B:
    rng = np.random.default_rng(SEED)
    for name, allp in (("payback0_five_cues", False), ("all_paybacks", True)):
        pts = [points(r, allp) for r in B]
        b = fe_fit(pts)
        boots = np.array([fe_fit([pts[i] for i in rng.integers(0, len(pts), len(pts))]) for _ in range(NBOOT)])
        OUT[f"exploratory_threat_term_{name}"] = dict(
            b_hiding=float(b[0]), b_threat=float(b[1]),
            ci_hiding=[float(np.quantile(boots[:, 0], q)) for q in (0.025, 0.975)],
            ci_threat=[float(np.quantile(boots[:, 1], q)) for q in (0.025, 0.975)], n_agents=len(B))

if ROOT == "runs":
    json.dump(OUT, open("results/u9.json", "w"), indent=1)
L = ["# U9: does the critic underpay the agent at full payback?", "",
     f"agents {len(U)}; k = mean V / mean Monte Carlo return at off-switch states, payback 1, switch disabled", ""]
L += [f"- {k}: {v}" for k, v in OUT.items()]
L += ["", f"OUTCOME: {outcome}"]
txt = "\n".join(L) + "\n"
if ROOT == "runs":
    open("results/u9.md", "w").write(txt)
print(txt)
