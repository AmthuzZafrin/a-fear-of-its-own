"""Stage U5 analysis (HYPOTHESES.md, Batch 2): does the stakes sign replicate, with the actions fixed, in new agents
(seeds 10-19), and is it absent in their belief-only twins?
python analyze_u5.py  -> results/u5.json, results/u5.md
"""
import glob
import json
import os

import numpy as np
from scipy import stats as st

ROOT = os.environ.get("U5_ROOT", "runs")          # overridden only to test this script on fake data
OUT = os.environ.get("U5_OUT", "results")
os.makedirs(OUT, exist_ok=True)
SESOI = {"hide": 0.25 * 4.51, "forage": 0.25 * 2.30}   # 25% of the original agents' effects (U1b fixed-action control)


def runs(pat):
    return sorted(glob.glob(os.path.join(ROOT, pat)))


def J(d, name):
    return json.load(open(os.path.join(d, name)))


def ci(x):
    x = np.asarray(x, float)
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def fin(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def one(x):
    x = fin(x)
    t, p = st.ttest_1samp(x, 0.0)
    return dict(mean=float(x.mean()), ci=ci(x), t=float(t), p=float(p), n=int(len(x)), all_positive=bool((x > 0).all()))


def tost(x, b):
    x = fin(x)
    p = max(st.ttest_1samp(x, -b, alternative="greater").pvalue, st.ttest_1samp(x, b, alternative="less").pvalue)
    return dict(mean=float(x.mean()), bound=float(b), p=float(p), equivalent=bool(p < 0.05))


AG, TW = runs("rep_mortal_tau12_s1[0-9]"), runs("rep_twinbelief_tau12_s1[0-9]")
OA, OT = [J(d, "u5.json") for d in AG], [J(d, "u5.json") for d in TW]


def diff(O, pol, key="hl", body="wounded"):
    return np.array([o[pol][body][key] - o[pol]["healthy"][key] for o in O], float)


R = dict(n_agents=len(AG), n_twins=len(TW))
P = {}
for pol in ("hide", "forage"):
    da, dt = diff(OA, pol), diff(OT, pol)
    t, p = st.ttest_ind(fin(da), fin(dt), equal_var=False)
    P[pol] = dict(agents=one(da), twins=one(dt), agents_vs_twins=dict(diff=float(fin(da).mean() - fin(dt).mean()), t=float(t), p=float(p)),
                  twins_equivalence=tost(dt, SESOI[pol]),
                  hl={g: {b: float(np.nanmean([o[pol][b]["hl"] for o in O])) for b in ("healthy", "wounded", "hungry")}
                      for g, O in (("agents", OA), ("twins", OT))})
# primary: always hide; Holm over P1 and P2
p1, p2 = P["hide"]["agents"]["p"], P["hide"]["agents_vs_twins"]["p"]
order = sorted([("P1", p1), ("P2", p2)], key=lambda kv: kv[1])
run_, holm = 0.0, {}
for i, (k, p) in enumerate(order):
    run_ = max(run_, min(1.0, (2 - i) * p))
    holm[k] = run_
prim = dict(P1=dict(test="agents: fixed-hide half-life longer when wounded", p_holm=holm["P1"],
                    passed=bool(holm["P1"] < 0.05 and P["hide"]["agents"]["mean"] > 0)),
            P2=dict(test="agents' difference exceeds the twins'", p_holm=holm["P2"],
                    passed=bool(holm["P2"] < 0.05 and P["hide"]["agents_vs_twins"]["diff"] > 0)),
            P3=dict(test="twins' difference equivalent to zero (+-25% of the original agents' effect)",
                    p=P["hide"]["twins_equivalence"]["p"], passed=P["hide"]["twins_equivalence"]["equivalent"]))
R["primary"] = prim
R["replicated"] = bool(all(v["passed"] for v in prim.values()))
R["by_policy"] = P
# secondary: free behaviour (as the original U1b T2), the slowdown control, the twins' classic signs
U_A, U_T = [J(d, "u1b.json") for d in AG], [J(d, "u1b.json") for d in TW]
fa = np.array([u["T1T2"]["wounded"]["hl"] - u["T1T2"]["healthy"]["hl"] for u in U_A], float)
ft = np.array([u["T1T2"]["wounded"]["hl"] - u["T1T2"]["healthy"]["hl"] for u in U_T], float)
t, p = st.ttest_ind(fin(fa), fin(ft), equal_var=False)
R["free_behaviour"] = dict(agents=one(fa), twins=one(ft), agents_vs_twins_p=float(p))
A4 = [J(d, "u4a.json") for d in AG]


def lr(r, s):
    h, w = r["healthy"][s]["hl"], r["wounded"][s]["hl"]
    return np.log(w / h) if (np.isfinite(h) and np.isfinite(w) and h > 0 and w > 0) else np.nan


sl = {}
for s in ("food", "object", "push_random"):
    x = np.array([lr(r, "threat") - lr(r, s) for r in A4], float)
    x = fin(x)
    sl[s] = dict(mean=float(x.mean()), p=float(st.ttest_1samp(x, 0).pvalue) if len(x) >= 5 else float("nan"), n=int(len(x)))
R["slowdown_control"] = sl
cs = dict(persistence=one([u["persist4"]["after_supp"] for u in U_T]),
          necessity=one([np.log10(u["ablate"]["random_life"] / u["ablate"]["axis_life"]) for u in U_T]),
          hunger=one([u["hunger_forage"]["hungry"] - u["hunger_forage"]["fed"] for u in U_T]))
R["twin_classic_signs"] = {k: dict(v, shown=bool(v["p"] < 0.05 and v["mean"] > 0)) for k, v in cs.items()}
R["life"] = dict(agents=float(np.mean([u["life"] for u in U_A])), twins=float(np.mean([u["life"] for u in U_T])))
json.dump(R, open(os.path.join(OUT, "u5.json"), "w"), indent=1)

f = lambda v: f"{v['mean']:+.2f} {np.round(v['ci'], 2).tolist()} (p={v['p']:.2g}, n={v['n']}, all positive={v['all_positive']})"
L = ["# U5: replication of the stakes sign on new agents (seeds 10-19), actions fixed", "",
     f"agents {len(AG)}, twins {len(TW)}; natural lifetime agents {R['life']['agents']:.0f}, twins {R['life']['twins']:.0f}", ""]
for pol in ("hide", "forage"):
    v = P[pol]
    L.append(f"- always {pol}: half-life healthy→wounded agents {v['hl']['agents']['healthy']:.2f}→{v['hl']['agents']['wounded']:.2f}, "
             f"twins {v['hl']['twins']['healthy']:.2f}→{v['hl']['twins']['wounded']:.2f}")
    L.append(f"  agents {f(v['agents'])}; twins {f(v['twins'])}; agents vs twins p={v['agents_vs_twins']['p']:.2g}; "
             f"twins equivalent to zero (±{v['twins_equivalence']['bound']:.2f}): p={v['twins_equivalence']['p']:.2g} → {v['twins_equivalence']['equivalent']}")
for k, v in prim.items():
    L.append(f"- {k} ({v['test']}): " + (f"Holm p={v['p_holm']:.2g}" if "p_holm" in v else f"p={v['p']:.2g}") + f", pass={v['passed']}")
L += [f"- REPLICATED: {R['replicated']}", "",
      f"- free behaviour: agents {f(R['free_behaviour']['agents'])}; twins {f(R['free_behaviour']['twins'])}; vs p={R['free_behaviour']['agents_vs_twins_p']:.2g}",
      f"- slowdown control (threat minus other, log ratio): {sl}",
      f"- twins' classic signs: " + "; ".join(f"{k} {v['mean']:+.3f} (p={v['p']:.2g}) shown={v['shown']}" for k, v in R['twin_classic_signs'].items())]
open(os.path.join(OUT, "u5.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
