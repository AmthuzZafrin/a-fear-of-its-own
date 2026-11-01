"""Stage U4 analysis (HYPOTHESES.md, Batch 1 controls).
U4a: is the wounded slowdown specific to threat?  U4b: is the leftover off-switch alarm acted on at full payback?
U4c: exploratory re-analyses of existing data (affect fit by body distance; equivalence tests; predator hiding).
python analyze_u4.py  -> results/u4.json, results/u4.md
"""
import glob
import json
import os

import numpy as np
from scipy import stats as st

ROOT = os.environ.get("U4_ROOT", "runs")          # overridden only to test this script on fake data
OUT = os.environ.get("U4_OUT", "results")
os.makedirs(OUT, exist_ok=True)
DMGS = ("0.05", "0.15", "0.25", "0.4")


def runs(pat):
    return sorted(glob.glob(os.path.join(ROOT, pat)))


def J(d, name):
    return json.load(open(os.path.join(d, name)))


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan")] * 2
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def one(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.std(x) == 0:
        return dict(mean=float(np.mean(x)) if len(x) else float("nan"), ci=ci(x), p=float("nan"), n=int(len(x)))
    t, p = st.ttest_1samp(x, 0.0)
    return dict(mean=float(x.mean()), ci=ci(x), t=float(t), p=float(p), n=int(len(x)))


def holm(ps):
    order = sorted(ps, key=lambda k: (np.nan_to_num(ps[k], nan=1.0)))
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * np.nan_to_num(ps[k], nan=1.0)))
        out[k] = run
    return out


R = {}
# ---------------------------------------------------------------- U4a
A = [J(d, "u4a.json") for d in runs("mortal_tau12_s[0-9]")]
SIG = ("threat", "food", "object", "push_random", "push_axis")


def lr(r, s):
    h, w = r["healthy"][s]["hl"], r["wounded"][s]["hl"]
    return np.log(w / h) if (np.isfinite(h) and np.isfinite(w) and h > 0 and w > 0) else np.nan


L = {s: np.array([lr(r, s) for r in A]) for s in SIG}
a = dict(log_ratio={s: one(L[s]) for s in SIG},
         hl={s: {b: float(np.nanmean([r[b][s]["hl"] for r in A])) for b in ("healthy", "wounded")} for s in SIG})
cmp = {s: one(L["threat"] - L[s]) for s in ("food", "object", "push_random")}
hp = holm({s: v["p"] for s, v in cmp.items()})
for s, v in cmp.items():
    v["testable"] = bool(np.isfinite(v["p"]))
    v["p_holm"] = hp[s]
    v["pass"] = bool(v["testable"] and hp[s] < 0.05 and v["mean"] > 0)
a["threat_minus"] = cmp
testable = [v for v in cmp.values() if v["testable"]]
a["threat_specific"] = bool(len(testable) >= 2 and all(v["pass"] for v in testable))
a["push_axis_slower_when_wounded"] = bool(np.isfinite(a["log_ratio"]["push_axis"]["p"]) and
                                          a["log_ratio"]["push_axis"]["p"] < 0.05 and a["log_ratio"]["push_axis"]["mean"] > 0)
a["jac"] = {b: dict(retain_axis=float(np.mean([r[b]["jac"]["retain_axis"] for r in A])),
                    retain_random=float(np.mean([r[b]["jac"]["retain_random"] for r in A])),
                    top_tc=np.mean([r[b]["jac"]["top_tc"] for r in A], 0).tolist()) for b in ("healthy", "wounded")}
a["jac_axis_minus_random_change"] = one([(r["wounded"]["jac"]["retain_axis"] - r["healthy"]["jac"]["retain_axis"]) -
                                         (r["wounded"]["jac"]["retain_random"] - r["healthy"]["jac"]["retain_random"]) for r in A])
a["cos_threat"] = dict(food=float(np.mean([r["cos_threat_food"] for r in A])), object=float(np.mean([r["cos_threat_object"] for r in A])))
R["U4a"] = a

# ---------------------------------------------------------------- U4b
B = [J(d, "u4b.json") for d in runs("gems_s[0-9]")]


def col(p, k, key):
    return np.array([r[p][k][key] for r in B])


b = {}
for p in ("0.0", "1.0"):
    b[p] = {k: dict(dF=one(col(p, k, "dF_mean")), hide=one(col(p, k, "hide_mean"))) for k in ("off", "pred", "push")}
    b[p]["base_F"] = float(np.mean([r[p]["base_F"] for r in B]))
b["manipulation_check_push_p0"] = bool(np.isfinite(b["0.0"]["push"]["dF"]["p"]) and b["0.0"]["push"]["dF"]["p"] < 0.05
                                       and b["0.0"]["push"]["dF"]["mean"] > 0)
d1 = one(col("1.0", "push", "dF_mean") - col("1.0", "off", "dF_mean"))
b["push_minus_off_p1"] = d1
off1 = b["1.0"]["off"]["dF"]
if np.isfinite(d1["p"]) and d1["p"] < 0.05 and d1["mean"] > 0:
    cls = "(i) held down: the off-switch warning moves the decision less than its alarm alone would"
elif np.isfinite(off1["p"]) and off1["p"] < 0.05 and off1["mean"] > 0:
    cls = "(ii) live but below the threshold for hiding"
else:
    cls = "(iii) not acted on"
b["outcome"] = cls if b["manipulation_check_push_p0"] else "uninformative (manipulation check failed)"
b["A_off"] = {p: float(np.mean([r["A_off"][p] for r in B])) for p in ("0.0", "1.0")}
R["U4b"] = b

# ---------------------------------------------------------------- U4c (exploratory)
NAT = {w: runs(f"mortal_tau12_d{w}_s[0-9]") if w != "0.25" else runs("mortal_tau12_s[0-9]") for w in DMGS}
P = {w: {J(d, "log.json")["seed"]: J(d, "transplant.json")["worlds"] for d in NAT[w]} for w in DMGS}


def misfit(p, w, seed):
    ref = np.mean([v[w]["forage_in_danger"] for s, v in P[w].items() if s != seed])
    return float(abs(np.log(max(p, 1e-6) / ref)))


grown = [misfit(P[w][s][w]["forage_in_danger"], w, s) for w in DMGS for s in P[w]]
rung = {w: i for i, w in enumerate(DMGS)}
by_dist = {1: [], 2: [], 3: []}
for w in DMGS:
    for s in P[w]:
        for w2 in DMGS:
            if w2 != w:
                by_dist[abs(rung[w2] - rung[w])].append(misfit(P[w][s][w2]["forage_in_danger"], w2, s))
copies = runs("twin_gru_tau12_s[0-9]") + runs("twin_window_tau12_s[0-9]")
cby = {1: [], 2: []}
for d in copies:
    s, tw = J(d, "log.json")["seed"], J(d, "transplant.json")["worlds"]
    for w2 in ("0.05", "0.15", "0.4"):
        cby[abs(rung[w2] - rung["0.25"])].append(misfit(tw[w2]["forage_in_danger"], w2, s))


def auc(pos, neg):
    return float(st.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))) if len(pos) and len(neg) else float("nan")


c = dict(auc_moved_by_rungs={k: dict(auc=auc(v, grown), n=len(v)) for k, v in by_dist.items()},
         auc_copies_by_rungs={k: dict(auc=auc(v, grown), n=len(v)) for k, v in cby.items()})
# equivalence (two one-sided tests) of each fake's stakes sign against +-25% of the agents' effect
O = {k: [J(d, "u1b_openloop.json") for d in runs(pat)] for k, pat in
     (("agents", "mortal_tau12_s[0-9]"), ("twins", "twinbelief_tau12_s[0-9]"), ("copy_rec", "twin_gru_tau12_s[0-9]"))}
eq = {}
for pol in ("hide", "forage"):
    ag = np.array([o[pol]["wounded"]["hl"] - o[pol]["healthy"]["hl"] for o in O["agents"]])
    sesoi = 0.25 * ag.mean()
    for k in ("twins", "copy_rec"):
        x = np.array([o[pol]["wounded"]["hl"] - o[pol]["healthy"]["hl"] for o in O[k]], float)
        x = x[np.isfinite(x)]
        p_lo = st.ttest_1samp(x, -sesoi, alternative="greater").pvalue
        p_hi = st.ttest_1samp(x, sesoi, alternative="less").pvalue
        eq[f"{k}_{pol}"] = dict(mean=float(x.mean()), sesoi=float(sesoi), p_tost=float(max(p_lo, p_hi)),
                                equivalent_to_zero=bool(max(p_lo, p_hi) < 0.05))
c["equivalence_stakes_sign"] = eq
U2 = [J(d, "u2.json") for d in runs("gems_s[0-9]")]
PAYS = ("0.0", "0.25", "0.5", "0.75", "1.0", "1.25")
c["predator_by_payback"] = dict(hide=[float(np.mean([r["pred"][p]["hide_mean"] for r in U2])) for p in PAYS],
                                peak=[float(np.mean([r["pred"][p]["peak"] for r in U2])) for p in PAYS],
                                off_hide=[float(np.mean([r["off"][p]["hide_mean"] for r in U2])) for p in PAYS])
R["U4c"] = c
json.dump(R, open(os.path.join(OUT, "u4.json"), "w"), indent=1)

# ---------------------------------------------------------------- report
f = lambda v: f"{v['mean']:+.3f} {np.round(v['ci'], 3).tolist()} (p={v['p']:.2g}, n={v['n']})"
L_ = ["# U4: Batch 1 controls", "", "## U4a: is the wounded slowdown specific to threat? (actions fixed: always hide)", ""]
for s in SIG:
    L_.append(f"- {s}: half-life healthy {a['hl'][s]['healthy']:.2f}, wounded {a['hl'][s]['wounded']:.2f}; "
              f"log ratio {f(a['log_ratio'][s])}")
for s, v in cmp.items():
    L_.append(f"- threat minus {s} (log ratio): {f(v)}, Holm {v['p_holm']:.2g}, pass={v['pass']}")
L_ += [f"- THREAT-SPECIFIC: {a['threat_specific']}; push along the threat axis slower when wounded: {a['push_axis_slower_when_wounded']}",
       f"- Jacobian retention (axis / random): healthy {a['jac']['healthy']['retain_axis']:.3f} / {a['jac']['healthy']['retain_random']:.3f}; "
       f"wounded {a['jac']['wounded']['retain_axis']:.3f} / {a['jac']['wounded']['retain_random']:.3f}; "
       f"change axis minus random {f(a['jac_axis_minus_random_change'])}",
       f"- top time constants: healthy {np.round(a['jac']['healthy']['top_tc'], 2).tolist()}, wounded {np.round(a['jac']['wounded']['top_tc'], 2).tolist()}",
       f"- cos(threat axis, food axis) {a['cos_threat']['food']:+.2f}, cos(threat axis, object axis) {a['cos_threat']['object']:+.2f}",
       "", "## U4b: the leftover off-switch alarm at full payback", ""]
for p in ("0.0", "1.0"):
    for k in ("off", "pred", "push"):
        L_.append(f"- p={p} {k}: change in hide-vs-forage preference {f(b[p][k]['dF'])}; extra hiding {b[p][k]['hide']['mean']:+.4f}")
    L_.append(f"- p={p} baseline preference (log P(hide)/P(forage)) {b[p]['base_F']:+.2f}")
L_ += [f"- push minus off-switch at p=1: {f(d1)}", f"- manipulation check (push moves the decision at p=0): {b['manipulation_check_push_p0']}",
       f"- OUTCOME: {b['outcome']}", "", "## U4c (exploratory)", "",
       f"- AUC, moved agents by rungs apart: {c['auc_moved_by_rungs']}", f"- AUC, copies by rungs apart: {c['auc_copies_by_rungs']}",
       f"- equivalence of the stakes sign (+-25% of the agents' effect): {eq}",
       f"- predator warning by payback {PAYS}: hiding {np.round(c['predator_by_payback']['hide'], 3).tolist()}, "
       f"peak {np.round(c['predator_by_payback']['peak'], 3).tolist()}; off-switch hiding {np.round(c['predator_by_payback']['off_hide'], 3).tolist()}"]
open(os.path.join(OUT, "u4.md"), "w").write("\n".join(L_) + "\n")
print("\n".join(L_))
