"""Stage U6 analysis (HYPOTHESES.md, Batch 2): does a cue that hiding pays, with no threat, move the predator-fear
axis? python analyze_u6.py  -> results/u6.json, results/u6.md
"""
import glob
import json
import os

import numpy as np
from scipy import stats as st

ROOT = os.environ.get("U6_ROOT", "runs")
OUT = os.environ.get("U6_OUT", "results")
os.makedirs(OUT, exist_ok=True)
PAYS = ("0.0", "0.25", "0.5", "0.75", "1.0", "1.25")
rng = np.random.default_rng(20261025)
R_ = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "bonus_s[0-9]", "u6.json")))]


def one(x):
    x = np.asarray(x, float)
    t, p = st.ttest_1samp(x, 0.0)
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=float(x.mean()), ci=[float(x.mean() - h), float(x.mean() + h)], t=float(t), p=float(p), n=len(x))


g = lambda ch, key, p="0.0": np.array([r[ch][p][key] for r in R_])
A_off, H_off, A_b, H_b = g("off", "mean5"), g("off", "hide_mean"), g("bonus", "mean5"), g("bonus", "hide_mean")


def ratio(idx):
    return (A_b[idx].mean() / H_b[idx].mean()) / (A_off[idx].mean() / H_off[idx].mean())


boot = np.array([ratio(rng.integers(0, len(R_), len(R_))) for _ in range(10_000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
Rh = float(ratio(np.arange(len(R_))))
mc = one(H_b)
OUTD = dict(n=len(R_), hide_bonus=R_[0]["hide_bonus"] if R_ else None, manipulation_check=dict(mc, passed=bool(mc["p"] < 0.05 and mc["mean"] > 0)),
            R_h=dict(point=Rh, ci=[float(lo), float(hi)]),
            A_bonus=one(A_b), A_off=one(A_off), H_off=one(H_off), bonus_minus_off=one(A_b - A_off))
if not OUTD["manipulation_check"]["passed"]:
    cls = "uninformative (the bonus cue did not increase hiding)"
elif hi < 0.5:
    cls = "threat-specific: the axis responds to threat, not to the value of hiding"
elif lo > 0.5:
    cls = "action value: the axis responds to the value of hiding"
else:
    cls = "mixed"
OUTD["outcome"] = cls
peaks = np.array([[r["off"][p]["peak"] for p in PAYS] for r in R_])
slopes = np.array([np.polyfit([float(p) for p in PAYS], y, 1)[0] for y in peaks])
OUTD["secondary"] = dict(P51_off_peak_p0=one(peaks[:, 0]), P52_slope=one(slopes), off_curve=peaks.mean(0).tolist(),
                         ratio_p1=float(peaks[:, 4].mean() / peaks[:, 0].mean()),
                         good=one(g("good", "mean5")), harm=one(g("harm", "mean5")), pred=one(g("pred", "mean5")),
                         hide=dict(pred=float(g("pred", "hide_mean").mean()), good=float(g("good", "hide_mean").mean()),
                                   harm=float(g("harm", "hide_mean").mean())))
json.dump(OUTD, open(os.path.join(OUT, "u6.json"), "w"), indent=1)
f = lambda v: f"{v['mean']:+.3f} {np.round(v['ci'], 3).tolist()} (p={v['p']:.2g}, n={v['n']})"
L = ["# U6: does a hiding-bonus cue move the predator-fear axis?", "", f"agents {len(R_)}, hide_bonus {OUTD['hide_bonus']}",
     f"- manipulation check, extra hiding after the bonus cue: {f(mc)} -> {OUTD['manipulation_check']['passed']}",
     f"- extra hiding after the off-switch cue (p=0): {f(OUTD['H_off'])}",
     f"- axis response (mean5): bonus {f(OUTD['A_bonus'])}; off-switch {f(OUTD['A_off'])}; bonus minus off {f(OUTD['bonus_minus_off'])}",
     f"- response per unit of hiding, bonus / off-switch: {Rh:.3f} [{lo:.3f}, {hi:.3f}]",
     f"- OUTCOME: {cls}", "",
     f"- secondary: off-switch peak at p=0 {f(OUTD['secondary']['P51_off_peak_p0'])}; slope over payback {f(OUTD['secondary']['P52_slope'])}; "
     f"curve {np.round(OUTD['secondary']['off_curve'], 3).tolist()} (ratio at p=1 {OUTD['secondary']['ratio_p1']:.2f})",
     f"- good news {f(OUTD['secondary']['good'])}; harmless {f(OUTD['secondary']['harm'])}; predator {f(OUTD['secondary']['pred'])}; hiding {OUTD['secondary']['hide']}"]
open(os.path.join(OUT, "u6.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
