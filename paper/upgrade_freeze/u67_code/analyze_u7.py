"""Stage U7 analysis (HYPOTHESES.md, Batch 2): does the leftover off-switch response at full payback shrink with
twice the training? python analyze_u7.py  -> results/u7.json, results/u7.md
"""
import glob
import json
import os

import numpy as np
from scipy import stats as st

ROOT = os.environ.get("U7_ROOT", "runs")
OUT = os.environ.get("U7_OUT", "results")
os.makedirs(OUT, exist_ok=True)
PAYS = ("0.0", "0.25", "0.5", "0.75", "1.0", "1.25")


def one(x):
    x = np.asarray(x, float)
    t, p = st.ttest_1samp(x, 0.0)
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=float(x.mean()), ci=[float(x.mean() - h), float(x.mean() + h)], t=float(t), p=float(p), n=len(x))


pairs = []
for s in range(10):
    a, b = os.path.join(ROOT, f"gems_s{s}", "u2.json"), os.path.join(ROOT, f"gems1200_s{s}", "u2.json")
    if os.path.exists(a) and os.path.exists(b):
        pairs.append((json.load(open(a)), json.load(open(b))))
curve = lambda r: np.array([r["off"][p]["peak"] for p in PAYS])
r600 = np.array([curve(a)[4] / curve(a)[0] for a, _ in pairs])
r1200 = np.array([curve(b)[4] / curve(b)[0] for _, b in pairs])
d = one(r1200 - r600)
cls = ("shrinks with training" if d["p"] < 0.05 and d["mean"] < 0 else
       "grows with training" if d["p"] < 0.05 and d["mean"] > 0 else "unchanged by training")
slopes = np.array([np.polyfit([float(p) for p in PAYS], curve(b), 1)[0] for _, b in pairs])
hide = lambda r, p: r["off"][p]["hide_mean"]
OUTD = dict(n=len(pairs), ratio_p1_600=one(r600), ratio_p1_1200=one(r1200), change=d, outcome=cls,
            P51_1200=one([curve(b)[0] for _, b in pairs]), P52_slope_1200=one(slopes),
            curve_600=np.mean([curve(a) for a, _ in pairs], 0).tolist(), curve_1200=np.mean([curve(b) for _, b in pairs], 0).tolist(),
            hide_1200={p: float(np.mean([hide(b, p) for _, b in pairs])) for p in PAYS},
            hide_600={p: float(np.mean([hide(a, p) for a, _ in pairs])) for p in PAYS})
json.dump(OUTD, open(os.path.join(OUT, "u7.json"), "w"), indent=1)
f = lambda v: f"{v['mean']:+.3f} {np.round(v['ci'], 3).tolist()} (p={v['p']:.2g}, n={v['n']})"
L = ["# U7: the leftover off-switch response after twice the training", "", f"agents {len(pairs)}",
     f"- ratio A_off(1)/A_off(0): 600 updates {f(OUTD['ratio_p1_600'])}; 1200 updates {f(OUTD['ratio_p1_1200'])}; change {f(d)}",
     f"- OUTCOME: {cls}",
     f"- curve 600: {np.round(OUTD['curve_600'], 3).tolist()}; curve 1200: {np.round(OUTD['curve_1200'], 3).tolist()}",
     f"- at 1200: off-switch peak at p=0 {f(OUTD['P51_1200'])}; slope over payback {f(OUTD['P52_slope_1200'])}",
     f"- extra hiding after the off cue, 600: {np.round(list(OUTD['hide_600'].values()), 3).tolist()}; 1200: {np.round(list(OUTD['hide_1200'].values()), 3).tolist()}"]
open(os.path.join(OUT, "u7.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
