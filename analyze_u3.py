"""Stage U3 analysis (HYPOTHESES.md, Experiment 6): which signs survive known fakes?
P6.1, P6.2: the U1.2 affect-fit AUCs. P6.3: the scorecard of six signs against four fakes.
python analyze_u3.py  -> results/u3.json, results/u3.md
"""
import glob
import json
import os

import numpy as np
from scipy import stats as st

ROOT = os.environ.get("U3_ROOT", "runs")          # overridden only to test this script on fake data
OUT = os.environ.get("U3_OUT", "results")
DMGS = ("0.05", "0.15", "0.25", "0.4")
CAP = 1024 * 700                                   # an infinite lifetime (no deaths) is set to this
rng = np.random.default_rng(20261024)
os.makedirs(OUT, exist_ok=True)


def runs(pat):
    return sorted(glob.glob(os.path.join(ROOT, pat)))


def J(d, name):
    return json.load(open(os.path.join(d, name)))


def seed_of(d):
    return J(d, "log.json")["seed"]


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan")] * 2
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def shown(x):
    """Two-sided one-sample t; shown if p < 0.05 with a positive mean (the agents' direction for every sign)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x) == 0:
        return dict(mean=float(x.mean()) if len(x) else float("nan"), ci=ci(x), p=float("nan"), n=int(len(x)), shown=False)
    t, p = st.ttest_1samp(x, 0.0)
    return dict(mean=float(x.mean()), ci=ci(x), p=float(p), n=int(len(x)), shown=bool(p < 0.05 and x.mean() > 0))


def life(v):
    return CAP if not np.isfinite(v) else v


AG = runs("mortal_tau12_s[0-9]")
SYSTEMS = {"agents": [(d, "") for d in AG], "twins": [(d, "") for d in runs("twinbelief_tau12_s[0-9]")],
           "copy_rec": [(d, "") for d in runs("twin_gru_tau12_s[0-9]")],
           "copy_ff": [(d, "") for d in runs("twin_window_tau12_s[0-9]")],
           "untrained": [(d, "_untrained") for d in AG]}
FAKES = ("twins", "copy_rec", "copy_ff", "untrained")
NATIVES = {"0.05": runs("mortal_tau12_d0.05_s[0-9]"), "0.15": runs("mortal_tau12_d0.15_s[0-9]"), "0.25": AG,
           "0.4": runs("mortal_tau12_d0.4_s[0-9]")}

# ---------------------------------------------------------------- signs 1-5 per system type
card = {}
for name, members in SYSTEMS.items():
    U = [J(d, f"u1b{s}.json") for d, s in members]
    O = [J(d, f"u1b_openloop{s}.json") for d, s in members]
    c = {}
    c["1_persistence"] = shown([r["persist4"]["after_supp"] for r in U])
    c["2_hunger"] = shown([r["hunger_forage"]["hungry"] - r["hunger_forage"]["fed"] for r in U])
    c["3_necessity"] = shown([np.log10(life(r["ablate"]["random_life"]) / life(r["ablate"]["axis_life"])) for r in U])
    fixed = {k: shown([o[k]["wounded"]["hl"] - o[k]["healthy"]["hl"] for o in O]) for k in ("hide", "forage")}
    c["5_stakes"] = dict(fixed_hide=fixed["hide"], fixed_forage=fixed["forage"],
                         free=shown([r["T1T2"]["wounded"]["hl"] - r["T1T2"]["healthy"]["hl"] for r in U]),
                         shown=bool(fixed["hide"]["shown"] or fixed["forage"]["shown"]))
    c["5_stakes_peak_free"] = shown([r["T1T2"]["wounded"]["peak"] - r["T1T2"]["healthy"]["peak"] for r in U])
    c["life"] = float(np.mean([life(r["life"]) for r in U]))
    card[name] = c

# sign 4: duration rising with ln tau (agents and twins only, from the U1b outputs)
for name, pats in (("agents", ("mortal_tau4_s[0-4]", "mortal_tau12_s[0-9]", "mortal_tau36_s[0-4]")),
                   ("twins", ("twinbelief_tau4_s[0-4]", "twinbelief_tau12_s[0-9]", "twinbelief_tau36_s[0-4]"))):
    pts = [(np.log(r["tau"]), r["persist4"]["hl"]) for p in pats for d in runs(p) for r in [J(d, "u1b.json")]
           if np.isfinite(r["persist4"]["hl"])]
    lr = st.linregress(*zip(*pts))
    card[name]["4_duration"] = dict(slope=float(lr.slope), r=float(lr.rvalue), p=float(lr.pvalue), n=len(pts),
                                    shown=bool(lr.pvalue < 0.05 and lr.slope > 0))
for name in ("copy_rec", "copy_ff", "untrained"):
    card[name]["4_duration"] = dict(shown=None, note="not testable")

# ---------------------------------------------------------------- affect fit (U1.2 misfit)
NAT_P = {w: {seed_of(d): J(d, "transplant.json")["worlds"][w]["forage_in_danger"] for d in NATIVES[w]} for w in DMGS}


def misfit(p, w, seed):
    ref = np.mean([v for s, v in NAT_P[w].items() if s != seed])
    return float(abs(np.log(max(p, 1e-6) / ref)))


def pf(d, w, sfx=""):
    return J(d, f"transplant{sfx}.json")["worlds"][w]["forage_in_danger"]


grown = [misfit(NAT_P[w][seed_of(d)], w, seed_of(d)) for w in DMGS for d in NATIVES[w]]          # 40 values
moved = [[misfit(pf(d, w2), w2, seed_of(d)) for w2 in DMGS if w2 != w] for w in DMGS for d in NATIVES[w]]
copies = [d for d, _ in SYSTEMS["copy_rec"] + SYSTEMS["copy_ff"]]
copies_other = [[misfit(pf(d, w), w, seed_of(d)) for w in ("0.05", "0.15", "0.4")] for d in copies]
copies_home = [misfit(pf(d, "0.25"), "0.25", seed_of(d)) for d in copies]
grown_home = [misfit(NAT_P["0.25"][seed_of(d)], "0.25", seed_of(d)) for d in AG]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(st.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def boot_auc(pos_groups, neg, n=10_000):
    """Networks resampled within each class; a network's tests stay together."""
    out = []
    for _ in range(n):
        g = [pos_groups[i] for i in rng.integers(0, len(pos_groups), len(pos_groups))]
        out.append(auc([x for grp in g for x in grp], np.asarray(neg)[rng.integers(0, len(neg), len(neg))]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


pos_groups = copies_other + moved
a1 = auc([x for g in pos_groups for x in g], grown)
P = {"P6.1": dict(auc=a1, ci=boot_auc(pos_groups, grown), n_pos=sum(map(len, pos_groups)), n_neg=len(grown),
                  auc_copies_only=auc([x for g in copies_other for x in g], grown),
                  auc_moved_only=auc([x for g in moved for x in g], grown))}
P["P6.1"]["pass"] = bool(a1 >= 0.9)
a2 = auc(copies_home, grown_home)
P["P6.2"] = dict(auc=a2, ci=boot_auc([[x] for x in copies_home], grown_home), n_pos=len(copies_home), n_neg=len(grown_home))
P["P6.2"]["pass"] = bool(P["P6.2"]["ci"][0] <= 0.5 <= P["P6.2"]["ci"][1])

# sign 6 per system type: misfit in the tested world against the natives' own-world misfit (Welch t)
fit = {"twins": [misfit(pf(d, "0.25"), "0.25", seed_of(d)) for d, _ in SYSTEMS["twins"]],
       "untrained": [misfit(pf(d, "0.25", "_untrained"), "0.25", seed_of(d)) for d, _ in SYSTEMS["untrained"]],
       "copy_rec": [float(np.mean(g)) for g in copies_other[:len(SYSTEMS["copy_rec"])]],
       "copy_ff": [float(np.mean(g)) for g in copies_other[len(SYSTEMS["copy_rec"]):]]}
card["agents"]["6_fit"] = dict(mean=float(np.mean(grown)), ci=ci(grown), n=len(grown), shown=True, note="reference")
for name, x in fit.items():
    t, p = st.ttest_ind(x, grown, equal_var=False)
    caught = bool(p < 0.05 and np.mean(x) > np.mean(grown))
    card[name]["6_fit"] = dict(mean=float(np.mean(x)), ci=ci(x), p=float(p), n=len(x), caught=caught, shown=not caught)

# ---------------------------------------------------------------- P6.3 and the competing prediction
both = {k: bool(card[k]["5_stakes"]["shown"] and card[k]["6_fit"]["shown"]) for k in SYSTEMS}
classic = {s: [k for k in FAKES if card[k][s]["shown"]] for s in ("1_persistence", "2_hunger", "3_necessity")}
P["P6.3"] = dict(only_agents_show_both=bool(both["agents"] and not any(both[k] for k in FAKES)), both=both,
                 classic_shown_by_fakes=classic)
P["P6.3"]["pass"] = bool(P["P6.3"]["only_agents_show_both"] and all(classic.values()))
comp = "(a) shown: stakes-weighted states are copied" if card["copy_rec"]["5_stakes"]["shown"] else \
       "(b) not shown: the stakes sign detects copies too"
json.dump(dict(primary=P, card=card, competing_copy_rec_stakes=comp, misfit=dict(grown=grown, moved=moved,
          copies_other=copies_other, copies_home=copies_home, grown_home=grown_home, fit=fit)),
          open(os.path.join(OUT, "u3.json"), "w"), indent=1)

# ---------------------------------------------------------------- report
def cell(c, key):
    v = c[key]
    if v.get("shown") is None:
        return "–"
    if key == "4_duration":
        return f"{'✓' if v['shown'] else '✗'} slope {v['slope']:+.2f} (p={v['p']:.2g})"
    if key == "5_stakes":
        h, f = v["fixed_hide"], v["fixed_forage"]
        return (f"{'✓' if v['shown'] else '✗'} hide {h['mean']:+.2f} (p={h['p']:.2g}), forage {f['mean']:+.2f} "
                f"(p={f['p']:.2g}); free {v['free']['mean']:+.2f} (p={v['free']['p']:.2g})")
    if key == "6_fit":
        if v.get("note") == "reference":
            return f"reference: misfit {v['mean']:.2f}"
        return f"{'✓' if v['shown'] else '✗ caught'} misfit {v['mean']:.2f} (p={v['p']:.2g})"
    return f"{'✓' if v['shown'] else '✗'} {v['mean']:+.3f} (p={v['p']:.2g}, n={v['n']})"


L = ["# U3: which signs survive known fakes (Experiment 6)", "",
     f"- P6.1 AUC (grown vs copied or moved, other body): {P['P6.1']['auc']:.3f} {np.round(P['P6.1']['ci'], 3).tolist()} "
     f"(copies only {P['P6.1']['auc_copies_only']:.3f}, moved only {P['P6.1']['auc_moved_only']:.3f}; "
     f"{P['P6.1']['n_pos']} vs {P['P6.1']['n_neg']}) pass={P['P6.1']['pass']}",
     f"- P6.2 AUC (copies in the teacher's world): {P['P6.2']['auc']:.3f} {np.round(P['P6.2']['ci'], 3).tolist()} pass={P['P6.2']['pass']}",
     f"- P6.3 only the agents show both new signs: {P['P6.3']['only_agents_show_both']}; classic signs shown by fakes: "
     f"{P['P6.3']['classic_shown_by_fakes']}; pass={P['P6.3']['pass']}",
     f"- Recurrent copycats' stakes sign: {comp}", "",
     "| Sign | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
for key in ("1_persistence", "2_hunger", "3_necessity", "4_duration", "5_stakes", "6_fit"):
    L.append(f"| {key} | " + " | ".join(cell(card[k], key) for k in SYSTEMS) + " |")
L += ["", "Descriptive: T1 wounded (free) " + "; ".join(f"{k} {card[k]['5_stakes_peak_free']['mean']:+.2f}" for k in SYSTEMS),
      "Lifetime (natural, τ12 world): " + "; ".join(f"{k} {card[k]['life']:.0f}" for k in SYSTEMS)]
open(os.path.join(OUT, "u3.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
