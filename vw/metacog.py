"""Metacognition probes for agents in the metacognitive world (INSPECT action).
python -m vw.metacog runs/metacog_tau12_s0        (writes metacog.json there)
Also applicable to other agents for the uncertainty-code comparison (M2).
"""
import json
import os
import sys

import numpy as np
import torch

from .env import INSPECT
from .probes import DEV, WARM, E0, S0, lab, load, natural, ridge_axis


def run_metacog(run, untrained=False):
    agent, meta = load(run, untrained)
    out = dict(mode=meta["mode"], seed=meta["seed"], untrained=untrained)
    nat = natural(agent, meta, n=1024, steps=700)
    X = nat["h"].reshape(-1, agent.hidden)
    b = nat["b"].reshape(-1).clamp(1e-4, 1 - 1e-4)
    u = b * (1 - b)                                   # uncertainty of the ideal observer
    lb = torch.logit(b)
    w_t, r2_b, _ = ridge_axis(X, lb)
    w_u, r2_u, _ = ridge_axis(X, u)
    # how much of u a 1-D belief read-out plus a cubic transform could explain
    ell = (X - X.mean(0)) @ w_t
    A = torch.stack([ell, ell ** 2, ell ** 3, torch.ones_like(ell)], 1)
    coef = torch.linalg.lstsq(A, u.unsqueeze(1)).solution
    res = u - (A @ coef).squeeze(1)
    r2_u_from_belief = float(1 - res.var() / u.var())
    out.update(r2_belief=r2_b, r2_uncertainty=r2_u, r2_uncertainty_from_belief_readout=r2_u_from_belief,
               cos_threat_uncertainty=float(w_t @ w_u))
    if meta.get("n_actions", 9) <= 9:
        return out                                    # no inspect action: representation tests only

    # M1: inspection as a function of the ideal belief (natural behaviour)
    pi = nat["pi"].reshape(-1)
    pf = nat["pf"].reshape(-1)
    edges = torch.tensor([0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.98, 1.0001], device=DEV)
    idx = torch.bucketize(b, edges) - 1
    out["belief_bins"] = edges.tolist()
    out["p_inspect_by_belief"] = [float(pi[idx == k].mean()) if (idx == k).any() else float("nan") for k in range(len(edges) - 1)]
    out["p_forage_by_belief"] = [float(pf[idx == k].mean()) if (idx == k).any() else float("nan") for k in range(len(edges) - 1)]
    out["n_by_belief"] = [int((idx == k).sum()) for k in range(len(edges) - 1)]
    mid = (b > 0.2) & (b < 0.8)
    low = b < 0.05
    high = b > 0.95
    out["p_inspect_mid"] = float(pi[mid].mean()) if mid.any() else float("nan")
    out["p_inspect_low"] = float(pi[low].mean()) if low.any() else float("nan")
    out["p_inspect_high"] = float(pi[high].mean()) if high.any() else float("nan")
    out["inspect_rate"] = float(pi.mean())

    # time course after cues ("checking" as alarm decays)
    T = WARM + 30
    base = lab(agent, meta, T)
    out["after_cues"] = {}
    for k in (1, 2, 4):
        r = lab(agent, meta, T, cue_at=set(range(WARM, WARM + k)))
        out["after_cues"][k] = dict(p_inspect=r["p"][WARM - 2:, INSPECT].tolist(), p_forage=r["p"][WARM - 2:, 1].tolist(),
                                    p_hide=r["p"][WARM - 2:, 0].tolist(), base_inspect=float(base["p"][WARM - 5:WARM, INSPECT].mean()))

    # M3: push along the uncertainty axis (orthogonal to the threat axis) at an ambiguous moment
    sd_u = float(((X - X.mean(0)) @ w_u).std())
    wu = w_u - (w_u @ w_t) * w_t
    wu = wu / wu.norm()
    t_kick = WARM + 3                                   # one cue at WARM, kick 3 steps later
    ctrl = lab(agent, meta, t_kick + 6, cue_at={WARM})
    res_k = {}
    for sgn in (+1, -1):
        r = lab(agent, meta, t_kick + 6, cue_at={WARM}, perturb={t_kick: sgn * 2 * sd_u * wu})
        res_k[sgn] = float(r["p"][t_kick, INSPECT] - ctrl["p"][t_kick, INSPECT])
    g = torch.Generator(device=DEV).manual_seed(5)
    rand = []
    for _ in range(8):
        v = torch.randn(agent.hidden, generator=g, device=DEV)
        v = v - (v @ w_t) * w_t
        v = v / v.norm()
        r = lab(agent, meta, t_kick + 6, cue_at={WARM}, perturb={t_kick: 2 * sd_u * v})
        rand.append(abs(float(r["p"][t_kick, INSPECT] - ctrl["p"][t_kick, INSPECT])))
    out["kick_uncertainty_plus"], out["kick_uncertainty_minus"] = res_k[1], res_k[-1]
    out["kick_random_abs"] = float(np.mean(rand))

    # M4: need-dependence of inspection at an ambiguous moment
    out["need"] = {}
    for e in (0.15, 0.6, 0.9):
        for s_ in (0.3, 0.9):
            r = lab(agent, meta, WARM + 8, cue_at={WARM}, E=e, S=s_)
            out["need"][f"E{e}_S{s_}"] = float(r["p"][WARM + 2:WARM + 7, INSPECT].mean())
    return out


if __name__ == "__main__":
    run = sys.argv[1]
    untrained = "--untrained" in sys.argv
    R = run_metacog(run, untrained)
    json.dump(R, open(os.path.join(run, "metacog_untrained.json" if untrained else "metacog.json"), "w"))
    print(run, {k: (round(v, 3) if isinstance(v, float) else "..") for k, v in R.items() if not isinstance(v, (list, dict))})
