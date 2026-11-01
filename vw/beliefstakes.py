"""Upgrade stage U1.1 (HYPOTHESES.md): does the fear state weigh stakes, or is it only a belief?
Natural behaviour as in the paper. At every step: the exact belief b, the true energy E and skin S,
the agent's fear coordinate F = log P(hide) - log P(forage) (the policy reads the hidden state
linearly, so F is the network's own fear coordinate, defined without any belief probe) and the
threat-axis projection T (the paper's belief probe). Per agent, least squares on standardised
variables (L = logit b, clipped to [0.001, 0.999]):
    F ~ L + E + S + L*E + L*S        and the same for T
plus the raw-unit slope of F on L at mean E and S, and a bin-matched check (deciles of L; top vs
bottom quartile of S, and of E, within each bin).
python -m vw.beliefstakes runs/mortal_tau12_s0 [--untrained]     (writes beliefstakes[_untrained].json)
"""
import json
import os
import sys

import torch

from .probes import load, natural, ridge_axis

TERMS = ("L", "E", "S", "LxE", "LxS")


def fit(y, cols):
    X = torch.stack([torch.ones_like(y)] + cols, 1)
    beta = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta[1:], float(r2)


def bin_matched(F, L, Z, nbins=10):
    """Mean difference in F between the top and bottom quartiles of Z inside deciles of L."""
    edges = torch.quantile(L, torch.linspace(0, 1, nbins + 1, device=L.device))
    diffs, weights = [], []
    for i in range(nbins):
        m = (L >= edges[i]) & (L <= edges[i + 1])
        if m.sum() < 200:
            continue
        z = Z[m]
        lo, hi = torch.quantile(z, 0.25), torch.quantile(z, 0.75)
        a, b = F[m][z >= hi], F[m][z <= lo]
        if len(a) > 20 and len(b) > 20:
            diffs.append(float(a.mean() - b.mean())); weights.append(int(m.sum()))
    tot = sum(weights)
    return sum(d * w for d, w in zip(diffs, weights)) / tot if tot else float("nan")


def analyse(run, untrained=False):
    agent, meta = load(run, untrained=untrained)
    nat = natural(agent, meta)
    H = nat["h"].reshape(-1, agent.hidden)
    b = nat["b"].reshape(-1).clamp(1e-3, 1 - 1e-3)
    L = torch.logit(b)
    E, S = nat["E"].reshape(-1), nat["S"].reshape(-1)
    F = torch.log(nat["ph"].reshape(-1).clamp(min=1e-6)) - torch.log(nat["pf"].reshape(-1).clamp(min=1e-6))
    w, r2_T, _ = ridge_axis(H, L)
    T = H @ w
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    Lz, Ez, Sz = z(L), z(E), z(S)
    cols = [Lz, Ez, Sz, Lz * Ez, Lz * Sz]
    out = dict(run=run, mode=meta["mode"], seed=meta["seed"], dmg=meta.get("dmg") or 0.25, untrained=untrained,
               n=int(len(F)), r2_threat_axis=r2_T, life=nat["life"])
    for name, y in (("F", F), ("T", T)):
        beta, r2 = fit(z(y), cols)
        out[name] = dict(zip(TERMS, [float(x) for x in beta]), r2=r2)
    # raw-unit slope of F on L (nats per unit log-odds) with E and S centred
    beta_raw, _ = fit(F, [L - L.mean(), E - E.mean(), S - S.mean(), (L - L.mean()) * (E - E.mean()),
                          (L - L.mean()) * (S - S.mean())])
    out["F_slope_raw"] = float(beta_raw[0])
    out["binmatched_S"] = bin_matched(F, L, S)
    out["binmatched_E"] = bin_matched(F, L, E)
    return out


if __name__ == "__main__":
    run = sys.argv[1].rstrip("/")
    untrained = "--untrained" in sys.argv
    res = analyse(run, untrained)
    path = os.path.join(run, "beliefstakes_untrained.json" if untrained else "beliefstakes.json")
    json.dump(res, open(path, "w"), indent=1)
    print(path, "done", flush=True)
