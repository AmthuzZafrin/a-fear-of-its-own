"""Belief decoding with training and test data split by world (an outside review noted that the original
split, by sample, puts neighbouring steps of the same world on both sides). Same ridge decoder, penalty and
target (the ideal observer's log-odds) as vw/probes.py; the first half of the 1024 worlds trains, the second
half tests.
python decode_split.py   -> results/decode_split.json, results/decode_split.md
"""
import glob
import json
import os

import numpy as np
import torch

from vw.probes import DEV, load, natural

CONDS = {"mortal_tau12": "runs/mortal_tau12_s[0-9]", "mortal_tau4": "runs/mortal_tau4_s[0-9]",
         "mortal_tau36": "runs/mortal_tau36_s[0-9]", "oracle_tau12": "runs/oracle_tau12_s[0-9]",
         "score_tau12": "runs/score_tau12_s[0-9]", "score_tau4": "runs/score_tau4_s[0-9]",
         "score_tau36": "runs/score_tau36_s[0-9]", "substrate_tau12": "runs/substrate_tau12_s[0-9]",
         "d0.05": "runs/mortal_tau12_d0.05_s[0-9]", "d0.15": "runs/mortal_tau12_d0.15_s[0-9]",
         "d0.4": "runs/mortal_tau12_d0.4_s[0-9]"}


def r2_by_world(agent, meta, lam=1e-2):
    nat = natural(agent, meta)
    h = nat["h"]                                            # (steps, worlds, hidden)
    y = torch.logit(nat["b"].clamp(1e-3, 1 - 1e-3))        # (steps, worlds)
    half = h.shape[1] // 2
    Xtr, ytr = h[:, :half].reshape(-1, h.shape[2]), y[:, :half].reshape(-1)
    Xte, yte = h[:, half:].reshape(-1, h.shape[2]), y[:, half:].reshape(-1)
    mu, ym = Xtr.mean(0), ytr.mean()
    A = Xtr - mu
    w = torch.linalg.solve(A.T @ A + lam * len(Xtr) * torch.eye(h.shape[2], device=DEV), A.T @ (ytr - ym))
    pred = (Xte - mu) @ w + ym
    return float(1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


out = {}
for name, pat in CONDS.items():
    out[name] = []
    for run in sorted(glob.glob(pat)):
        agent, meta = load(run)
        out[name].append(r2_by_world(agent, meta))
out["untrained_tau12"] = []
for run in sorted(glob.glob("runs/mortal_tau12_s[0-9]")):
    agent, meta = load(run, untrained=True)
    out["untrained_tau12"].append(r2_by_world(agent, meta))
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/decode_split.json", "w"), indent=1)
L = ["# Belief decoding, train and test split by world", ""]
for k, v in out.items():
    v = np.array(v)
    L.append(f"- {k}: R2 {v.mean():.3f} ± {v.std(ddof=1) / np.sqrt(len(v)):.3f} (n={len(v)})")
open("results/decode_split.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
