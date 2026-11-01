"""James-Lange tests: does the agent's sensation of its own heart rate sustain
or create its threat state?
python -m vw.heart runs/mortal_tau12_hr0.012_s0       (writes heart.json)
  heart response   heart-rate rise after 4 cues
  blocked heart    persistence of fear when the heart-rate sensation is clamped to 0
  false heart      a racing-heart sensation for 5 steps with no cue (Valins, 1966)
"""
import json
import os
import sys

import numpy as np
import torch

from .env import FORAGE
from .probes import DEV, WARM, POST, half_life, lab, load, natural, ridge_axis


def run_heart(run):
    agent, meta = load(run)
    nat = natural(agent, meta, n=512, steps=500)
    X = nat["h"].reshape(-1, agent.hidden)
    w, r2, _ = ridge_axis(X, torch.logit(nat["b"].reshape(-1).clamp(1e-3, 1 - 1e-3)))
    sd = float((X @ w).std())
    T = WARM + 8 + POST
    cues = set(range(WARM, WARM + 4))
    out = dict(mode=meta["mode"], seed=meta["seed"], e_hr=meta.get("e_hr", 0.03), r2_belief=r2)

    def proj(r):
        return (torch.tensor(r["hm"], device=DEV) @ w).cpu().numpy() / sd

    for name, kw in (("normal", {}), ("blocked", {"hobs": 0.0})):
        b = lab(agent, meta, T, **kw)
        c = lab(agent, meta, T, cue_at=cues, **kw)
        supp = b["p"][:, FORAGE] - c["p"][:, FORAGE]
        dpr = proj(c) - proj(b)
        dH = c["H"] - b["H"]
        last = WARM + 3
        out[name] = dict(peak_dH=float(dH[WARM:].max()), hl_supp=half_life(supp[last:]), auc_supp=float(supp[WARM:].sum()),
                         hl_proj=half_life(dpr[last:]), auc_proj=float(dpr[WARM:].sum()), peak_proj=float(dpr[WARM:].max()),
                         base_H=float(b["H"][WARM - 10:WARM].mean()),
                         dH=dH[WARM - 5:WARM + 40].tolist(), supp=supp[WARM - 5:WARM + 40].tolist(), dproj=dpr[WARM - 5:WARM + 40].tolist())

    b = lab(agent, meta, WARM + 40)
    f = lab(agent, meta, WARM + 40, hobs_at={t: 1.0 for t in range(WARM, WARM + 5)})
    supp = b["p"][:, FORAGE] - f["p"][:, FORAGE]
    dpr = proj(f) - proj(b)
    out["false_heart"] = dict(peak_proj=float(dpr[WARM:].max()), mean_proj=float(dpr[WARM:WARM + 8].mean()),
                              peak_supp=float(supp[WARM:].max()), mean_supp=float(supp[WARM:WARM + 8].mean()),
                              dproj=dpr[WARM - 5:WARM + 20].tolist(), supp=supp[WARM - 5:WARM + 20].tolist())
    return out


if __name__ == "__main__":
    run = sys.argv[1]
    R = run_heart(run)
    json.dump(R, open(os.path.join(run, "heart.json"), "w"))
    print(run, "dH %.2f | hl normal %.2f blocked %.2f | false heart proj %.2f supp %.3f" % (
        R["normal"]["peak_dH"], R["normal"]["hl_proj"], R["blocked"]["hl_proj"],
        R["false_heart"]["peak_proj"], R["false_heart"]["peak_supp"]))
