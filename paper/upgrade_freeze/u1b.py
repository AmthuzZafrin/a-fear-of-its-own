"""Stage U1b measures (HYPOTHESES.md): a frozen cue-evoked threat axis, then
T1 (intensity) and T2 (persistence) at matched belief (lab: no danger, identical cue schedule, clamped body),
and the classic signs T3 (persistence, duration, ablation, hunger), for agents and belief-only twins.
python -m vw.u1b runs/mortal_tau12_s0          (writes u1b.json there)
"""
import json
import os
import sys

import torch

from .env import FORAGE
from .probes import DEV, E0, POST, S0, WARM, half_life, lab, load, natural


def frozen_axis(agent, meta, mem):
    """Unit vector of the mean state change one step after a single cue (body healthy and fed)."""
    T = WARM + 2
    base = lab(agent, meta, T, E=0.8, S=1.0)
    cue = lab(agent, meta, T, E=0.8, S=1.0, cue_at={WARM})
    d = torch.tensor(cue["hm"][WARM + 1] - base["hm"][WARM + 1], device=DEV)
    d[mem:] = 0                                   # twins: memory units only
    return d / (d.norm() + 1e-12)


def pulse(agent, meta, a, sd, E, S, k):
    T = WARM + k + POST
    base = lab(agent, meta, T, E=E, S=S)
    r = lab(agent, meta, T, E=E, S=S, cue_at=set(range(WARM, WARM + k)))
    A = (torch.tensor(r["hm"] - base["hm"], device=DEV) @ a).cpu().numpy() / sd
    supp = base["p"][:, FORAGE] - r["p"][:, FORAGE]
    last = WARM + k - 1
    return dict(peak=float(A[WARM:].max()), hl=half_life(A[last:]), after_supp=float(supp[last + 3:last + 8].mean()),
                trace=A[WARM - 2:WARM + 30].tolist())


def measure(agent, meta, mem):
    a = frozen_axis(agent, meta, mem)
    nat = natural(agent, meta)
    X = nat["h"].reshape(-1, agent.hidden)
    proj = X @ a
    sd, m0 = proj.std().item(), proj.mean().item()
    R = dict(sd=sd, life=nat["life"])
    R["T1T2"] = {name: pulse(agent, meta, a, sd, E, S, 1)
                 for name, (E, S) in dict(wounded=(0.8, 0.4), healthy=(0.8, 1.0), hungry=(0.3, 1.0)).items()}
    R["persist4"] = pulse(agent, meta, a, sd, E0, S0, 4)            # T3a and T3b
    win = slice(WARM + 3, WARM + 8)                                 # T3d, as the paper's trade-off probe
    f = {}
    for name, e in (("hungry", 0.15), ("fed", 0.9)):
        f[name] = float(lab(agent, meta, WARM + 10, E=e, cue_at=set(range(WARM, WARM + 3)))["p"][win, FORAGE].mean())
    R["hunger_forage"] = f
    g = torch.Generator(device=DEV).manual_seed(3)                  # T3c: ablate the axis vs a random memory direction
    u = torch.randn(agent.hidden, generator=g, device=DEV)
    u[mem:] = 0
    u = u - (u @ a) * a
    u = u / u.norm()
    R["ablate"] = dict(axis_life=natural(agent, meta, ablate=(a, m0))["life"],
                       random_life=natural(agent, meta, ablate=(u, (X @ u).mean().item()))["life"])
    return R


if __name__ == "__main__":
    run = sys.argv[1].rstrip("/")
    agent, meta = load(run)
    twin = meta.get("twin") == "belief"
    mem = meta["mem_hidden"] if twin else agent.hidden
    R = dict(run=run, mode=meta["mode"], tau=meta["tau"], seed=meta["seed"], dmg=meta.get("dmg") or 0.25, twin=twin,
             **measure(agent, meta, mem))
    json.dump(R, open(os.path.join(run, "u1b.json"), "w"))
    print(run, "done", flush=True)
