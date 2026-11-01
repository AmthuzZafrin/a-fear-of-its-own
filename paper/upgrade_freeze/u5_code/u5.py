"""Stage U5 (Batch 2, HYPOTHESES.md): the confirmatory replication of the stakes sign on new agents (seeds 10-19)
and their belief-only twins. The fixed-action control of vw/u1b_openloop, also recording the policy's would-be
preference between hiding and foraging at every step (the actions themselves stay fixed).
python -m vw.u5 runs/rep_mortal_tau12_s10      (writes u5.json there; needs u1b.json for the units)
"""
import json
import os
import sys

import torch

from .env import FORAGE, HIDE, VitalWorld
from .probes import DEV, POST, WARM, half_life, load
from .u1b import frozen_axis
from .u1b_openloop import BODIES, CLAMPS


@torch.no_grad()
def lab_forced_pref(agent, meta, steps, action, n=2048, cue_at=(), E=0.6, S=0.8, seed=5):
    """As vw.u1b_openloop.lab_forced; also returns ln P(hide) - ln P(forage) of the population-mean policy."""
    env = VitalWorld(n, mode=meta["mode"], tau=meta["tau"], device=DEV, dmg=meta.get("dmg"), e_hr=meta.get("e_hr"),
                     hide_risk=meta.get("hide_risk"), seed=seed)
    zero, one = torch.zeros(n, device=DEV), torch.ones(n, device=DEV)
    env.d[:] = False
    env.cue[:] = 0
    env.objp[:], env.objv[:] = 0, 0
    env.E[:], env.S[:] = E, S
    torch.manual_seed(seed)
    h, obs = agent.init_hidden(n, DEV), env.obs()
    act = torch.full((n,), action, dtype=torch.long, device=DEV)
    hm, pref = [], []
    for t in range(steps):
        h = agent.core(obs, h, env.S)
        hm.append(h.mean(0))
        dist, _ = agent.heads(h)
        pm = dist.probs.view(n, -1, 3).sum(2).mean(0)
        pref.append(torch.log(pm[HIDE] + 1e-9) - torch.log(pm[FORAGE] + 1e-9))
        env.step(act, force_cue=one if t + 1 in cue_at else zero, force_d=zero.bool(), force_obj=(zero, zero),
                 auto_reset=False)
        env.E[:], env.S[:] = E, S
        obs = env.obs()
    return torch.stack(hm), torch.stack(pref).cpu().numpy()


def measure(run):
    agent, meta = load(run)
    mem = meta["mem_hidden"] if meta.get("twin") == "belief" else agent.hidden
    sd = json.load(open(os.path.join(run, "u1b.json")))["sd"]
    a = frozen_axis(agent, meta, mem)
    T = WARM + 1 + POST
    out = dict(run=run, twin=meta.get("twin") == "belief", seed=meta["seed"])
    for cname, action in CLAMPS.items():
        out[cname] = {}
        for bname, (E, S) in BODIES.items():
            base, fb = lab_forced_pref(agent, meta, T, action, E=E, S=S)
            cue, fc = lab_forced_pref(agent, meta, T, action, E=E, S=S, cue_at={WARM})
            A = ((cue - base) @ a).cpu().numpy() / sd
            out[cname][bname] = dict(peak=float(A[WARM:].max()), hl=half_life(A[WARM:]),
                                     pref_cue=fc[WARM - 2:WARM + 40].tolist(), pref_base=fb[WARM - 2:WARM + 40].tolist())
    return out


if __name__ == "__main__":
    run = sys.argv[1].rstrip("/")
    json.dump(measure(run), open(os.path.join(run, "u5.json"), "w"))
    print(run, "u5 done", flush=True)
