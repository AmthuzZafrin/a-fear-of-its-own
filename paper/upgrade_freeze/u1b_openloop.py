"""Exploratory open-loop control for U1b (HYPOTHESES.md, 24 Oct 2026): T1/T2 with the actions clamped
identically in both body states (always hide, or always forage; heart held), so behaviour cannot feed back
into belief updating. Same frozen axis, seeds and units as vw/u1b.py.
python -m vw.u1b_openloop runs/mortal_tau12_s0      (writes u1b_openloop.json there)
python -m vw.u1b_openloop --summary                  (writes results/u1b_openloop.md)
"""
import glob
import json
import os
import sys

import numpy as np
import torch

from .env import FORAGE, HIDE, VitalWorld
from .probes import DEV, POST, WARM, half_life, load
from .u1b import frozen_axis

CLAMPS = {"hide": HIDE * 3 + 1, "forage": FORAGE * 3 + 1}       # heart rate held
BODIES = dict(wounded=(0.8, 0.4), healthy=(0.8, 1.0), hungry=(0.3, 1.0))


@torch.no_grad()
def lab_forced(agent, meta, steps, action, n=2048, cue_at=(), E=0.6, S=0.8, seed=5):
    """As probes.lab, but every copy takes the same fixed action at every step."""
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
    hm = []
    for t in range(steps):
        h = agent.core(obs, h, env.S)
        hm.append(h.mean(0))
        env.step(act, force_cue=one if t + 1 in cue_at else zero, force_d=zero.bool(), force_obj=(zero, zero),
                 auto_reset=False)
        env.E[:], env.S[:] = E, S
        obs = env.obs()
    return torch.stack(hm)


def measure(run):
    agent, meta = load(run)
    mem = meta["mem_hidden"] if meta.get("twin") == "belief" else agent.hidden
    sd = json.load(open(os.path.join(run, "u1b.json")))["sd"]
    a = frozen_axis(agent, meta, mem)
    T = WARM + 1 + POST
    out = dict(run=run, twin=meta.get("twin") == "belief")
    for cname, action in CLAMPS.items():
        out[cname] = {}
        for bname, (E, S) in BODIES.items():
            base = lab_forced(agent, meta, T, action, E=E, S=S)
            cue = lab_forced(agent, meta, T, action, E=E, S=S, cue_at={WARM})
            A = ((cue - base) @ a).cpu().numpy() / sd
            out[cname][bname] = dict(peak=float(A[WARM:].max()), hl=half_life(A[WARM:]))
    return out


def summary():
    from scipy import stats as st
    L = ["# U1b open-loop control (exploratory): actions clamped identically in both body states", ""]
    for cname in CLAMPS:
        for grp, pat in (("agents", "runs/mortal_tau12_s[0-9]"), ("twins", "runs/twinbelief_tau12_s[0-9]")):
            R = [json.load(open(p)) for p in sorted(glob.glob(pat + "/u1b_openloop.json"))]
            row = []
            for test, key, other in (("T1 wounded", "peak", "wounded"), ("T1 hungry", "peak", "hungry"), ("T2 wounded", "hl", "wounded")):
                x = np.array([r[cname][other][key] - r[cname]["healthy"][key] for r in R], float)
                x = x[np.isfinite(x)]
                t, p = st.ttest_1samp(x, 0.0)
                row.append(f"{test} {x.mean():+.3f} (n={len(x)}, p={p:.2g})")
            hh = np.nanmean([r[cname]["healthy"]["hl"] for r in R])
            hw = np.nanmean([r[cname]["wounded"]["hl"] for r in R])
            L.append(f"- clamp {cname}, {grp}: " + "; ".join(row) + f"; half-life healthy {hh:.2f}, wounded {hw:.2f}")
    os.makedirs("results", exist_ok=True)
    open("results/u1b_openloop.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary()
    else:
        run = sys.argv[1].rstrip("/")
        json.dump(measure(run), open(os.path.join(run, "u1b_openloop.json"), "w"))
        print(run, "done", flush=True)
