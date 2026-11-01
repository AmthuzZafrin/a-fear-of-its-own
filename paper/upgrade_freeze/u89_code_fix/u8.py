"""Stage U8 (HYPOTHESES.md, 31 Oct 2026): the stakes sign in the decision, and the stakes law.
For a VitalWorld agent or belief-only twin, in the lab with the actions fixed (always foraging, or always hiding;
heart rate held) and the body clamped healthy or wounded, a single warning. Records the policy's preference
ln P(hide) - ln P(forage) (population mean), the frozen threat-axis response and the critic's value, and the
critic's value at the clamped integrity levels needed for the loss ratio L_w / L_h.
python -m vw.u8 runs/mortal_tau4_s0          (writes u8.json there; needs u1b.json for the axis units)
python -m vw.u8 runs/mortal_tau4_s0 --untrained   (code test only: a random network; writes nothing)
"""
import json
import os
import sys

import numpy as np
import torch

from .env import FORAGE, HIDE, VitalWorld
from .probes import DEV, POST, WARM, half_life, load, natural
from .u1b import frozen_axis
from .u1b_openloop import CLAMPS

BODIES = dict(healthy=(0.8, 1.0), wounded=(0.8, 0.4))


@torch.no_grad()
def lab_full(agent, meta, steps, action, n=2048, cue_at=(), E=0.8, S=1.0, seed=5):
    """As vw.u5.lab_forced_pref (actions fixed, body clamped, no real danger); also returns the mean value."""
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
    hm, pref, val = [], [], []
    for t in range(steps):
        h = agent.core(obs, h, env.S)
        hm.append(h.mean(0))
        dist, v = agent.heads(h)
        pm = dist.probs.view(n, -1, 3).sum(2).mean(0)
        pref.append(torch.log(pm[HIDE] + 1e-9) - torch.log(pm[FORAGE] + 1e-9))
        val.append(v.mean())
        env.step(act, force_cue=one if t + 1 in cue_at else zero, force_d=zero.bool(), force_obj=(zero, zero),
                 auto_reset=False)
        env.E[:], env.S[:] = E, S
        obs = env.obs()
    return torch.stack(hm), torch.stack(pref).cpu().numpy(), torch.stack(val).cpu().numpy()


def alarm_duration(pref):
    """Steps after the warning (seen at step WARM) until the preference first falls below 0, linearly interpolated.
    None if the preference is already >= 0 at the step before the warning (the network always hides);
    0 if it is below 0 at the warning step itself; POST if it never falls below 0 within the window (capped)."""
    if pref[WARM - 1] >= 0:
        return None, "always hides"
    if pref[WARM] < 0:
        return 0.0, "ok"
    for t in range(WARM + 1, WARM + POST + 1):
        if pref[t] < 0:
            return float((t - 1 - WARM) + pref[t - 1] / (pref[t - 1] - pref[t])), "ok"
    return float(POST), "capped"


def measure(run, untrained=False):
    agent, meta = load(run, untrained=untrained)
    mem = meta["mem_hidden"] if meta.get("twin") == "belief" else agent.hidden
    a = frozen_axis(agent, meta, mem)
    if os.path.exists(os.path.join(run, "u1b.json")):
        sd = json.load(open(os.path.join(run, "u1b.json")))["sd"]
    else:                      # fix of 31 Oct, 4:42 pm: units as in vw/u1b.py (s.d. of the natural-behaviour projection)
        sd = (natural(agent, meta)["h"].reshape(-1, agent.hidden) @ a).std().item()
    T = WARM + 1 + POST
    dmg = float(meta.get("dmg") or 0.25)
    out = dict(run=run, twin=meta.get("twin") == "belief", seed=meta["seed"], tau=float(meta["tau"]), dmg=dmg)
    for cname, action in CLAMPS.items():
        out[cname] = {}
        for bname, (E, S) in BODIES.items():
            base, pb, vb = lab_full(agent, meta, T, action, E=E, S=S)
            cue, pc, vc = lab_full(agent, meta, T, action, E=E, S=S, cue_at={WARM})
            A = ((cue - base) @ a).cpu().numpy() / sd
            seg = A[WARM:WARM + POST]
            pk = float(seg.max())
            hl = half_life(A[WARM:])
            D, status = alarm_duration(pc)
            out[cname][bname] = dict(D=D, D_status=status, pref_before=float(pc[WARM - 1]), pref_cue=pc[WARM - 2:].tolist(),
                                     pref_base=pb[WARM - 2:].tolist(), peak=pk, hl=hl,
                                     hl_at_window_end=bool(hl >= len(A[WARM:]) - int(np.argmax(A[WARM:])) - 0.5),
                                     area=float(seg.sum() / pk) if pk >= 0.02 else float("nan"),
                                     value_before=float(vb[WARM - 10:WARM].mean()))
    # the critic's value at clamped integrity levels (always foraging, no warning, end of warm-up)
    vals = {}
    for S in (1.0, 1.0 - dmg, 0.4, 0.4 - dmg):
        if S <= 0:
            vals[f"{S:.2f}"] = 0.0                     # an attack would kill: nothing more to gain
        else:
            _, _, v = lab_full(agent, meta, WARM, CLAMPS["forage"], E=0.8, S=S)
            vals[f"{S:.2f}"] = float(v[WARM - 10:WARM].mean())
    out["values"] = vals
    out["L_h"] = vals[f"{1.0:.2f}"] - vals[f"{1.0 - dmg:.2f}"]
    out["L_w"] = vals[f"{0.4:.2f}"] - vals[f"{0.4 - dmg:.2f}"]
    return out


if __name__ == "__main__":
    run = sys.argv[1].rstrip("/")
    if "--untrained" in sys.argv:
        o = measure(run, untrained=True)
        print("code test (untrained network):", {k: o[k] for k in ("L_h", "L_w")},
              {c: {b: (o[c][b]["D"], o[c][b]["D_status"]) for b in BODIES} for c in CLAMPS})
    else:
        json.dump(measure(run), open(os.path.join(run, "u8.json"), "w"))
        print(run, "u8 done", flush=True)
