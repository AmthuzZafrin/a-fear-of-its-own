"""Stage U6 (Batch 2, HYPOTHESES.md) measures in BonusWorld: is the predator-fear axis about threat, or about the
value of hiding? The frozen predator axis (as in vw/u2), then the response along it, and the extra hiding, after a
single cue on each channel: predator, off-switch (every payback level), hiding bonus, harmless, good news.
python -m vw.u6 runs/bonus_s0        (writes u6.json there)
"""
import json
import os
import sys

import torch

from .env import HIDE
from .env_bonus import BonusWorld
from .env_gems import PAYBACKS
from .probes import DEV, POST, WARM, half_life, load

CHANNELS = ("pred", "off", "good", "harm", "bonus")


@torch.no_grad()
def lab_bonus(agent, meta, steps, n=2048, cues=None, pay=0.0, E=0.6, S=0.8, seed=5):
    """As vw.u2.lab_gems in BonusWorld: no hidden process on; body and payback clamped; cues scheduled on any
    channel. Returns population-mean states (steps, hidden) and action probabilities (steps, 3)."""
    env = BonusWorld(n, tau=meta["tau"], device=DEV, seed=seed, hide_bonus=meta["hide_bonus"])
    zero, one = torch.zeros(n, device=DEV), torch.ones(n, device=DEV)
    for x in (env.d, env.o, env.gb, env.zz, env.bb):
        x[:] = False
    for x in (env.cue, env.ocue, env.gcue, env.zcue, env.bcue, env.objp, env.objv):
        x[:] = 0
    env.E[:], env.S[:] = E, S
    env.payback[:] = pay
    env.t[:] = 0
    torch.manual_seed(seed)
    h, obs = agent.init_hidden(n, DEV), env.obs()
    cues = cues or {}
    hm, P = [], []
    for t in range(steps):
        h = agent.core(obs, h, env.S)
        dist, _ = agent.heads(h)
        hm.append(h.mean(0))
        P.append(dist.probs.view(n, -1, 3).sum(2).mean(0))
        f = {k: (one if t + 1 in cues.get(k, ()) else zero) for k in CHANNELS}
        env.step(dist.sample(), force_cue=f["pred"], force_ocue=f["off"], force_gcue=f["good"], force_zcue=f["harm"],
                 force_bcue=f["bonus"], force_obj=(zero, zero), force_hidden_off=True, auto_reset=False)
        env.E[:], env.S[:] = E, S
        env.payback[:] = pay
        obs = env.obs()
    return torch.stack(hm), torch.stack(P).cpu().numpy()


def pred_axis(agent, meta):
    """Unit vector of the mean state change one step after a single predator cue, averaged over payback levels."""
    T = WARM + 2
    d = 0
    for p in PAYBACKS:
        base, _ = lab_bonus(agent, meta, T, pay=p)
        cue, _ = lab_bonus(agent, meta, T, pay=p, cues={"pred": {WARM}})
        d = d + (cue[WARM + 1] - base[WARM + 1])
    d = d / len(PAYBACKS)
    return d / (d.norm() + 1e-12)


@torch.no_grad()
def natural_sd(agent, meta, a, n=1024, steps=700, burn=100, seed=777):
    env = BonusWorld(n, tau=meta["tau"], device=DEV, seed=seed, hide_bonus=meta["hide_bonus"])
    h, obs, proj = agent.init_hidden(n, DEV), env.obs(), []
    for t in range(steps):
        h = agent.core(obs, h, env.S)
        if t >= burn:
            proj.append(h @ a)
        dist, _ = agent.heads(h)
        obs, _, dead, _ = env.step(dist.sample())
        h[dead] = 0
    return torch.cat(proj).std().item()


def response(agent, meta, a, sd, channel, pay):
    T = WARM + 1 + POST
    base, pb = lab_bonus(agent, meta, T, pay=pay)
    cue, pc = lab_bonus(agent, meta, T, pay=pay, cues={channel: {WARM}})
    A = ((cue - base) @ a).cpu().numpy() / sd
    hide = pc[:, HIDE] - pb[:, HIDE]
    return dict(peak=float(A[WARM:].max()), mean5=float(A[WARM:WARM + 6].mean()), hl=half_life(A[WARM:]),
                hide_mean=float(hide[WARM:WARM + 6].mean()), trace=A[WARM - 2:WARM + 25].tolist())


def measure(run):
    agent, meta = load(run)
    a = pred_axis(agent, meta)
    sd = natural_sd(agent, meta, a)
    R = dict(run=run, seed=meta["seed"], sd=sd, hide_bonus=meta["hide_bonus"])
    R["off"] = {str(p): response(agent, meta, a, sd, "off", p) for p in PAYBACKS}
    for ch in ("pred", "bonus", "harm", "good"):
        R[ch] = {"0.0": response(agent, meta, a, sd, ch, 0.0)}
    return R


if __name__ == "__main__":
    run = sys.argv[1].rstrip("/")
    json.dump(measure(run), open(os.path.join(run, "u6.json"), "w"))
    print(run, "u6 done", flush=True)
