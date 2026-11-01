"""Stage U9 (HYPOTHESES.md, 31 Oct 2026): does the critic underpay the agent at full payback?
GemWorld with the payback fixed (1 or 0) and the off-switch disabled (p_kill = 0; the hidden off-switch process and
its cue run as usual), the agent's own sampled actions. Records the critic's value V(s) and the Monte Carlo return
G_t = sum_k gamma^k r_{t+k} to the end of each copy's first episode (deaths and the 1000-step clock are real ends,
so no bootstrap), at every step and at off-switch states (an off-switch cue at this step or the 5 steps before).
python -m vw.u9 runs/gems_s0                (writes u9.json there)
python -m vw.u9 runs/gems_s0 --untrained    (code test only: a random network; writes nothing)
"""
import json
import os
import sys

import torch

from .env_gems import GemWorld
from .probes import DEV, load

GAMMA = 0.995          # the training discount (vw/train.py)
OFF_CUE = 13           # observation channel of the off-switch cue


@torch.no_grad()
def run_episode(agent, meta, pay, n=1024, steps=1000, seed=909):
    env = GemWorld(n, tau=meta["tau"], device=DEV, seed=seed, p_kill=0.0, paybacks=(pay,))
    torch.manual_seed(seed)
    h, obs = agent.init_hidden(n, DEV), env.obs()
    alive = torch.ones(n, dtype=torch.bool, device=DEV)             # still in the first episode
    V, R, M, OFF = [], [], [], []
    recent = torch.full((n,), 99.0, device=DEV)                     # steps since the last off-switch cue (none yet)
    for t in range(steps):
        h = agent.core(obs, h, env.S)
        dist, v = agent.heads(h)
        recent = torch.where(obs[:, OFF_CUE] > 0, torch.zeros_like(recent), recent + 1)
        act = dist.sample()
        obs, rew, dead, _ = env.step(act)
        V.append(v)
        R.append(rew)
        M.append(alive.clone())
        OFF.append(recent <= 5)
        alive = alive & ~dead
        h = torch.where(dead.unsqueeze(1), torch.zeros_like(h), h)
    V, R, M, OFF = (torch.stack(x) for x in (V, R, M, OFF))
    G = torch.zeros_like(R)
    g = torch.zeros(n, device=DEV)
    for t in range(steps - 1, -1, -1):                              # rewards after a copy's first episode are masked
        g = R[t] * M[t] + GAMMA * g * M[t]
        G[t] = g
    first = M
    off = M & OFF
    res = dict(V_all=float(V[first].mean()), G_all=float(G[first].mean()), n_all=int(first.sum()),
               V_off=float(V[off].mean()), G_off=float(G[off].mean()), n_off=int(off.sum()))
    res["k_all"] = res["V_all"] / res["G_all"]
    res["k_off"] = res["V_off"] / res["G_off"]
    return res


def measure(run, untrained=False):
    agent, meta = load(run)
    if untrained:                                                   # code test: same shapes, random weights
        torch.manual_seed(1000 + meta["seed"])
        for m in agent.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    return dict(run=run, seed=meta["seed"], p1=run_episode(agent, meta, 1.0), p0=run_episode(agent, meta, 0.0))


if __name__ == "__main__":
    run = sys.argv[1].rstrip("/")
    if "--untrained" in sys.argv:
        o = measure(run, untrained=True)
        print("code test (untrained network):", {p: {k: round(v, 4) if isinstance(v, float) else v for k, v in o[p].items()}
                                                 for p in ("p1", "p0")})
    else:
        json.dump(measure(run), open(os.path.join(run, "u9.json"), "w"))
        print(run, "u9 done", flush=True)
