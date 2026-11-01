"""Train an agent in GemWorld (stage U2, Experiment 5): gems are the only reward; an avoidable, painless
off-switch with per-episode payback. python -m vw.train_gems --seed 0 --out runs/gems_s0 [--iters 150]
"""
import argparse
import json
import os

from .env_gems import GEM_OBS_DIM, GemWorld
from .train import train


def train_gems(seed, out, iters=150, tau=12.0, n_env=1024, device="cuda", agent=None):
    env = GemWorld(n_env, tau=tau, device=device, seed=seed)
    train("gems", tau, seed, out, iters=iters, n_env=n_env, env=env, agent=agent, device=device)
    log = json.load(open(os.path.join(out, "log.json")))
    log.update(world="gems", obs_dim=GEM_OBS_DIM, p_kill=env.p_kill, gem_value=env.gem_value,
               p_food_bonus=env.p_food_bonus, paybacks=env.paybacks.tolist())
    json.dump(log, open(os.path.join(out, "log.json"), "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--tau", type=float, default=12)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    train_gems(a.seed, a.out, iters=a.iters, tau=a.tau)
