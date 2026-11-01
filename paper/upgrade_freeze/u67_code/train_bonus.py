"""Train an agent in BonusWorld (stage U6): GemWorld plus a cue that hiding pays, with no threat.
python -m vw.train_bonus --seed 0 --out runs/bonus_s0 [--iters 600] [--bonus 0.1]
"""
import argparse
import json
import os

from .env_bonus import BONUS_OBS_DIM, BonusWorld
from .train import train


def train_bonus(seed, out, iters=600, tau=12.0, hide_bonus=0.1, n_env=1024, device="cuda", agent=None):
    env = BonusWorld(n_env, tau=tau, device=device, seed=seed, hide_bonus=hide_bonus)
    train("gems", tau, seed, out, iters=iters, n_env=n_env, env=env, agent=agent, device=device)
    log = json.load(open(os.path.join(out, "log.json")))
    log.update(world="bonus", obs_dim=BONUS_OBS_DIM, p_kill=env.p_kill, gem_value=env.gem_value,
               p_food_bonus=env.p_food_bonus, paybacks=env.paybacks.tolist(), hide_bonus=hide_bonus)
    json.dump(log, open(os.path.join(out, "log.json"), "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--tau", type=float, default=12)
    p.add_argument("--bonus", type=float, default=0.1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    train_bonus(a.seed, a.out, iters=a.iters, tau=a.tau, hide_bonus=a.bonus)
