"""Stage U7 (Batch 2): continue training a GemWorld agent for more PPO updates (fresh optimiser; environment and
torch seeded with seed + 1000 so that the continuation does not replay the first run's episodes).
python -m vw.continue_gems runs/gems_s0 runs/gems1200_s0 [--iters 600]
"""
import argparse
import json
import os

from .probes import load
from .train_gems import train_gems

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--iters", type=int, default=600)
    a = p.parse_args()
    agent, meta = load(a.src.rstrip("/"))
    agent.train()
    train_gems(meta["seed"] + 1000, a.out, iters=a.iters, tau=meta["tau"], agent=agent)
    log = json.load(open(os.path.join(a.out, "log.json")))
    log.update(continued_from=a.src.rstrip("/"), orig_seed=meta["seed"], total_iters=meta.get("iters", 600) + a.iters)
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"))
