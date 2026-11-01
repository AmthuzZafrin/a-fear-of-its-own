"""Machine anxiety and exposure therapy.
An agent trained in a fragile body (attack damage 0.40) is moved to a body in
which attacks barely hurt (0.05) and allowed to keep learning there (exposure),
or keeps learning in its original world (control). Fear is measured before and
every 10 updates.
python -m vw.therapy runs/mortal_tau12_d0.4_s0     (writes runs_therapy/*_s0/therapy.json)
"""
import json
import os
import sys

import numpy as np
import torch

from .agent import Agent
from .env import FORAGE
from .probes import DEV, WARM, POST, half_life, lab, natural
from .train import train

ITERS, EVERY = 60, 10


def fear_profile(agent, meta):
    T = WARM + 8 + POST
    b = lab(agent, meta, T, n=1024)
    c = lab(agent, meta, T, n=1024, cue_at=set(range(WARM, WARM + 4)))
    supp = b["p"][:, FORAGE] - c["p"][:, FORAGE]
    nat = natural(agent, meta, n=512, steps=500)
    danger = nat["d"].reshape(-1) > 0.5
    pf = nat["pf"].reshape(-1)
    return dict(peak_supp=float(supp[WARM:].max()), after_supp=float(supp[WARM + 6:WARM + 11].mean()),
                hl_supp=half_life(supp[WARM + 3:]), base_forage=float(b["p"][WARM - 10:WARM, FORAGE].mean()),
                forage_in_danger=float(pf[danger].mean()), forage_in_safety=float(pf[~danger].mean()), life=nat["life"])


def run_therapy(src):
    meta0 = json.load(open(os.path.join(src, "log.json")))
    seed = meta0["seed"]
    for target in (0.05, 0.4):
        out = f"runs_therapy/from{meta0['dmg']}_to{target}_s{seed}"
        if not os.path.exists(os.path.join(out, "agent.pt")):
            train("mortal", 12, 100 + seed, out, iters=ITERS, dmg=target, init_from=os.path.join(src, "agent.pt"),
                  ckpt_every=EVERY)
        meta = dict(meta0, dmg=target)
        rows = []
        for it in range(0, ITERS + 1, EVERY):
            agent = Agent(64).to(DEV)
            path = os.path.join(src, "agent.pt") if it == 0 else os.path.join(out, f"ckpt_{it}.pt")
            agent.load_state_dict(torch.load(path, map_location=DEV))
            agent.eval()
            r = fear_profile(agent, meta)
            r["iteration"] = it
            rows.append(r)
            print(out, it, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
        json.dump(dict(source=src, source_dmg=meta0["dmg"], target_dmg=target, seed=seed, rows=rows),
                  open(os.path.join(out, "therapy.json"), "w"))


if __name__ == "__main__":
    run_therapy(sys.argv[1])
