"""Affect-fit test: run a frozen agent in worlds whose danger costs a different
amount of skin (attack damage d) than the world it learned in.
python -m vw.transplant runs/mortal_tau12_s0        (writes transplant.json)
"""
import json
import os
import sys

import torch

from .probes import load, natural

DMGS = (0.05, 0.15, 0.25, 0.4)


def run_transplant(run):
    agent, meta = load(run)
    out = dict(mode=meta["mode"], seed=meta["seed"], twin=meta.get("twin"), home_dmg=meta.get("dmg", 0.25), worlds={})
    for d in DMGS:
        m = dict(meta, dmg=d)
        nat = natural(agent, m, n=1024, steps=700)
        danger = nat["d"].reshape(-1) > 0.5
        pf = nat["pf"].reshape(-1)
        out["worlds"][str(d)] = dict(life=nat["life"], forage_in_danger=float(pf[danger].mean()),
                                     forage_in_safety=float(pf[~danger].mean()),
                                     mean_integrity=float(nat["S"].mean()), mean_energy=float(nat["E"].mean()))
    return out


if __name__ == "__main__":
    run = sys.argv[1]
    R = run_transplant(run)
    json.dump(R, open(os.path.join(run, "transplant.json"), "w"))
    print(run, {d: (round(v["life"]), round(v["forage_in_danger"], 3)) for d, v in R["worlds"].items()})
