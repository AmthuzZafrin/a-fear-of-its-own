"""Stage U3 (Experiment 6) measures: the U1b measure and the fixed-action control on copycats and untrained
networks, and the transplant measure on belief-only twins and untrained networks. The U1b, fixed-action and
transplant code is called unchanged; the two wrappers below only take a network that is passed in.
python -m vw.u3 u1b runs/twin_gru_tau12_s0                    (writes u1b.json and u1b_openloop.json there)
python -m vw.u3 u1b runs/mortal_tau12_s0 --untrained          (writes u1b_untrained.json, u1b_openloop_untrained.json)
python -m vw.u3 transplant runs/twinbelief_tau12_s0           (writes transplant.json)
python -m vw.u3 transplant runs/mortal_tau12_s0 --untrained   (writes transplant_untrained.json)
"""
import json
import os
import sys

from .probes import POST, WARM, half_life, load, natural
from .transplant import DMGS
from .u1b import frozen_axis
from .u1b import measure as u1b_measure
from .u1b_openloop import BODIES, CLAMPS, lab_forced


def openloop(agent, meta, mem, sd):
    """As vw.u1b_openloop.measure, for a network passed in."""
    a = frozen_axis(agent, meta, mem)
    T = WARM + 1 + POST
    out = {}
    for cname, action in CLAMPS.items():
        out[cname] = {}
        for bname, (E, S) in BODIES.items():
            base = lab_forced(agent, meta, T, action, E=E, S=S)
            cue = lab_forced(agent, meta, T, action, E=E, S=S, cue_at={WARM})
            A = ((cue - base) @ a).cpu().numpy() / sd
            out[cname][bname] = dict(peak=float(A[WARM:].max()), hl=half_life(A[WARM:]))
    return out


def transplant(agent, meta):
    """As vw.transplant.run_transplant, for a network passed in."""
    out = dict(mode=meta["mode"], seed=meta["seed"], twin=meta.get("twin"), home_dmg=meta.get("dmg", 0.25), worlds={})
    for d in DMGS:
        nat = natural(agent, dict(meta, dmg=d), n=1024, steps=700)
        danger = nat["d"].reshape(-1) > 0.5
        pf = nat["pf"].reshape(-1)
        out["worlds"][str(d)] = dict(life=nat["life"], forage_in_danger=float(pf[danger].mean()),
                                     forage_in_safety=float(pf[~danger].mean()),
                                     mean_integrity=float(nat["S"].mean()), mean_energy=float(nat["E"].mean()))
    return out


if __name__ == "__main__":
    what, run = sys.argv[1], sys.argv[2].rstrip("/")
    untrained = "--untrained" in sys.argv
    agent, meta = load(run, untrained=untrained)
    sfx = "_untrained" if untrained else ""
    if what == "u1b":
        mem = meta["mem_hidden"] if meta.get("twin") == "belief" else agent.hidden
        R = dict(run=run, mode=meta["mode"], tau=meta["tau"], seed=meta["seed"], dmg=meta.get("dmg") or 0.25,
                 twin=meta.get("twin"), untrained=untrained, **u1b_measure(agent, meta, mem))
        json.dump(R, open(os.path.join(run, f"u1b{sfx}.json"), "w"))
        O = dict(run=run, twin=meta.get("twin"), untrained=untrained, **openloop(agent, meta, mem, R["sd"]))
        json.dump(O, open(os.path.join(run, f"u1b_openloop{sfx}.json"), "w"))
    elif what == "transplant":
        json.dump(transplant(agent, meta), open(os.path.join(run, f"transplant{sfx}.json"), "w"))
    else:
        raise SystemExit(f"unknown measure {what}")
    print(run, what, "untrained" if untrained else "trained", "done", flush=True)
