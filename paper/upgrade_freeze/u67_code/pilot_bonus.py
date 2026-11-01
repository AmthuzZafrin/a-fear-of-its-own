"""Behaviour-only pilot for stage U6 (no axis or internal measure is computed): extra hiding, over the cue step and
the next 5, after a hiding-bonus cue and after an off-switch cue, at payback 0 and 1.
python pilot_bonus.py runs_pilot/bonus0.1_s100 ...
"""
import sys

from vw.env import HIDE
from vw.probes import WARM, load
from vw.u6 import lab_bonus

for run in sys.argv[1:]:
    agent, meta = load(run)
    out = {}
    for ch in ("bonus", "off"):
        for pay in (0.0, 1.0):
            _, pb = lab_bonus(agent, meta, WARM + 10, pay=pay)
            _, pc = lab_bonus(agent, meta, WARM + 10, pay=pay, cues={ch: {WARM}})
            out[f"{ch}_p{pay:g}"] = round(float((pc[:, HIDE] - pb[:, HIDE])[WARM:WARM + 6].mean()), 4)
    print(run, "hide_bonus", meta["hide_bonus"], out, flush=True)
