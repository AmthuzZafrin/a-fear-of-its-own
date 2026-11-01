"""Move Score runs trained with the original (pilot-1) exchange rates to the
score_cheap condition. They are recognised by the absence of `score_rates` in
log.json (that field was added together with the corrected rates)."""
import glob
import json
import os

for d in sorted(glob.glob("runs/score_tau*_s*")):
    f = os.path.join(d, "log.json")
    if not os.path.exists(f):
        continue
    L = json.load(open(f))
    if "score_rates" in L:
        continue                      # corrected run, leave it
    L["mode"] = "score_cheap"
    json.dump(L, open(f, "w"))
    p = os.path.join(d, "probes.json")
    if os.path.exists(p):
        R = json.load(open(p))
        R["mode"] = "score_cheap"
        json.dump(R, open(p, "w"))
    new = d.replace("runs/score_", "runs/score_cheap_")
    os.rename(d, new)
    print("moved", d, "->", new)
