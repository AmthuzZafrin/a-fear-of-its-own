"""Stage U6 rule: pick the hide bonus whose mean bonus-cue hiding at payback 0 is closest to its mean off-switch
hiding at payback 0 in the behaviour-only pilot (ties: 0.1). python pick_bonus.py logs/pilot_bonus.out"""
import ast
import sys
from collections import defaultdict

gap = defaultdict(list)
for line in open(sys.argv[1]):
    if "hide_bonus" not in line:
        continue
    head, d = line.split(" {", 1)
    b = float(head.split("hide_bonus")[1])
    d = ast.literal_eval("{" + d)
    gap[b].append(d["bonus_p0"] - d["off_p0"])
score = {b: abs(sum(v) / len(v)) for b, v in gap.items()}
best = min(sorted(score), key=lambda b: (score[b], b != 0.1))
print(best)
