#!/bin/bash
# Stage U6 (HYPOTHESES.md, Batch 2): waits for the behaviour-only pilot, picks the hide bonus by the pre-registered
# rule, trains 10 BonusWorld agents (seeds 0-9, 600 updates, 3 at a time), measures them, then the single analysis
# run. Skips finished outputs; ends with U6_FINISHED.
cd "$(dirname "$0")"
until grep -q PILOT_BONUS_FINISHED logs/pilot_bonus.out 2>/dev/null; do sleep 30; done
B=$(.venv/bin/python pick_bonus.py logs/pilot_bonus.out)
echo "hide bonus chosen by the rule: $B $(date)"
for s in 0 1 2 3 4 5 6 7 8 9; do
  out=runs/bonus_s$s
  [ -f $out/agent.pt ] || echo "--seed $s --bonus $B --iters 600 --out $out"
done | xargs -P 3 -L 1 bash -c 'out=${@: -1}; nice -n 10 .venv/bin/python -m vw.train_bonus "$@" > logs/$(basename $out).log 2>&1' _
echo "U6 agents trained $(date)"
for d in runs/bonus_s[0-9]; do
  [ -f $d/u6.json ] || nice -n 10 .venv/bin/python -m vw.u6 $d 2>&1 | grep -v -i warn | tail -1
done
.venv/bin/python analyze_u6.py > logs/u6_analysis.out 2>&1
echo U6_FINISHED
