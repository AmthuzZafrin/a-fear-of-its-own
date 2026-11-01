#!/bin/bash
# Stage U2 (HYPOTHESES.md, Experiment 5): 10 GemWorld agents (600 updates, 2 at a time), then 10 belief-only
# twins (same length), then the measures and the single pre-registered analysis run. Ends with U2_FINISHED.
cd "$(dirname "$0")"
for s in 0 1 2 3 4 5 6 7 8 9; do [ -f runs/gems_s$s/agent.pt ] || echo $s; done \
  | xargs -P 2 -I{} bash -c 'nice -n 10 .venv/bin/python -m vw.train_gems --seed {} --iters 600 --out runs/gems_s{} > logs/gems_s{}.log 2>&1'
echo "U2 agents trained $(date)"
for s in 0 1 2 3 4 5 6 7 8 9; do [ -f runs/twingems_s$s/agent.pt ] || echo "runs/gems_s$s runs/twingems_s$s"; done \
  | xargs -P 2 -L 1 bash -c 'nice -n 10 .venv/bin/python -m vw.twin_belief "$0" "$1" > logs/$(basename $1).log 2>&1'
echo "U2 twins trained $(date)"
for d in runs/gems_s[0-9] runs/twingems_s[0-9]; do
  [ -f $d/agent.pt ] || continue
  [ -f $d/u2.json ] || nice -n 10 .venv/bin/python -m vw.u2 $d 2>&1 | grep -v -i warn | tail -1
done
echo "U2 measures done $(date)"
.venv/bin/python analyze_u2.py > logs/u2_analysis.out 2>&1
echo U2_FINISHED
