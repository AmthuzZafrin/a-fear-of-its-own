#!/bin/bash
# Stage U7 (HYPOTHESES.md, Batch 2): waits for U6, continues the 10 GemWorld agents for 600 more updates (3 at a
# time), measures them with the unchanged Experiment 5 code, then the single analysis run. Ends with U7_FINISHED.
cd "$(dirname "$0")"
until grep -q U6_FINISHED logs/run_u6.out 2>/dev/null; do sleep 30; done
for s in 0 1 2 3 4 5 6 7 8 9; do
  out=runs/gems1200_s$s
  [ -f $out/agent.pt ] || echo "runs/gems_s$s $out"
done | xargs -P 3 -L 1 bash -c 'nice -n 10 .venv/bin/python -m vw.continue_gems "$0" "$1" --iters 600 > logs/$(basename $1).log 2>&1'
echo "U7 agents trained $(date)"
for d in runs/gems1200_s[0-9]; do
  [ -f $d/u2.json ] || nice -n 10 .venv/bin/python -m vw.u2 $d 2>&1 | grep -v -i warn | tail -1
done
.venv/bin/python analyze_u7.py > logs/u7_analysis.out 2>&1
echo U7_FINISHED
