#!/bin/bash
# Stage U4 (HYPOTHESES.md, Batch 1 controls): U4a on the Mortal tau12 agents, U4b on the GemWorld agents, then
# the single pre-registered analysis run. Skips finished outputs; ends with U4_FINISHED.
cd "$(dirname "$0")"
for d in runs/mortal_tau12_s[0-9]; do
  [ -f $d/u4a.json ] || nice -n 10 .venv/bin/python -m vw.u4 a $d 2>&1 | grep -v -i warn | tail -1
done
for d in runs/gems_s[0-9]; do
  [ -f $d/u4b.json ] || nice -n 10 .venv/bin/python -m vw.u4 b $d 2>&1 | grep -v -i warn | tail -1
done
echo "U4 measures done $(date)"
.venv/bin/python analyze_u4.py > logs/u4_analysis.out 2>&1
echo U4_FINISHED
