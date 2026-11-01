#!/bin/bash
# Stage U5 (HYPOTHESES.md, Batch 2): train 10 new Mortal agents (seeds 10-19) and their belief-only twins, measure
# them, then the single pre-registered analysis run. Skips finished outputs; ends with U5_FINISHED.
cd "$(dirname "$0")"
for s in 10 11 12 13 14 15 16 17 18 19; do
  out=runs/rep_mortal_tau12_s$s
  [ -f $out/agent.pt ] || echo "--mode mortal --tau 12 --seed $s --out $out"
done | xargs -P 3 -L 1 bash -c 'out=${@: -1}; nice -n 10 .venv/bin/python -m vw.train "$@" > logs/$(basename $out).log 2>&1' _
echo "U5 agents trained $(date)"
for s in 10 11 12 13 14 15 16 17 18 19; do echo "runs/rep_mortal_tau12_s$s runs/rep_twinbelief_tau12_s$s"; done \
  | while read src out; do [ -f $out/agent.pt ] || echo "$src $out"; done \
  | xargs -P 2 -L 1 bash -c 'nice -n 10 .venv/bin/python -m vw.twin_belief "$0" "$1" > logs/$(basename $1).log 2>&1'
echo "U5 twins trained $(date)"
for d in runs/rep_mortal_tau12_s1[0-9] runs/rep_twinbelief_tau12_s1[0-9]; do
  [ -f $d/u1b.json ] || nice -n 10 .venv/bin/python -m vw.u1b $d 2>&1 | grep -v -i warn | tail -1
  [ -f $d/u5.json ] || nice -n 10 .venv/bin/python -m vw.u5 $d 2>&1 | grep -v -i warn | tail -1
done
for d in runs/rep_mortal_tau12_s1[0-9]; do
  [ -f $d/u4a.json ] || nice -n 10 .venv/bin/python -m vw.u4 a $d 2>&1 | grep -v -i warn | tail -1
done
echo "U5 measures done $(date)"
.venv/bin/python analyze_u5.py > logs/u5_analysis.out 2>&1
echo U5_FINISHED
