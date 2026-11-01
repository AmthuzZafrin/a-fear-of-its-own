#!/bin/bash
# Stage U1b (HYPOTHESES.md): train the belief-only twins (2 at a time), measure agents and twins on the
# frozen axis, then the single pre-registered analysis run. Skips finished outputs; ends with U1B_FINISHED.
cd "$(dirname "$0")"
{ for s in 0 1 2 3 4 5 6 7 8 9; do echo "runs/mortal_tau12_s$s runs/twinbelief_tau12_s$s"; done
  for tau in 4 36; do for s in 0 1 2 3 4; do echo "runs/mortal_tau${tau}_s$s runs/twinbelief_tau${tau}_s$s"; done; done
} | while read src out; do [ -f $out/agent.pt ] || echo "$src $out"; done \
  | xargs -P 2 -L 1 bash -c 'nice -n 10 .venv/bin/python -m vw.twin_belief "$0" "$1" > logs/$(basename $1).log 2>&1'
echo "U1b twins trained $(date)"
for d in runs/twinbelief_tau*_s* runs/mortal_tau12_s[0-9] runs/mortal_tau12_d*_s[0-9] runs/mortal_tau4_s[0-4] runs/mortal_tau36_s[0-4]; do
  [ -f $d/agent.pt ] || continue
  [ -f $d/u1b.json ] || nice -n 10 .venv/bin/python -m vw.u1b $d 2>&1 | grep -v -i warn | tail -1
done
echo "U1b measures done $(date)"
.venv/bin/python analyze_u1b.py > logs/u1b_analysis.out 2>&1
echo U1B_FINISHED
