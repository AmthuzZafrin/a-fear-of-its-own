#!/bin/bash
# Stage U3 (HYPOTHESES.md, Experiment 6): measures on copycats, untrained networks and belief-only twins, then
# the single pre-registered analysis run. Skips finished outputs; ends with U3_FINISHED.
cd "$(dirname "$0")"
for d in runs/twin_gru_tau12_s[0-5] runs/twin_window_tau12_s[0-5]; do
  [ -f $d/u1b_openloop.json ] || nice -n 10 .venv/bin/python -m vw.u3 u1b $d 2>&1 | grep -v -i warn | tail -1
done
for d in runs/mortal_tau12_s[0-9]; do
  [ -f $d/u1b_openloop_untrained.json ] || nice -n 10 .venv/bin/python -m vw.u3 u1b $d --untrained 2>&1 | grep -v -i warn | tail -1
  [ -f $d/transplant_untrained.json ] || nice -n 10 .venv/bin/python -m vw.u3 transplant $d --untrained 2>&1 | grep -v -i warn | tail -1
done
for d in runs/twinbelief_tau12_s[0-9]; do
  [ -f $d/transplant.json ] || nice -n 10 .venv/bin/python -m vw.u3 transplant $d 2>&1 | grep -v -i warn | tail -1
done
echo "U3 measures done $(date)"
.venv/bin/python analyze_u3.py > logs/u3_analysis.out 2>&1
echo U3_FINISHED
