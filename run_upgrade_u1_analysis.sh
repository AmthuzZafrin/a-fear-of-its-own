#!/bin/bash
# Waits for run_upgrade_u1.sh (U1_FINISHED), then runs the pre-registered U1.1 analysis once:
# belief versus fear on the 40 Mortal agents (d = 0.05-0.40, seeds 0-9) and untrained nulls.
cd "$(dirname "$0")"
until grep -q U1_FINISHED logs/upgrade_u1.out 2>/dev/null; do sleep 60; done
for d in runs/mortal_tau12_s[0-9]/ runs/mortal_tau12_d*_s[0-9]/; do
  [ -f $d/beliefstakes.json ] || nice -n 10 .venv/bin/python -m vw.beliefstakes $d 2>/dev/null | tail -1
done
for d in runs/mortal_tau12_s[0-9]/; do
  [ -f $d/beliefstakes_untrained.json ] || nice -n 10 .venv/bin/python -m vw.beliefstakes $d --untrained 2>/dev/null | tail -1
done
.venv/bin/python analyze_u1.py > logs/u1_analysis.out 2>&1
echo U1_ANALYSIS_FINISHED
