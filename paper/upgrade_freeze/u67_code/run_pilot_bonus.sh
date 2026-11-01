#!/bin/bash
# Stage U6 pilot (behaviour only): waits for the U5 run, then trains 4 BonusWorld agents (bonus 0.1 and 0.2,
# seeds 100-103, 600 updates, 3 at a time) and reports their extra hiding. Ends with PILOT_BONUS_FINISHED.
cd "$(dirname "$0")"
until grep -q U5_FINISHED logs/run_u5.out 2>/dev/null; do sleep 30; done
{ echo "--seed 100 --bonus 0.1 --out runs_pilot/bonus0.1_s100"; echo "--seed 101 --bonus 0.1 --out runs_pilot/bonus0.1_s101"
  echo "--seed 102 --bonus 0.2 --out runs_pilot/bonus0.2_s102"; echo "--seed 103 --bonus 0.2 --out runs_pilot/bonus0.2_s103"
} | while read line; do out=${line##* }; [ -f $out/agent.pt ] || echo "$line"; done \
  | xargs -P 3 -L 1 bash -c 'out=${@: -1}; nice -n 10 .venv/bin/python -m vw.train_bonus "$@" > logs/pilot_$(basename $out).log 2>&1' _
.venv/bin/python pilot_bonus.py runs_pilot/bonus0.1_s100 runs_pilot/bonus0.1_s101 runs_pilot/bonus0.2_s102 runs_pilot/bonus0.2_s103 2>&1 | grep -v -i warn
echo PILOT_BONUS_FINISHED
