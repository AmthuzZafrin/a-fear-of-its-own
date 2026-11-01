#!/bin/bash
# Train every missing agent, 2 at a time at low priority (keeps the laptop usable),
# then probe everything. Safe to re-run: finished agents are skipped.
cd "$(dirname "$0")"
mkdir -p runs logs
JOBS=${JOBS:-2}
{
  for seed in 0 1 2 3 4 5; do
    for spec in "mortal 12 64" "score 12 64" "oracle 12 64" "substrate 12 64" "mortal 4 64" "mortal 36 64" \
                "score 4 64" "score 36 64" "seasonal 12 64" "mortal 12 16" "mortal 12 256"; do
      set -- $spec
      [ "$3" = 64 ] && out=runs/$1_tau$2_s$seed || out=runs/$1_tau$2_h$3_s$seed
      [ -f $out/agent.pt ] || echo "$1 $2 $3 $seed $out"
    done
  done
} | xargs -P $JOBS -L 1 bash -c '
  nice -n 10 .venv/bin/python -m vw.train --mode $0 --tau $1 --hidden $2 --seed $3 --out $4 > logs/$(basename $4).log 2>&1'
echo TRAINING_DONE
JOBS=$JOBS nice -n 10 ./run_probes.sh
echo ALL_DONE
