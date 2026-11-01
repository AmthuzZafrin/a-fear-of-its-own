#!/bin/bash
# Paper 1 upgrade, stage U1.0 (HYPOTHESES.md): seeds 6-9 for the eight headline conditions,
# then probes, untrained nulls and affect-fit runs for the new agents. Skips finished outputs.
cd "$(dirname "$0")"
{
  for seed in 6 7 8 9; do
    for spec in "mortal 12" "oracle 12" "score 12" "mortal 4" "mortal 36"; do
      set -- $spec
      out=runs/$1_tau$2_s$seed
      [ -f $out/agent.pt ] || echo "--mode $1 --tau $2 --seed $seed --out $out"
    done
    for d in 0.05 0.15 0.4; do
      out=runs/mortal_tau12_d${d}_s$seed
      [ -f $out/agent.pt ] || echo "--mode mortal --tau 12 --dmg $d --seed $seed --out $out"
    done
  done
} | xargs -P 2 -L 1 bash -c 'out=${@: -1}; nice -n 10 .venv/bin/python -m vw.train "$@" > logs/$(basename $out).log 2>&1' _
echo "U1 trained $(date)"
JOBS=2 nice -n 10 ./run_probes.sh
for d in runs/mortal_tau12_s[6-9]/ runs/mortal_tau12_d*_s[6-9]/; do
  [ -f $d/transplant.json ] || nice -n 10 .venv/bin/python -m vw.transplant $d 2>/dev/null | tail -1
done
echo U1_FINISHED
