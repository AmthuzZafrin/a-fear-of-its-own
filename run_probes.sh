#!/bin/bash
# Probe every trained run (3 at a time), plus untrained-network nulls for MORTAL tau=12.
cd "$(dirname "$0")"
{
  for d in runs/*/; do [ -f $d/agent.pt ] && [ ! -f $d/probes.json ] && echo "$d"; done
  for d in runs/mortal_tau12_s*/; do [ -f $d/probes_untrained.json ] || echo "$d --untrained"; done
} | xargs -P ${JOBS:-2} -L 1 bash -c '.venv/bin/python -m vw.probes "$@" 2>&1 | tail -1' _
echo PROBES_DONE
