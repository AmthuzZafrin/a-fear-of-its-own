#!/usr/bin/env bash
# Stages U8 and U9 (HYPOTHESES.md, 31 Oct 2026): the measures, then the single pre-registered analyses.
# At most three networks are measured at once. Each measure skips networks that already have their output.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p logs
echo "=== U8/U9 started $(date) ===" >> logs/run_u89.out
ls -d runs/mortal_tau4_s[0-9] runs/mortal_tau12_s[0-9] runs/mortal_tau36_s[0-9] \
      runs/twinbelief_tau4_s[0-9] runs/twinbelief_tau12_s[0-9] runs/twinbelief_tau36_s[0-9] \
      runs/mortal_tau12_d0.05_s[0-9] runs/mortal_tau12_d0.15_s[0-9] runs/mortal_tau12_d0.4_s[0-9] |
    xargs -P 3 -I{} sh -c "[ -f {}/u8.json ] || $PY -m vw.u8 {} >> logs/u8.log 2>&1"
echo "U8 measures done $(date): $(ls runs/*/u8.json | wc -l) files" >> logs/run_u89.out
ls -d runs/gems_s[0-9] | xargs -P 3 -I{} sh -c "[ -f {}/u9.json ] || $PY -m vw.u9 {} >> logs/u9.log 2>&1"
echo "U9 measures done $(date): $(ls runs/gems_s*/u9.json | wc -l) files" >> logs/run_u89.out
$PY analyze_u8.py > logs/analyze_u8.log 2>&1 && echo "analyze_u8 done $(date)" >> logs/run_u89.out
$PY analyze_u9.py > logs/analyze_u9.log 2>&1 && echo "analyze_u9 done $(date)" >> logs/run_u89.out
echo "U89_FINISHED $(date)" >> logs/run_u89.out
