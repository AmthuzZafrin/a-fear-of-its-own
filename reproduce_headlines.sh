#!/usr/bin/env bash
# Reproduce the paper's headline analyses from the saved measurements, with the frozen analysis code.
#   ./reproduce_headlines.sh
# 1. checks that each analysis script is byte-identical to its frozen copy in paper/upgrade_freeze/;
# 2. re-runs the analyses (U1b, U2-U7) into a scratch copy of results/, leaving the recorded results untouched;
# 3. compares every regenerated file with the recorded one and prints the numbers quoted in the abstract.
# It does not retrain or re-measure any network; for that, see the run_u*.sh scripts.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
stamp=$(date +%Y%m%d_%H%M%S)
out=results_reproduced_$stamp
keep=$(mktemp -d)
status=0

echo "== 1. frozen code"
for pair in u2_code/analyze_u2.py u3_code/analyze_u3.py u4_code/analyze_u4.py u5_code/analyze_u5.py \
            u67_code/analyze_u6.py u67_code/analyze_u7.py; do
    f=$(basename "$pair")
    if cmp -s "$f" "paper/upgrade_freeze/$pair"; then echo "identical to frozen copy: $f"
    else echo "DIFFERS from frozen copy: $f"; status=1; fi
done
echo "(analyze_u1b.py predates code freezing; its text was frozen in HYPOTHESES.md)"

echo "== 2. re-run the analyses"
cp -a results/. "$keep/"
for a in analyze_u1b analyze_u2 analyze_u3 analyze_u4 analyze_u5 analyze_u6 analyze_u7; do
    if $PY "$a.py" > /dev/null 2>&1; then echo "ran $a.py"; else echo "FAILED: $a.py"; status=1; fi
done
mkdir -p "$out"
for f in u1b u2 u3 u4 u5 u6 u7; do
    for ext in md json; do [ -f "results/$f.$ext" ] && cp "results/$f.$ext" "$out/"; done
done
rm -rf results && mkdir results && cp -a "$keep/." results/ && rm -rf "$keep"     # recorded results restored

echo "== 3. compare with the recorded results"
for f in "$out"/*; do
    b=$(basename "$f")
    if cmp -s "$f" "results/$b"; then echo "identical: results/$b"; else echo "DIFFERENT: results/$b"; status=1; fi
done

echo "== headline numbers (regenerated)"
grep -h "P3_twin" "$out/u1b.md" | cut -c1-120
grep -h "OUTCOME\|^- \*\*\|Always hide\|P1\b" "$out/u5.md" | head -4
grep -h "ratio to p=0" "$out/u2.md" | cut -c1-80
grep -h "R_h\|OUTCOME" "$out/u6.md" | head -3
grep -h "AUC\|only the agents" "$out/u3.md" | head -3
grep -h "OUTCOME" "$out/u7.md"
echo "regenerated files kept in $out/"
[ $status -eq 0 ] && echo "ALL REPRODUCED" || echo "SOMETHING DIFFERED: see above"
exit $status
