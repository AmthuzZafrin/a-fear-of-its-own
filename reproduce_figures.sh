#!/bin/bash
# Re-create every analysis, figure and the paper from the saved network outputs (runs/*/*.json), without
# retraining. Writes into a separate folder (repro/) so the originals are untouched; compare with diff afterwards.
# Usage: ./reproduce_figures.sh          (about 10 minutes on a laptop; needs .venv and tools/tectonic)
set -e
cd "$(dirname "$0")"
mkdir -p repro
cp -r results repro/results_original_$(date +%Y%m%d_%H%M)
for s in analyze.py analyze_stage3.py make_appendix.py make_fig_u1b.py make_fig_u2.py \
         analyze_u1.py analyze_u1b.py analyze_u2.py analyze_u3.py analyze_u4.py; do
  echo "== $s"; .venv/bin/python $s > repro/$(basename $s .py).out 2>&1 || echo "   FAILED (see repro/$(basename $s .py).out)"
done
(cd paper && ../tools/tectonic -X compile paper.tex > ../repro/paper_build.out 2>&1) && echo "paper rebuilt: paper/paper.pdf"
echo "Done. Compare results/*.md with repro/results_original_*/ (e.g. diff -r) to check that every number is reproduced."
