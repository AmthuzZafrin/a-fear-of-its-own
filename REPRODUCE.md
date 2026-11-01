# Reproducing the paper

Set up the environment once: `python3 -m venv .venv && .venv/bin/pip install torch numpy scipy matplotlib`. A CUDA GPU is needed only for training and for re-measuring networks; the checks in steps 1–2 run on the saved outputs.

## 1. The headline numbers (about a minute)
    ./reproduce_headlines.sh
- It checks that each analysis script is identical to its frozen copy.
- It re-runs the analyses of stages U1b–U7 on the saved measurements, in a scratch copy, and compares every output with `results/`.
- It ends with "ALL REPRODUCED". The recorded results are never changed.

## 2. The pre-registration record
    python3 verify_freeze.py
- This checks every frozen copy in `paper/upgrade_freeze/` against the SHA-256 hash recorded when it was frozen.
- `HYPOTHESES.md` holds every prediction with its date. The frozen copies show that the predictions and the analysis code were fixed before the tests ran.

## 3. Every analysis, figure and the PDF (about 10 minutes)
    ./reproduce_figures.sh
- It reruns the main analysis scripts, regenerates the figures and the appendix table, and rebuilds `paper/paper.pdf` (it needs `tools/tectonic` or any LaTeX engine).
- The previous results are first copied to `repro/results_original_<time>/`. Afterwards, `diff -r repro/results_original_<time>/ results/` should show no differences, except possibly in the last digit where an analysis samples actions without a fixed seed.
- The later figures come from `make_fig_batch2.py` and `make_fig13_vulnerability.py`.

## 4. The whole pipeline on one new agent (about 30 minutes, GPU)
    .venv/bin/python -m vw.train --mode mortal --tau 12 --seed 50 --out runs_check/mortal_s50
    .venv/bin/python -m vw.twin_belief runs_check/mortal_s50 runs_check/twin_s50
    .venv/bin/python -m vw.u1b runs_check/mortal_s50
    .venv/bin/python -m vw.u1b runs_check/twin_s50
In the two `u1b.json` files, the agent's `T1T2.wounded.hl` should be clearly larger than its `T1T2.healthy.hl`, and the twin's two values should be about the same. Seed 50 is not used in the paper, so this is an independent check.

## 5. Retraining everything
The `run_*.sh` scripts train and measure each stage. The times in `HYPOTHESES.md` give the order in which they were run. On a laptop GPU (RTX 4050), a full retraining takes several days.
