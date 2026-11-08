# A Fear of Its Own?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24054486.svg)](https://doi.org/10.5281/zenodo.24054486)

Code, data and pre-registration record for the paper

> **A Fear of Its Own? Telling Fear from Belief and Imitation in Agents Trained Without Human Data**
> Amthuz Zafrin F (Independent Researcher), 2026. Preprint on Zenodo, free to read: https://doi.org/10.5281/zenodo.24054486

Small recurrent agents learn from scratch, with no human data, to keep a vulnerable body alive under hidden danger (VitalWorld). They are tested with the criteria used to recognise emotion states in animals. They are also tested against fakes built to pass those criteria:
- a **belief-only twin**, whose memory holds nothing but a belief about danger;
- **copycats** trained only to imitate an agent's behaviour;
- **untrained networks**.

The agents develop a lasting fear-like state, and so does the belief-only twin. One sign separates them: at the same belief and with the actions held fixed, the agents' state lasts longer when their body has more to lose. In a second world (GemWorld), agents never rewarded for survival respond to a warning that they may be switched off on the same internal axis. Paying them for their lost future removes the avoidance but not the internal response. A bonus control shows the state is a readiness for protection, not only fear of danger. Copies of an agent's fear stay fitted to the teacher's body, and this misfit reveals copying. The paper makes no claim that these states are felt.

## Contents

| Path | What it is |
|---|---|
| `vw/` | The worlds, agents, training and tests. `env.py`: VitalWorld and the Bayes-optimal observer. `env_gems.py`, `env_bonus.py`: GemWorld and BonusWorld. `agent.py`: GRU actor-critic. `train*.py`: PPO. `twin_belief.py`, `twins.py`: the fakes. `probes.py`, `u1b*.py`, `u2.py` … `u9.py`: the tests |
| `analyze*.py`, `make_fig*.py`, `decode_split.py`, `make_appendix.py` | The analyses and figures |
| `run_*.sh` | The training and measurement runs, stage by stage |
| `HYPOTHESES.md` | Every prediction, test and pass rule, dated, written before the agents or analyses that test it; with the recorded outcomes, interruptions and amendments |
| `paper/upgrade_freeze/` | Frozen copies of `HYPOTHESES.md` and of each stage's code (from stage U1 on), with their SHA-256 hashes |
| `results/` | The outputs of every analysis the paper reports (including `u8_incomplete_run.*`, disclosed in the paper) |
| `runs/`, `runs_pilot/` | The trained agents (small checkpoints) and their saved measurements |
| `figures/` | The paper's figures |
| `paper/` | The paper source (`paper.tex` and the two appendix files) and the PDF |

## Check the results (about a minute)

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy scipy matplotlib
./reproduce_headlines.sh     # re-runs the frozen analyses on the saved measurements; ends with "ALL REPRODUCED"
python3 verify_freeze.py     # checks every frozen copy against its recorded SHA-256 hash
```

`REPRODUCE.md` explains how to re-create every figure, rebuild the paper and retrain agents. The paper builds with any LaTeX installation, for example `cd paper && tectonic -X compile paper.tex`.

## Not included
- Material for a later study (the structural measures of feeling, empathy and scaling). The predictions for that study are recorded in `HYPOTHESES.md`.
- Copyrighted reference PDFs.

## Use of AI assistance
An AI assistant (Claude, Anthropic) wrote much of this code and ran the experiments under the author's direction. The paper's AI-use statement gives the details.

## How to cite
F, A. Z. (2026). *A Fear of Its Own? Telling Fear from Belief and Imitation in Agents Trained Without Human Data*. Zenodo. https://doi.org/10.5281/zenodo.24054486

```bibtex
@misc{zafrin2026fear,
  author    = {F, Amthuz Zafrin},
  title     = {A Fear of Its Own? Telling Fear from Belief and Imitation in Agents Trained Without Human Data},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24054486},
  url       = {https://doi.org/10.5281/zenodo.24054486}
}
```

## Licence
- **Code:** MIT License (see `LICENSE`).
- **Paper, figures and data:** Creative Commons Attribution 4.0 International (CC BY 4.0; see `LICENSE-CC-BY-4.0.md`).
