# Reproducibility Notes

This repository is a portfolio-ready reconstruction of a final-year project. The
code is cleaned up for reviewability, while the figures in `assets/` are
historical outputs from the original dissertation work.

## Environment

The original experiments used the classic Gym-based Level-Based Foraging
package. That stack is easiest to reproduce with Python 3.7 or another
environment that can install `gym==0.21.0`, `torch==1.13.1`, and the pinned
`lbforaging` Git dependency in `requirements.txt`.

For quick verification, each entrypoint supports `--smoke`. Smoke mode uses a
tiny built-in two-agent grid-world so that the command-line path, logging, and
checkpoint-writing behavior can be checked without rendering or a long training
run.

## Expected Runtime

Smoke runs should finish in seconds:

```bash
python -m marl_lbf.train_tabular --episodes 10 --smoke
python -m marl_lbf.train_dqn --episodes 10 --smoke
python -m marl_lbf.train_forced_coop --episodes 10 --smoke
```

Full LBF runs are much slower. DQN experiments may require thousands of episodes
before the replay buffer is warm and the target network has synchronized enough
times to show stable learning.

## Historical Figures

The README figures are preserved project evidence, not regenerated artifacts
from the cleaned package. The cleaned package writes new metrics and checkpoints
under `runs/` by default so new experiments do not overwrite historical results.

## Known Limitations

- The DQN agents train independently, so other agents' policies change while
  each learner is estimating action values.
- The forced-cooperation experiment relies on reward shaping to make sparse
  cooperative rewards learnable without communication.
- Hyperparameters are intentionally close to the dissertation experiments rather
  than exhaustively tuned for modern MARL baselines.
