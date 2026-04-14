# LunarLander-v3 PPO smoke test (SB3 + Gymnasium)

This folder is intentionally minimal: it installs into your existing Conda env (`ppo`) and runs a short PPO training smoke test on `LunarLander-v3`.

## Setup (local)

Activate your existing Conda env:

```bash
conda activate ppo
```

From this directory, install deps:

```bash
python -m pip install -U pip
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install --no-cache-dir -e .
```

## Run

```bash
python src/train_ppo_lunarlander.py
```

Artifacts (TensorBoard logs + saved model) go under `runs/`.

### Optional: shorter/faster

```bash
python src/train_ppo_lunarlander.py --total-timesteps 10000 --n-envs 4
```

## Notes / Troubleshooting

- **Box2D install issues**: `LunarLander-v3` depends on Box2D bindings via `gymnasium[box2d]`. If install fails on your system, you may need OS build deps (commonly `swig`, `gcc`, `python-devel`-equivalents) before `pip install -e .` succeeds.
- **No rendering by default**: this smoke test does not render; it’s focused on verifying training starts and produces artifacts.

