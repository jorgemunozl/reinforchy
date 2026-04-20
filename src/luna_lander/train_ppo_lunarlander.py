from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


@dataclass(frozen=True)
class RunConfig:
    env_id: str
    seed: int
    total_timesteps: int
    n_envs: int
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PPO smoke test on LunarLander-v3")
    p.add_argument(
        "--play",
        action="store_true",
        help="Load a saved model and run it in the environment (rendering optional).",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a saved SB3 .zip model (required with --play).",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Render the environment (requires a display for render_mode=human).",
    )
    p.add_argument(
        "--record-video-dir",
        type=str,
        default=None,
        help="If set, record an .mp4 into this directory (uses rgb_array render mode).",
    )
    p.add_argument("--episodes", type=int, default=3, help="Episodes to run in --play mode.")
    p.add_argument("--env-id", type=str, default="LunarLander-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=50_000)
    p.add_argument("--n-envs", type=int, default=8)

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    return p


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _play(model_path: str, env_id: str, seed: int, episodes: int, render: bool, record_video_dir: str | None) -> None:
    if record_video_dir is not None:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    env = gym.make(env_id, render_mode=render_mode)

    if record_video_dir is not None:
        video_dir = Path(record_video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_folder=str(video_dir), name_prefix=f"{env_id}_ppo")

    model = PPO.load(model_path, device=_resolve_device())

    for ep in range(1, episodes + 1):
        obs, _info = env.reset(seed=seed + ep)
        terminated = False
        truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            ep_reward += float(reward)

            if render and record_video_dir is None:
                env.render()

        print(f"[play] episode={ep} reward={ep_reward:.2f}")

    env.close()


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.play:
        if not args.model_path:
            raise SystemExit("--model-path is required with --play")
        _play(
            model_path=args.model_path,
            env_id=args.env_id,
            seed=args.seed,
            episodes=args.episodes,
            render=args.render,
            record_video_dir=args.record_video_dir,
        )
        return

    cfg = RunConfig(
        env_id=args.env_id,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
    )

    set_random_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"{cfg.env_id}_ppo_seed{cfg.seed}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device()
    print(f"[info] env_id={cfg.env_id} seed={cfg.seed} n_envs={cfg.n_envs} device={device}")
    print(f"[info] total_timesteps={cfg.total_timesteps} out_dir={out_dir}")

    def make_env() -> gym.Env:
        env = gym.make(cfg.env_id)
        env.reset(seed=cfg.seed)
        return env

    vec_env = make_vec_env(make_env, n_envs=cfg.n_envs, seed=cfg.seed)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        verbose=1,
        tensorboard_log=str(out_dir / "tb"),
        device=device,
    )

    model.learn(total_timesteps=cfg.total_timesteps, progress_bar=True)

    model_path = out_dir / "ppo_lunarlander.zip"
    model.save(model_path)
    print(f"[ok] saved model to {model_path}")

    vec_env.close()


if __name__ == "__main__":
    main()

