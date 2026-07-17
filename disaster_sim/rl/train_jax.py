"""
CLI entry point for JAX-accelerated PPO training.

Usage:
    python -m disaster_sim.rl.train_jax                          # defaults
    python -m disaster_sim.rl.train_jax --agent-type ambulance   # train ambulance
    python -m disaster_sim.rl.train_jax --preset small --n-envs 2048
    python -m disaster_sim.rl.train_jax --config configs/ppo_jax.yaml
"""

from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import pickle
import sys

import jax
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RL agent with JAX PPO (100 %% GPU)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides individual flags)",
    )
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--agent-type", type=str, default="drone")
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--n-minibatches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cities", type=int, default=32)
    parser.add_argument("--save-dir", type=str, default="models/")
    args = parser.parse_args()

    # ---- Device check ----
    print(f"JAX version  : {jax.__version__}")
    print(f"JAX backend  : {jax.default_backend()}")
    for d in jax.devices():
        print(f"  device     : {d}")
    print()

    if jax.default_backend() == "cpu":
        print(
            "WARNING: JAX is using the CPU backend.  "
            "Install jax[cuda12] for GPU acceleration.\n"
        )

    # ---- Build config ----
    from disaster_sim.rl.jax_ppo import PPOConfig, train

    # Start with defaults from CLI args
    cfg_dict = {
        "preset": args.preset,
        "agent_type": args.agent_type,
        "n_envs": args.n_envs,
        "total_timesteps": args.total_timesteps,
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "n_minibatches": args.n_minibatches,
        "learning_rate": args.lr,
        "n_cities": args.n_cities,
    }

    # If a YAML config is supplied, overlay its values
    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        # Map any YAML keys to PPOConfig field names
        key_map = {"learning_rate": "learning_rate", "lr": "learning_rate"}
        for k, v in yaml_cfg.items():
            field = key_map.get(k, k)
            cfg_dict[field] = v

    # Filter to valid PPOConfig fields
    valid_fields = {f.name for f in PPOConfig.__dataclass_fields__.values()}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}

    config = PPOConfig(**cfg_dict)

    # ---- Train ----
    runner_state, _ = train(config, seed=args.seed)

    # ---- Save ----
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir, f"{config.agent_type}_jax_ppo.pkl",
    )

    train_state = runner_state[0]
    params_cpu = jax.device_get(train_state.params)
    with open(save_path, "wb") as f:
        pickle.dump(params_cpu, f)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
