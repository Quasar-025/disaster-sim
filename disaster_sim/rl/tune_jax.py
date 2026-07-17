"""
Hyperparameter tuning script for JAX PPO.

Usage:
    python -m disaster_sim.rl.tune_jax --trials 5 --timesteps 2000000
"""

import argparse
import itertools
import os
import pickle
import random
import time

import jax
import yaml

from disaster_sim.rl.jax_ppo import PPOConfig, train

def main():
    parser = argparse.ArgumentParser(description="Tune RL agent hyperparameters")
    parser.add_argument("--trials", type=int, default=4, help="Number of configs to try")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Timesteps per trial")
    parser.add_argument("--agent-type", type=str, default="drone")
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default="models/")
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")

    # Define search space
    search_space = {
        "learning_rate": [1e-4, 3e-4, 5e-4],
        "ent_coef": [0.01, 0.05, 0.1],
        "gamma": [0.99, 0.995],
        "n_epochs": [4, 8],
    }

    # Generate all combinations
    keys, values = zip(*search_space.items())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Shuffle and pick `args.trials` configs
    random.seed(42)
    random.shuffle(all_configs)
    configs_to_try = all_configs[:args.trials]

    best_reward = -float('inf')
    best_config = None
    best_params = None

    print(f"Starting tuning with {len(configs_to_try)} trials...")
    print("-" * 60)

    for i, cfg in enumerate(configs_to_try):
        print(f"\nTrial {i+1}/{len(configs_to_try)}")
        print(f"Config: {cfg}")
        
        # Build PPO config
        ppo_cfg = PPOConfig(
            agent_type=args.agent_type,
            n_envs=args.n_envs,
            total_timesteps=args.timesteps,
            **cfg
        )
        
        # Train
        runner_state, metrics = train(ppo_cfg, seed=i)
        
        # Evaluate
        final_reward = float(metrics["mean_reward"])
        final_kl = float(metrics["approx_kl"])
        
        print(f"--> Result: Reward = {final_reward:+.4f}, KL = {final_kl:.4f}")
        
        if final_reward > best_reward:
            print(f"*** New Best! ***")
            best_reward = final_reward
            best_config = cfg
            train_state = runner_state[0]
            best_params = jax.device_get(train_state.params)

    print("\n" + "=" * 60)
    print("Tuning Complete!")
    print(f"Best Reward: {best_reward:+.4f}")
    print(f"Best Config: {best_config}")
    
    # Save best model
    if best_params is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"{args.agent_type}_jax_ppo.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(best_params, f)
        print(f"Saved best model to {save_path}")

        # Update config file with best params
        config_path = f"configs/ppo_jax_{args.agent_type}_best.yaml"
        with open(config_path, "w") as f:
            yaml.dump(best_config, f)
        print(f"Saved best hyperparameters to {config_path}")

if __name__ == "__main__":
    main()
