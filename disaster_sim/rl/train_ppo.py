"""
Training script for PPO.
Uses Stable Baselines 3 to train a single agent on the DisasterEnv.
"""

import argparse
import os
import yaml
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from disaster_sim.rl.env import DisasterEnv
from disaster_sim.rl.networks import DisasterCNN
from disaster_sim.rl.callbacks import TensorboardCallback


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on DisasterEnv")
    parser.add_argument("--config", type=str, default="configs/ppo_small.yaml", help="Path to PPO config file")
    parser.add_argument("--preset", type=str, default="small", help="City preset name")
    parser.add_argument("--n-envs", type=int, default=20, help="Number of parallel environments")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set up environment
    def make_env():
        return DisasterEnv(preset=args.preset, render_mode=None)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

    # Prepare model arguments
    policy_kwargs = dict(
        features_extractor_class=DisasterCNN,
        features_extractor_kwargs=dict(features_dim=config.get("features_dim", 128)),
    )
    
    log_dir = "tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print(f"Initializing PPO with {args.n_envs} environments...")
    model = PPO(
        config["policy"],
        vec_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
    )

    callback = TensorboardCallback()

    print(f"Starting training for {config['total_timesteps']} timesteps...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback,
            tb_log_name=f"ppo_{args.preset}",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        
    # Save the model
    save_path = f"models/ppo_{args.preset}_latest"
    os.makedirs("models", exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

if __name__ == "__main__":
    main()
