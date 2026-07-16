"""
Train RL Agents using Stable Baselines3 (PPO)

Trains a Drone policy and an Ambulance policy.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from disaster_sim.rl.env import DisasterEnv
from disaster_sim.rl.networks import DisasterCNN


from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(agent_type):
    def _init():
        return DisasterEnv(preset="medium", target_agent_type=agent_type)
    return _init

def train_agent(agent_type: str, total_timesteps: int = 50000):
    print(f"--- Starting Training for {agent_type.upper()} ---")
    
    # Vectorized Environment using SubprocVecEnv to utilize all CPU cores
    env = make_vec_env(
        make_env(agent_type),
        n_envs=20,
        vec_env_cls=SubprocVecEnv
    )
    
    # Save checkpoints every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=2500,  # 2500 * 16 envs = 40k steps (adjusted frequency)
        save_path='./models/logs/',
        name_prefix=f'{agent_type}_model'
    )
    
    # Prepare model arguments with our Custom CNN to properly process the 6x15x15 spatial grid.
    # An MlpPolicy would flatten the grid and destroy spatial information, crippling the agent.
    policy_kwargs = dict(
        features_extractor_class=DisasterCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    # Initialize PPO with CnnPolicy
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./tensorboard_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=1024, # Increased batch size for better GPU utilization
        device="cuda", # RTX 4060
        policy_kwargs=policy_kwargs
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # Save final model
    os.makedirs("./models", exist_ok=True)
    model.save(f"./models/{agent_type}_ppo")
    print(f"--- Finished Training for {agent_type.upper()} ---")


if __name__ == "__main__":
    # Train Drone for scouting/exploring
    train_agent("drone", total_timesteps=2_000_000)
    
    # Train Ambulance for rescuing
    train_agent("ambulance", total_timesteps=2_000_000)
