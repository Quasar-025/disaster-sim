"""
Experiment Runner for automated benchmarking.
Runs headless simulations to gather statistical data for evaluation.
"""

import argparse
import csv
import os
import time

from disaster_sim.rl.env import DisasterEnv
from disaster_sim.rl.pz_env import DisasterMultiAgentEnv

def run_experiments(num_episodes: int, preset: str, output_csv: str):
    print(f"Running {num_episodes} benchmarking episodes with preset '{preset}'...")
    
    # Use multi-agent env by default for benchmarking
    env = DisasterMultiAgentEnv(preset=preset, render_mode=None)
    
    results = []
    
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Episode", "Timesteps", "Total_Victims", "Rescued", 
            "Transported", "Dead", "Coverage", "Total_Energy_Consumed", "Collisions"
        ])
        
        for ep in range(1, num_episodes + 1):
            obs, info = env.reset()
            
            done = False
            total_steps = 0
            
            while not done:
                # Provide random actions for now (can be swapped for trained models)
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                
                obs, rewards, terminations, truncations, infos = env.step(actions)
                
                total_steps += 1
                
                if not env.agents:
                    done = True
                    
            world = env.world
            
            rescued = world.total_victims_rescued
            transported = world.total_victims_transported
            dead = sum(1 for v in world.victims.values() if not v.is_alive)
            total = len(world.victims)
            coverage = world.coverage
            energy = world.total_energy_consumed
            collisions = world.total_collisions
            
            writer.writerow([
                ep, total_steps, total, rescued, 
                transported, dead, coverage, energy, collisions
            ])
            
            print(f"Episode {ep}/{num_episodes} - Steps: {total_steps}, Rescued: {rescued}/{total}, Dead: {dead}, Coverage: {coverage:.1%}")
            
    print(f"Experiments complete. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--preset", type=str, default="medium", help="City configuration preset")
    parser.add_argument("--output", type=str, default="results/benchmark.csv", help="Output CSV file path")
    args = parser.parse_args()
    
    run_experiments(args.episodes, args.preset, args.output)
