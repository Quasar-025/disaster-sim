"""
Gymnasium Environment for Single-Agent Training.
Wraps the WorldState and PhysicsEngine into a standard RL interface.
"""

from __future__ import annotations

import argparse
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from disaster_sim.digital_twin.city_config import get_config
from disaster_sim.digital_twin.city_generator import CityGenerator
from disaster_sim.digital_twin.world_state import WorldState
from disaster_sim.digital_twin.disaster_dynamics import DisasterDynamics
from disaster_sim.engine.physics import Action, PhysicsEngine
from disaster_sim.rl.observation import ObservationBuilder
from disaster_sim.rl.reward import RewardCalculator


class DisasterEnv(gym.Env):
    """
    Single-agent Gymnasium environment for disaster response.
    Controls the first agent in the fleet (usually a drone or ground_robot).
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, preset: str = "small", render_mode: str | None = None, fov_size: int = 21, target_agent_type: str = "drone"):
        super().__init__()
        self.preset = preset
        self.render_mode = render_mode
        self.fov_size = fov_size
        self.target_agent_type = target_agent_type
        
        self.config = get_config(preset)
        self.action_space = spaces.Discrete(len(Action))
        # Observation space: 6 channels, fov_size x fov_size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(6, fov_size, fov_size), 
            dtype=np.float32
        )
        
        self.world: WorldState | None = None
        self.physics: PhysicsEngine | None = None
        self.dynamics: DisasterDynamics | None = None
        self.obs_builder: ObservationBuilder | None = None
        self.reward_calc: RewardCalculator | None = None
        self.agent_id: str = ""
        self.renderer = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        if seed is not None:
            self.config.seed = seed
            
        generator = CityGenerator(self.config)
        city = generator.generate()
        
        self.world = WorldState(city)
        self.physics = PhysicsEngine(self.world)
        self.dynamics = DisasterDynamics(self.world)
        self.obs_builder = ObservationBuilder(self.world, self.fov_size)
        self.reward_calc = RewardCalculator(self.world)
        
        # Pick the first agent of the target type
        self.agent_id = next(
            (aid for aid, a in self.world.agents.items() if a.agent_type == self.target_agent_type), 
            list(self.world.agents.keys())[0]
        )
        
        if self.render_mode == "human":
            from disaster_sim.engine.renderer import CityRenderer
            self.renderer = CityRenderer(city, self.world)
            if city.width <= 50:
                self.renderer.cell_size = 12  # Make small maps bigger
            
        obs = self.obs_builder.get_observation(self.agent_id)
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Record pre-state to compute accurate rewards
        agent = self.world.agents[self.agent_id]
        prev_carrying = agent.carrying_victim
        prev_explored = agent.cells_explored
        
        # Take step
        valid_action = self.physics.step(self.agent_id, action)
        
        # Other agents do nothing (idle) in this single-agent version
        
        # Sync communications for MARL network constraints
        self.physics.sync_communications()
        
        self.world.timestep += 1
        
        # Advance disaster physics every 10 steps (slower than agent movement)
        if self.world.timestep % 10 == 0:
            self.dynamics.step()
        
        # Compute exact reward logic
        cells_revealed_this_step = agent.cells_explored - prev_explored
        rescued_victim_id = agent.carrying_victim if prev_carrying is None and agent.carrying_victim is not None else None
        
        reward = self.reward_calc.calculate_reward(
            self.agent_id, action, valid_action, cells_revealed_this_step, rescued_victim_id
        )
            
        obs = self.obs_builder.get_observation(self.agent_id)
        
        terminated = not agent.is_alive or self.world.victims_remaining == 0
        truncated = self.world.timestep >= 1000
        
        info = self._get_info()
        
        if self.render_mode == "human" and self.renderer:
            import pygame
            # Process events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            # Simple tick render
            self._render_human()
            
        return obs, reward, terminated, truncated, info

    def _get_info(self) -> dict:
        return {
            "coverage": self.world.coverage,
            "victims_remaining": self.world.victims_remaining,
            "battery": self.world.agents[self.agent_id].battery_fraction,
            "collisions": self.world.total_collisions
        }

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            if not self.renderer:
                from disaster_sim.engine.renderer import CityRenderer
                self.renderer = CityRenderer(self.world.city, self.world)
            import pygame
            surface = self.renderer.render_to_surface()
            return pygame.surfarray.array3d(surface)

    def _render_human(self):
        import pygame
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((self.renderer.WINDOW_WIDTH, self.renderer.WINDOW_HEIGHT))
            pygame.display.set_caption("DisasterEnv Interactive")
            self.clock = pygame.time.Clock()
            
        self.screen.fill((20, 20, 30))
        city_surface = self.renderer.render_to_surface()
        
        # Center the city in the viewport
        view_width = self.renderer.WINDOW_WIDTH - self.renderer.INFO_PANEL_WIDTH
        view_height = self.renderer.WINDOW_HEIGHT
        
        city_w = city_surface.get_width()
        city_h = city_surface.get_height()
        
        offset_x = max(0, (view_width - city_w) // 2)
        offset_y = max(0, (view_height - city_h) // 2)
        
        self.screen.blit(city_surface, (offset_x, offset_y))
        self.renderer._render_info_panel(self.screen)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])


# ---------------------------------------------------------------------------
# Interactive Play and Random Baseline
# ---------------------------------------------------------------------------

def play_interactive():
    """Play the environment with keyboard controls."""
    import pygame
    pygame.init()
    
    env = DisasterEnv(preset="small", render_mode="human")
    obs, info = env.reset()
    
    running = True
    print("\n--- INTERACTIVE MODE ---")
    print("Controls: Arrow Keys to move, Space to interact (rescue/charge)")
    print("Agent ID:", env.agent_id)
    
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = Action.UP
                elif event.key == pygame.K_DOWN:
                    action = Action.DOWN
                elif event.key == pygame.K_LEFT:
                    action = Action.LEFT
                elif event.key == pygame.K_RIGHT:
                    action = Action.RIGHT
                elif event.key == pygame.K_SPACE:
                    action = Action.INTERACT
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    
        if action is not None:
            obs, reward, term, trunc, info = env.step(action)
            print(f"Action: {Action(action).name} | Reward: {reward:.2f} | Battery: {info['battery']:.2f} | Victims left: {info['victims_remaining']}")
            if term or trunc:
                print("Episode finished!")
                env.reset()
                
        # Call the environment's built-in render method
        env.render()
            
    pygame.quit()


def play_random():
    """Run a random agent baseline."""
    env = DisasterEnv(preset="small", render_mode=None)
    obs, info = env.reset()
    
    total_reward = 0.0
    steps = 0
    
    print("\n--- RANDOM AGENT BASELINE ---")
    while True:
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if term or trunc:
            break
            
    print(f"Finished in {steps} steps.")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Coverage: {info['coverage']:.1%}")
    print(f"Victims remaining: {info['victims_remaining']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Play with keyboard")
    parser.add_argument("--random", action="store_true", help="Run random agent baseline")
    args = parser.parse_args()
    
    if args.interactive:
        play_interactive()
    else:
        play_random()
