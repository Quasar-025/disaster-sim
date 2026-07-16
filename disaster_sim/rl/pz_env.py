"""
PettingZoo Multi-Agent Environment

Wraps the WorldState into a ParallelEnv where multiple agents (drones, ambulances, police)
act simultaneously. The reward is shared / localized depending on agent roles.
"""

import functools
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from disaster_sim.digital_twin.city_config import get_config, Action
from disaster_sim.digital_twin.city_generator import CityGenerator
from disaster_sim.digital_twin.world_state import WorldState
from disaster_sim.engine.physics import PhysicsEngine
from disaster_sim.rl.observation import ObservationBuilder
from disaster_sim.rl.reward import RewardCalculator
from disaster_sim.digital_twin.disaster_dynamics import DisasterDynamics


class DisasterMultiAgentEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "disaster_pz_v0"}

    def __init__(self, preset="medium", render_mode=None, fov_size=15):
        self.preset = preset
        self.render_mode = render_mode
        self.fov_size = fov_size
        
        self.config = get_config(preset)
        self.possible_agents = []
        
        # We will dynamically populate possible_agents during reset based on the generated city,
        # but PettingZoo usually requires it statically. We'll pre-generate a dummy city to get agent IDs.
        dummy_gen = CityGenerator(self.config)
        dummy_city = dummy_gen.generate()
        self.possible_agents = [spawn.id for spawn in dummy_city.agent_spawns if "traffic_light" not in spawn.agent_type and "citizen" not in spawn.agent_type]
        
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.world = None
        self.physics = None
        self.dynamics = None
        self.obs_builder = None
        self.reward_calc = None
        self.renderer = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(
            low=0.0, high=1.0, 
            shape=(6, self.fov_size, self.fov_size), 
            dtype=np.float32
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(Action))

    def render(self):
        if self.render_mode == "human" and self.renderer:
            import pygame
            self.renderer.render_to_surface()
            pygame.display.flip()
        elif self.render_mode == "rgb_array" and self.renderer:
            import pygame
            return pygame.surfarray.array3d(self.renderer.render_to_surface())

    def close(self):
        if self.renderer is not None:
            import pygame
            pygame.quit()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.config.seed = seed
            
        generator = CityGenerator(self.config)
        city = generator.generate()
        
        self.world = WorldState(city)
        self.physics = PhysicsEngine(self.world)
        self.dynamics = DisasterDynamics(self.world)
        self.obs_builder = ObservationBuilder(self.world, self.fov_size)
        self.reward_calc = RewardCalculator(self.world)
        
        # Agents alive at the start
        self.agents = [a for a in self.possible_agents]
        
        if self.render_mode == "human":
            from disaster_sim.engine.renderer import CityRenderer
            self.renderer = CityRenderer(city, self.world)
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.renderer.WINDOW_WIDTH, self.renderer.WINDOW_HEIGHT))

        observations = {
            agent: self.obs_builder.get_observation(agent) for agent in self.agents
        }
        infos = {agent: self._get_info(agent) for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        """
        Takes a dict of actions keyed by agent.
        Returns (observations, rewards, terminations, truncations, infos)
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # 1. Record pre-states
        pre_states = {}
        for agent_id in self.agents:
            agent = self.world.agents[agent_id]
            pre_states[agent_id] = {
                "explored": agent.cells_explored,
                "carrying": agent.carrying_victim
            }

        # 2. Execute actions
        for agent_id, action in actions.items():
            if self.world.agents[agent_id].is_alive:
                self.physics.step(agent_id, action)
                
        # Sync communications for MARL network constraints
        self.physics.sync_communications()
                
        # 3. Advance World State
        self.world.timestep += 1
        if self.world.timestep % 10 == 0:
            self.dynamics.step()

        # 4. Compute returns
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        global_termination = self.world.victims_remaining == 0
        global_truncation = self.world.timestep >= 1000

        for agent_id in self.agents:
            agent = self.world.agents[agent_id]
            pre = pre_states[agent_id]
            
            cells_rev = agent.cells_explored - pre["explored"]
            rescued_victim_id = agent.carrying_victim if pre["carrying"] is None and agent.carrying_victim is not None else None
            
            # Simple individual reward (can be made cooperative)
            reward = self.reward_calc.calculate_reward(agent_id, actions[agent_id], True, cells_rev, rescued_victim_id)
                
            rewards[agent_id] = reward
            observations[agent_id] = self.obs_builder.get_observation(agent_id)
            
            terminations[agent_id] = not agent.is_alive or global_termination
            truncations[agent_id] = global_truncation
            infos[agent_id] = self._get_info(agent_id)

        # 5. Clean up dead agents from self.agents
        self.agents = [agent for agent in self.agents if not terminations[agent]]

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _get_info(self, agent_id):
        agent = self.world.agents[agent_id]
        return {
            "battery": agent.battery_fraction,
            "status": agent.status
        }
