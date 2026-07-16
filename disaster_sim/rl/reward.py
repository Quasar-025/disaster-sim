"""
Reward Function for Reinforcement Learning.
Calculates the reward for an agent at a given timestep.
"""

from __future__ import annotations

from disaster_sim.digital_twin.world_state import WorldState
from disaster_sim.digital_twin.city_config import SEVERITY_REWARD_MULTIPLIER


class RewardCalculator:
    """Calculates dense and sparse rewards for RL agents."""
    
    # Reward weights
    REWARD_EXPLORE = 0.1       # Per newly revealed cell
    REWARD_RESCUE = 100.0      # Base reward for rescuing a victim
    REWARD_TRANSPORT = 50.0    # Base reward for dropping at hospital
    PENALTY_COLLISION = -5.0   # Penalty for hitting obstacles/agents
    PENALTY_TIME = -0.01       # Small penalty every step to encourage speed
    
    def __init__(self, world: WorldState):
        self.world = world
        self.previous_coverage = world.coverage
        
    def calculate_reward(
        self,
        agent_id: str,
        action: int,
        valid_action: bool,
        cells_revealed_this_step: int,
        rescued_victim_id: str | None = None
    ) -> float:
        """Calculate the reward for an agent after taking an action."""
        
        agent = self.world.agents[agent_id]
        reward = self.PENALTY_TIME
        
        if not valid_action:
            reward += self.PENALTY_COLLISION
            
        # Exploration reward
        reward += cells_revealed_this_step * self.REWARD_EXPLORE
        
        if rescued_victim_id:
            victim = self.world.victims[rescued_victim_id]
            multiplier = SEVERITY_REWARD_MULTIPLIER.get(victim.severity, 1.0)
            reward += self.REWARD_RESCUE * multiplier
            
        return reward
