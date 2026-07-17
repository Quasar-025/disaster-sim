"""
Physics Engine — Manages agent movement, collision detection, and battery drain.
"""

from __future__ import annotations

import enum

from disaster_sim.digital_twin.world_state import AgentState, WorldState
from disaster_sim.digital_twin.city_config import Terrain


class Action(enum.IntEnum):
    """Discrete action space for RL agents."""
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    INTERACT = 5  # Context-sensitive: Rescue, drop-off, or charge


class PhysicsEngine:
    """Handles step-by-step state updates for agents."""

    def __init__(self, world: WorldState):
        self.world = world

    def step(self, agent_id: str, action: int) -> bool:
        """Apply an action for an agent. Returns True if the action was valid (no collision)."""
        agent = self.world.agents.get(agent_id)
        if not agent or not agent.is_alive:
            return False

        # Drain battery
        agent.battery -= agent.spec.battery_drain_rate
        if agent.battery <= 0:
            agent.battery = 0
            agent.status = "dead"
            return False

        valid_action = True
        dr, dc = 0, 0
        
        agent.prev_row = agent.row
        agent.prev_col = agent.col

        if action == Action.UP:
            dr = -1
        elif action == Action.DOWN:
            dr = 1
        elif action == Action.LEFT:
            dc = -1
        elif action == Action.RIGHT:
            dc = 1
        elif action == Action.INTERACT:
            self._handle_interaction(agent)
            return True

        if dr != 0 or dc != 0:
            target_r = int(agent.row + dr)
            target_c = int(agent.col + dc)

            # Check passability
            if self.world.is_passable(target_r, target_c, agent.agent_type):
                # Simple collision check with other agents (only ground units block each other)
                collision = False
                if agent.spec.movement_type != "aerial":
                    for other in self.world.active_agents:
                        if other.id != agent.id and other.spec.movement_type != "aerial":
                            if other.grid_pos == (target_r, target_c):
                                collision = True
                                break

                if collision:
                    self.world.total_collisions += 1
                    valid_action = False
                else:
                    agent.row += dr
                    agent.col += dc
            else:
                self.world.total_collisions += 1
                valid_action = False

        self.world.total_energy_consumed += agent.spec.battery_drain_rate
        
        # Reveal fog of war
        cells_revealed = self.world._reveal_around(agent)
        agent.cells_explored += cells_revealed

        return valid_action

    def _handle_interaction(self, agent: AgentState) -> None:
        """Handle context-sensitive interactions (Rescue/Charge)."""
        r, c = agent.grid_pos

        # Charging
        terrain = self.world.terrain_at(r, c)
        if terrain == Terrain.CHARGING_STATION:
            # Instantly charge (for simplification in early RL)
            agent.battery = agent.spec.battery_capacity
            agent.status = "charging"
            return

        # Rescuing
        if agent.spec.can_rescue and not agent.carrying_victim:
            victims = self.world.get_victims_in_range(r, c, radius=1)
            if victims:
                victim = victims[0]
                victim.rescued = True
                victim.assigned_agent = agent.id
                agent.carrying_victim = victim.id
                agent.status = "rescuing"
                self.world.total_victims_rescued += 1
                return
        
        # Drop off (simplified: if carrying victim and at hospital)
        if agent.carrying_victim and terrain == Terrain.HOSPITAL:
            victim = self.world.victims[agent.carrying_victim]
            victim.transported = True
            agent.carrying_victim = None
            agent.status = "idle"
            self.world.total_victims_transported += 1
            return

    def sync_communications(self) -> None:
        """
        Merge local_explored maps and known victims for agents that are within communication range
        of each other or a hospital (base station).
        """
        import numpy as np
        comm_range = self.world.city.config.communication_range
        active_agents = self.world.active_agents
        
        for i, a1 in enumerate(active_agents):
            if a1.local_explored is None:
                continue
                
            # Check if near base station (hospital)
            near_base_station = False
            for (hr, hc) in self.world.city.hospitals:
                if abs(a1.row - hr) + abs(a1.col - hc) <= comm_range:
                    near_base_station = True
                    break
                    
            if near_base_station:
                # Sync with global network
                a1.local_explored[:] = self.world.explored[:]
                for v in self.world.victims.values():
                    if v.detected and v.is_alive and not v.rescued:
                        a1.local_victims_known.add(v.id)
            
            # Mesh networking: sync with nearby agents
            for j in range(i + 1, len(active_agents)):
                a2 = active_agents[j]
                if a2.local_explored is None:
                    continue
                
                dist = abs(a1.row - a2.row) + abs(a1.col - a2.col)
                if dist <= comm_range:
                    # Sync explored maps
                    np.logical_or(a1.local_explored, a2.local_explored, out=a1.local_explored)
                    np.logical_or(a1.local_explored, a2.local_explored, out=a2.local_explored)
                    
                    # Sync known victims
                    union_victims = a1.local_victims_known.union(a2.local_victims_known)
                    a1.local_victims_known = set(union_victims)
                    a2.local_victims_known = set(union_victims)
