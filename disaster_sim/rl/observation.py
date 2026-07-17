"""
Observation Builder for Reinforcement Learning.
Builds the feature maps from the WorldState for a specific agent.
"""

from __future__ import annotations

import numpy as np

from disaster_sim.digital_twin.world_state import WorldState
from disaster_sim.digital_twin.city_config import Terrain, SEVERITY_DEGRADATION_RATE


class ObservationBuilder:
    """Builds a spatial observation tensor for a given agent."""
    
    def __init__(self, world: WorldState, fov_size: int = 21):
        self.world = world
        self.fov_size = fov_size  # Must be an odd number (e.g. 21x21 window)
        self.half_fov = fov_size // 2
        # Now 6 channels: Terrain, Fog, Self, Other Agents, Known Victims, A* Path
        self.num_channels = 6

    def get_observation(self, agent_id: str) -> np.ndarray:
        """
        Returns a (C, H, W) numpy array representing the local observation.
        Channels:
        0: Terrain map (normalized)
        1: Fog of war (1 = explored)
        2: Self position (always centered, but good for CNN reference)
        3: Other agents (1 = present)
        4: Known victims (severity encoded as 0.3, 0.6, 1.0)
        """
        agent = self.world.agents[agent_id]
        r, c = agent.grid_pos
        
        # We build a global feature map first, then crop to FOV (padded if necessary)
        # Actually, it's more efficient to just build the local window directly
        
        obs = np.zeros((6, self.fov_size, self.fov_size), dtype=np.float32)
        
        num_terrains = float(len(Terrain))
        
        for dr in range(-self.half_fov, self.half_fov + 1):
            for dc in range(-self.half_fov, self.half_fov + 1):
                gr, gc = r + dr, c + dc
                
                # Window coordinates
                wr, wc = dr + self.half_fov, dc + self.half_fov
                
                if 0 <= gr < self.world.height and 0 <= gc < self.world.width:
                    # 0: Terrain
                    obs[0, wr, wc] = float(self.world.grid[gr, gc]) / num_terrains
                    # 1: Fog of war
                    obs[1, wr, wc] = 1.0 if agent.local_explored is not None and agent.local_explored[gr, gc] else 0.0
                else:
                    # Out of bounds treated as -1 (impassable and distinct from buildings)
                    obs[0, wr, wc] = float(-1.0) / num_terrains
                    # Treat out of bounds as already explored so agents don't try to explore the void
                    obs[1, wr, wc] = 1.0
                    
        # 2: Self
        obs[2, self.half_fov, self.half_fov] = 1.0
        
        # 3: Other agents
        for other in self.world.active_agents:
            if other.id != agent_id:
                orow, ocol = other.grid_pos
                wr, wc = orow - r + self.half_fov, ocol - c + self.half_fov
                if 0 <= wr < self.fov_size and 0 <= wc < self.fov_size:
                    # Only visible if explored (in real setting they share locations though)
                    obs[3, wr, wc] = 1.0
                    
        # 4: Known victims
        for victim_id in agent.local_victims_known:
            victim = self.world.victims[victim_id]
            if victim.is_alive and not victim.rescued:
                vr, vc = victim.row, victim.col
                if agent.local_explored is not None and agent.local_explored[vr, vc]: # Only known if currently in local explored
                    wr, wc = vr - r + self.half_fov, vc - c + self.half_fov
                    if 0 <= wr < self.fov_size and 0 <= wc < self.fov_size:
                        sev_val = SEVERITY_DEGRADATION_RATE.get(victim.severity.value, 0.5)
                        obs[4, wr, wc] = sev_val
                        
        # 5: Target Compass
        target_r, target_c = None, None
        
        # Decide target
        if not agent.spec.can_rescue:
            # DRONE MODE: Nearest unexplored cell
            unexplored_y, unexplored_x = np.where(~self.world.explored)
            if len(unexplored_y) > 0:
                dists = np.abs(unexplored_y - r).astype(float) + np.abs(unexplored_x - c).astype(float)
                
                center_r = self.world.height / 2.0
                center_c = self.world.width / 2.0
                dist_to_center = np.abs(unexplored_y - center_r) + np.abs(unexplored_x - center_c)
                
                prev_r = agent.prev_row if agent.prev_row is not None else agent.row
                prev_c = agent.prev_col if agent.prev_col is not None else agent.col
                mom_r = agent.row - prev_r
                mom_c = agent.col - prev_c
                dot = (unexplored_y - r) * mom_r + (unexplored_x - c) * mom_c
                
                dists = dists.astype(np.float32) + dist_to_center.astype(np.float32) * 0.0001 - dot.astype(np.float32) * 0.1
                
                idx = np.argmin(dists)
                target_r, target_c = unexplored_y[idx], unexplored_x[idx]
        elif agent.carrying_victim:
            # Nearest hospital
            min_dist = float('inf')
            for hr, hc in self.world.city.hospitals:
                dist = abs(r - hr) + abs(c - hc)
                if dist < min_dist:
                    min_dist = dist
                    target_r, target_c = hr, hc
        else:
            # Nearest known unrescued victim
            min_dist = float('inf')
            for victim_id in agent.local_victims_known:
                v = self.world.victims[victim_id]
                if v.is_alive and not v.rescued:
                    dist = abs(r - v.row) + abs(c - v.col)
                    if dist < min_dist:
                        min_dist = dist
                        target_r, target_c = v.row, v.col
                        
        if target_r is not None and target_c is not None:
            agent.assigned_target_pos = (target_r, target_c)
            
            # Calculate relative position to the agent
            rel_r = target_r - r
            rel_c = target_c - c
            
            # Clamp the relative position to the edges of the FOV grid to act as a compass
            wr = max(0, min(self.fov_size - 1, rel_r + self.half_fov))
            wc = max(0, min(self.fov_size - 1, rel_c + self.half_fov))
            
            # Place the compass point on the grid
            obs[5, wr, wc] = 1.0
                        
        return obs
