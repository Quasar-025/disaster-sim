"""
Disaster Injector — Dynamically applies catastrophic events to the running simulation.
"""

import numpy as np

from disaster_sim.digital_twin.city_config import Terrain
from disaster_sim.digital_twin.world_state import WorldState


class DisasterInjector:
    """Provides methods to inject disasters into a live WorldState."""

    def __init__(self, world: WorldState, seed: int = None):
        self.world = world
        self.rng = np.random.default_rng(seed)

    def trigger_earthquake(self, epicenter_r: int, epicenter_c: int, magnitude: float = 7.0) -> dict:
        """
        Triggers an earthquake at the specified location.
        Damages buildings and scatters debris based on distance and magnitude.
        
        Args:
            epicenter_r: Row of epicenter
            epicenter_c: Col of epicenter
            magnitude: Earthquake magnitude (Richter scale roughly 5.0 to 9.0)
            
        Returns:
            A summary dictionary of the damage caused.
        """
        # Base damage radius scales exponentially with magnitude roughly
        damage_radius = int(2 ** (magnitude - 4))
        
        buildings_destroyed = 0
        roads_blocked = 0
        
        h, w = self.world.height, self.world.width
        
        for r in range(max(0, epicenter_r - damage_radius), min(h, epicenter_r + damage_radius + 1)):
            for c in range(max(0, epicenter_c - damage_radius), min(w, epicenter_c + damage_radius + 1)):
                dist = np.sqrt((r - epicenter_r)**2 + (c - epicenter_c)**2)
                if dist <= damage_radius:
                    # Probability of damage decays with distance
                    prob = 1.0 - (dist / damage_radius)
                    # Scale probability by magnitude intensity (very loose approximation)
                    intensity = prob * (magnitude / 7.0)
                    
                    terrain = self.world.terrain_at(r, c)
                    
                    if terrain == Terrain.BUILDING:
                        if self.rng.random() < intensity:
                            self.world.grid[r, c] = Terrain.BUILDING_DAMAGED.value
                            buildings_destroyed += 1
                            
                    elif terrain == Terrain.ROAD:
                        # Debris falling on roads
                        if self.rng.random() < intensity * 0.5:
                            self.world.grid[r, c] = Terrain.DEBRIS.value
                            roads_blocked += 1
                            
                    elif terrain in (Terrain.EMPTY, Terrain.PARK):
                        # Ground cracking / debris
                        if self.rng.random() < intensity * 0.3:
                            self.world.grid[r, c] = Terrain.DEBRIS.value
                            
        return {
            "event": "earthquake",
            "magnitude": magnitude,
            "epicenter": (epicenter_r, epicenter_c),
            "buildings_destroyed": buildings_destroyed,
            "roads_blocked": roads_blocked
        }

    def trigger_flood(self, source_r: int, source_c: int, volume: int = 100) -> dict:
        """
        Triggers a flash flood spreading outward from a source.
        
        Args:
            source_r: Row of water source
            source_c: Col of water source
            volume: Number of cells to flood
            
        Returns:
            A summary dictionary of the flood.
        """
        cells_flooded = 0
        h, w = self.world.height, self.world.width
        
        # Simple BFS spread for flood
        queue = [(source_r, source_c)]
        visited = set(queue)
        
        while queue and cells_flooded < volume:
            r, c = queue.pop(0)
            
            terrain = self.world.terrain_at(r, c)
            
            # Water cannot overwrite intact buildings (for now) but fills roads and empty space
            if terrain not in (Terrain.BUILDING, Terrain.BUILDING_DAMAGED, Terrain.WATER):
                self.world.grid[r, c] = Terrain.WATER.value
                cells_flooded += 1
                
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if (nr, nc) not in visited:
                        # Water prefers flowing into lower terrain, but here we just do uniform spread
                        visited.add((nr, nc))
                        queue.append((nr, nc))
                        
        return {
            "event": "flood",
            "source": (source_r, source_c),
            "cells_flooded": cells_flooded
        }
