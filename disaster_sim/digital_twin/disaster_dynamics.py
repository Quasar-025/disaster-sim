"""
Disaster Spread Dynamics (Cellular Automata)

Simulates the physics-based spread of disasters using 2D Cellular Automata.
- Fire spread: influenced by wind direction, fuel types (buildings, debris, parks).
- Flood spread: influenced by terrain permeability and existing water levels.
"""

import numpy as np
from scipy.signal import convolve2d

from disaster_sim.digital_twin.world_state import WorldState
from disaster_sim.digital_twin.city_config import Terrain


class DisasterDynamics:
    """Manages the Cellular Automata updates for the WorldState."""

    def __init__(self, world: WorldState):
        self.world = world
        self.rng = np.random.default_rng()
        
        # Wind vector (vx, vy). Positive vx = East, Positive vy = South
        self.wind_vx = 0.5
        self.wind_vy = -0.2

    def step(self):
        """Advance the CA simulation by one tick."""
        self._step_fire()
        self._step_flood()

    def _step_fire(self):
        """
        Fire Spread CA.
        Rules:
        - Fire spreads to adjacent combustible cells.
        - Probability of ignition is biased by wind direction.
        - Burning cells eventually burn out.
        """
        grid = self.world.grid
        
        # Identify current burning cells
        burning_mask = (grid == Terrain.FIRE.value)
        
        # Identify combustible cells
        combustible_mask = np.isin(grid, [
            Terrain.BUILDING.value,
            Terrain.BUILDING_DAMAGED.value,
            Terrain.DEBRIS.value,
            Terrain.PARK.value,
        ])
        
        if not np.any(burning_mask):
            return

        # Base ignition probability matrix (3x3 kernel)
        # Center is the burning cell itself
        base_ignition_prob = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 0.0, 0.2],
            [0.1, 0.2, 0.1]
        ])
        
        # Bias the ignition probability based on wind
        # If wind is blowing East (vx > 0), the right side of the kernel should have higher prob
        wind_bias = np.zeros((3, 3))
        wind_bias[1, 2] = max(0, self.wind_vx)  # East
        wind_bias[1, 0] = max(0, -self.wind_vx) # West
        wind_bias[2, 1] = max(0, self.wind_vy)  # South
        wind_bias[0, 1] = max(0, -self.wind_vy) # North
        
        # Diagonal wind bias (approximate)
        wind_bias[0, 2] = max(0, self.wind_vx) * max(0, -self.wind_vy) # NE
        wind_bias[2, 2] = max(0, self.wind_vx) * max(0, self.wind_vy)  # SE
        wind_bias[0, 0] = max(0, -self.wind_vx) * max(0, -self.wind_vy) # NW
        wind_bias[2, 0] = max(0, -self.wind_vx) * max(0, self.wind_vy)  # SW
        
        kernel = base_ignition_prob + (wind_bias * 0.5)
        
        # Convolve burning cells with the ignition kernel to get ignition pressure on each cell
        ignition_pressure = convolve2d(burning_mask.astype(float), kernel, mode='same', boundary='fill', fillvalue=0)
        
        # Roll random numbers to see which cells ignite
        ignition_rolls = self.rng.random(grid.shape)
        
        # New fires: must be combustible AND roll < pressure
        new_fires = combustible_mask & (ignition_rolls < ignition_pressure)
        
        # Apply new fires
        grid[new_fires] = Terrain.FIRE.value
        
        # Burning out logic (random chance for a fire to turn into debris/empty)
        burn_out_rolls = self.rng.random(grid.shape)
        burn_out_mask = burning_mask & (burn_out_rolls < 0.05) # 5% chance per tick to burn out
        grid[burn_out_mask] = Terrain.DEBRIS.value

    def _step_flood(self):
        """
        Flood Spread CA.
        Rules:
        - Water spreads to adjacent non-building cells.
        - Slower spread than fire.
        """
        grid = self.world.grid
        water_mask = (grid == Terrain.WATER.value)
        
        if not np.any(water_mask):
            return
            
        permeable_mask = np.isin(grid, [
            Terrain.EMPTY.value,
            Terrain.ROAD.value,
            Terrain.PARK.value,
            Terrain.DEBRIS.value
        ])
        
        # Simple orthagonal spread kernel
        kernel = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])
        
        # Convolve to find cells adjacent to water
        water_neighbors = convolve2d(water_mask.astype(float), kernel, mode='same', boundary='fill', fillvalue=0)
        
        # 20% chance for water to spread to an adjacent permeable cell per tick
        spread_rolls = self.rng.random(grid.shape)
        new_water = permeable_mask & (water_neighbors > 0) & (spread_rolls < 0.2)
        
        grid[new_water] = Terrain.WATER.value

