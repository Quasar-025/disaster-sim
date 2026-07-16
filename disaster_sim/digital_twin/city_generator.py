"""
Procedural City Generator — Creates a disaster-zone city layout on a 2D grid.

Generation Pipeline:
    1. Lay down road network (major + minor grid pattern)
    2. Place buildings in blocks between roads
    3. Add parks and open spaces
    4. Generate rivers and water features
    5. Apply disaster damage (collapse buildings, spread debris)
    6. Ignite fire zones
    7. Place hospitals and charging stations on roads
    8. Spawn victims near disaster zones
    9. Validate connectivity (ensure road network is traversable)

All generation is deterministic given a seed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from disaster_sim.digital_twin.city_config import (
    AGENT_TYPES,
    CityConfig,
    Severity,
    Terrain,
    get_config,
)


# ---------------------------------------------------------------------------
# Data Structures for Generated City
# ---------------------------------------------------------------------------

@dataclass
class VictimInfo:
    """A victim placed in the city."""

    id: str
    row: int
    col: int
    severity: Severity
    time_remaining: float  # Steps until condition degrades


@dataclass
class AgentSpawn:
    """Spawn point for an agent."""

    id: str
    agent_type: str
    row: int
    col: int


@dataclass
class GeneratedCity:
    """Output of the city generation process.

    Contains the terrain grid, victim placements, agent spawn points,
    and metadata about the generation.
    """

    grid: np.ndarray               # Shape: (height, width), dtype=int8 (Terrain values)
    config: CityConfig
    victims: list[VictimInfo]
    agent_spawns: list[AgentSpawn]
    hospitals: list[tuple[int, int]]
    charging_stations: list[tuple[int, int]]
    road_cells: list[tuple[int, int]]

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    @property
    def height(self) -> int:
        return self.grid.shape[0]

    def terrain_at(self, row: int, col: int) -> Terrain:
        """Get the terrain type at a given position."""
        return Terrain(self.grid[row, col])

    def is_passable(self, row: int, col: int, agent_type: str) -> bool:
        """Check if a cell is passable by a given agent type."""
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        terrain = self.terrain_at(row, col)
        spec = AGENT_TYPES[agent_type]
        return terrain.name in spec.terrain_passable

    def summary(self) -> str:
        """Return a text summary of the generated city."""
        counts = {}
        for t in Terrain:
            counts[t.name] = int(np.sum(self.grid == t.value))
        lines = [
            f"City: {self.width}×{self.height} (seed={self.config.seed})",
            f"Terrain Distribution:",
        ]
        for name, count in counts.items():
            pct = 100.0 * count / (self.width * self.height)
            lines.append(f"  {name:20s}: {count:6d} ({pct:5.1f}%)")
        lines.append(f"Victims: {len(self.victims)}")
        severity_counts = {}
        for v in self.victims:
            severity_counts[v.severity.value] = severity_counts.get(v.severity.value, 0) + 1
        for sev, cnt in severity_counts.items():
            lines.append(f"  {sev}: {cnt}")
        lines.append(f"Agent Spawns: {len(self.agent_spawns)}")
        for spawn in self.agent_spawns:
            lines.append(f"  {spawn.id} ({spawn.agent_type}) at ({spawn.row}, {spawn.col})")
        lines.append(f"Hospitals: {len(self.hospitals)}")
        lines.append(f"Charging Stations: {len(self.charging_stations)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# City Generator
# ---------------------------------------------------------------------------

class CityGenerator:
    """Procedural city generator.

    Creates a complete disaster-zone city from a CityConfig specification.
    All random operations use a seeded numpy RNG for reproducibility.
    """

    def __init__(self, config: CityConfig):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.grid = np.full((config.height, config.width), Terrain.EMPTY.value, dtype=np.int8)
        self._road_cells: list[tuple[int, int]] = []
        self._building_cells: list[tuple[int, int]] = []
        self._hospitals: list[tuple[int, int]] = []
        self._charging_stations: list[tuple[int, int]] = []
        self._victims: list[VictimInfo] = []
        self._spawns: list[AgentSpawn] = []

    def generate(self) -> GeneratedCity:
        """Run the full generation pipeline and return the result."""
        self._lay_roads()
        self._place_buildings()
        self._place_parks()
        self._generate_water()
        self._apply_disaster()
        self._place_fires()
        self._place_hospitals()
        self._place_charging_stations()
        self._spawn_victims()
        self._spawn_agents()
        self._place_traffic_lights()
        self._spawn_citizens()

        return GeneratedCity(
            grid=self.grid.copy(),
            config=self.config,
            victims=self._victims,
            agent_spawns=self._spawns,
            hospitals=self._hospitals,
            charging_stations=self._charging_stations,
            road_cells=self._road_cells,
        )

    # -- Step 1: Road Network --

    def _lay_roads(self) -> None:
        """Create a grid-based road network with major and minor roads."""
        cfg = self.config
        h, w = cfg.height, cfg.width

        # Major roads — spine of the city
        for r in range(0, h, cfg.major_road_spacing):
            for c in range(w):
                for dr in range(cfg.road_width):
                    rr = min(r + dr, h - 1)
                    self.grid[rr, c] = Terrain.ROAD.value

        for c in range(0, w, cfg.major_road_spacing):
            for r in range(h):
                for dc in range(cfg.road_width):
                    cc = min(c + dc, w - 1)
                    self.grid[r, cc] = Terrain.ROAD.value

        # Minor roads within blocks
        for r in range(0, h, cfg.minor_road_spacing):
            for c in range(w):
                if self.grid[r, c] == Terrain.EMPTY.value:
                    self.grid[r, c] = Terrain.ROAD.value

        for c in range(0, w, cfg.minor_road_spacing):
            for r in range(h):
                if self.grid[r, c] == Terrain.EMPTY.value:
                    self.grid[r, c] = Terrain.ROAD.value

        # Border roads
        self.grid[0, :] = Terrain.ROAD.value
        self.grid[h - 1, :] = Terrain.ROAD.value
        self.grid[:, 0] = Terrain.ROAD.value
        self.grid[:, w - 1] = Terrain.ROAD.value

        # Collect road cells
        self._road_cells = [
            (r, c) for r in range(h) for c in range(w)
            if self.grid[r, c] == Terrain.ROAD.value
        ]

    # -- Step 2: Buildings --

    def _place_buildings(self) -> None:
        """Fill empty cells between roads with buildings of varying sizes."""
        cfg = self.config
        h, w = cfg.height, cfg.width

        # Find empty cells (candidates for buildings)
        empty_mask = self.grid == Terrain.EMPTY.value
        total_empty = int(np.sum(empty_mask))
        target_building_cells = int(total_empty * cfg.building_density)
        placed = 0

        # Try to place buildings of random sizes
        attempts = 0
        max_attempts = target_building_cells * 4

        while placed < target_building_cells and attempts < max_attempts:
            attempts += 1
            bh = self.rng.integers(cfg.min_building_size, cfg.max_building_size + 1)
            bw = self.rng.integers(cfg.min_building_size, cfg.max_building_size + 1)
            r = self.rng.integers(0, h - bh + 1)
            c = self.rng.integers(0, w - bw + 1)

            # Check if entire footprint is empty
            footprint = self.grid[r:r + bh, c:c + bw]
            if np.all(footprint == Terrain.EMPTY.value):
                self.grid[r:r + bh, c:c + bw] = Terrain.BUILDING.value
                for dr in range(bh):
                    for dc in range(bw):
                        self._building_cells.append((r + dr, c + dc))
                placed += bh * bw

    # -- Step 3: Parks --

    def _place_parks(self) -> None:
        """Place parks as circular open spaces in the city."""
        cfg = self.config
        h, w = cfg.height, cfg.width

        for _ in range(cfg.num_parks):
            cr = self.rng.integers(cfg.park_radius + 5, h - cfg.park_radius - 5)
            cc = self.rng.integers(cfg.park_radius + 5, w - cfg.park_radius - 5)

            for dr in range(-cfg.park_radius, cfg.park_radius + 1):
                for dc in range(-cfg.park_radius, cfg.park_radius + 1):
                    r, c = cr + dr, cc + dc
                    if 0 <= r < h and 0 <= c < w:
                        dist = (dr * dr + dc * dc) ** 0.5
                        if dist <= cfg.park_radius and self.grid[r, c] != Terrain.ROAD.value:
                            self.grid[r, c] = Terrain.PARK.value

    # -- Step 4: Water (Rivers + Flood Zones) --

    def _generate_water(self) -> None:
        """Generate rivers (serpentine paths) and flood zones."""
        cfg = self.config
        h, w = cfg.height, cfg.width

        # Rivers: flow from top to bottom with horizontal meandering
        for _ in range(cfg.num_rivers):
            col = self.rng.integers(w // 4, 3 * w // 4)
            drift = 0.0

            for row in range(h):
                drift += self.rng.normal(0, 0.3)
                drift = np.clip(drift, -2, 2)
                center_col = int(col + drift)

                for dc in range(-cfg.river_width // 2, cfg.river_width // 2 + 1):
                    c = center_col + dc
                    if 0 <= c < w:
                        self.grid[row, c] = Terrain.WATER.value

                col = center_col

        # Flood zones: circular water patches
        for _ in range(cfg.flood_zone_count):
            cr = self.rng.integers(10, h - 10)
            cc = self.rng.integers(10, w - 10)
            radius = self.rng.integers(
                max(2, cfg.flood_zone_radius // 2),
                cfg.flood_zone_radius + 1,
            )

            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    r, c = cr + dr, cc + dc
                    if 0 <= r < h and 0 <= c < w:
                        dist = (dr * dr + dc * dc) ** 0.5
                        if dist <= radius:
                            # Don't flood roads (they're infrastructure)
                            if self.grid[r, c] not in (
                                Terrain.ROAD.value,
                                Terrain.HOSPITAL.value,
                                Terrain.CHARGING_STATION.value,
                            ):
                                self.grid[r, c] = Terrain.WATER.value

    # -- Step 5: Disaster Damage --

    def _apply_disaster(self) -> None:
        """Damage buildings and spread debris around collapse sites."""
        cfg = self.config
        h, w = cfg.height, cfg.width

        if not self._building_cells:
            return

        # Pick buildings to damage (unique building origins)
        building_set = set(self._building_cells)
        damaged_origins = []

        # Sample random building cells and mark their buildings as damaged
        num_to_damage = min(cfg.num_damaged_buildings, len(self._building_cells) // 4)
        if num_to_damage == 0:
            return

        indices = self.rng.choice(len(self._building_cells), size=num_to_damage, replace=False)

        for idx in indices:
            br, bc = self._building_cells[idx]
            if self.grid[br, bc] == Terrain.BUILDING.value:
                self.grid[br, bc] = Terrain.BUILDING_DAMAGED.value
                damaged_origins.append((br, bc))

        # Spread debris around damaged buildings
        for dr, dc in damaged_origins:
            for ddr in range(-cfg.debris_spread, cfg.debris_spread + 1):
                for ddc in range(-cfg.debris_spread, cfg.debris_spread + 1):
                    r, c = dr + ddr, dc + ddc
                    if 0 <= r < h and 0 <= c < w:
                        dist = abs(ddr) + abs(ddc)  # Manhattan distance
                        if dist <= cfg.debris_spread:
                            # Probability decreases with distance
                            prob = 1.0 - (dist / (cfg.debris_spread + 1))
                            if self.rng.random() < prob * 0.6:
                                if self.grid[r, c] in (
                                    Terrain.EMPTY.value,
                                    Terrain.PARK.value,
                                ):
                                    self.grid[r, c] = Terrain.DEBRIS.value
                                elif self.grid[r, c] == Terrain.ROAD.value:
                                    # Roads get blocked by debris sometimes
                                    if self.rng.random() < 0.3:
                                        self.grid[r, c] = Terrain.DEBRIS.value

    # -- Step 6: Fire --

    def _place_fires(self) -> None:
        """Ignite fire zones near damaged buildings."""
        cfg = self.config
        h, w = cfg.height, cfg.width

        # Find damaged building cells as fire ignition candidates
        damaged = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if self.grid[r, c] == Terrain.BUILDING_DAMAGED.value
        ]

        if not damaged:
            return

        num_fires = min(cfg.num_fire_zones, len(damaged))
        fire_indices = self.rng.choice(len(damaged), size=num_fires, replace=False)

        for idx in fire_indices:
            fr, fc = damaged[idx]
            radius = cfg.fire_spread_radius

            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    r, c = fr + dr, fc + dc
                    if 0 <= r < h and 0 <= c < w:
                        dist = (dr * dr + dc * dc) ** 0.5
                        if dist <= radius:
                            # Fire spreads to ground-level terrain
                            if self.grid[r, c] in (
                                Terrain.EMPTY.value,
                                Terrain.DEBRIS.value,
                                Terrain.PARK.value,
                                Terrain.BUILDING_DAMAGED.value,
                            ):
                                self.grid[r, c] = Terrain.FIRE.value

    # -- Step 7: Hospitals --

    def _place_hospitals(self) -> None:
        """Place hospitals on road cells, spread around the city edges."""
        cfg = self.config
        h, w = cfg.height, cfg.width

        # Place hospitals near edges for realism
        edge_roads = [
            (r, c) for r, c in self._road_cells
            if r < 10 or r >= h - 10 or c < 10 or c >= w - 10
        ]

        if not edge_roads:
            edge_roads = self._road_cells

        if not edge_roads:
            return

        # Pick well-spaced hospital locations
        placed: list[tuple[int, int]] = []
        candidates = list(edge_roads)
        self.rng.shuffle(candidates)

        for r, c in candidates:
            if len(placed) >= cfg.num_hospitals:
                break
            # Ensure minimum spacing between hospitals
            if all(
                abs(r - pr) + abs(c - pc) > max(h, w) // (cfg.num_hospitals + 1)
                for pr, pc in placed
            ):
                self.grid[r, c] = Terrain.HOSPITAL.value
                placed.append((r, c))

        # If spacing constraint was too strict, just place remaining
        if len(placed) < cfg.num_hospitals:
            for r, c in candidates:
                if len(placed) >= cfg.num_hospitals:
                    break
                if (r, c) not in placed:
                    self.grid[r, c] = Terrain.HOSPITAL.value
                    placed.append((r, c))

        self._hospitals = placed

    # -- Step 8: Charging Stations --

    def _place_charging_stations(self) -> None:
        """Place charging stations on road cells, distributed across the city."""
        cfg = self.config
        h, w = cfg.height, cfg.width

        # Prefer interior roads
        interior_roads = [
            (r, c) for r, c in self._road_cells
            if 5 < r < h - 5 and 5 < c < w - 5
            and self.grid[r, c] == Terrain.ROAD.value  # Not already a hospital
        ]

        if not interior_roads:
            interior_roads = [
                (r, c) for r, c in self._road_cells
                if self.grid[r, c] == Terrain.ROAD.value
            ]

        if not interior_roads:
            return

        placed: list[tuple[int, int]] = []
        candidates = list(interior_roads)
        self.rng.shuffle(candidates)

        for r, c in candidates:
            if len(placed) >= cfg.num_charging_stations:
                break
            # Ensure spacing
            if all(
                abs(r - pr) + abs(c - pc) > max(h, w) // (cfg.num_charging_stations + 1)
                for pr, pc in placed
            ):
                self.grid[r, c] = Terrain.CHARGING_STATION.value
                placed.append((r, c))

        # Fill remaining if spacing was too strict
        if len(placed) < cfg.num_charging_stations:
            for r, c in candidates:
                if len(placed) >= cfg.num_charging_stations:
                    break
                if (r, c) not in placed and self.grid[r, c] == Terrain.ROAD.value:
                    self.grid[r, c] = Terrain.CHARGING_STATION.value
                    placed.append((r, c))

        self._charging_stations = placed

    # -- Step 9: Victims --

    def _spawn_victims(self) -> None:
        """Place victims near disaster zones (debris, damaged buildings, fire edges)."""
        cfg = self.config
        h, w = cfg.height, cfg.width

        # Candidate cells: debris, empty/park cells near damaged buildings, road near disaster
        candidates: list[tuple[int, int]] = []

        for r in range(h):
            for c in range(w):
                terrain = self.grid[r, c]
                if terrain == Terrain.DEBRIS.value:
                    candidates.extend([(r, c)] * 3)  # Higher weight for debris
                elif terrain in (Terrain.ROAD.value, Terrain.EMPTY.value, Terrain.PARK.value):
                    # Check if near a disaster zone
                    for dr in range(-3, 4):
                        for dc in range(-3, 4):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if self.grid[nr, nc] in (
                                    Terrain.BUILDING_DAMAGED.value,
                                    Terrain.DEBRIS.value,
                                    Terrain.FIRE.value,
                                ):
                                    candidates.append((r, c))
                                    break
                        else:
                            continue
                        break

        if not candidates:
            # Fallback: place on any accessible cell
            candidates = [
                (r, c) for r in range(h) for c in range(w)
                if self.grid[r, c] in (Terrain.ROAD.value, Terrain.EMPTY.value)
            ]

        if not candidates:
            return

        # Sample victim positions (with replacement if needed)
        num_victims = min(cfg.num_victims, len(candidates))
        indices = self.rng.choice(len(candidates), size=num_victims, replace=False)

        # Determine severity distribution
        severities = []
        for sev_name, frac in cfg.victim_severity_distribution.items():
            count = max(1, int(num_victims * frac))
            severities.extend([Severity(sev_name)] * count)
        # Trim or pad
        severities = severities[:num_victims]
        while len(severities) < num_victims:
            severities.append(Severity.STABLE)
        self.rng.shuffle(severities)

        for i, idx in enumerate(indices):
            r, c = candidates[idx]
            victim = VictimInfo(
                id=f"victim_{i:03d}",
                row=r,
                col=c,
                severity=severities[i],
                time_remaining=cfg.victim_time_limit * (1.0 / SEVERITY_DEGRADATION_RATE[severities[i].value])
                if severities[i].value in ("critical",) else cfg.victim_time_limit,
            )
            self._victims.append(victim)

    # -- Step 10: Agent Spawns --

    def _spawn_agents(self) -> None:
        """Create agent spawn points on road cells near charging stations."""
        cfg = self.config
        h, w = cfg.height, cfg.width

        # Prefer spawning near charging stations, else near city edges
        spawn_candidates = []

        # Near charging stations
        for sr, sc in self._charging_stations:
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    r, c = sr + dr, sc + dc
                    if 0 <= r < h and 0 <= c < w:
                        if self.grid[r, c] == Terrain.ROAD.value:
                            spawn_candidates.append((r, c))

        if not spawn_candidates:
            spawn_candidates = [
                (r, c) for r, c in self._road_cells
                if self.grid[r, c] == Terrain.ROAD.value
            ]

        if not spawn_candidates:
            return

        agent_counter = 0
        for agent_type, count in cfg.fleet.items():
            for i in range(count):
                idx = self.rng.integers(0, len(spawn_candidates))
                r, c = spawn_candidates[idx]
                spawn = AgentSpawn(
                    id=f"{agent_type}_{i:02d}",
                    agent_type=agent_type,
                    row=r,
                    col=c,
                )
                self._spawns.append(spawn)
                agent_counter += 1

    # -- Step 11: Citizens and Traffic Lights --

    def _place_traffic_lights(self) -> None:
        """Place traffic light agents at road intersections."""
        h, w = self.config.height, self.config.width
        tl_idx = 0
        
        for r, c in self._road_cells:
            # Check 4-way adjacent cells
            adjacent_roads = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and self.grid[nr, nc] == Terrain.ROAD.value:
                    adjacent_roads += 1
            
            # If it's an intersection (3 or 4 adjacent roads), spawn a traffic light
            if adjacent_roads >= 3:
                # Add some spacing to avoid spamming them on wide roads
                if tl_idx % 4 == 0: 
                    spawn = AgentSpawn(
                        id=f"traffic_light_{tl_idx:03d}",
                        agent_type="traffic_light",
                        row=r,
                        col=c,
                    )
                    self._spawns.append(spawn)
                tl_idx += 1

    def _spawn_citizens(self) -> None:
        """Spawn citizen agents randomly in parks and open areas."""
        h, w = self.config.height, self.config.width
        cit_idx = 0
        num_citizens = max(5, self.config.num_victims // 2)
        
        candidates = [
            (r, c) for r in range(h) for c in range(w) 
            if self.grid[r, c] in (Terrain.PARK.value, Terrain.EMPTY.value)
        ]
        
        if not candidates:
            return
            
        indices = self.rng.choice(len(candidates), size=min(num_citizens, len(candidates)), replace=False)
        for idx in indices:
            r, c = candidates[idx]
            spawn = AgentSpawn(
                id=f"citizen_{cit_idx:03d}",
                agent_type="citizen",
                row=r,
                col=c,
            )
            self._spawns.append(spawn)
            cit_idx += 1


# ---------------------------------------------------------------------------
# Severity degradation rate lookup (needed for victim time calculation)
# ---------------------------------------------------------------------------

SEVERITY_DEGRADATION_RATE: dict[str, float] = {
    "critical": 1.0,
    "serious": 0.5,
    "stable": 0.2,
}


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for city generation."""
    parser = argparse.ArgumentParser(
        description="Generate a procedural disaster-zone city."
    )
    parser.add_argument(
        "--preset", "-p",
        default="medium",
        help="Preset name (small/medium/large) or path to YAML config",
    )
    parser.add_argument(
        "--size", "-s",
        type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
        help="Override city dimensions (e.g., --size 200 200)",
    )
    parser.add_argument(
        "--seed",
        type=int, default=None,
        help="Random seed (overrides preset/config seed)",
    )
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Open a Pygame window to visualize the generated city",
    )
    parser.add_argument(
        "--save-image", "-o",
        type=str, default=None,
        help="Save the rendered city to an image file (PNG)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print a text summary of the generated city",
    )
    args = parser.parse_args()

    config = get_config(args.preset)

    if args.size:
        config.width, config.height = args.size
    if args.seed is not None:
        config.seed = args.seed

    config.validate()

    print(f"Generating city ({config.width}×{config.height}, seed={config.seed})...")
    generator = CityGenerator(config)
    city = generator.generate()

    if args.summary:
        print()
        print(city.summary())

    if args.render or args.save_image:
        from disaster_sim.engine.renderer import CityRenderer
        renderer = CityRenderer(city)

        if args.save_image:
            renderer.save_image(args.save_image)
            print(f"\nSaved city image to: {args.save_image}")

        if args.render:
            print("\nRendering city (close window to exit)...")
            renderer.run()


if __name__ == "__main__":
    main()
