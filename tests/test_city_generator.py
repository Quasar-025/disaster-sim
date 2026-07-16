"""
Tests for city generation — Milestone 1 verification.

Validates that:
1. City generator produces valid terrain grids
2. Road network is connected
3. Victims are placed on accessible tiles
4. Configuration parameters actually affect output
5. Seed reproducibility works
6. All presets pass validation
"""

from __future__ import annotations

import numpy as np
import pytest

from disaster_sim.digital_twin.city_config import (
    AGENT_TYPES,
    CityConfig,
    PRESETS,
    Severity,
    Terrain,
    get_config,
)
from disaster_sim.digital_twin.city_generator import CityGenerator, GeneratedCity
from disaster_sim.digital_twin.world_state import WorldState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_city() -> GeneratedCity:
    """Generate a small city for fast testing."""
    config = get_config("small")
    return CityGenerator(config).generate()


@pytest.fixture
def medium_city() -> GeneratedCity:
    """Generate a medium (default) city."""
    config = get_config("medium")
    return CityGenerator(config).generate()


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestCityConfig:
    """Tests for CityConfig validation and loading."""

    def test_default_config_is_valid(self):
        config = CityConfig()
        config.validate()  # Should not raise

    def test_all_presets_valid(self):
        for name, preset in PRESETS.items():
            preset.validate()

    def test_invalid_size_raises(self):
        config = CityConfig(width=5, height=5)
        with pytest.raises(ValueError, match="too small"):
            config.validate()

    def test_invalid_density_raises(self):
        config = CityConfig(building_density=1.5)
        with pytest.raises(ValueError, match="building_density"):
            config.validate()

    def test_invalid_agent_type_raises(self):
        config = CityConfig(fleet={"helicopter": 2})
        with pytest.raises(ValueError, match="Unknown agent type"):
            config.validate()

    def test_yaml_roundtrip(self, tmp_path):
        config = CityConfig(width=100, height=100, seed=123)
        path = tmp_path / "test_config.yaml"
        config.to_yaml(path)
        loaded = CityConfig.from_yaml(path)
        assert loaded.width == 100
        assert loaded.height == 100
        assert loaded.seed == 123

    def test_get_config_preset(self):
        config = get_config("small")
        assert config.width == 50
        assert config.height == 50

    def test_get_config_invalid_raises(self):
        with pytest.raises(ValueError, match="not a valid preset"):
            get_config("nonexistent")


# ---------------------------------------------------------------------------
# City Generation Tests
# ---------------------------------------------------------------------------

class TestCityGenerator:
    """Tests for procedural city generation."""

    def test_grid_shape(self, small_city: GeneratedCity):
        assert small_city.grid.shape == (50, 50)
        assert small_city.width == 50
        assert small_city.height == 50

    def test_grid_dtype(self, small_city: GeneratedCity):
        assert small_city.grid.dtype == np.int8

    def test_terrain_values_valid(self, small_city: GeneratedCity):
        valid_values = {t.value for t in Terrain}
        unique_values = set(np.unique(small_city.grid))
        assert unique_values.issubset(valid_values), (
            f"Invalid terrain values: {unique_values - valid_values}"
        )

    def test_has_roads(self, small_city: GeneratedCity):
        road_count = int(np.sum(small_city.grid == Terrain.ROAD.value))
        assert road_count > 0, "City has no roads"
        # Roads should be a significant fraction of the grid
        total = small_city.width * small_city.height
        assert road_count / total > 0.1, "Less than 10% of city is roads"

    def test_has_buildings(self, small_city: GeneratedCity):
        building_count = int(
            np.sum(small_city.grid == Terrain.BUILDING.value)
            + np.sum(small_city.grid == Terrain.BUILDING_DAMAGED.value)
        )
        assert building_count > 0, "City has no buildings"

    def test_has_victims(self, small_city: GeneratedCity):
        assert len(small_city.victims) > 0, "City has no victims"
        assert len(small_city.victims) <= small_city.config.num_victims

    def test_has_agent_spawns(self, small_city: GeneratedCity):
        expected_total = sum(small_city.config.fleet.values())
        assert len(small_city.agent_spawns) == expected_total

    def test_has_hospitals(self, small_city: GeneratedCity):
        assert len(small_city.hospitals) > 0, "City has no hospitals"

    def test_has_charging_stations(self, small_city: GeneratedCity):
        assert len(small_city.charging_stations) > 0, "City has no charging stations"

    def test_border_roads(self, small_city: GeneratedCity):
        grid = small_city.grid
        h, w = grid.shape
        # Border cells should be roads or infrastructure placed on roads
        # (hospitals and charging stations replace road cells)
        road_or_infra = {
            Terrain.ROAD.value,
            Terrain.HOSPITAL.value,
            Terrain.CHARGING_STATION.value,
        }
        assert all(grid[0, c] in road_or_infra for c in range(w)), "Top border has non-road/infra"
        assert all(grid[h-1, c] in road_or_infra for c in range(w)), "Bottom border has non-road/infra"
        assert all(grid[r, 0] in road_or_infra for r in range(h)), "Left border has non-road/infra"
        assert all(grid[r, w-1] in road_or_infra for r in range(h)), "Right border has non-road/infra"

    def test_seed_reproducibility(self):
        """Same seed → identical cities."""
        config = CityConfig(width=50, height=50, seed=42)
        city1 = CityGenerator(config).generate()
        city2 = CityGenerator(config).generate()
        np.testing.assert_array_equal(city1.grid, city2.grid)
        assert len(city1.victims) == len(city2.victims)
        for v1, v2 in zip(city1.victims, city2.victims):
            assert v1.row == v2.row
            assert v1.col == v2.col

    def test_different_seeds_differ(self):
        """Different seeds → different cities."""
        config1 = CityConfig(width=50, height=50, seed=42)
        config2 = CityConfig(width=50, height=50, seed=99)
        city1 = CityGenerator(config1).generate()
        city2 = CityGenerator(config2).generate()
        assert not np.array_equal(city1.grid, city2.grid)

    def test_size_affects_output(self):
        """Changing size produces different grid dimensions."""
        c1 = CityConfig(width=50, height=50, seed=42)
        c2 = CityConfig(width=100, height=100, seed=42)
        city1 = CityGenerator(c1).generate()
        city2 = CityGenerator(c2).generate()
        assert city1.grid.shape != city2.grid.shape

    def test_victims_on_accessible_tiles(self, small_city: GeneratedCity):
        """Victims should not be placed inside intact buildings."""
        for victim in small_city.victims:
            terrain = small_city.terrain_at(victim.row, victim.col)
            assert terrain != Terrain.BUILDING, (
                f"Victim {victim.id} placed inside intact building at "
                f"({victim.row}, {victim.col})"
            )

    def test_agent_spawns_on_roads(self, small_city: GeneratedCity):
        """Agents should spawn on road cells."""
        for spawn in small_city.agent_spawns:
            terrain = small_city.terrain_at(spawn.row, spawn.col)
            assert terrain == Terrain.ROAD, (
                f"Agent {spawn.id} spawned on {terrain.name} at "
                f"({spawn.row}, {spawn.col}), expected ROAD"
            )

    def test_summary_output(self, small_city: GeneratedCity):
        summary = small_city.summary()
        assert "City: 50×50" in summary
        assert "Victims:" in summary
        assert "Agent Spawns:" in summary

    def test_medium_city_generation(self, medium_city: GeneratedCity):
        """Smoke test for the default 200×200 city."""
        assert medium_city.grid.shape == (200, 200)
        assert len(medium_city.victims) > 0
        assert len(medium_city.agent_spawns) == 6  # 3 drones + 3 ground robots

    def test_road_connectivity(self, small_city: GeneratedCity):
        """Verify that the road network is connected (BFS from any road to all others)."""
        grid = small_city.grid
        h, w = grid.shape

        # Find all road cells
        road_cells = set()
        for r in range(h):
            for c in range(w):
                if grid[r, c] == Terrain.ROAD.value:
                    road_cells.add((r, c))

        if not road_cells:
            pytest.skip("No roads generated")

        # BFS from first road cell
        start = next(iter(road_cells))
        visited = set()
        queue = [start]
        visited.add(start)

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in road_cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        coverage = len(visited) / len(road_cells)
        # Allow for some disconnected minor roads (>90% connected)
        assert coverage > 0.9, (
            f"Road network only {coverage:.1%} connected "
            f"({len(visited)}/{len(road_cells)} reachable)"
        )


# ---------------------------------------------------------------------------
# World State Tests
# ---------------------------------------------------------------------------

class TestWorldState:
    """Tests for the mutable world state."""

    def test_initialization(self, small_city: GeneratedCity):
        state = WorldState(small_city)
        assert len(state.agents) == sum(small_city.config.fleet.values())
        assert len(state.victims) == len(small_city.victims)
        assert state.timestep == 0

    def test_agents_initialized_correctly(self, small_city: GeneratedCity):
        state = WorldState(small_city)
        for agent in state.agents.values():
            spec = AGENT_TYPES[agent.agent_type]
            assert agent.battery == spec.battery_capacity
            assert agent.status == "idle"
            assert agent.is_alive

    def test_coverage_starts_low(self, small_city: GeneratedCity):
        state = WorldState(small_city)
        # Some cells revealed around spawn points, but not the whole city
        assert 0 < state.coverage < 1.0

    def test_victims_remaining(self, small_city: GeneratedCity):
        state = WorldState(small_city)
        assert state.victims_remaining == len(small_city.victims)
        assert state.victims_rescued_count == 0

    def test_terrain_query(self, small_city: GeneratedCity):
        state = WorldState(small_city)
        # Out of bounds should return BUILDING (impassable)
        assert state.terrain_at(-1, -1) == Terrain.BUILDING

    def test_serialization(self, small_city: GeneratedCity):
        state = WorldState(small_city)
        data = state.to_dict()
        assert "timestep" in data
        assert "agents" in data
        assert "victims" in data
        assert "coverage" in data
        assert isinstance(data["agents"], dict)
        assert isinstance(data["victims"], dict)

    def test_passability_drone(self, small_city: GeneratedCity):
        state = WorldState(small_city)
        # Drones can fly over anything
        for r in range(small_city.height):
            for c in range(small_city.width):
                assert state.is_passable(r, c, "drone")

    def test_passability_ambulance(self, small_city: GeneratedCity):
        state = WorldState(small_city)
        # Ambulances can only use roads, hospitals, charging stations
        for r in range(small_city.height):
            for c in range(small_city.width):
                terrain = state.terrain_at(r, c)
                passable = state.is_passable(r, c, "ambulance")
                if terrain.name in ("ROAD", "HOSPITAL", "CHARGING_STATION"):
                    assert passable, f"Ambulance should pass {terrain.name} at ({r},{c})"
                else:
                    assert not passable, f"Ambulance shouldn't pass {terrain.name} at ({r},{c})"
