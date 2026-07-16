"""
City Configuration — Data classes and presets for procedural city generation.

Defines the terrain types, agent type specifications, victim/hazard models,
and configurable city generation parameters. Supports YAML config loading
and built-in presets (small/medium/large).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Terrain Types
# ---------------------------------------------------------------------------

class Terrain(enum.IntEnum):
    """Grid cell terrain types.

    Integer values are used as array indices for fast lookup and as channels
    in the observation space for RL training.
    """

    EMPTY = 0           # Open ground, passable by all ground units
    ROAD = 1            # Paved road — fast movement, required for ambulances
    BUILDING = 2        # Intact building — impassable
    BUILDING_DAMAGED = 3  # Damaged building — impassable, may have victims nearby
    DEBRIS = 4          # Rubble from collapsed buildings — slow traversal for ground
    WATER = 5           # River / flood zone — only boats can traverse
    FIRE = 6            # Active fire — dangerous, blocks ground units
    HOSPITAL = 7        # Medical facility — ambulance destination
    CHARGING_STATION = 8  # Agent recharging point
    PARK = 9            # Open green space — passable, potential victim location


# Terrain rendering colors (R, G, B) for the debug renderer
TERRAIN_COLORS: dict[Terrain, tuple[int, int, int]] = {
    Terrain.EMPTY: (60, 60, 60),
    Terrain.ROAD: (180, 180, 180),
    Terrain.BUILDING: (100, 100, 120),
    Terrain.BUILDING_DAMAGED: (130, 90, 70),
    Terrain.DEBRIS: (110, 85, 60),
    Terrain.WATER: (40, 100, 180),
    Terrain.FIRE: (220, 80, 30),
    Terrain.HOSPITAL: (220, 220, 240),
    Terrain.CHARGING_STATION: (50, 200, 100),
    Terrain.PARK: (60, 130, 60),
}


# ---------------------------------------------------------------------------
# Victim Severity
# ---------------------------------------------------------------------------

class Severity(enum.Enum):
    """Victim condition severity levels.

    Determines rescue priority and reward multiplier.
    """

    CRITICAL = "critical"    # Reward ×3, degrades fastest
    SERIOUS = "serious"      # Reward ×2
    STABLE = "stable"        # Reward ×1, degrades slowest


SEVERITY_REWARD_MULTIPLIER: dict[Severity, float] = {
    Severity.CRITICAL: 3.0,
    Severity.SERIOUS: 2.0,
    Severity.STABLE: 1.0,
}

SEVERITY_DEGRADATION_RATE: dict[Severity, float] = {
    Severity.CRITICAL: 1.0,   # Degrades 1 unit per timestep
    Severity.SERIOUS: 0.5,
    Severity.STABLE: 0.2,
}


# ---------------------------------------------------------------------------
# Agent Type Specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentTypeSpec:
    """Immutable specification for an agent type.

    Defines movement capabilities, resource constraints, and special abilities
    for each class of rescue vehicle.
    """

    name: str
    movement_type: str          # "aerial", "ground", "road_only", "water_only"
    speed: float                # Grid cells per step
    battery_capacity: float     # Total energy units
    battery_drain_rate: float   # Energy consumed per step
    payload_capacity: float     # Kilograms
    sensor_range: int           # Visibility radius in grid cells
    can_rescue: bool            # Can extract victims
    can_clear_debris: bool      # Can remove debris tiles
    can_transport: bool         # Can carry rescued victims to hospital
    terrain_passable: frozenset[str]  # Set of Terrain names this agent can traverse


# Built-in agent type registry
AGENT_TYPES: dict[str, AgentTypeSpec] = {
    "citizen": AgentTypeSpec(
        name="citizen",
        movement_type="ground",
        speed=0.5,
        battery_capacity=100.0, # Health / Stamina
        battery_drain_rate=0.5,
        payload_capacity=0.0,
        sensor_range=2,
        can_rescue=False,
        can_clear_debris=False,
        can_transport=False,
        terrain_passable=frozenset({"EMPTY", "ROAD", "DEBRIS", "PARK"}),
    ),
    "traffic_light": AgentTypeSpec(
        name="traffic_light",
        movement_type="stationary",
        speed=0.0,
        battery_capacity=1000000.0,
        battery_drain_rate=0.0,
        payload_capacity=0.0,
        sensor_range=5,
        can_rescue=False,
        can_clear_debris=False,
        can_transport=False,
        terrain_passable=frozenset({"ROAD"}),
    ),
    "drone": AgentTypeSpec(
        name="drone",
        movement_type="aerial",
        speed=3.0,
        battery_capacity=100.0,
        battery_drain_rate=0.5,
        payload_capacity=0.0,
        sensor_range=8,
        can_rescue=False,
        can_clear_debris=False,
        can_transport=False,
        terrain_passable=frozenset({
            "EMPTY", "ROAD", "BUILDING", "BUILDING_DAMAGED",
            "DEBRIS", "WATER", "FIRE", "HOSPITAL", "CHARGING_STATION", "PARK",
        }),
    ),
    "ground_robot": AgentTypeSpec(
        name="ground_robot",
        movement_type="ground",
        speed=1.5,
        battery_capacity=200.0,
        battery_drain_rate=0.3,
        payload_capacity=50.0,
        sensor_range=4,
        can_rescue=True,
        can_clear_debris=False,
        can_transport=False,
        terrain_passable=frozenset({
            "EMPTY", "ROAD", "DEBRIS", "HOSPITAL", "CHARGING_STATION", "PARK",
        }),
    ),
    "ambulance": AgentTypeSpec(
        name="ambulance",
        movement_type="road_only",
        speed=4.0,
        battery_capacity=500.0,
        battery_drain_rate=0.1,
        payload_capacity=200.0,
        sensor_range=3,
        can_rescue=True,
        can_clear_debris=False,
        can_transport=True,
        terrain_passable=frozenset({"ROAD", "HOSPITAL", "CHARGING_STATION"}),
    ),
    "boat": AgentTypeSpec(
        name="boat",
        movement_type="water_only",
        speed=2.0,
        battery_capacity=300.0,
        battery_drain_rate=0.2,
        payload_capacity=100.0,
        sensor_range=5,
        can_rescue=True,
        can_clear_debris=False,
        can_transport=True,
        terrain_passable=frozenset({"WATER"}),
    ),
    "heavy_lifter": AgentTypeSpec(
        name="heavy_lifter",
        movement_type="ground",
        speed=0.5,
        battery_capacity=400.0,
        battery_drain_rate=0.8,
        payload_capacity=500.0,
        sensor_range=2,
        can_rescue=False,
        can_clear_debris=True,
        can_transport=False,
        terrain_passable=frozenset({
            "EMPTY", "ROAD", "DEBRIS", "HOSPITAL", "CHARGING_STATION", "PARK",
        }),
    ),
}


# ---------------------------------------------------------------------------
# City Configuration
# ---------------------------------------------------------------------------

@dataclass
class HospitalConfig:
    """Config and baseline capacity for a hospital."""
    icu_beds: int = 10
    specialists: list[str] = field(default_factory=lambda: ["trauma", "burn"])
    blood_supply: float = 100.0


@dataclass
class CityConfig:
    """Configuration for procedural city generation.

    All parameters can be overridden via YAML config files.
    Sensible defaults target a 200×200 medium-scale city.
    """

    # Grid dimensions
    width: int = 200
    height: int = 200

    # Random seed for reproducibility
    seed: int = 42

    # Road network
    major_road_spacing: int = 25     # Distance between major roads
    minor_road_spacing: int = 8      # Distance between minor roads within blocks
    road_width: int = 1              # Road width in cells (1 for grid env)

    # Buildings
    building_density: float = 0.45   # Fraction of non-road cells filled with buildings
    min_building_size: int = 2       # Minimum building footprint side length
    max_building_size: int = 6       # Maximum building footprint side length

    # Disaster
    num_damaged_buildings: int = 30   # Number of buildings to mark as damaged
    debris_spread: int = 3            # How far debris spreads from damaged buildings
    num_fire_zones: int = 12          # Number of active fire zones
    fire_spread_radius: int = 2       # Initial fire radius around ignition point

    # Water
    num_rivers: int = 1               # Number of rivers through the city
    river_width: int = 3              # River width in cells
    flood_zone_count: int = 5         # Number of additional flood zones
    flood_zone_radius: int = 8        # Radius of each flood zone

    # Points of interest
    num_hospitals: int = 3
    num_charging_stations: int = 4
    num_parks: int = 6
    park_radius: int = 4

    # Victims
    num_victims: int = 50
    victim_severity_distribution: dict[str, float] = field(
        default_factory=lambda: {"critical": 0.2, "serious": 0.4, "stable": 0.4}
    )
    victim_time_limit: float = 500.0  # Steps before a stable victim degrades

    # Agent fleet (default: drone + ground robot only for Phase 1)
    fleet: dict[str, int] = field(
        default_factory=lambda: {"drone": 3, "ground_robot": 3}
    )

    # Communication
    communication_range: int = 20     # Max grid cells for agent-to-agent communication

    def validate(self) -> None:
        """Validate configuration parameters, raising ValueError on issues."""
        if self.width < 20 or self.height < 20:
            raise ValueError(f"City too small: {self.width}×{self.height}. Minimum is 20×20.")
        if self.width > 1000 or self.height > 1000:
            raise ValueError(f"City too large: {self.width}×{self.height}. Maximum is 1000×1000.")
        if self.building_density < 0 or self.building_density > 1:
            raise ValueError(f"building_density must be in [0, 1], got {self.building_density}")
        if self.num_victims < 0:
            raise ValueError(f"num_victims must be non-negative, got {self.num_victims}")
        for agent_type in self.fleet:
            if agent_type not in AGENT_TYPES:
                raise ValueError(
                    f"Unknown agent type '{agent_type}'. "
                    f"Valid types: {list(AGENT_TYPES.keys())}"
                )
        severity_sum = sum(self.victim_severity_distribution.values())
        if abs(severity_sum - 1.0) > 0.01:
            raise ValueError(
                f"victim_severity_distribution must sum to 1.0, got {severity_sum}"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> CityConfig:
        """Load configuration from a YAML file.

        Only keys matching CityConfig fields are loaded; unknown keys
        are silently ignored to allow forward-compatible config files.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        config = cls(**filtered)
        config.validate()
        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, CityConfig] = {
    "small": CityConfig(
        width=50, height=50,
        major_road_spacing=12, minor_road_spacing=4,
        num_damaged_buildings=8, num_fire_zones=4,
        num_rivers=0, flood_zone_count=1, flood_zone_radius=5,
        num_hospitals=1, num_charging_stations=2, num_parks=2,
        num_victims=10,
        fleet={"drone": 2, "ground_robot": 1},
    ),
    "medium": CityConfig(),  # 200×200 defaults
    "large": CityConfig(
        width=400, height=400,
        major_road_spacing=40, minor_road_spacing=12,
        building_density=0.40,
        num_damaged_buildings=80, debris_spread=4,
        num_fire_zones=30, fire_spread_radius=3,
        num_rivers=2, river_width=4, flood_zone_count=10, flood_zone_radius=12,
        num_hospitals=6, num_charging_stations=8, num_parks=12,
        num_victims=150,
        fleet={"drone": 8, "ground_robot": 5, "ambulance": 3, "boat": 2, "heavy_lifter": 2},
    ),
}


def get_config(preset_or_path: str) -> CityConfig:
    """Get a CityConfig from a preset name or YAML file path.

    Args:
        preset_or_path: One of "small", "medium", "large", or a path to a YAML file.

    Returns:
        A validated CityConfig instance.
    """
    if preset_or_path in PRESETS:
        config = PRESETS[preset_or_path]
        config.validate()
        return config

    path = Path(preset_or_path)
    if path.exists():
        return CityConfig.from_yaml(path)

    raise ValueError(
        f"'{preset_or_path}' is not a valid preset ({list(PRESETS.keys())}) "
        f"or file path."
    )
