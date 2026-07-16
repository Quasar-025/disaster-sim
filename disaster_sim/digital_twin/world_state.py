"""
World State — Mutable state tracking for the disaster simulation.

Manages the runtime state of the simulation including agent positions,
victim statuses, hazard progression, and discovered map information.
This is the single source of truth that the RL environment, renderer,
and backend API all read from and write to.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from disaster_sim.digital_twin.city_config import (
    AGENT_TYPES,
    AgentTypeSpec,
    Severity,
    Terrain,
    HospitalConfig,
)
from disaster_sim.digital_twin.city_generator import AgentSpawn, GeneratedCity, VictimInfo


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Runtime state of a single agent."""

    id: str
    agent_type: str
    row: float
    col: float
    battery: float          # 0.0 – max capacity
    status: str             # "idle", "searching", "rescuing", "returning", "charging", "dead"
    assigned_victim: Optional[str] = None
    assigned_target_pos: Optional[tuple[int, int]] = None
    carrying_victim: Optional[str] = None
    path: list[tuple[int, int]] = field(default_factory=list)
    cells_explored: int = 0
    local_explored: Optional[np.ndarray] = None
    local_victims_known: set[str] = field(default_factory=set)

    @property
    def spec(self) -> AgentTypeSpec:
        return AGENT_TYPES[self.agent_type]

    @property
    def battery_fraction(self) -> float:
        return self.battery / self.spec.battery_capacity

    @property
    def is_alive(self) -> bool:
        return self.status != "dead"

    @property
    def grid_pos(self) -> tuple[int, int]:
        return (int(round(self.row)), int(round(self.col)))


# ---------------------------------------------------------------------------
# Victim State
# ---------------------------------------------------------------------------

@dataclass
class VictimState:
    """Runtime state of a single victim."""

    id: str
    row: int
    col: int
    severity: Severity
    time_remaining: float
    detected: bool = False
    rescued: bool = False
    transported: bool = False  # Delivered to hospital
    assigned_agent: Optional[str] = None

    @property
    def is_alive(self) -> bool:
        return self.time_remaining > 0 and not self.rescued


# ---------------------------------------------------------------------------
# Hospital & Sensor State
# ---------------------------------------------------------------------------

@dataclass
class HospitalState:
    """Runtime state of a hospital."""
    id: str
    row: int
    col: int
    config: HospitalConfig
    current_patients: int = 0
    available_blood: float = 100.0
    
    @property
    def available_icu_beds(self) -> int:
        return max(0, self.config.icu_beds - self.current_patients)


@dataclass
class SensorState:
    """Simulated IoT sensor telemetry."""
    id: str
    row: int
    col: int
    water_level: float = 0.0
    smoke_density: float = 0.0
    temperature: float = 22.0
    air_quality_index: int = 50


# ---------------------------------------------------------------------------
# World State
# ---------------------------------------------------------------------------

class WorldState:
    """Mutable state of the simulation world.

    Initialized from a GeneratedCity and updated each simulation step.
    """

    def __init__(self, city: GeneratedCity):
        self.city = city
        self.grid = city.grid.copy()  # Mutable copy — terrain can change (fire spread, debris clear)
        self.width = city.width
        self.height = city.height
        self.timestep = 0

        # Initialize agents from spawn points
        self.agents: dict[str, AgentState] = {}
        for spawn in city.agent_spawns:
            spec = AGENT_TYPES[spawn.agent_type]
            self.agents[spawn.id] = AgentState(
                id=spawn.id,
                agent_type=spawn.agent_type,
                row=float(spawn.row),
                col=float(spawn.col),
                battery=spec.battery_capacity,
                status="idle",
            )
            self.agents[spawn.id].local_explored = np.zeros((self.height, self.width), dtype=bool)

        # Initialize victims
        self.victims: dict[str, VictimState] = {}
        for v in city.victims:
            self.victims[v.id] = VictimState(
                id=v.id,
                row=v.row,
                col=v.col,
                severity=v.severity,
                time_remaining=v.time_remaining,
            )

        # Initialize hospitals
        self.hospitals: dict[str, HospitalState] = {}
        h_config = HospitalConfig() # Baseline config for now
        for i, (hr, hc) in enumerate(city.hospitals):
            hid = f"hospital_{i:02d}"
            self.hospitals[hid] = HospitalState(
                id=hid, row=hr, col=hc, config=h_config, available_blood=h_config.blood_supply
            )
            
        # Initialize environmental sensors (grid of sensors every 20 cells)
        self.sensors: dict[str, SensorState] = {}
        sensor_idx = 0
        for r in range(10, self.height, 20):
            for c in range(10, self.width, 20):
                sid = f"sensor_{sensor_idx:03d}"
                self.sensors[sid] = SensorState(id=sid, row=r, col=c)
                sensor_idx += 1

        # Fog of war — tracks which cells have been explored
        self.explored = np.zeros((self.height, self.width), dtype=bool)
        # Initially reveal cells around agent spawn points
        for agent in self.agents.values():
            self._reveal_around(agent)

        # Metrics accumulators
        self.total_collisions = 0
        self.total_energy_consumed = 0.0
        self.total_victims_rescued = 0
        self.total_victims_detected = 0
        self.total_victims_transported = 0

    def _reveal_around(self, agent: AgentState) -> int:
        """Reveal cells within an agent's sensor range. Returns count of newly revealed."""
        spec = agent.spec
        r, c = agent.grid_pos
        newly_revealed = 0
        for dr in range(-spec.sensor_range, spec.sensor_range + 1):
            for dc in range(-spec.sensor_range, spec.sensor_range + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    dist = (dr * dr + dc * dc) ** 0.5
                    if dist <= spec.sensor_range:
                        if agent.local_explored is not None and not agent.local_explored[nr, nc]:
                            agent.local_explored[nr, nc] = True
                            newly_revealed += 1
                        if not self.explored[nr, nc]:
                            self.explored[nr, nc] = True
                            
                        # If a victim is here, add to local known
                        for victim in self.victims.values():
                            if victim.row == nr and victim.col == nc and victim.is_alive and not victim.rescued:
                                agent.local_victims_known.add(victim.id)
                                victim.detected = True

        return newly_revealed

    # -- Queries --

    @property
    def coverage(self) -> float:
        """Fraction of the city that has been explored."""
        return float(np.mean(self.explored))

    @property
    def victims_remaining(self) -> int:
        """Number of victims not yet rescued."""
        return sum(1 for v in self.victims.values() if v.is_alive and not v.rescued)

    @property
    def victims_rescued_count(self) -> int:
        return sum(1 for v in self.victims.values() if v.rescued)

    @property
    def victims_detected_count(self) -> int:
        return sum(1 for v in self.victims.values() if v.detected)

    @property
    def active_agents(self) -> list[AgentState]:
        """Agents that are alive and not dead."""
        return [a for a in self.agents.values() if a.is_alive]

    @property
    def avg_battery(self) -> float:
        """Average battery fraction across active agents."""
        active = self.active_agents
        if not active:
            return 0.0
        return sum(a.battery_fraction for a in active) / len(active)

    def terrain_at(self, row: int, col: int) -> Terrain:
        """Get terrain at position, handling bounds."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return Terrain(self.grid[row, col])
        return Terrain.BUILDING  # Out of bounds = impassable

    def is_passable(self, row: int, col: int, agent_type: str) -> bool:
        """Check if a cell is passable for a given agent type."""
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        terrain = self.terrain_at(row, col)
        spec = AGENT_TYPES[agent_type]
        return terrain.name in spec.terrain_passable

    def get_victims_in_range(self, row: int, col: int, radius: int) -> list[VictimState]:
        """Get unrescued victims within a given radius of a position."""
        result = []
        for v in self.victims.values():
            if v.is_alive and not v.rescued:
                dist = abs(v.row - row) + abs(v.col - col)
                if dist <= radius:
                    result.append(v)
        return result

    def get_nearest_hospital(self, row: int, col: int) -> Optional[tuple[int, int]]:
        """Get the nearest hospital position."""
        if not self.city.hospitals:
            return None
        best = None
        best_dist = float("inf")
        for hr, hc in self.city.hospitals:
            dist = abs(hr - row) + abs(hc - col)
            if dist < best_dist:
                best_dist = dist
                best = (hr, hc)
        return best

    def get_nearest_charging_station(self, row: int, col: int) -> Optional[tuple[int, int]]:
        """Get the nearest charging station position."""
        if not self.city.charging_stations:
            return None
        best = None
        best_dist = float("inf")
        for sr, sc in self.city.charging_stations:
            dist = abs(sr - row) + abs(sc - col)
            if dist < best_dist:
                best_dist = dist
                best = (sr, sc)
        return best

    # -- Serialization --

    def to_dict(self) -> dict:
        """Serialize world state for WebSocket/API transmission."""
        return {
            "timestep": self.timestep,
            "coverage": round(self.coverage, 4),
            "victims_remaining": self.victims_remaining,
            "victims_rescued": self.victims_rescued_count,
            "victims_detected": self.victims_detected_count,
            "total_collisions": self.total_collisions,
            "total_energy": round(self.total_energy_consumed, 2),
            "avg_battery": round(self.avg_battery, 4),
            "agents": {
                aid: {
                    "type": a.agent_type,
                    "row": round(a.row, 2),
                    "col": round(a.col, 2),
                    "battery": round(a.battery_fraction, 4),
                    "status": a.status,
                    "assigned_victim": a.assigned_victim,
                }
                for aid, a in self.agents.items()
            },
            "victims": {
                vid: {
                    "row": v.row,
                    "col": v.col,
                    "severity": v.severity.value,
                    "detected": v.detected,
                    "rescued": v.rescued,
                    "time_remaining": round(v.time_remaining, 1),
                }
                for vid, v in self.victims.items()
            },
            "hospitals": {
                hid: {
                    "row": h.row,
                    "col": h.col,
                    "available_icu_beds": h.available_icu_beds,
                    "available_blood": round(h.available_blood, 1),
                }
                for hid, h in self.hospitals.items()
            },
            "sensors": {
                sid: {
                    "row": s.row,
                    "col": s.col,
                    "water_level": round(s.water_level, 2),
                    "smoke_density": round(s.smoke_density, 2),
                    "temperature": round(s.temperature, 1),
                    "aqi": s.air_quality_index,
                }
                for sid, s in self.sensors.items()
            }
        }
