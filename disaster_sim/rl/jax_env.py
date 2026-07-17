"""
JAX-Accelerated Disaster Simulation Environment.

Encodes the entire environment state as JAX pytrees for 100% GPU execution.
All operations (movement, collision, fog-of-war, observations, rewards,
disaster dynamics) are jit-compiled JAX functions that can be vmapped over
thousands of parallel environments.

This module replaces the CPU-bound env.py + observation.py + reward.py +
disaster_dynamics.py for RL training. The original CPU files remain for
interactive play and the backend API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from disaster_sim.digital_twin.city_config import (
    AGENT_TYPES,
    Severity,
    Terrain,
    get_config,
)
from disaster_sim.digital_twin.city_generator import CityGenerator


# ---------------------------------------------------------------------------
# Terrain Constants (compile-time, for use inside JIT)
# ---------------------------------------------------------------------------

_TERRAIN_EMPTY = int(Terrain.EMPTY.value)
_TERRAIN_ROAD = int(Terrain.ROAD.value)
_TERRAIN_BUILDING = int(Terrain.BUILDING.value)
_TERRAIN_BUILDING_DAMAGED = int(Terrain.BUILDING_DAMAGED.value)
_TERRAIN_DEBRIS = int(Terrain.DEBRIS.value)
_TERRAIN_WATER = int(Terrain.WATER.value)
_TERRAIN_FIRE = int(Terrain.FIRE.value)
_TERRAIN_HOSPITAL = int(Terrain.HOSPITAL.value)
_TERRAIN_CHARGING = int(Terrain.CHARGING_STATION.value)
_TERRAIN_PARK = int(Terrain.PARK.value)
_NUM_TERRAINS = len(Terrain)
_NUM_ACTIONS = 6  # STAY, UP, DOWN, LEFT, RIGHT, INTERACT


# ---------------------------------------------------------------------------
# Environment State  (JAX pytree — every leaf is a device array)
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    """Mutable environment state.  All leaves are JAX arrays on GPU."""

    grid: Any               # [H, W]  int32   — terrain grid (mutable: fire/flood)
    agent_row: Any           # []      int32
    agent_col: Any           # []      int32
    agent_prev_row: Any      # []      int32   — previous position (anti-oscillation)
    agent_prev_col: Any      # []      int32
    agent_battery: Any       # []      float32
    agent_alive: Any         # []      bool
    agent_carrying: Any      # []      int32   (-1 = not carrying)
    agent_cells_explored: Any  # []    int32
    explored: Any            # [H, W]  bool    — fog of war
    victim_row: Any          # [V]     int32
    victim_col: Any          # [V]     int32
    victim_alive: Any        # [V]     bool
    victim_rescued: Any      # [V]     bool
    victim_detected: Any     # [V]     bool
    victim_severity: Any     # [V]     int32   (0=critical 1=serious 2=stable)
    victim_mask: Any         # [V]     bool    (valid slots)
    hospital_row: Any        # [Nh]    int32
    hospital_col: Any        # [Nh]    int32
    n_hospitals: Any         # []      int32
    charging_row: Any        # [Nc]    int32
    charging_col: Any        # [Nc]    int32
    n_charging: Any          # []      int32
    timestep: Any            # []      int32
    total_collisions: Any    # []      int32
    total_rescued: Any       # []      int32


# ---------------------------------------------------------------------------
# Environment Parameters  (frozen Python dataclass — compile-time constant)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvParams:
    """Static environment configuration.

    Captured in closures and treated as compile-time constants by JAX JIT.
    Python primitives / tuples only (no JAX arrays) so that it hashes
    correctly for JIT caching.
    """

    # Grid
    height: int = 200
    width: int = 200
    fov_size: int = 21
    max_steps: int = 1000

    # Agent
    sensor_range: int = 8
    battery_capacity: float = 100.0
    battery_drain_rate: float = 0.5
    can_rescue: bool = False
    is_aerial: bool = True

    # Terrain passability lookup (indexed by Terrain int value)
    passable_terrain: tuple = (True,) * _NUM_TERRAINS  # drone default: all

    # Padded array limits
    max_victims: int = 50
    max_hospitals: int = 10
    max_charging: int = 10

    # Reward weights
    reward_explore: float = 0.02
    reward_rescue: float = 100.0
    penalty_collision: float = -0.2
    penalty_time: float = -0.002
    penalty_revisit: float = -0.003
    penalty_oscillation: float = -0.03

    # Severity → reward multiplier  (critical, serious, stable)
    severity_reward_mult: tuple = (3.0, 2.0, 1.0)
    # Severity → observation encoding value
    severity_obs_values: tuple = (1.0, 0.5, 0.2)

    num_terrains: int = _NUM_TERRAINS


# ---------------------------------------------------------------------------
# Pre-generated City Pool  (JAX pytree of arrays, lives on GPU)
# ---------------------------------------------------------------------------

class CityData(NamedTuple):
    """Pool of N pre-generated cities stored as padded JAX arrays."""

    grids: Any             # [N, H, W]  int32
    victim_row: Any        # [N, V]     int32
    victim_col: Any        # [N, V]     int32
    victim_severity: Any   # [N, V]     int32
    victim_mask: Any       # [N, V]     bool
    hospital_row: Any      # [N, Nh]    int32
    hospital_col: Any      # [N, Nh]    int32
    hospital_mask: Any     # [N, Nh]    bool
    charging_row: Any      # [N, Nc]    int32
    charging_col: Any      # [N, Nc]    int32
    charging_mask: Any     # [N, Nc]    bool
    agent_start_row: Any   # [N]        int32
    agent_start_col: Any   # [N]        int32


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_params_for_agent(
    preset: str,
    agent_type: str,
    fov_size: int = 21,
    max_steps: int = 1000,
) -> EnvParams:
    """Build an ``EnvParams`` tuned for *agent_type* on *preset*."""
    config = get_config(preset)
    spec = AGENT_TYPES[agent_type]

    passable = tuple(
        Terrain(t).name in spec.terrain_passable for t in range(_NUM_TERRAINS)
    )

    return EnvParams(
        height=config.height,
        width=config.width,
        fov_size=fov_size,
        max_steps=max_steps,
        sensor_range=spec.sensor_range,
        battery_capacity=spec.battery_capacity,
        battery_drain_rate=spec.battery_drain_rate,
        can_rescue=spec.can_rescue,
        is_aerial=(spec.movement_type == "aerial"),
        passable_terrain=passable,
        max_victims=max(config.num_victims, 1),
        max_hospitals=max(config.num_hospitals, 1),
        max_charging=max(config.num_charging_stations, 1),
        num_terrains=_NUM_TERRAINS,
    )


def generate_city_pool(
    preset: str,
    agent_type: str,
    n_cities: int,
    params: EnvParams,
    base_seed: int = 42,
) -> CityData:
    """Generate *n_cities* on CPU, pad, and transfer to GPU as a CityData."""

    severity_to_int = {
        Severity.CRITICAL: 0,
        Severity.SERIOUS: 1,
        Severity.STABLE: 2,
    }

    grids = []
    v_rows, v_cols, v_sevs, v_masks = [], [], [], []
    h_rows, h_cols, h_masks = [], [], []
    c_rows, c_cols, c_masks = [], [], []
    a_rows, a_cols = [], []

    for i in range(n_cities):
        config = get_config(preset)
        config.seed = base_seed + i
        city = CityGenerator(config).generate()

        grids.append(city.grid.astype(np.int32))

        # --- victims (pad to max_victims) ---
        vr = np.zeros(params.max_victims, dtype=np.int32)
        vc = np.zeros(params.max_victims, dtype=np.int32)
        vs = np.full(params.max_victims, 2, dtype=np.int32)   # default: stable
        vm = np.zeros(params.max_victims, dtype=bool)
        for j, v in enumerate(city.victims[: params.max_victims]):
            vr[j], vc[j] = v.row, v.col
            vs[j] = severity_to_int.get(v.severity, 2)
            vm[j] = True
        v_rows.append(vr); v_cols.append(vc)
        v_sevs.append(vs); v_masks.append(vm)

        # --- hospitals ---
        hr = np.zeros(params.max_hospitals, dtype=np.int32)
        hc = np.zeros(params.max_hospitals, dtype=np.int32)
        hm = np.zeros(params.max_hospitals, dtype=bool)
        for j, (r, c) in enumerate(city.hospitals[: params.max_hospitals]):
            hr[j], hc[j], hm[j] = r, c, True
        h_rows.append(hr); h_cols.append(hc); h_masks.append(hm)

        # --- charging stations ---
        cr = np.zeros(params.max_charging, dtype=np.int32)
        cc = np.zeros(params.max_charging, dtype=np.int32)
        cm = np.zeros(params.max_charging, dtype=bool)
        for j, (r, c) in enumerate(city.charging_stations[: params.max_charging]):
            cr[j], cc[j], cm[j] = r, c, True
        c_rows.append(cr); c_cols.append(cc); c_masks.append(cm)

        # --- agent spawn ---
        spawn = next(
            (s for s in city.agent_spawns if s.agent_type == agent_type),
            city.agent_spawns[0] if city.agent_spawns else None,
        )
        a_rows.append(spawn.row if spawn else 0)
        a_cols.append(spawn.col if spawn else 0)

    return CityData(
        grids=jnp.array(np.stack(grids)),
        victim_row=jnp.array(np.stack(v_rows)),
        victim_col=jnp.array(np.stack(v_cols)),
        victim_severity=jnp.array(np.stack(v_sevs)),
        victim_mask=jnp.array(np.stack(v_masks)),
        hospital_row=jnp.array(np.stack(h_rows)),
        hospital_col=jnp.array(np.stack(h_cols)),
        hospital_mask=jnp.array(np.stack(h_masks)),
        charging_row=jnp.array(np.stack(c_rows)),
        charging_col=jnp.array(np.stack(c_cols)),
        charging_mask=jnp.array(np.stack(c_masks)),
        agent_start_row=jnp.array(a_rows, dtype=jnp.int32),
        agent_start_col=jnp.array(a_cols, dtype=jnp.int32),
    )


# ===================================================================
#  Core environment functions  (all pure-JAX, JIT-compilable)
# ===================================================================

def reset(
    key: jax.Array,
    city_pool: CityData,
    params: EnvParams,
) -> tuple[EnvState, jax.Array]:
    """Sample a random city and return ``(state, obs)``."""

    key1, key2 = jax.random.split(key)
    n_cities = city_pool.grids.shape[0]
    idx = jax.random.randint(key1, (), 0, n_cities)

    grid = city_pool.grids[idx]
    a_row = city_pool.agent_start_row[idx]
    a_col = city_pool.agent_start_col[idx]

    explored = jnp.zeros((params.height, params.width), dtype=jnp.bool_)
    explored, cells_rev = _reveal_fog(explored, a_row, a_col, params)

    v_alive = city_pool.victim_mask[idx].copy()      # alive = valid initially
    v_rescued = jnp.zeros(params.max_victims, dtype=jnp.bool_)
    v_detected = _detect_victims(
        a_row, a_col,
        city_pool.victim_row[idx], city_pool.victim_col[idx],
        city_pool.victim_mask[idx], v_alive, v_rescued,
        params,
    )

    state = EnvState(
        grid=grid,
        agent_row=a_row,
        agent_col=a_col,
        agent_prev_row=a_row,
        agent_prev_col=a_col,
        agent_battery=jnp.float32(params.battery_capacity),
        agent_alive=jnp.bool_(True),
        agent_carrying=jnp.int32(-1),
        agent_cells_explored=cells_rev,
        explored=explored,
        victim_row=city_pool.victim_row[idx],
        victim_col=city_pool.victim_col[idx],
        victim_alive=v_alive,
        victim_rescued=v_rescued,
        victim_detected=v_detected,
        victim_severity=city_pool.victim_severity[idx],
        victim_mask=city_pool.victim_mask[idx],
        hospital_row=city_pool.hospital_row[idx],
        hospital_col=city_pool.hospital_col[idx],
        n_hospitals=jnp.sum(city_pool.hospital_mask[idx]).astype(jnp.int32),
        charging_row=city_pool.charging_row[idx],
        charging_col=city_pool.charging_col[idx],
        n_charging=jnp.sum(city_pool.charging_mask[idx]).astype(jnp.int32),
        timestep=jnp.int32(0),
        total_collisions=jnp.int32(0),
        total_rescued=jnp.int32(0),
    )

    obs = build_observation(state, params)
    return state, obs


def step(
    key: jax.Array,
    state: EnvState,
    action: jax.Array,
    params: EnvParams,
) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, dict]:
    """Advance one step.  Returns ``(state, obs, reward, done, info)``."""

    passable_lut = jnp.array(params.passable_terrain)

    # ---- battery drain ----
    new_battery = state.agent_battery - jnp.float32(params.battery_drain_rate)
    battery_dead = new_battery <= 0.0
    new_battery = jnp.maximum(new_battery, jnp.float32(0.0))
    agent_alive = state.agent_alive & ~battery_dead

    # ---- movement ----
    #  STAY=0  UP=1  DOWN=2  LEFT=3  RIGHT=4  INTERACT=5
    dr = jnp.array([0, -1, 1, 0, 0, 0], dtype=jnp.int32)[action]
    dc = jnp.array([0, 0, 0, -1, 1, 0], dtype=jnp.int32)[action]

    target_r = state.agent_row + dr
    target_c = state.agent_col + dc

    in_bounds = (
        (target_r >= 0) & (target_r < params.height)
        & (target_c >= 0) & (target_c < params.width)
    )
    safe_r = jnp.clip(target_r, 0, params.height - 1)
    safe_c = jnp.clip(target_c, 0, params.width - 1)

    terrain_at_target = state.grid[safe_r, safe_c]
    passable = passable_lut[terrain_at_target]

    is_move = (action >= 1) & (action <= 4)
    can_move = in_bounds & passable & agent_alive
    new_row = jnp.where(is_move & can_move, safe_r, state.agent_row)
    new_col = jnp.where(is_move & can_move, safe_c, state.agent_col)
    collision = is_move & ~can_move & agent_alive
    new_collisions = state.total_collisions + collision.astype(jnp.int32)

    # ---- interaction ----
    is_interact = (action == 5) & agent_alive
    terrain_here = state.grid[state.agent_row, state.agent_col]

    #  Charging station
    at_charger = terrain_here == _TERRAIN_CHARGING
    new_battery = jnp.where(
        is_interact & at_charger,
        jnp.float32(params.battery_capacity),
        new_battery,
    )

    #  Rescue — pick up nearest victim within Manhattan dist ≤ 1
    victim_dist = (
        jnp.abs(state.victim_row - state.agent_row)
        + jnp.abs(state.victim_col - state.agent_col)
    )
    rescuable = (
        state.victim_mask & state.victim_alive
        & ~state.victim_rescued & (victim_dist <= 1)
    )
    not_carrying = state.agent_carrying == -1
    can_pickup = (
        is_interact & jnp.bool_(params.can_rescue)
        & not_carrying & jnp.any(rescuable)
    )
    first_rescuable = jnp.argmax(rescuable.astype(jnp.int32))

    new_carrying = jnp.where(can_pickup, first_rescuable, state.agent_carrying)
    new_victim_rescued = jnp.where(
        can_pickup,
        state.victim_rescued.at[first_rescuable].set(True),
        state.victim_rescued,
    )
    new_total_rescued = state.total_rescued + can_pickup.astype(jnp.int32)

    #  Drop-off at hospital
    at_hospital = terrain_here == _TERRAIN_HOSPITAL
    can_dropoff = is_interact & (state.agent_carrying >= 0) & at_hospital
    new_carrying = jnp.where(can_dropoff, jnp.int32(-1), new_carrying)

    # ---- fog of war ----
    new_explored, cells_rev = _reveal_fog(
        state.explored, new_row, new_col, params,
    )
    new_cells_explored = state.agent_cells_explored + cells_rev

    # ---- victim detection ----
    new_detected = _detect_victims(
        new_row, new_col,
        state.victim_row, state.victim_col,
        state.victim_mask, state.victim_alive, new_victim_rescued,
        params,
    )
    new_detected = state.victim_detected | new_detected

    # ---- disaster dynamics (every 10 steps) ----
    new_timestep = state.timestep + 1
    do_dynamics = (new_timestep % 10) == 0
    key, dyn_key = jax.random.split(key)
    new_grid = jax.lax.cond(
        do_dynamics,
        lambda g, k: _step_dynamics(g, k),
        lambda g, k: g,
        state.grid, dyn_key,
    )

    # ---- reward ----
    valid_action = ~collision
    actually_moved = (new_row != state.agent_row) | (new_col != state.agent_col)
    went_back = (
        (new_row == state.agent_prev_row) & (new_col == state.agent_prev_col)
        & actually_moved
    )
    reward = _compute_reward(
        cells_rev, can_pickup, state.victim_severity,
        first_rescuable, valid_action, actually_moved, went_back, params,
    )

    # ---- termination ----
    victims_remaining = jnp.sum(
        state.victim_mask & state.victim_alive & ~new_victim_rescued,
    )
    done = ~agent_alive | (new_timestep >= params.max_steps) | (victims_remaining == 0)

    # ---- assemble new state ----
    new_state = EnvState(
        grid=new_grid,
        agent_row=new_row,
        agent_col=new_col,
        agent_prev_row=jnp.where(actually_moved, state.agent_row, state.agent_prev_row),
        agent_prev_col=jnp.where(actually_moved, state.agent_col, state.agent_prev_col),
        agent_battery=new_battery,
        agent_alive=agent_alive,
        agent_carrying=new_carrying,
        agent_cells_explored=new_cells_explored,
        explored=new_explored,
        victim_row=state.victim_row,
        victim_col=state.victim_col,
        victim_alive=state.victim_alive,
        victim_rescued=new_victim_rescued,
        victim_detected=new_detected,
        victim_severity=state.victim_severity,
        victim_mask=state.victim_mask,
        hospital_row=state.hospital_row,
        hospital_col=state.hospital_col,
        n_hospitals=state.n_hospitals,
        charging_row=state.charging_row,
        charging_col=state.charging_col,
        n_charging=state.n_charging,
        timestep=new_timestep,
        total_collisions=new_collisions,
        total_rescued=new_total_rescued,
    )

    obs = build_observation(new_state, params)

    info = {
        "coverage": jnp.mean(new_explored.astype(jnp.float32)),
        "victims_remaining": victims_remaining,
        "battery": new_battery / jnp.float32(params.battery_capacity),
        "collisions": new_collisions,
    }

    return new_state, obs, reward, done, info


# ===================================================================
#  Observation Builder
# ===================================================================

def build_observation(state: EnvState, params: EnvParams) -> jax.Array:
    """Build a ``[7, fov, fov]`` float32 observation tensor.

    Channels:
        0 — terrain (normalised)
        1 — fog-of-war (1 = explored)
        2 — self position (centre)
        3 — other agents (zeros for single-agent training)
        4 — known victims (severity-encoded)
        5 — target compass
        6 — previous position (symmetry breaking for collision loops)
    """
    fov = params.fov_size
    half = fov // 2

    # Pad for edge-safe slicing.  Out-of-bounds → -1 (distinct from passable buildings).
    padded_grid = jnp.pad(
        state.grid, half,
        mode="constant", constant_values=-1,
    )
    padded_explored = jnp.pad(
        state.explored.astype(jnp.int32), half,
        mode="constant", constant_values=1,
    )

    # dynamic_slice at (agent_row, agent_col) in padded coords gives
    # a window centred on the agent.
    terrain_win = jax.lax.dynamic_slice(
        padded_grid, (state.agent_row, state.agent_col), (fov, fov),
    )
    explored_win = jax.lax.dynamic_slice(
        padded_explored, (state.agent_row, state.agent_col), (fov, fov),
    )

    ch0 = terrain_win.astype(jnp.float32) / jnp.float32(params.num_terrains)
    ch1 = explored_win.astype(jnp.float32)
    ch2 = jnp.zeros((fov, fov), dtype=jnp.float32).at[half, half].set(1.0)
    ch3 = jnp.zeros((fov, fov), dtype=jnp.float32)          # other agents

    # ---- victims ----
    rel_r = state.victim_row - state.agent_row + half
    rel_c = state.victim_col - state.agent_col + half
    in_fov = (rel_r >= 0) & (rel_r < fov) & (rel_c >= 0) & (rel_c < fov)
    known = (
        state.victim_detected & state.victim_alive
        & ~state.victim_rescued & state.victim_mask
    )
    visible = in_fov & known
    sev_vals = jnp.array(params.severity_obs_values)
    vals = sev_vals[state.victim_severity] * visible.astype(jnp.float32)
    safe_vr = jnp.clip(rel_r, 0, fov - 1)
    safe_vc = jnp.clip(rel_c, 0, fov - 1)
    ch4 = jnp.zeros((fov, fov), dtype=jnp.float32).at[safe_vr, safe_vc].add(vals)

    # ---- compass ----
    ch5 = _build_compass(state, params, half, fov)

    # ---- previous position (memory) ----
    rel_prev_r = state.agent_prev_row - state.agent_row + half
    rel_prev_c = state.agent_prev_col - state.agent_col + half
    safe_prev_r = jnp.clip(rel_prev_r, 0, fov - 1)
    safe_prev_c = jnp.clip(rel_prev_c, 0, fov - 1)
    ch6 = jnp.zeros((fov, fov), dtype=jnp.float32).at[safe_prev_r, safe_prev_c].set(1.0)

    return jnp.stack([ch0, ch1, ch2, ch3, ch4, ch5, ch6])


def _build_compass(
    state: EnvState, params: EnvParams, half: int, fov: int,
) -> jax.Array:
    """Channel 5: directional gradient pointing toward the best target.

    Instead of a single pixel (easily lost by convolutions), this fills
    the entire FOV with a gradient whose sign and magnitude indicate the
    direction and rough distance to the target.  The CNN can learn from
    this large-scale signal far more easily.
    """

    a_r, a_c = state.agent_row, state.agent_col

    # Nearest unrescued detected victim
    v_dist = jnp.abs(state.victim_row - a_r) + jnp.abs(state.victim_col - a_c)
    valid_v = (
        state.victim_detected & state.victim_alive
        & ~state.victim_rescued & state.victim_mask
    )
    v_dist = jnp.where(valid_v, v_dist, jnp.int32(99999))
    nv = jnp.argmin(v_dist)
    has_victim = jnp.any(valid_v)

    # Nearest hospital
    h_dist = jnp.abs(state.hospital_row - a_r) + jnp.abs(state.hospital_col - a_c)
    valid_h = jnp.arange(params.max_hospitals) < state.n_hospitals
    h_dist = jnp.where(valid_h, h_dist, jnp.int32(99999))
    nh = jnp.argmin(h_dist)
    has_hosp = state.n_hospitals > 0

    carrying = state.agent_carrying >= 0
    # If drone, target unexplored. If rescuer, target hosp/victim.
    if not params.can_rescue:
        # DRONE MODE: Nearest unexplored cell
        rs = jnp.arange(params.height)[:, None]
        cs = jnp.arange(params.width)[None, :]
        u_dist = jnp.abs(rs - a_r) + jnp.abs(cs - a_c)
        
        # Tie-breaker: prefer cells further from the center (outward exploration)
        # Tie-breaker: strongly prefer cells in the direction of current momentum
        # to prevent arbitrary turning and tie-breaking biases.
        mom_r = state.agent_row - state.agent_prev_row
        mom_c = state.agent_col - state.agent_prev_col
        dot = (rs - a_r) * mom_r + (cs - a_c) * mom_c
        
        center_r = params.height / 2.0
        center_c = params.width / 2.0
        dist_to_center = jnp.abs(rs - center_r) + jnp.abs(cs - center_c)
        
        u_dist_float = u_dist.astype(jnp.float32) - dist_to_center * 0.0001 - dot.astype(jnp.float32) * 0.1
        
        u_dist_float = jnp.where(~state.explored, u_dist_float, jnp.float32(99999.0))
        u_flat = jnp.argmin(u_dist_float)
        target_r = u_flat // params.width
        target_c = u_flat % params.width
        has_target = jnp.any(~state.explored)
    else:
        # RESCUE MODE
        target_r = jnp.where(
            carrying & has_hosp, state.hospital_row[nh],
            jnp.where(has_victim, state.victim_row[nv], a_r),
        )
        target_c = jnp.where(
            carrying & has_hosp, state.hospital_col[nh],
            jnp.where(has_victim, state.victim_col[nv], a_c),
        )
        has_target = (carrying & has_hosp) | has_victim

    # ---- Directional gradient compass ----
    # Compute normalised direction to target
    dy = (target_r - a_r).astype(jnp.float32)
    dx = (target_c - a_c).astype(jnp.float32)
    dist = jnp.maximum(jnp.abs(dy) + jnp.abs(dx), jnp.float32(1.0))
    dir_r = dy / dist
    dir_c = dx / dist

    # Build gradient: each pixel's value = dot(direction, offset_from_centre)
    rows = jnp.arange(fov, dtype=jnp.float32) - float(half)
    cols = jnp.arange(fov, dtype=jnp.float32) - float(half)
    grid_r = rows[:, None]  # [fov, 1]
    grid_c = cols[None, :]  # [1, fov]
    gradient = (dir_r * grid_r + dir_c * grid_c) / float(half)
    gradient = jnp.clip(gradient, -1.0, 1.0)

    blank = jnp.zeros((fov, fov), dtype=jnp.float32)
    return jnp.where(has_target, gradient, blank)


# ===================================================================
#  Internal helpers
# ===================================================================

def _reveal_fog(
    explored: jax.Array,
    agent_row: jax.Array,
    agent_col: jax.Array,
    params: EnvParams,
) -> tuple[jax.Array, jax.Array]:
    """Reveal cells within the sensor circle.  Returns (explored, n_new)."""

    sr = params.sensor_range
    offsets = jnp.arange(-sr, sr + 1)
    dr, dc = jnp.meshgrid(offsets, offsets, indexing="ij")
    dist = jnp.sqrt(dr.astype(jnp.float32) ** 2 + dc.astype(jnp.float32) ** 2)
    circle = dist <= sr

    act_r = agent_row + dr
    act_c = agent_col + dc
    valid = (
        (act_r >= 0) & (act_r < params.height)
        & (act_c >= 0) & (act_c < params.width)
        & circle
    )
    safe_r = jnp.clip(act_r, 0, params.height - 1)
    safe_c = jnp.clip(act_c, 0, params.width - 1)

    was = explored[safe_r, safe_c]
    n_new = jnp.sum(valid & ~was).astype(jnp.int32)

    new_explored = explored.at[safe_r, safe_c].max(valid)
    return new_explored, n_new


def _detect_victims(
    a_row, a_col,
    v_row, v_col, v_mask, v_alive, v_rescued,
    params: EnvParams,
) -> jax.Array:
    """Return bool[V] of victims inside the sensor radius."""

    dist = jnp.sqrt(
        (v_row - a_row).astype(jnp.float32) ** 2
        + (v_col - a_col).astype(jnp.float32) ** 2
    )
    return (dist <= params.sensor_range) & v_mask & v_alive & ~v_rescued


def _compute_reward(
    cells_revealed, rescued_this_step,
    victim_severity, rescued_idx,
    valid_action, actually_moved, went_back, params: EnvParams,
) -> jax.Array:
    """Compute scalar step reward.

    Reward table (per step):
        Exploring (cells > 0):       +cells * 0.02 - 0.002  ≈ +0.2
        Traversing explored area:    -0.002 - 0.003         = -0.005
        Oscillating (went back):     -0.002 - 0.003 - 0.03  = -0.035
        Staying still:               -0.002 - 0.003         = -0.005
        Collision:                   -0.002 - 5.0           = -5.002
    """

    reward = jnp.float32(params.penalty_time)
    reward = jnp.where(
        ~valid_action,
        reward + jnp.float32(params.penalty_collision),
        reward,
    )
    # Exploration reward — per newly revealed cell
    reward = reward + cells_revealed.astype(jnp.float32) * jnp.float32(
        params.reward_explore,
    )

    # Revisit penalty — any step without new exploration costs a little extra
    unproductive = cells_revealed == 0
    reward = jnp.where(
        unproductive,
        reward + jnp.float32(params.penalty_revisit),
        reward,
    )

    # Oscillation penalty — going back to previous position is heavily punished
    reward = jnp.where(
        went_back,
        reward + jnp.float32(params.penalty_oscillation),
        reward,
    )

    sev_mult = jnp.array(params.severity_reward_mult)
    rescue_rew = jnp.float32(params.reward_rescue) * sev_mult[victim_severity[rescued_idx]]
    reward = jnp.where(rescued_this_step, reward + rescue_rew, reward)

    return reward


# ===================================================================
#  Disaster dynamics  (fire + flood cellular automata)
# ===================================================================

def _convolve2d_same(inp: jax.Array, kernel: jax.Array) -> jax.Array:
    """2-D convolution with 'same' padding using ``jax.lax.conv``."""
    inp4 = inp[None, None, :, :].astype(jnp.float32)
    # Flip kernel for proper convolution (not correlation)
    ker4 = kernel[None, None, ::-1, ::-1].astype(jnp.float32)
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    out = jax.lax.conv_general_dilated(
        inp4, ker4,
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
    )
    return out[0, 0]


def _step_dynamics(grid: jax.Array, key: jax.Array) -> jax.Array:
    key1, key2 = jax.random.split(key)
    grid = _step_fire(grid, key1)
    grid = _step_flood(grid, key2)
    return grid


def _step_fire(grid: jax.Array, key: jax.Array) -> jax.Array:
    """Fire-spread cellular automaton step (wind-biased)."""

    burning = (grid == _TERRAIN_FIRE).astype(jnp.float32)
    combustible = (
        (grid == _TERRAIN_BUILDING)
        | (grid == _TERRAIN_BUILDING_DAMAGED)
        | (grid == _TERRAIN_DEBRIS)
        | (grid == _TERRAIN_PARK)
    )

    # Wind parameters (compile-time constants)
    wx, wy = 0.5, -0.2
    base = jnp.array([
        [0.1, 0.2, 0.1],
        [0.2, 0.0, 0.2],
        [0.1, 0.2, 0.1],
    ])
    wind = jnp.array([
        [max(0, -wx) * max(0, -wy), max(0, -wy), max(0, wx) * max(0, -wy)],
        [max(0, -wx),               0.0,          max(0, wx)],
        [max(0, -wx) * max(0, wy),  max(0, wy),   max(0, wx) * max(0, wy)],
    ])
    kernel = base + wind * 0.5

    pressure = _convolve2d_same(burning, kernel)

    k1, k2 = jax.random.split(key)
    rolls = jax.random.uniform(k1, grid.shape)
    new_fire = combustible & (rolls < pressure)
    grid = jnp.where(new_fire, _TERRAIN_FIRE, grid)

    # Burn-out (5 % per tick → debris)
    burn_rolls = jax.random.uniform(k2, grid.shape)
    burn_out = (grid == _TERRAIN_FIRE) & (burn_rolls < 0.05)
    grid = jnp.where(burn_out, _TERRAIN_DEBRIS, grid)

    return grid


def _step_flood(grid: jax.Array, key: jax.Array) -> jax.Array:
    """Flood-spread cellular automaton step."""

    water = (grid == _TERRAIN_WATER).astype(jnp.float32)
    permeable = (
        (grid == _TERRAIN_EMPTY) | (grid == _TERRAIN_ROAD)
        | (grid == _TERRAIN_PARK) | (grid == _TERRAIN_DEBRIS)
    )
    kernel = jnp.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    neighbours = _convolve2d_same(water, kernel)

    rolls = jax.random.uniform(key, grid.shape)
    new_water = permeable & (neighbours > 0) & (rolls < 0.2)
    grid = jnp.where(new_water, _TERRAIN_WATER, grid)

    return grid
