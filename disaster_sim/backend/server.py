"""
FastAPI Backend Server

Exposes the digital twin via WebSocket for real-time visualization
and REST API for disaster injection.

Drones run the trained JAX PPO policy.
Ambulances use A* pathfinding to rescue and transport victims.
"""

import asyncio
import json
import os
import pickle
import random
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import numpy as np

from disaster_sim.digital_twin.city_config import get_config, Terrain, AGENT_TYPES
from disaster_sim.digital_twin.city_generator import CityGenerator
from disaster_sim.digital_twin.world_state import WorldState
from disaster_sim.engine.physics import PhysicsEngine, Action
from disaster_sim.digital_twin.disaster_dynamics import DisasterDynamics
from disaster_sim.digital_twin.disaster_injector import DisasterInjector
from disaster_sim.predictive.cloud_coordinator import CloudCoordinator
from disaster_sim.routing.astar import astar_path


# ---------------------------------------------------------------------------
#  JAX drone inference (lazy-loaded so server starts even without JAX)
# ---------------------------------------------------------------------------

_jax_ready = False
_jax_network = None
_jax_params = None
_jax_get_action = None
_jax_build_obs = None
_jax_env_params = None
_jax_build_state = None
_jax_rng = None

def _init_jax_drone():
    """Load JAX model + JIT-compile inference function (called once)."""
    global _jax_ready, _jax_network, _jax_params, _jax_get_action
    global _jax_build_obs, _jax_env_params, _jax_build_state, _jax_rng

    model_path = os.path.join("models", "drone_jax_ppo.pkl")
    if not os.path.exists(model_path):
        print(f"[server] No JAX drone model at {model_path} — drones will random-walk.")
        return

    try:
        import jax
        import jax.numpy as jnp
        from disaster_sim.rl.jax_networks import ActorCritic
        from disaster_sim.rl.jax_env import build_observation, make_params_for_agent
        from disaster_sim.rl.test_jax import _build_jax_state_from_world

        with open(model_path, "rb") as f:
            _jax_params = pickle.load(f)

        _jax_network = ActorCritic(action_dim=6, features_dim=128)
        _jax_env_params = make_params_for_agent("medium", "drone")
        _jax_build_obs = build_observation
        _jax_build_state = _build_jax_state_from_world

        _jax_rng = jax.random.PRNGKey(42)

        @jax.jit
        def _get_action(params, obs_batch, rng_key):
            pi, value = _jax_network.apply(params, obs_batch)
            action = pi.sample(seed=rng_key)
            return action[0], value[0]

        _jax_get_action = _get_action
        _jax_ready = True
        print("[server] JAX drone model loaded and JIT-compiled.")
    except Exception as e:
        print(f"[server] Failed to load JAX drone: {e}")


# ---------------------------------------------------------------------------
#  Ambulance AI — A* dispatch
# ---------------------------------------------------------------------------

def _ambulance_action(agent, world: WorldState) -> int:
    """Pick the next action for an ambulance using A* pathfinding.

    Priority:
    1. If carrying victim → navigate to nearest hospital, interact when there.
    2. If not carrying → navigate to nearest known victim, interact when adjacent.
    3. No known targets → random walk on roads.
    """
    r, c = agent.grid_pos

    # --- Already has a path? Follow it. ---
    if agent.path:
        next_r, next_c = agent.path[0]
        dr, dc = next_r - r, next_c - c

        # If we've arrived at this waypoint, pop it
        if dr == 0 and dc == 0:
            agent.path.pop(0)
            if agent.path:
                next_r, next_c = agent.path[0]
                dr, dc = next_r - r, next_c - c
            else:
                dr, dc = 0, 0

        # Convert delta to action
        if dr == -1 and dc == 0:
            return Action.UP
        elif dr == 1 and dc == 0:
            return Action.DOWN
        elif dr == 0 and dc == -1:
            return Action.LEFT
        elif dr == 0 and dc == 1:
            return Action.RIGHT

    # --- Carrying victim → go to hospital ---
    if agent.carrying_victim:
        terrain = world.terrain_at(r, c)
        if terrain == Terrain.HOSPITAL:
            agent.path = []
            return Action.INTERACT

        hospital_pos = world.get_nearest_hospital(r, c)
        if hospital_pos:
            path = astar_path(world, (r, c), hospital_pos, agent.agent_type)
            if path:
                agent.path = path
                if agent.path:
                    next_r, next_c = agent.path[0]
                    dr, dc = next_r - r, next_c - c
                    if dr == -1:
                        return Action.UP
                    elif dr == 1:
                        return Action.DOWN
                    elif dc == -1:
                        return Action.LEFT
                    elif dc == 1:
                        return Action.RIGHT
        return Action.STAY

    # --- Not carrying → find nearest known victim ---
    # Check if we're close enough to rescue a victim (using ambulance sensor range = 3)
    rescue_radius = 3
    nearby = world.get_victims_in_range(r, c, radius=rescue_radius)
    if nearby:
        for v in nearby:
            if not v.rescued and v.detected:
                agent.path = []
                return Action.INTERACT

    # Find all potential targets from known victims
    valid_victims = []
    for vid in agent.local_victims_known:
        v = world.victims.get(vid)
        if v and v.is_alive and not v.rescued and (v.assigned_agent is None or v.assigned_agent == agent.id):
            dist = abs(v.row - r) + abs(v.col - c)
            valid_victims.append((dist, v))
            
    # Sort by distance
    valid_victims.sort(key=lambda x: x[0])

    for dist, best_victim in valid_victims:
        target = (best_victim.row, best_victim.col)

        # A* to victim — try cells within rescue_radius, sorted by distance to target
        path = None
        candidates = []
        for dr in range(-rescue_radius, rescue_radius + 1):
            for dc in range(-rescue_radius, rescue_radius + 1):
                if abs(dr) + abs(dc) <= rescue_radius:
                    adj = (target[0] + dr, target[1] + dc)
                    if world.is_passable(adj[0], adj[1], agent.agent_type):
                        candidates.append(adj)
        
        # Sort candidates by distance to the ambulance's current position to minimize travel
        candidates.sort(key=lambda pos: abs(pos[0] - r) + abs(pos[1] - c))
        
        for adj in candidates:
            p = astar_path(world, (r, c), adj, agent.agent_type)
            # If path is [] (already there) or has elements, it's valid
            if p is not None:
                path = p
                break

        if path is not None:
            # We found a reachable victim!
            if agent.assigned_victim and agent.assigned_victim != best_victim.id:
                old_v = world.victims.get(agent.assigned_victim)
                if old_v and old_v.assigned_agent == agent.id:
                    old_v.assigned_agent = None

            # Assign self so other ambulances don't compete
            best_victim.assigned_agent = agent.id
            agent.assigned_victim = best_victim.id
            agent.path = path
            
            if agent.path: # path length > 0
                next_r, next_c = agent.path[0]
                ddr, ddc = next_r - r, next_c - c
                if ddr == -1:
                    return Action.UP
                elif ddr == 1:
                    return Action.DOWN
                elif ddc == -1:
                    return Action.LEFT
                elif ddc == 1:
                    return Action.RIGHT
            else:
                # Path is [], meaning we are already at a valid cell! 
                return Action.INTERACT

    # --- No target — stay still instead of random vibrating ---
    return Action.STAY


# ---------------------------------------------------------------------------
#  FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="Disaster Sim Digital Twin")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimRunner:
    """Manages the simulation loop and state."""

    def __init__(self):
        import time
        self.config = get_config("medium")
        self.config.seed = int(time.time() * 1000) % 1000000
        gen = CityGenerator(self.config)
        self.city = gen.generate()

        self.world = WorldState(self.city)
        self.physics = PhysicsEngine(self.world)
        self.dynamics = DisasterDynamics(self.world)
        self.injector = DisasterInjector(self.world)
        self.coordinator = CloudCoordinator(self.world)

        self.running = False
        self.clients: list[WebSocket] = []
        self.tick_interval = 0.05  # 20 Hz default

        # Drone prev-position tracking for JAX obs builder
        self._drone_prev: dict[str, tuple[int, int]] = {}
        for aid, a in self.world.agents.items():
            if a.agent_type == "drone":
                self._drone_prev[aid] = (int(a.row), int(a.col))

        # Initialize JAX drone model
        _init_jax_drone()
        
        self.rng_key = None
        if _jax_ready:
            import jax
            self.rng_key = jax.random.PRNGKey(self.config.seed)

    def _get_drone_action(self, agent) -> int:
        """Get action from JAX model or fall back to random."""
        if not _jax_ready:
            return random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])

        import jax
        import jax.numpy as jnp
        import random

        if agent.id not in self._drone_prev:
            # Fake initial momentum to break symmetry
            self._drone_prev[agent.id] = (int(agent.row) + random.choice([-1, 1]), int(agent.col) + random.choice([-1, 1]))
        prev = self._drone_prev[agent.id]
        jax_state = _jax_build_state(
            self.world, agent.id, _jax_env_params,
            prev_row=prev[0], prev_col=prev[1],
        )
        obs = _jax_build_obs(jax_state, _jax_env_params)
        obs_batched = jnp.expand_dims(obs, axis=0)
        global _jax_rng
        _jax_rng, act_key = jax.random.split(_jax_rng)
        action, _ = _jax_get_action(_jax_params, obs_batched, act_key)
        return int(action)

    async def run_loop(self):
        """Main simulation loop."""
        self.running = True
        while self.running:
            # 1. Agent actions
            for agent in self.world.active_agents:
                if agent.agent_type == "drone":
                    action = self._get_drone_action(agent)
                    old_r, old_c = int(agent.row), int(agent.col)
                    self.physics.step(agent.id, action)
                    # Update prev position only if actually moved (matches training behavior)
                    if int(agent.row) != old_r or int(agent.col) != old_c:
                        self._drone_prev[agent.id] = (old_r, old_c)

                elif agent.agent_type == "ambulance":
                    action = _ambulance_action(agent, self.world)
                    success = self.physics.step(agent.id, action)
                    if not success and action != Action.STAY and hasattr(agent, 'path'):
                        agent.path = []

                elif agent.agent_type == "traffic_light":
                    pass  # Handled by coordinator

                else:
                    # Ground robots etc — random walk
                    action = random.choice([
                        Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY
                    ])
                    self.physics.step(agent.id, action)

            # 2. Sync communications (mesh network)
            self.physics.sync_communications()

            # 3. Advance world physics
            self.world.timestep += 1
            if self.world.timestep % 10 == 0:
                self.dynamics.step()

            # 4. Cloud coordinator
            self.coordinator.tick()

            # 5. Broadcast state
            await self.broadcast_state()

            await asyncio.sleep(self.tick_interval)

    async def broadcast_state(self):
        """Serialize and send state to all connected WebSockets."""
        if not self.clients:
            return

        state = self.world.to_dict()
        state["grid"] = self.world.grid.tolist()

        message = json.dumps(state)
        dead = []
        for client in self.clients:
            try:
                await client.send_text(message)
            except Exception:
                dead.append(client)
        for c in dead:
            self.clients.remove(c)


runner = SimRunner()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(runner.run_loop())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    runner.clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle speed control messages
            try:
                msg = json.loads(data)
                if msg.get("type") == "set_speed":
                    speed = float(msg.get("value", 20))
                    runner.tick_interval = max(0.01, 1.0 / speed)
            except (json.JSONDecodeError, ValueError):
                pass
    except WebSocketDisconnect:
        runner.clients.remove(websocket)


# -- REST API for Disaster Injection --

class EarthquakeRequest(BaseModel):
    r: int
    c: int
    magnitude: float

class FloodRequest(BaseModel):
    r: int
    c: int
    volume: int

@app.post("/api/disaster/earthquake")
async def trigger_earthquake(req: EarthquakeRequest):
    result = runner.injector.trigger_earthquake(req.r, req.c, req.magnitude)
    return {"status": "success", "data": result}

@app.post("/api/disaster/flood")
async def trigger_flood(req: FloodRequest):
    result = runner.injector.trigger_flood(req.r, req.c, req.volume)
    return {"status": "success", "data": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
