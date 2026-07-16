"""
FastAPI Backend Server

Exposes the digital twin via WebSocket for real-time visualization
and REST API for disaster injection.
"""

import asyncio
import json
import random
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import os
from stable_baselines3 import PPO

from disaster_sim.digital_twin.city_config import get_config, Terrain
from disaster_sim.digital_twin.city_generator import CityGenerator
from disaster_sim.digital_twin.world_state import WorldState
from disaster_sim.engine.physics import PhysicsEngine, Action
from disaster_sim.digital_twin.disaster_dynamics import DisasterDynamics
from disaster_sim.digital_twin.disaster_injector import DisasterInjector
from disaster_sim.predictive.cloud_coordinator import CloudCoordinator
from disaster_sim.rl.observation import ObservationBuilder


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
        self.config = get_config("medium")
        self.config.seed = 42
        
        gen = CityGenerator(self.config)
        self.city = gen.generate()
        
        self.world = WorldState(self.city)
        self.physics = PhysicsEngine(self.world)
        self.dynamics = DisasterDynamics(self.world)
        self.injector = DisasterInjector(self.world)
        self.coordinator = CloudCoordinator(self.world)
        
        self.running = False
        self.clients: list[WebSocket] = []
        
        # Load RL Models if they exist
        self.models = {}
        if os.path.exists("./models/drone_ppo.zip"):
            self.models["drone"] = PPO.load("./models/drone_ppo")
        if os.path.exists("./models/ambulance_ppo.zip"):
            self.models["ambulance"] = PPO.load("./models/ambulance_ppo")
            
        # Need observation builder for inference
        self.obs_builder = ObservationBuilder(self.world, fov_size=15)

    async def run_loop(self):
        """Main simulation loop running at ~10Hz."""
        self.running = True
        while self.running:
            # 1. Agent actions (Trained Inference or Random Walk)
            for agent in self.world.active_agents:
                if agent.agent_type in self.models:
                    obs = self.obs_builder.get_observation(agent.id)
                    action, _ = self.models[agent.agent_type].predict(obs, deterministic=False)
                else:
                    action = random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY])
                    
                # Auto-interact to help undertrained models rescue/dropoff
                if getattr(agent.spec, 'can_rescue', False) and not agent.carrying_victim:
                    victims = self.world.get_victims_in_range(int(agent.row), int(agent.col), radius=1)
                    if victims and not victims[0].rescued:
                        action = Action.INTERACT
                elif getattr(agent.spec, 'can_transport', False) and agent.carrying_victim:
                    if self.world.terrain_at(int(agent.row), int(agent.col)) == Terrain.HOSPITAL:
                        action = Action.INTERACT

                self.physics.step(agent.id, int(action))

            # Sync communications for MARL network constraints
            self.physics.sync_communications()

            # 2. Advance World Physics
            self.world.timestep += 1
            if self.world.timestep % 10 == 0:
                self.dynamics.step()
                
            # 3. AI Cloud Coordinator
            self.coordinator.tick()

            # 4. Broadcast state
            await self.broadcast_state()
            
            await asyncio.sleep(0.1)

    async def broadcast_state(self):
        """Serialize and send state to all connected WebSockets."""
        if not self.clients:
            return

        state = self.world.to_dict()
        
        # Add predictive heatmaps
        # Convert keys from tuples to strings for JSON
        congestion_map = {
            f"{r},{c}": val for (r, c), val in getattr(self.coordinator, 'current_traffic_map', {}).items()
        }
        state["predictive_traffic"] = congestion_map
        
        # Flatten grid to list
        state["grid"] = self.world.grid.tolist()

        message = json.dumps(state)
        for client in self.clients:
            try:
                await client.send_text(message)
            except Exception:
                pass


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
            # Wait for client messages if any (e.g., manual overrides)
            data = await websocket.receive_text()
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
