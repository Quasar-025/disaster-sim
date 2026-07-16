# Project Architecture & Workflow Documentation

This document describes the internal workings of the Disaster Sim project, breaking down the architecture layer by layer and explaining the end-to-end lifecycle of the simulation and training process.

---

## System Architecture

The project is divided into several loosely coupled modules to separate the game engine, reinforcement learning environment, backend server, and frontend visualization.

### 1. The Digital Twin (Core Engine)
Located in `disaster_sim/digital_twin/` and `disaster_sim/engine/`.
* **CityGenerator**: Procedurally generates the 2D grid map based on configuration presets (small, medium, large). It lays out roads, buildings, hospitals, and spawns randomized victims and agents.
* **WorldState**: The central source of truth. It holds the grid representation, the list of active agents (drones, ambulances), victims, and keeps track of global metrics (coverage, rescued count, battery levels).
* **PhysicsEngine**: Handles the logic for agent movement, collision detection, and interactions (like picking up or dropping off a victim).
* **DisasterDynamics & DisasterInjector**: Manages hazards. The `DisasterInjector` allows external triggers (like an earthquake API call), while `DisasterDynamics` simulates the natural spread and consequences over time (e.g., floods expanding, victim health degrading).

### 2. Reinforcement Learning (RL) Layer
Located in `disaster_sim/rl/`.
* **DisasterEnv**: A standard Gymnasium environment that wraps the `WorldState` and `PhysicsEngine`. It exposes standard `reset()` and `step()` functions so RL algorithms can interact with the digital twin.
* **ObservationBuilder**: The "eyes" of the AI. Instead of giving the AI raw data, this builds a 6-channel `15x15` tensor representing the agent's immediate Field of View (FOV). The channels encode: Terrain, Fog of War, Self Position, Other Agents, Known Victims, and an A* Path breadcrumb trail to their target.
* **RewardCalculator**: Evaluates the agent's actions and provides scalar feedback (positive rewards for rescuing/exploring, negative penalties for collisions or battery drain).
* **Networks & Training**: Uses `Stable Baselines3`. The custom `DisasterCNN` processes the 6-channel spatial observation. Scripts like `train.py` use `SubprocVecEnv` to spawn multiple parallel environments (e.g., 20 cores) to gather data rapidly, which is then fed into the PPO algorithm running on the GPU.

### 3. Backend & API (The Orchestrator)
Located in `disaster_sim/backend/server.py`.
* **SimRunner Loop**: A continuous `async` loop that ticks at ~10Hz. On every tick, it:
  1. Asks the trained PPO models for the next action for each agent (or chooses randomly if models aren't trained yet).
  2. Steps the `PhysicsEngine` to move the agents.
  3. Steps the `DisasterDynamics` (every 10 ticks) to update hazards.
  4. Serializes the `WorldState` and broadcasts it to all connected frontend clients via WebSockets.
* **REST API**: Exposes endpoints (like `/api/disaster/earthquake`) to allow external applications (or the frontend) to inject new disasters dynamically while the simulation is running.

### 4. Frontend Visualization
Located in `frontend/`.
* A Vite/React application that connects to the FastAPI WebSocket.
* It receives the flattened grid and entity positions at 10Hz and renders them in real-time, providing an interactive dashboard to monitor the autonomous fleet's progress.

---

## Step-by-Step Workflow (How it runs currently)

### Scenario A: Training the AI
1. The user runs `python -m disaster_sim.rl.train`.
2. The script initializes **N parallel instances** (e.g., 20) of `DisasterEnv` using `SubprocVecEnv`. Each instance generates a randomized digital twin city.
3. The environments run in the background (CPU-bound). For every step, the `ObservationBuilder` uses A* search and grid-cropping to generate the 6x15x15 visual input.
4. Once enough steps are gathered (e.g., 40,000 steps), the data is sent to the GPU.
5. The `DisasterCNN` evaluates the spatial grids, and the PPO algorithm updates the neural network weights to maximize the rewards provided by `RewardCalculator`.
6. This loops for millions of timesteps until the model is saved to `models/drone_ppo.zip` and `models/ambulance_ppo.zip`.

### Scenario B: Running the Live Digital Twin Server
1. The user runs the backend server (`uvicorn disaster_sim.backend.server:app`).
2. `server.py` initializes a single instance of `WorldState` and `PhysicsEngine`.
3. It checks the `models/` directory. If it finds the `.zip` files generated during Scenario A, it loads the trained neural networks into memory.
4. The `run_loop()` starts. 10 times a second, it passes the current local observation of each agent through the neural network to predict the optimal move.
5. The agents move around the digital twin autonomously, rescuing victims and exploring.
6. The updated state is blasted out over WebSockets.
7. (Optional) The user clicks a button on the frontend, hitting a REST endpoint to trigger an earthquake, changing the map dynamically and forcing the AI to adapt.
