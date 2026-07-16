# Disaster Sim: Autonomous Multi-Agent Response System

A real-time Digital Twin and Reinforcement Learning (RL) simulation environment for autonomous disaster response. This project simulates a city environment where autonomous agents (Drones, Ambulances) learn to navigate, scout, and rescue victims using Deep Reinforcement Learning (PPO).

## What is Built Currently

* **Digital Twin Engine:** A custom 2D simulation environment (`disaster_sim.digital_twin`) that generates cities with distinct terrains, building layouts, and disaster dynamics (earthquakes, floods).
* **Reinforcement Learning Environment:** A fully integrated Gymnasium environment (`DisasterEnv`) featuring spatial 6-channel grid observations (terrain, fog of war, known victims, A* paths, agent locations).
* **Multi-Agent PPO Training Pipeline:** Training scripts powered by PyTorch and Stable Baselines3. Includes a custom Convolutional Neural Network (`DisasterCNN`) for processing spatial grid data, and utilizes `SubprocVecEnv` for aggressive multi-core parallel data collection.
* **Backend API & WebSocket Server:** A FastAPI backend (`server.py`) that runs the simulation loop at 10Hz, exposing real-time state via WebSockets for the frontend, and REST endpoints for triggering disaster events on the fly.
* **Interactive Frontend:** A Vite + React (Three.js/Fiber) frontend for visualizing the digital twin and monitoring the autonomous fleet.

## What Remains (Roadmap)

* **Computer Vision (Milestone 7):** Integration of OpenCV and YOLO (Ultralytics) for simulated drone camera feeds and real-time victim detection.
* **Advanced Route Optimization (Milestone 5):** Upgrading from basic A* to global fleet-level route optimization using Google OR-Tools and NetworkX for multi-vehicle routing protocols (VRP).
* **Predictive Cloud Coordinator:** Expanding the AI Cloud Coordinator to predict disaster spread (e.g., flood expansion) and pre-emptively route agents.
* **MARL (Multi-Agent RL):** Moving from independent single-agent PPO policies to full PettingZoo-based cooperative Multi-Agent Reinforcement Learning.

---

## Setup Instructions

### 1. Python Backend & RL Environment

It is highly recommended to use a virtual environment (e.g., `venv` or `conda`). Python 3.10+ is required.

```bash
# Clone the repository
git clone <your-repo-url>
cd disaster

# Install the package in editable mode with ALL dependencies 
# (Includes RL, Computer Vision, Routing, and Backend modules)
pip install -e .[all]
```

### 2. Frontend Visualization

The frontend is built with Vite, React, and Three.js. You need Node.js installed.

```bash
cd frontend
npm install
```

---

## Running the Project

### Start the Backend Server (Digital Twin)
This starts the FastAPI server which runs the simulation loop and serves the WebSocket connections.
```bash
python -m disaster_sim.backend.server
# or
uvicorn disaster_sim.backend.server:app --host 0.0.0.0 --port 8000
```

### Start the Frontend Web App
In a new terminal window:
```bash
cd frontend
npm run dev
```
Navigate to the `localhost` URL provided in the terminal to view the simulation live.

### Run Reinforcement Learning Training
To train the agents using the PPO algorithm (will utilize your CPU heavily for rollout generation and GPU for optimization):

```bash
# Train the baseline models (Drone & Ambulance sequentially)
python -m disaster_sim.rl.train

# Or run the configurable PPO script with specific presets
python -m disaster_sim.rl.train_ppo --preset small --n-envs 20
```

*Note on Training:* By default, training heavily leverages multiprocessing (`n_envs=20`). Ensure you have sufficient CPU cores and RAM. If you encounter a `MemoryError` on Windows, reduce `n_envs` in the scripts.

### Play Interactively
You can play the environment manually to test mechanics:
```bash
python -m disaster_sim.rl.env --interactive
```
