"""
Traffic & Congestion Predictor (Graph Neural Network)

Uses a native PyTorch Graph Convolutional Network (GCN) to predict
future traffic congestion on the road network based on:
- Current vehicle density
- Road damage / Debris
- Weather intensity
- Proximity to active disasters (Fires, Floods)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from disaster_sim.digital_twin.world_state import WorldState
from disaster_sim.digital_twin.city_config import Terrain


class GraphConvLayer(nn.Module):
    """Simple Graph Convolutional Layer (A * X * W)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, in_features)
        # adj: (N, N) adjacency matrix (preferably normalized)
        support = self.linear(x)          # (N, out_features)
        output = torch.matmul(adj, support) # (N, out_features)
        return output + self.bias


class TrafficGNN(nn.Module):
    """
    GNN for predicting traffic congestion in the digital twin.
    Inputs: Node features (N, 5)
    Outputs: Congestion prediction (N, 1) -> 0.0 to 1.0
    """
    def __init__(self, num_features: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.gc1 = GraphConvLayer(num_features, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gc3 = GraphConvLayer(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = torch.sigmoid(self.gc3(x, adj))
        return x


class TrafficPredictorSystem:
    """Wrapper to integrate the GNN with the WorldState."""
    
    def __init__(self, world: WorldState):
        self.world = world
        self.model = TrafficGNN()
        self.model.eval()  # Inference mode
        
        # Build the static road graph
        self.road_nodes = self._extract_road_nodes()
        self.num_nodes = len(self.road_nodes)
        self.adj_matrix = self._build_adjacency_matrix()
        
    def _extract_road_nodes(self) -> list[tuple[int, int]]:
        """Find all road, hospital, and charging station cells (navigable for vehicles)."""
        nodes = []
        h, w = self.world.height, self.world.width
        for r in range(h):
            for c in range(w):
                terrain = self.world.terrain_at(r, c)
                if terrain in (Terrain.ROAD, Terrain.HOSPITAL, Terrain.CHARGING_STATION):
                    nodes.append((r, c))
        return nodes
        
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """Build a normalized adjacency matrix for the road network."""
        N = self.num_nodes
        adj = np.zeros((N, N), dtype=np.float32)
        
        node_to_idx = {pos: i for i, pos in enumerate(self.road_nodes)}
        
        for i, (r, c) in enumerate(self.road_nodes):
            adj[i, i] = 1.0  # Self-loop
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in node_to_idx:
                    j = node_to_idx[(nr, nc)]
                    adj[i, j] = 1.0
                    
        # Row-normalize the adjacency matrix (D^-1 * A)
        rowsum = adj.sum(axis=1)
        rowsum[rowsum == 0] = 1.0
        adj = adj / rowsum[:, np.newaxis]
        
        return torch.from_numpy(adj)
        
    def _extract_node_features(self) -> torch.Tensor:
        """
        Extract real-time features for each road node.
        Features: [vehicle_count, is_damaged, weather_impact, proximity_to_disaster, time_factor]
        """
        N = self.num_nodes
        features = np.zeros((N, 5), dtype=np.float32)
        
        # 1. Count vehicles (agents) on each node
        vehicle_counts = {}
        for agent in self.world.active_agents:
            # We care about vehicles causing traffic (ambulances, ground_robots)
            if agent.agent_type in ("ambulance", "ground_robot", "heavy_lifter"):
                pos = agent.grid_pos
                vehicle_counts[pos] = vehicle_counts.get(pos, 0) + 1
                
        for i, (r, c) in enumerate(self.road_nodes):
            # F0: Vehicle count
            features[i, 0] = vehicle_counts.get((r, c), 0) / 3.0  # Normalize roughly max 3 per cell
            
            # F1: Road damage / Debris
            terrain = self.world.terrain_at(r, c)
            features[i, 1] = 1.0 if terrain == Terrain.DEBRIS else 0.0
            
            # F2: Weather (Placeholder - reading from a global weather state if added, else 0)
            features[i, 2] = 0.5  # E.g., medium rain
            
            # F3: Proximity to active disaster (Fire/Flood)
            # Rough heuristic: check 5x5 window
            disaster_score = 0.0
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.world.height and 0 <= nc < self.world.width:
                        t = self.world.terrain_at(nr, nc)
                        if t in (Terrain.FIRE, Terrain.WATER, Terrain.BUILDING_DAMAGED):
                            disaster_score += 1.0
            features[i, 3] = min(1.0, disaster_score / 10.0)
            
            # F4: Time factor (rush hour simulation based on timestep)
            # Cycle every 500 steps
            time_cycle = (self.world.timestep % 500) / 500.0
            features[i, 4] = np.sin(time_cycle * np.pi)
            
        return torch.from_numpy(features)

    def predict_congestion(self) -> dict[tuple[int, int], float]:
        """
        Run the GNN to predict traffic congestion for the entire road network.
        Returns a mapping from (row, col) to congestion probability (0.0 to 1.0).
        """
        if self.num_nodes == 0:
            return {}
            
        features = self._extract_node_features()
        
        with torch.no_grad():
            predictions = self.model(features, self.adj_matrix)
            
        preds_np = predictions.squeeze(-1).numpy()
        
        result = {}
        for i, pos in enumerate(self.road_nodes):
            result[pos] = float(preds_np[i])
            
        return result
