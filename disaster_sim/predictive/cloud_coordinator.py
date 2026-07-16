"""
Cloud Coordinator AI (Centralized Dispatch)

Acts as the "God-view" dispatcher. Reads the WorldState and GNN traffic predictions
to assign optimal targets to agents, balance hospital loads, and preempt
traffic lights for emergency vehicles.
"""

from disaster_sim.digital_twin.world_state import WorldState
from disaster_sim.digital_twin.city_config import Terrain
from disaster_sim.predictive.traffic_gnn import TrafficPredictorSystem
from disaster_sim.predictive.hospital_optimizer import HospitalOptimizer


class CloudCoordinator:
    """Centralized coordinator for the disaster response fleet."""
    
    def __init__(self, world: WorldState):
        self.world = world
        self.traffic_predictor = TrafficPredictorSystem(world)
        self.hospital_optimizer = HospitalOptimizer(world)
        
        # A dictionary mapping ambulance_id -> assigned hospital target
        self.ambulance_assignments = {}

    def tick(self) -> None:
        """
        Execute one coordination cycle.
        1. Predict traffic.
        2. Optimize hospital assignments for victims in transit.
        3. Preempt traffic lights along emergency routes.
        """
        # 1. Update Traffic Predictions
        # (Stored in memory, can be pushed to frontend)
        self.current_traffic_map = self.traffic_predictor.predict_congestion()
        
        # 2. Hospital Load Balancing
        # Find all victims currently being carried by agents
        victims_in_transit = []
        for agent in self.world.active_agents:
            if agent.carrying_victim:
                victims_in_transit.append(agent.carrying_victim)
                
        # Optimize targets
        assignments = self.hospital_optimizer.optimize_distribution(victims_in_transit)
        self.hospital_optimizer.apply_assignments(assignments)
        
        # 3. Traffic Light Preemption
        self._preempt_traffic_lights()

    def _preempt_traffic_lights(self) -> None:
        """
        Clear traffic lights for ambulances by forcing them to 'green' (allow pass).
        This is simulated by checking if an ambulance is near a traffic light.
        """
        # For simplicity in this CA model, we find all traffic light agents
        # and see if an ambulance is approaching.
        ambulances = [a for a in self.world.active_agents if a.agent_type == "ambulance"]
        
        if not ambulances:
            return
            
        for tl_agent in self.world.active_agents:
            if tl_agent.agent_type == "traffic_light":
                # By default, let's assume traffic lights alternate (status: red/green)
                # But if an ambulance is within 3 cells, force it to 'preempted'
                near_ambulance = False
                tl_r, tl_c = tl_agent.grid_pos
                
                for amb in ambulances:
                    ar, ac = amb.grid_pos
                    if abs(ar - tl_r) + abs(ac - tl_c) <= 3:
                        near_ambulance = True
                        break
                        
                if near_ambulance:
                    tl_agent.status = "preempted_green"
                else:
                    # Normal cycle based on timestep
                    if (self.world.timestep // 10) % 2 == 0:
                        tl_agent.status = "green"
                    else:
                        tl_agent.status = "red"
