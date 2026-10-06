"""
Hospital Load Balancing Optimizer

Uses bipartite matching (linear sum assignment) to optimally distribute
critical patients across the city's hospitals, minimizing transport distance
while strictly respecting ICU bed and blood supply constraints.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from disaster_sim.digital_twin.world_state import WorldState


class HospitalOptimizer:
    """Optimizes patient distribution to hospitals."""

    def __init__(self, world: WorldState):
        self.world = world

    def optimize_distribution(self, victims_to_transport: list[str]) -> dict[str, str]:
        """
        Assigns victims to hospitals.
        
        Args:
            victims_to_transport: List of victim IDs that need immediate hospital care.
            
        Returns:
            A dictionary mapping Victim ID -> Hospital ID.
        """
        if not victims_to_transport or not self.world.hospitals:
            return {}

        # Build a list of available hospital "slots" (each slot is 1 ICU bed)
        hospital_slots = []
        for hid, hospital in self.world.hospitals.items():
            # A hospital can take patients up to its available ICU beds
            # We also require at least 1 unit of blood per critical patient (simplified)
            capacity = min(
                hospital.available_icu_beds,
                int(hospital.available_blood)
            )
            for _ in range(capacity):
                hospital_slots.append(hid)
                
        if not hospital_slots:
            # No capacity anywhere
            return {}
            
        # Cost matrix: rows = victims, cols = hospital slots
        num_victims = len(victims_to_transport)
        num_slots = len(hospital_slots)
        
        cost_matrix = np.zeros((num_victims, num_slots), dtype=np.float32)
        
        for i, vid in enumerate(victims_to_transport):
            victim = self.world.victims[vid]
            for j, hid in enumerate(hospital_slots):
                hospital = self.world.hospitals[hid]
                
                # Cost is primarily Manhattan distance
                dist = abs(victim.row - hospital.row) + abs(victim.col - hospital.col)
                cost_matrix[i, j] = dist
                
        # Scipy's linear_sum_assignment finds the minimum cost bipartite matching
        # It handles rectangular matrices (e.g. if num_victims > num_slots or vice versa)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignments = {}
        for i, j in zip(row_ind, col_ind):
            vid = victims_to_transport[i]
            hid = hospital_slots[j]
            assignments[vid] = hid
            
        return assignments

    def apply_assignments(self, assignments: dict[str, str]) -> None:
        """
        Actually commits the assignments to the WorldState,
        deducting resources from the target hospitals.
        """
        for vid, hid in assignments.items():
            victim = self.world.victims[vid]
            hospital = self.world.hospitals[hid]
            
            # Just update expected hospital resources. 
            # The actual transport logic and state updates happen in physics.py when the ambulance arrives.
            hospital.current_patients += 1
            if victim.severity.value == "critical":
                hospital.available_blood -= 2.0
            else:
                hospital.available_blood -= 0.5
