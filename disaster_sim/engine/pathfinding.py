"""
A* Pathfinding Module for Hybrid AI Navigation.
"""

from __future__ import annotations

import heapq
from typing import Optional

from disaster_sim.digital_twin.world_state import WorldState, AgentState
from disaster_sim.digital_twin.city_config import Terrain

def heuristic(r1: int, c1: int, r2: int, c2: int) -> float:
    # Manhattan distance
    return abs(r1 - r2) + abs(c1 - c2)

def is_passable_optimistic(world: WorldState, agent: AgentState, r: int, c: int) -> bool:
    """
    Check if a cell is passable based on the agent's LOCAL memory.
    If unexplored, assume it's passable (optimism under uncertainty).
    """
    if r < 0 or r >= world.height or c < 0 or c >= world.width:
        return False
        
    # If not explored locally, assume it is EMPTY (passable by default)
    if agent.local_explored is None or not agent.local_explored[r, c]:
        # Wait, if movement type is water_only, EMPTY is not passable!
        # So optimistic means we assume it's whatever terrain we need.
        # But actually, if they don't know, maybe we just say it's passable if it's not a known obstacle.
        # For simplicity, if unexplored, assume passable unless they are water_only (then they shouldn't explore randomly).
        if agent.spec.movement_type == "water_only":
            return False # Boats can't be optimistic on land
        return True
        
    terrain_idx = world.grid[r, c]
    terrain = Terrain(terrain_idx)
    return terrain.name in agent.spec.terrain_passable

def a_star_search(world: WorldState, agent: AgentState, target_r: int, target_c: int) -> list[tuple[int, int]]:
    """
    Finds the shortest path from agent's current position to target using A*.
    Returns a list of (row, col) tuples representing the path, starting from the next step.
    Returns empty list if no path is found.
    """
    start_r, start_c = agent.grid_pos
    
    if start_r == target_r and start_c == target_c:
        return []
        
    # Priority queue stores (f_score, count, (r, c))
    open_set = []
    heapq.heappush(open_set, (0, 0, (start_r, start_c)))
    
    came_from = {}
    g_score = {(start_r, start_c): 0}
    f_score = {(start_r, start_c): heuristic(start_r, start_c, target_r, target_c)}
    
    count = 1
    max_nodes = 2000
    
    while open_set and count < max_nodes:
        _, _, current = heapq.heappop(open_set)
        
        if current == (target_r, target_c):
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
            
        r, c = current
        
        # 4-way movement
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            
            if not is_passable_optimistic(world, agent, nr, nc):
                continue
                
            tentative_g = g_score[current] + 1
            
            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative_g
                f = tentative_g + heuristic(nr, nc, target_r, target_c)
                f_score[(nr, nc)] = f
                heapq.heappush(open_set, (f, count, (nr, nc)))
                count += 1
                
    return []
