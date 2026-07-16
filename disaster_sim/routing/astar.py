"""
A* Pathfinding for disaster simulation.
"""

from __future__ import annotations

import heapq
from typing import Optional

from disaster_sim.digital_twin.world_state import WorldState


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Manhattan distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_path(
    world: WorldState,
    start: tuple[int, int],
    goal: tuple[int, int],
    agent_type: str
) -> Optional[list[tuple[int, int]]]:
    """Find the shortest path from start to goal for a specific agent type."""
    
    # If the goal is not passable, we can't reach it
    if not world.is_passable(goal[0], goal[1], agent_type):
        return None

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_node = (r + dr, c + dc)
            if not world.is_passable(next_node[0], next_node[1], agent_type):
                continue
                
            new_cost = cost_so_far[current] + 1.0  # Uniform cost for grid steps
            
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(goal, next_node)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current

    if goal not in came_from:
        return None  # No path found

    # Reconstruct path
    path = []
    current_node = goal
    while current_node != start:
        path.append(current_node)
        current_node = came_from[current_node] # type: ignore
    path.reverse()
    
    return path
