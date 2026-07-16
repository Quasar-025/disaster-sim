"""
City Renderer — Pygame-based 2D visualization for debugging and validation.

Renders the city grid with color-coded terrain, agent positions,
victim locations, and an info overlay. Supports both interactive
windowed mode and static image export.

Controls:
    Arrow keys / WASD: Pan the camera
    +/- or scroll: Zoom in/out
    R: Reset view
    G: Toggle grid lines
    F: Toggle fog of war
    V: Toggle victim markers
    ESC / Q: Close window
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from disaster_sim.digital_twin.city_generator import GeneratedCity
    from disaster_sim.digital_twin.world_state import WorldState

from disaster_sim.digital_twin.city_config import (
    TERRAIN_COLORS,
    Severity,
    Terrain,
)


# Agent type → marker color
AGENT_COLORS: dict[str, tuple[int, int, int]] = {
    "drone": (0, 180, 255),       # Cyan-blue
    "ground_robot": (0, 220, 80),  # Green
    "ambulance": (255, 255, 100),  # Yellow
    "boat": (100, 160, 255),       # Light blue
    "heavy_lifter": (255, 160, 0), # Orange
}

# Victim severity → marker color
SEVERITY_COLORS: dict[Severity, tuple[int, int, int]] = {
    Severity.CRITICAL: (255, 30, 30),   # Bright red
    Severity.SERIOUS: (255, 140, 30),   # Orange
    Severity.STABLE: (255, 220, 50),    # Yellow
}


class CityRenderer:
    """Pygame renderer for 2D city visualization.

    Can render from either a static GeneratedCity or a live WorldState.
    """

    DEFAULT_CELL_SIZE = 4   # Pixels per grid cell at 1x zoom
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    INFO_PANEL_WIDTH = 280
    FPS = 30

    def __init__(
        self,
        city: GeneratedCity,
        world_state: Optional[WorldState] = None,
    ):
        self.city = city
        self.world_state = world_state
        self.cell_size = self.DEFAULT_CELL_SIZE
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.show_grid = False
        self.show_fog = False
        self.show_victims = True
        self._pygame_initialized = False

    def _ensure_pygame(self) -> None:
        """Lazy-initialize Pygame."""
        if not self._pygame_initialized:
            import pygame
            pygame.init()
            self._pygame_initialized = True

    def render_to_surface(self) -> "pygame.Surface":
        """Render the city to a Pygame surface (no window needed)."""
        import pygame

        self._ensure_pygame()

        w = self.city.width * self.cell_size
        h = self.city.height * self.cell_size
        surface = pygame.Surface((w, h))
        surface.fill((30, 30, 30))

        grid = self.world_state.grid if self.world_state else self.city.grid

        # Draw terrain
        for row in range(self.city.height):
            for col in range(self.city.width):
                terrain = Terrain(grid[row, col])
                color = TERRAIN_COLORS.get(terrain, (50, 50, 50))

                # Fire flickering effect
                if terrain == Terrain.FIRE:
                    flicker = np.random.randint(-20, 20)
                    color = (
                        max(0, min(255, color[0] + flicker)),
                        max(0, min(255, color[1] + flicker // 2)),
                        color[2],
                    )

                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(surface, color, rect)

        # Draw grid lines
        if self.show_grid and self.cell_size >= 4:
            grid_color = (50, 50, 50)
            for row in range(self.city.height + 1):
                y = row * self.cell_size
                pygame.draw.line(surface, grid_color, (0, y), (w, y))
            for col in range(self.city.width + 1):
                x = col * self.cell_size
                pygame.draw.line(surface, grid_color, (x, 0), (x, h))

        # Draw victims
        if self.show_victims:
            victims = (
                self.world_state.victims.values()
                if self.world_state
                else self.city.victims
            )
            for v in victims:
                if self.world_state:
                    if v.rescued:
                        continue
                    severity = v.severity
                    row, col = v.row, v.col
                else:
                    severity = v.severity
                    row, col = v.row, v.col

                color = SEVERITY_COLORS.get(severity, (255, 255, 255))
                cx = col * self.cell_size + self.cell_size // 2
                cy = row * self.cell_size + self.cell_size // 2
                radius = max(2, self.cell_size // 2 + 1)
                pygame.draw.circle(surface, color, (cx, cy), radius)
                # White outline for visibility
                pygame.draw.circle(surface, (255, 255, 255), (cx, cy), radius, 1)

        # Draw agent positions
        agents = (
            self.world_state.agents.values()
            if self.world_state
            else [
                type("A", (), {"agent_type": s.agent_type, "row": s.row, "col": s.col, "id": s.id})
                for s in self.city.agent_spawns
            ]
        )
        for agent in agents:
            color = AGENT_COLORS.get(agent.agent_type, (200, 200, 200))
            cx = int(agent.col) * self.cell_size + self.cell_size // 2
            cy = int(agent.row) * self.cell_size + self.cell_size // 2
            size = max(3, self.cell_size // 2 + 2)

            # Draw agent marker (diamond shape)
            points = [
                (cx, cy - size),       # Top
                (cx + size, cy),       # Right
                (cx, cy + size),       # Bottom
                (cx - size, cy),       # Left
            ]
            pygame.draw.polygon(surface, color, points)
            pygame.draw.polygon(surface, (255, 255, 255), points, 1)

        # Draw hospital markers (cross)
        for hr, hc in self.city.hospitals:
            cx = hc * self.cell_size + self.cell_size // 2
            cy = hr * self.cell_size + self.cell_size // 2
            s = max(3, self.cell_size)
            pygame.draw.line(surface, (255, 50, 50), (cx - s, cy), (cx + s, cy), 2)
            pygame.draw.line(surface, (255, 50, 50), (cx, cy - s), (cx, cy + s), 2)

        # Draw charging station markers (lightning bolt)
        for sr, sc in self.city.charging_stations:
            cx = sc * self.cell_size + self.cell_size // 2
            cy = sr * self.cell_size + self.cell_size // 2
            s = max(2, self.cell_size // 2)
            pygame.draw.circle(surface, (50, 255, 100), (cx, cy), s + 1)
            pygame.draw.circle(surface, (255, 255, 255), (cx, cy), s + 1, 1)

        # Draw fog of war
        if self.show_fog and self.world_state is not None:
            fog_surface = pygame.Surface((w, h), pygame.SRCALPHA)
            for row in range(self.city.height):
                for col in range(self.city.width):
                    if not self.world_state.explored[row, col]:
                        rect = pygame.Rect(
                            col * self.cell_size,
                            row * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        )
                        pygame.draw.rect(fog_surface, (0, 0, 0, 160), rect)
            surface.blit(fog_surface, (0, 0))

        return surface

    def _render_info_panel(self, screen: "pygame.Surface") -> None:
        """Draw an information panel on the right side of the screen."""
        import pygame

        panel_x = self.WINDOW_WIDTH - self.INFO_PANEL_WIDTH
        panel_rect = pygame.Rect(panel_x, 0, self.INFO_PANEL_WIDTH, self.WINDOW_HEIGHT)
        pygame.draw.rect(screen, (25, 25, 35), panel_rect)
        pygame.draw.line(
            screen, (60, 60, 80),
            (panel_x, 0), (panel_x, self.WINDOW_HEIGHT), 2,
        )

        font = pygame.font.SysFont("consolas", 14)
        title_font = pygame.font.SysFont("consolas", 16, bold=True)
        y = 15
        line_height = 20

        def draw_text(text: str, color: tuple = (200, 200, 220), bold: bool = False) -> None:
            nonlocal y
            f = title_font if bold else font
            surf = f.render(text, True, color)
            screen.blit(surf, (panel_x + 12, y))
            y += line_height

        # Title
        draw_text("DISASTER RESPONSE", (100, 200, 255), bold=True)
        draw_text("COMMAND CENTER", (100, 200, 255), bold=True)
        y += 5

        # City info
        draw_text(f"City: {self.city.width}x{self.city.height}")
        draw_text(f"Seed: {self.city.config.seed}")
        y += 5

        # Terrain legend
        draw_text("--- TERRAIN ---", bold=True)
        legend_items = [
            (Terrain.ROAD, "Road"),
            (Terrain.BUILDING, "Building"),
            (Terrain.BUILDING_DAMAGED, "Damaged"),
            (Terrain.DEBRIS, "Debris"),
            (Terrain.WATER, "Water"),
            (Terrain.FIRE, "Fire"),
            (Terrain.PARK, "Park"),
        ]
        for terrain, name in legend_items:
            color = TERRAIN_COLORS[terrain]
            rect = pygame.Rect(panel_x + 12, y + 3, 12, 12)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (100, 100, 100), rect, 1)
            surf = font.render(f" {name}", True, (200, 200, 220))
            screen.blit(surf, (panel_x + 28, y))
            y += line_height

        y += 5

        # Victim info
        draw_text("--- VICTIMS ---", bold=True)
        total_v = len(self.city.victims)
        if self.world_state:
            rescued = self.world_state.victims_rescued_count
            detected = self.world_state.victims_detected_count
            draw_text(f"Total: {total_v}")
            draw_text(f"Detected: {detected}", (100, 255, 100))
            draw_text(f"Rescued: {rescued}", (100, 255, 100))
            draw_text(f"Remaining: {total_v - rescued}")
        else:
            draw_text(f"Total: {total_v}")
            for sev in Severity:
                count = sum(1 for v in self.city.victims if v.severity == sev)
                color = SEVERITY_COLORS[sev]
                draw_text(f"  {sev.value}: {count}", color)

        y += 5

        # Agent info
        draw_text("--- AGENTS ---", bold=True)
        agents = (
            list(self.world_state.agents.values())
            if self.world_state
            else [
                type("A", (), {
                    "id": s.id, "agent_type": s.agent_type,
                    "battery_fraction": 1.0, "status": "spawn",
                })
                for s in self.city.agent_spawns
            ]
        )
        for agent in agents:
            a_color = AGENT_COLORS.get(agent.agent_type, (200, 200, 200))
            batt = getattr(agent, "battery_fraction", 1.0)
            status = getattr(agent, "status", "spawn")
            batt_pct = int(batt * 100)
            draw_text(f"{agent.id}", a_color)
            draw_text(f"  🔋{batt_pct}% | {status}")

        y += 10

        # Controls
        draw_text("--- CONTROLS ---", bold=True)
        draw_text("Arrow/WASD: Pan")
        draw_text("+/-/Scroll: Zoom")
        draw_text("G: Grid  F: Fog")
        draw_text("V: Victims  R: Reset")
        draw_text("ESC/Q: Quit")

    def save_image(self, path: str) -> None:
        """Render the city and save to a PNG file."""
        import pygame

        self._ensure_pygame()
        surface = self.render_to_surface()
        pygame.image.save(surface, path)

    def run(self) -> None:
        """Run the interactive Pygame window."""
        import pygame

        self._ensure_pygame()

        screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption(
            f"Disaster City — {self.city.width}×{self.city.height} "
            f"(seed={self.city.config.seed})"
        )
        clock = pygame.time.Clock()

        running = True
        pan_speed = 10

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key == pygame.K_f:
                        self.show_fog = not self.show_fog
                    elif event.key == pygame.K_v:
                        self.show_victims = not self.show_victims
                    elif event.key == pygame.K_r:
                        self.camera_x = 0
                        self.camera_y = 0
                        self.cell_size = self.DEFAULT_CELL_SIZE
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.cell_size = min(20, self.cell_size + 1)
                    elif event.key == pygame.K_MINUS:
                        self.cell_size = max(1, self.cell_size - 1)
                elif event.type == pygame.MOUSEWHEEL:
                    if event.y > 0:
                        self.cell_size = min(20, self.cell_size + 1)
                    elif event.y < 0:
                        self.cell_size = max(1, self.cell_size - 1)

            # Keyboard panning
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.camera_x -= pan_speed
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.camera_x += pan_speed
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.camera_y -= pan_speed
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.camera_y += pan_speed

            # Render
            screen.fill((20, 20, 30))

            city_surface = self.render_to_surface()
            view_width = self.WINDOW_WIDTH - self.INFO_PANEL_WIDTH
            view_height = self.WINDOW_HEIGHT

            # Center the city in the viewport
            city_w = city_surface.get_width()
            city_h = city_surface.get_height()
            offset_x = (view_width - city_w) // 2 - int(self.camera_x)
            offset_y = (view_height - city_h) // 2 - int(self.camera_y)

            # Clip to viewport
            view_surface = pygame.Surface((view_width, view_height))
            view_surface.fill((20, 20, 30))
            view_surface.blit(city_surface, (offset_x, offset_y))
            screen.blit(view_surface, (0, 0))

            self._render_info_panel(screen)

            pygame.display.flip()
            clock.tick(self.FPS)

        pygame.quit()
