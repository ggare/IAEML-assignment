"""Pygame frontend for the robotaxi environment (same as main.py, but with
the RoboTaxiEnv and the scripted driver in eval mode).

The stock renderer opens an 800x400 window (map 20x10 cells at 40 px) and
prints the info dict right on top of the map, so the numbers overlap the
obstacles and rays. Here the window is enlarged and the stats are moved
into a dark side panel.

Run from the repository ``src/`` directory:

    python main_robotaxi.py            # scripted driver, map 0
    MAP=1 python main_robotaxi.py      # map 1
    CELL=72 HUD=400 python main_robotaxi.py   # even bigger window
"""

import os

import jax
import pygame
from jax import numpy as jnp

from env.robotaxi import RoboTaxiEnv
from utils.renderer import PygameFrontend
from utils.autoreset import AutoResetWrapper
from utils.scripted_driver import ScriptedDriver

MAP_ID = int(os.environ.get("MAP", 0))

# window settings (can be tuned via environment variables)
CELL_SIZE = int(os.environ.get("CELL", 56))    # pixels per map cell (stock: 40)
HUD_WIDTH = int(os.environ.get("HUD", 330))    # width of the side stats panel

PANEL_BG = (24, 26, 32)
PANEL_FG = (235, 235, 235)
ACCENT = (242, 183, 5)


class WideFrontend(PygameFrontend):
    """Stock renderer, but with a larger window and the info block moved off
    the map into a dark side panel (the original draws it at (10, 10) right
    on top of the map)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        h, w = int(self.params.map_height_width[0]), int(self.params.map_height_width[1])

        # enlarge the window: bigger map area + a side panel for the stats
        self.CELL_SIZE = CELL_SIZE
        self.COLLIDER_RADIUS_PX = CELL_SIZE // 2
        self.AGENT_RADIUS_PX = CELL_SIZE // 2
        self.GOAL_RADIUS_PX = CELL_SIZE // 2
        self.OBSTACLE_SIDE_PX = CELL_SIZE
        self.map_w_px = w * CELL_SIZE
        self.screen = pygame.display.set_mode(
            (self.map_w_px + HUD_WIDTH, h * CELL_SIZE)
        )
        pygame.display.set_caption(f"robotaxi — Assignment 1 (map {MAP_ID})")

    def draw_info(self, screen, info, position=None, font_size=17,
                  line_height=28, color=PANEL_FG):
        """Draw the stats in the side panel instead of on top of the map."""
        screen.fill(
            PANEL_BG,
            pygame.Rect(self.map_w_px, 0, HUD_WIDTH, screen.get_height()),
        )
        font = pygame.font.SysFont("Arial", font_size)
        bold = pygame.font.SysFont("Arial", font_size, bold=True)

        x, y = self.map_w_px + 14, 12
        screen.blit(bold.render(f"robotaxi · map {MAP_ID}", True, ACCENT), (x, y))
        y += int(1.6 * line_height)

        for key, value in info.items():
            if key == "direction_of_path":
                text = f"{key}: x={float(value[0]):+.2f}  y={float(value[1]):+.2f}"
            elif key in ("distance_to_path", "distance_to_goal", "speed"):
                text = f"{key}: {float(value):.2f}"
            elif key == "time":
                text = f"time: {int(value)} ({int(value) / self.params.fps:.1f} s)"
            else:  # success / collision flags
                text = f"{key}: {bool(value)}"
            screen.blit(font.render(text, True, color), (x, y))
            y += line_height

        y += line_height
        screen.blit(
            font.render("scripted driver (pure pursuit)", True, (150, 155, 165)),
            (x, y),
        )


key = jax.random.PRNGKey(0)
env_params, init_state = RoboTaxiEnv.init_params(
    key=key,
    map_id=MAP_ID,
    max_steps=1200,
    discretization_scale=1,
    path_length=100,
    fps=60,
    perception_radius=5.0,
    num_ray_sensors=32,
)

driver = ScriptedDriver(env_params)


def agent_fn(state):
    """The renderer's eval mode calls agent_fn(state); observations are
    recomputed from the state on the fly."""
    obs = RoboTaxiEnv.get_observation(state, env_params)
    return jnp.asarray(driver.act(obs, state))


env = AutoResetWrapper(RoboTaxiEnv, env_params, init_state)

frontend = WideFrontend(env, env_params, init_state, eval_mode=True,
                        agent_fn=agent_fn)
frontend.run()
