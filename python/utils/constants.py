"""
Sensonaut - Shared Constants

This module consolidates all constants used across the codebase to ensure
consistency and make parameter tuning easier.
"""
import numpy as np
import math

# =============================================================================
# Grid World Constants
# =============================================================================
GRID_W = 13  # Grid width (x-axis cells)
GRID_H = 29  # Grid height (y-axis cells)

# Car footprint dimensions (in grid cells)
CAR_W = 2  # Car width (cells)
CAR_H = 4  # Car height (cells)

# Slot layout for vehicle placement (4x4 grid of slots)
SLOT_COL_ANCHORS = [1, 4, 7, 10]
SLOT_ROW_ANCHORS = [0, 10, 15, 25]

# =============================================================================
# Unity World Coordinate Bounds
# =============================================================================
# These define the mapping between grid coordinates and Unity world space
X_MIN = 25.0   # Unity x minimum (maps to grid x=0)
X_MAX = 38.0   # Unity x maximum (maps to grid x=GRID_W-1)
Z_MIN = -14.5  # Unity z minimum (maps to grid y=GRID_H-1)
Z_MAX = 14.5   # Unity z maximum (maps to grid y=0)

# =============================================================================
# Belief Grid for Bayesian Filtering
# =============================================================================
# Theta grid: bearing estimates from -pi to +pi (361 points for 1-degree resolution)
THETA_GRID = np.linspace(-math.pi, math.pi, 361)

# R grid: distance estimates from 0.5 to 30.0 meters (30 points)
R_GRID = np.linspace(0.5, 30.0, 30)

# =============================================================================
# Physics Constants
# =============================================================================
HEAD_RADIUS = 0.0875  # Human head radius in meters (for ITD calculation)
SPEED_OF_SOUND = 343.0  # Speed of sound in m/s at room temperature

# Derived constant for ITD calculations (in microseconds per meter)
SPEED_OF_SOUND_US = SPEED_OF_SOUND / 1_000_000  # m/microsecond

# =============================================================================
# Default Noise Parameters
# =============================================================================
ITD_NOISE_SCALE = 30.0   # Microseconds - uncertainty in ITD measurements
ILD_NOISE_SCALE = 0.1    # dB - uncertainty in ILD measurements

# =============================================================================
# Visual Parameters
# =============================================================================
DEFAULT_FOV_DEG = 110  # Meta Quest 3 horizontal field of view in degrees
DEFAULT_FOV_RAD = math.radians(DEFAULT_FOV_DEG)

# =============================================================================
# Color/Prefab Mapping
# =============================================================================
# Maps color indices to prefab names and display colors
COLOR_TO_PREFAB = {
    0: "black",
    1: "red",
    2: "white"
}

PREFAB_TO_COLOR = {v: k for k, v in COLOR_TO_PREFAB.items()}

# Display colors for visualization (hex)
COLOR_MAP = {
    "black": "#2e2e2e",
    "red": "#27d6d0",   # Note: displayed as cyan for visibility
    "white": "#dddddd"
}

# =============================================================================
# Action Space
# =============================================================================
ACTION_TURN_LEFT = 0
ACTION_TURN_RIGHT = 1
ACTION_MOVE_FORWARD = 2
ACTION_COMMIT = 3
ACTION_STAY = 4

ACTION_NAMES = {
    ACTION_TURN_LEFT: "turn_left",
    ACTION_TURN_RIGHT: "turn_right",
    ACTION_MOVE_FORWARD: "move_forward",
    ACTION_COMMIT: "commit",
    ACTION_STAY: "stay"
}

NUM_ACTIONS = 5
