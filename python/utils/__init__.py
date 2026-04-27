"""
Sensonaut Utilities Package

Provides shared utilities for the Sensonaut audio-visual search project:
- constants: Shared constants (grid dimensions, physics, colors, etc.)
- coordinates: Coordinate transformation functions
- audio_features: Audio processing (ITD/ILD computation)
- unity_client: Unity communication client
- metrics: Evaluation metrics for comparing human vs. model behavior
"""

from .constants import (
    # Grid constants
    GRID_W, GRID_H, CAR_W, CAR_H,
    # Unity bounds
    X_MIN, X_MAX, Z_MIN, Z_MAX,
    # Belief grids
    THETA_GRID, R_GRID,
    # Physics
    HEAD_RADIUS, SPEED_OF_SOUND, SPEED_OF_SOUND_US,
    # Noise
    ITD_NOISE_SCALE, ILD_NOISE_SCALE,
    # Visual
    DEFAULT_FOV_DEG, DEFAULT_FOV_RAD,
    # Colors
    COLOR_TO_PREFAB, PREFAB_TO_COLOR, COLOR_MAP,
    # Actions
    ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_MOVE_FORWARD,
    ACTION_COMMIT, ACTION_STAY, ACTION_NAMES, NUM_ACTIONS,
)

from .coordinates import (
    grid_to_unity,
    unity_to_grid,
    world_to_egocentric,
    egocentric_to_world,
    wrap_angle,
    wrap_angle_positive,
    angle_difference,
    heading_vector,
    heading_from_vector,
)

__all__ = [
    # Constants
    'GRID_W', 'GRID_H', 'CAR_W', 'CAR_H',
    'X_MIN', 'X_MAX', 'Z_MIN', 'Z_MAX',
    'THETA_GRID', 'R_GRID',
    'HEAD_RADIUS', 'SPEED_OF_SOUND', 'SPEED_OF_SOUND_US',
    'ITD_NOISE_SCALE', 'ILD_NOISE_SCALE',
    'DEFAULT_FOV_DEG', 'DEFAULT_FOV_RAD',
    'COLOR_TO_PREFAB', 'PREFAB_TO_COLOR', 'COLOR_MAP',
    'ACTION_TURN_LEFT', 'ACTION_TURN_RIGHT', 'ACTION_MOVE_FORWARD',
    'ACTION_COMMIT', 'ACTION_STAY', 'ACTION_NAMES', 'NUM_ACTIONS',
    # Coordinates
    'grid_to_unity', 'unity_to_grid',
    'world_to_egocentric', 'egocentric_to_world',
    'wrap_angle', 'wrap_angle_positive', 'angle_difference',
    'heading_vector', 'heading_from_vector',
]
