"""
Sensonaut Analysis Package

This package contains analysis tools for comparing human and model behavior
in the audio-visual search task.

Modules:
- compare_traj: Trajectory comparison and visualization
- belief_action_comparison: Belief state and action analysis
- human_actions_and_beliefs: Human behavior inference
- plot_beliefs: Belief visualization utilities
- plot_models_vs_humans_facets: Multi-facet comparison plots
- visualize_mismatch_cases: Error case analysis
- visualize_movement: Movement heatmaps and patterns
"""

# Re-export commonly used constants and functions from utils
import sys
import os

# Add parent directory to path for utils imports
_CUR = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_CUR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.constants import (
    GRID_W, GRID_H, CAR_W, CAR_H,
    X_MIN, X_MAX, Z_MIN, Z_MAX,
    THETA_GRID, R_GRID,
    COLOR_MAP,
)

from utils.coordinates import (
    grid_to_unity,
    unity_to_grid,
    wrap_angle,
)

__all__ = [
    'GRID_W', 'GRID_H', 'CAR_W', 'CAR_H',
    'X_MIN', 'X_MAX', 'Z_MIN', 'Z_MAX',
    'THETA_GRID', 'R_GRID',
    'COLOR_MAP',
    'grid_to_unity', 'unity_to_grid', 'wrap_angle',
]
