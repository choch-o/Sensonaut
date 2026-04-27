"""
Sensonaut - Coordinate Transformation Utilities

This module provides consistent coordinate transformations between:
- Grid coordinates (discrete cell-based, used internally)
- Unity world coordinates (continuous 3D space)
- Egocentric polar coordinates (r, theta relative to agent)

Coordinate Systems:
------------------
Grid: (gx, gy) where (0,0) is top-left, x increases right, y increases down
Unity: (ux, uz) where x,z are Unity world coordinates (y is up/height)
Polar: (r, theta) where r is distance, theta is angle from forward (-pi to pi)
"""
import math
import numpy as np
from typing import Tuple

from .constants import GRID_W, GRID_H, X_MIN, X_MAX, Z_MIN, Z_MAX


def grid_to_unity(gx: float, gy: float,
                  grid_w: int = GRID_W, grid_h: int = GRID_H,
                  x_min: float = X_MIN, x_max: float = X_MAX,
                  z_min: float = Z_MIN, z_max: float = Z_MAX) -> Tuple[float, float]:
    """
    Convert grid coordinates to Unity world coordinates.

    Grid (0,0) maps to (x_min, z_max) - top-left
    Grid (grid_w-1, grid_h-1) maps to (x_max, z_min) - bottom-right

    Args:
        gx: Grid x coordinate (column)
        gy: Grid y coordinate (row)
        grid_w: Grid width (default from constants)
        grid_h: Grid height (default from constants)
        x_min, x_max, z_min, z_max: Unity coordinate bounds

    Returns:
        (ux, uz): Unity world coordinates
    """
    gw = max(1, grid_w - 1)
    gh = max(1, grid_h - 1)

    # Normalize to [0, 1]
    nx = float(gx) / gw
    ny = float(gy) / gh

    # Linear map to Unity space
    ux = x_min + nx * (x_max - x_min)
    uz = z_max + ny * (z_min - z_max)  # Note: z decreases as grid y increases

    return ux, uz


def unity_to_grid(ux: float, uz: float,
                  grid_w: int = GRID_W, grid_h: int = GRID_H,
                  x_min: float = X_MIN, x_max: float = X_MAX,
                  z_min: float = Z_MIN, z_max: float = Z_MAX,
                  clamp: bool = True) -> Tuple[float, float]:
    """
    Convert Unity world coordinates to grid coordinates.

    Inverse of grid_to_unity().

    Args:
        ux: Unity x coordinate
        uz: Unity z coordinate
        grid_w: Grid width (default from constants)
        grid_h: Grid height (default from constants)
        x_min, x_max, z_min, z_max: Unity coordinate bounds
        clamp: If True, clamp output to valid grid bounds

    Returns:
        (gx, gy): Grid coordinates (may be fractional)
    """
    gw = max(1, grid_w - 1)
    gh = max(1, grid_h - 1)

    # Normalize from Unity space to [0, 1]
    nx = (ux - x_min) / (x_max - x_min)
    ny = (uz - z_max) / (z_min - z_max)  # Note: reversed order

    # Map to grid coordinates
    gx = nx * gw
    gy = ny * gh

    if clamp:
        gx = max(0, min(grid_w - 1, gx))
        gy = max(0, min(grid_h - 1, gy))

    return gx, gy


def world_to_egocentric(wx: float, wy: float,
                        agent_x: float, agent_y: float,
                        agent_heading: float) -> Tuple[float, float]:
    """
    Convert world (grid) coordinates to agent-relative polar coordinates.

    Args:
        wx, wy: World position (grid coordinates)
        agent_x, agent_y: Agent position (grid coordinates)
        agent_heading: Agent heading in radians (0 = north/up)

    Returns:
        (r, theta): Distance and relative bearing
            r: Distance from agent to target
            theta: Relative bearing in radians, wrapped to (-pi, pi]
                   0 = directly ahead, negative = left, positive = right
    """
    dx = float(wx) - float(agent_x)
    dy = float(wy) - float(agent_y)

    r = math.hypot(dx, dy)

    # Absolute angle: atan2(dx, -dy) gives angle from north (up)
    theta_abs = math.atan2(dx, -dy) % (2 * math.pi)

    # Relative angle: subtract agent heading and wrap to (-pi, pi]
    theta_rel = (theta_abs - agent_heading + math.pi) % (2 * math.pi) - math.pi

    return r, theta_rel


def egocentric_to_world(r: float, theta: float,
                        agent_x: float, agent_y: float,
                        agent_heading: float,
                        grid_w: int = GRID_W, grid_h: int = GRID_H,
                        clamp: bool = True) -> Tuple[float, float]:
    """
    Convert agent-relative polar coordinates to world (grid) coordinates.

    Inverse of world_to_egocentric().

    Args:
        r: Distance from agent
        theta: Relative bearing in radians (0 = ahead)
        agent_x, agent_y: Agent position (grid coordinates)
        agent_heading: Agent heading in radians (0 = north/up)
        grid_w, grid_h: Grid dimensions for clamping
        clamp: If True, clamp output to valid grid bounds

    Returns:
        (wx, wy): World position (grid coordinates)
    """
    # Convert relative angle to absolute
    abs_angle = agent_heading + theta

    # Calculate offset from agent
    dx = r * math.sin(abs_angle)
    dy = -r * math.cos(abs_angle)

    wx = agent_x + dx
    wy = agent_y + dy

    if clamp:
        wx = max(0, min(grid_w - 1, wx))
        wy = max(0, min(grid_h - 1, wy))

    return wx, wy


def wrap_angle(angle: float) -> float:
    """
    Wrap angle to (-pi, pi].

    Args:
        angle: Angle in radians

    Returns:
        Wrapped angle in range (-pi, pi]
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def wrap_angle_positive(angle: float) -> float:
    """
    Wrap angle to [0, 2*pi).

    Args:
        angle: Angle in radians

    Returns:
        Wrapped angle in range [0, 2*pi)
    """
    return angle % (2 * math.pi)


def angle_difference(a1: float, a2: float) -> float:
    """
    Compute the signed angular difference between two angles.

    Args:
        a1: First angle in radians
        a2: Second angle in radians

    Returns:
        Signed difference (a1 - a2) wrapped to (-pi, pi]
    """
    return wrap_angle(a1 - a2)


def heading_vector(heading: float) -> np.ndarray:
    """
    Convert heading angle to a unit direction vector.

    Convention: heading=0 points north (up, negative y in grid)

    Args:
        heading: Heading in radians

    Returns:
        numpy array [dx, dy] representing unit direction
    """
    dx = math.sin(heading)
    dy = -math.cos(heading)
    return np.array([dx, dy])


def heading_from_vector(dx: float, dy: float) -> float:
    """
    Convert a direction vector to heading angle.

    Args:
        dx: X component of direction
        dy: Y component of direction

    Returns:
        Heading in radians [0, 2*pi)
    """
    return math.atan2(dx, -dy) % (2 * math.pi)
