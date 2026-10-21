"""局部集合轴向边界与扳手误差计算。"""

from __future__ import annotations

import numpy as np


def axial_support_bounds(local_set) -> tuple[np.ndarray, np.ndarray]:
    """返回扳手六个坐标分量的理论下界与上界。"""
    lower = np.zeros(6)
    upper = np.zeros(6)
    for axis in range(6):
        direction = np.zeros(6)
        direction[axis] = 1.0
        positive = local_set.support(direction)
        negative = local_set.support(-direction)
        if not positive.feasible or not negative.feasible:
            raise RuntimeError(f"support query failed on wrench axis {axis}")
        upper[axis] = positive.wrench[axis]
        lower[axis] = negative.wrench[axis]
    return lower, upper


def spherical_directions(count: int = 32) -> np.ndarray:
    """Return approximately uniform unit directions on the 3-D sphere."""
    if count < 4:
        raise ValueError("count must be at least 4")
    index = np.arange(count, dtype=float) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * index
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))


def support_wrench_samples(local_set, count: int = 32) -> np.ndarray:
    """Return complete six-dimensional boundary-wrench witnesses.

    Half the queries maximize directions in resultant-force space and half in
    resultant-moment space.  This avoids mixing force and moment units in one
    arbitrary six-dimensional norm.  Each row remains one complete feasible
    wrench, so its force and moment components must be displayed together.
    """
    if count < 8:
        raise ValueError("count must be at least 8 so both projections have four directions")
    force_count = count // 2
    moment_count = count - force_count
    query_directions = []
    for direction in spherical_directions(force_count):
        query_directions.append(np.concatenate((np.zeros(3), direction)))
    for direction in spherical_directions(moment_count):
        query_directions.append(np.concatenate((direction, np.zeros(3))))
    samples = []
    for direction in query_directions:
        result = local_set.support(direction)
        if not result.feasible or result.wrench is None:
            raise RuntimeError("six-dimensional support query failed")
        samples.append(result.wrench)
    return np.asarray(samples, dtype=float)
