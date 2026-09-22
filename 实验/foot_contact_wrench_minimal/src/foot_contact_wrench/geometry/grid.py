"""规则矩形足底的有限面积离散。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def rectangular_cell_grid(
    size_m: tuple[float, float] | list[float],
    resolution_m: float | tuple[float, float] | list[float],
    z_m: float = 0.0,
) -> tuple[FloatArray, FloatArray, tuple[int, int]]:
    """返回单元中心、单元面积和网格形状 ``(ny, nx)``。"""
    size = np.asarray(size_m, dtype=float).reshape(2)
    resolution = np.broadcast_to(np.asarray(resolution_m, dtype=float), (2,)).copy()
    if np.any(size <= 0.0) or np.any(resolution <= 0.0):
        raise ValueError("grid size and resolution must be positive")
    counts_float = size / resolution
    counts = np.rint(counts_float).astype(int)
    if not np.allclose(counts_float, counts, atol=1e-10):
        raise ValueError("foot size must be divisible by grid resolution")
    nx, ny = map(int, counts)
    x = -0.5 * size[0] + (np.arange(nx) + 0.5) * resolution[0]
    y = -0.5 * size[1] + (np.arange(ny) + 0.5) * resolution[1]
    points = np.asarray([[x_i, y_i, z_m] for y_i in y for x_i in x])
    areas = np.full(nx * ny, resolution[0] * resolution[1])
    return points, areas, (ny, nx)
