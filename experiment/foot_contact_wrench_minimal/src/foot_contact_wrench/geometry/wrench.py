"""Geometry helpers for force-to-wrench mappings."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def skew(vector: ArrayLike) -> FloatArray:
    """Return the 3x3 cross-product matrix of a three-vector."""
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def force_to_wrench_matrix(position: ArrayLike) -> FloatArray:
    """Map a force at ``position`` to [moment; force] about the origin."""
    position_array = np.asarray(position, dtype=float).reshape(3)
    return np.vstack((skew(position_array), np.eye(3)))


def shift_wrench_reference(wrench: ArrayLike, displacement: ArrayLike) -> FloatArray:
    """Shift [moment; force] to a point displaced from the old reference.

    ``displacement`` points from the new reference point to the old one.
    """
    wrench_array = np.asarray(wrench, dtype=float).reshape(6)
    moment = wrench_array[:3] + np.cross(displacement, wrench_array[3:])
    return np.concatenate((moment, wrench_array[3:]))
