"""Discrete, coplanar point-contact model for a rigid foot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from foot_contact_wrench.geometry.wrench import force_to_wrench_matrix

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _unit(vector: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(array)
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return array / norm


@dataclass(frozen=True)
class DiscreteContactModel:
    """Fixed contact geometry and parameters for one local wrench-set slice."""

    points: FloatArray
    cell_areas: FloatArray
    normal: FloatArray
    tangent_basis: FloatArray
    friction: FloatArray
    normal_force_max: FloatArray
    active: BoolArray

    @classmethod
    def create(
        cls,
        points: ArrayLike,
        normal: ArrayLike,
        tangent_basis: ArrayLike,
        friction: ArrayLike,
        normal_force_max: ArrayLike | None = None,
        active: ArrayLike | None = None,
        cell_areas: ArrayLike | None = None,
    ) -> "DiscreteContactModel":
        points_array = np.asarray(points, dtype=float)
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            raise ValueError("points must have shape (M, 3)")
        point_count = points_array.shape[0]
        if cell_areas is None:
            area_array = np.ones(point_count)
        else:
            area_array = np.broadcast_to(
                np.asarray(cell_areas, dtype=float), (point_count,)
            ).copy()
            if np.any(area_array <= 0.0):
                raise ValueError("cell_areas must be positive")

        normal_array = _unit(normal, "normal")
        tangent_array = np.asarray(tangent_basis, dtype=float)
        if tangent_array.shape != (3, 2):
            raise ValueError("tangent_basis must have shape (3, 2)")
        gram = tangent_array.T @ tangent_array
        if not np.allclose(gram, np.eye(2), atol=1e-10):
            raise ValueError("tangent_basis columns must be orthonormal")
        if not np.allclose(tangent_array.T @ normal_array, 0.0, atol=1e-10):
            raise ValueError("tangent_basis must be orthogonal to normal")

        friction_array = np.broadcast_to(
            np.asarray(friction, dtype=float), (point_count,)
        ).copy()
        if np.any(friction_array < 0.0):
            raise ValueError("friction coefficients must be nonnegative")

        if normal_force_max is None:
            max_array = np.full(point_count, np.inf)
        else:
            max_array = np.broadcast_to(
                np.asarray(normal_force_max, dtype=float), (point_count,)
            ).copy()
            if np.any(max_array < 0.0):
                raise ValueError("normal_force_max must be nonnegative")

        if active is None:
            active_array = np.ones(point_count, dtype=bool)
        else:
            active_array = np.broadcast_to(
                np.asarray(active, dtype=bool), (point_count,)
            ).copy()
        if not np.any(active_array):
            raise ValueError("at least one contact point must be active")
        max_array[~active_array] = 0.0

        return cls(
            points=points_array,
            cell_areas=area_array,
            normal=normal_array,
            tangent_basis=tangent_array,
            friction=friction_array,
            normal_force_max=max_array,
            active=active_array,
        )

    @property
    def point_count(self) -> int:
        return self.points.shape[0]

    @property
    def wrench_maps(self) -> FloatArray:
        """Return the stacked force-to-wrench maps with shape (M, 6, 3)."""
        return np.stack([force_to_wrench_matrix(point) for point in self.points])

    def wrench(self, normal_forces: ArrayLike, tangential_forces: ArrayLike) -> FloatArray:
        """Compute the resultant [moment; force] from point-force variables."""
        normal_array = np.asarray(normal_forces, dtype=float).reshape(self.point_count)
        tangent_array = np.asarray(tangential_forces, dtype=float).reshape(
            self.point_count, 2
        )
        point_forces = (
            normal_array[:, None] * self.normal
            + tangent_array @ self.tangent_basis.T
        )
        return np.einsum("mij,mj->i", self.wrench_maps, point_forces)

    def validate_load(self, normal_load: float, tolerance: float = 1e-10) -> None:
        """Raise when the fixed normal-load slice is empty by construction."""
        if normal_load < -tolerance:
            raise ValueError("normal_load must be nonnegative")
        capacity = float(np.sum(self.normal_force_max[self.active]))
        if np.isfinite(capacity) and normal_load > capacity + tolerance:
            raise ValueError(
                f"normal_load={normal_load} exceeds active-point capacity={capacity}"
            )
