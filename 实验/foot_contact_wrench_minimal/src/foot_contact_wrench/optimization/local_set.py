"""Convex queries for a fixed-load local contact wrench set."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from foot_contact_wrench.contact.discrete import DiscreteContactModel

FloatArray = NDArray[np.float64]
OPTIMAL_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


@dataclass(frozen=True)
class WrenchSolution:
    """A wrench-set query result and its feasible contact-force witness."""

    feasible: bool
    status: str
    wrench: FloatArray | None
    normal_forces: FloatArray | None
    tangential_forces: FloatArray | None
    objective: float | None = None


class LocalWrenchSet:
    """Fixed-parameter, fixed-normal-load local wrench-set slice."""

    def __init__(
        self,
        model: DiscreteContactModel,
        normal_load: float,
        solver: str = "CLARABEL",
        tolerance: float = 1e-7,
    ) -> None:
        model.validate_load(normal_load, tolerance)
        self.model = model
        self.normal_load = float(normal_load)
        self.solver = solver
        self.tolerance = float(tolerance)

    def _variables_and_constraints(
        self,
    ) -> tuple[cp.Variable, cp.Variable, cp.Expression, list[cp.Constraint]]:
        point_count = self.model.point_count
        normal_forces = cp.Variable(point_count, name="normal_forces")
        tangential_forces = cp.Variable((point_count, 2), name="tangential_forces")

        force_matrix = (
            cp.reshape(normal_forces, (point_count, 1), order="C")
            @ self.model.normal.reshape(1, 3)
            + tangential_forces @ self.model.tangent_basis.T
        )
        wrench = sum(
            self.model.wrench_maps[index] @ force_matrix[index, :]
            for index in range(point_count)
        )

        constraints: list[cp.Constraint] = [
            normal_forces >= 0.0,
            cp.sum(normal_forces) == self.normal_load,
        ]
        finite = np.isfinite(self.model.normal_force_max)
        if np.any(finite):
            constraints.append(
                normal_forces[finite] <= self.model.normal_force_max[finite]
            )
        inactive = ~self.model.active
        if np.any(inactive):
            constraints.extend(
                [normal_forces[inactive] == 0.0, tangential_forces[inactive, :] == 0.0]
            )
        for index in np.flatnonzero(self.model.active):
            constraints.append(
                cp.norm(tangential_forces[index, :], 2)
                <= self.model.friction[index] * normal_forces[index]
            )
        return normal_forces, tangential_forces, wrench, constraints

    def _solve(self, problem: cp.Problem) -> None:
        options: dict[str, float | bool] = {"verbose": False}
        if self.solver == "CLARABEL":
            options.update(
                {
                    "tol_gap_abs": self.tolerance,
                    "tol_feas": self.tolerance,
                }
            )
        elif self.solver == "SCS":
            options.update({"eps": self.tolerance})
        problem.solve(solver=self.solver, **options)

    @staticmethod
    def _result(
        problem: cp.Problem,
        wrench: cp.Expression,
        normal_forces: cp.Variable,
        tangential_forces: cp.Variable,
        objective: float | None = None,
    ) -> WrenchSolution:
        feasible = problem.status in OPTIMAL_STATUSES
        if not feasible:
            return WrenchSolution(False, problem.status, None, None, None, None)
        return WrenchSolution(
            True,
            problem.status,
            np.asarray(wrench.value, dtype=float).reshape(6),
            np.asarray(normal_forces.value, dtype=float).reshape(-1),
            np.asarray(tangential_forces.value, dtype=float),
            objective,
        )

    def support(self, direction: ArrayLike) -> WrenchSolution:
        """Maximize ``direction @ wrench`` and return the boundary witness."""
        direction_array = np.asarray(direction, dtype=float).reshape(6)
        if np.linalg.norm(direction_array) <= 0.0:
            raise ValueError("support direction must be nonzero")
        normal_forces, tangential_forces, wrench, constraints = (
            self._variables_and_constraints()
        )
        objective = cp.Maximize(direction_array @ wrench)
        problem = cp.Problem(objective, constraints)
        self._solve(problem)
        value = float(problem.value) if problem.status in OPTIMAL_STATUSES else None
        return self._result(
            problem, wrench, normal_forces, tangential_forces, value
        )

    def support_reduced(self, direction: ArrayLike) -> float:
        """Support value after analytically eliminating each tangent disk.

        This is the linear-program form of the support equation in the
        derivation. It is useful as an independent cross-check of the SOCP.
        """
        direction_array = np.asarray(direction, dtype=float).reshape(6)
        if np.linalg.norm(direction_array) <= 0.0:
            raise ValueError("support direction must be nonzero")
        coefficients = np.empty(self.model.point_count)
        for index, mapping in enumerate(self.model.wrench_maps):
            q = mapping.T @ direction_array
            coefficients[index] = (
                float(q @ self.model.normal)
                + self.model.friction[index]
                * np.linalg.norm(self.model.tangent_basis.T @ q)
            )
        normal_forces = cp.Variable(self.model.point_count)
        constraints: list[cp.Constraint] = [
            normal_forces >= 0.0,
            cp.sum(normal_forces) == self.normal_load,
        ]
        finite = np.isfinite(self.model.normal_force_max)
        if np.any(finite):
            constraints.append(
                normal_forces[finite] <= self.model.normal_force_max[finite]
            )
        inactive = ~self.model.active
        if np.any(inactive):
            constraints.append(normal_forces[inactive] == 0.0)
        problem = cp.Problem(cp.Maximize(coefficients @ normal_forces), constraints)
        self._solve(problem)
        if problem.status not in OPTIMAL_STATUSES:
            raise RuntimeError(f"reduced support query failed: {problem.status}")
        return float(problem.value)

    def contains(self, target_wrench: ArrayLike) -> WrenchSolution:
        """Test exact set membership up to the configured numerical tolerance."""
        target = np.asarray(target_wrench, dtype=float).reshape(6)
        normal_forces, tangential_forces, wrench, constraints = (
            self._variables_and_constraints()
        )
        constraints.append(cp.norm(wrench - target, 2) <= self.tolerance)
        problem = cp.Problem(cp.Minimize(0.0), constraints)
        self._solve(problem)
        result = self._result(problem, wrench, normal_forces, tangential_forces)
        if result.feasible and result.wrench is not None:
            residual = np.linalg.norm(result.wrench - target)
            if residual > 10.0 * self.tolerance:
                return WrenchSolution(False, "residual_too_large", None, None, None)
        return result

    def radial_capacity(
        self, working_wrench: ArrayLike, direction: ArrayLike
    ) -> WrenchSolution:
        """Find the nonnegative boundary distance from a feasible working wrench."""
        working = np.asarray(working_wrench, dtype=float).reshape(6)
        direction_array = np.asarray(direction, dtype=float).reshape(6)
        if np.linalg.norm(direction_array) <= 0.0:
            raise ValueError("radial direction must be nonzero")
        if not self.contains(working).feasible:
            raise ValueError("working_wrench must belong to the local wrench set")

        normal_forces, tangential_forces, wrench, constraints = (
            self._variables_and_constraints()
        )
        alpha = cp.Variable(nonneg=True, name="alpha")
        constraints.append(wrench == working + alpha * direction_array)
        problem = cp.Problem(cp.Maximize(alpha), constraints)
        self._solve(problem)
        value = float(alpha.value) if problem.status in OPTIMAL_STATUSES else None
        return self._result(
            problem, wrench, normal_forces, tangential_forces, value
        )
