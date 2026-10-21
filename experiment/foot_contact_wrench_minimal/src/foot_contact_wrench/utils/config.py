"""Configuration loading for local wrench-set experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from foot_contact_wrench.contact.discrete import DiscreteContactModel
from foot_contact_wrench.geometry.grid import rectangular_cell_grid
from foot_contact_wrench.optimization.local_set import LocalWrenchSet


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream)
    if not isinstance(content, dict):
        raise ValueError("configuration root must be a mapping")
    return content


def local_set_from_config(config: dict[str, Any]) -> LocalWrenchSet:
    if "contact_grid" in config:
        grid = config["contact_grid"]
        points, cell_areas, _ = rectangular_cell_grid(
            size_m=grid["size_m"],
            resolution_m=grid["resolution_m"],
            z_m=float(grid.get("z_m", 0.0)),
        )
    else:
        points = config["contact_points_m"]
        cell_areas = config.get("cell_areas_m2")
    model = DiscreteContactModel.create(
        points=points,
        cell_areas=cell_areas,
        normal=config["normal"],
        tangent_basis=np.asarray(config["tangent_basis"], dtype=float),
        friction=config["friction_coefficients"],
        normal_force_max=config.get("normal_force_max_n"),
        active=config.get("active_mask"),
    )
    return LocalWrenchSet(
        model=model,
        normal_load=float(config["normal_load_n"]),
        solver=str(config.get("solver", "CLARABEL")),
        tolerance=float(config.get("solver_tolerance", 1e-7)),
    )
