"""在 MuJoCo 用户场景中绘制扳手和轴向局部能力范围。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike


ACTUAL_FORCE = np.array([0.90, 0.18, 0.12, 1.0], dtype=np.float32)
ACTUAL_MOMENT = np.array([0.98, 0.50, 0.10, 1.0], dtype=np.float32)
MODEL_FORCE = np.array([0.10, 0.55, 0.95, 1.0], dtype=np.float32)
MODEL_MOMENT = np.array([0.05, 0.78, 0.72, 1.0], dtype=np.float32)
RANGE_FORCE = np.array([0.20, 0.75, 0.28, 0.42], dtype=np.float32)
RANGE_MOMENT = np.array([0.20, 0.75, 0.28, 0.42], dtype=np.float32)


@dataclass(frozen=True)
class WrenchSceneOptions:
    force_scale_m_per_n: float = 0.003
    moment_scale_m_per_nm: float = 0.04
    arrow_width_m: float = 0.003
    range_width_m: float = 0.0018


def _add_arrow(scene, start, vector, rgba, width, label) -> None:
    vector = np.asarray(vector, dtype=float).reshape(3)
    length = float(np.linalg.norm(vector))
    if length <= 1e-10 or scene.ngeom >= scene.maxgeom:
        return
    direction = vector / length
    helper = (
        np.array([0.0, 0.0, 1.0])
        if abs(direction[2]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    axis_x = np.cross(helper, direction)
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(direction, axis_x)
    rotation = np.column_stack((axis_x, axis_y, direction))
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([width, width, length]),
        pos=np.asarray(start, dtype=float),
        mat=rotation.reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    geom.label = label
    scene.ngeom += 1


def add_wrench(
    scene,
    reference_world: ArrayLike,
    wrench_world: ArrayLike,
    options: WrenchSceneOptions,
    force_rgba: ArrayLike,
    moment_rgba: ArrayLike,
    label_prefix: str,
    origin_offset_world: ArrayLike | None = None,
) -> None:
    """将一个六维扳手画成相邻的合力箭头和合力矩箭头。"""
    wrench = np.asarray(wrench_world, dtype=float).reshape(6)
    start = np.asarray(reference_world, dtype=float).reshape(3)
    if origin_offset_world is not None:
        # Optional compatibility argument; the bench uses the COM directly.
        start = start + np.asarray(origin_offset_world, dtype=float).reshape(3)
    _add_arrow(
        scene,
        start,
        options.force_scale_m_per_n * wrench[3:],
        force_rgba,
        options.arrow_width_m,
        f"{label_prefix} F",
    )
    _add_arrow(
        scene,
        start,
        options.moment_scale_m_per_nm * wrench[:3],
        moment_rgba,
        options.arrow_width_m,
        f"{label_prefix} M",
    )


def add_wrench_cone(
    scene,
    reference_world: ArrayLike,
    rotation_world_from_foot: ArrayLike,
    wrench_samples_foot: ArrayLike,
    options: WrenchSceneOptions,
    force_rgba: ArrayLike = RANGE_FORCE,
    moment_rgba: ArrayLike = RANGE_MOMENT,
    label_prefix: str = "local wrench",
) -> None:
    """Draw one local-wrench envelope from complete six-dimensional samples.

    Each row is one feasible wrench.  Its resultant-force and resultant-
    moment arrows are drawn together, so the display never pairs independent
    force and moment boundary queries.
    """
    reference = np.asarray(reference_world, dtype=float).reshape(3)
    rotation = np.asarray(rotation_world_from_foot, dtype=float).reshape(3, 3)
    wrench_samples = np.asarray(wrench_samples_foot, dtype=float).reshape(-1, 6)
    for index, wrench in enumerate(wrench_samples):
        _add_arrow(
            scene,
            reference,
            options.force_scale_m_per_n * (rotation @ wrench[3:]),
            force_rgba,
            options.range_width_m,
            f"{label_prefix} {index} force",
        )
        _add_arrow(
            scene,
            reference,
            options.moment_scale_m_per_nm * (rotation @ wrench[:3]),
            moment_rgba,
            options.range_width_m,
            f"{label_prefix} {index} moment",
        )
