"""Continuous partial-support geometry and quasi-static loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from foot_contact_wrench.contact.discrete import DiscreteContactModel
from foot_contact_wrench.optimization.local_set import LocalWrenchSet

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LoadingState:
    """One point on the prescribed quasi-static loading trajectory."""

    phase: str
    phase_progress: float
    path_progress: float
    normal_force_n: float
    application_point_foot_m: FloatArray
    complete: bool = False


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not safe:
        raise ValueError("scenario name must contain at least one safe character")
    return safe


def _support_boxes(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    boxes = scenario.get("support_boxes")
    if not isinstance(boxes, list) or not boxes:
        raise ValueError(f"scenario {scenario.get('name')} needs support_boxes")
    for box in boxes:
        center = np.asarray(box["center_xy_m"], dtype=float).reshape(2)
        size = np.asarray(box["size_xy_m"], dtype=float).reshape(2)
        if not np.all(np.isfinite(center)) or np.any(size <= 0.0):
            raise ValueError("support box center must be finite and size must be positive")
    return boxes


def build_partial_contact_xml(
    base_xml_path: str | Path,
    output_xml_path: str | Path,
    scenarios: list[dict[str, Any]],
    support_geom_prefix: str = "support_surface_",
    support_height_m: float = 0.01,
    support_friction: ArrayLike = (0.5, 0.005, 0.0001),
) -> Path:
    """Add continuous, coplanar support patches for all experiment scenarios.

    The support patches belong to the simulated environment and are independent
    of the foot's discrete pressure cells. Only one scenario is enabled at run
    time; the scenario identity is never passed to the local-set estimator.
    """
    if support_height_m <= 0.0:
        raise ValueError("support_height_m must be positive")
    friction = np.asarray(support_friction, dtype=float).reshape(3)
    if np.any(friction < 0.0):
        raise ValueError("support_friction must be nonnegative")

    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("base MuJoCo model must contain worldbody")
    for geom in worldbody.findall("geom"):
        if geom.attrib.get("name") == "ground":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("rgba", "0.7 0.7 0.7 0.04")

    seen_names: set[str] = set()
    for scenario in scenarios:
        scenario_name = _safe_name(scenario["name"])
        if scenario_name in seen_names:
            raise ValueError(f"duplicate scenario name: {scenario_name}")
        seen_names.add(scenario_name)
        for index, box in enumerate(_support_boxes(scenario)):
            center = np.asarray(box["center_xy_m"], dtype=float).reshape(2)
            size = np.asarray(box["size_xy_m"], dtype=float).reshape(2)
            worldbody.append(
                ET.Element(
                    "geom",
                    {
                        "name": f"{support_geom_prefix}{scenario_name}_{index:02d}",
                        "type": "box",
                        "pos": (
                            f"{center[0]:.9g} {center[1]:.9g} "
                            f"{-0.5 * support_height_m:.9g}"
                        ),
                        "size": (
                            f"{0.5 * size[0]:.9g} {0.5 * size[1]:.9g} "
                            f"{0.5 * support_height_m:.9g}"
                        ),
                        "friction": " ".join(f"{value:.9g}" for value in friction),
                        "condim": "3",
                        # Keep every patch in MuJoCo's compiled collision-pair
                        # candidates. ``set_support_scenario`` disables all
                        # nonselected patches immediately after model loading.
                        "contype": "1",
                        "conaffinity": "1",
                        "rgba": "0.72 0.55 0.18 0.05",
                    },
                )
            )

    output_path = Path(output_xml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return output_path


def set_support_scenario(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario_name: str,
    support_geom_prefix: str = "support_surface_",
) -> int:
    """Enable one continuous support geometry in the simulation environment."""
    selected_prefix = f"{support_geom_prefix}{_safe_name(scenario_name)}_"
    selected_count = 0
    available_count = 0
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if not name.startswith(support_geom_prefix):
            continue
        available_count += 1
        enabled = name.startswith(selected_prefix)
        model.geom_contype[geom_id] = 1 if enabled else 0
        model.geom_conaffinity[geom_id] = 1 if enabled else 0
        model.geom_rgba[geom_id, 3] = 0.78 if enabled else 0.025
        selected_count += int(enabled)
    if available_count == 0:
        raise ValueError("model contains no partial-support surfaces")
    if selected_count == 0:
        raise ValueError(f"unknown support scenario: {scenario_name}")
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    return selected_count


def pressure_active_mask(
    normal_forces: ArrayLike, threshold_n: float = 1e-3
) -> BoolArray:
    """Infer loaded pressure cells without reading environment geometry."""
    if threshold_n < 0.0:
        raise ValueError("threshold_n must be nonnegative")
    forces = np.asarray(normal_forces, dtype=float).reshape(-1)
    if np.any(forces < -1e-10):
        raise ValueError("normal forces must be nonnegative")
    return forces > float(threshold_n)


def observed_normal_wrench_set(
    base_config: dict[str, Any],
    points_foot: ArrayLike,
    normal_forces: ArrayLike,
    normal_foot: ArrayLike,
    active_force_threshold_n: float = 1e-3,
) -> LocalWrenchSet | None:
    """Build the current normal-contact set from pressure data alone.

    This first-stage set represents the observable ``[Mx, My, Fz]`` slice.
    Tangential forces are fixed to zero, so an unverified friction coefficient
    cannot enlarge the reported roll/pitch capability. Loads below the small
    threshold are discarded only to remove contact-solver numerical residue.
    """
    points = np.asarray(points_foot, dtype=float).reshape(-1, 3)
    forces = np.asarray(normal_forces, dtype=float).reshape(-1)
    if len(forces) != len(points):
        raise ValueError("normal force count must match contact point count")
    active = pressure_active_mask(forces, active_force_threshold_n)
    if not np.any(active):
        return None
    normal_load = float(np.sum(forces[active]))
    if normal_load <= 0.0:
        return None

    model = DiscreteContactModel.create(
        points=points,
        normal=normal_foot,
        tangent_basis=np.asarray(base_config["tangent_basis"], dtype=float),
        friction=0.0,
        normal_force_max=base_config.get("normal_force_max_n"),
        active=active,
    )
    try:
        return LocalWrenchSet(
            model=model,
            normal_load=normal_load,
            solver=str(base_config.get("solver", "CLARABEL")),
            tolerance=max(float(base_config.get("solver_tolerance", 1e-7)), 1e-6),
        )
    except ValueError:
        # The measured load violates the assumed per-cell capacity. In that
        # case there is no valid set under the current theoretical assumptions.
        return None


def loading_state(
    time_s: float, experiment_config: dict[str, Any], scenario: dict[str, Any]
) -> LoadingState:
    """Evaluate settle, preload, reference-hold, and sweep phases."""
    settle = float(experiment_config["settle_duration_s"])
    preload = float(experiment_config["preload_ramp_duration_s"])
    hold = float(experiment_config["reference_hold_duration_s"])
    sweep = float(experiment_config["sweep_duration_s"])
    if min(settle, preload, hold, sweep) < 0.0 or sweep <= 0.0:
        raise ValueError("phase durations must be nonnegative and sweep must be positive")

    load_path = scenario["load_path"]
    start_xy = np.asarray(load_path["start_xy_m"], dtype=float).reshape(2)
    end_xy = np.asarray(load_path["end_xy_m"], dtype=float).reshape(2)
    height = float(experiment_config.get("load_application_height_m", 0.025))
    target_force = float(experiment_config["applied_normal_force_n"])
    if target_force < 0.0:
        raise ValueError("applied_normal_force_n must be nonnegative")

    t = max(0.0, float(time_s))
    if t < settle:
        return LoadingState("settle", t / settle if settle else 1.0, 0.0, 0.0, np.r_[start_xy, height])
    t -= settle
    if t < preload:
        progress = t / preload if preload else 1.0
        return LoadingState("preload", progress, 0.0, target_force * progress, np.r_[start_xy, height])
    t -= preload
    if t < hold:
        return LoadingState("reference", t / hold if hold else 1.0, 0.0, target_force, np.r_[start_xy, height])
    t -= hold
    if t < sweep:
        progress = t / sweep
        point = (1.0 - progress) * start_xy + progress * end_xy
        return LoadingState("sweep", progress, progress, target_force, np.r_[point, height])
    return LoadingState("complete", 1.0, 1.0, target_force, np.r_[end_xy, height], True)


def vertical_point_load_wrench_world(
    data: mujoco.MjData,
    body_id: int,
    application_point_foot_m: ArrayLike,
    downward_force_n: float,
) -> FloatArray:
    """Return a vertical point load as an equivalent COM wrench in world axes."""
    if downward_force_n < 0.0:
        raise ValueError("downward_force_n must be nonnegative")
    rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    arm_world = rotation @ np.asarray(application_point_foot_m, dtype=float).reshape(3)
    force_world = np.array([0.0, 0.0, -float(downward_force_n)])
    return np.concatenate((np.cross(arm_world, force_world), force_world))


def tilt_angle_rad(data: mujoco.MjData, body_id: int) -> float:
    """Return the angle between foot and world normals."""
    rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    cosine = float(np.clip(rotation[:, 2] @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
    return float(np.arccos(cosine))


def normal_path_wrench_direction(
    start_point_foot_m: ArrayLike,
    end_point_foot_m: ArrayLike,
    applied_normal_force_n: float,
) -> tuple[FloatArray, float]:
    """Return unit contact-wrench direction and full path magnitude.

    Moving a downward external point load changes the equilibrium ground wrench
    by the negative of the external moment change.
    """
    start = np.asarray(start_point_foot_m, dtype=float).reshape(3)
    end = np.asarray(end_point_foot_m, dtype=float).reshape(3)
    force = np.array([0.0, 0.0, -float(applied_normal_force_n)])
    external_change = np.cross(end - start, force)
    contact_change = np.concatenate((-external_change, np.zeros(3)))
    magnitude = float(np.linalg.norm(contact_change))
    if magnitude <= 0.0:
        raise ValueError("load path must create a nonzero contact-moment change")
    return contact_change / magnitude, magnitude
