"""MuJoCo helpers for the simplified single-foot digital bench."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ContactWrenchSample:
    """Ground-on-foot wrench expressed at the foot COM."""

    wrench_world: FloatArray
    wrench_foot: FloatArray
    pressure_model_wrench_world: FloatArray
    pressure_model_wrench_foot: FloatArray
    com_position_world: FloatArray
    normal_forces: FloatArray
    normal_force_names: tuple[str, ...]
    contact_constraint_count: int
    contact_count: int

def load_bench_model(path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    return model, data


def initialize_free_foot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_position_world_m: NDArray[np.float64],
    initial_linear_velocity_world_mps: NDArray[np.float64] | None = None,
    initial_angular_velocity_world_radps: NDArray[np.float64] | None = None,
) -> None:
    """Reset the free-foot bench to a world-frame pose and velocity state."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[3] = 1.0
    data.qpos[:3] = np.asarray(initial_position_world_m, dtype=float).reshape(3)
    if initial_linear_velocity_world_mps is not None:
        data.qvel[:3] = np.asarray(
            initial_linear_velocity_world_mps, dtype=float
        ).reshape(3)
    if initial_angular_velocity_world_radps is not None:
        data.qvel[3:6] = np.asarray(
            initial_angular_velocity_world_radps, dtype=float
        ).reshape(3)
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def apply_world_wrench_at_com(
    data: mujoco.MjData, body_id: int, wrench_moment_force: NDArray[np.float64]
) -> None:
    """Apply world-frame ``[moment; force]`` at a body's center of mass."""
    wrench = np.asarray(wrench_moment_force, dtype=float).reshape(6)
    data.xfrc_applied[body_id, :3] = wrench[3:]
    data.xfrc_applied[body_id, 3:] = wrench[:3]


def _contact_force_world(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> FloatArray:
    force_contact = np.zeros(6)
    mujoco.mj_contactForce(model, data, index, force_contact)
    # Use the batched contact arrays. Direct scalar-contact frame access can
    # expose an unpopulated view in the Python binding.
    rotation = np.asarray(data.contact.frame[index], dtype=float).reshape(3, 3)
    force_world = rotation.T @ force_contact[:3]
    torque_world = rotation.T @ force_contact[3:]
    return np.concatenate((torque_world, force_world))


def contact_wrench_about_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    ground_geom_name: str = "ground",
    contact_geom_prefix: str = "contact_",
    contact_plane_site_name: str = "contact_plane_center",
    support_geom_prefix: str = "",
    normal_world: NDArray[np.float64] | None = None,
) -> ContactWrenchSample:
    """Sum ground-on-foot contacts and express the result at the foot COM.

    MuJoCo's contact frame normal is oriented according to the geom ordering;
    the sign is corrected from body/ground ordering below before accumulation.
    ``contact_plane_site_name`` identifies the contact plane used by the
    pressure reconstruction; it is not the wrench reference point.
    """
    body_id = model.body(body_name).id
    contact_plane_site_id = model.site(contact_plane_site_name).id
    ground_id = model.geom(ground_geom_name).id
    contact_wrench = np.zeros(6)
    contact_geom_names = tuple(
        model.geom(geom_id).name
        for geom_id in range(model.ngeom)
        if model.geom(geom_id).name.startswith(contact_geom_prefix)
    )
    point_index_by_name = {
        name: index for index, name in enumerate(contact_geom_names)
    }
    normal_forces = np.zeros(len(contact_geom_names))
    contact_constraint_count = 0
    com_position = data.xipos[body_id].copy()
    rotation_world_from_foot = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    foot_normal_world = rotation_world_from_foot[:, 2]
    support_normal_world = foot_normal_world if normal_world is None else np.asarray(
        normal_world, dtype=float
    ).reshape(3).copy()
    support_normal_norm = np.linalg.norm(support_normal_world)
    if support_normal_norm <= 1e-12:
        raise ValueError("normal_world must be nonzero")
    support_normal_world /= support_normal_norm

    def is_support_geom(geom_id: int) -> bool:
        name = model.geom(geom_id).name
        return geom_id == ground_id or (
            bool(support_geom_prefix) and name.startswith(support_geom_prefix)
        )

    for index in range(data.ncon):
        geom1 = int(data.contact.geom1[index])
        geom2 = int(data.contact.geom2[index])
        foot_geom = None
        # mj_contactForce returns the contact force on geom1 expressed in the
        # contact frame. We want ground-on-foot, so the sign depends on which
        # geom is the foot.
        sign = -1.0
        if is_support_geom(geom2) and model.geom(geom1).name.startswith(contact_geom_prefix):
            foot_geom = geom1
        elif is_support_geom(geom1) and model.geom(geom2).name.startswith(contact_geom_prefix):
            foot_geom = geom2
            sign = 1.0
        if foot_geom is None:
            continue

        contact_point = np.asarray(data.contact.pos[index], dtype=float).copy()
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force_contact)
        rotation = np.asarray(data.contact.frame[index], dtype=float).reshape(3, 3)
        force_world = sign * rotation.T @ force_contact[:3]
        torque_world = sign * rotation.T @ force_contact[3:]
        arm = contact_point - com_position
        contact_wrench[:3] += np.cross(arm, force_world) + torque_world
        contact_wrench[3:] += force_world
        point_index = point_index_by_name[model.geom(foot_geom).name]
        normal_forces[point_index] += float(
            max(0.0, force_world @ support_normal_world)
        )
        contact_constraint_count += 1

    contact_count = int(np.count_nonzero(normal_forces > 1e-10))
    pressure_model_wrench_foot = np.zeros(6)
    com_position_foot = np.asarray(model.body_ipos[body_id], dtype=float)
    contact_plane_position_foot = np.asarray(
        model.site_pos[contact_plane_site_id], dtype=float
    )
    for name, normal_force in zip(contact_geom_names, normal_forces):
        geom_id = model.geom(name).id
        arm_foot = np.array(
            [model.geom_pos[geom_id, 0], model.geom_pos[geom_id, 1], contact_plane_position_foot[2]]
        ) - com_position_foot
        force_foot = normal_force * (rotation_world_from_foot.T @ support_normal_world)
        pressure_model_wrench_foot[:3] += np.cross(arm_foot, force_foot)
        pressure_model_wrench_foot[3:] += force_foot

    wrench_foot = np.concatenate(
        (
            rotation_world_from_foot.T @ contact_wrench[:3],
            rotation_world_from_foot.T @ contact_wrench[3:],
        )
    )
    pressure_model_wrench_world = np.concatenate(
        (
            rotation_world_from_foot @ pressure_model_wrench_foot[:3],
            rotation_world_from_foot @ pressure_model_wrench_foot[3:],
        )
    )

    return ContactWrenchSample(
        wrench_world=contact_wrench,
        wrench_foot=wrench_foot,
        pressure_model_wrench_world=pressure_model_wrench_world,
        pressure_model_wrench_foot=pressure_model_wrench_foot,
        com_position_world=com_position,
        normal_forces=normal_forces,
        normal_force_names=contact_geom_names,
        contact_constraint_count=contact_constraint_count,
        contact_count=contact_count,
    )


def run_free_body_bench(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration_s: float,
    initial_position: FloatArray,
    initial_linear_velocity: FloatArray | None = None,
    initial_angular_velocity: FloatArray | None = None,
) -> list[dict[str, float | int | list[float]]]:
    """Run the free foot under gravity and record contact truth and IMU state."""
    initialize_free_foot(
        model,
        data,
        np.asarray(initial_position, dtype=float),
        None if initial_linear_velocity is None else np.asarray(initial_linear_velocity, dtype=float),
        None if initial_angular_velocity is None else np.asarray(initial_angular_velocity, dtype=float),
    )

    records: list[dict[str, float | int | list[float]]] = []
    steps = int(np.ceil(duration_s / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        sample = contact_wrench_about_body(model, data, "foot")
        body_id = model.body("foot").id
        records.append(
            {
                "time_s": float(data.time),
                "contact_count": sample.contact_count,
                "contact_constraint_count": sample.contact_constraint_count,
                "normal_forces_n": sample.normal_forces.tolist(),
                "normal_force_names": list(sample.normal_force_names),
                "contact_wrench_world": sample.wrench_world.tolist(),
                "contact_wrench_foot": sample.wrench_foot.tolist(),
                "pressure_model_wrench_world": sample.pressure_model_wrench_world.tolist(),
                "pressure_model_wrench_foot": sample.pressure_model_wrench_foot.tolist(),
                "com_position_world_m": sample.com_position_world.tolist(),
                "position_world_m": data.xpos[body_id].tolist(),
                "linear_velocity_world_mps": data.cvel[body_id, 3:].tolist(),
                "angular_velocity_world_radps": data.cvel[body_id, :3].tolist(),
            }
        )
    return records
