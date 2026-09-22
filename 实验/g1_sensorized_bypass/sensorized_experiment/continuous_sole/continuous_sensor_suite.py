"""Continuous-sole virtual array with cell-wise traction integration.

Unlike the earlier array logger, this module does not interpret existing small
foot geometries as taxels.  The generated model has one collision box named
``*_continuous_sole`` for each foot.  Every solved sole/terrain contact is a
quadrature sample of the continuous traction field and is accumulated into a
grid cell:

    F_ij = integral_Aij t(r) dA ~= sum_{k in Aij} f_k

The reported ``pressure_avg_pa`` is the nominal-sole-normal component of this
resultant divided by the *whole cell area*.  It is therefore the average
pressure a taxel covering that cell would report; it is not a claim that the
solver has recovered a continuous pressure density or the active contact area
inside a cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


FOOT_BODIES = {
    "left": "left_ankle_roll_link",
    "right": "right_ankle_roll_link",
}
SOLE_GEOMS = {
    "left": "left_continuous_sole",
    "right": "right_continuous_sole",
}

# These bounds exactly match the footprint of the generated thin box sole.
SOLE_X_BOUNDS_M = (-0.08, 0.15)
SOLE_Y_BOUNDS_M = (-0.06, 0.06)
ARRAY_NX = 8
ARRAY_NY = 4
CELL_WIDTH_M = (SOLE_X_BOUNDS_M[1] - SOLE_X_BOUNDS_M[0]) / ARRAY_NX
CELL_LENGTH_M = (SOLE_Y_BOUNDS_M[1] - SOLE_Y_BOUNDS_M[0]) / ARRAY_NY
CELL_AREA_M2 = CELL_WIDTH_M * CELL_LENGTH_M


@dataclass
class ContinuousFootReading:
    """Integrated continuous-sole channels for one foot at one time step."""

    cell_force_foot_n: np.ndarray
    cell_pressure_avg_pa: np.ndarray
    cell_contact_count: np.ndarray
    macro_wrench_ankle: np.ndarray
    cop_foot_m: np.ndarray
    cop_valid: int
    contacts: int
    outside_contacts: int
    outside_force_foot_n: np.ndarray
    imu_gyro_rad_s: np.ndarray
    imu_accel_m_s2: np.ndarray
    force_closure_error_n: float


def _body_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _sole_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
    if geom_id < 0:
        raise RuntimeError(f"continuous sole geom not found: {name}")
    return geom_id


def _contact_force_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_id: int,
    sole_geom_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the force and contact torque acting on the continuous sole."""

    contact = data.contact[contact_id]
    geom1, geom2 = int(contact.geom[0]), int(contact.geom[1])
    body1 = int(model.geom_bodyid[geom1])
    body2 = int(model.geom_bodyid[geom2])
    if geom1 == sole_geom_id and body2 == 0:
        sign = -1.0
    elif geom2 == sole_geom_id and body1 == 0:
        sign = 1.0
    else:
        return None

    contact_wrench = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, contact_id, contact_wrench)
    frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
    force_world = sign * frame.T @ contact_wrench[:3]
    torque_world = sign * frame.T @ contact_wrench[3:]
    return force_world, torque_world


def _imu(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    velocity_local = np.zeros(6, dtype=np.float64)
    acceleration_local = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity_local,
        1,
    )
    mujoco.mj_objectAcceleration(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        acceleration_local,
        1,
    )
    rotation_world_from_body = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    gravity_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    specific_force_local = acceleration_local[3:] - rotation_world_from_body.T @ gravity_world
    return velocity_local[:3].copy(), specific_force_local


def read_continuous_foot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot: str,
) -> ContinuousFootReading:
    """Integrate all sole/terrain contact samples into the 8×4 taxel grid."""

    if foot not in FOOT_BODIES:
        raise ValueError(f"unknown foot {foot!r}")
    foot_body_id = _body_id(model, FOOT_BODIES[foot])
    sole_geom_id = _sole_id(model, SOLE_GEOMS[foot])
    if int(model.geom_bodyid[sole_geom_id]) != foot_body_id:
        raise RuntimeError(f"{SOLE_GEOMS[foot]} is not attached to {FOOT_BODIES[foot]}")

    rotation_world_from_foot = np.asarray(data.xmat[foot_body_id], dtype=np.float64).reshape(3, 3)
    foot_origin_world = np.asarray(data.xpos[foot_body_id], dtype=np.float64)
    x_edges = np.linspace(SOLE_X_BOUNDS_M[0], SOLE_X_BOUNDS_M[1], ARRAY_NX + 1)
    y_edges = np.linspace(SOLE_Y_BOUNDS_M[0], SOLE_Y_BOUNDS_M[1], ARRAY_NY + 1)

    cell_force = np.zeros((ARRAY_NX, ARRAY_NY, 3), dtype=np.float64)
    cell_contact_count = np.zeros((ARRAY_NX, ARRAY_NY), dtype=np.int32)
    outside_force = np.zeros(3, dtype=np.float64)
    outside_contacts = 0
    cop_numerator = np.zeros(2, dtype=np.float64)
    cop_denominator = 0.0
    wrench_world = np.zeros(6, dtype=np.float64)
    contact_count = 0

    for contact_id in range(data.ncon):
        force_torque = _contact_force_world(model, data, contact_id, sole_geom_id)
        if force_torque is None:
            continue
        contact_count += 1
        force_world, torque_world_at_contact = force_torque
        position_world = np.asarray(data.contact[contact_id].pos, dtype=np.float64)
        position_foot = rotation_world_from_foot.T @ (position_world - foot_origin_world)
        force_foot = rotation_world_from_foot.T @ force_world
        x, y = float(position_foot[0]), float(position_foot[1])
        # Contact positions on a box boundary can differ from the nominal
        # edge by a few floating-point ulps.  Clamp them before binning so a
        # force physically on the sole boundary is not mislabeled as outside.
        eps = 1e-9
        x_binned = float(np.clip(x, SOLE_X_BOUNDS_M[0] + eps, SOLE_X_BOUNDS_M[1] - eps))
        y_binned = float(np.clip(y, SOLE_Y_BOUNDS_M[0] + eps, SOLE_Y_BOUNDS_M[1] - eps))
        ix = int(np.searchsorted(x_edges, x_binned, side="right") - 1)
        iy = int(np.searchsorted(y_edges, y_binned, side="right") - 1)
        if 0 <= ix < ARRAY_NX and 0 <= iy < ARRAY_NY:
            cell_force[ix, iy] += force_foot
            cell_contact_count[ix, iy] += 1
        else:
            outside_contacts += 1
            outside_force += force_foot

        # Nominal sole-normal load is used for CoP weighting.  On a sloped or
        # rough surface this intentionally remains a reported convention, not
        # an estimate of terrain-normal pressure.
        normal_load = max(float(force_foot[2]), 0.0)
        if normal_load > 0.0:
            cop_numerator += normal_load * position_foot[:2]
            cop_denominator += normal_load

        lever_arm = position_world - foot_origin_world
        wrench_world[:3] += force_world
        wrench_world[3:] += np.cross(lever_arm, force_world)
        wrench_world[3:] += torque_world_at_contact

    if cop_denominator > 1e-9:
        cop = np.array(
            [cop_numerator[0] / cop_denominator, cop_numerator[1] / cop_denominator, 0.0],
            dtype=np.float64,
        )
        cop_valid = 1
    else:
        cop = np.zeros(3, dtype=np.float64)
        cop_valid = 0

    wrench_ankle = np.empty(6, dtype=np.float64)
    wrench_ankle[:3] = rotation_world_from_foot.T @ wrench_world[:3]
    wrench_ankle[3:] = rotation_world_from_foot.T @ wrench_world[3:]
    array_force = np.sum(cell_force, axis=(0, 1)) + outside_force
    force_closure_error = float(np.linalg.norm(array_force - wrench_ankle[:3]))
    pressure_avg = np.maximum(cell_force[:, :, 2], 0.0) / CELL_AREA_M2
    gyro, accel = _imu(model, data, foot_body_id)

    return ContinuousFootReading(
        cell_force_foot_n=cell_force,
        cell_pressure_avg_pa=pressure_avg,
        cell_contact_count=cell_contact_count,
        macro_wrench_ankle=wrench_ankle,
        cop_foot_m=cop,
        cop_valid=cop_valid,
        contacts=contact_count,
        outside_contacts=outside_contacts,
        outside_force_foot_n=outside_force,
        imu_gyro_rad_s=gyro,
        imu_accel_m_s2=accel,
        force_closure_error_n=force_closure_error,
    )
