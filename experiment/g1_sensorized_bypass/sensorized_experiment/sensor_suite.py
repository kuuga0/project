"""Passive sensor channels for the G1 walking experiment.

The controller is deliberately not changed by this module.  It reads the
MuJoCo state after a physics step and exposes three different views of the
same ground contact:

* a virtual configurable plantar array (the current experiment uses 8 x 4),
  with a triaxial force for every cell;
* the ground-contact wrench transported to the ankle-roll body origin;
* an ankle IMU (local angular velocity and specific force).

The array is a software instrument for simulation.  It bins solved contact
points by their position in the foot frame; it is not presented as a MuJoCo
``geom`` force sensor and it does not replace a physical sensor model.
Keeping this distinction explicit is important for the later sim-to-real
interpretation of the results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np


FOOT_BODIES = {
    "left": "left_ankle_roll_link",
    "right": "right_ankle_roll_link",
}

# The coordinate convention follows the G1 ankle-roll body.  The x direction
# is approximately toeward and y is lateral.  These bounds cover the four
# small sole contact geoms in g1_12dof.xml and leave a little margin for mesh
# contacts on uneven terrain.
ARRAY_X_BOUNDS_M = (-0.08, 0.15)
ARRAY_Y_BOUNDS_M = (-0.06, 0.06)
# High-resolution experiment setting.  The previous 4 x 2 data are retained
# as a separately named baseline directory; changing these values changes the
# CSV schema produced by the logger, so never mix the resulting CSV files in a
# single analysis call.
ARRAY_NX = 8
ARRAY_NY = 4


@dataclass
class ContactSample:
    """One solved environment contact expressed in both useful frames."""

    position_world_m: np.ndarray
    position_foot_m: np.ndarray
    force_world_n: np.ndarray
    force_foot_n: np.ndarray
    torque_world_nm: np.ndarray
    normal_load_n: float


@dataclass
class FootReading:
    """All channels produced for one foot at one simulation instant."""

    cell_force_foot_n: np.ndarray
    macro_wrench_ankle: np.ndarray
    cop_foot_m: np.ndarray
    cop_valid: int
    contacts: int
    outside_contacts: int
    outside_force_foot_n: np.ndarray
    imu_gyro_rad_s: np.ndarray
    imu_accel_m_s2: np.ndarray


def _body_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _foot_geom_ids(model: mujoco.MjModel, foot_body_id: int) -> set[int]:
    """Return every collision geom attached to the ankle-roll body."""

    return {
        gid
        for gid in range(model.ngeom)
        if int(model.geom_bodyid[gid]) == foot_body_id
    }


def _contact_force_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_id: int,
    foot_geom_ids: set[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return force and contact torque acting on the foot in world axes.

    ``mj_contactForce`` reports the force on geom 1 in the contact frame.  A
    contact can list the foot as geom 1 or geom 2, hence the explicit sign
    handling.  The same sign is applied to the contact torque.  Only contacts
    against world-body geoms are retained; robot self contacts are excluded.
    """

    contact = data.contact[contact_id]
    geom1, geom2 = int(contact.geom[0]), int(contact.geom[1])
    body1 = int(model.geom_bodyid[geom1])
    body2 = int(model.geom_bodyid[geom2])

    if geom1 in foot_geom_ids and body2 == 0:
        sign = -1.0
    elif geom2 in foot_geom_ids and body1 == 0:
        sign = 1.0
    else:
        return None

    contact_wrench = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, contact_id, contact_wrench)
    frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
    # MuJoCo stores the contact-frame axes as matrix columns.  Transpose maps
    # the contact-frame force/torque into world coordinates.
    force_world = sign * frame.T @ contact_wrench[:3]
    torque_world = sign * frame.T @ contact_wrench[3:]
    return force_world, torque_world


def _imu(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute ideal local gyro and accelerometer channels.

    ``mj_objectVelocity`` and ``mj_objectAcceleration`` provide Cartesian
    body quantities in the body frame when ``flg_local=1``.  An accelerometer
    measures specific force, so gravity is removed from the inertial linear
    acceleration.  At rest on the ground this convention yields roughly
    +9.81 m/s^2 along the sensor's upward axis.
    """

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


def read_foot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot: str,
) -> FootReading:
    """Read one foot's array, ankle wrench/CoP, and IMU channels."""

    if foot not in FOOT_BODIES:
        raise ValueError(f"unknown foot {foot!r}; expected one of {tuple(FOOT_BODIES)}")

    foot_body_id = _body_id(model, FOOT_BODIES[foot])
    foot_geom_ids = _foot_geom_ids(model, foot_body_id)
    rotation_world_from_foot = np.asarray(data.xmat[foot_body_id], dtype=np.float64).reshape(3, 3)
    foot_origin_world = np.asarray(data.xpos[foot_body_id], dtype=np.float64)

    ankle_body_id = foot_body_id
    ankle_origin_world = foot_origin_world
    rotation_world_from_ankle = rotation_world_from_foot

    contacts: list[ContactSample] = []
    for contact_id in range(data.ncon):
        force_torque = _contact_force_world(model, data, contact_id, foot_geom_ids)
        if force_torque is None:
            continue
        force_world, torque_world_at_contact = force_torque
        position_world = np.asarray(data.contact[contact_id].pos, dtype=np.float64).copy()
        position_foot = rotation_world_from_foot.T @ (position_world - foot_origin_world)
        force_foot = rotation_world_from_foot.T @ force_world
        normal_load = max(float(force_world[2]), 0.0)
        contacts.append(
            ContactSample(
                position_world_m=position_world,
                position_foot_m=position_foot,
                force_world_n=force_world,
                force_foot_n=force_foot,
                torque_world_nm=torque_world_at_contact,
                normal_load_n=normal_load,
            )
        )

    x_edges = np.linspace(ARRAY_X_BOUNDS_M[0], ARRAY_X_BOUNDS_M[1], ARRAY_NX + 1)
    y_edges = np.linspace(ARRAY_Y_BOUNDS_M[0], ARRAY_Y_BOUNDS_M[1], ARRAY_NY + 1)
    cell_force = np.zeros((ARRAY_NX, ARRAY_NY, 3), dtype=np.float64)
    outside_force = np.zeros(3, dtype=np.float64)
    outside_contacts = 0
    cop_numerator = np.zeros(2, dtype=np.float64)
    cop_denominator = 0.0

    wrench_world = np.zeros(6, dtype=np.float64)
    for sample in contacts:
        x, y = sample.position_foot_m[:2]
        ix = int(np.searchsorted(x_edges, x, side="right") - 1)
        iy = int(np.searchsorted(y_edges, y, side="right") - 1)
        inside = 0 <= ix < ARRAY_NX and 0 <= iy < ARRAY_NY
        if inside:
            cell_force[ix, iy] += sample.force_foot_n
        else:
            outside_contacts += 1
            outside_force += sample.force_foot_n

        cop_weight = sample.normal_load_n
        if cop_weight > 0.0:
            cop_numerator += cop_weight * sample.position_foot_m[:2]
            cop_denominator += cop_weight

        lever_arm = sample.position_world_m - ankle_origin_world
        wrench_world[:3] += sample.force_world_n
        wrench_world[3:] += np.cross(lever_arm, sample.force_world_n)
        wrench_world[3:] += sample.torque_world_nm

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
    wrench_ankle[:3] = rotation_world_from_ankle.T @ wrench_world[:3]
    wrench_ankle[3:] = rotation_world_from_ankle.T @ wrench_world[3:]
    gyro, accel = _imu(model, data, ankle_body_id)

    return FootReading(
        cell_force_foot_n=cell_force,
        macro_wrench_ankle=wrench_ankle,
        cop_foot_m=cop,
        cop_valid=cop_valid,
        contacts=len(contacts),
        outside_contacts=outside_contacts,
        outside_force_foot_n=outside_force,
        imu_gyro_rad_s=gyro,
        imu_accel_m_s2=accel,
    )


def iter_cell_names(foot: str) -> Iterable[str]:
    """Yield stable CSV names in the same order as ``cell_force_foot_n``."""

    for ix in range(ARRAY_NX):
        for iy in range(ARRAY_NY):
            yield f"{foot}_cell_{ix}_{iy}"
