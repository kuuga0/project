"""Ideal IMU/pressure-array contact-wrench reconstruction for the foot bench."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SensorWrenchEstimate:
    """Three comparable wrench estimates, all ordered ``[moment; force]``."""

    imu_wrench_world: FloatArray
    pressure_wrench_world: FloatArray
    fused_wrench_world: FloatArray
    gyro_foot_radps: FloatArray
    specific_force_foot_mps2: FloatArray


def sensor_vector(
    model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str
) -> FloatArray:
    """Read one MuJoCo sensor without depending on a hard-coded address."""
    sensor_id = model.sensor(sensor_name).id
    address = int(model.sensor_adr[sensor_id])
    dimension = int(model.sensor_dim[sensor_id])
    return np.asarray(data.sensordata[address : address + dimension], dtype=float).copy()


def foot_imu_sample(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[FloatArray, FloatArray]:
    """Return ideal gyro and accelerometer readings in the foot frame.

    MuJoCo's accelerometer output is treated as specific force
    ``f_B = R_WB^T (a_W - g_W)``. This is the quantity an ideal accelerometer
    measures and is the correct input for Newton's equation below.
    """
    gyro = sensor_vector(model, data, "foot_gyro")
    accelerometer = sensor_vector(model, data, "foot_accelerometer")
    return gyro, accelerometer


def body_inertia_world(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> tuple[float, FloatArray, FloatArray]:
    """Return body mass, body-frame inertia, and world-from-body rotation."""
    mass = float(model.body_mass[body_id])
    inertia_body = np.asarray(model.body_inertia[body_id], dtype=float).copy()
    rotation_world_from_foot = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    return mass, inertia_body, rotation_world_from_foot


def pressure_wrench_from_normal_forces(
    normal_forces: ArrayLike,
    contact_points_foot: ArrayLike,
    contact_plane_z_m: float,
) -> FloatArray:
    """Map ideal per-cell normal forces to a wrench about the foot COM."""
    forces = np.asarray(normal_forces, dtype=float).reshape(-1)
    points = np.asarray(contact_points_foot, dtype=float).reshape(-1, 3).copy()
    if len(forces) != len(points):
        raise ValueError("normal_forces and contact_points_foot must have equal length")
    points[:, 2] = float(contact_plane_z_m)
    wrench = np.zeros(6)
    for point, normal_force in zip(points, forces):
        force = float(normal_force) * np.array([0.0, 0.0, 1.0])
        wrench[:3] += np.cross(point, force)
        wrench[3:] += force
    return wrench


def imu_wrench_estimate(
    mass_kg: float,
    inertia_body: ArrayLike,
    rotation_world_from_foot: ArrayLike,
    gyro_foot_radps: ArrayLike,
    gyro_previous_foot_radps: ArrayLike,
    specific_force_foot_mps2: ArrayLike,
    timestep_s: float,
    external_wrench_world: ArrayLike,
) -> FloatArray:
    """Estimate ground-on-foot wrench from ideal IMU and known excitation.

    The gyro derivative is taken in body coordinates, so Euler's equation is
    evaluated with the body-frame inertia. The returned wrench is world-frame
    and about the foot COM. External test excitation is explicitly removed.
    """
    if timestep_s <= 0.0:
        raise ValueError("timestep_s must be positive")
    rotation = np.asarray(rotation_world_from_foot, dtype=float).reshape(3, 3)
    inertia = np.asarray(inertia_body, dtype=float).reshape(3)
    gyro = np.asarray(gyro_foot_radps, dtype=float).reshape(3)
    gyro_previous = np.asarray(gyro_previous_foot_radps, dtype=float).reshape(3)
    specific_force = np.asarray(specific_force_foot_mps2, dtype=float).reshape(3)
    external = np.asarray(external_wrench_world, dtype=float).reshape(6)

    # Specific force already excludes gravity: m(a-g) is the non-gravitational
    # resultant. Subtract the known external test wrench to leave contact.
    contact_force_world = mass_kg * (rotation @ specific_force) - external[3:]
    angular_acceleration_body = (gyro - gyro_previous) / float(timestep_s)
    gyroscopic_term_body = np.cross(gyro, inertia * gyro)
    contact_moment_body = (
        inertia * angular_acceleration_body
        + gyroscopic_term_body
        - rotation.T @ external[:3]
    )
    contact_moment_world = rotation @ contact_moment_body
    return np.concatenate((contact_moment_world, contact_force_world))


def fuse_pressure_and_imu(
    imu_wrench_world: ArrayLike,
    pressure_wrench_world: ArrayLike,
    rotation_world_from_foot: ArrayLike,
    contact_plane_z_m: float,
) -> FloatArray:
    """Fuse normal pressure terms with inertial tangential/contact-torque terms.

    Fusion is performed in the foot frame. The pressure array measures the
    normal-force terms ``Fz, sum(y Fz), -sum(x Fz)``. The IMU provides ``Fx``
    and ``Fy``, whose common contact-plane lever arm adds ``-z Fy`` and
    ``z Fx`` to ``Mx`` and ``My`` respectively. ``Mz`` stays inertial because
    a normal-only array does not observe tangential friction torque.
    """
    imu = np.asarray(imu_wrench_world, dtype=float).reshape(6)
    pressure = np.asarray(pressure_wrench_world, dtype=float).reshape(6)
    rotation = np.asarray(rotation_world_from_foot, dtype=float).reshape(3, 3)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("rotation_world_from_foot must be orthonormal")
    imu_foot = np.concatenate((rotation.T @ imu[:3], rotation.T @ imu[3:]))
    pressure_foot = np.concatenate(
        (rotation.T @ pressure[:3], rotation.T @ pressure[3:])
    )
    z = float(contact_plane_z_m)
    fused_foot = imu_foot.copy()
    fused_foot[0] = pressure_foot[0] - z * imu_foot[4]
    fused_foot[1] = pressure_foot[1] + z * imu_foot[3]
    fused_foot[5] = pressure_foot[5]
    return np.concatenate(
        (rotation @ fused_foot[:3], rotation @ fused_foot[3:])
    )


def sensor_wrench_estimate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    normal_forces: ArrayLike,
    contact_points_foot: ArrayLike,
    contact_plane_z_m: float,
    gyro_previous_foot_radps: ArrayLike,
    external_wrench_world: ArrayLike,
) -> SensorWrenchEstimate:
    """Build pressure, IMU, and fused estimates for one simulation sample."""
    gyro, specific_force = foot_imu_sample(model, data)
    mass, inertia_body, rotation = body_inertia_world(model, data, body_id)
    imu_wrench = imu_wrench_estimate(
        mass,
        inertia_body,
        rotation,
        gyro,
        gyro_previous_foot_radps,
        specific_force,
        model.opt.timestep,
        external_wrench_world,
    )
    pressure_foot = pressure_wrench_from_normal_forces(
        normal_forces, contact_points_foot, contact_plane_z_m
    )
    pressure_world = np.concatenate(
        (rotation @ pressure_foot[:3], rotation @ pressure_foot[3:])
    )
    fused = fuse_pressure_and_imu(
        imu_wrench,
        pressure_world,
        rotation,
        contact_plane_z_m,
    )
    return SensorWrenchEstimate(
        imu_wrench_world=imu_wrench,
        pressure_wrench_world=pressure_world,
        fused_wrench_world=fused,
        gyro_foot_radps=gyro,
        specific_force_foot_mps2=specific_force,
    )
