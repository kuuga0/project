import numpy as np

from foot_contact_wrench.estimation.sensor_wrench import (
    fuse_pressure_and_imu,
    imu_wrench_estimate,
    pressure_wrench_from_normal_forces,
)


def test_pressure_array_maps_normal_loads_about_foot_com():
    wrench = pressure_wrench_from_normal_forces(
        normal_forces=[10.0, 20.0],
        contact_points_foot=[[0.10, 0.05, -0.03], [-0.10, -0.05, -0.03]],
        contact_plane_z_m=-0.03,
    )
    assert np.allclose(wrench, [-0.5, 1.0, 0.0, 0.0, 0.0, 30.0])


def test_imu_dynamics_subtracts_known_external_wrench():
    estimate = imu_wrench_estimate(
        mass_kg=4.0,
        inertia_body=[1.0, 2.0, 3.0],
        rotation_world_from_foot=np.eye(3),
        gyro_foot_radps=[0.0, 0.0, 0.0],
        gyro_previous_foot_radps=[0.0, 0.0, 0.0],
        specific_force_foot_mps2=[1.0, 2.0, 3.0],
        timestep_s=0.01,
        external_wrench_world=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0],
    )
    assert np.allclose(estimate, [-0.2, -0.3, -0.4, 3.0, 6.0, 9.0])


def test_fusion_adds_tangential_force_lever_arm_to_pressure_moments():
    imu = np.array([0.7, 1.1, 2.0, 4.0, 5.0, 30.0])
    pressure = np.array([0.2, -0.4, 0.0, 0.0, 0.0, 29.0])
    fused = fuse_pressure_and_imu(imu, pressure, np.eye(3), contact_plane_z_m=-0.03)
    assert np.allclose(fused, [0.35, -0.52, 2.0, 4.0, 5.0, 29.0])
