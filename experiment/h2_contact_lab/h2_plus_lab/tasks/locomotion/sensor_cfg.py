"""Sensor configuration accessors; parameters and metadata live in sensor_cfg.yaml."""

from pathlib import Path

from ...config import build_config, config_data

_FOOTRIX = config_data("sensors.footrix")
H2_PLUS_FOOTRIX_CELL_COUNT = _FOOTRIX["cell_count"]
H2_PLUS_FOOTRIX_CHANNELS_PER_CELL = _FOOTRIX["channels_per_cell"]
H2_PLUS_FOOTRIX_CELL_RADIUS_M = _FOOTRIX["cell_radius_m"]
H2_PLUS_FOOTRIX_CELL_THICKNESS_M = _FOOTRIX["cell_thickness_m"]
H2_PLUS_FOOTRIX_SURFACE_TOLERANCE_M = _FOOTRIX["surface_tolerance_m"]
H2_PLUS_FOOTRIX_PROJECTION_METHOD = _FOOTRIX["projection_method"]
H2_PLUS_FOOTRIX_LAYOUT_Z_M = _FOOTRIX["layout_z_m"]
H2_PLUS_FOOTRIX_LAYOUT_PATH = Path(_FOOTRIX["layout_path"])
H2_PLUS_FOOTRIX_LAYOUT_STATUS = _FOOTRIX["layout_status"]

H2_PLUS_IMU_NOISE_CFG = config_data("sensors.imu_noise")  # reserved; raw recorder adds no noise
H2_PLUS_IMU_UPDATE_PERIOD_S = config_data("sensors.torso_imu")["update_period"]
H2_PLUS_IMU_HISTORY_LENGTH = config_data("sensors.torso_imu")["history_length"]
H2_PLUS_IMU_GRAVITY_BIAS_M_S2 = tuple(config_data("sensors.torso_imu")["gravity_bias"])
H2_PLUS_FOOT_IMU_ROTATION_WXYZ = tuple(config_data("sensors.left_foot_imu")["offset"]["rot"])
H2_PLUS_LEFT_FOOT_CONTACT_CFG = build_config("sensors.left_foot_contact")
H2_PLUS_RIGHT_FOOT_CONTACT_CFG = build_config("sensors.right_foot_contact")
H2_PLUS_BODY_CONTACT_CFG = build_config("sensors.body_contact")
H2_PLUS_TORSO_IMU_CFG = build_config("sensors.torso_imu")
H2_PLUS_LEFT_FOOT_IMU_CFG = build_config("sensors.left_foot_imu")
H2_PLUS_RIGHT_FOOT_IMU_CFG = build_config("sensors.right_foot_imu")
H2_PLUS_RAY_CASTER_CFG = build_config("sensors.ray_caster")

__all__ = [name for name in globals() if name.startswith("H2_PLUS_")]
