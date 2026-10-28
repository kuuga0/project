"""Sensor configurations and deferred sensor interfaces for H2 Plus.

The first locomotion policy is intentionally blind to terrain geometry.  The
ray-caster configuration is still defined here so that a later ablation or
terrain-aware policy can enable it without changing the scene API.  The torso
and two foot IMUs are attached for recording only; they are deliberately not
inserted into the first policy observation group.  FOOTRIX remains an interface
until its simulation mapping is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns


H2_PLUS_TERRAIN_FILTER_PATHS = ["/World/ground"]

H2_PLUS_IMU_UPDATE_PERIOD_S = 1.0 / 120.0
H2_PLUS_IMU_HISTORY_LENGTH = 6
H2_PLUS_IMU_GRAVITY_BIAS_M_S2 = (0.0, 0.0, 9.81)
H2_PLUS_FOOT_IMU_ROTATION_WXYZ = (0.7071067812, 0.0, 0.0, -0.7071067812)


@dataclass(frozen=True)
class H2PlusImuNoiseCfg:
    """Provisional recording noise until hardware noise is characterized."""

    linear_acceleration_std_m_s2: float = 0.05
    angular_velocity_std_rad_s: float = 0.005
    linear_acceleration_bias_std_m_s2: float = 0.02
    angular_velocity_bias_std_rad_s: float = 0.001


H2_PLUS_IMU_NOISE_CFG = H2PlusImuNoiseCfg()


H2_PLUS_LEFT_FOOT_CONTACT_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_ankle_pitch_link",
    update_period=0.0,
    history_length=3,
    filter_prim_paths_expr=H2_PLUS_TERRAIN_FILTER_PATHS,
    track_air_time=True,
    track_contact_points=True,
    track_friction_forces=True,
    max_contact_data_count_per_prim=64,
)


H2_PLUS_RIGHT_FOOT_CONTACT_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_ankle_pitch_link",
    update_period=0.0,
    history_length=3,
    filter_prim_paths_expr=H2_PLUS_TERRAIN_FILTER_PATHS,
    track_air_time=True,
    track_contact_points=True,
    track_friction_forces=True,
    max_contact_data_count_per_prim=64,
)


H2_PLUS_BODY_CONTACT_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/(pelvis|torso_link)",
    update_period=0.0,
    history_length=1,
    filter_prim_paths_expr=H2_PLUS_TERRAIN_FILTER_PATHS,
    track_air_time=False,
    track_contact_points=False,
    track_friction_forces=False,
    max_contact_data_count_per_prim=16,
)


H2_PLUS_TORSO_IMU_CFG = ImuCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    update_period=H2_PLUS_IMU_UPDATE_PERIOD_S,
    history_length=H2_PLUS_IMU_HISTORY_LENGTH,
    offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    gravity_bias=H2_PLUS_IMU_GRAVITY_BIAS_M_S2,
    debug_vis=False,
)


H2_PLUS_LEFT_FOOT_IMU_CFG = ImuCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_ankle_pitch_link",
    update_period=H2_PLUS_IMU_UPDATE_PERIOD_S,
    history_length=H2_PLUS_IMU_HISTORY_LENGTH,
    offset=ImuCfg.OffsetCfg(
        pos=(-0.047, 0.0373, -0.0222),
        rot=H2_PLUS_FOOT_IMU_ROTATION_WXYZ,
    ),
    gravity_bias=H2_PLUS_IMU_GRAVITY_BIAS_M_S2,
    debug_vis=False,
)


H2_PLUS_RIGHT_FOOT_IMU_CFG = ImuCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_ankle_pitch_link",
    update_period=H2_PLUS_IMU_UPDATE_PERIOD_S,
    history_length=H2_PLUS_IMU_HISTORY_LENGTH,
    offset=ImuCfg.OffsetCfg(
        pos=(-0.047, -0.0373, -0.0222),
        rot=H2_PLUS_FOOT_IMU_ROTATION_WXYZ,
    ),
    gravity_bias=H2_PLUS_IMU_GRAVITY_BIAS_M_S2,
    debug_vis=False,
)


# This sensor is deliberately not inserted into the first policy observation
# group.  It remains available for a later terrain-aware or ablation policy.
H2_PLUS_RAY_CASTER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/pelvis",
    update_period=1.0 / 60.0,
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(1.6, 1.0)),
    mesh_prim_paths=["/World/ground"],
    debug_vis=False,
)


@dataclass(frozen=True)
class H2PlusImuInterface:
    """Mount metadata for an active Isaac Lab IMU sensor."""

    enabled: bool = True
    cfg: ImuCfg | None = None
    mount_link: str | None = None
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class H2PlusFootrixInterface:
    """Reserved FOOTRIX mapping slot, not a fabricated contact sensor."""

    enabled: bool = False
    cell_count: int = 154
    channels_per_cell: int = 3
    cfg: Any | None = None


@dataclass(frozen=True)
class H2PlusSensorInterfaces:
    """Named sensor objects consumed by the scene and recording configs."""

    left_foot_contact: ContactSensorCfg
    right_foot_contact: ContactSensorCfg
    body_contact: ContactSensorCfg
    ray_caster: RayCasterCfg
    ray_caster_enabled: bool = False
    imu: H2PlusImuInterface = H2PlusImuInterface()
    imu_cfgs: tuple[ImuCfg, ImuCfg, ImuCfg] = ()
    footrix: H2PlusFootrixInterface = H2PlusFootrixInterface()


H2_PLUS_SENSOR_INTERFACES = H2PlusSensorInterfaces(
    left_foot_contact=H2_PLUS_LEFT_FOOT_CONTACT_CFG,
    right_foot_contact=H2_PLUS_RIGHT_FOOT_CONTACT_CFG,
    body_contact=H2_PLUS_BODY_CONTACT_CFG,
    ray_caster=H2_PLUS_RAY_CASTER_CFG,
    imu=H2PlusImuInterface(
        enabled=True,
        cfg=H2_PLUS_TORSO_IMU_CFG,
        mount_link="torso_link",
        translation_m=(0.0, 0.0, 0.0),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    ),
    imu_cfgs=(H2_PLUS_TORSO_IMU_CFG, H2_PLUS_LEFT_FOOT_IMU_CFG, H2_PLUS_RIGHT_FOOT_IMU_CFG),
)


__all__ = [
    "H2PlusFootrixInterface",
    "H2PlusImuInterface",
    "H2PlusImuNoiseCfg",
    "H2PlusSensorInterfaces",
    "H2_PLUS_BODY_CONTACT_CFG",
    "H2_PLUS_FOOT_IMU_ROTATION_WXYZ",
    "H2_PLUS_IMU_GRAVITY_BIAS_M_S2",
    "H2_PLUS_IMU_HISTORY_LENGTH",
    "H2_PLUS_IMU_NOISE_CFG",
    "H2_PLUS_IMU_UPDATE_PERIOD_S",
    "H2_PLUS_LEFT_FOOT_CONTACT_CFG",
    "H2_PLUS_LEFT_FOOT_IMU_CFG",
    "H2_PLUS_RAY_CASTER_CFG",
    "H2_PLUS_RIGHT_FOOT_CONTACT_CFG",
    "H2_PLUS_RIGHT_FOOT_IMU_CFG",
    "H2_PLUS_SENSOR_INTERFACES",
    "H2_PLUS_TORSO_IMU_CFG",
    "H2_PLUS_TERRAIN_FILTER_PATHS",
]
