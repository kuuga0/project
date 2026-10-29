"""Sensor configurations and sensor interfaces for H2 Plus.

The first locomotion policy is intentionally blind to terrain geometry.  The
ray-caster configuration is still defined here so that a later ablation or
terrain-aware policy can enable it without changing the scene API.  The torso
and two foot IMUs are attached for recording only; they are deliberately not
inserted into the first policy observation group.  FOOTRIX uses a dedicated
visual marker layer: each of its 154 cells is an independent fixed child link
with a thin circular visual shape and no collision.  A PhysX contact-report
callback reads the complete foot collision and distributes contact impulses to
nearby cells in the planar measurement layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns


H2_PLUS_TERRAIN_FILTER_PATHS = ["/World/ground"]

H2_PLUS_IMU_UPDATE_PERIOD_S = 1.0 / 120.0
H2_PLUS_IMU_HISTORY_LENGTH = 6
H2_PLUS_IMU_GRAVITY_BIAS_M_S2 = (0.0, 0.0, 9.81)
H2_PLUS_FOOT_IMU_ROTATION_WXYZ = (0.7071067812, 0.0, 0.0, -0.7071067812)
H2_PLUS_FOOTRIX_CELL_COUNT = 154
H2_PLUS_FOOTRIX_CHANNELS_PER_CELL = 3
H2_PLUS_FOOTRIX_CELL_RADIUS_M = 0.003401643
H2_PLUS_FOOTRIX_CELL_THICKNESS_M = 0.004
# The layout spacing is about 8--10 mm; 12 mm normally includes multiple
# neighboring cells while keeping the kernel local.
H2_PLUS_FOOTRIX_SUPPORT_RADIUS_M = 0.012
H2_PLUS_FOOTRIX_PROJECTION_METHOD = "planar_linear_weighted_force_bin"
# Effective common layout plane, aligned to the current foot mesh bottom in the
# ankle_pitch_link frame.  The image-derived JSON remains the raw source.
H2_PLUS_FOOTRIX_LAYOUT_Z_M = -0.0521
H2_PLUS_FOOTRIX_LAYOUT_PATH = Path(
    "experiment/h2_contact_lab/output/history/windows_footrix_points_checks_20261015/footrix_layout.json"
)
H2_PLUS_FOOTRIX_LAYOUT_STATUS = "image-derived canonical left/right mirrored layouts; effective z aligned to current foot mesh bottom; installation coordinates remain replaceable"


@dataclass(frozen=True)
class H2PlusImuNoiseCfg:
    """Reserved noise parameters; the current RecorderManager stores raw IMU values."""

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
    # Termination uses net force on both bodies, not a filtered force matrix.
    filter_prim_paths_expr=[],
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


@dataclass(frozen=True)
class H2PlusFootrixMarkerCfg:
    """Visual measurement-point layout; markers have no collision geometry."""

    marker_prim_path: str
    source_contact_sensor: str
    enabled: bool = True
    cell_count: int = H2_PLUS_FOOTRIX_CELL_COUNT
    channels_per_cell: int = H2_PLUS_FOOTRIX_CHANNELS_PER_CELL
    marker_collision_enabled: bool = False
    projection_method: str = H2_PLUS_FOOTRIX_PROJECTION_METHOD
    support_radius_m: float = H2_PLUS_FOOTRIX_SUPPORT_RADIUS_M


# The full ankle-pitch foot mesh remains the only foot-ground collision body.
# These fixed links are visual-only markers.  A PhysX contact-report callback
# reads contact impulses from the full foot and distributes them to nearby
# points with normalized planar weights.
H2_PLUS_LEFT_FOOTRIX_CFG = H2PlusFootrixMarkerCfg(
    marker_prim_path="{ENV_REGEX_NS}/Robot/(left_ankle_pitch_link/)?left_footrix_cell_.*",
    source_contact_sensor="left_foot_contact",
)


H2_PLUS_RIGHT_FOOTRIX_CFG = H2PlusFootrixMarkerCfg(
    marker_prim_path="{ENV_REGEX_NS}/Robot/(right_ankle_pitch_link/)?right_footrix_cell_.*",
    source_contact_sensor="right_foot_contact",
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
    """FOOTRIX marker layout and full-foot contact-report projection config."""

    enabled: bool = True
    cell_count: int = H2_PLUS_FOOTRIX_CELL_COUNT
    channels_per_cell: int = H2_PLUS_FOOTRIX_CHANNELS_PER_CELL
    cfg: tuple[H2PlusFootrixMarkerCfg, H2PlusFootrixMarkerCfg] | None = None
    layout_path: str = str(H2_PLUS_FOOTRIX_LAYOUT_PATH)
    layout_status: str = H2_PLUS_FOOTRIX_LAYOUT_STATUS
    cell_layout_frame: str = "ankle_pitch_link"
    layout_z_m: float = H2_PLUS_FOOTRIX_LAYOUT_Z_M
    measurement_frame: str = "world"
    cell_radius_m: float = H2_PLUS_FOOTRIX_CELL_RADIUS_M
    cell_thickness_m: float = H2_PLUS_FOOTRIX_CELL_THICKNESS_M
    support_radius_m: float = H2_PLUS_FOOTRIX_SUPPORT_RADIUS_M
    projection_method: str = H2_PLUS_FOOTRIX_PROJECTION_METHOD
    marker_collision_enabled: bool = False
    visualized_in_urdf: bool = True


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
    footrix_left: H2PlusFootrixMarkerCfg | None = None
    footrix_right: H2PlusFootrixMarkerCfg | None = None
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
    footrix_left=H2_PLUS_LEFT_FOOTRIX_CFG,
    footrix_right=H2_PLUS_RIGHT_FOOTRIX_CFG,
    footrix=H2PlusFootrixInterface(
        enabled=True,
        cfg=(H2_PLUS_LEFT_FOOTRIX_CFG, H2_PLUS_RIGHT_FOOTRIX_CFG),
    ),
)


__all__ = [
    "H2PlusFootrixInterface",
    "H2PlusFootrixMarkerCfg",
    "H2PlusImuInterface",
    "H2PlusImuNoiseCfg",
    "H2PlusSensorInterfaces",
    "H2_PLUS_BODY_CONTACT_CFG",
    "H2_PLUS_FOOT_IMU_ROTATION_WXYZ",
    "H2_PLUS_FOOTRIX_CELL_COUNT",
    "H2_PLUS_FOOTRIX_CHANNELS_PER_CELL",
    "H2_PLUS_FOOTRIX_CELL_RADIUS_M",
    "H2_PLUS_FOOTRIX_CELL_THICKNESS_M",
    "H2_PLUS_FOOTRIX_SUPPORT_RADIUS_M",
    "H2_PLUS_FOOTRIX_PROJECTION_METHOD",
    "H2_PLUS_FOOTRIX_LAYOUT_Z_M",
    "H2_PLUS_FOOTRIX_LAYOUT_PATH",
    "H2_PLUS_FOOTRIX_LAYOUT_STATUS",
    "H2_PLUS_IMU_GRAVITY_BIAS_M_S2",
    "H2_PLUS_IMU_HISTORY_LENGTH",
    "H2_PLUS_IMU_NOISE_CFG",
    "H2_PLUS_IMU_UPDATE_PERIOD_S",
    "H2_PLUS_LEFT_FOOT_CONTACT_CFG",
    "H2_PLUS_LEFT_FOOT_IMU_CFG",
    "H2_PLUS_LEFT_FOOTRIX_CFG",
    "H2_PLUS_RAY_CASTER_CFG",
    "H2_PLUS_RIGHT_FOOT_CONTACT_CFG",
    "H2_PLUS_RIGHT_FOOT_IMU_CFG",
    "H2_PLUS_RIGHT_FOOTRIX_CFG",
    "H2_PLUS_SENSOR_INTERFACES",
    "H2_PLUS_TORSO_IMU_CFG",
    "H2_PLUS_TERRAIN_FILTER_PATHS",
]
