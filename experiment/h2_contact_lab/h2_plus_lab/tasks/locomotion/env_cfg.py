"""H2 Plus locomotion scenes, shared MDP settings and train/play configurations.

Edit shared manager terms here or override them in the task-specific classes at
the end of this file. Sensor parameters live in sensor_cfg.py and terrain
geometry/materials in terrain_cfg.py. Gym IDs are registered in __init__.py.
"""

from __future__ import annotations

import math
from pathlib import Path
from time import time_ns

import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg, RecorderTermCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ...assets.h2_plus import H2_PLUS_CFG
from .terrain_cfg import (
    HARD_MATERIAL,
    LOW_FRICTION_MATERIAL,
    H2_PLUS_DEFORMABLE_GROUND_CFG,
    H2_PLUS_TERRAIN_CATALOG,
)
from .sensor_cfg import (
    H2_PLUS_BODY_CONTACT_CFG,
    H2_PLUS_LEFT_FOOT_IMU_CFG,
    H2_PLUS_LEFT_FOOT_CONTACT_CFG,
    H2_PLUS_RIGHT_FOOT_IMU_CFG,
    H2_PLUS_RIGHT_FOOT_CONTACT_CFG,
    H2_PLUS_TORSO_IMU_CFG,
)
from . import mdp


def _apply_rewards_yaml(rewards, path: Path | None = None) -> None:
    """Apply numeric reward settings before task-specific and Hydra overrides."""
    path = Path(path) if path is not None else Path(__file__).with_name("agents") / "rsl_rl_locomotion_rewards.yaml"
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) != {"rewards"} or not isinstance(config["rewards"], dict):
        raise ValueError(f"{path}: expected a mapping with a single 'rewards' section")

    def number(value, key):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{path}: {key} must be a finite number")
        return float(value)

    updates = []
    for name, settings in config["rewards"].items():
        term = getattr(rewards, name, None)
        if term is None or not hasattr(term, "weight"):
            raise ValueError(f"{path}: unknown reward term {name!r}")
        if not isinstance(settings, dict) or set(settings) - {"weight", "params"}:
            raise ValueError(f"{path}: rewards.{name} only accepts weight and params")
        weight = number(settings["weight"], f"{name}.weight") if "weight" in settings else term.weight
        params = settings.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"{path}: {name}.params must be a mapping")
        values = {}
        for key, value in params.items():
            current = getattr(term, "params", {}).get(key)
            if type(current) not in (int, float):
                raise ValueError(f"{path}: {name}.params.{key} is not an existing numeric parameter")
            value = number(value, f"{name}.params.{key}")
            if key == "std" and value <= 0:
                raise ValueError(f"{path}: {name}.params.std must be positive")
            if key in {"threshold", "command_threshold"} and value < 0:
                raise ValueError(f"{path}: {name}.params.{key} must be non-negative")
            values[key] = value
        updates.append((term, weight, values))
    # Validate the whole file first, so a typo never leaves a half-applied config.
    for term, weight, values in updates:
        term.weight = weight
        term.params.update(values)


# Scene composition

@configclass
class H2PlusSceneCfg(InteractiveSceneCfg):
    """Reusable robot, terrain and lighting scene for later environments."""

    num_envs: int = 1
    env_spacing: float = 3.0

    robot: ArticulationCfg = H2_PLUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    terrain: TerrainImporterCfg = H2_PLUS_TERRAIN_CATALOG["ordinary_hard"].importer_cfg

    # Locomotion needs net force and air time, not pairwise contact records.
    left_foot_contact: ContactSensorCfg = H2_PLUS_LEFT_FOOT_CONTACT_CFG.replace(
        filter_prim_paths_expr=[], track_contact_points=False, track_friction_forces=False,
    )
    # Locomotion needs net force and air time, not pairwise contact records.
    right_foot_contact: ContactSensorCfg = H2_PLUS_RIGHT_FOOT_CONTACT_CFG.replace(
        filter_prim_paths_expr=[], track_contact_points=False, track_friction_forces=False,
    )
    body_contact: ContactSensorCfg = H2_PLUS_BODY_CONTACT_CFG
    torso_imu: ImuCfg = H2_PLUS_TORSO_IMU_CFG
    left_foot_imu: ImuCfg = H2_PLUS_LEFT_FOOT_IMU_CFG
    right_foot_imu: ImuCfg = H2_PLUS_RIGHT_FOOT_IMU_CFG
    # The full ankle-pitch links are the only foot collision bodies.  FOOTRIX
    # marker geometry is part of the robot asset; its point forces are produced
    # by the PhysX contact-report recorder from these two full-foot links.

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8)),
    )


@configclass
class H2PlusLowFrictionSceneCfg(H2PlusSceneCfg):
    """Scene with two adjacent rigid regions and genuinely different materials."""

    # A single TerrainImporter cannot carry both materials, so this scene
    # replaces it with two independently spawned rigid cuboids.
    env_spacing: float = 5.0
    terrain = None
    reference_region = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Terrain/ReferenceRegion",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 2.0, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=HARD_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -1.0, -0.05)),
    )
    low_friction_region = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Terrain/LowFrictionRegion",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 2.0, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=LOW_FRICTION_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.20)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 1.0, -0.05)),
    )

    body_contact = H2_PLUS_BODY_CONTACT_CFG


@configclass
class H2PlusDeformableSceneCfg(H2PlusSceneCfg):
    """Scene with a real volume-deformable slab over a rigid backing."""

    env_spacing: float = 5.0
    terrain = None
    deformable_backing = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Terrain/FlexibleBacking",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=HARD_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.225)),
    )
    # PhysX deformable objects cannot use replicated physics.
    replicate_physics: bool = False
    deformable_ground = H2_PLUS_DEFORMABLE_GROUND_CFG

    body_contact = H2_PLUS_BODY_CONTACT_CFG


def scene_for_terrain(name: str) -> InteractiveSceneCfg:
    """Return the scene implementation for a named H2 Plus terrain."""

    spec = H2_PLUS_TERRAIN_CATALOG[name]
    if name == "low_friction_difference":
        return H2PlusLowFrictionSceneCfg()
    if name == "deformable_ground":
        if spec.deformable_cfg is None:
            raise ValueError("deformable_ground is missing its DeformableObjectCfg")
        return H2PlusDeformableSceneCfg()
    if spec.importer_cfg is None:
        raise ValueError(
            f"Terrain {name!r} has no scene implementation or TerrainImporterCfg."
        )
    return H2PlusSceneCfg(terrain=spec.importer_cfg)


# Shared MDP and environment settings

H2_PLUS_LEG_JOINTS = [
    ".*_hip_(pitch|roll|yaw)_joint",
    ".*_knee_joint",
    ".*_ankle_(roll|pitch)_joint",
]


H2_PLUS_WAIST_JOINTS = ["waist_(yaw|roll|pitch)_joint"]


H2_PLUS_SHOULDER_ELBOW_JOINTS = [
    ".*_shoulder_(pitch|roll|yaw)_joint",
    ".*_elbow_joint",
]


H2_PLUS_ACTIVE_JOINTS = H2_PLUS_LEG_JOINTS + H2_PLUS_WAIST_JOINTS + H2_PLUS_SHOULDER_ELBOW_JOINTS


H2_PLUS_PASSIVE_JOINTS = ["head_(pitch|yaw)_joint", ".*_wrist_(roll|pitch|yaw)_joint"]


@configclass
class H2PlusActionsCfg:
    """Twenty-three policy actions: legs, waist, shoulders and elbows."""

    legs = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=H2_PLUS_LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )
    waist = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=H2_PLUS_WAIST_JOINTS,
        scale=0.15,
        use_default_offset=True,
    )
    shoulders_elbows = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=H2_PLUS_SHOULDER_ELBOW_JOINTS,
        scale=0.20,
        use_default_offset=True,
    )


@configclass
class H2PlusObservationsCfg:
    """Blind policy observations; no RayCaster term is included here."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=H2_PLUS_ACTIVE_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=H2_PLUS_ACTIVE_JOINTS)},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class H2PlusCommandsCfg:
    """Forward walking and yaw commands used by the first locomotion task."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class H2PlusRewardsCfg:
    """Reward functions and bindings; tuning overrides live in agents/rsl_rl_locomotion_rewards.yaml."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=mdp.h2_plus_feet_air_time_positive_biped,
        weight=0.2,
        params={"command_name": "base_velocity", "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.h2_plus_feet_slide,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_pitch_link", "right_ankle_pitch_link"]
            )
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=H2_PLUS_ACTIVE_JOINTS)},
    )
    passive_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=H2_PLUS_PASSIVE_JOINTS)},
    )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.06,
            "asset_cfg": SceneEntityCfg("robot", joint_names=H2_PLUS_ACTIVE_JOINTS),
        },
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_acceleration = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=H2_PLUS_ACTIVE_JOINTS)},
    )


@configclass
class H2PlusTerminationsCfg:
    """Timeout, body contact, orientation and terrain boundary conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.h2_plus_body_contact,
        params={
            "sensor_cfg": SceneEntityCfg("body_contact", body_names=["pelvis", "torso_link"]),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 0.5},
    )


@configclass
class H2PlusEventsCfg:
    """Deterministic reset events; disturbances remain a later extension."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.05, 0.05)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class H2PlusSensorRecorderTermCfg(RecorderTermCfg):
    class_type: type = mdp.H2PlusSensorRecorder
    env_ids: list[int] | None = [0]  # null records all parallel environments
    record_every: int = 1  # control steps; terminal frames are always included


@configclass
class H2PlusRecorderCfg(RecorderManagerBaseCfg):
    dataset_filename = "sensors"
    dataset_export_mode = DatasetExportMode.EXPORT_ALL
    export_in_record_pre_reset = True
    export_in_close = True
    sensors: H2PlusSensorRecorderTermCfg = H2PlusSensorRecorderTermCfg()


@configclass
class H2PlusLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """H2 Plus blind locomotion environment assembled from project modules."""

    scene: H2PlusSceneCfg = H2PlusSceneCfg(num_envs=1024)
    observations: H2PlusObservationsCfg = H2PlusObservationsCfg()
    actions: H2PlusActionsCfg = H2PlusActionsCfg()
    commands: H2PlusCommandsCfg = H2PlusCommandsCfg()
    rewards: H2PlusRewardsCfg = H2PlusRewardsCfg()
    terminations: H2PlusTerminationsCfg = H2PlusTerminationsCfg()
    events: H2PlusEventsCfg = H2PlusEventsCfg()
    recorders: H2PlusRecorderCfg = H2PlusRecorderCfg()

    def __post_init__(self):
        _apply_rewards_yaml(self.rewards)
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=self.decimation)
        if self.recorders is not None:
            # Safe unique fallback for environments created outside train.py/play.py.
            self.recorders.dataset_export_dir_path = str(
                Path(__file__).resolve().parents[3] / "output" / "sensors" / f"env_{time_ns()}"
            )


# Flat train/play configurations

@configclass
class H2PlusFlatEnvCfg(H2PlusLocomotionEnvCfg):
    scene: H2PlusSceneCfg = H2PlusSceneCfg(num_envs=1024)


@configclass
class H2PlusFlatEnvCfg_PLAY(H2PlusFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


# Low-friction train/play configurations

@configclass
class H2PlusLowFrictionEnvCfg(H2PlusLocomotionEnvCfg):
    scene: H2PlusLowFrictionSceneCfg = H2PlusLowFrictionSceneCfg(num_envs=1024)


@configclass
class H2PlusLowFrictionEnvCfg_PLAY(H2PlusLowFrictionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


# Deformable train/play configurations

@configclass
class H2PlusDeformableEnvCfg(H2PlusLocomotionEnvCfg):
    scene: H2PlusDeformableSceneCfg = H2PlusDeformableSceneCfg(num_envs=64)


@configclass
class H2PlusDeformableEnvCfg_PLAY(H2PlusDeformableEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
