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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

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


def _apply_rewards_yaml(
    rewards,
    path: Path | None = None,
    *,
    events=None,
    observations=None,
) -> None:
    """Apply reward and optional domain-randomization settings from the YAML file."""
    path = Path(path) if path is not None else Path(__file__).with_name("agents") / "rsl_rl_locomotion_rewards.yaml"
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if (
        not isinstance(config, dict)
        or "rewards" not in config
        or set(config) - {"rewards", "domain_randomization"}
        or not isinstance(config["rewards"], dict)
    ):
        raise ValueError(f"{path}: expected 'rewards' and optional 'domain_randomization' sections")

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
            if key in {"threshold", "command_threshold", "contact_force_threshold"} and value < 0:
                raise ValueError(f"{path}: {name}.params.{key} must be non-negative")
            values[key] = value
        updates.append((term, weight, values))
    if "domain_randomization" in config:
        if events is None or observations is None:
            raise ValueError(f"{path}: domain_randomization requires environment events and observations")
        _apply_domain_randomization_yaml(events, observations, config["domain_randomization"], path)
    # Validate the whole file first, so a typo never leaves a half-applied config.
    for term, weight, values in updates:
        term.weight = weight
        term.params.update(values)


def _apply_domain_randomization_yaml(events, observations, settings, path: Path) -> None:
    """Apply validated domain-randomization ranges to the existing event terms."""

    if not isinstance(settings, dict):
        raise ValueError(f"{path}: domain_randomization must be a mapping")
    allowed = {
        "enabled",
        "foot_material",
        "body_mass",
        "body_com",
        "reset_joints",
        "reset_base",
        "push",
        "observation_noise",
    }
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown domain_randomization keys: {sorted(unknown)}")

    def number(value, key):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{path}: domain_randomization.{key} must be a finite number")
        return float(value)

    def boolean(value, key):
        if type(value) is not bool:
            raise ValueError(f"{path}: domain_randomization.{key} must be true or false")
        return value

    def pair(value, key, *, nonnegative=False, positive=False):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{path}: domain_randomization.{key} must contain two numbers")
        result = (number(value[0], f"{key}[0]"), number(value[1], f"{key}[1]"))
        if result[0] > result[1]:
            raise ValueError(f"{path}: domain_randomization.{key} lower bound exceeds upper bound")
        if nonnegative and result[0] < 0:
            raise ValueError(f"{path}: domain_randomization.{key} must be non-negative")
        if positive and result[0] <= 0:
            raise ValueError(f"{path}: domain_randomization.{key} must be positive")
        return result

    def section(name, allowed_keys):
        value = settings.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"{path}: domain_randomization.{name} must be a mapping")
        unknown_keys = set(value) - set(allowed_keys)
        if unknown_keys:
            raise ValueError(f"{path}: unknown domain_randomization.{name} keys: {sorted(unknown_keys)}")
        return value

    event_updates = []

    foot_material = section(
        "foot_material",
        {"static_friction_range", "dynamic_friction_range", "restitution_range", "num_buckets"},
    )
    foot_params = {}
    if "static_friction_range" in foot_material:
        foot_params["static_friction_range"] = pair(
            foot_material["static_friction_range"], "foot_material.static_friction_range", nonnegative=True
        )
    if "dynamic_friction_range" in foot_material:
        foot_params["dynamic_friction_range"] = pair(
            foot_material["dynamic_friction_range"], "foot_material.dynamic_friction_range", nonnegative=True
        )
    if "restitution_range" in foot_material:
        foot_params["restitution_range"] = pair(
            foot_material["restitution_range"], "foot_material.restitution_range", nonnegative=True
        )
    if "num_buckets" in foot_material:
        buckets = foot_material["num_buckets"]
        if type(buckets) is not int or buckets <= 0:
            raise ValueError(f"{path}: domain_randomization.foot_material.num_buckets must be a positive integer")
        foot_params["num_buckets"] = buckets
    if foot_params:
        event_updates.append(("randomize_foot_material", foot_params))

    body_mass = section("body_mass", {"scale_range"})
    if "scale_range" in body_mass:
        event_updates.append(
            (
                "randomize_body_mass",
                {"mass_distribution_params": pair(body_mass["scale_range"], "body_mass.scale_range", positive=True)},
            )
        )

    body_com = section("body_com", {"x", "y", "z"})
    if body_com:
        com_range = {
            axis: pair(body_com[axis], f"body_com.{axis}") for axis in ("x", "y", "z") if axis in body_com
        }
        event_updates.append(("randomize_body_com", {"com_range": com_range}))

    reset_joints = section("reset_joints", {"position_range", "velocity_range"})
    if reset_joints:
        event_updates.append(
            (
                "reset_joints",
                {
                    key: pair(reset_joints[key], f"reset_joints.{key}")
                    for key in ("position_range", "velocity_range")
                    if key in reset_joints
                },
            )
        )

    reset_base = section("reset_base", {"pose_range", "velocity_range"})
    base_params = {}
    for group_name, group_keys in (
        ("pose_range", ("x", "y", "yaw")),
        ("velocity_range", ("x", "y", "z", "roll", "pitch", "yaw")),
    ):
        group = reset_base.get(group_name, {})
        if not isinstance(group, dict):
            raise ValueError(f"{path}: domain_randomization.reset_base.{group_name} must be a mapping")
        unknown_keys = set(group) - set(group_keys)
        if unknown_keys:
            raise ValueError(
                f"{path}: unknown domain_randomization.reset_base.{group_name} keys: {sorted(unknown_keys)}"
            )
        if group:
            base_params[group_name] = {
                key: pair(group[key], f"reset_base.{group_name}.{key}") for key in group
            }
    if base_params:
        event_updates.append(("reset_base", base_params))

    push = section("push", {"enabled", "interval_range_s", "velocity_range"})
    push_params = {}
    push_enabled = True
    if "enabled" in push:
        push_enabled = boolean(push["enabled"], "push.enabled")
    if "interval_range_s" in push:
        interval = pair(push["interval_range_s"], "push.interval_range_s", nonnegative=True)
    else:
        interval = None
    if "velocity_range" in push:
        velocity = push["velocity_range"]
        if not isinstance(velocity, dict):
            raise ValueError(f"{path}: domain_randomization.push.velocity_range must be a mapping")
        unknown_keys = set(velocity) - {"x", "y", "z", "roll", "pitch", "yaw"}
        if unknown_keys:
            raise ValueError(f"{path}: unknown domain_randomization.push.velocity_range keys: {sorted(unknown_keys)}")
        push_params["velocity_range"] = {
            key: pair(velocity[key], f"push.velocity_range.{key}") for key in velocity
        }
    if interval is not None or push_params:
        event_updates.append(("push_robot", push_params, interval))

    observation_noise = section(
        "observation_noise",
        {"enabled", "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"},
    )
    noise_enabled = None
    if "enabled" in observation_noise:
        noise_enabled = boolean(observation_noise["enabled"], "observation_noise.enabled")
    noise_updates = []
    for term_name in ("base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
        if term_name in observation_noise:
            noise_updates.append(
                (term_name, pair(observation_noise[term_name], f"observation_noise.{term_name}"))
            )

    enabled = boolean(settings.get("enabled", True), "enabled")
    if not enabled:
        _disable_domain_randomization_components(events, observations)
        return

    for update in event_updates:
        name, params = update[:2]
        term = getattr(events, name, None)
        if term is None:
            raise ValueError(f"{path}: event term {name!r} is unavailable")
        term.params.update(params)
        if name == "push_robot" and len(update) == 3 and update[2] is not None:
            term.interval_range_s = update[2]
    if not push_enabled:
        events.push_robot = None
    if noise_enabled is not None:
        observations.policy.enable_corruption = noise_enabled
    for term_name, noise_range in noise_updates:
        term = getattr(observations.policy, term_name, None)
        if term is None or term.noise is None:
            raise ValueError(f"{path}: observation term {term_name!r} has no noise configuration")
        term.noise.n_min, term.noise.n_max = noise_range


def _disable_domain_randomization_components(events, observations) -> None:
    """Disable randomization while preserving the mandatory scene reset event."""

    events.randomize_foot_material = None
    events.randomize_body_mass = None
    events.randomize_body_com = None
    events.push_robot = None
    events.reset_joints.params.update(position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
    events.reset_base.params.update(
        pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        velocity_range={
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    )
    observations.policy.enable_corruption = False


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
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.10, n_max=0.10))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=H2_PLUS_ACTIVE_JOINTS)},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=H2_PLUS_ACTIVE_JOINTS)},
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class H2PlusCommandsCfg:
    """Forward walking and yaw commands used by the first locomotion task."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.20),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.20, 0.20),
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
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "command_threshold": 0.1,
        },
    )
    feet_slide = RewTerm(
        func=mdp.h2_plus_feet_slide,
        weight=-0.05,
        params={
            "contact_force_threshold": 1.0,
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
    """Training randomization events shared by the H2 Plus locomotion tasks."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_foot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_pitch_link", "right_ankle_pitch_link"]
            ),
            "static_friction_range": (0.60, 1.00),
            "dynamic_friction_range": (0.40, 0.60),
            "restitution_range": (0.00, 0.05),
            "num_buckets": 32,
        },
    )
    randomize_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "torso_link"]),
            "mass_distribution_params": (1.0 / 1.05, 1.05),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    randomize_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "torso_link"]),
            "com_range": {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (-0.003, 0.003),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.98, 1.02), "velocity_range": (-0.05, 0.05)},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.03, 0.03)},
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.10, 0.10),
            },
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "velocity_range": {
                "x": (-0.20, 0.20),
                "y": (-0.20, 0.20),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        },
    )


@configclass
class H2PlusCurriculumCfg:
    """Command-difficulty curriculum for scenes without a terrain generator."""

    command_range = CurrTerm(
        func=mdp.h2_plus_command_curriculum,
        params={
            "start_step": 0,
            "ramp_steps": 600_000,
            "initial_lin_vel_x": 0.20,
            "target_lin_vel_x": 1.00,
            "initial_ang_vel_z": 0.20,
            "target_ang_vel_z": 1.00,
        },
    )
    # Reserved for a future TerrainImporter generator-backed rough-terrain task.
    terrain_levels: CurrTerm | None = None


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
    curriculum: H2PlusCurriculumCfg | None = H2PlusCurriculumCfg()
    recorders: H2PlusRecorderCfg = H2PlusRecorderCfg()

    def __post_init__(self):
        _apply_rewards_yaml(self.rewards, events=self.events, observations=self.observations)
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=self.decimation)
        if self.recorders is not None:
            # Safe unique fallback for environments created outside train.py/play.py.
            self.recorders.dataset_export_dir_path = str(
                Path(__file__).resolve().parents[3] / "output" / "sensors" / f"env_{time_ns()}"
            )


def _disable_training_randomization(env_cfg: H2PlusLocomotionEnvCfg) -> None:
    """Keep play deterministic while leaving training randomization in place."""

    env_cfg.curriculum = None
    _disable_domain_randomization_components(env_cfg.events, env_cfg.observations)


# Flat train/play configurations

@configclass
class H2PlusFlatEnvCfg(H2PlusLocomotionEnvCfg):
    scene: H2PlusSceneCfg = H2PlusSceneCfg(num_envs=1024)


@configclass
class H2PlusFlatEnvCfg_PLAY(H2PlusFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _disable_training_randomization(self)
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
        _disable_training_randomization(self)
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
        _disable_training_randomization(self)
        self.scene.num_envs = 16
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
