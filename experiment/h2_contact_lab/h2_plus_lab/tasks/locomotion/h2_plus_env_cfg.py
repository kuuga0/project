"""Manager-based blind locomotion configuration for H2 Plus.

This is the first task-level MDP configuration.  The policy observes body and
joint state plus contact-derived terms, but no terrain height scan.  The
RayCaster configuration remains available in ``sensor_cfg.py`` for a later
terrain-aware ablation.  Torso and foot IMUs are attached to the scene for
recording only and are intentionally absent from the policy observation group.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import H2PlusDeformableSceneCfg, H2PlusLowFrictionSceneCfg, H2PlusSceneCfg


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
    """Initial locomotion reward weights; all remain tuning parameters."""

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
class H2PlusLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """H2 Plus blind locomotion environment assembled from project modules."""

    scene: H2PlusSceneCfg = H2PlusSceneCfg()
    observations: H2PlusObservationsCfg = H2PlusObservationsCfg()
    actions: H2PlusActionsCfg = H2PlusActionsCfg()
    commands: H2PlusCommandsCfg = H2PlusCommandsCfg()
    rewards: H2PlusRewardsCfg = H2PlusRewardsCfg()
    terminations: H2PlusTerminationsCfg = H2PlusTerminationsCfg()
    events: H2PlusEventsCfg = H2PlusEventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=self.decimation)


@configclass
class H2PlusLocomotionEnvCfg_PLAY(H2PlusLocomotionEnvCfg):
    """Small deterministic play scene for a later trained checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class H2PlusLowFrictionLocomotionEnvCfg(H2PlusLocomotionEnvCfg):
    """Blind locomotion on two adjacent rigid regions with different friction."""

    scene: H2PlusLowFrictionSceneCfg = H2PlusLowFrictionSceneCfg()


@configclass
class H2PlusDeformableLocomotionEnvCfg(H2PlusLocomotionEnvCfg):
    """Blind locomotion on the volume-deformable ground slab."""

    scene: H2PlusDeformableSceneCfg = H2PlusDeformableSceneCfg()


__all__ = [
    "H2PlusActionsCfg",
    "H2PlusCommandsCfg",
    "H2PlusEventsCfg",
    "H2PlusLocomotionEnvCfg",
    "H2PlusLocomotionEnvCfg_PLAY",
    "H2PlusLowFrictionLocomotionEnvCfg",
    "H2PlusDeformableLocomotionEnvCfg",
    "H2PlusObservationsCfg",
    "H2PlusRewardsCfg",
    "H2PlusTerminationsCfg",
]
