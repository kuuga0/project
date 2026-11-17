# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), angular_command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize posture drift only when both translation and yaw are near zero."""
    command = env.command_manager.get_command(command_name)
    standing = (torch.norm(command[:, :2], dim=1) < command_threshold) & (
        command[:, 2].abs() < angular_command_threshold
    )
    return mdp.joint_deviation_l1(env, asset_cfg) * standing


def h2_plus_feet_air_time_positive_biped(
    env,
    command_name: str,
    threshold: float,
    command_threshold: float = 0.05,
    sensor_names: Sequence[str] = ("left_foot_contact", "right_foot_contact"),
    angular_command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward alternating two-foot stance using the two H2 foot sensors."""

    air_time = []
    contact_time = []
    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        air_time.append(sensor.data.current_air_time[:, 0])
        contact_time.append(sensor.data.current_contact_time[:, 0])

    air_time = torch.stack(air_time, dim=1)
    contact_time = torch.stack(contact_time, dim=1)
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    command = env.command_manager.get_command(command_name)
    moving = (torch.norm(command[:, :2], dim=1) > command_threshold) | (
        command[:, 2].abs() > angular_command_threshold
    )
    reward *= moving
    return reward


def h2_plus_feet_slide(
    env,
    asset_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
    sensor_names: Sequence[str] = ("left_foot_contact", "right_foot_contact"),
) -> torch.Tensor:
    """Penalize tangential velocity of H2 feet while they are in contact."""

    contacts = []
    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        contacts.append((sensor.data.net_forces_w[:, 0, :].norm(dim=-1) > contact_force_threshold)
                        & (sensor.data.current_contact_time[:, 0] > 0.0))
    contacts = torch.stack(contacts, dim=1)

    asset = env.scene[asset_cfg.name]
    foot_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return torch.sum(foot_velocity * contacts, dim=1)


def h2_plus_joint_reference_l1(env, reference_angle: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint-coordinate reference independent of startup pose and action offset."""
    positions = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return (positions - reference_angle).abs().sum(dim=1)


def h2_plus_joint_reference_deadband_l2(
    env, asset_cfg: SceneEntityCfg, reference_angle: float, tolerance: float, std: float,
) -> torch.Tensor:
    """Allow small posture adjustments, penalizing only excessive joint deviation."""
    position = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    excess = ((position - reference_angle).abs() - tolerance).clamp(min=0)
    return (excess / std).square().mean(dim=1)


def h2_plus_torso_orientation_l2(env, asset_cfg: SceneEntityCfg,
                               roll_weight: float = 1.0, pitch_weight: float = 1.0) -> torch.Tensor:
    """Weight lateral and forward torso tilt separately via projected gravity."""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    gravity = asset.data.GRAVITY_VEC_W[:, None, :].expand(-1, quat.shape[1], -1)
    projected = quat_apply_inverse(quat.reshape(-1, 4), gravity.reshape(-1, 3))
    penalty = pitch_weight * projected[:, 0].square() + roll_weight * projected[:, 1].square()
    return penalty.reshape(quat.shape[:2]).mean(-1)


def h2_plus_torso_roll_velocity_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Damp lateral rocking about torso's heading-aligned horizontal forward axis.

    A yaw-only frame keeps pure world-vertical turning out of this penalty,
    including when the torso is pitched. This is angular rocking, not lateral
    translation of the torso.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    angular_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    angular_heading = quat_apply_inverse(yaw_quat(quat.reshape(-1, 4)), angular_w.reshape(-1, 3))
    return angular_heading[:, 0].square().reshape(quat.shape[:2]).mean(-1)


def h2_plus_body_lateral_position_l2(
    env, asset_cfg: SceneEntityCfg, feet_cfg: SceneEntityCfg, command_name: str,
    tolerance: float, std: float, angular_command_threshold: float,
) -> torch.Tensor:
    """Limit sideways body excursion relative to foot midpoint during straight motion."""
    asset = env.scene[asset_cfg.name]
    midpoint = env.scene[feet_cfg.name].data.body_pos_w[:, feet_cfg.body_ids, :].mean(dim=1)
    offset = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - midpoint[:, None, :]
    heading = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, offset.shape[1], -1)
    local = quat_apply_inverse(heading.reshape(-1, 4), offset.reshape(-1, 3)).reshape_as(offset)
    excess = (local[:, :, 1].abs() - tolerance).clamp(min=0)
    straight = env.command_manager.get_command(command_name)[:, 2].abs() < angular_command_threshold
    return (excess / std).square().mean(dim=1) * straight


def h2_plus_body_lateral_velocity_l2(
    env, asset_cfg: SceneEntityCfg, command_name: str, tolerance: float, std: float,
    angular_command_threshold: float,
) -> torch.Tensor:
    """Limit sideways translation, allowing commanded lateral motion and small balance shifts."""
    asset = env.scene[asset_cfg.name]
    velocity = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    heading = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, velocity.shape[1], -1)
    local = quat_apply_inverse(heading.reshape(-1, 4), velocity.reshape(-1, 3)).reshape_as(velocity)
    command = env.command_manager.get_command(command_name)
    excess = ((local[:, :, 1] - command[:, 1:2]).abs() - tolerance).clamp(min=0)
    return (excess / std).square().mean(dim=1) * (command[:, 2].abs() < angular_command_threshold)


def h2_plus_arm_leg_coordination_exp(
    env, asset_cfg: SceneEntityCfg, feet_cfg: SceneEntityCfg, command_name: str,
    phase_length: float, shoulder_amplitude: float, elbow_amplitude: float, std: float,
    command_threshold: float, angular_command_threshold: float,
) -> torch.Tensor:
    """Soft fore-aft arm cue from measured left/right step positions, no gait clock.

    Explicit joint order: left/right shoulder pitch, then left/right elbow.
    For this URDF positive shoulder pitch swings the arm backward. Thus a
    forward ipsilateral foot cues a backward shoulder and the opposite elbow
    excursion. This is a heuristic coordination reference, not motion imitation.
    """
    asset = env.scene[asset_cfg.name]
    foot_offset = env.scene[feet_cfg.name].data.body_pos_w[:, feet_cfg.body_ids, :] - asset.data.root_pos_w[:, None, :]
    heading = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, foot_offset.shape[1], -1)
    feet_local = quat_apply_inverse(heading.reshape(-1, 4), foot_offset.reshape(-1, 3)).reshape_as(foot_offset)
    phase = torch.tanh((feet_local[:, 0, 0] - feet_local[:, 1, 0]) / (2*phase_length))
    amplitudes = phase.new_tensor([shoulder_amplitude, -shoulder_amplitude, -elbow_amplitude, elbow_amplitude])
    reference = phase[:, None] * amplitudes
    actual = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = ((actual - reference) / std).square().mean(dim=1)
    command = env.command_manager.get_command(command_name)
    moving = (command[:, :2].norm(dim=1) > command_threshold) | (command[:, 2].abs() > angular_command_threshold)
    return torch.exp(-error) * phase.abs() * moving


def h2_plus_contact_foot_flat_reward(
    env, asset_cfg: SceneEntityCfg, tilt_std: float, contact_force_threshold: float,
    settle_time_s: float,
    sensor_names: Sequence[str] = ("left_foot_contact", "right_foot_contact"),
) -> torch.Tensor:
    """Reward upright level feet during established contact, ignoring yaw.

    Local +z is the sole normal for these ankle_pitch bodies. This is an
    orientation proxy on horizontal ground, not a contact-area measurement.
    Average over eligible contacting feet so double support adds no extra bonus.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    gravity = asset.data.GRAVITY_VEC_W[:, None, :].expand(-1, quat.shape[1], -1)
    projected = quat_apply_inverse(quat.reshape(-1, 4), gravity.reshape(-1, 3)).reshape(
        quat.shape[0], quat.shape[1], 3
    )
    score = torch.exp(-projected[:, :, :2].square().sum(-1) / tilt_std**2)
    score = score * (projected[:, :, 2] < 0)  # upside-down feet must not score as flat
    contact = torch.stack([
        (env.scene.sensors[name].data.net_forces_w[:, 0, :].norm(dim=-1) > contact_force_threshold)
        & (env.scene.sensors[name].data.current_contact_time[:, 0] > settle_time_s)
        for name in sensor_names
    ], dim=1)
    return (score * contact).sum(dim=1) / contact.sum(dim=1).clamp(min=1)


def h2_plus_torso_vertical_velocity_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World vertical torso speed penalty; no prescribed absolute height."""
    return env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, 2].square().mean(dim=1)


def h2_plus_joint_motion_reward(
    env, command_name: str, asset_cfg: SceneEntityCfg, speed_scale: float,
    speed_deadband: float, command_threshold: float, angular_command_threshold: float,
    tracking_std: float, yaw_tracking_std: float,
) -> torch.Tensor:
    """Bounded measured activity bonus gated by motion commands and velocity tracking.

    Peaks at deadband+speed_scale and decays for excessive speed; this does not
    guarantee useful swing timing or coordination. No posture reference is imposed.
    """
    asset = env.scene[asset_cfg.name]
    speed = asset.data.joint_vel[:, asset_cfg.joint_ids].abs()
    x = ((speed - speed_deadband).clamp(min=0) / speed_scale).square().clamp(max=100)
    activity = (x * torch.exp(1.0 - x)).mean(dim=1)
    command = env.command_manager.get_command(command_name)
    moving = (command[:, :2].norm(dim=1) > command_threshold) | (
        command[:, 2].abs() > angular_command_threshold
    )
    xy_error = (command[:, :2] - asset.data.root_lin_vel_b[:, :2]).square().sum(dim=1)
    yaw_error = (command[:, 2] - asset.data.root_ang_vel_b[:, 2]).square()
    tracking = torch.exp(-xy_error / tracking_std**2 - yaw_error / yaw_tracking_std**2)
    return activity * tracking * moving


class H2PlusJointTargetLimitPenalty(ManagerTermBase):
    """Penalize unreachable position requests before native action clipping.

    Actual joint angles alone cannot expose a policy leaning into a hard stop.
    Read native affine preprocessing before its clip, preserving the raw actions.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        action = env.action_manager.get_term(cfg.params["action_name"])
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        _, action_names = asset.find_joints(action.cfg.joint_names, preserve_order=action.cfg.preserve_order)
        self._action = action
        self._columns = [action_names.index(asset.joint_names[i]) for i in asset_cfg.joint_ids]

    def __call__(self, env, action_name: str, asset_cfg: SceneEntityCfg, std: float,
                 margin_ratio: float = 0.0) -> torch.Tensor:
        del action_name
        if not 0.0 <= margin_ratio < 0.5:
            raise ValueError("Joint target margin_ratio must be in [0, 0.5)")
        # JointAction has no public pre-clip target; use its native scale/offset
        # tensors so regex scales and default-pose offsets stay exactly aligned.
        raw_target = self._action.raw_actions * self._action._scale + self._action._offset
        target = raw_target[:, self._columns]
        limits = env.scene[asset_cfg.name].data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
        margin = (limits[:, :, 1] - limits[:, :, 0]) * margin_ratio
        violation = (limits[:, :, 0] + margin - target).clamp(min=0) + (target - limits[:, :, 1] + margin).clamp(min=0)
        return torch.log1p((violation / std).square()).mean(dim=1)


class H2PlusJointInactivity(ManagerTermBase):
    """Penalize insufficient joint excursion over a recent moving-command window.

    Measuring range instead of instantaneous speed allows stance and swing
    reversals. A held joint still counts as inactive; there is no angle target.
    Native RewardManager resets this history separately for each reset environment.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        steps = max(2, math.ceil(cfg.params["window_s"] / env.step_dt) + 1)
        asset_cfg = cfg.params["asset_cfg"]
        positions = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
        if positions.shape[1] == 0:
            raise ValueError("joint_inactivity requires at least one joint")
        self._history = torch.zeros((steps, *positions.shape), device=positions.device, dtype=positions.dtype)
        self._age = torch.zeros(env.num_envs, dtype=torch.long, device=positions.device)
        self._cursor = 0

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._history[:, ids] = 0
        self._age[ids] = 0

    @torch.no_grad()
    def __call__(self, env, command_name: str, asset_cfg: SceneEntityCfg, window_s: float,
                 min_excursion: float, command_threshold: float, angular_command_threshold: float) -> torch.Tensor:
        del window_s  # Config fixes buffer duration when the native term is constructed.
        self._history[self._cursor] = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
        self._cursor = (self._cursor + 1) % self._history.shape[0]
        command = env.command_manager.get_command(command_name)
        moving = (command[:, :2].norm(dim=1) > command_threshold) | (
            command[:, 2].abs() > angular_command_threshold
        )
        self._age = torch.where(moving, (self._age + 1).clamp(max=self._history.shape[0]), 0)
        excursion = self._history.amax(dim=0) - self._history.amin(dim=0)
        deficit = (1.0 - excursion / min_excursion).clamp(min=0.0, max=1.0).square().mean(dim=1)
        return deficit * (self._age >= self._history.shape[0])


def h2_plus_swing_clearance_exp(
    env, command_name: str, asset_cfg: SceneEntityCfg, target_height: float,
    height_std: float, speed_scale: float, command_threshold: float, angular_command_threshold: float,
    sensor_names: Sequence[str] = ("left_foot_contact", "right_foot_contact"),
) -> torch.Tensor:
    """Reward target swing height relative to support foot, weighted by swing speed.

    Uses no terrain scan. This is a relative ankle-height proxy, not sole clearance
    above arbitrary terrain. Exactly one foot must be supporting; no reward for hopping.
    """
    asset = env.scene[asset_cfg.name]
    height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    speed = (asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1) / speed_scale).clamp(max=1.0)
    contact = torch.stack([
        env.scene.sensors[name].data.current_contact_time[:, 0] > 0 for name in sensor_names
    ], dim=1)
    single = contact.sum(dim=1) == 1
    support_height = (height * contact).sum(dim=1, keepdim=True)
    error = ((height - support_height - target_height) / height_std).square()
    command = env.command_manager.get_command(command_name)
    moving = (command[:, :2].norm(dim=1) > command_threshold) | (
        command[:, 2].abs() > angular_command_threshold
    )
    return (torch.exp(-error) * (~contact) * speed).sum(dim=1) * single * moving


def h2_plus_joint_limit_margin_l2(env, asset_cfg: SceneEntityCfg, margin_ratio: float) -> torch.Tensor:
    """Zero in the central joint range; quadratic cost only near either limit."""
    if not 0.0 < margin_ratio < 0.5:
        raise ValueError("Joint limit margin_ratio must be in (0, 0.5)")
    asset = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    position = asset.data.joint_pos[:, asset_cfg.joint_ids]
    margin = (limits[:, :, 1] - limits[:, :, 0]) * margin_ratio
    lower = (limits[:, :, 0] + margin - position).clamp(min=0) / margin
    upper = (position - limits[:, :, 1] + margin).clamp(min=0) / margin
    return (lower.square() + upper.square()).mean(dim=1)


def h2_plus_foot_heading_l2(env, asset_cfg: SceneEntityCfg, tolerance: float, std: float) -> torch.Tensor:
    """Align each foot's horizontal +x heading with pelvis heading, independently.

    Relative angles are wrapped at pi; yawing the entire robot adds no cost.
    Both stance and swing feet participate. Roll/pitch are handled by other terms.
    """
    asset = env.scene[asset_cfg.name]
    quats = torch.cat((asset.data.root_quat_w[:, None, :],
                       asset.data.body_quat_w[:, asset_cfg.body_ids, :]), dim=1)
    w, x, y, z = quats.unbind(dim=-1)
    heading = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y.square() + z.square()))
    difference = heading[:, 1:] - heading[:, :1]
    relative = torch.atan2(torch.sin(difference), torch.cos(difference))
    excess = (relative.abs() - tolerance).clamp(min=0)
    return (excess / std).square().mean(dim=1)


def h2_plus_stance_width_l2(env, asset_cfg: SceneEntityCfg, min_width: float,
                          max_width: float, width_std: float) -> torch.Tensor:
    """Penalize excessive or crossed lateral stance, allowing a band of widths.

    Measures ankle-link origins in pelvis yaw coordinates, not sole outer edges.
    """
    if min_width > max_width:
        raise ValueError("stance_width requires min_width <= max_width")
    asset = env.scene[asset_cfg.name]
    offset = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, None, :]
    quat = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, offset.shape[1], -1)
    local = quat_apply_inverse(quat.reshape(-1, 4), offset.reshape(-1, 3)).reshape_as(offset)
    width = local[:, 0, 1] - local[:, 1, 1]  # explicit left/right order preserves crossing sign
    error = (min_width - width).clamp(min=0) + (width - max_width).clamp(min=0)
    return (error / width_std).square()


def h2_plus_feet_lateral_symmetry_l2(
    env, command_name: str, asset_cfg: SceneEntityCfg, lateral_std: float,
    command_threshold: float, angular_command_threshold: float,
) -> torch.Tensor:
    """Center feet laterally about pelvis during straight motion; allow alternating x.

    Does not force simultaneous joint motion or equal sagittal joint angles.
    """
    asset = env.scene[asset_cfg.name]
    offset = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, None, :]
    quat = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, offset.shape[1], -1)
    local = quat_apply_inverse(quat.reshape(-1, 4), offset.reshape(-1, 3)).reshape_as(offset)
    command = env.command_manager.get_command(command_name)
    straight = (command[:, :2].norm(dim=1) > command_threshold) & (
        command[:, 2].abs() < angular_command_threshold
    )
    return (local[:, :, 1].sum(dim=1) / lateral_std).square() * straight
