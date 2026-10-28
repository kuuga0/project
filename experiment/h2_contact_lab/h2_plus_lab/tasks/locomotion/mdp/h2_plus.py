"""H2 Plus-specific MDP terms.

The generic Isaac Lab locomotion terms remain reusable.  These small wrappers
only handle the project choice of keeping left and right foot sensors as
separate sensor objects instead of pretending that FOOTRIX is already
available as a body-level ContactSensor.
"""

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _body_slice(body_ids):
    """Return an index that selects resolved sensor body IDs."""

    return slice(None) if body_ids is None else body_ids


def h2_plus_feet_air_time_positive_biped(
    env,
    command_name: str,
    threshold: float,
    sensor_names: Sequence[str] = ("left_foot_contact", "right_foot_contact"),
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
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def h2_plus_feet_slide(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_names: Sequence[str] = ("left_foot_contact", "right_foot_contact"),
) -> torch.Tensor:
    """Penalize tangential velocity of H2 feet while they are in contact."""

    contacts = []
    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        foot_force_history = sensor.data.net_forces_w_history[:, :, 0, :]
        contacts.append(foot_force_history.norm(dim=-1).amax(dim=1) > 1.0)
    contacts = torch.stack(contacts, dim=1)

    asset = env.scene[asset_cfg.name]
    foot_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return torch.sum(foot_velocity * contacts, dim=1)


def h2_plus_body_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return whether pelvis or torso contact exceeds the force threshold."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.net_forces_w_history[:, :, _body_slice(sensor_cfg.body_ids), :]
    force_norm = force_history.norm(dim=-1)
    return force_norm.amax(dim=(1, 2)) > threshold


__all__ = [
    "h2_plus_body_contact",
    "h2_plus_feet_air_time_positive_biped",
    "h2_plus_feet_slide",
]
