"""Curriculum terms for H2 Plus locomotion tasks."""

from __future__ import annotations

from typing import Sequence


def h2_plus_command_curriculum(
    env,
    env_ids: Sequence[int],
    start_step: int = 0,
    ramp_steps: int = 600_000,
    initial_lin_vel_x: float = 0.20,
    target_lin_vel_x: float = 1.00,
    initial_ang_vel_z: float = 0.20,
    target_ang_vel_z: float = 1.00,
) -> dict[str, float]:
    """Linearly widen forward and yaw command ranges during training.

    The term changes only the command manager configuration.  It does not
    alter terrain geometry, rewards, observations, or the policy action space.
    ``env_ids`` is accepted for the standard curriculum-manager interface;
    command ranges are intentionally shared by all parallel environments.
    """

    del env_ids
    progress = (int(getattr(env, "common_step_counter", 0)) - start_step) / max(ramp_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    max_lin_vel_x = initial_lin_vel_x + progress * (target_lin_vel_x - initial_lin_vel_x)
    max_ang_vel_z = initial_ang_vel_z + progress * (target_ang_vel_z - initial_ang_vel_z)

    command_cfg = env.command_manager.get_term_cfg("base_velocity")
    command_cfg.ranges.lin_vel_x = (0.0, max_lin_vel_x)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (-max_ang_vel_z, max_ang_vel_z)
    command_cfg.ranges.heading = (0.0, 0.0)
    env.command_manager.set_term_cfg("base_velocity", command_cfg)

    return {
        "progress": progress,
        "lin_vel_x_max": max_lin_vel_x,
        "ang_vel_z_abs_max": max_ang_vel_z,
    }
