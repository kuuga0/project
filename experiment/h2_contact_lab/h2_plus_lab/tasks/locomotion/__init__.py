"""H2 Plus locomotion task registration."""

from __future__ import annotations

import gymnasium as gym

from .agents.h2_plus import agents as h2_plus_agents


_H2_PLUS_TASK_ID = "Isaac-H2Plus-Blind-Locomotion-v0"
_H2_PLUS_PLAY_TASK_ID = "Isaac-H2Plus-Blind-Locomotion-Play-v0"
_H2_PLUS_LOW_FRICTION_TASK_ID = "Isaac-H2Plus-Blind-Locomotion-LowFriction-v0"
_H2_PLUS_DEFORMABLE_TASK_ID = "Isaac-H2Plus-Blind-Locomotion-Deformable-v0"


def _register_h2_plus_tasks() -> None:
    """Register project-local tasks when the package is imported."""

    if _H2_PLUS_TASK_ID not in gym.registry:
        gym.register(
            id=_H2_PLUS_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.h2_plus_env_cfg:H2PlusLocomotionEnvCfg",
                "rsl_rl_cfg_entry_point": f"{h2_plus_agents.__name__}.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
            },
        )
    if _H2_PLUS_PLAY_TASK_ID not in gym.registry:
        gym.register(
            id=_H2_PLUS_PLAY_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.h2_plus_env_cfg:H2PlusLocomotionEnvCfg_PLAY",
                "rsl_rl_cfg_entry_point": f"{h2_plus_agents.__name__}.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
            },
        )
    if _H2_PLUS_LOW_FRICTION_TASK_ID not in gym.registry:
        gym.register(
            id=_H2_PLUS_LOW_FRICTION_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.h2_plus_env_cfg:H2PlusLowFrictionLocomotionEnvCfg",
                "rsl_rl_cfg_entry_point": f"{h2_plus_agents.__name__}.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
            },
        )
    if _H2_PLUS_DEFORMABLE_TASK_ID not in gym.registry:
        gym.register(
            id=_H2_PLUS_DEFORMABLE_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.h2_plus_env_cfg:H2PlusDeformableLocomotionEnvCfg",
                "rsl_rl_cfg_entry_point": f"{h2_plus_agents.__name__}.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
            },
        )


_register_h2_plus_tasks()

__all__ = [
    "_H2_PLUS_DEFORMABLE_TASK_ID",
    "_H2_PLUS_LOW_FRICTION_TASK_ID",
    "_H2_PLUS_PLAY_TASK_ID",
    "_H2_PLUS_TASK_ID",
]
