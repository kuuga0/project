"""H2 Plus locomotion tasks.

Add a gym.register call here when introducing another environment configuration.
Keep entry points as strings so registration does not initialize Isaac Sim.
"""

import gymnasium as gym

gym.register(
    id="Isaac-H2Plus-Blind-Locomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:H2PlusFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-H2Plus-Blind-Locomotion-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:H2PlusFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-H2Plus-Blind-Locomotion-LowFriction-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:H2PlusLowFrictionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-H2Plus-Blind-Locomotion-LowFriction-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:H2PlusLowFrictionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-H2Plus-Blind-Locomotion-Deformable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:H2PlusDeformableEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-H2Plus-Blind-Locomotion-Deformable-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:H2PlusDeformableEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
    },
)
