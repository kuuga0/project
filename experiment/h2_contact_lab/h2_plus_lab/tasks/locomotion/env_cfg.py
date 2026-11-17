"""Native ManagerBased environment; all task parameters live in YAML files.

MDP functions remain in mdp/. Registration remains in __init__.py. Hydra builds
native config objects; no reward/domain-specific YAML parsing is maintained here.
"""

from pathlib import Path
from time import time_ns
from typing import ClassVar

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from ...config import environment_data, instantiate_config


@configclass
class H2PlusLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    terrain_profile: ClassVar[str] = "flat"
    play_profile: ClassVar[bool] = False

    def __post_init__(self):
        data = environment_data(self.terrain_profile, self.play_profile)
        unknown = set(data) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown environment settings: {sorted(unknown)}")
        values = instantiate_config(data)
        for name, value in values.items():
            setattr(self, name, value)
        if self.recorders is not None:
            self.recorders.dataset_export_dir_path = str(
                Path(self.recorders.dataset_export_dir_path) / f"env_{time_ns()}"
            )


@configclass
class H2PlusFlatEnvCfg(H2PlusLocomotionEnvCfg):
    pass


@configclass
class H2PlusFlatEnvCfg_PLAY(H2PlusFlatEnvCfg):
    play_profile: ClassVar[bool] = True


@configclass
class H2PlusLowFrictionEnvCfg(H2PlusLocomotionEnvCfg):
    terrain_profile: ClassVar[str] = "low_friction"


@configclass
class H2PlusLowFrictionEnvCfg_PLAY(H2PlusLowFrictionEnvCfg):
    play_profile: ClassVar[bool] = True


@configclass
class H2PlusDeformableEnvCfg(H2PlusLocomotionEnvCfg):
    terrain_profile: ClassVar[str] = "deformable"


@configclass
class H2PlusDeformableEnvCfg_PLAY(H2PlusDeformableEnvCfg):
    play_profile: ClassVar[bool] = True
