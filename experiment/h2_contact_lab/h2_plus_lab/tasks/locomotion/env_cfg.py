"""Scene-level configuration for H2 Plus locomotion development.

The first policy is blind to terrain geometry, but it still uses foot and
body contact sensors.  The RayCaster and FOOTRIX entries remain deferred
interfaces.  The torso and foot IMUs are instantiated for recording and are
excluded from the first policy observation group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
    H2_PLUS_RAY_CASTER_CFG,
    H2_PLUS_RIGHT_FOOT_IMU_CFG,
    H2_PLUS_RIGHT_FOOT_CONTACT_CFG,
    H2_PLUS_SENSOR_INTERFACES,
    H2_PLUS_TORSO_IMU_CFG,
)


@dataclass(frozen=True)
class H2PlusDeferredInterfaces:
    """Named slots for active and intentionally deferred interfaces."""

    contact_sensor_cfg: Any | None = None
    ray_caster_cfg: Any | None = None
    imu_cfg: Any | None = None
    footrix_adapter_cfg: Any | None = None
    actions_cfg: Any | None = None
    observations_cfg: Any | None = None
    commands_cfg: Any | None = None
    rewards_cfg: Any | None = None
    terminations_cfg: Any | None = None
    events_cfg: Any | None = None


H2_PLUS_DEFERRED_INTERFACES = H2PlusDeferredInterfaces(
    contact_sensor_cfg=H2_PLUS_SENSOR_INTERFACES,
    # The ray caster is defined but deliberately not attached to the first
    # blind-walking scene.  A later terrain-aware scene can use this object.
    ray_caster_cfg=H2_PLUS_RAY_CASTER_CFG,
    imu_cfg=H2_PLUS_SENSOR_INTERFACES.imu_cfgs,
    footrix_adapter_cfg=H2_PLUS_SENSOR_INTERFACES.footrix,
)


@configclass
class H2PlusSceneCfg(InteractiveSceneCfg):
    """Reusable robot, terrain and lighting scene for later environments."""

    num_envs: int = 1
    env_spacing: float = 3.0

    robot: ArticulationCfg = H2_PLUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    terrain: TerrainImporterCfg = H2_PLUS_TERRAIN_CATALOG["ordinary_hard"].importer_cfg

    left_foot_contact: ContactSensorCfg = H2_PLUS_LEFT_FOOT_CONTACT_CFG
    right_foot_contact: ContactSensorCfg = H2_PLUS_RIGHT_FOOT_CONTACT_CFG
    body_contact: ContactSensorCfg = H2_PLUS_BODY_CONTACT_CFG
    torso_imu: ImuCfg = H2_PLUS_TORSO_IMU_CFG
    left_foot_imu: ImuCfg = H2_PLUS_LEFT_FOOT_IMU_CFG
    right_foot_imu: ImuCfg = H2_PLUS_RIGHT_FOOT_IMU_CFG

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8)),
    )


@configclass
class H2PlusLowFrictionSceneCfg(H2PlusSceneCfg):
    """Scene with two adjacent rigid regions and genuinely different materials."""

    # A single TerrainImporter cannot carry both materials, so this scene
    # replaces it with two independently spawned rigid cuboids.
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

    left_foot_contact = H2_PLUS_LEFT_FOOT_CONTACT_CFG.replace(
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Terrain/ReferenceRegion",
            "{ENV_REGEX_NS}/Terrain/LowFrictionRegion",
        ]
    )
    right_foot_contact = H2_PLUS_RIGHT_FOOT_CONTACT_CFG.replace(
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Terrain/ReferenceRegion",
            "{ENV_REGEX_NS}/Terrain/LowFrictionRegion",
        ]
    )
    body_contact = H2_PLUS_BODY_CONTACT_CFG.replace(
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Terrain/ReferenceRegion",
            "{ENV_REGEX_NS}/Terrain/LowFrictionRegion",
        ]
    )


@configclass
class H2PlusDeformableSceneCfg(H2PlusSceneCfg):
    """Scene with a real volume-deformable slab over a rigid backing."""

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
    deformable_ground = H2_PLUS_DEFORMABLE_GROUND_CFG

    left_foot_contact = H2_PLUS_LEFT_FOOT_CONTACT_CFG.replace(
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Terrain/FlexibleGround",
            "{ENV_REGEX_NS}/Terrain/FlexibleBacking",
        ]
    )
    right_foot_contact = H2_PLUS_RIGHT_FOOT_CONTACT_CFG.replace(
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Terrain/FlexibleGround",
            "{ENV_REGEX_NS}/Terrain/FlexibleBacking",
        ]
    )
    body_contact = H2_PLUS_BODY_CONTACT_CFG.replace(
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Terrain/FlexibleGround",
            "{ENV_REGEX_NS}/Terrain/FlexibleBacking",
        ]
    )


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


__all__ = [
    "H2PlusDeferredInterfaces",
    "H2PlusDeformableSceneCfg",
    "H2PlusLowFrictionSceneCfg",
    "H2_PLUS_DEFERRED_INTERFACES",
    "H2PlusSceneCfg",
    "scene_for_terrain",
]
