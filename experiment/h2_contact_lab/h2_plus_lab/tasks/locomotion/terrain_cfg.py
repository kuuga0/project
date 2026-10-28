"""Terrain definitions for the H2 Plus development task.

The catalog contains ordinary hard ground, two adjacent rigid regions with
different friction materials, random-grid rough ground, and a supported
volume-deformable ground slab.  The multi-material and deformable variants
are scene assets rather than one ``TerrainImporterCfg`` because an importer
does not express those physics assignments faithfully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg


@dataclass(frozen=True)
class H2PlusSurfaceRegionCfg:
    """A rigid terrain region to be spawned by the scene layer later."""

    name: str
    size_m: tuple[float, float, float]
    center_m: tuple[float, float, float]
    static_friction: float
    dynamic_friction: float


@dataclass(frozen=True)
class H2PlusTerrainSpec:
    """A named terrain variant plus its optional scene assets."""

    name: str
    description: str
    importer_cfg: TerrainImporterCfg | None
    regions: tuple[H2PlusSurfaceRegionCfg, ...] = ()
    deformable_cfg: DeformableObjectCfg | None = None
    deformable_reserved: bool = True


HARD_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.8,
    dynamic_friction=0.7,
    restitution=0.0,
)

LOW_FRICTION_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.25,
    dynamic_friction=0.20,
    restitution=0.0,
)


H2_PLUS_HARD_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    terrain_generator=None,
    physics_material=HARD_MATERIAL,
)


H2_PLUS_ROUGH_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=20261027,
        curriculum=False,
        size=(4.0, 4.0),
        border_width=0.25,
        border_height=1.0,
        num_rows=1,
        num_cols=1,
        color_scheme="height",
        horizontal_scale=0.10,
        vertical_scale=0.005,
        slope_threshold=None,
        sub_terrains={
            "random_grid_roughness": HfRandomUniformTerrainCfg(
                proportion=1.0,
                noise_range=(-0.08, 0.08),
                noise_step=0.02,
                downsampled_scale=0.25,
                size=(4.0, 4.0),
                border_width=0.25,
                horizontal_scale=0.10,
                vertical_scale=0.005,
                slope_threshold=None,
            )
        },
        difficulty_range=(0.0, 1.0),
        use_cache=False,
    ),
    physics_material=HARD_MATERIAL,
)


# Isaac Lab 2.3.x uses these deformable configuration names.  The
# ``deformable_props`` slot enables a real PhysX deformable body; this is not a
# rigid proxy or a height-field perturbation.
H2_PLUS_DEFORMABLE_GROUND_CFG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Terrain/FlexibleGround",
    spawn=sim_utils.MeshCuboidCfg(
        size=(4.0, 4.0, 0.20),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            deformable_enabled=True,
            rest_offset=0.0,
            contact_offset=0.001,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(
            density=650.0,
            static_friction=0.8,
            dynamic_friction=0.7,
            youngs_modulus=250000.0,
            poissons_ratio=0.35,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.42, 0.28)),
    ),
    # The scene adds a rigid backing at z=-0.20.  The slab starts with its
    # bottom face on that backing and can deform above it.
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.10)),
    debug_vis=False,
)


H2_PLUS_TERRAIN_CATALOG: dict[str, H2PlusTerrainSpec] = {
    "ordinary_hard": H2PlusTerrainSpec(
        name="ordinary_hard",
        description="Uniform rigid plane used as the reference condition.",
        importer_cfg=H2_PLUS_HARD_TERRAIN_CFG,
        regions=(
            H2PlusSurfaceRegionCfg(
                name="hard_ground",
                size_m=(4.0, 4.0, 0.10),
                center_m=(0.0, 0.0, -0.05),
                static_friction=0.8,
                dynamic_friction=0.7,
            ),
        ),
    ),
    "low_friction_difference": H2PlusTerrainSpec(
        name="low_friction_difference",
        description=(
            "Two adjacent rigid regions: positive Y is low friction and negative Y "
            "is the ordinary reference material."
        ),
        # The scene layer must spawn these regions separately.  A single plane
        # importer cannot express the two material assignments faithfully.
        importer_cfg=None,
        regions=(
            H2PlusSurfaceRegionCfg(
                name="positive_y_low_friction",
                size_m=(4.0, 2.0, 0.10),
                center_m=(0.0, 1.0, -0.05),
                static_friction=0.25,
                dynamic_friction=0.20,
            ),
            H2PlusSurfaceRegionCfg(
                name="negative_y_ordinary_friction",
                size_m=(4.0, 2.0, 0.10),
                center_m=(0.0, -1.0, -0.05),
                static_friction=0.8,
                dynamic_friction=0.7,
            ),
        ),
    ),
    "local_roughness": H2PlusTerrainSpec(
        name="local_roughness",
        description=(
            "Random-grid height field for local support changes; it is not a single "
            "raised edge or a single fixed height offset."
        ),
        importer_cfg=H2_PLUS_ROUGH_TERRAIN_CFG,
        regions=(),
    ),
    "deformable_ground": H2PlusTerrainSpec(
        name="deformable_ground",
        description=(
            "A four-by-four-meter volume-deformable slab on a rigid backing. "
            "The slab uses PhysX FEM material parameters rather than a rigid proxy."
        ),
        importer_cfg=None,
        regions=(),
        deformable_cfg=H2_PLUS_DEFORMABLE_GROUND_CFG,
        deformable_reserved=False,
    ),
}


def get_terrain_spec(name: str) -> H2PlusTerrainSpec:
    """Return a named terrain specification for the later scene builder."""

    try:
        return H2_PLUS_TERRAIN_CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"Unknown H2 Plus terrain variant: {name!r}") from exc


def build_region_material(region: H2PlusSurfaceRegionCfg) -> Any:
    """Build a rigid material config for a custom region spawn.

    This helper is kept separate from the scene so that a later scene builder
    can choose whether to use cuboids, meshes, or a custom terrain spawner.
    """

    return sim_utils.RigidBodyMaterialCfg(
        static_friction=region.static_friction,
        dynamic_friction=region.dynamic_friction,
        restitution=0.0,
    )


__all__ = [
    "H2PlusSurfaceRegionCfg",
    "H2PlusTerrainSpec",
    "H2_PLUS_DEFORMABLE_GROUND_CFG",
    "H2_PLUS_HARD_TERRAIN_CFG",
    "H2_PLUS_ROUGH_TERRAIN_CFG",
    "H2_PLUS_TERRAIN_CATALOG",
    "build_region_material",
    "get_terrain_spec",
]
