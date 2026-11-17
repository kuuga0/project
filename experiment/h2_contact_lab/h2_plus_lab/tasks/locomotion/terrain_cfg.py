"""Reusable native terrain configurations; parameters live in terrain_cfg.yaml."""

from ...config import build_config

HARD_MATERIAL = build_config("terrain.hard_material")
LOW_FRICTION_MATERIAL = build_config("terrain.low_friction_material")
H2_PLUS_HARD_TERRAIN_CFG = build_config("terrain.flat")
H2_PLUS_ROUGH_TERRAIN_CFG = build_config("terrain.rough")
H2_PLUS_DEFORMABLE_GROUND_CFG = build_config("terrain.deformable")
