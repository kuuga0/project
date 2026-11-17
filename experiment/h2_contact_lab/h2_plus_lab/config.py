"""Compose YAML parameters and construct native Isaac Lab configuration objects."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin

from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

PACKAGE_DIR = Path(__file__).resolve().parent
LAB_DIR = PACKAGE_DIR.parent
LOC_DIR = PACKAGE_DIR / "tasks" / "locomotion"


@dataclass
class ManagerTermsCfg:
    """Attribute-based term container expected by Isaac Lab's managers.

    Term instances are native *TermCfg objects. The containing group starts
    empty; managers add runtime attributes such as debug_vis after parsing.
    """


def _tuple_spec(annotation):
    """Read tuple element types, including postponed/optional annotations."""
    if isinstance(annotation, str):
        def parse(node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                sides = [n for n in (node.left, node.right) if not (isinstance(n, ast.Constant) and n.value is None)]
                return parse(sides[0]) if len(sides) == 1 else None
            if not isinstance(node, ast.Subscript):
                return None
            name = getattr(node.value, "id", getattr(node.value, "attr", ""))
            if name == "Optional":
                return parse(node.slice)
            if name not in ("tuple", "Tuple"):
                return None
            items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            return tuple(Ellipsis if isinstance(n, ast.Constant) and n.value is Ellipsis
                         else {"float": float, "int": int}.get(getattr(n, "id", "")) for n in items)
        try:
            return parse(ast.parse(annotation, mode="eval").body)
        except SyntaxError:
            return None
    if annotation is tuple or get_origin(annotation) is tuple:
        return get_args(annotation)
    if get_origin(annotation) in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _tuple_spec(args[0]) if len(args) == 1 else None
    return None


def restore_config_types(value):
    """Restore native tuple fields after Hydra; keep list-valued fields as lists.

    USD treats an untyped list as VtValue[], not necessarily as a Gf vector.
    Use the native config declaration rather than guessing from vector lengths.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            value[key] = restore_config_types(child)
    elif isinstance(value, list):
        value[:] = [restore_config_types(child) for child in value]
    elif isinstance(value, tuple):
        return tuple(restore_config_types(child) for child in value)
    elif is_dataclass(value) and not isinstance(value, type):
        native_fields = {field.name: field for field in fields(value)}
        for name, child in vars(value).copy().items():
            spec = _tuple_spec(native_fields[name].type) if name in native_fields else None
            if spec is not None and isinstance(child, (list, tuple)):
                repeated = len(spec) == 2 and spec[1] is Ellipsis
                if spec and not repeated and len(child) != len(spec):
                    raise ValueError(f"{type(value).__name__}.{name} expects {len(spec)} values, got {len(child)}")
                items = []
                for index, item in enumerate(child):
                    expected = spec[0] if repeated else (spec[index] if spec else None)
                    if expected in (float, int):
                        if type(item) not in (float, int) or (expected is int and type(item) is not int):
                            raise TypeError(f"{type(value).__name__}.{name} contains an invalid numeric value")
                        item = expected(item)
                    items.append(item)
                child = tuple(items)
            setattr(value, name, restore_config_types(child))
    return value


def instantiate_config(values):
    return restore_config_types(instantiate(values, _convert_="all"))


def config_group(class_path: str, **values):
    """Add scene entities/manager terms to a native extensible config group."""
    cls = get_class(class_path)
    constructor_fields = {field.name for field in fields(cls) if field.init}
    cfg = cls(**{key: value for key, value in values.items() if key in constructor_fields})
    for key, value in values.items():
        if key not in constructor_fields:
            setattr(cfg, key, value)
    return cfg


def load_settings():
    """Load the module YAML files; paths derive from this file, never the cwd."""
    files = (LOC_DIR / "env_cfg.yaml", PACKAGE_DIR / "assets/h2_plus.yaml",
             LOC_DIR / "sensor_cfg.yaml", LOC_DIR / "terrain_cfg.yaml")
    settings = OmegaConf.merge(*(OmegaConf.load(path) for path in files))
    settings.paths = {"project": str(LAB_DIR.parents[1]), "lab": str(LAB_DIR)}
    reward_file = OmegaConf.load(LOC_DIR / "agents/rsl_rl_locomotion_rewards.yaml")
    if set(reward_file) != {"rewards"}:
        raise ValueError("Reward YAML accepts only 'rewards'; edit events/noise in env_cfg.yaml")
    declared = set(settings.environment.rewards) - {"_target_", "class_path"}
    unknown = set(reward_file.rewards) - declared
    if unknown:
        raise ValueError(f"Unknown reward terms: {sorted(unknown)}")
    settings.environment.rewards = OmegaConf.merge(settings.environment.rewards, reward_file.rewards)
    return settings


def _validate(settings, values) -> None:
    def finite(value):
        if isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, list):
            for child in value: finite(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Environment YAML contains a non-finite number")
    finite(values)
    for name, term in values.get("rewards", {}).items():
        if name in {"_target_", "class_path"}:
            continue
        if term is not None and (not isinstance(term, dict) or type(term.get("weight")) not in (float, int)):
            raise ValueError(f"rewards.{name}.weight must be a finite number")
    cfg = OmegaConf.create(values)
    for constraint, paths in settings.checks.items():
        for path in paths:
            value = OmegaConf.select(cfg, path)
            if type(value) not in (float, int) or value < 0 or (constraint == "positive" and value == 0):
                raise ValueError(f"{path} must be {constraint}")


def environment_data(profile: str = "flat", play: bool = False) -> dict:
    """Compose base → terrain → play; Hydra command-line overrides run later."""
    settings = load_settings()
    if profile not in settings.terrains:
        raise ValueError(f"Unknown terrain profile: {profile}")
    settings.active = OmegaConf.merge(settings.environment, settings.terrains[profile], settings.play if play else {})
    values = OmegaConf.to_container(settings.active, resolve=True)
    _validate(settings, values)
    return values


def build_config(section: str):
    """Construct a reusable asset/sensor/terrain from its native YAML spec."""
    settings = load_settings()
    values = OmegaConf.to_container(OmegaConf.select(settings, section), resolve=True)
    return instantiate_config(values)


def config_data(section: str):
    """Read metadata/constants without importing the simulation runtime."""
    settings = load_settings()
    value = OmegaConf.select(settings, section)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
