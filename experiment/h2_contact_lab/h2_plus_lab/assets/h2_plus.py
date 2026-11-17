"""H2 Plus asset configuration loaded from h2_plus.yaml."""

from pathlib import Path

from ..config import build_config, config_data

H2_PLUS_CFG = build_config("robot")
H2_PLUS_URDF = Path(config_data("robot.spawn")["asset_path"])
H2_PLUS_USD_DIR = Path(config_data("robot.spawn")["usd_dir"])

__all__ = ["H2_PLUS_CFG", "H2_PLUS_URDF", "H2_PLUS_USD_DIR"]
