"""Project-local launcher for the Isaac Lab RSL-RL train/play scripts.

The wrapper imports the project package before handing control to Isaac Lab so
that the H2 Plus Gym task registrations exist in the same Python process.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


TASK_IDS = {
    "train": "Isaac-H2Plus-Blind-Locomotion-v0",
    "play": "Isaac-H2Plus-Blind-Locomotion-Play-v0",
}


def _parse_launcher_args(argv: list[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=tuple(TASK_IDS), default="train")
    args, remaining = parser.parse_known_args(argv)
    return args.mode, remaining


def main(argv: list[str] | None = None) -> int:
    mode, remaining = _parse_launcher_args(argv)
    import h2_plus_lab  # noqa: F401  # registers the Gym tasks in this process

    isaaclab_root = Path(os.environ.get("ISAACLAB_ROOT", "/wyp/IsaacLab")).expanduser()
    script = isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / f"{mode}.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"Isaac Lab {mode} script was not found at {script}. "
            "Set ISAACLAB_ROOT to the local IsaacLab checkout."
        )

    if "--task" not in remaining:
        remaining.extend(["--task", TASK_IDS[mode]])
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
