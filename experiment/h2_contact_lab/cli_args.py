# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from time import time_ns
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def checkpoint_path(value: str) -> str:
    """Resolve a local checkpoint file before launching Isaac Sim."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Checkpoint file does not exist: {path}")
    return str(path)


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    arg_group.add_argument(
        "--record_sensors", action="store_true", default=False,
        help="Save task sensor data through RecorderManager (disabled by default).",
    )
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument(
        "--checkpoint", "--ckpt", dest="checkpoint", type=checkpoint_path, default=None,
        help="Path to a local checkpoint file; automatically enables training resume. Relative paths are also accepted.",
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
        agent_cfg.resume = True
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg


def configure_hydra_output(hydra_args: list[str]) -> None:
    """Set the project-local snapshot directory without changing the caller's cwd."""
    if not any(arg.lstrip("+").startswith("hydra.run.dir=") for arg in hydra_args):
        lab_dir = Path(__file__).resolve().parent
        hydra_args.append(f"hydra.run.dir={lab_dir}/output/hydra/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}")


def configure_recording_output(env_cfg, run_dir: str, mode: str, task_name: str, enabled: bool = False) -> None:
    """Enable recording explicitly; otherwise remove recorder terms before env creation."""
    if not enabled:
        env_cfg.recorders = None
        return
    recorders = getattr(env_cfg, "recorders", None)
    if recorders is None:
        raise ValueError("--record_sensors requires a task with a recorder configuration")
    lab_dir = Path(__file__).resolve().parent
    recorders.dataset_export_dir_path = str(
        lab_dir / "output" / "sensors" / mode / f"{Path(run_dir).name}_{time_ns()}"
    )
    env_cfg.env_name = task_name
