"""View this project's training curves without starting Isaac Sim or training."""

from __future__ import annotations

import argparse
from pathlib import Path


LOG_ROOT = Path(__file__).resolve().parent / "log" / "rsl_rl"


def select_log_dir(root: Path = LOG_ROOT, run: Path | None = None, all_runs: bool = False) -> Path:
    if run is not None:
        directory = run.expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Training run directory does not exist: {directory}")
        return directory
    root = root.resolve()
    if all_runs:
        return root
    latest = None
    for event in root.rglob("events.out.tfevents.*"):
        try:
            info = event.stat()
        except FileNotFoundError:
            continue
        if not event.is_file() or info.st_size == 0:
            continue
        relative = event.relative_to(root)
        # Layout: experiment/run[/optional-summary-subdir]/event file.
        directory = root.joinpath(*relative.parts[:2]) if len(relative.parts) >= 3 else event.parent
        candidate = (info.st_mtime_ns, str(directory))
        if latest is None or candidate > latest:
            latest = candidate
    if latest is None:
        raise FileNotFoundError("No training event data yet. Start PPO first, or use --all to open the log root.")
    return Path(latest[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="View only the most recently updated training run by default.",
        epilog="Other options are forwarded to TensorBoard, for example --port 6007.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--latest", action="store_true", help="View the most recently updated run (default).")
    selection.add_argument("--all", action="store_true", help="View all training runs.")
    selection.add_argument("--run", type=Path, help="View one specified run directory.")
    args, remaining = parser.parse_known_args(argv)
    try:
        log_dir = LOG_ROOT if "--version" in remaining else select_log_dir(LOG_ROOT, args.run, args.all)
    except FileNotFoundError as error:
        parser.error(str(error))
    from tensorboard import program

    print(f"TensorBoard log directory: {log_dir}", flush=True)
    tensorboard = program.TensorBoard()
    tensorboard.configure(argv=[
        str(Path(__file__).resolve()),
        "--logdir", str(log_dir),
        "--host", "127.0.0.1",
        "--port", "6006",
        *remaining,
    ])
    return tensorboard.main()


if __name__ == "__main__":
    raise SystemExit(main())
