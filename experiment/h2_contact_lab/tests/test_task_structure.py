"""Task registration and CLI contracts; no SimulationApp or environment creation."""

import argparse
import ast
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

import gymnasium as gym
import cli_args
import h2_plus_lab


class TaskStructureTests(unittest.TestCase):
    def test_registered_entry_points_resolve_to_declared_classes(self):
        ids = {
            "Isaac-H2Plus-Blind-Locomotion-v0",
            "Isaac-H2Plus-Blind-Locomotion-Play-v0",
            "Isaac-H2Plus-Blind-Locomotion-LowFriction-v0",
            "Isaac-H2Plus-Blind-Locomotion-LowFriction-Play-v0",
            "Isaac-H2Plus-Blind-Locomotion-Deformable-v0",
            "Isaac-H2Plus-Blind-Locomotion-Deformable-Play-v0",
        }
        for task_id in ids:
            spec = gym.spec(task_id)
            self.assertEqual(spec.entry_point, "isaaclab.envs:ManagerBasedRLEnv")
            for key in ("env_cfg_entry_point", "rsl_rl_cfg_entry_point"):
                module, class_name = spec.kwargs[key].split(":")
                path = LAB_DIR.joinpath(*module.split(".")).with_suffix(".py")
                tree = ast.parse(path.read_text())
                self.assertIn(class_name, {node.name for node in tree.body if isinstance(node, ast.ClassDef)})
        self.assertNotIn("isaaclab.sim", sys.modules)
        self.assertNotIn("omni", sys.modules)

    def test_new_task_package_is_discovered_without_changing_launcher(self):
        task_id = "H2-Test-Discovery-v0"
        module_name = "h2_plus_lab.tasks.discovery_example"
        with tempfile.TemporaryDirectory() as root:
            package = Path(root) / "discovery_example"
            package.mkdir()
            (package / "__init__.py").write_text(
                f'import gymnasium as gym\ngym.register(id="{task_id}", entry_point="example:Env")\n'
            )
            h2_plus_lab.tasks.__path__.append(root)
            try:
                importlib.invalidate_caches()
                h2_plus_lab.tasks.register_task_packages()
                self.assertEqual(gym.spec(task_id).entry_point, "example:Env")
            finally:
                h2_plus_lab.tasks.__path__.remove(root)
                gym.registry.pop(task_id, None)
                sys.modules.pop(module_name, None)
                vars(h2_plus_lab.tasks).pop("discovery_example", None)

    def test_hydra_paths_do_not_depend_on_working_directory(self):
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as root:
            try:
                os.chdir(root)
                args = []
                cli_args.configure_hydra_output(args)
                self.assertTrue(args[0].startswith(f"hydra.run.dir={LAB_DIR}/output/hydra/"))
                cli_args.configure_hydra_output(args)
                self.assertEqual(len(args), 1)
                explicit = ["hydra.run.dir=/tmp/explicit-h2-output"]
                cli_args.configure_hydra_output(explicit)
                self.assertEqual(len(explicit), 1)
            finally:
                os.chdir(old_cwd)

    def test_recording_paths_are_unique_and_disabled_recorders_are_supported(self):
        env_cfg = SimpleNamespace(recorders=SimpleNamespace())
        cli_args.configure_recording_output(env_cfg, "/tmp/run", "train", "task-a", enabled=True)
        first = Path(env_cfg.recorders.dataset_export_dir_path)
        self.assertTrue(first.is_relative_to(LAB_DIR / "output/sensors/train"))
        self.assertEqual(env_cfg.env_name, "task-a")
        cli_args.configure_recording_output(env_cfg, "/tmp/run", "train", "task-a", enabled=True)
        self.assertNotEqual(str(first), env_cfg.recorders.dataset_export_dir_path)
        cli_args.configure_recording_output(SimpleNamespace(recorders=None), "/tmp/run", "train", "task-a")

    def test_sensor_recording_switch_defaults_to_off(self):
        parser = argparse.ArgumentParser()
        cli_args.add_rsl_rl_args(parser)
        self.assertFalse(parser.parse_args([]).record_sensors)
        self.assertTrue(parser.parse_args(["--record_sensors"]).record_sensors)
        cfg = SimpleNamespace(recorders=SimpleNamespace())
        cli_args.configure_recording_output(cfg, "/tmp/run", "train", "task-a")
        self.assertIsNone(cfg.recorders)
        with self.assertRaises(ValueError):
            cli_args.configure_recording_output(cfg, "/tmp/run", "train", "task-a", enabled=True)

    def test_checkpoint_file_path_enables_resume_without_run_directory(self):
        parser = argparse.ArgumentParser()
        cli_args.add_rsl_rl_args(parser)
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "a run" / "model_100.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"path test only; not model data")
            for option in ("--checkpoint", "--ckpt"):
                with self.subTest(option=option):
                    args = parser.parse_args([option, str(checkpoint)])
                    cfg = SimpleNamespace(resume=False, logger="tensorboard")
                    cli_args.update_rsl_rl_cfg(cfg, args)
                    self.assertTrue(cfg.resume)
                    self.assertIsNone(args.load_run)
                    self.assertEqual(cfg.load_checkpoint, str(checkpoint.resolve()))
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                relative = parser.parse_args(["--checkpoint", "a run/model_100.pt"])
                self.assertEqual(relative.checkpoint, str(checkpoint.resolve()))
            finally:
                os.chdir(old_cwd)
            with self.assertRaises(argparse.ArgumentTypeError):
                cli_args.checkpoint_path(str(checkpoint.parent))
            with self.assertRaises(argparse.ArgumentTypeError):
                cli_args.checkpoint_path(str(checkpoint.parent / "missing.pt"))

    def test_cli_overrides_keep_configuration_defaults_when_unspecified(self):
        parser = argparse.ArgumentParser()
        cli_args.add_rsl_rl_args(parser)
        cfg = SimpleNamespace(resume=True, logger="tensorboard", experiment_name="original")
        cli_args.update_rsl_rl_cfg(cfg, parser.parse_args([]))
        self.assertTrue(cfg.resume)
        self.assertEqual(cfg.experiment_name, "original")
        cli_args.update_rsl_rl_cfg(cfg, parser.parse_args(["--experiment_name", "custom", "--resume"]))
        self.assertEqual(cfg.experiment_name, "custom")
        self.assertTrue(cfg.resume)


if __name__ == "__main__":
    unittest.main()
