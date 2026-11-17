"""Task registration and CLI contracts; no SimulationApp or environment creation."""

import argparse
import ast
import importlib
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

import gymnasium as gym
import cli_args
import h2_plus_lab
import view_logs


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

    def test_markers_are_massless_visuals_without_extra_links_or_colliders(self):
        from h2_plus_lab.config import config_data
        merge = config_data("robot.spawn")["merge_fixed_joints"]
        urdf = LAB_DIR.parent / "robot_asset/h2_plus/H2_Plus_with_footrix_no_dex_hands.urdf"
        robot = ET.parse(urdf).getroot()
        links = {link.get("name"): link for link in robot.findall("link")}
        fixed_parents = {
            joint.find("child").get("link"): joint.find("parent").get("link")
            for joint in robot.findall("joint") if joint.get("type") == "fixed"
        } if merge else {}

        def body(name):
            while name in fixed_parents:
                name = fixed_parents[name]
            return name

        active = [joint for joint in robot.findall("joint") if joint.get("type") == "revolute"]
        self.assertEqual(len({body(name) for name in links}), 32)
        self.assertEqual(len(active), 31)
        for joint in active:
            self.assertNotEqual(body(joint.find("parent").get("link")), body(joint.find("child").get("link")))
        self.assertEqual(len(links), 32)
        self.assertFalse(any("_footrix_cell_" in name for name in links))
        from h2_plus_lab.config import config_data
        import json
        layout = json.loads((LAB_DIR.parents[1] / config_data("sensors.footrix")["layout_path"]).read_text())
        for side in ("left", "right"):
            foot = links[f"{side}_ankle_pitch_link"]
            markers = [v for v in foot.findall("visual") if "_footrix_cell_" in v.get("name", "")]
            self.assertEqual(len(markers), 154)
            self.assertEqual(len(foot.findall("collision")), 1)
            for i, visual in enumerate(markers):
                self.assertIsNone(visual.find("inertial"))
                self.assertIsNone(visual.find("collision"))
                xyz = list(map(float, visual.find("origin").get("xyz").split()))
                expected = layout["feet"][side]["observation_layout"]["sample_positions_in_foot_link_m"][i]
                for actual, target in zip(xyz, expected):
                    self.assertAlmostEqual(actual, target, places=7)
        for side in ("left", "right"):
            self.assertIsNotNone(links[f"{side}_ankle_pitch_link"].find("collision"))

    def test_tensorboard_selects_latest_event_writer_not_newest_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "experiment/old_run"
            recent = root / "experiment/new_run"
            empty = root / "experiment/empty_run"
            for directory in (old, recent, empty):
                directory.mkdir(parents=True)
            first = old / "events.out.tfevents.old"
            second = recent / "events.out.tfevents.new"
            first.write_bytes(b"event data")
            second.write_bytes(b"event data")
            (empty / "events.out.tfevents.empty").touch()
            os.utime(first, ns=(10, 10))
            os.utime(second, ns=(20, 20))
            self.assertEqual(view_logs.select_log_dir(root), recent)
            os.utime(first, ns=(30, 30))  # continued training in an older run
            self.assertEqual(view_logs.select_log_dir(root), old)
            nested = old / "summaries"
            nested.mkdir()
            event = nested / "events.out.tfevents.child"
            event.write_bytes(b"event data")
            os.utime(event, ns=(40, 40))
            self.assertEqual(view_logs.select_log_dir(root), old)

    def test_tensorboard_selection_handles_explicit_all_and_missing_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(view_logs.select_log_dir(root, all_runs=True), root)
            self.assertEqual(view_logs.select_log_dir(root, run=root), root)
            with self.assertRaises(FileNotFoundError):
                view_logs.select_log_dir(root)
            with self.assertRaises(FileNotFoundError):
                view_logs.select_log_dir(root, run=root / "missing")

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
