"""Validate reward tuning without importing or launching Isaac Sim."""

import ast
import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

LOC = Path(__file__).resolve().parents[1] / "h2_plus_lab/tasks/locomotion"
SOURCE = LOC / "env_cfg.py"
TREE = ast.parse(SOURCE.read_text())


class MdpFunctions:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def term(*, func, weight, params=None):
    return SimpleNamespace(func=func, weight=weight, params={} if params is None else params)


# Load the actual pure configuration function and declarations, without Sim imports.
loader = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_rewards_yaml")
reward_class = copy.deepcopy(next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "H2PlusRewardsCfg"))
reward_class.decorator_list = []
namespace = {
    "__file__": str(SOURCE), "Path": Path, "math": math, "yaml": yaml,
    "RewTerm": term, "mdp": MdpFunctions(), "SceneEntityCfg": lambda *args, **kwargs: object(),
    "H2_PLUS_ACTIVE_JOINTS": ["active"], "H2_PLUS_PASSIVE_JOINTS": ["passive"],
}
exec(compile(ast.Module(body=[loader, reward_class], type_ignores=[]), str(SOURCE), "exec"), namespace)
apply_yaml = namespace["_apply_rewards_yaml"]


def rewards():
    return SimpleNamespace(**copy.deepcopy({name: value for name, value in vars(namespace["H2PlusRewardsCfg"]).items() if not name.startswith("_")}))


class RewardYamlTests(unittest.TestCase):
    def test_shipped_yaml_matches_existing_defaults_and_preserves_bindings(self):
        cfg = rewards()
        weights = {name: value.weight for name, value in vars(cfg).items()}
        foot_binding = cfg.feet_slide.params["asset_cfg"]
        apply_yaml(cfg)
        self.assertEqual(weights, {name: value.weight for name, value in vars(cfg).items()})
        self.assertIs(cfg.feet_slide.params["asset_cfg"], foot_binding)
        self.assertEqual(cfg.track_lin_vel_xy.params["std"], 0.5)
        self.assertEqual(cfg.track_lin_vel_xy.params["command_name"], "base_velocity")
        config = yaml.safe_load((LOC / "agents/rsl_rl_locomotion_rewards.yaml").read_text())
        self.assertEqual(set(config["rewards"]), set(weights))

    def test_partial_tuning_updates_only_selected_numbers(self):
        cfg = rewards()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "tuning.yaml"
            path.write_text("rewards:\n  action_rate:\n    weight: -0.03\n  feet_air_time:\n    params:\n      threshold: 0.6\n")
            apply_yaml(cfg, path)
        self.assertEqual(cfg.action_rate.weight, -0.03)
        self.assertEqual(cfg.feet_air_time.params["threshold"], 0.6)
        self.assertEqual(cfg.feet_air_time.weight, 0.2)
        self.assertEqual(cfg.termination_penalty.weight, -200.0)

    def test_invalid_values_or_keys_fail_before_any_changes(self):
        invalid = [
            "rewards: []", "wrong_section: {}",
            "rewards:\n  typo_reward: {weight: 1}",
            "rewards:\n  action_rate: {weight: .nan}",
            "rewards:\n  action_rate: {weight: false}",
            "rewards:\n  action_rate: {unknown: 1}",
            "rewards:\n  track_lin_vel_xy: {params: {std: 0}}",
            "rewards:\n  feet_air_time: {params: {threshold: -1}}",
            "rewards:\n  feet_slide: {params: {asset_cfg: 1}}",
            "rewards:\n  track_lin_vel_xy: {params: {command_name: 1}}",
            "rewards:\n  action_rate: {weight: -9}\n  typo_reward: {weight: 1}",
        ]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "invalid.yaml"
            for text in invalid:
                cfg = rewards()
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    apply_yaml(cfg, path)
                self.assertEqual(cfg.action_rate.weight, -0.01)


if __name__ == "__main__":
    unittest.main()
