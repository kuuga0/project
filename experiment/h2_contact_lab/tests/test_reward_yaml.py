"""YAML composition and native API regressions; no SimulationApp or training."""

import ast
import math
import logging
from abc import ABC
import copy
import importlib.util
import runpy
import re
import xml.etree.ElementTree as ET
import sys
import unittest
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from typing import ClassVar

import torch
from hydra.utils import instantiate

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from h2_plus_lab import config


@dataclass
class ExampleGroup:
    enabled: bool = True


@dataclass
class ExampleTerm:
    weight: float = 1.0


@dataclass
class ExampleVectors:
    color: tuple[float, float, float] = (1., 1., 1.)
    position: tuple[float, float, float] | None = None
    rotation: "tuple[float, float, float, float]" = (1., 0., 0., 0.)
    names: list[str] = field(default_factory=list)
    tags: list[tuple[str, str]] = field(default_factory=list)


class ManagerTestTerm:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.action_dim = 2


@dataclass
class ManagerTestTermCfg:
    class_type: type = ManagerTestTerm
    debug_vis: bool = True


class ManagerTestBase:
    """Replace simulator plumbing, retaining native manager constructors/parsers."""
    def __init__(self, cfg, env):
        self.cfg = copy.deepcopy(cfg)
        self._env = env
        if self.cfg:
            self._prepare_terms()

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device


class RewardYamlTests(unittest.TestCase):
    def test_knee_inactivity_uses_motion_window_and_native_reset(self):
        isaac = Path(importlib.util.find_spec("isaaclab").origin).parent
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        base = next(n for n in ast.parse((isaac / "managers/manager_base.py").read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == "ManagerTermBase")
        term_class = next(n for n in ast.parse((LAB / "h2_plus_lab/tasks/locomotion/mdp/rewards.py").read_text()).body
                          if isinstance(n, ast.ClassDef) and n.name == "H2PlusJointInactivity")
        scope = {"torch": torch, "math": math, "ABC": ABC}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, base, term_class], type_ignores=[])), "<native term base and knee history>", "exec"), scope)
        for term_name in ("knee_inactivity", "elbow_inactivity"):
            positions = torch.zeros(4, 2)
            commands = torch.tensor([[.5, 0., 0.], [.5, 0., 0.], [0., 0., 0.], [0., 0., .3]])
            env = NS(num_envs=4, device="cpu", step_dt=1/30, scene={"robot": NS(data=NS(joint_pos=positions))},
                     command_manager=NS(get_command=lambda name: commands))
            params = config.environment_data()["rewards"][term_name]["params"].copy()
            params["asset_cfg"] = NS(name="robot", joint_ids=[0, 1])
            term = scope["H2PlusJointInactivity"](NS(params=params), env)
            for index in range(term._history.shape[0]):
                positions[0] = .8  # held flexed posture is still inactive
                positions[1] = (.3 if index % 2 else 0.)
                positions[3, 0] = (.3 if index % 2 else 0.)  # one active knee during turning
                result = term(env, **params)
                if index < term._history.shape[0]-1:
                    self.assertTrue((result == 0).all())
            torch.testing.assert_close(result, torch.tensor([1., 0., 0., .5]))
            # Exercise the actual native RewardManager reset hook for per-env histories.
            manager = next(n for n in ast.parse((isaac / "managers/reward_manager.py").read_text()).body
                           if isinstance(n, ast.ClassDef) and n.name == "RewardManager")
            reset = next(n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == "reset")
            exec(compile(ast.fix_missing_locations(ast.Module(body=[future, reset], type_ignores=[])), "<native reward reset>", "exec"), scope)
            scope["reset"](NS(_episode_sums={}, _class_term_cfgs=[NS(func=term)]), env_ids=[0, 3])
            result = term(env, **params)
            self.assertEqual(result[0].item(), 0.)
            self.assertEqual(result[3].item(), 0.)
            commands[1] = 0.
            term(env, **params)
            commands[1, 0] = .5
            self.assertEqual(term(env, **params)[1].item(), 0.)  # fresh grace period after standing

    def test_native_joint_action_accepts_larger_knee_scale_without_reordering(self):
        isaac = Path(importlib.util.find_spec("isaaclab").origin).parent
        strings = runpy.run_path(str(isaac / "utils/string.py"))
        tree = ast.parse((isaac / "envs/mdp/actions/joint_actions.py").read_text())
        action_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "JointAction")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        class ActionBase:
            def __init__(self, cfg, env):
                self.cfg, self._asset = cfg, env.asset
                self.num_envs, self.device = env.num_envs, env.device
        names = [f"{side}_{part}_joint" for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_roll", "ankle_pitch")
                 for side in ("left", "right")]
        asset = NS(num_joints=31, find_joints=lambda patterns, preserve_order: (list(range(12)), names))
        scope = {"torch": torch, "ActionTerm": ActionBase, "logger": logging.getLogger(__name__),
                 "string_utils": NS(resolve_matching_names_values=strings["resolve_matching_names_values"])}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, action_class], type_ignores=[])), "<native joint scale>", "exec"), scope)
        values = config.environment_data()["actions"]["legs"]
        cfg = NS(joint_names=values["joint_names"], scale=values["scale"], preserve_order=False, offset=0., clip=values.get("clip"))
        action = scope["JointAction"](cfg, NS(asset=asset, num_envs=1, device="cpu"))
        action.process_actions(torch.ones((1, 12)))
        expected = torch.tensor([[.5 if "knee" in name else .25 for name in names]])
        torch.testing.assert_close(action.processed_actions, expected)
        self.assertEqual(action._joint_names, names)
        extreme = torch.full((1, 12), 2.)
        extreme[0, 6:8] = torch.tensor([-20., 20.])
        action.process_actions(extreme)
        robot_xml = ET.parse(LAB.parent / "robot_asset/h2_plus/H2_Plus_with_footrix_no_dex_hands.urdf").getroot()
        limit = robot_xml.find("joint[@name='left_knee_joint']/limit")
        bounds = [float(limit.get("lower")), float(limit.get("upper"))]
        torch.testing.assert_close(action.processed_actions[0, 6:8], torch.tensor(bounds))
        torch.testing.assert_close(action.raw_actions, extreme)  # clipping keeps pre-clip samples available
        self.assertEqual(values["clip"][".*_knee_joint"], bounds)
        for index, name in ((8, "left_ankle_roll_joint"), (9, "right_ankle_roll_joint")):
            limit = robot_xml.find(f"joint[@name='{name}']/limit")
            bounds = [float(limit.get("lower")), float(limit.get("upper"))]
            self.assertEqual(values["clip"][name], bounds)
            self.assertAlmostEqual(action.processed_actions[0, index].item(), bounds[1], places=6)

    def test_target_limit_penalty_reads_unclipped_targets_with_correct_joint_mapping(self):
        isaac = Path(importlib.util.find_spec("isaaclab").origin).parent
        base = next(n for n in ast.parse((isaac / "managers/manager_base.py").read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == "ManagerTermBase")
        cls = next(n for n in ast.parse((LAB / "h2_plus_lab/tasks/locomotion/mdp/rewards.py").read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "H2PlusJointTargetLimitPenalty")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        scope = {"ABC": ABC, "torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, base, cls], type_ignores=[])), "<native base and target limit term>", "exec"), scope)
        action = NS(cfg=NS(joint_names=[".*"], preserve_order=False), raw_actions=torch.tensor([[-4., 4., 100.], [1., 1., 100.], [-10., 10., 100.]]),
                    _scale=torch.tensor([[.5, .5, .25]]), _offset=torch.tensor([[.1, .2, 0.]]), processed_actions=torch.zeros(3, 3))
        asset = NS(joint_names=["right_knee_joint", "hip_joint", "left_knee_joint"],
                   find_joints=lambda names, preserve_order: ([2, 0, 1], ["left_knee_joint", "right_knee_joint", "hip_joint"]),
                   data=NS(soft_joint_pos_limits=torch.tensor([[[-.08725, 2.53025]]*3]*3)))
        env = NS(scene={"robot": asset}, action_manager=NS(get_term=lambda name: action))
        params = config.environment_data()["rewards"]["knee_target_limits"]["params"].copy()
        params["asset_cfg"] = NS(name="robot", joint_ids=[2, 0])
        term = scope["H2PlusJointTargetLimitPenalty"](NS(params=params), env)
        result = term(env, **params)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(result[0].item(), 0.)
        self.assertEqual(result[1].item(), 0.)  # hip's huge target is not in this configured penalty
        self.assertGreater(result[2].item(), result[0].item())
        action.raw_actions[1, :2] = 0.  # valid physical targets, but inside the optional end margin
        self.assertEqual(term(env, **params)[1].item(), 0.)
        self.assertGreater(term(env, **params, margin_ratio=.15)[1].item(), 0.)
        with self.assertRaises(ValueError):
            term(env, **params, margin_ratio=.5)

    def test_sensor_config_can_read_scalar_and_container_metadata(self):
        self.assertAlmostEqual(config.config_data("sensors.torso_imu.update_period"), 1/120)
        self.assertEqual(config.config_data("sensors.torso_imu.history_length"), 6)
        self.assertIsInstance(config.config_data("sensors.footrix"), dict)

    def test_lateral_body_translation_is_distinct_from_roll_and_allows_balance_and_turning(self):
        pos = torch.zeros(5, 4, 3)
        pos[:, 2, 1], pos[:, 3, 1] = .1, -.1
        pos[0, :2, 1] = .02
        pos[1:3, :2, 1] = .105
        pos[3, :2, 0] = 100.
        velocity = torch.zeros_like(pos)
        velocity[0, :2, 1] = .04
        velocity[1:3, :2, 1] = .35
        velocity[3, :2, 0] = 3.
        velocity[4, :2, 1] = .35
        pos += torch.tensor([1., 2., 0.])  # common translation does not change the foot-midpoint metric
        command = torch.zeros(5, 3); command[:, 0] = .5
        command[2, 2] = .5; command[4, 1] = .35
        env = NS(scene={"robot": NS(data=NS(body_pos_w=pos, body_link_lin_vel_w=velocity,
                 root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(5, 1)))},
                 command_manager=NS(get_command=lambda name: command))
        rewards = config.environment_data()["rewards"]
        for term, function in (("body_lateral_position", "h2_plus_body_lateral_position_l2"),
                               ("body_lateral_velocity", "h2_plus_body_lateral_velocity_l2")):
            params = rewards[term]["params"].copy()
            params["asset_cfg"] = NS(name="robot", body_ids=[0, 1])
            if "feet_cfg" in params:
                params["feet_cfg"] = NS(name="robot", body_ids=[2, 3])
            result = self.posture_functions()[function](env, **params)
            torch.testing.assert_close(result, torch.tensor([0., 1., 0., 0., 0.]), atol=1e-5, rtol=1e-5)

    def test_arm_coordination_rewards_alternating_fore_aft_motion_not_frozen_arms(self):
        params = config.environment_data()["rewards"]["arm_leg_coordination"]["params"].copy()
        phase = math.tanh(1.)
        positions = torch.zeros(5, 2, 3)
        positions[:, 0, 0], positions[:, 1, 0] = params["phase_length"], -params["phase_length"]
        positions[3] = 0.; positions[4] *= -1
        amplitudes = torch.tensor([params["shoulder_amplitude"], -params["shoulder_amplitude"],
                                   -params["elbow_amplitude"], params["elbow_amplitude"]])
        joints = (phase*amplitudes)[None, :].repeat(5, 1)
        joints[1] = 0.; joints[3] = 0.; joints[4] *= -1
        commands = torch.tensor([[.5, 0., 0.]]).repeat(5, 1); commands[2] = 0.
        env = NS(scene={"robot": NS(data=NS(body_pos_w=positions, joint_pos=joints,
                 root_pos_w=torch.zeros(5, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(5, 1)))},
                 command_manager=NS(get_command=lambda name: commands))
        params["asset_cfg"] = NS(name="robot", joint_ids=[0, 1, 2, 3])
        params["feet_cfg"] = NS(name="robot", body_ids=[0, 1])
        result = self.posture_functions()["h2_plus_arm_leg_coordination_exp"](env, **params)
        self.assertGreater(result[0].item(), result[1].item())
        torch.testing.assert_close(result[[0, 4]], torch.tensor([phase, phase]))
        torch.testing.assert_close(result[[2, 3]], torch.zeros(2))
        arm_patterns = config.environment_data()["rewards"]["arm_posture"]["params"]["asset_cfg"]["joint_names"]
        self.assertFalse(any(re.fullmatch(p, "left_shoulder_pitch_joint") for p in arm_patterns))

    def test_shoulder_roll_penalties_allow_small_motion_and_exclude_other_arm_joints(self):
        rewards = config.environment_data()["rewards"]
        params = rewards["shoulder_roll_posture"]["params"].copy()
        tolerance, std = params["tolerance"], params["std"]
        positions = torch.tensor([[0., 0.], [.1, -.1], [tolerance, -tolerance],
                                  [tolerance+std, -tolerance-std]])
        env = NS(scene={"robot": NS(data=NS(joint_pos=positions))})
        params["asset_cfg"] = NS(name="robot", joint_ids=[0, 1])
        result = self.posture_functions()["h2_plus_joint_reference_deadband_l2"](env, **params)
        torch.testing.assert_close(result, torch.tensor([0., 0., 0., 1.]), atol=1e-6, rtol=1e-5)
        for term in ("shoulder_roll_posture", "shoulder_roll_velocity", "shoulder_roll_target_limits"):
            patterns = rewards[term]["params"]["asset_cfg"]["joint_names"]
            for side in ("left", "right"):
                self.assertTrue(any(re.fullmatch(p, side+"_shoulder_roll_joint") for p in patterns))
                for part in ("shoulder_pitch", "shoulder_yaw", "elbow"):
                    self.assertFalse(any(re.fullmatch(p, side+"_"+part+"_joint") for p in patterns))
        clips = config.environment_data()["actions"]["shoulders_elbows"]["clip"]
        robot = ET.parse(LAB.parent / "robot_asset/h2_plus/H2_Plus_with_footrix_no_dex_hands.urdf").getroot()
        for pattern, bounds in clips.items():
            matches = [j for j in robot.findall("joint") if re.fullmatch(pattern, j.get("name"))]
            self.assertTrue(matches)
            for joint in matches:
                limit = joint.find("limit")
                self.assertEqual(bounds, [float(limit.get("lower")), float(limit.get("upper"))])

    def test_ankle_roll_normal_range_is_free_and_cost_rises_at_limits(self):
        params = config.environment_data()["rewards"]["ankle_roll_limits"]["params"].copy()
        limits = torch.tensor([[[-.34906585, .29670597], [-.29670597, .34906585]]]).repeat(6, 1, 1)
        margin = (limits[:, :, 1]-limits[:, :, 0])*params["margin_ratio"]
        positions = torch.zeros(6, 2)
        positions[1] = limits[1, :, 0]+margin[1]
        positions[2] = limits[2, :, 1]-margin[2]
        positions[3] = limits[3, :, 0]+margin[3]/2
        positions[4] = limits[4, :, 0]
        positions[5] = limits[5, :, 1]
        env = NS(scene={"robot": NS(data=NS(joint_pos=positions, soft_joint_pos_limits=limits))})
        params["asset_cfg"] = NS(name="robot", joint_ids=[0, 1])
        function = self.posture_functions()["h2_plus_joint_limit_margin_l2"]
        actual = function(env, **params)
        torch.testing.assert_close(actual, torch.tensor([0., 0., 0., .25, 1., 1.]), atol=1e-6, rtol=1e-5)
        for ratio in (0., -.1, .5, 1.):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                function(env, asset_cfg=params["asset_cfg"], margin_ratio=ratio)

    def test_stance_width_catches_symmetric_wide_and_crossed_feet(self):
        widths = torch.tensor([.15724, .25, .4, 0., -.2])
        positions = torch.zeros(5, 2, 3)
        positions[:, 0, 1] = widths/2
        positions[:, 1, 1] = -widths/2
        data = NS(body_pos_w=positions, root_pos_w=torch.zeros(5, 3),
                  root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(5, 1))
        env = NS(scene={"robot": NS(data=data)})
        params = config.environment_data()["rewards"]["stance_width"]["params"].copy()
        params["asset_cfg"] = NS(name="robot", body_ids=[0, 1])
        result = self.posture_functions()["h2_plus_stance_width_l2"](env, **params)
        torch.testing.assert_close(result, torch.tensor([0., 0., 2.25, 1., 9.]), atol=1e-6, rtol=1e-5)
        # The old centering metric is zero even for the wide case; width is separately needed.
        self.assertEqual(positions[2, :, 1].sum().item(), 0.)

    def test_slide_penalty_ignores_airborne_feet_with_stale_force_history(self):
        class Scene(dict):
            pass
        velocity = torch.tensor([[[.4, .3, 0.], [1.2, 0., 0.]]]).repeat(3, 1, 1)
        scene = Scene(robot=NS(data=NS(body_lin_vel_w=velocity)))
        scene.sensors = {}
        for side, active_row in (("left", 0), ("right", 2)):
            force = torch.zeros(3, 1, 3); force[active_row, 0, 2] = 10.
            duration = torch.zeros(3, 1); duration[active_row, 0] = .1
            scene.sensors[side+"_foot_contact"] = NS(data=NS(net_forces_w=force, current_contact_time=duration,
                net_forces_w_history=torch.ones(3, 3, 1, 3)*10))
        result = self.posture_functions()["h2_plus_feet_slide"](NS(scene=scene), NS(name="robot", body_ids=[0, 1]), 1.)
        torch.testing.assert_close(result, torch.tensor([.5, 0., 1.2]))

    def posture_functions(self):
        tree = ast.parse((LAB / "h2_plus_lab/tasks/locomotion/mdp/rewards.py").read_text())
        names = {"h2_plus_body_lateral_position_l2", "h2_plus_body_lateral_velocity_l2", "h2_plus_arm_leg_coordination_exp", "h2_plus_joint_reference_l1", "h2_plus_joint_reference_deadband_l2", "h2_plus_torso_orientation_l2", "h2_plus_torso_roll_velocity_l2", "h2_plus_swing_clearance_exp", "h2_plus_feet_lateral_symmetry_l2", "h2_plus_contact_foot_flat_reward", "h2_plus_joint_limit_margin_l2", "h2_plus_stance_width_l2", "h2_plus_feet_slide", "h2_plus_foot_heading_l2"}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        def inverse(q, v):
            xyz = q[:, 1:]
            return v - 2*q[:, :1]*torch.cross(xyz, v, dim=-1) + 2*torch.cross(xyz, torch.cross(xyz, v, dim=-1), dim=-1)
        isaac = Path(importlib.util.find_spec("isaaclab").origin).parent
        native_yaw = next(n for n in ast.parse((isaac / "utils/math.py").read_text()).body
                          if isinstance(n, ast.FunctionDef) and n.name == "yaw_quat")
        native_yaw.decorator_list = []
        scope = {"torch": torch, "quat_apply_inverse": inverse,
                 "normalize": lambda q: torch.nn.functional.normalize(q, dim=-1)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, native_yaw]+functions, type_ignores=[])), "<posture rewards>", "exec"), scope)
        return scope

    def test_torso_lateral_tilt_is_stronger_without_penalizing_pitch_or_yaw_rate(self):
        c, s = math.cos(math.pi/12), math.sin(math.pi/12)
        quats = torch.tensor([[[c, s, 0., 0.]], [[c, 0., s, 0.]], [[2**-.5, 0., 0., 2**-.5]],
                              [[c, 0., s, 0.]], [[2**-.5, 0., 0., 2**-.5]]])
        angular = torch.tensor([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]],
                                [[0., 0., 1.]], [[0., 1., 0.]]])
        data = NS(body_quat_w=quats, body_ang_vel_w=angular,
                  GRAVITY_VEC_W=torch.tensor([[0., 0., -1.]]).repeat(5, 1))
        env = NS(scene={"robot": NS(data=data)})
        cfg = NS(name="robot", body_ids=[0])
        functions = self.posture_functions()
        params = config.environment_data()["rewards"]["torso_orientation"]["params"].copy()
        params["asset_cfg"] = cfg
        result = functions["h2_plus_torso_orientation_l2"](env, **params)
        torch.testing.assert_close(result, torch.tensor([1.5, .25, 0., .25, 0.]), atol=1e-6, rtol=1e-5)
        result = functions["h2_plus_torso_roll_velocity_l2"](env, cfg)
        torch.testing.assert_close(result, torch.tensor([1., 0., 0., 0., 1.]), atol=1e-6, rtol=1e-5)

    def test_pelvis_roll_penalties_measure_pelvis_even_when_torso_is_upright(self):
        rewards = config.environment_data()["rewards"]
        for name in ("pelvis_roll", "pelvis_roll_velocity"):
            self.assertEqual(rewards[name]["params"]["asset_cfg"]["body_names"], ["pelvis"])
            self.assertLess(rewards[name]["weight"], 0.)
        c, s = math.cos(math.pi/18), math.sin(math.pi/18)
        quats = torch.tensor([[[c, s, 0., 0.], [1., 0., 0., 0.]],
                              [[c, 0., s, 0.], [1., 0., 0., 0.]],
                              [[2**-.5, 0., 0., 2**-.5], [1., 0., 0., 0.]]])
        angular = torch.zeros(3, 2, 3)
        angular[0, 0, 0] = 1.
        angular[1, 0, 1] = 1.
        angular[2, 0, 2] = 1.
        env = NS(scene={"robot": NS(data=NS(body_quat_w=quats, body_ang_vel_w=angular,
                  GRAVITY_VEC_W=torch.tensor([[0., 0., -1.]]).repeat(3, 1)))})
        functions = self.posture_functions()
        params = rewards["pelvis_roll"]["params"].copy()
        params["asset_cfg"] = NS(name="robot", body_ids=[0])
        result = functions["h2_plus_torso_orientation_l2"](env, **params)
        torch.testing.assert_close(result, torch.tensor([math.sin(math.pi/9)**2, 0., 0.]), atol=1e-6, rtol=1e-5)
        velocity = functions["h2_plus_torso_roll_velocity_l2"](env, params["asset_cfg"])
        torch.testing.assert_close(velocity, torch.tensor([1., 0., 0.]), atol=1e-6, rtol=1e-5)
        torso = functions["h2_plus_torso_orientation_l2"](env, NS(name="robot", body_ids=[1]))
        torch.testing.assert_close(torso, torch.zeros(3))

    def test_foot_heading_penalizes_toe_in_out_and_allows_whole_robot_turning(self):
        def yaw(degrees):
            half = torch.deg2rad(torch.tensor(degrees))/2
            return torch.stack((half.cos(), torch.zeros_like(half), torch.zeros_like(half), half.sin()), dim=-1)
        root = yaw([0., 0., 0., 90., 179., 0., 0.])
        feet = yaw([[0., 0.], [20., -20.], [-20., 20.], [90., 90.], [-179., -179.], [20., 20.], [0., 0.]])
        c, s = math.cos(math.pi/12), math.sin(math.pi/12)
        feet[6] = torch.tensor([[c, s, 0., 0.], [c, 0., s, 0.]])  # roll/pitch only
        env = NS(scene={"robot": NS(data=NS(root_quat_w=root, body_quat_w=feet))})
        params = config.environment_data()["rewards"]["foot_heading"]["params"].copy()
        params["asset_cfg"] = NS(name="robot", body_ids=[0, 1])
        result = self.posture_functions()["h2_plus_foot_heading_l2"](env, **params)
        torch.testing.assert_close(result[[0, 3, 4, 6]], torch.zeros(4), atol=1e-7, rtol=1e-6)
        self.assertGreater(result[1].item(), 0.)
        torch.testing.assert_close(result[1], result[2])
        torch.testing.assert_close(result[1], result[5])  # parallel but both skewed still costs

    def test_foot_flat_reward_uses_contact_and_ignores_yaw(self):
        count = 8
        quat = torch.tensor([[[1., 0., 0., 0.]]]).repeat(count, 2, 1)
        quat[1, 0] = torch.tensor([2**-.5, 0., 0., 2**-.5])  # level with yaw
        quat[2, 0] = torch.tensor([2**-.5, 2**-.5, 0., 0.])  # rolled 90 deg
        quat[3, 0] = torch.tensor([0., 1., 0., 0.])  # upside down
        quat[0, 1] = quat[3, 0]  # airborne foot tilt must not affect support reward
        contact = torch.tensor([[1., 0.]]).repeat(count, 1)
        contact[4] = 0.  # flight
        contact[5] = 1.  # double support should still score 1, not 2
        duration = contact.clone()
        duration[6, 0] = .02  # new touchdown not eligible
        forces = torch.zeros(count, 2, 3); forces[:, :, 2] = contact*10
        forces[7, 0, 2] = .5  # below force threshold despite stale contact duration
        class Scene(dict):
            pass
        data = NS(body_quat_w=quat, GRAVITY_VEC_W=torch.tensor([[0., 0., -1.]]).repeat(count, 1))
        scene = Scene(robot=NS(data=data))
        scene.sensors = {name: NS(data=NS(net_forces_w=forces[:, i:i+1], current_contact_time=duration[:, i:i+1]))
                         for i, name in enumerate(("left_foot_contact", "right_foot_contact"))}
        params = config.environment_data()["rewards"]["contact_foot_flat"]["params"].copy()
        params["asset_cfg"] = NS(name="robot", body_ids=[0, 1])
        result = self.posture_functions()["h2_plus_contact_foot_flat_reward"](NS(scene=scene), **params)
        torch.testing.assert_close(result, torch.tensor([1., 1., 0., 0., 0., 1., 0., 0.]), atol=1e-6, rtol=1e-5)

    def test_torso_tilt_is_detected_even_with_upright_pelvis(self):
        functions = self.posture_functions()
        # Upright, 90-degree roll, and pure yaw: only roll should be penalized.
        quat = torch.tensor([[[1., 0., 0., 0.]], [[2**-.5, 2**-.5, 0., 0.]], [[2**-.5, 0., 0., 2**-.5]]])
        data = NS(body_quat_w=quat, GRAVITY_VEC_W=torch.tensor([[0., 0., -1.]]).repeat(3, 1))
        env = NS(scene={"robot": NS(data=data)})
        result = functions["h2_plus_torso_orientation_l2"](env, NS(name="robot", body_ids=[0]))
        torch.testing.assert_close(result, torch.tensor([0., 1., 0.]))

    def test_swing_clearance_gates_support_motion_and_commands(self):
        functions = self.posture_functions()
        count = 7
        positions = torch.zeros(count, 2, 3)
        positions[0, 1, 2] = .06  # target clearance
        positions[2] += 5.0      # elevated support surface: translation invariant
        velocities = torch.ones(count, 2, 3)
        velocities[5] = 0.
        commands = torch.tensor([[.5, 0., 0.]]).repeat(count, 1)
        commands[6] = 0.
        contact = torch.tensor([[1., 0.]]).repeat(count, 1)
        contact[3] = 1.  # double support
        contact[4] = 0.  # flight
        sensors = {name: NS(data=NS(current_contact_time=contact[:, i:i+1]))
                   for i, name in enumerate(("left_foot_contact", "right_foot_contact"))}
        class Scene(dict):
            pass
        scene = Scene(robot=NS(data=NS(body_pos_w=positions, body_lin_vel_w=velocities)))
        scene.sensors = sensors
        env = NS(scene=scene, command_manager=NS(get_command=lambda name: commands))
        params = config.environment_data()["rewards"]["swing_clearance"]["params"].copy()
        params["asset_cfg"] = NS(name="robot", body_ids=[0, 1])
        result = functions["h2_plus_swing_clearance_exp"](env, **params)
        torch.testing.assert_close(result, torch.tensor([1., math.exp(-1.44), math.exp(-1.44), 0., 0., 0., 0.]))

    def test_lateral_symmetry_allows_alternating_steps_and_turning(self):
        functions = self.posture_functions()
        positions = torch.tensor([[[.3, .1, 0.], [-.3, -.1, .06]],
                                  [[-.3, .1, .06], [.3, -.1, 0.]],
                                  [[.3, .2, 0.], [-.3, 0., .06]],
                                  [[.3, .2, 0.], [-.3, 0., .06]]])
        commands = torch.tensor([[.5, 0., 0.], [.5, 0., 0.], [.5, 0., 0.], [.5, 0., .5]])
        data = NS(body_pos_w=positions, root_pos_w=torch.zeros(4, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1))
        env = NS(scene={"robot": NS(data=data)}, command_manager=NS(get_command=lambda name: commands))
        params = config.environment_data()["rewards"]["feet_lateral_symmetry"]["params"].copy()
        params["asset_cfg"] = NS(name="robot", body_ids=[0, 1])
        result = functions["h2_plus_feet_lateral_symmetry_l2"](env, **params)
        torch.testing.assert_close(result, torch.tensor([0., 0., 4., 0.]))

    def test_zero_startup_and_independent_posture_references(self):
        initial = config.config_data("robot.init_state")
        self.assertEqual(initial["joint_pos"], {".*": 0.})
        self.assertEqual(initial["pos"], [0., 0., 1.02])
        cfg = config.environment_data()
        self.assertEqual(cfg["rewards"]["arm_posture"]["params"]["reference_angle"], 0.)
        self.assertNotIn("knee_posture", cfg["rewards"])
        for term in ("arm_posture", "stand_still", "waist_posture", "passive_joint_deviation"):
            patterns = cfg["rewards"][term]["params"]["asset_cfg"]["joint_names"]
            for name in ("left_knee_joint", "right_knee_joint", "left_elbow_joint", "right_elbow_joint"):
                self.assertFalse(any(re.fullmatch(pattern, name) for pattern in patterns), (term, name))
        env = NS(scene={"robot": NS(data=NS(joint_pos=torch.tensor([[0., 0.], [.35, .35]])))})
        result = self.posture_functions()["h2_plus_joint_reference_l1"](
            env, reference_angle=.35, asset_cfg=NS(name="robot", joint_ids=[0, 1]))
        torch.testing.assert_close(result, torch.tensor([.7, 0.]))

    def test_vertical_penalty_and_bounded_motion_bonus(self):
        tree = ast.parse((LAB / "h2_plus_lab/tasks/locomotion/mdp/rewards.py").read_text())
        names = {"h2_plus_torso_vertical_velocity_l2", "h2_plus_joint_motion_reward"}
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        scope = {"torch": torch}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+functions, type_ignores=[])), "<motion rewards>", "exec"), scope)
        rewards = config.environment_data()["rewards"]
        for term in ("knee_motion", "elbow_motion"):
            params = rewards[term]["params"].copy()
            peak = params["speed_deadband"] + params["speed_scale"]
            speed = torch.tensor([0., params["speed_deadband"], peak, -peak, 20., peak, peak, peak])
            commands = torch.tensor([[.5, 0., 0.]]).repeat(8, 1)
            commands[5] = 0.
            commands[7] = torch.tensor([0., 0., .3])
            linear = commands.clone(); linear[:, 2] = 0.; linear[6, 0] = 3.
            angular = torch.zeros(8, 3); angular[7, 2] = .3
            velocity = torch.zeros(8, 1, 3); velocity[0, 0, 2] = 2.; velocity[1, 0, 2] = -2.
            data = NS(joint_vel=speed[:, None].repeat(1, 2), root_lin_vel_b=linear,
                      root_ang_vel_b=angular, body_lin_vel_w=velocity)
            env = NS(scene={"robot": NS(data=data)}, command_manager=NS(get_command=lambda name: commands))
            params["asset_cfg"] = NS(name="robot", joint_ids=[0, 1])
            result = scope["h2_plus_joint_motion_reward"](env, **params)
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(((result >= 0) & (result <= 1.000001)).all())
            torch.testing.assert_close(result[[0, 1, 2, 3, 5, 7]], torch.tensor([0., 0., 1., 1., 0., 1.]))
            self.assertLess(result[4].item(), .001)  # excessive speed loses its bonus
            self.assertLess(result[6].item(), .001)  # poor tracking suppresses bonus
            vertical = scope["h2_plus_torso_vertical_velocity_l2"](env, NS(name="robot", body_ids=[0]))
            torch.testing.assert_close(vertical, torch.tensor([4., 4., 0., 0., 0., 0., 0., 0.]))

    def test_yaml_profiles_preserve_training_and_play_behavior(self):
        for profile, count, spacing in (("flat", 1024, 3.), ("low_friction", 1024, 5.), ("deformable", 64, 5.)):
            with self.subTest(profile=profile):
                train = config.environment_data(profile)
                play = config.environment_data(profile, True)
                self.assertEqual(train["scene"]["num_envs"], count)
                self.assertEqual(train["scene"]["env_spacing"], spacing)
                self.assertEqual(train["decimation"], 4)
                self.assertAlmostEqual(train["sim"]["dt"], 1/120)
                self.assertEqual(train["rewards"]["action_rate"]["weight"], -.01)
                self.assertEqual(train["rewards"]["feet_slide"]["params"]["asset_cfg"]["name"], "robot")
                self.assertTrue(train["observations"]["policy"]["enable_corruption"])
                self.assertEqual(train["events"]["reset_joints"]["params"]["position_range"], [-.02, .02])
                self.assertEqual(play["commands"]["base_velocity"]["rel_standing_envs"], 0.)
                self.assertEqual(play["scene"]["num_envs"], 16)
                self.assertEqual(play["scene"]["env_spacing"], spacing)
                self.assertFalse(play["observations"]["policy"]["enable_corruption"])
                self.assertIsNone(play["curriculum"])
                self.assertIsNone(play["events"]["push_robot"])
                self.assertIsNone(play["events"]["randomize_foot_material"])
                self.assertEqual(play["events"]["reset_joints"]["params"]["position_range"], [0., 0.])
                self.assertEqual(play["commands"]["base_velocity"]["ranges"]["lin_vel_x"], [.5, .5])

    def test_yaml_reward_overrides_keep_native_bindings(self):
        settings = config.load_settings()
        original = copy.deepcopy(config.environment_data()["rewards"]["feet_slide"])
        settings.environment.rewards.action_rate.weight = -.03
        with patch.object(config, "load_settings", return_value=settings):
            values = config.environment_data()
        self.assertEqual(values["rewards"]["action_rate"]["weight"], -.03)
        self.assertEqual(values["rewards"]["feet_slide"], original)

    def test_invalid_reward_values_are_rejected(self):
        for path, value in (("action_rate.weight", float("nan")), ("action_rate.weight", False),
                            ("track_lin_vel_xy.params.std", 0.), ("feet_air_time.params.threshold", -1.),
                            ("knee_inactivity.params.window_s", 0.), ("knee_inactivity.params.min_excursion", 0.),
                            ("knee_target_limits.params.std", 0.)):
            settings = config.load_settings()
            from omegaconf import OmegaConf
            OmegaConf.update(settings, "environment.rewards."+path, value)
            with self.subTest(path=path), patch.object(config, "load_settings", return_value=settings), self.assertRaises(ValueError):
                config.environment_data()

    def test_hydra_constructs_typed_extensible_groups(self):
        group = instantiate({
            "_target_": "h2_plus_lab.config.config_group", "class_path": f"{__name__}.ExampleGroup",
            "enabled": False, "reward": {"_target_": f"{__name__}.ExampleTerm", "weight": .2},
        }, _convert_="all")
        self.assertIsInstance(group, ExampleGroup)
        self.assertIsInstance(group.reward, ExampleTerm)
        self.assertFalse(group.enabled)
        self.assertEqual(group.reward.weight, .2)

    def test_native_configclass_initializes_profiles_and_rejects_unknown_settings(self):
        isaac = Path(importlib.util.find_spec("isaaclab").origin).parent
        tree = ast.parse((isaac / "utils/configclass.py").read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
        # Only serialization helpers are stubbed; exercise the real constructor/deepcopy decorator.
        scope = {"__name__": __name__, "class_to_dict": lambda obj: vars(obj), "update_class_from_dict": lambda *args: None}
        exec(compile(tree, "<native configclass>", "exec"), scope)
        decorate = scope["configclass"]
        base = decorate(type("Base", (), {"__annotations__": {"scene": object, "recorders": object}, "scene": MISSING, "recorders": None}))
        calls = []
        def data(profile, play):
            calls.append((profile, play))
            return {"scene": NS(num_envs=16 if play else 1024), "recorders": None}
        source = LAB / "h2_plus_lab/tasks/locomotion/env_cfg.py"
        classes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef)]
        scope = {"__name__": __name__, "ClassVar": ClassVar, "configclass": decorate, "ManagerBasedRLEnvCfg": base,
                 "instantiate_config": lambda values: values, "environment_data": data}
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), scope)
        for name, profile in (("H2PlusFlatEnvCfg", "flat"), ("H2PlusLowFrictionEnvCfg", "low_friction"), ("H2PlusDeformableEnvCfg", "deformable")):
            for play in (False, True):
                obj = scope[name+("_PLAY" if play else "")]()
                self.assertEqual(calls[-1], (profile, play))
                self.assertEqual(obj.scene.num_envs, 16 if play else 1024)
        scope["environment_data"] = lambda *args: {"misspelled_setting": 1}
        with self.assertRaises(ValueError):
            scope["H2PlusFlatEnvCfg"]()

    def test_course_finishes_within_current_training_budget(self):
        tree = ast.parse((LAB / "h2_plus_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        params = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id in ("max_iterations", "num_steps_per_env")}
        cfg = config.environment_data()
        curriculum = cfg["curriculum"]["command_range"]["params"]
        self.assertLess(curriculum["start_step"]+curriculum["ramp_steps"], params["max_iterations"]*params["num_steps_per_env"])
        self.assertGreater(cfg["commands"]["base_velocity"]["rel_standing_envs"], 0.)

    def test_turning_is_not_standing_and_can_receive_air_time_reward(self):
        source = ast.parse((LAB / "h2_plus_lab/tasks/locomotion/mdp/rewards.py").read_text())
        names = {"stand_still_joint_deviation_l1", "h2_plus_feet_air_time_positive_biped"}
        functions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        scope = {"torch": torch, "SceneEntityCfg": lambda name: NS(name=name),
                 "mdp": NS(joint_deviation_l1=lambda env, cfg: env.deviation)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+functions, type_ignores=[])), "<actual rewards>", "exec"), scope)
        commands = torch.tensor([[0., 0., 0.], [0., 0., .2], [.1, 0., 0.], [.02, 0., .04]])
        sensor = lambda air, contact: NS(data=NS(current_air_time=torch.full((4, 1), air), current_contact_time=torch.full((4, 1), contact)))
        env = NS(command_manager=NS(get_command=lambda name: commands), deviation=torch.ones(4),
                 scene=NS(sensors={"left_foot_contact": sensor(0., .2), "right_foot_contact": sensor(.3, 0.)}))
        rewards = config.environment_data()["rewards"]
        stand = rewards["stand_still"]["params"]
        result = scope["stand_still_joint_deviation_l1"](env, command_name=stand["command_name"],
            command_threshold=stand["command_threshold"], angular_command_threshold=stand["angular_command_threshold"])
        torch.testing.assert_close(result, torch.tensor([1., 0., 0., 1.]))
        result = scope["h2_plus_feet_air_time_positive_biped"](env, **rewards["feet_air_time"]["params"])
        torch.testing.assert_close(result, torch.tensor([0., .2, .2, 0.]))

    def test_non_foot_contact_penalty_excludes_the_two_supporting_feet(self):
        cfg = config.environment_data()
        expression = cfg["scene"]["body_contact"]["prim_path"].rsplit("/", 1)[-1]
        robot = ET.parse(LAB.parent / "robot_asset/h2_plus/H2_Plus_with_footrix_no_dex_hands.urdf").getroot()
        fixed = {joint.find("child").get("link") for joint in robot.findall("joint") if joint.get("type") == "fixed"}
        bodies = {link.get("name") for link in robot.findall("link")} - fixed
        selected = {name for name in bodies if re.fullmatch(expression, name)}
        self.assertEqual(selected, bodies-{"left_ankle_pitch_link", "right_ankle_pitch_link"})
        self.assertEqual(set(cfg["terminations"]["body_contact"]["params"]["sensor_cfg"]["body_names"]), {"pelvis", "torso_link"})
        self.assertLess(cfg["rewards"]["non_foot_contact"]["weight"], 0.)
        isaac = Path(importlib.util.find_spec("isaaclab").origin).parent
        tree = ast.parse((isaac / "envs/mdp/rewards.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "undesired_contacts")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        scope = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])), "<native contacts>", "exec"), scope)
        force = torch.zeros(3, 3, 2, 3)
        force[1, 0, 0, 2] = 2.
        force[2, 1, 0, 2] = .5
        env = NS(scene=NS(sensors={"body_contact": NS(data=NS(net_forces_w_history=force))}))
        result = scope["undesired_contacts"](env, threshold=cfg["rewards"]["non_foot_contact"]["params"]["threshold"], sensor_cfg=NS(name="body_contact", body_ids=slice(None)))
        torch.testing.assert_close(result, torch.tensor([0, 1, 0]))

    def test_yaml_vectors_restore_native_tuples_without_changing_lists(self):
        values = {"spawn": {"_target_": f"{__name__}.ExampleVectors", "color": [1, 0, 1],
                  "position": [0, 0, 1], "rotation": [1, 0, 0, 0], "names": ["hip", "knee"],
                  "tags": [["class", "robot"]]}}
        cfg = config.instantiate_config(values)["spawn"]
        self.assertIsInstance(cfg.color, tuple)
        self.assertEqual(cfg.color, (1., 0., 1.))
        self.assertTrue(all(type(value) is float for value in cfg.color))
        self.assertIsInstance(cfg.position, tuple)
        self.assertIsInstance(cfg.rotation, tuple)
        self.assertIsInstance(cfg.names, list)
        self.assertIsInstance(cfg.tags, list)
        # Simulate a Hydra override after construction and exercise pre-gym normalization.
        cfg.color = [0.8, 0.8, 0.8]
        cfg.position = None
        config.restore_config_types(cfg)
        self.assertEqual(cfg.color, (0.8, 0.8, 0.8))
        self.assertIsNone(cfg.position)
        self.assertIs(config.restore_config_types(cfg), cfg)

    def test_native_vector_shape_and_numeric_type_errors_fail_early(self):
        for color in ([1, 1], [1, 1, 1, 1], [True, 0, 0], ["red", 0, 0]):
            with self.subTest(color=color), self.assertRaises((ValueError, TypeError)):
                config.restore_config_types(ExampleVectors(color=color))

    def test_all_manager_groups_have_attribute_based_specs(self):
        for profile in ("flat", "low_friction", "deformable"):
            for play in (False, True):
                cfg = config.environment_data(profile, play)
                for name in ("actions", "commands", "rewards", "terminations", "events", "curriculum", "observations"):
                    if cfg[name] is None:
                        continue
                    with self.subTest(profile=profile, play=play, group=name):
                        self.assertEqual(cfg[name]["_target_"], "h2_plus_lab.config.config_group")
                        self.assertEqual(cfg[name]["class_path"], "h2_plus_lab.config.ManagerTermsCfg")
                self.assertIsNone(cfg["commands"]["base_velocity"]["ranges"]["heading"])
                self.assertFalse(cfg["commands"]["base_velocity"]["heading_command"])

    def test_native_command_and_action_manager_constructors_accept_typed_groups(self):
        isaac = Path(importlib.util.find_spec("isaaclab").origin).parent
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        for file_name, class_name, term_name in (("command_manager.py", "CommandManager", "base_velocity"),
                                                 ("action_manager.py", "ActionManager", "legs")):
            tree = ast.parse((isaac / "managers" / file_name).read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            scope = {"ManagerBase": ManagerTestBase, "torch": torch,
                     "CommandTermCfg": ManagerTestTermCfg, "CommandTerm": ManagerTestTerm,
                     "ActionTermCfg": ManagerTestTermCfg, "ActionTerm": ManagerTestTerm}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), "<native manager>", "exec"), scope)
            env = NS(num_envs=2, device="cpu")
            terms = {term_name: ManagerTestTermCfg()}
            with self.subTest(manager=class_name):
                # Reproduce the reported failure with the old plain-dict representation.
                with self.assertRaises(AttributeError):
                    scope[class_name](terms, env)
                group = config.config_group("h2_plus_lab.config.ManagerTermsCfg", **terms)
                manager = scope[class_name](group, env)
                self.assertTrue(manager.cfg.debug_vis)
                self.assertIn(term_name, manager._terms)
                self.assertNotIn("debug_vis", vars(group))  # runtime mutation stays in manager's copy
                if class_name == "ActionManager":
                    self.assertEqual(tuple(manager.action.shape), (2, 2))

    def test_configured_native_joint_reset_perturbs_zero_defaults(self):
        isaac = Path(importlib.util.find_spec("isaaclab").origin).parent
        event = config.environment_data()["events"]["reset_joints"]
        name = event["func"].split(":")[-1]
        tree = ast.parse((isaac / "envs/mdp/events.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        scope = {"SceneEntityCfg": lambda name: NS(name=name, joint_ids=slice(None)),
                 "math_utils": NS(sample_uniform=lambda low, high, shape, device: torch.full(shape, high, device=device))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])), "<native reset>", "exec"), scope)
        state = {}
        data = NS(default_joint_pos=torch.zeros(2, 3), default_joint_vel=torch.zeros(2, 3),
                  soft_joint_pos_limits=torch.tensor([[[-1., 1.]]*3]*2), soft_joint_vel_limits=torch.ones(2, 3))
        asset = NS(data=data, write_joint_state_to_sim=lambda pos, vel, **kwargs: state.update(pos=pos, vel=vel))
        env = NS(scene={"robot": asset})
        scope[name](env, torch.tensor([0, 1]), **event["params"])
        torch.testing.assert_close(state["pos"], torch.full((2, 3), .02))
        torch.testing.assert_close(state["vel"], torch.full((2, 3), .05))
        scope[name](env, torch.tensor([0, 1]), **config.environment_data(play=True)["events"]["reset_joints"]["params"])
        self.assertTrue(torch.all(state["pos"] == 0))
        self.assertTrue(torch.all(state["vel"] == 0))

    def test_curriculum_uses_native_command_manager_public_api(self):
        isaac = Path(importlib.util.find_spec("isaaclab").origin).parent
        tree = ast.parse((isaac / "managers/command_manager.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CommandManager")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        scope = {"ManagerBase": object}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), "<native CommandManager>", "exec"), scope)
        manager = scope["CommandManager"].__new__(scope["CommandManager"])
        ranges = NS()
        manager._terms = {"base_velocity": NS(cfg=NS(ranges=ranges))}
        curriculum = runpy.run_path(str(LAB / "h2_plus_lab/tasks/locomotion/mdp/curriculums.py"))["h2_plus_command_curriculum"]
        env = NS(command_manager=manager, common_step_counter=0)
        params = config.environment_data()["curriculum"]["command_range"]["params"]
        ramp = params["ramp_steps"]
        for step, expected in ((0, .2), (ramp//2, .6), (ramp, 1.), (ramp+ramp//2, 1.)):
            env.common_step_counter = step
            result = curriculum(env, [0], **params)
            self.assertAlmostEqual(ranges.lin_vel_x[1], expected)
            self.assertAlmostEqual(result["ang_vel_z_abs_max"], expected)
            self.assertIsNone(ranges.heading)


if __name__ == "__main__":
    unittest.main()
