"""Native RecorderManager/HDF5 lifecycle with synthetic sensor buffers, no Sim.

Only ManagerBase (which imports omni.timeline) and configclass are replaced by
small construction stubs. RecorderManager callbacks, EpisodeData and the HDF5
handler are loaded unchanged from the installed Isaac Lab source.
"""

import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import h5py
import torch

LAB = Path(__file__).resolve().parents[1]
ISAAC = Path(importlib.util.find_spec("isaaclab").origin).parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def module(name, **members):
    result = types.ModuleType(name)
    result.__dict__.update(members)
    return result


# Dataset classes have no simulator dependencies.
datasets = module("_h2_test_datasets", __path__=[str(ISAAC / "utils/datasets")])
sys.modules[datasets.__name__] = datasets
EpisodeData = importlib.import_module("_h2_test_datasets.episode_data").EpisodeData
HDF5Handler = importlib.import_module("_h2_test_datasets.hdf5_dataset_file_handler").HDF5DatasetFileHandler


class TermBase:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env

    def reset(self, env_ids=None):
        pass


stubs = {
    "_h2_test_managers": module("_h2_test_managers", __path__=[]),
    "_h2_test_managers.manager_base": module("base", ManagerBase=object, ManagerTermBase=TermBase),
    "_h2_test_managers.manager_term_cfg": module("cfg", RecorderTermCfg=object),
    "isaaclab.utils": module("utils", configclass=lambda cls: cls),
    "isaaclab.utils.datasets": module("datasets", EpisodeData=EpisodeData, HDF5DatasetFileHandler=HDF5Handler),
}
with patch.dict(sys.modules, stubs):
    native = load_file("_h2_test_managers.recorder_manager", ISAAC / "managers/recorder_manager.py")
with patch.dict(sys.modules, {"isaaclab.managers.recorder_manager": native}):
    recording = load_file("_h2_sensor_recorder_test", LAB / "h2_plus_lab/tasks/locomotion/mdp/recorders.py")


class Scene(dict):
    pass


def make_env(directory, num_envs=2):
    urdf = directory / "robot.urdf"
    urdf.write_text("<robot name='synthetic-test'/>")
    data = NS(
        root_pos_w=torch.zeros(num_envs, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]] * num_envs),
        root_lin_vel_w=torch.zeros(num_envs, 3), root_ang_vel_w=torch.zeros(num_envs, 3),
        joint_pos=torch.zeros(num_envs, 2), joint_vel=torch.zeros(num_envs, 2),
        joint_pos_target=torch.ones(num_envs, 2),
    )
    scene = Scene(robot=NS(data=data, joint_names=["joint_a", "joint_b"], cfg=NS(spawn=NS(asset_path=str(urdf)))))
    scene.sensors = {}
    for name in recording.H2PlusSensorRecorder.IMUS:
        scene.sensors[name] = NS(
            cfg=NS(prim_path=name, update_period=1/120, offset=NS(pos=(0, 0, 0), rot=(1, 0, 0, 0)), gravity_bias=(0, 0, 9.81)),
            data=NS(lin_acc_b=torch.zeros(num_envs, 3), ang_vel_b=torch.ones(num_envs, 3)),
        )
    for name in recording.H2PlusSensorRecorder.CONTACTS:
        scene.sensors[name] = NS(
            cfg=NS(prim_path=name, update_period=0, force_threshold=1., track_air_time=name != "body_contact"),
            body_names=[name],
            data=NS(net_forces_w=torch.zeros(num_envs, 1, 3), current_air_time=torch.zeros(num_envs, 1), current_contact_time=torch.zeros(num_envs, 1)),
        )
    resets = torch.zeros(num_envs, dtype=torch.bool)
    cfg = NS(seed=42, log_dir=str(directory / "log"), env_name="test-task", sim=NS(dt=1/120, render_interval=4),
             decimation=4, scene=NS(num_envs=num_envs), recorders=NS(dataset_export_dir_path=str(directory / "dataset")))
    return NS(
        num_envs=num_envs, device="cpu", cfg=cfg, scene=scene, physics_dt=1/120, step_dt=1/30, _sim_step_counter=0,
        reset_buf=resets, reset_terminated=resets.clone(), reset_time_outs=resets.clone(), reward_buf=torch.ones(num_envs),
        action_manager=NS(action=torch.zeros(num_envs, 2), active_terms=["legs"], action_term_dim=[2]),
        command_manager=NS(get_command=lambda name: torch.zeros(num_envs, 3)),
        termination_manager=NS(active_terms=["bad_orientation"], get_term=lambda name: resets),
    )


def attach_manager(env, ids, stride):
    term = recording.H2PlusSensorRecorder(NS(env_ids=ids, record_every=stride), env)
    cfg = native.RecorderManagerBaseCfg()
    cfg.dataset_export_dir_path = env.cfg.recorders.dataset_export_dir_path
    cfg.dataset_filename = "sensors"
    cfg.export_in_close = True
    manager = native.RecorderManager.__new__(native.RecorderManager)
    manager.cfg, manager._env = cfg, env
    manager._term_names, manager._terms = ["sensors"], {"sensors": term}
    manager._episodes = {i: EpisodeData() for i in range(env.num_envs)}
    manager._exported_successful_episode_count = {}
    manager._exported_failed_episode_count = {}
    manager._failed_episode_dataset_file_handler = None
    manager._dataset_file_handler = HDF5Handler()
    manager._dataset_file_handler.create(str(Path(cfg.dataset_export_dir_path) / "sensors"), env_name="test-task")
    env.recorder_manager = manager
    return manager


def step(env, manager, value):
    env._sim_step_counter += 4
    env.scene["robot"].data.joint_pos.fill_(value)
    env.scene.sensors["torso_imu"].data.lin_acc_b.fill_(value)
    env.scene.sensors["left_foot_contact"].data.net_forces_w[:, :, 2] = value
    env.action_manager.action.fill_(value / 10)
    manager.record_post_step()


class SensorRecordingTests(unittest.TestCase):
    def test_terminal_frame_reset_and_partial_close_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = make_env(Path(tmp))
            manager = attach_manager(env, [1], 2)
            manager.record_post_physics_decimation_step()
            self.assertTrue(manager.get_episode(1).is_empty())
            step(env, manager, 1)  # downsampled away
            step(env, manager, 2)
            env.reset_buf[1] = env.reset_terminated[1] = True
            step(env, manager, 3)  # terminal frame must be kept despite stride
            env.scene.sensors["torso_imu"].data.lin_acc_b.fill_(999)  # prove snapshots don't alias
            manager.record_pre_reset([0, 1])
            manager.reset([0, 1])
            manager.record_post_reset([0, 1])
            env.reset_buf.zero_(); env.reset_terminated.zero_()
            step(env, manager, 4)
            step(env, manager, 5)
            manager.close()  # native export_in_close saves unfinished episode
            manager.close()  # environment destruction may close it again
            with h5py.File(Path(tmp) / "dataset/sensors.hdf5", "r") as file:
                self.assertEqual(len(file["data"]), 2)  # env 0 has no data
                complete, partial = file["data/demo_0"], file["data/demo_1"]
                self.assertEqual(complete.attrs["num_samples"], 2)
                self.assertEqual(complete["env_id"][:, 0].tolist(), [1, 1])
                self.assertEqual(complete["episode_step"][:, 0].tolist(), [2, 3])
                self.assertEqual(complete["imu/torso_imu/linear_accel_b_m_s2"][:, 0].tolist(), [2, 3])
                self.assertEqual(complete["terminated"][:].tolist(), [False, True])
                self.assertAlmostEqual(complete["simulation_time_s"][-1, 0], 12/120)
                self.assertEqual(partial["episode_id"][:, 0].tolist(), [1])
                self.assertFalse(partial["reset"][-1])
                self.assertEqual(file["data"].attrs["total"], 3)
            metadata = json.loads((Path(tmp) / "dataset/metadata.json").read_text())
            self.assertEqual(metadata["recorded_env_ids"], [1])
            self.assertEqual(metadata["nominal_record_hz"], 15)
            self.assertFalse(metadata["imu_noise_added_by_recorder"])

    def test_all_envs_and_raw_invalid_sample_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = make_env(Path(tmp))
            manager = attach_manager(env, None, 1)
            env.scene.sensors["right_foot_imu"].data.ang_vel_b[0, 0] = float("nan")
            step(env, manager, 10)
            manager.close()
            with h5py.File(Path(tmp) / "dataset/sensors.hdf5", "r") as file:
                self.assertEqual(len(file["data"]), 2)
                sample = file["data/demo_0"]
                self.assertFalse(sample["imu/right_foot_imu/finite"][0])
                self.assertTrue(sample["contact/left_foot_contact/in_contact"][0, 0])
                self.assertFalse(sample["contact/right_foot_contact/in_contact"][0, 0])

    def test_bad_selection_or_stride_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = make_env(Path(tmp))
            for ids, stride in [([], 1), ([0, 0], 1), ([2], 1), ([-1], 1), ([True], 1), ([0], 0)]:
                with self.subTest(ids=ids, stride=stride), self.assertRaises(ValueError):
                    recording.H2PlusSensorRecorder(NS(env_ids=ids, record_every=stride), env)


if __name__ == "__main__":
    unittest.main()
