"""Native RecorderManager/HDF5 lifecycle with synthetic sensor buffers, no Sim.

Simulator-facing construction and the PhysX event source are stubbed. Layout
constants come from the actual sensor config; recorder callbacks, EpisodeData,
projection calculations and HDF5 handling execute unchanged.
"""

import ast
import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
import weakref
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
loc = LAB / "h2_plus_lab/tasks/locomotion"
constants = [n for n in ast.parse((loc / "sensor_cfg.py").read_text()).body
    if isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.startswith("H2_PLUS_FOOTRIX_") for t in n.targets)]
from h2_plus_lab.config import config_data
sensor_constants = module("_h2_test_locomotion.sensor_cfg", Path=Path, _FOOTRIX=config_data("sensors.footrix"))
exec(compile(ast.Module(body=constants, type_ignores=[]), str(loc / "sensor_cfg.py"), "exec"), sensor_constants.__dict__)
with patch.dict(sys.modules, {
    "isaaclab.managers.recorder_manager": native,
    "_h2_test_locomotion": module("_h2_test_locomotion", __path__=[str(loc)]),
    "_h2_test_locomotion.mdp": module("_h2_test_locomotion.mdp", __path__=[str(loc / "mdp")]),
    "_h2_test_locomotion.sensor_cfg": sensor_constants,
}):
    recording = load_file("_h2_test_locomotion.mdp.recorders", loc / "mdp/recorders.py")


class ContactSubscription:
    def __init__(self, callback):
        self.callback = callback


class ContactInterface:
    def __init__(self):
        self.subscriptions = []

    def subscribe_contact_report_events(self, callback):
        subscription = ContactSubscription(callback)
        self.subscriptions.append(weakref.ref(subscription))
        return subscription

    def emit(self, headers, contacts):
        for reference in self.subscriptions:
            subscription = reference()
            if subscription is not None:
                subscription.callback(headers, contacts)

    @property
    def active_count(self):
        return sum(reference() is not None for reference in self.subscriptions)


def contact_runtime(interface):
    physx = module("omni.physx", get_physx_simulation_interface=lambda: interface)
    return patch.dict(sys.modules, {
        "omni": module("omni", __path__=[], physx=physx), "omni.physx": physx,
        "pxr": module("pxr", PhysicsSchemaTools=NS(intToSdfPath=lambda path: path)),
    })


def contact_header(env_id):
    return NS(actor0=f"/World/envs/env_{env_id}/Robot/left_ankle_pitch_link", actor1="/World/ground",
        contact_data_offset=0, num_contact_data=1)


def make_term(env, ids, stride):
    interface = ContactInterface()
    with contact_runtime(interface):
        term = recording.H2PlusSensorRecorder(NS(env_ids=ids, record_every=stride), env)
    env.contact_interface = interface
    return term


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
        body_pos_w=torch.zeros(num_envs, 2, 3),
        body_quat_w=torch.tensor([[[1., 0., 0., 0.]] * 2] * num_envs),
    )
    scene = Scene(robot=NS(data=data, joint_names=["joint_a", "joint_b"], body_names=["left_ankle_pitch_link", "right_ankle_pitch_link"], cfg=NS(spawn=NS(asset_path=str(urdf)))))
    data._physics_sim_view = NS(update_articulations_kinematic=lambda: None)
    scene["robot"].root_physx_view = NS(get_link_transforms=lambda: torch.cat(
        (data.body_pos_w, data.body_quat_w[:, :, [1, 2, 3, 0]]), dim=-1))
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
    term = make_term(env, ids, stride)
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
    position = manager._terms["sensors"]._footrix_layout_b["left"][0].tolist()
    for _ in range(4):
        for env_id in range(env.num_envs):
            env.contact_interface.emit([contact_header(env_id)], [NS(position=position, impulse=(0., 0., value * env.physics_dt))])
        manager.record_post_physics_decimation_step()
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
            self.assertEqual(env.contact_interface.active_count, 0)
            with h5py.File(Path(tmp) / "dataset/sensors.hdf5", "r") as file:
                self.assertEqual(len(file["data"]), 2)  # env 0 has no data
                complete, partial = file["data/demo_0"], file["data/demo_1"]
                self.assertEqual(complete.attrs["num_samples"], 2)
                self.assertEqual(complete["env_id"][:, 0].tolist(), [1, 1])
                self.assertEqual(complete["episode_step"][:, 0].tolist(), [2, 3])
                self.assertEqual(complete["imu/torso_imu/linear_accel_b_m_s2"][:, 0].tolist(), [2, 3])
                self.assertEqual(complete["terminated"][:].tolist(), [False, True])
                forces = torch.tensor(complete["footrix/left/force_w_N"][:])
                torch.testing.assert_close(forces.sum(dim=1)[:, 2], torch.tensor([2., 3.]))
                self.assertEqual(complete["footrix/left/source_contact_count"][:, 0].tolist(), [4, 4])
                self.assertAlmostEqual(complete["simulation_time_s"][-1, 0], 12/120)
                self.assertEqual(partial["episode_id"][:, 0].tolist(), [1])
                self.assertFalse(partial["reset"][-1])
                self.assertEqual(file["data"].attrs["total"], 3)
            metadata = json.loads((Path(tmp) / "dataset/metadata.json").read_text())
            self.assertEqual(metadata["recorded_env_ids"], [1])
            self.assertEqual(metadata["nominal_record_hz"], 15)
            self.assertFalse(metadata["imu_noise_added_by_recorder"])
            self.assertEqual(metadata["schema_version"], 2)
            self.assertNotIn("support_radius_m", metadata["footrix"])
            self.assertEqual(metadata["footrix"]["projection_method"], "strict_circular_contact_regions")

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

    def test_inverse_rotation_for_known_unit_quaternion(self):
        vectors = torch.eye(3)
        quat = torch.tensor([2**-0.5, 0., 0., 2**-0.5]).expand(3, -1)
        result = recording.H2PlusSensorRecorder._quat_rotate_inverse(quat, vectors)
        expected = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result.norm(dim=-1), vectors.norm(dim=-1))

    def test_strict_disks_exclude_gaps_and_side_contacts_without_spreading(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = make_env(Path(tmp)); term = make_term(env, [1], 1)
            try:
                radius = recording.H2_PLUS_FOOTRIX_CELL_RADIUS_M
                layout = term._footrix_layout_b["left"]
                layout.fill_(100.)
                layout[0] = torch.tensor([0., 0., -0.0521])
                layout[1] = torch.tensor([3*radius, 0., -0.0521])
                impulses = torch.tensor([[.02, -.01, .1], [-.03, .005, .05], [0., 0., .2], [0., 0., .3]])
                packets = {(1, "left"): [
                    ((radius/2, 0., -.0521), tuple(impulses[0].tolist())),
                    ((3*radius, 0., -.0521), tuple(impulses[1].tolist())),
                    ((1.5*radius, 0., -.0521), tuple(impulses[2].tolist())),  # gap
                    ((0., 0., 0.), tuple(impulses[3].tolist())),  # same xy but not sole surface
                ], (1, "right"): []}
                result = term._project_footrix_contacts(env, [1], packets)
                forces = result["left"]["force_w_N"][0]
                expected = impulses[:2] / env.step_dt
                torch.testing.assert_close(forces[:2], expected, rtol=1e-5, atol=2e-5)
                self.assertTrue((forces[2:] == 0).all())
                torch.testing.assert_close(result["left"]["unmeasured_force_w_N"][0]*env.step_dt, impulses[2:].sum(0))
                torch.testing.assert_close((forces.sum(0)+result["left"]["unmeasured_force_w_N"][0])*env.step_dt, impulses.sum(0))
                self.assertEqual(result["left"]["selected_contact_count"].item(), 2)
                self.assertEqual(result["left"]["unmeasured_contact_count"].item(), 2)
                self.assertEqual(result["left"]["cell_contact_count"][0, :2].tolist(), [1, 1])
            finally:
                term.close("")

    def test_substep_pose_used_even_when_final_foot_moves_to_another_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = make_env(Path(tmp)); term = make_term(env, [1], 1)
            try:
                local = term._footrix_layout_b["left"][0].clone()
                robot = env.scene["robot"]
                robot.data.body_quat_w[1, 0] = torch.tensor([2**-.5, 0., 0., 2**-.5])
                for shift in (0., .05):
                    robot.data.body_pos_w[1, 0] = torch.tensor([shift, -.1, 0.])
                    world = torch.tensor([-local[1]+shift, local[0]-.1, local[2]])
                    env.contact_interface.emit([contact_header(1)], [NS(position=world.tolist(), impulse=(0., 0., .01))])
                    term.record_post_physics_decimation_step()
                packets = {(1, side): term._footrix_local_packets.pop((1, side), []) for side in term.FOOTRIX_SIDES}
                result = term._project_footrix_contacts(env, [1], packets)
                self.assertAlmostEqual(result["left"]["force_w_N"][0, 0, 2].item()*env.step_dt, .02, places=6)
                self.assertEqual(result["left"]["selected_contact_count"].item(), 2)
            finally:
                term.close("")

    def test_overlapping_regions_and_invalid_radius_are_rejected(self):
        radius = recording.H2_PLUS_FOOTRIX_CELL_RADIUS_M
        for value in (0., float("nan"), .01):
            with self.subTest(radius=value), patch.object(recording, "H2_PLUS_FOOTRIX_CELL_RADIUS_M", value), self.assertRaises(ValueError):
                recording.H2PlusSensorRecorder._load_footrix_layout("cpu")

    def test_left_and_right_layouts_are_loaded_separately(self):
        positions = recording.H2PlusSensorRecorder._load_footrix_layout("cpu")
        project = Path(recording.__file__).resolve().parents[6]
        source = json.loads((project / recording.H2_PLUS_FOOTRIX_LAYOUT_PATH).read_text())
        for side in ("left", "right"):
            expected = torch.tensor(source["feet"][side]["observation_layout"]["sample_positions_in_foot_link_m"])
            expected[:, 2] = recording.H2_PLUS_FOOTRIX_LAYOUT_Z_M
            torch.testing.assert_close(positions[side], expected)
        self.assertNotEqual(positions["left"].data_ptr(), positions["right"].data_ptr())

    def test_unselected_envs_are_not_buffered_and_closed_callback_is_inert(self):
        interface = ContactInterface()
        with contact_runtime(interface):
            buffer = recording._FootrixContactReportBuffer(1/120, env_ids=[0])
        contact = NS(position=(0., 0., 0.), impulse=(0., 0., 1.))
        for _ in range(100):
            interface.emit([contact_header(1)], [contact]); buffer.pop([0])
        self.assertEqual(len(buffer._packets), 0)
        interface.emit([contact_header(0)], [contact])
        self.assertEqual(len(buffer._packets[(0, "left")]), 1)
        buffer.close(); buffer.close()
        self.assertEqual(interface.active_count, 0)
        buffer._on_contact_report([contact_header(0)], [contact])
        self.assertEqual(len(buffer._packets), 0)

    def test_reset_discards_only_reset_env_contact_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = make_env(Path(tmp)); manager = attach_manager(env, None, 1)
            contact = NS(position=(0., 0., 0.), impulse=(0., 0., 1.))
            for env_id in range(env.num_envs):
                env.contact_interface.emit([contact_header(env_id)], [contact])
            manager.record_post_physics_decimation_step()
            term = manager._terms["sensors"]
            self.assertIn((0, "left"), term._footrix_local_packets)
            self.assertIn((1, "left"), term._footrix_local_packets)
            for env_id in range(env.num_envs):
                env.contact_interface.emit([contact_header(env_id)], [contact])
            manager.reset([0])
            reports = manager._terms["sensors"]._footrix_contact_reports
            self.assertNotIn((0, "left"), reports._packets)
            self.assertEqual(len(reports._packets[(1, "left")]), 1)
            self.assertNotIn((0, "left"), term._footrix_local_packets)
            self.assertIn((1, "left"), term._footrix_local_packets)
            manager.close()
            self.assertEqual(env.contact_interface.active_count, 0)

    def test_bad_selection_or_stride_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = make_env(Path(tmp))
            for ids, stride in [([], 1), ([0, 0], 1), ([2], 1), ([-1], 1), ([True], 1), ([0], 0)]:
                with self.subTest(ids=ids, stride=stride), self.assertRaises(ValueError):
                    recording.H2PlusSensorRecorder(NS(env_ids=ids, record_every=stride), env)


if __name__ == "__main__":
    unittest.main()
