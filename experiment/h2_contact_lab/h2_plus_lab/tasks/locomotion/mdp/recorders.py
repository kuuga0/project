"""Record sensors from the live training scene through Isaac Lab RecorderManager.

Post-step runs after scene updates and before automatic resets. Sampling is at
control-step resolution; the physics-substep callback in Lab 2.3.2 is too early
for reading fresh scene sensor buffers. Native episode buffering/export remains
owned by RecorderManager, including reset boundaries and HDF5 serialization.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from isaaclab.managers.recorder_manager import RecorderTerm

from ..sensor_cfg import (
    H2_PLUS_FOOTRIX_CELL_COUNT,
    H2_PLUS_FOOTRIX_CELL_RADIUS_M,
    H2_PLUS_FOOTRIX_CELL_THICKNESS_M,
    H2_PLUS_FOOTRIX_CHANNELS_PER_CELL,
    H2_PLUS_FOOTRIX_LAYOUT_PATH,
    H2_PLUS_FOOTRIX_LAYOUT_Z_M,
    H2_PLUS_FOOTRIX_LAYOUT_STATUS,
    H2_PLUS_FOOTRIX_PROJECTION_METHOD,
    H2_PLUS_FOOTRIX_SUPPORT_RADIUS_M,
)

__all__ = ["H2PlusSensorRecorder"]


class _FootrixContactReportBuffer:
    """Collect full-foot PhysX contact impulses before point projection."""

    SIDES = ("left", "right")

    def __init__(self, physics_dt: float):
        try:
            import omni.physx
            from pxr import PhysicsSchemaTools
        except ImportError as exc:  # pragma: no cover - only reached outside Kit
            raise RuntimeError("FOOTRIX contact reports require the Isaac Sim omni.physx runtime") from exc
        self._physics_dt = float(physics_dt)
        self._physics_schema_tools = PhysicsSchemaTools
        self._packets = defaultdict(list)
        # Current Omni Physics documents expose this on the simulation
        # interface; older Isaac Sim builds exposed the same method through
        # get_physx_interface().
        get_simulation_interface = getattr(omni.physx, "get_physx_simulation_interface", None)
        simulation_interface = (
            get_simulation_interface() if get_simulation_interface is not None else omni.physx.get_physx_interface()
        )
        self._subscription = simulation_interface.subscribe_contact_report_events(self._on_contact_report)

    @staticmethod
    def _side_for_path(path: str):
        if "left_ankle_pitch_link" in path:
            return "left"
        if "right_ankle_pitch_link" in path:
            return "right"
        return None

    def _on_contact_report(self, contact_headers, contact_data):
        for header in contact_headers:
            actor0 = str(self._physics_schema_tools.intToSdfPath(header.actor0))
            actor1 = str(self._physics_schema_tools.intToSdfPath(header.actor1))
            side0 = self._side_for_path(actor0)
            side1 = self._side_for_path(actor1)
            if side0 is None and side1 is None:
                continue
            foot_is_actor0 = side0 is not None
            side = side0 if foot_is_actor0 else side1
            foot_path = actor0 if foot_is_actor0 else actor1
            match = re.search(r"/envs/env_(\d+)/", foot_path)
            if match is None:
                continue
            env_id = int(match.group(1))
            sign = 1.0 if foot_is_actor0 else -1.0
            start = int(header.contact_data_offset)
            stop = start + int(header.num_contact_data)
            for index in range(start, stop):
                contact = contact_data[index]
                position = tuple(float(contact.position[i]) for i in range(3))
                impulse = tuple(sign * float(contact.impulse[i]) for i in range(3))
                self._packets[(env_id, side)].append((position, impulse))

    def pop(self, env_ids):
        packets = {}
        for env_id in env_ids:
            for side in self.SIDES:
                packets[(int(env_id), side)] = self._packets.pop((int(env_id), side), [])
        return packets

    def close(self):
        self._subscription = None
        self._packets.clear()


class H2PlusSensorRecorder(RecorderTerm):
    """Capture selected environments without constructing another simulation."""

    IMUS = ("torso_imu", "left_foot_imu", "right_foot_imu")
    CONTACTS = ("left_foot_contact", "right_foot_contact", "body_contact")
    FOOTRIX_SIDES = ("left", "right")
    FOOTRIX_CELL_COUNT = H2_PLUS_FOOTRIX_CELL_COUNT

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids = list(range(env.num_envs)) if cfg.env_ids is None else list(cfg.env_ids)
        if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < env.num_envs for i in ids):
            raise ValueError("recorders.sensors.env_ids must contain unique valid environment indices, or null for all")
        if type(cfg.record_every) is not int or cfg.record_every < 1:
            raise ValueError("recorders.sensors.record_every must be a positive integer")
        self._ids = ids
        self._steps = {i: 0 for i in ids}
        self._episode_ids = {i: 0 for i in ids}
        self._metadata_written = False
        self._footrix_layout_b = self._load_footrix_layout(env.device)
        robot = env.scene["robot"]
        self._footrix_body_indices = {
            "left": robot.body_names.index("left_ankle_pitch_link"),
            "right": robot.body_names.index("right_ankle_pitch_link"),
        }
        self._footrix_contact_reports = _FootrixContactReportBuffer(env.physics_dt)

    @staticmethod
    def _load_footrix_layout(device):
        # mdp/recorders.py -> project/experiment/h2_contact_lab/...; the
        # layout path is project-relative, so parents[6] is the project root.
        project_root = Path(__file__).resolve().parents[6]
        path = project_root / H2_PLUS_FOOTRIX_LAYOUT_PATH
        layout = json.loads(path.read_text(encoding="utf-8"))
        positions = {}
        for side in ("left", "right"):
            side_positions = layout["feet"][side]["observation_layout"]["sample_positions_in_foot_link_m"]
            if len(side_positions) != H2_PLUS_FOOTRIX_CELL_COUNT:
                raise ValueError(
                    f"FOOTRIX {side} layout must contain {H2_PLUS_FOOTRIX_CELL_COUNT} measurement points"
                )
            positions[side] = torch.tensor(side_positions, dtype=torch.float32, device=device)
            positions[side][:, 2] = H2_PLUS_FOOTRIX_LAYOUT_Z_M
        return positions

    @staticmethod
    def _quat_rotate_inverse(quat, vector):
        q_xyz = quat[..., 1:]
        q_w = quat[..., 0:1]
        cross_1 = torch.cross(q_xyz, vector, dim=-1)
        cross_2 = torch.cross(q_xyz, vector + q_w * cross_1, dim=-1)
        return vector - 2.0 * cross_2

    def _project_footrix_contacts(self, env, ids, packets):
        robot = env.scene["robot"]
        projected = {
            "left": torch.zeros(
                (len(ids), self.FOOTRIX_CELL_COUNT, 3), dtype=torch.float32, device=env.device
            ),
            "right": torch.zeros(
                (len(ids), self.FOOTRIX_CELL_COUNT, 3), dtype=torch.float32, device=env.device
            ),
        }
        source_counts = {
            "left": torch.zeros((len(ids), 1), dtype=torch.int64),
            "right": torch.zeros((len(ids), 1), dtype=torch.int64),
        }
        for row, env_id in enumerate(ids):
            for side in self.FOOTRIX_SIDES:
                samples = packets[(int(env_id), side)]
                source_counts[side][row, 0] = len(samples)
                if not samples:
                    continue
                positions_w = torch.tensor(
                    [sample[0] for sample in samples], dtype=torch.float32, device=env.device
                )
                impulses = torch.tensor(
                    [sample[1] for sample in samples], dtype=torch.float32, device=env.device
                )
                body_index = self._footrix_body_indices[side]
                body_pos_w = robot.data.body_pos_w[env_id, body_index]
                body_quat_w = robot.data.body_quat_w[env_id, body_index]
                positions_b = self._quat_rotate_inverse(
                    body_quat_w.expand_as(positions_w), positions_w - body_pos_w
                )
                forces_w = impulses / float(env.step_dt)
                # FOOTRIX cells form a planar sole layout.  Use local x/y
                # distances so small residual sole-height errors do not
                # change the force distribution.
                distances = torch.cdist(
                    positions_b[:, :2], self._footrix_layout_b[side][:, :2]
                )
                weights = torch.clamp(
                    1.0 - distances / float(H2_PLUS_FOOTRIX_SUPPORT_RADIUS_M), min=0.0
                )
                weight_sums = weights.sum(dim=1, keepdim=True)
                missing = weight_sums.squeeze(1) <= torch.finfo(weights.dtype).eps
                if missing.any():
                    nearest = distances[missing].argmin(dim=1)
                    weights[missing] = 0.0
                    weights[missing, nearest] = 1.0
                    weight_sums = weights.sum(dim=1, keepdim=True)
                weights = weights / weight_sums.clamp_min(torch.finfo(weights.dtype).eps)
                projected[side][row] += torch.mm(weights.transpose(0, 1), forces_w)
        finite = {
            side: torch.isfinite(projected[side]).all(-1)
            for side in self.FOOTRIX_SIDES
        }
        in_contact = {
            side: finite[side] & (projected[side].norm(dim=-1) > 1.0)
            for side in self.FOOTRIX_SIDES
        }
        return {
            side: {
                "force_w_N": projected[side].detach().cpu().clone(),
                "finite": finite[side].detach().cpu().clone(),
                "in_contact": in_contact[side].detach().cpu().clone(),
                "source_contact_count": source_counts[side],
            }
            for side in self.FOOTRIX_SIDES
        }

    def reset(self, env_ids=None):
        """Advance episode IDs only after an episode actually had a step."""
        ids = self._ids if env_ids is None else env_ids
        for env_id in ids:
            env_id = int(env_id)
            if env_id in self._steps:
                if self._steps[env_id] > 0:
                    self._episode_ids[env_id] += 1
                self._steps[env_id] = 0

    @torch.no_grad()
    def record_post_step(self):
        env = self._env
        for env_id in self._ids:
            self._steps[env_id] += 1
        done = env.reset_buf[self._ids].detach().cpu().tolist()
        contact_packets = self._footrix_contact_reports.pop(self._ids)
        # Always capture a terminal frame, even when downsampling.
        ids = [i for i, terminal in zip(self._ids, done) if self._steps[i] % self.cfg.record_every == 0 or terminal]
        if not ids:
            return None, None
        if not self._metadata_written:
            self._write_metadata()
        robot = env.scene["robot"]

        def snapshot(tensor):
            return tensor[ids].detach().cpu().clone()

        def column(values, dtype=torch.int64):
            return torch.tensor(values, dtype=dtype).reshape(len(ids), 1)

        imu = {}
        for name in self.IMUS:
            data = env.scene.sensors[name].data
            acceleration = snapshot(data.lin_acc_b)
            angular_velocity = snapshot(data.ang_vel_b)
            imu[name] = {
                "linear_accel_b_m_s2": acceleration,
                "angular_velocity_b_rad_s": angular_velocity,
                "finite": torch.isfinite(acceleration).all(-1) & torch.isfinite(angular_velocity).all(-1),
            }
        contacts = {}
        for name in self.CONTACTS:
            sensor = env.scene.sensors[name]
            data = sensor.data
            force = snapshot(data.net_forces_w)
            finite = torch.isfinite(force).all(-1)
            contacts[name] = {
                "net_normal_force_w_N": force,
                "finite": finite,
                "in_contact": finite & (force.norm(dim=-1) > sensor.cfg.force_threshold),
            }
            if sensor.cfg.track_air_time:
                contacts[name]["current_air_time_s"] = snapshot(data.current_air_time)
                contacts[name]["current_contact_time_s"] = snapshot(data.current_contact_time)

        footrix = self._project_footrix_contacts(env, ids, contact_packets)

        frame = {
            # Native HDF5 exporter uses this top-level key to count samples.
            "actions": snapshot(env.action_manager.action),
            "env_id": column(ids),
            "episode_id": column([self._episode_ids[i] for i in ids]),
            "episode_step": column([self._steps[i] for i in ids]),
            "physics_step": column([env._sim_step_counter] * len(ids)),
            "simulation_time_s": column([env._sim_step_counter * env.physics_dt] * len(ids), torch.float64),
            "robot": {
                "root_pos_w_m": snapshot(robot.data.root_pos_w),
                "root_quat_wxyz": snapshot(robot.data.root_quat_w),
                "root_linear_velocity_w_m_s": snapshot(robot.data.root_lin_vel_w),
                "root_angular_velocity_w_rad_s": snapshot(robot.data.root_ang_vel_w),
                "joint_pos_rad": snapshot(robot.data.joint_pos),
                "joint_vel_rad_s": snapshot(robot.data.joint_vel),
                "joint_pos_target_rad": snapshot(robot.data.joint_pos_target),
            },
            "velocity_command": snapshot(env.command_manager.get_command("base_velocity")),
            "imu": imu,
            "contact": contacts,
            "footrix": footrix,
            "reward": snapshot(env.reward_buf),
            "terminated": snapshot(env.reset_terminated),
            "time_out": snapshot(env.reset_time_outs),
            "reset": snapshot(env.reset_buf),
            "termination_terms": {
                name: snapshot(env.termination_manager.get_term(name))
                for name in env.termination_manager.active_terms
            },
        }
        manager = env.recorder_manager
        for env_id in ids:
            episode = manager.get_episode(env_id)
            episode.env_id = env_id
            episode.seed = env.cfg.seed
        for key, value in frame.items():
            # The native post-step callback assumes all envs when it receives a
            # return value. Supply explicit IDs through its public buffer API.
            manager.add_to_episodes(key, value, env_ids=ids)
        return None, None

    def _write_metadata(self):
        env = self._env
        robot = env.scene["robot"]
        sensors = {}
        for name in self.IMUS + self.CONTACTS:
            sensor = env.scene.sensors[name]
            cfg = sensor.cfg
            description = {"prim_path": cfg.prim_path, "update_period_s": cfg.update_period}
            if name in self.IMUS:
                description.update(
                    offset_pos_m=list(cfg.offset.pos), offset_rotation_wxyz=list(cfg.offset.rot),
                    gravity_bias_m_s2=list(cfg.gravity_bias),
                )
            else:
                description.update(body_names=list(sensor.body_names), force_threshold_N=cfg.force_threshold)
            sensors[name] = description
        urdf = Path(robot.cfg.spawn.asset_path)
        metadata = {
            "schema_version": 1,
            "task": getattr(env.cfg, "env_name", None),
            "training_log_dir": env.cfg.log_dir,
            "physics_dt_s": env.physics_dt,
            "control_dt_s": env.step_dt,
            "sample_every_control_steps": self.cfg.record_every,
            "nominal_record_hz": 1.0 / (env.step_dt * self.cfg.record_every),
            "terminal_frames_always_recorded": True,
            "callback": "record_post_step: after scene update, before reset",
            "num_envs": env.num_envs,
            "recorded_env_ids": self._ids,
            "joint_names": list(robot.joint_names),
            "action_terms": list(env.action_manager.active_terms),
            "action_term_dimensions": list(env.action_manager.action_term_dim),
            "sensors": sensors,
            "footrix": {
                "cell_count": self.FOOTRIX_CELL_COUNT,
                "channels_per_cell": H2_PLUS_FOOTRIX_CHANNELS_PER_CELL,
                "measurement_frame": "ankle_pitch_link",
                "force_frame": "world",
                "layout_path": str(H2_PLUS_FOOTRIX_LAYOUT_PATH),
                "layout_z_m": H2_PLUS_FOOTRIX_LAYOUT_Z_M,
                "layout_z_calibration": "JSON sample positions and effective plane aligned to current foot mesh bottom",
                "layout_status": H2_PLUS_FOOTRIX_LAYOUT_STATUS,
                "cell_radius_m": H2_PLUS_FOOTRIX_CELL_RADIUS_M,
                "cell_thickness_m": H2_PLUS_FOOTRIX_CELL_THICKNESS_M,
                "support_radius_m": H2_PLUS_FOOTRIX_SUPPORT_RADIUS_M,
                "projection_method": H2_PLUS_FOOTRIX_PROJECTION_METHOD,
                "projection_distance_metric": "xy_in_ankle_pitch_link",
                "source": "PhysX contact-report impulses from complete ankle-pitch foot collision",
                "marker_collision_enabled": False,
                "full_foot_collision_preserved": True,
                "coordinate_order": "image-derived left layout and its lateral right-foot mirror; replaceable",
            },
            "urdf": str(urdf),
            "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
            "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "imu_noise_added_by_recorder": False,
            "contact_semantics": "Aggregate contact entries are PhysX net forces in world frame; FOOTRIX entries are contact-report impulses from the complete foot collision distributed to nearby layout cells with normalized planar linear weights, so they preserve total reported impulse but depend on the selected point-distribution model",
            "success_semantics": "Native HDF5 success is not a locomotion quality label; use termination fields",
            "partial_episode": "A final sample with reset=false may be an interrupted or manually reset episode",
        }
        directory = Path(env.cfg.recorders.dataset_export_dir_path)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "metadata.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
        self._metadata_written = True
