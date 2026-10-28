"""Record sensors from the live training scene through Isaac Lab RecorderManager.

Post-step runs after scene updates and before automatic resets. Sampling is at
control-step resolution; the physics-substep callback in Lab 2.3.2 is too early
for reading fresh scene sensor buffers. Native episode buffering/export remains
owned by RecorderManager, including reset boundaries and HDF5 serialization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from isaaclab.managers.recorder_manager import RecorderTerm

__all__ = ["H2PlusSensorRecorder"]


class H2PlusSensorRecorder(RecorderTerm):
    """Capture selected environments without constructing another simulation."""

    IMUS = ("torso_imu", "left_foot_imu", "right_foot_imu")
    CONTACTS = ("left_foot_contact", "right_foot_contact", "body_contact")

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
            "urdf": str(urdf),
            "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
            "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "imu_noise_added_by_recorder": False,
            "contact_semantics": "PhysX aggregate normal contact force in world frame; not friction or FOOTRIX cells",
            "success_semantics": "Native HDF5 success is not a locomotion quality label; use termination fields",
            "partial_episode": "A final sample with reset=false may be an interrupted or manually reset episode",
        }
        directory = Path(env.cfg.recorders.dataset_export_dir_path)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "metadata.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
        self._metadata_written = True
