"""Run the pretrained G1 walker while passively logging sensor channels.

Example (from the repository root)::

    uv run --no-project --with mujoco --with torch --with numpy \
      python biped_walking/sensorized_experiment/run_sensorized_walk.py \
      --scene rough --speed 0.8 --duration 8 --episodes 2

The policy, gains, terrain XMLs, and dynamics timestep are inherited from the
downloaded candidate project.  This file only adds a side-channel logger.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CANDIDATE_ROOT = PROJECT_ROOT / "external" / "candidates" / "humanoid-sim-eval-learning"
sys.path.insert(0, str(CANDIDATE_ROOT))

from eval_speed_test import (  # noqa: E402  (candidate constants are intentional)
    ACTION_SCALE,
    ANG_VEL_SCALE,
    CMD_SCALE,
    CONTROL_DECIMATION,
    DEFAULT_ANGLES,
    DIRECTION_MAP,
    DOF_POS_SCALE,
    DOF_VEL_SCALE,
    FALL_HEIGHT,
    KDS,
    KPS,
    NUM_ACTIONS,
    NUM_OBS,
    QUAT_MAP,
    SIMULATION_DT,
    get_gravity_orientation,
    pd_control,
)
from config import POLICY_PATH  # noqa: E402

from sensor_suite import (  # noqa: E402
    ARRAY_NX,
    ARRAY_NY,
    FOOT_BODIES,
    read_foot,
)


SCENES = {
    "flat",
    "slope1",
    "slope2",
    "slope3",
    "stairs",
    "perlin",
    "rough",
}


def _field_names() -> list[str]:
    fields = [
        "scene",
        "direction",
        "commanded_speed_m_s",
        "episode",
        "time_s",
        "sim_step",
        "outcome",
        "pelvis_x_m",
        "pelvis_y_m",
        "pelvis_z_m",
        "pelvis_vx_m_s",
        "pelvis_vy_m_s",
        "pelvis_vz_m_s",
        "pelvis_roll_rad",
        "pelvis_pitch_rad",
        "pelvis_yaw_rad",
    ]
    for foot in FOOT_BODIES:
        fields.extend(
            [
                f"{foot}_contacts",
                f"{foot}_outside_contacts",
                f"{foot}_cop_valid",
                f"{foot}_cop_x_m",
                f"{foot}_cop_y_m",
                f"{foot}_cop_z_m",
            ]
        )
        for component in ("fx", "fy", "fz", "tx", "ty", "tz"):
            fields.append(f"{foot}_ankle_{component}")
        for component in ("gx", "gy", "gz"):
            fields.append(f"{foot}_imu_{component}_rad_s")
        for component in ("ax", "ay", "az"):
            fields.append(f"{foot}_imu_{component}_m_s2")
        for component in ("fx", "fy", "fz"):
            fields.append(f"{foot}_outside_{component}")
        for ix in range(ARRAY_NX):
            for iy in range(ARRAY_NY):
                for component in ("fx", "fy", "fz"):
                    fields.append(f"{foot}_cell_{ix}_{iy}_{component}")
    return fields


FIELD_NAMES = _field_names()


def _quat_to_rpy(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw for the MuJoCo wxyz quaternion."""

    qw, qx, qy, qz = [float(v) for v in quaternion_wxyz]
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch_arg = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(pitch_arg)
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _put_vector(row: dict[str, Any], prefix: str, values: np.ndarray, names: tuple[str, ...]) -> None:
    for name, value in zip(names, values):
        row[f"{prefix}_{name}"] = float(value)


def _sensor_row(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: str,
    direction: str,
    speed: float,
    episode: int,
    sim_step: int,
    outcome: str,
) -> dict[str, Any]:
    pelvis_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
    pelvis_rpy = _quat_to_rpy(np.asarray(data.xquat[pelvis_id], dtype=np.float64))
    row: dict[str, Any] = {
        "scene": scene,
        "direction": direction,
        "commanded_speed_m_s": float(speed),
        "episode": int(episode),
        "time_s": float(sim_step * SIMULATION_DT),
        "sim_step": int(sim_step),
        "outcome": outcome,
        "pelvis_x_m": float(data.qpos[0]),
        "pelvis_y_m": float(data.qpos[1]),
        "pelvis_z_m": float(data.qpos[2]),
        "pelvis_vx_m_s": float(data.qvel[0]),
        "pelvis_vy_m_s": float(data.qvel[1]),
        "pelvis_vz_m_s": float(data.qvel[2]),
        "pelvis_roll_rad": float(pelvis_rpy[0]),
        "pelvis_pitch_rad": float(pelvis_rpy[1]),
        "pelvis_yaw_rad": float(pelvis_rpy[2]),
    }

    for foot in FOOT_BODIES:
        reading = read_foot(model, data, foot)
        row[f"{foot}_contacts"] = int(reading.contacts)
        row[f"{foot}_outside_contacts"] = int(reading.outside_contacts)
        row[f"{foot}_cop_valid"] = int(reading.cop_valid)
        row[f"{foot}_cop_x_m"] = float(reading.cop_foot_m[0])
        row[f"{foot}_cop_y_m"] = float(reading.cop_foot_m[1])
        row[f"{foot}_cop_z_m"] = float(reading.cop_foot_m[2])
        _put_vector(
            row,
            f"{foot}_ankle",
            reading.macro_wrench_ankle,
            ("fx", "fy", "fz", "tx", "ty", "tz"),
        )
        _put_vector(
            row,
            f"{foot}_imu",
            reading.imu_gyro_rad_s,
            ("gx_rad_s", "gy_rad_s", "gz_rad_s"),
        )
        _put_vector(
            row,
            f"{foot}_imu",
            reading.imu_accel_m_s2,
            ("ax_m_s2", "ay_m_s2", "az_m_s2"),
        )
        _put_vector(row, f"{foot}_outside", reading.outside_force_foot_n, ("fx", "fy", "fz"))
        for ix in range(ARRAY_NX):
            for iy in range(ARRAY_NY):
                _put_vector(
                    row,
                    f"{foot}_cell_{ix}_{iy}",
                    reading.cell_force_foot_n[ix, iy],
                    ("fx", "fy", "fz"),
                )

    return row


def _load_model(scene: str) -> mujoco.MjModel:
    scene_path = CANDIDATE_ROOT / "scenes" / f"{scene}.xml"
    if not scene_path.exists():
        raise FileNotFoundError(scene_path)
    old_cwd = Path.cwd()
    try:
        # Scene includes are relative to scenes/, so MuJoCo must resolve the
        # include while its process cwd is the scene directory.
        os.chdir(scene_path.parent)
        model = mujoco.MjModel.from_xml_path(scene_path.name)
    finally:
        os.chdir(old_cwd)
    model.opt.timestep = SIMULATION_DT
    return model


def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    policy: torch.jit.ScriptModule,
    scene: str,
    direction: str,
    speed: float,
    episode: int,
    duration_s: float,
) -> tuple[list[dict[str, Any]], str, float]:
    mujoco.mj_resetData(model, data)
    initial_quat = QUAT_MAP[direction]
    if initial_quat is not None:
        data.qpos[3:7] = initial_quat
    mujoco.mj_forward(model, data)

    cmd = DIRECTION_MAP[direction](speed)
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    max_steps = int(round(duration_s / SIMULATION_DT))
    rows: list[dict[str, Any]] = []

    for sim_step in range(1, max_steps + 1):
        tau = pd_control(
            target_dof_pos,
            data.qpos[7:],
            KPS,
            np.zeros_like(KDS),
            data.qvel[6:],
            KDS,
        )
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        if data.qpos[2] < FALL_HEIGHT:
            outcome = "fall"
            survival_time = (sim_step - 1) * SIMULATION_DT
            break

        if sim_step % CONTROL_DECIMATION == 0:
            rows.append(
                _sensor_row(
                    model,
                    data,
                    scene,
                    direction,
                    speed,
                    episode,
                    sim_step,
                    "running",
                )
            )

            qj = data.qpos[7:]
            dqj = data.qvel[6:]
            quat = data.qpos[3:7]
            omega = data.qvel[3:6]
            gravity = get_gravity_orientation(quat)
            counter = sim_step
            t = counter * SIMULATION_DT
            phase = (t % 0.8) / 0.8
            obs[:3] = omega * ANG_VEL_SCALE
            obs[3:6] = gravity
            obs[6:9] = cmd * CMD_SCALE
            obs[9 : 9 + NUM_ACTIONS] = (qj - DEFAULT_ANGLES) * DOF_POS_SCALE
            obs[9 + NUM_ACTIONS : 9 + 2 * NUM_ACTIONS] = dqj * DOF_VEL_SCALE
            obs[9 + 2 * NUM_ACTIONS : 9 + 3 * NUM_ACTIONS] = action
            obs[9 + 3 * NUM_ACTIONS : 9 + 3 * NUM_ACTIONS + 2] = [
                np.sin(2 * np.pi * phase),
                np.cos(2 * np.pi * phase),
            ]
            with torch.inference_mode():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().cpu().numpy().squeeze()
            target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES
    else:
        outcome = "timeout"
        survival_time = duration_s

    for row in rows:
        row["outcome"] = outcome
    return rows, outcome, float(survival_time)


def run(
    scene: str,
    direction: str,
    speed: float,
    duration_s: float,
    episodes: int,
    output_csv: Path,
) -> dict[str, Any]:
    model = _load_model(scene)
    data = mujoco.MjData(model)
    policy = torch.jit.load(str(POLICY_PATH), map_location="cpu")
    policy.eval()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summary_episodes: list[dict[str, Any]] = []
    for episode in range(episodes):
        rows, outcome, survival_time = run_episode(
            model,
            data,
            policy,
            scene,
            direction,
            speed,
            episode,
            duration_s,
        )
        all_rows.extend(rows)
        if rows:
            distance_x = float(rows[-1]["pelvis_x_m"] - rows[0]["pelvis_x_m"])
            distance_y = float(rows[-1]["pelvis_y_m"] - rows[0]["pelvis_y_m"])
            final_pelvis_z = float(rows[-1]["pelvis_z_m"])
        else:
            distance_x = 0.0
            distance_y = 0.0
            final_pelvis_z = 0.0
        summary_episodes.append(
            {
                "episode": episode,
                "outcome": outcome,
                "survival_time_s": survival_time,
                "logged_rows": len(rows),
                "pelvis_delta_x_m": distance_x,
                "pelvis_delta_y_m": distance_y,
                "final_pelvis_z_m": final_pelvis_z,
            }
        )
        print(
            f"scene={scene} direction={direction} speed={speed:.2f} "
            f"episode={episode} outcome={outcome} time={survival_time:.3f}s rows={len(rows)}"
        )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_NAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "scene": scene,
        "direction": direction,
        "speed_m_s": speed,
        "duration_s": duration_s,
        "episodes": summary_episodes,
        "rows": len(all_rows),
        "dt_s": SIMULATION_DT,
        "logging_hz": 1.0 / (SIMULATION_DT * CONTROL_DECIMATION),
        "array_shape": [ARRAY_NX, ARRAY_NY, 3],
        "array_cells_per_foot": ARRAY_NX * ARRAY_NY,
        "array_force_channels_both_feet": 2 * ARRAY_NX * ARRAY_NY * 3,
        "policy_path": str(POLICY_PATH),
    }
    summary_path = output_csv.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output_csv}")
    print(f"wrote {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(SCENES), default="rough")
    parser.add_argument("--direction", choices=sorted(DIRECTION_MAP), default="f")
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "outputs" / "sensor_log.csv",
        help="CSV output path; a same-stem JSON summary is written beside it",
    )
    args = parser.parse_args()
    if args.speed <= 0 or args.duration <= 0 or args.episodes <= 0:
        parser.error("speed, duration, and episodes must be positive")
    run(args.scene, args.direction, args.speed, args.duration, args.episodes, args.output)


if __name__ == "__main__":
    main()
