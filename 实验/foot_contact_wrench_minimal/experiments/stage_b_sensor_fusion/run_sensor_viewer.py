"""Interactively compare ideal IMU/pressure estimates with MuJoCo contact truth."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from foot_contact_wrench.estimation.sensor_wrench import sensor_wrench_estimate
from foot_contact_wrench.mujoco_io.bench import (
    apply_world_wrench_at_com,
    contact_wrench_about_body,
    initialize_free_foot,
    load_bench_model,
)
from foot_contact_wrench.mujoco_io.sensor_bench import sinusoidal_excitation_wrench
from foot_contact_wrench.utils.config import load_yaml, local_set_from_config
from foot_contact_wrench.visualization.wrench_scene import (
    ACTUAL_FORCE,
    ACTUAL_MOMENT,
    MODEL_FORCE,
    MODEL_MOMENT,
    WrenchSceneOptions,
    add_wrench,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "bench" / "stage_b_sensor_bench.yml"

IMU_FORCE = np.array([0.95, 0.45, 0.10, 1.0], dtype=np.float32)
IMU_MOMENT = np.array([0.88, 0.20, 0.68, 1.0], dtype=np.float32)
FUSED_FORCE = np.array([0.42, 0.20, 0.90, 1.0], dtype=np.float32)
FUSED_MOMENT = np.array([0.55, 0.25, 0.95, 1.0], dtype=np.float32)
EXCITATION_FORCE = np.array([0.95, 0.78, 0.08, 1.0], dtype=np.float32)
EXCITATION_MOMENT = np.array([0.78, 0.28, 0.88, 1.0], dtype=np.float32)


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    base_config = load_yaml(ROOT / config["base_config"])
    model, data = load_bench_model(str(ROOT / base_config["model_xml"]))
    model.opt.timestep = float(base_config["timestep_s"])
    model.opt.gravity[:] = np.asarray(base_config["gravity_mps2"], dtype=float)
    body_id = model.body("foot").id
    points_foot = np.asarray(local_set_from_config(base_config).model.points, dtype=float)
    contact_plane_z = float(base_config["contact_grid"]["z_m"])
    warmup_duration = float(config["warmup_duration_s"])
    duration = float(config["duration_s"])
    viewer_config = base_config.get("viewer", {})
    options = WrenchSceneOptions(
        force_scale_m_per_n=float(viewer_config.get("force_scale_m_per_n", 0.003)),
        moment_scale_m_per_nm=float(viewer_config.get("moment_scale_m_per_nm", 0.04)),
    )
    state = {
        "actual": True,
        "imu": True,
        "pressure": True,
        "fused": True,
        "excitation": True,
        "paused": False,
        "reset": False,
    }
    gyro_previous = np.zeros(3)

    def reset() -> None:
        nonlocal gyro_previous
        initialize_free_foot(
            model,
            data,
            base_config["initial_position_m"],
            base_config["initial_linear_velocity_mps"],
            base_config["initial_angular_velocity_radps"],
        )
        gyro_previous = np.zeros(3)

    def key_callback(keycode: int) -> None:
        if keycode == ord("1"):
            state["actual"] = not state["actual"]
        elif keycode == ord("2"):
            state["imu"] = not state["imu"]
        elif keycode == ord("3"):
            state["pressure"] = not state["pressure"]
        elif keycode == ord("4"):
            state["fused"] = not state["fused"]
        elif keycode == ord("5"):
            state["excitation"] = not state["excitation"]
        elif keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (ord("R"), ord("r")):
            state["reset"] = True

    reset()
    print("IMU + 足底阵列动态观察器")
    print("按键：1 真实接触，2 IMU，3 阵列，4 融合，5 已知测试激励")
    print("      Space 暂停/继续，R 重置；每轮 6 秒后自动重放")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=True, show_right_ui=True
    ) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.02]
        viewer.cam.distance = 0.48
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        while viewer.is_running():
            loop_start = time.perf_counter()
            if state["reset"] or data.time >= duration:
                reset()
                state["reset"] = False
            if not state["paused"]:
                external = sinusoidal_excitation_wrench(
                    config, max(0.0, data.time - warmup_duration)
                )
                if data.time < warmup_duration:
                    external[:] = 0.0
                apply_world_wrench_at_com(data, body_id, external)
                mujoco.mj_step(model, data)
            else:
                external = sinusoidal_excitation_wrench(
                    config, max(0.0, data.time - warmup_duration)
                )
                if data.time < warmup_duration:
                    external[:] = 0.0

            sample = contact_wrench_about_body(model, data, "foot")
            estimate = sensor_wrench_estimate(
                model,
                data,
                body_id,
                sample.normal_forces,
                points_foot,
                contact_plane_z,
                gyro_previous,
                external,
            )
            if not state["paused"]:
                gyro_previous = estimate.gyro_foot_radps.copy()
            force_error = np.linalg.norm(
                estimate.fused_wrench_world[3:] - sample.wrench_world[3:]
            )
            moment_error = np.linalg.norm(
                estimate.fused_wrench_world[:3] - sample.wrench_world[:3]
            )

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                if state["actual"]:
                    add_wrench(
                        viewer.user_scn, sample.com_position_world, sample.wrench_world,
                        options, ACTUAL_FORCE, ACTUAL_MOMENT, "actual contact",
                    )
                if state["imu"]:
                    add_wrench(
                        viewer.user_scn, sample.com_position_world, estimate.imu_wrench_world,
                        options, IMU_FORCE, IMU_MOMENT, "imu dynamics",
                    )
                if state["pressure"]:
                    add_wrench(
                        viewer.user_scn, sample.com_position_world, estimate.pressure_wrench_world,
                        options, MODEL_FORCE, MODEL_MOMENT, "pressure array",
                    )
                if state["fused"]:
                    add_wrench(
                        viewer.user_scn, sample.com_position_world, estimate.fused_wrench_world,
                        options, FUSED_FORCE, FUSED_MOMENT, "imu pressure fusion",
                    )
                if state["excitation"]:
                    add_wrench(
                        viewer.user_scn, sample.com_position_world, external,
                        options, EXCITATION_FORCE, EXCITATION_MOMENT, "known excitation",
                    )
                viewer.cam.lookat[:] = sample.com_position_world

            left_text = (
                "IDEAL SENSOR COMPARISON\n"
                f"Time: {data.time:.3f} s / {duration:.1f} s\n"
                f"Phase: {'SETTLE' if data.time < warmup_duration else 'EXCITATION'}"
                f"{'  [PAUSED]' if state['paused'] else ''}\n"
                f"Active contacts: {sample.contact_count}/50\n"
                f"Fused force error: {force_error:.3e} N\n"
                f"Fused moment error: {moment_error:.3e} N m\n"
                "Known excitation is subtracted in IMU estimate"
            )
            right_text = (
                "DISPLAY\n"
                f"1 Actual: {'ON' if state['actual'] else 'OFF'}\n"
                f"2 IMU: {'ON' if state['imu'] else 'OFF'}\n"
                f"3 Pressure: {'ON' if state['pressure'] else 'OFF'}\n"
                f"4 Fusion: {'ON' if state['fused'] else 'OFF'}\n"
                f"5 Excitation: {'ON' if state['excitation'] else 'OFF'}\n"
                "\nSpace Pause\nR Reset"
            )
            viewer.set_texts(
                [
                    (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, left_text, ""),
                    (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT, right_text, ""),
                ]
            )
            viewer.sync()
            delay = model.opt.timestep - (time.perf_counter() - loop_start)
            if delay > 0.0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
