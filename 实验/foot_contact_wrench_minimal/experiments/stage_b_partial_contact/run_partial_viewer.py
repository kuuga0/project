"""在 MuJoCo 中观察不完整连续支撑下的准静态加载。"""

from __future__ import annotations

import argparse
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
from foot_contact_wrench.mujoco_io.partial_contact import (
    build_partial_contact_xml,
    loading_state,
    observed_normal_wrench_set,
    set_support_scenario,
    tilt_angle_rad,
    vertical_point_load_wrench_world,
)
from foot_contact_wrench.utils.config import load_yaml, local_set_from_config
from foot_contact_wrench.visualization.wrench_scene import (
    ACTUAL_FORCE,
    ACTUAL_MOMENT,
    MODEL_FORCE,
    MODEL_MOMENT,
    RANGE_FORCE,
    RANGE_MOMENT,
    WrenchSceneOptions,
    add_wrench,
    add_wrench_cone,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "bench" / "stage_b_partial_contact.yml"
OUTPUT_DIR = ROOT / "outputs" / "stage_b_partial_contact"
EXTERNAL_FORCE = np.array([0.95, 0.78, 0.08, 1.0], dtype=np.float32)
EXTERNAL_MOMENT = np.array([0.78, 0.28, 0.88, 1.0], dtype=np.float32)
FUSED_FORCE = np.array([0.42, 0.20, 0.90, 1.0], dtype=np.float32)
FUSED_MOMENT = np.array([0.55, 0.25, 0.95, 1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=None, help="起始工况名称")
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    return parser.parse_args()


def reduced_boundary_samples(local_set, count: int = 32) -> np.ndarray:
    """Sample the pressure-observable roll/pitch boundary at fixed Fz."""
    samples = []
    for angle in np.linspace(0.0, 2.0 * np.pi, count, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0, 0.0, 0.0])
        result = local_set.support(direction)
        if result.feasible and result.wrench is not None:
            samples.append(result.wrench)
    return np.asarray(samples, dtype=float).reshape(-1, 6)


def main() -> None:
    args = parse_args()
    config = load_yaml(CONFIG_PATH)
    base_config = load_yaml(ROOT / config["base_config"])
    points_foot = np.asarray(local_set_from_config(base_config).model.points, dtype=float)
    model_xml = OUTPUT_DIR / "generated_partial_contact_bench.xml"
    build_partial_contact_xml(
        ROOT / base_config["model_xml"],
        model_xml,
        config["scenarios"],
        support_geom_prefix=config["support_geom_prefix"],
        support_height_m=float(config["support_height_m"]),
        support_friction=config["support_friction"],
    )
    model, data = load_bench_model(str(model_xml))
    model.opt.timestep = float(base_config["timestep_s"])
    model.opt.gravity[:] = np.asarray(base_config["gravity_mps2"], dtype=float)
    body_id = model.body("foot").id
    scenarios = list(config["scenarios"])
    names = [scenario["name"] for scenario in scenarios]
    if args.scenario is not None and args.scenario not in names:
        raise ValueError(f"未知工况 {args.scenario}，可选：{', '.join(names)}")
    scenario_index = names.index(args.scenario) if args.scenario else 0
    options = WrenchSceneOptions(
        force_scale_m_per_n=float(base_config["viewer"]["force_scale_m_per_n"]),
        moment_scale_m_per_nm=float(base_config["viewer"]["moment_scale_m_per_nm"]),
    )
    state = {
        "actual": True,
        "fused": True,
        "pressure": True,
        "range": True,
        "external": True,
        "paused": False,
        "reset": False,
        "next": False,
    }
    gyro_previous = np.zeros(3)
    boundary_samples = np.empty((0, 6))
    reference_set = None
    previous_phase = "settle"

    def reset() -> None:
        nonlocal gyro_previous, boundary_samples, reference_set, previous_phase
        initialize_free_foot(
            model,
            data,
            base_config["initial_position_m"],
            base_config["initial_linear_velocity_mps"],
            base_config["initial_angular_velocity_radps"],
        )
        set_support_scenario(
            model, data, scenarios[scenario_index]["name"], config["support_geom_prefix"]
        )
        gyro_previous = np.zeros(3)
        boundary_samples = np.empty((0, 6))
        reference_set = None
        previous_phase = "settle"

    def key_callback(keycode: int) -> None:
        toggles = {
            ord("1"): "actual",
            ord("2"): "fused",
            ord("3"): "pressure",
            ord("4"): "range",
            ord("5"): "external",
        }
        if keycode in toggles:
            state[toggles[keycode]] = not state[toggles[keycode]]
        elif keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (ord("R"), ord("r")):
            state["reset"] = True
        elif keycode in (ord("N"), ord("n")):
            state["next"] = True

    reset()
    print("单足不完整连续支撑准静态观察器")
    print("1 实际扳手，2 融合扳手，3 压力扳手，4 参考局部集合，5 外部加载")
    print("Space 暂停/继续，R 重置，N 下一工况")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=True, show_right_ui=True
    ) as viewer:
        viewer.cam.distance = 0.48
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        while viewer.is_running():
            loop_start = time.perf_counter()
            if state["next"]:
                scenario_index = (scenario_index + 1) % len(scenarios)
                state["next"] = False
                reset()
            if state["reset"]:
                state["reset"] = False
                reset()

            scenario = scenarios[scenario_index]
            load = loading_state(float(data.time), config, scenario)
            external = vertical_point_load_wrench_world(
                data, body_id, load.application_point_foot_m, load.normal_force_n
            )
            if not state["paused"] and not load.complete:
                apply_world_wrench_at_com(data, body_id, external)
                mujoco.mj_step(model, data)

            sample = contact_wrench_about_body(
                model,
                data,
                "foot",
                support_geom_prefix=config["support_geom_prefix"],
                normal_world=config["support_normal_world"],
            )
            estimate = sensor_wrench_estimate(
                model,
                data,
                body_id,
                sample.normal_forces,
                points_foot,
                float(base_config["contact_grid"]["z_m"]),
                gyro_previous,
                external,
            )
            if not state["paused"]:
                gyro_previous = estimate.gyro_foot_radps.copy()

            if load.phase == "reference":
                candidate_set = observed_normal_wrench_set(
                    base_config,
                    points_foot,
                    sample.normal_forces,
                    base_config["normal"],
                    float(config["pressure_active_threshold_n"]),
                )
                if candidate_set is not None:
                    reference_set = candidate_set
            if (
                previous_phase == "reference"
                and load.phase == "sweep"
                and reference_set is not None
            ):
                boundary_samples = reduced_boundary_samples(reference_set)
            previous_phase = load.phase

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                if state["actual"]:
                    add_wrench(
                        viewer.user_scn,
                        sample.com_position_world,
                        sample.wrench_world,
                        options,
                        ACTUAL_FORCE,
                        ACTUAL_MOMENT,
                        "actual contact",
                    )
                if state["fused"]:
                    add_wrench(
                        viewer.user_scn,
                        sample.com_position_world,
                        estimate.fused_wrench_world,
                        options,
                        FUSED_FORCE,
                        FUSED_MOMENT,
                        "IMU-pressure model",
                    )
                if state["pressure"]:
                    add_wrench(
                        viewer.user_scn,
                        sample.com_position_world,
                        sample.pressure_model_wrench_world,
                        options,
                        MODEL_FORCE,
                        MODEL_MOMENT,
                        "pressure model",
                    )
                if state["range"] and len(boundary_samples):
                    rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
                    add_wrench_cone(
                        viewer.user_scn,
                        sample.com_position_world,
                        rotation,
                        boundary_samples,
                        options,
                        RANGE_FORCE,
                        RANGE_MOMENT,
                        "reference local set",
                    )
                if state["external"]:
                    add_wrench(
                        viewer.user_scn,
                        sample.com_position_world,
                        external,
                        options,
                        EXTERNAL_FORCE,
                        EXTERNAL_MOMENT,
                        "external load",
                    )
                viewer.cam.lookat[:] = sample.com_position_world

            viewer.set_texts(
                [
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        (
                            f"{scenario['label_zh']}\n阶段 {load.phase}  进度 {load.path_progress:.3f}\n"
                            f"承载点 {sample.contact_count}/50  倾角 {np.degrees(tilt_angle_rad(data, body_id)):.3f} deg"
                        ),
                        "",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                        "1 实际  2 融合  3 压力\n4 局部集合  5 外部加载\nSpace 暂停  R 重置  N 下一工况",
                        "",
                    ),
                ]
            )
            viewer.sync()
            delay = model.opt.timestep / max(float(args.realtime_factor), 1e-6) - (
                time.perf_counter() - loop_start
            )
            if delay > 0.0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
