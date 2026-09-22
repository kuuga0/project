"""Interactively observe a theoretical wrench boundary sweep in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from foot_contact_wrench.analysis.comparison import support_wrench_samples
from foot_contact_wrench.mujoco_io.bench import (
    apply_world_wrench_at_com,
    contact_wrench_about_body,
    initialize_free_foot,
    load_bench_model,
)
from foot_contact_wrench.utils.config import load_yaml, local_set_from_config
from foot_contact_wrench.visualization.wrench_scene import (
    ACTUAL_FORCE,
    ACTUAL_MOMENT,
    MODEL_FORCE,
    MODEL_MOMENT,
    WrenchSceneOptions,
    add_wrench,
    add_wrench_cone,
)


ROOT = Path(__file__).resolve().parents[2]
SWEEP_CONFIG_PATH = ROOT / "configs" / "bench" / "stage_b_wrench_sweep.yml"

APPLIED_FORCE = np.array([0.95, 0.78, 0.08, 1.0], dtype=np.float32)
APPLIED_MOMENT = np.array([0.78, 0.28, 0.88, 1.0], dtype=np.float32)
OVER_LIMIT = np.array([0.95, 0.08, 0.58, 1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default=None,
        help="Starting case name from stage_b_wrench_sweep.yml.",
    )
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=1.5,
        help="Seconds to hold the maximum load before advancing.",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="Simulation time divided by wall time.",
    )
    return parser.parse_args()


def working_wrench(local_set) -> np.ndarray:
    """Construct the fixed-normal-load equilibrium wrench used by the sweep."""
    active = np.flatnonzero(local_set.model.active)
    normal_forces = np.zeros(local_set.model.point_count)
    normal_forces[active] = local_set.normal_load / len(active)
    return local_set.model.wrench(
        normal_forces, np.zeros((local_set.model.point_count, 2))
    )


def failure_measure(data, body_id: int, mode: str) -> float:
    if mode == "slip":
        return float(np.linalg.norm(data.cvel[body_id, 3:5]))
    if mode == "yaw_slip":
        return float(abs(data.cvel[body_id, 2]))
    if mode == "tip":
        rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        cosine = np.clip(rotation[:, 2] @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0)
        return float(np.arccos(cosine))
    raise ValueError(f"unknown failure mode: {mode}")


def threshold_for(sweep_config, mode: str) -> float:
    thresholds = sweep_config["failure_thresholds"]
    return {
        "slip": float(thresholds["slip_speed_mps"]),
        "yaw_slip": float(thresholds["yaw_rate_radps"]),
        "tip": float(thresholds["tip_angle_rad"]),
    }[mode]


class DynamicSweep:
    """Own the current loading case and its continuously advancing state."""

    def __init__(
        self,
        base_config,
        sweep_config,
        model,
        data,
        local_set,
        start_case: str | None,
        hold_duration: float,
    ) -> None:
        self.base_config = base_config
        self.sweep_config = sweep_config
        self.model = model
        self.data = data
        self.local_set = local_set
        self.cases = list(sweep_config["cases"])
        self.body_id = model.body("foot").id
        self.hold_duration = max(0.0, float(hold_duration))
        self.settle_duration = float(sweep_config["settle_duration_s"])
        self.ramp_duration = float(sweep_config["ramp_duration_s"])
        self.ramp_ratio = float(sweep_config["ramp_to_theory_ratio"])
        self.persistence_steps = max(
            1,
            int(
                float(sweep_config["failure_persistence_s"])
                / self.model.opt.timestep
            ),
        )
        self.wrench_zero = working_wrench(local_set)
        self.theory_limits = [self._theory_limit(case) for case in self.cases]
        names = [case["name"] for case in self.cases]
        if start_case is not None and start_case not in names:
            raise ValueError(
                f"unknown case {start_case!r}; choose one of: {', '.join(names)}"
            )
        self.case_index = names.index(start_case) if start_case is not None else 0
        self.auto_advance = True
        self.paused = False
        self.single_step = False
        self.show_actual = True
        self.show_model = True
        self.show_range = True
        self.show_applied = True
        self.reset_case()

    @property
    def case(self):
        return self.cases[self.case_index]

    @property
    def theory_limit(self) -> float:
        return self.theory_limits[self.case_index]

    @property
    def external_direction(self) -> np.ndarray:
        direction = np.asarray(
            self.case["external_wrench_direction"], dtype=float
        ).reshape(6)
        return direction / np.linalg.norm(direction)

    @property
    def threshold(self) -> float:
        return threshold_for(self.sweep_config, self.case["failure_mode"])

    def _theory_limit(self, case) -> float:
        external_direction = np.asarray(
            case["external_wrench_direction"], dtype=float
        ).reshape(6)
        external_direction /= np.linalg.norm(external_direction)
        result = self.local_set.radial_capacity(
            self.wrench_zero, -external_direction
        )
        if not result.feasible or result.objective is None:
            raise RuntimeError(f"theory boundary query failed for {case['name']}")
        return float(result.objective)

    def reset_case(self) -> None:
        initialize_free_foot(
            self.model,
            self.data,
            self.base_config["initial_position_m"],
            self.base_config["initial_linear_velocity_mps"],
            self.base_config["initial_angular_velocity_radps"],
        )
        self.consecutive_failure = 0
        self.failure_detected = False
        self.failure_load: float | None = None
        self.last_measure = 0.0
        self.applied_load = 0.0
        self.external_wrench = np.zeros(6)

    def change_case(self, offset: int) -> None:
        self.case_index = (self.case_index + offset) % len(self.cases)
        self.reset_case()
        print(
            f"当前工况：{self.case_index + 1}/{len(self.cases)} "
            f"{self.case['label_zh']}，理论边界 "
            f"{self.theory_limit:.5g} {self.case['unit']}"
        )

    def phase(self) -> tuple[str, float]:
        if self.data.time < self.settle_duration:
            return "SETTLE", 0.0
        ramp_time = self.data.time - self.settle_duration
        if ramp_time < self.ramp_duration:
            return "RAMP", ramp_time / self.ramp_duration
        if ramp_time < self.ramp_duration + self.hold_duration:
            return "HOLD", 1.0
        return "COMPLETE", 1.0

    def step(self) -> None:
        phase, progress = self.phase()
        if phase == "COMPLETE":
            if self.auto_advance:
                self.change_case(1)
                phase, progress = self.phase()
            else:
                self.paused = True
                return

        if phase == "SETTLE":
            self.applied_load = 0.0
        else:
            self.applied_load = self.ramp_ratio * self.theory_limit * progress
        self.external_wrench = self.applied_load * self.external_direction
        apply_world_wrench_at_com(
            self.data, self.body_id, self.external_wrench
        )
        mujoco.mj_step(self.model, self.data)

        self.last_measure = failure_measure(
            self.data, self.body_id, self.case["failure_mode"]
        )
        if (
            self.last_measure >= self.threshold
            and self.applied_load >= 0.1 * self.theory_limit
        ):
            self.consecutive_failure += 1
        else:
            self.consecutive_failure = 0
        if (
            not self.failure_detected
            and self.consecutive_failure >= self.persistence_steps
        ):
            self.failure_detected = True
            self.failure_load = self.applied_load

    def utilization(self) -> float:
        return self.applied_load / self.theory_limit


def main() -> None:
    args = parse_args()
    sweep_config = load_yaml(SWEEP_CONFIG_PATH)
    base_config = load_yaml(ROOT / sweep_config["base_config"])
    model, data = load_bench_model(str(ROOT / base_config["model_xml"]))
    model.opt.timestep = float(base_config["timestep_s"])
    model.opt.gravity[:] = np.asarray(base_config["gravity_mps2"], dtype=float)
    local_set = local_set_from_config(base_config)
    viewer_config = base_config.get("viewer", {})
    options = WrenchSceneOptions(
        force_scale_m_per_n=float(viewer_config.get("force_scale_m_per_n", 0.003)),
        moment_scale_m_per_nm=float(
            viewer_config.get("moment_scale_m_per_nm", 0.04)
        ),
    )
    direction_count = int(viewer_config.get("cone_direction_count", 32))
    wrench_samples_foot = support_wrench_samples(local_set, direction_count)
    experiment = DynamicSweep(
        base_config,
        sweep_config,
        model,
        data,
        local_set,
        args.case,
        args.hold_duration,
    )

    def key_callback(keycode: int) -> None:
        if keycode == ord("1"):
            experiment.show_actual = not experiment.show_actual
        elif keycode == ord("2"):
            experiment.show_model = not experiment.show_model
        elif keycode == ord("3"):
            experiment.show_range = not experiment.show_range
        elif keycode == ord("4"):
            experiment.show_applied = not experiment.show_applied
        elif keycode == 32:
            experiment.paused = not experiment.paused
        elif keycode in (ord("S"), ord("s")):
            experiment.paused = True
            experiment.single_step = True
        elif keycode in (ord("R"), ord("r")):
            experiment.reset_case()
        elif keycode in (ord("N"), ord("n")):
            experiment.change_case(1)
        elif keycode in (ord("P"), ord("p")):
            experiment.change_case(-1)
        elif keycode in (ord("A"), ord("a")):
            experiment.auto_advance = not experiment.auto_advance

    print("动态 MuJoCo 扳手边界观察器")
    print("加载过程：沉降 -> 线性加载至理论边界的 1.30 倍 -> 保持 -> 下一工况")
    print("按键：1 真值，2 法向阵列模型，3 局部范围，4 外部加载")
    print("      Space 暂停/继续，S 单步，R 重置，N/P 下一/上一工况，A 自动轮播")
    experiment.change_case(0)

    realtime_factor = max(float(args.realtime_factor), 1e-6)
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.02]
        viewer.cam.distance = 0.48
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0

        while viewer.is_running():
            loop_start = time.perf_counter()
            should_step = not experiment.paused or experiment.single_step
            if should_step:
                experiment.step()
                experiment.single_step = False

            sample = contact_wrench_about_body(model, data, "foot")
            rotation_world_from_foot = np.asarray(
                data.xmat[experiment.body_id], dtype=float
            ).reshape(3, 3)
            utilization = experiment.utilization()
            phase, _ = experiment.phase()
            status = "FAILED" if experiment.failure_detected else "tracking"
            failure_load = (
                "--"
                if experiment.failure_load is None
                else f"{experiment.failure_load:.4g} {experiment.case['unit']}"
            )

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                if experiment.show_actual:
                    add_wrench(
                        viewer.user_scn,
                        sample.com_position_world,
                        sample.wrench_world,
                        options,
                        ACTUAL_FORCE,
                        ACTUAL_MOMENT,
                        "actual contact",
                    )
                if experiment.show_model:
                    add_wrench(
                        viewer.user_scn,
                        sample.com_position_world,
                        sample.pressure_model_wrench_world,
                        options,
                        MODEL_FORCE,
                        MODEL_MOMENT,
                        "normal-array model",
                    )
                if experiment.show_range:
                    add_wrench_cone(
                        viewer.user_scn,
                        sample.com_position_world,
                        rotation_world_from_foot,
                        wrench_samples_foot,
                        options,
                    )
                if experiment.show_applied:
                    load_color_force = OVER_LIMIT if utilization > 1.0 else APPLIED_FORCE
                    load_color_moment = OVER_LIMIT if utilization > 1.0 else APPLIED_MOMENT
                    add_wrench(
                        viewer.user_scn,
                        sample.com_position_world,
                        experiment.external_wrench,
                        options,
                        load_color_force,
                        load_color_moment,
                        "applied external",
                    )
                # Follow the foot after it starts to slide or tip.
                viewer.cam.lookat[:] = sample.com_position_world

            left_text = (
                "DYNAMIC WRENCH SWEEP\n"
                f"Case {experiment.case_index + 1}/{len(experiment.cases)}: "
                f"{experiment.case['name']}\n"
                f"Phase: {phase}{'  [PAUSED]' if experiment.paused else ''}\n"
                f"Load: {experiment.applied_load:.4g} {experiment.case['unit']}\n"
                f"Theory: {experiment.theory_limit:.4g} {experiment.case['unit']}\n"
                f"Utilization: {100.0 * utilization:.1f}%\n"
                f"Failure: {status}  first={failure_load}\n"
                f"Measure: {experiment.last_measure:.4g} / {experiment.threshold:.4g}\n"
                f"Active contacts: {sample.contact_count}/50"
            )
            right_text = (
                "DISPLAY\n"
                f"1 Actual contact: {'ON' if experiment.show_actual else 'OFF'}\n"
                f"2 Normal model: {'ON' if experiment.show_model else 'OFF'}\n"
                f"3 Local envelope: {'ON' if experiment.show_range else 'OFF'}\n"
                f"4 External load: {'ON' if experiment.show_applied else 'OFF'}\n"
                "\nCONTROL\n"
                "Space Pause   S Step\n"
                "R Reset       N/P Case\n"
                f"A Auto cycle: {'ON' if experiment.auto_advance else 'OFF'}"
            )
            viewer.set_texts(
                [
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        left_text,
                        "",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                        right_text,
                        "",
                    ),
                ]
            )
            viewer.sync()

            target = model.opt.timestep / realtime_factor
            delay = target - (time.perf_counter() - loop_start)
            if delay > 0.0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
