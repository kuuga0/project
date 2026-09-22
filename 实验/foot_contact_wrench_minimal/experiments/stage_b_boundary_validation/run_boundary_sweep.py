"""Run quasi-static external-wrench sweeps against theoretical local boundaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from foot_contact_wrench.mujoco_io.bench import (
    apply_world_wrench_at_com,
    contact_wrench_about_body,
    initialize_free_foot,
    load_bench_model,
)
from foot_contact_wrench.utils.config import load_yaml, local_set_from_config


ROOT = Path(__file__).resolve().parents[2]
SWEEP_CONFIG_PATH = ROOT / "configs" / "bench" / "stage_b_wrench_sweep.yml"
OUTPUT_DIR = ROOT / "outputs" / "stage_b_validation"


def working_wrench(local_set) -> np.ndarray:
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


def run_case(base_config, sweep_config, case) -> tuple[dict, list[dict]]:
    model, data = load_bench_model(str(ROOT / base_config["model_xml"]))
    model.opt.timestep = float(base_config["timestep_s"])
    model.opt.gravity[:] = np.asarray(base_config["gravity_mps2"], dtype=float)
    initialize_free_foot(
        model,
        data,
        base_config["initial_position_m"],
        base_config["initial_linear_velocity_mps"],
        base_config["initial_angular_velocity_radps"],
    )
    body_id = model.body("foot").id

    settle_steps = int(float(sweep_config["settle_duration_s"]) / model.opt.timestep)
    for _ in range(settle_steps):
        data.xfrc_applied[body_id, :] = 0.0
        mujoco.mj_step(model, data)

    local_set = local_set_from_config(base_config)
    wrench_zero = working_wrench(local_set)
    external_direction = np.asarray(
        case["external_wrench_direction"], dtype=float
    ).reshape(6)
    external_direction /= np.linalg.norm(external_direction)
    contact_direction = -external_direction
    theory = local_set.radial_capacity(wrench_zero, contact_direction)
    if not theory.feasible or theory.objective is None or theory.wrench is None:
        raise RuntimeError(f"theory boundary query failed for {case['name']}")
    theory_limit = float(theory.objective)
    ramp_duration = float(sweep_config["ramp_duration_s"])
    ramp_limit = float(sweep_config["ramp_to_theory_ratio"]) * theory_limit
    ramp_steps = int(ramp_duration / model.opt.timestep)
    persistence_steps = max(
        1,
        int(float(sweep_config["failure_persistence_s"]) / model.opt.timestep),
    )
    threshold = threshold_for(sweep_config, case["failure_mode"])
    consecutive_failure = 0
    failure_load: float | None = None
    failure_time: float | None = None
    records: list[dict] = []

    for step in range(ramp_steps + 1):
        ramp_time = step * model.opt.timestep
        applied_load = ramp_limit * min(ramp_time / ramp_duration, 1.0)
        external_wrench = applied_load * external_direction
        apply_world_wrench_at_com(data, body_id, external_wrench)
        mujoco.mj_step(model, data)
        measure = failure_measure(data, body_id, case["failure_mode"])
        if measure >= threshold and applied_load >= 0.1 * theory_limit:
            consecutive_failure += 1
        else:
            consecutive_failure = 0
        sample = contact_wrench_about_body(model, data, "foot")
        if step % 5 == 0 or consecutive_failure >= persistence_steps:
            records.append(
                {
                    "time_s": ramp_time,
                    "applied_load": applied_load,
                    "theory_limit": theory_limit,
                    "utilization": applied_load / theory_limit,
                    "failure_measure": measure,
                    "failure_threshold": threshold,
                    "contact_count": sample.contact_count,
                    "contact_wrench_foot": sample.wrench_foot.tolist(),
                    "com_position_world_m": sample.com_position_world.tolist(),
                }
            )
        if consecutive_failure >= persistence_steps:
            failure_load = applied_load
            failure_time = ramp_time
            break

    relative_error = (
        (failure_load - theory_limit) / theory_limit
        if failure_load is not None
        else None
    )
    summary = {
        "name": case["name"],
        "label_zh": case["label_zh"],
        "failure_mode": case["failure_mode"],
        "unit": case["unit"],
        "theory_limit": theory_limit,
        "simulation_failure_load": failure_load,
        "simulation_to_theory_ratio": (
            failure_load / theory_limit if failure_load is not None else None
        ),
        "signed_relative_error": relative_error,
        "failure_time_s": failure_time,
        "detected": failure_load is not None,
        "theory_boundary_wrench": theory.wrench.tolist(),
    }
    return summary, records


def main() -> None:
    sweep_config = load_yaml(SWEEP_CONFIG_PATH)
    base_config = load_yaml(ROOT / sweep_config["base_config"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for case in sweep_config["cases"]:
        print(f"运行工况：{case['label_zh']}")
        summary, records = run_case(base_config, sweep_config, case)
        summaries.append(summary)
        path = OUTPUT_DIR / f"{case['name']}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if summary["detected"]:
            print(
                f"  理论边界={summary['theory_limit']:.5g} {summary['unit']}，"
                f"仿真失效={summary['simulation_failure_load']:.5g} {summary['unit']}，"
                f"比值={summary['simulation_to_theory_ratio']:.3f}"
            )
        else:
            print("  扫描上限内未检测到持续失效")

    (OUTPUT_DIR / "validation_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / "validation_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fieldnames = [
            "name", "label_zh", "failure_mode", "unit", "theory_limit",
            "simulation_failure_load", "simulation_to_theory_ratio",
            "signed_relative_error", "failure_time_s", "detected",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"验证数据已写入：{OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
