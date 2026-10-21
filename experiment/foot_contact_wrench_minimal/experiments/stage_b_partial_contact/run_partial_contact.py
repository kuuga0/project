"""运行不完整连续支撑下的单足准静态加载试验。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
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
    normal_path_wrench_direction,
    observed_normal_wrench_set,
    pressure_active_mask,
    set_support_scenario,
    tilt_angle_rad,
    vertical_point_load_wrench_world,
)
from foot_contact_wrench.utils.config import load_yaml, local_set_from_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "bench" / "stage_b_partial_contact.yml"
OUTPUT_DIR = ROOT / "outputs" / "stage_b_partial_contact"


def wrench_world_to_foot(data: mujoco.MjData, body_id: int, wrench_world) -> np.ndarray:
    rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    wrench = np.asarray(wrench_world, dtype=float).reshape(6)
    return np.concatenate((rotation.T @ wrench[:3], rotation.T @ wrench[3:]))


def normal_projection(wrench) -> np.ndarray:
    """Keep the pressure-observable ``[Mx, My, Fz]`` components."""
    projected = np.zeros(6)
    source = np.asarray(wrench, dtype=float).reshape(6)
    projected[[0, 1, 5]] = source[[0, 1, 5]]
    return projected


def phase_end_time(config: dict) -> float:
    return sum(
        float(config[key])
        for key in (
            "settle_duration_s",
            "preload_ramp_duration_s",
            "reference_hold_duration_s",
            "sweep_duration_s",
        )
    )


def run_scenario(
    base_config: dict,
    config: dict,
    scenario: dict,
    model_xml: Path,
    points_foot: np.ndarray,
) -> tuple[list[dict], dict]:
    model, data = load_bench_model(str(model_xml))
    model.opt.timestep = float(base_config["timestep_s"])
    model.opt.gravity[:] = np.asarray(base_config["gravity_mps2"], dtype=float)
    body_id = model.body("foot").id
    initialize_free_foot(
        model,
        data,
        base_config["initial_position_m"],
        base_config["initial_linear_velocity_mps"],
        base_config["initial_angular_velocity_radps"],
    )
    support_surface_count = set_support_scenario(
        model, data, scenario["name"], config["support_geom_prefix"]
    )

    threshold_n = float(config["pressure_active_threshold_n"])
    sample_stride = max(1, int(config.get("sample_stride", 1)))
    persistence_steps = max(
        1, int(float(config["failure_persistence_s"]) / model.opt.timestep)
    )
    failure_tilt = float(config["failure_tilt_angle_rad"])
    total_steps = int(np.ceil(phase_end_time(config) / model.opt.timestep)) + 1
    gyro_previous = np.zeros(3)
    records: list[dict] = []
    reference_set = None
    reference_wrench = None
    theory_boundary_wrench = None
    theory_path_progress = None
    direction, full_path_magnitude = normal_path_wrench_direction(
        [*scenario["load_path"]["start_xy_m"], config["load_application_height_m"]],
        [*scenario["load_path"]["end_xy_m"], config["load_application_height_m"]],
        float(config["applied_normal_force_n"]),
    )
    consecutive_failure = 0
    failure_record = None
    failure_candidate_record = None
    last_stable_record = None
    previous_phase = "settle"

    for step in range(total_steps):
        state = loading_state(float(data.time), config, scenario)
        external = vertical_point_load_wrench_world(
            data,
            body_id,
            state.application_point_foot_m,
            state.normal_force_n,
        )
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
        gyro_previous = estimate.gyro_foot_radps.copy()
        active_mask = pressure_active_mask(sample.normal_forces, threshold_n)

        # Freeze the last stable reference-hold set before the load point moves.
        if state.phase == "reference":
            candidate_set = observed_normal_wrench_set(
                base_config,
                points_foot,
                sample.normal_forces,
                base_config["normal"],
                threshold_n,
            )
            if candidate_set is not None:
                reference_set = candidate_set
                filtered_forces = np.where(active_mask, sample.normal_forces, 0.0)
                reference_wrench = candidate_set.model.wrench(
                    filtered_forces, np.zeros((len(filtered_forces), 2))
                )
        if previous_phase == "reference" and state.phase == "sweep":
            if reference_set is None or reference_wrench is None:
                raise RuntimeError(f"{scenario['name']}: no valid pressure-derived reference set")
            theory = reference_set.radial_capacity(reference_wrench, direction)
            if not theory.feasible or theory.objective is None or theory.wrench is None:
                raise RuntimeError(f"{scenario['name']}: radial-capacity query failed")
            theory_path_progress = float(theory.objective) / full_path_magnitude
            theory_boundary_wrench = theory.wrench.copy()
        previous_phase = state.phase

        tilt = tilt_angle_rad(data, body_id)
        if state.phase == "sweep" and tilt >= failure_tilt:
            consecutive_failure += 1
        else:
            consecutive_failure = 0
            failure_candidate_record = None

        if (
            step % sample_stride != 0
            and consecutive_failure not in (1, persistence_steps)
        ):
            continue

        truth_foot = sample.wrench_foot
        fused_foot = wrench_world_to_foot(data, body_id, estimate.fused_wrench_world)
        row = {
            "scenario": scenario["name"],
            "scenario_label_zh": scenario["label_zh"],
            "time_s": float(data.time),
            "phase": state.phase,
            "path_progress": state.path_progress,
            "applied_normal_force_n": state.normal_force_n,
            "load_application_point_foot_m": state.application_point_foot_m.tolist(),
            "contact_count": sample.contact_count,
            "contact_constraint_count": sample.contact_constraint_count,
            "observed_active_count": int(np.count_nonzero(active_mask)),
            "observed_active_mask": active_mask.tolist(),
            "normal_forces_n": sample.normal_forces.tolist(),
            "total_normal_load_n": float(np.sum(sample.normal_forces)),
            "external_wrench_world": external.tolist(),
            "contact_wrench_world": sample.wrench_world.tolist(),
            "contact_wrench_foot": truth_foot.tolist(),
            "pressure_wrench_foot": sample.pressure_model_wrench_foot.tolist(),
            "imu_wrench_world": estimate.imu_wrench_world.tolist(),
            "fused_wrench_world": estimate.fused_wrench_world.tolist(),
            "fused_wrench_foot": fused_foot.tolist(),
            "normal_wrench_error": (
                normal_projection(fused_foot) - normal_projection(truth_foot)
            ).tolist(),
            "theory_path_progress": theory_path_progress,
            "tilt_angle_rad": tilt,
            "failure_threshold_rad": failure_tilt,
            "com_position_world_m": sample.com_position_world.tolist(),
        }
        records.append(row)
        if state.phase == "sweep" and tilt < failure_tilt:
            last_stable_record = row
        if consecutive_failure == 1:
            failure_candidate_record = row
        if consecutive_failure >= persistence_steps:
            failure_record = row
            break
        if state.complete:
            break

    if reference_set is None or reference_wrench is None:
        raise RuntimeError(f"{scenario['name']}: experiment ended without a reference set")
    if theory_path_progress is None or theory_boundary_wrench is None:
        theory = reference_set.radial_capacity(reference_wrench, direction)
        if not theory.feasible or theory.objective is None or theory.wrench is None:
            raise RuntimeError(f"{scenario['name']}: radial-capacity query failed")
        theory_path_progress = float(theory.objective) / full_path_magnitude
        theory_boundary_wrench = theory.wrench.copy()

    truth = np.asarray([row["contact_wrench_foot"] for row in records])
    fused = np.asarray([row["fused_wrench_foot"] for row in records])
    reduced_indices = [0, 1, 5]
    error = fused[:, reduced_indices] - truth[:, reduced_indices]
    summary = {
        "scenario": scenario["name"],
        "scenario_label_zh": scenario["label_zh"],
        "support_surface_count_truth": support_surface_count,
        "support_boxes_truth": scenario["support_boxes"],
        "sample_count": len(records),
        "reference_active_count": int(np.count_nonzero(reference_set.model.active)),
        "reference_normal_load_n": float(reference_set.normal_load),
        "theory_path_progress": theory_path_progress,
        "theory_boundary_wrench_foot": theory_boundary_wrench.tolist(),
        "last_stable_path_progress": (
            last_stable_record["path_progress"] if last_stable_record else None
        ),
        "first_failure_path_progress": (
            failure_candidate_record["path_progress"]
            if failure_candidate_record is not None and failure_record is not None
            else None
        ),
        "confirmed_failure_path_progress": (
            failure_record["path_progress"] if failure_record else None
        ),
        "failure_detected": failure_record is not None,
        "reduced_wrench_rmse_mx_my_fz": np.sqrt(np.mean(error * error, axis=0)).tolist(),
    }
    return records, summary


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    base_config = load_yaml(ROOT / config["base_config"])
    points_foot = np.asarray(local_set_from_config(base_config).model.points, dtype=float)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_xml = OUTPUT_DIR / "generated_partial_contact_bench.xml"
    build_partial_contact_xml(
        ROOT / base_config["model_xml"],
        model_xml,
        config["scenarios"],
        support_geom_prefix=config["support_geom_prefix"],
        support_height_m=float(config["support_height_m"]),
        support_friction=config["support_friction"],
    )

    all_records: list[dict] = []
    summaries: list[dict] = []
    for scenario in config["scenarios"]:
        records, summary = run_scenario(base_config, config, scenario, model_xml, points_foot)
        all_records.extend(records)
        summaries.append(summary)
        failure = summary["first_failure_path_progress"]
        failure_text = f"{failure:.3f}" if failure is not None else "扫描范围内未失效"
        print(
            f"{scenario['label_zh']}：参考承载点 {summary['reference_active_count']}/50，"
            f"理论边界进度 {summary['theory_path_progress']:.3f}，"
            f"首次失效进度 {failure_text}"
        )

    with (OUTPUT_DIR / "partial_contact_records.jsonl").open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (OUTPUT_DIR / "partial_contact_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / "partial_contact_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fields = [key for key in summaries[0] if key != "support_boxes_truth"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"记录已保存：{OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
