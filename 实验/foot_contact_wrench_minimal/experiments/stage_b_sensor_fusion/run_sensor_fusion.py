"""Run the ideal IMU + pressure-array single-foot sensor comparison."""

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
from foot_contact_wrench.mujoco_io.sensor_bench import sinusoidal_excitation_wrench
from foot_contact_wrench.utils.config import load_yaml, local_set_from_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "bench" / "stage_b_sensor_bench.yml"


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    base_config = load_yaml(ROOT / config["base_config"])
    model, data = load_bench_model(str(ROOT / base_config["model_xml"]))
    model.opt.timestep = float(base_config["timestep_s"])
    model.opt.gravity[:] = np.asarray(base_config["gravity_mps2"], dtype=float)
    local_set = local_set_from_config(base_config)
    points_foot = np.asarray(local_set.model.points, dtype=float)
    body_id = model.body("foot").id
    initialize_free_foot(
        model,
        data,
        base_config["initial_position_m"],
        base_config["initial_linear_velocity_mps"],
        base_config["initial_angular_velocity_radps"],
    )

    warmup_steps = int(float(config["warmup_duration_s"]) / model.opt.timestep)
    total_steps = int(float(config["duration_s"]) / model.opt.timestep)
    sample_stride = max(1, int(config.get("sample_stride", 1)))
    gyro_previous = np.zeros(3)
    records: list[dict] = []
    for step in range(total_steps):
        time_s = float(data.time)
        external = sinusoidal_excitation_wrench(
            config, max(0.0, time_s - float(config["warmup_duration_s"]))
        )
        if step < warmup_steps:
            external[:] = 0.0
        apply_world_wrench_at_com(data, body_id, external)
        mujoco.mj_step(model, data)
        sample = contact_wrench_about_body(model, data, "foot")
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
        if step % sample_stride == 0:
            records.append(
                {
                    "time_s": float(data.time),
                    "contact_count": sample.contact_count,
                    "contact_constraint_count": sample.contact_constraint_count,
                    "external_wrench_world": external.tolist(),
                    "contact_wrench_world": sample.wrench_world.tolist(),
                    "imu_wrench_world": estimate.imu_wrench_world.tolist(),
                    "pressure_wrench_world": estimate.pressure_wrench_world.tolist(),
                    "fused_wrench_world": estimate.fused_wrench_world.tolist(),
                    "gyro_foot_radps": estimate.gyro_foot_radps.tolist(),
                    "specific_force_foot_mps2": estimate.specific_force_foot_mps2.tolist(),
                    "normal_forces_n": sample.normal_forces.tolist(),
                    "com_position_world_m": sample.com_position_world.tolist(),
                }
            )

    output_dir = ROOT / "outputs" / "stage_b_sensor_bench"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sensor_wrench_records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    truth = np.asarray([row["contact_wrench_world"] for row in records])
    imu = np.asarray([row["imu_wrench_world"] for row in records])
    pressure = np.asarray([row["pressure_wrench_world"] for row in records])
    fused = np.asarray([row["fused_wrench_world"] for row in records])
    metrics = {
        "wrench_order": ["tau_x", "tau_y", "tau_z", "f_x", "f_y", "f_z"],
        "units": ["N m", "N m", "N m", "N", "N", "N"],
        "sample_count": len(records),
        "imu_rmse": np.sqrt(np.mean((imu - truth) ** 2, axis=0)).tolist(),
        "pressure_rmse": np.sqrt(np.mean((pressure - truth) ** 2, axis=0)).tolist(),
        "fused_rmse": np.sqrt(np.mean((fused - truth) ** 2, axis=0)).tolist(),
        "fused_max_abs_error": np.max(np.abs(fused - truth), axis=0).tolist(),
        "note": "理想无噪声 IMU 与 50 点法向阵列；已知外部激励在 IMU 动力学估计中扣除。",
    }
    (output_dir / "sensor_wrench_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "sensor_wrench_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", *metrics["wrench_order"]])
        writer.writerow(["imu_rmse", *metrics["imu_rmse"]])
        writer.writerow(["pressure_rmse", *metrics["pressure_rmse"]])
        writer.writerow(["fused_rmse", *metrics["fused_rmse"]])
        writer.writerow(["fused_max_abs_error", *metrics["fused_max_abs_error"]])

    print(f"仿真采样数：{len(records)}")
    print(f"末时刻活动接触点：{records[-1]['contact_count']}/50")
    print(f"IMU 扭矩/力 RMSE：[力矩; 力] {metrics['imu_rmse']}")
    print(f"压力阵列 RMSE：[力矩; 力] {metrics['pressure_rmse']}")
    print(f"融合扳手 RMSE：[力矩; 力] {metrics['fused_rmse']}")
    print(f"输出目录：{output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
