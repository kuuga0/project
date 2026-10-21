"""运行简化单足 MuJoCo 数字实验台，并生成中文结果摘要。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from foot_contact_wrench.mujoco_io.bench import load_bench_model, run_free_body_bench
from foot_contact_wrench.utils.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "bench" / "stage_b_bench.yml"


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    model_path = ROOT / str(config["model_xml"])
    model, data = load_bench_model(str(model_path))
    model.opt.timestep = float(config["timestep_s"])
    model.opt.gravity[:] = np.asarray(config["gravity_mps2"], dtype=float)
    records = run_free_body_bench(
        model,
        data,
        duration_s=float(config["duration_s"]),
        initial_position=np.asarray(config["initial_position_m"], dtype=float),
        initial_linear_velocity=np.asarray(
            config["initial_linear_velocity_mps"], dtype=float
        ),
        initial_angular_velocity=np.asarray(
            config["initial_angular_velocity_radps"], dtype=float
        ),
    )

    output_dir = ROOT / "outputs" / "stage_b_bench"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "contact_truth.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "time_s", "contact_count", "contact_constraint_count",
            "fz_n", "tau_x_nm", "tau_y_nm", "tau_z_nm"
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            wrench = record["contact_wrench_world"]
            writer.writerow(
                {
                    "time_s": record["time_s"],
                    "contact_count": record["contact_count"],
                    "contact_constraint_count": record["contact_constraint_count"],
                    "fz_n": wrench[5],
                    "tau_x_nm": wrench[0],
                    "tau_y_nm": wrench[1],
                    "tau_z_nm": wrench[2],
                }
            )

    contact_records = [record for record in records if record["contact_count"] > 0]
    final = records[-1]
    print(f"模型文件：{model_path.relative_to(ROOT)}")
    print(f"仿真步数：{len(records)}")
    print(f"存在接触的采样数：{len(contact_records)}")
    print(f"末时刻活动离散单元数：{final['contact_count']}（阵列总数 50）")
    print(f"末时刻 MuJoCo 接触约束数：{final['contact_constraint_count']}")
    print(f"末时刻地面对足的接触扳手 [力矩; 力]：{final['contact_wrench_world']}")
    print(f"输出目录：{output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
