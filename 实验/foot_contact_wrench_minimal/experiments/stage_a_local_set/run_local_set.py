"""运行阶段 A 基准算例，并保存查询结果。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from foot_contact_wrench.utils.config import load_yaml, local_set_from_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "local_set" / "stage_a_baseline.yml"
OUTPUT_DIR = ROOT / "outputs" / "stage_a_baseline"


def unit_direction(index: int, dimension: int = 6) -> np.ndarray:
    direction = np.zeros(dimension)
    direction[index] = 1.0
    return direction


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    local_set = local_set_from_config(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    support_rows: list[dict[str, float | int | str]] = []
    for axis in range(6):
        for sign in (-1.0, 1.0):
            direction = sign * unit_direction(axis)
            result = local_set.support(direction)
            support_rows.append(
                {
                    "axis": axis,
                    "sign": sign,
                    "objective": result.objective,
                    "status": result.status,
                    "wrench": json.dumps(result.wrench.tolist()),
                    "normal_forces": json.dumps(result.normal_forces.tolist()),
                    "tangential_forces": json.dumps(
                        result.tangential_forces.tolist()
                    ),
                }
            )

    working_normal = np.full(local_set.model.point_count, local_set.normal_load / local_set.model.point_count)
    working_wrench = local_set.model.wrench(
        working_normal, np.zeros((local_set.model.point_count, 2))
    )
    radial_rows: list[dict[str, float | int | str]] = []
    for axis in range(6):
        direction = unit_direction(axis)
        result = local_set.radial_capacity(working_wrench, direction)
        radial_rows.append(
            {
                "axis": axis,
                "objective": result.objective,
                "status": result.status,
                "working_wrench": json.dumps(working_wrench.tolist()),
                "boundary_wrench": json.dumps(result.wrench.tolist()),
            }
        )

    with (OUTPUT_DIR / "support_queries.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=support_rows[0].keys())
        writer.writeheader()
        writer.writerows(support_rows)
    with (OUTPUT_DIR / "radial_queries.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=radial_rows[0].keys())
        writer.writeheader()
        writer.writerows(radial_rows)
    (OUTPUT_DIR / "working_point.json").write_text(
        json.dumps(
            {
                "config": str(CONFIG_PATH.relative_to(ROOT)),
                "contact_points_m": local_set.model.points.tolist(),
                "cell_areas_m2": local_set.model.cell_areas.tolist(),
                "normal_force_max_n": local_set.model.normal_force_max.tolist(),
                "working_wrench": working_wrench.tolist(),
                "normal_forces": working_normal.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"配置文件：{CONFIG_PATH.relative_to(ROOT)}")
    print(f"输出目录：{OUTPUT_DIR.relative_to(ROOT)}")
    print(f"离散单元数：{local_set.model.point_count}")
    print(f"当前工作扳手 [力矩; 力]：{working_wrench}")
    for row in support_rows:
        print(
            f"集合边界：第 {row['axis']} 维，方向 {row['sign']:+.0f}，"
            f"值={row['objective']:.6f}，状态={row['status']}"
        )
    for row in radial_rows:
        print(
            f"径向容量：第 {row['axis']} 维，"
            f"值={row['objective']:.6f}，状态={row['status']}"
        )


if __name__ == "__main__":
    main()
