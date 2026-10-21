"""比较理论局部范围、MuJoCo 真值和法向阵列模型重建扳手。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from foot_contact_wrench.analysis.comparison import axial_support_bounds
from foot_contact_wrench.utils.config import load_yaml, local_set_from_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "bench" / "stage_b_bench.yml"
OUTPUT_DIR = ROOT / "outputs" / "stage_b_bench"
INPUT_PATH = OUTPUT_DIR / "contact_truth.jsonl"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def configure_chinese_font() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(FONT_PATH)
        ).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    configure_chinese_font()
    config = load_yaml(CONFIG_PATH)
    local_set = local_set_from_config(config)
    lower, upper = axial_support_bounds(local_set)
    records = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError("没有仿真记录，请先运行 run_contact_bench.py")
    time = np.asarray([row["time_s"] for row in records])
    actual = np.asarray([row["contact_wrench_foot"] for row in records])
    reconstructed = np.asarray(
        [row["pressure_model_wrench_foot"] for row in records]
    )
    labels = ["τ_x", "τ_y", "τ_z", "f_x", "f_y", "f_z"]
    units = ["N·m", "N·m", "N·m", "N", "N", "N"]
    actual_moment_norm = np.linalg.norm(actual[:, :3], axis=1)
    model_moment_norm = np.linalg.norm(reconstructed[:, :3], axis=1)
    actual_force_norm = np.linalg.norm(actual[:, 3:], axis=1)
    model_force_norm = np.linalg.norm(reconstructed[:, 3:], axis=1)

    fig = plt.figure(figsize=(12, 7))
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(2, 2, 1)
    axes[1, 0] = fig.add_subplot(2, 2, 3)
    axes[0, 1] = fig.add_subplot(2, 2, 2, projection="3d")
    axes[1, 1] = fig.add_subplot(2, 2, 4, projection="3d")
    axes[0, 0].plot(time, actual_force_norm, color="#c33c2f", linewidth=1.2, label="MuJoCo 实际值")
    axes[0, 0].plot(time, model_force_norm, color="#2676b8", linestyle="--", linewidth=1.0, label="模型重建值")
    axes[0, 0].set_title("合力大小 (N)")
    axes[1, 0].plot(time, actual_moment_norm, color="#c33c2f", linewidth=1.2)
    axes[1, 0].plot(time, model_moment_norm, color="#2676b8", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("合力矩大小 (N·m)")
    direction_time_index = len(time) - 1
    force_vectors = np.vstack((actual[direction_time_index, 3:], reconstructed[direction_time_index, 3:]))
    moment_vectors = np.vstack((actual[direction_time_index, :3], reconstructed[direction_time_index, :3]))
    for row, (vectors, title) in enumerate(((force_vectors, "合力方向"), (moment_vectors, "合力矩方向"))):
        axis = axes[row, 1]
        colors = ("#c33c2f", "#2676b8")
        names = ("MuJoCo 实际值", "模型重建值")
        for vector, color, name in zip(vectors, colors, names):
            norm = np.linalg.norm(vector)
            if norm > 1e-9:
                direction = vector / norm
                axis.quiver(0.0, 0.0, 0.0, *direction, length=1.0, normalize=False, color=color, label=name)
            else:
                axis.text2D(0.03, 0.92 if name == names[0] else 0.84, f"{name}：大小近似为零，方向未定义", color=color, transform=axis.transAxes)
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.05, 1.05)
        axis.set_zlim(-1.05, 1.05)
        axis.set_box_aspect((1, 1, 1))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(f"末时刻{title}")
        handles, direction_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, direction_labels, fontsize=8)
    for axis in axes[:, 0]:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(time[0], time[-1])
    axes[1, 0].set_xlabel("时间 (s)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(0.32, 0.92),
        ncol=2,
    )
    fig.suptitle(
        "仿真实际扳手与模型重建扳手的合力/合力矩对比\n"
        f"固定总法向载荷 N={local_set.normal_load:g} N；扳手参考点为足部质心",
        y=0.985,
    )
    fig.subplots_adjust(top=0.80, bottom=0.09, hspace=0.40, wspace=0.28)
    figure_path = OUTPUT_DIR / "theory_simulation_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    error = reconstructed - actual
    with (OUTPUT_DIR / "wrench_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = ["time_s"]
        for label in ("tau_x", "tau_y", "tau_z", "f_x", "f_y", "f_z"):
            fieldnames.extend([f"actual_{label}", f"model_{label}", f"error_{label}"])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, time_value in enumerate(time):
            row: dict[str, float] = {"time_s": float(time_value)}
            for axis_index, label in enumerate(
                ("tau_x", "tau_y", "tau_z", "f_x", "f_y", "f_z")
            ):
                row[f"actual_{label}"] = actual[row_index, axis_index]
                row[f"model_{label}"] = reconstructed[row_index, axis_index]
                row[f"error_{label}"] = error[row_index, axis_index]
            writer.writerow(row)
    metrics = {
        "wrench_order": ["tau_x", "tau_y", "tau_z", "f_x", "f_y", "f_z"],
        "units": units,
        "theory_lower": lower.tolist(),
        "theory_upper": upper.tolist(),
        "rmse_model_minus_actual": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "note": "模型值仅由离散法向载荷重建；未知切向分量由局部范围表达。所有扳手均相对于足部质心；理论范围按仿真载荷截面 N=39.24 N 计算。",
    }
    (OUTPUT_DIR / "wrench_comparison_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存对比图：{figure_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
