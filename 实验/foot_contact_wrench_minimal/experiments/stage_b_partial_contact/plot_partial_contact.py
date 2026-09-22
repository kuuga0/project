"""绘制准静态部分支撑试验的压力、倾转和理论边界对比。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "stage_b_partial_contact"
INPUT_PATH = OUTPUT_DIR / "partial_contact_records.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "partial_contact_summary.json"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def configure_chinese_font() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    configure_chinese_font()
    records = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summaries = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("没有记录，请先运行 run_partial_contact.py")
    names = list(dict.fromkeys(row["scenario"] for row in records))
    labels = {
        name: next(row["scenario_label_zh"] for row in records if row["scenario"] == name)
        for name in names
    }
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(names)))

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
    for color, name in zip(colors, names):
        rows = [row for row in records if row["scenario"] == name]
        time_s = np.asarray([row["time_s"] for row in rows])
        axes[0].plot(
            time_s,
            [row["total_normal_load_n"] for row in rows],
            color=color,
            label=labels[name],
        )
        axes[1].plot(
            time_s,
            np.degrees([row["tilt_angle_rad"] for row in rows]),
            color=color,
            label=labels[name],
        )
        errors = np.asarray([row["normal_wrench_error"] for row in rows])
        axes[2].plot(
            time_s,
            np.linalg.norm(errors[:, [0, 1]], axis=1),
            color=color,
            label=labels[name],
        )
        axes[3].plot(
            time_s,
            np.abs(errors[:, 5]),
            color=color,
            label=labels[name],
        )
    axes[0].set_title("准静态加载下的总法向载荷")
    axes[0].set_ylabel("Fz (N)")
    axes[1].set_title("足部倾角与首次失效判据")
    axes[1].set_ylabel("倾角 (deg)")
    axes[2].set_title("模型重建与 MuJoCo 真值的滚转/俯仰力矩误差")
    axes[2].set_ylabel("||(eMx, eMy)|| (N m)")
    axes[3].set_title("模型重建与 MuJoCo 真值的法向力误差")
    axes[3].set_ylabel("|eFz| (N)")
    axes[3].set_xlabel("时间 (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    timeseries_path = OUTPUT_DIR / "partial_contact_timeseries.png"
    fig.savefig(timeseries_path, dpi=180)
    plt.close(fig)

    reference_rows = []
    for name in names:
        candidates = [
            row for row in records if row["scenario"] == name and row["phase"] == "reference"
        ]
        reference_rows.append(candidates[-1])
    common_pressure_max = max(
        float(np.max(row["normal_forces_n"])) for row in reference_rows
    )
    fig, axes = plt.subplots(
        1, len(reference_rows), figsize=(3.2 * len(reference_rows), 3.8), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, row in zip(axes, reference_rows):
        pressure = np.asarray(row["normal_forces_n"]).reshape(5, 10)
        image = axis.imshow(
            pressure,
            origin="lower",
            extent=[-100, 100, -50, 50],
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=max(common_pressure_max, 1e-12),
        )
        axis.set_title(f"{row['scenario_label_zh']}\n承载点 {row['observed_active_count']}/50")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="法向力 (N)")
    pressure_path = OUTPUT_DIR / "partial_contact_pressure_maps.png"
    fig.savefig(pressure_path, dpi=180)
    plt.close(fig)

    x = np.arange(len(summaries), dtype=float)
    width = 0.25
    theory = np.asarray([row["theory_path_progress"] for row in summaries], dtype=float)
    stable = np.asarray(
        [np.nan if row["last_stable_path_progress"] is None else row["last_stable_path_progress"] for row in summaries]
    )
    failure = np.asarray(
        [np.nan if row["first_failure_path_progress"] is None else row["first_failure_path_progress"] for row in summaries]
    )
    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    axis.bar(x - width, theory, width, label="压力条件模型边界")
    axis.bar(x, stable, width, label="最后稳定")
    axis.bar(x + width, failure, width, label="首次失效")
    axis.set_xticks(x, [row["scenario_label_zh"] for row in summaries], rotation=12)
    axis.set_ylabel("载荷作用点归一化路径进度")
    axis.set_title("理论局部边界与准静态仿真失效区间")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    boundary_path = OUTPUT_DIR / "partial_contact_boundary_comparison.png"
    fig.savefig(boundary_path, dpi=180)
    plt.close(fig)

    print(f"已保存时序图：{timeseries_path.relative_to(ROOT)}")
    print(f"已保存压力分布图：{pressure_path.relative_to(ROOT)}")
    print(f"已保存边界对比图：{boundary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
