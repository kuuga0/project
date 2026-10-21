"""Plot ideal IMU/pressure-array wrench estimates against contact truth."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "stage_b_sensor_bench"
INPUT_PATH = OUTPUT_DIR / "sensor_wrench_records.jsonl"
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
    records = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError("没有传感器记录，请先运行 run_sensor_fusion.py")
    time = np.asarray([row["time_s"] for row in records])
    truth = np.asarray([row["contact_wrench_world"] for row in records])
    imu = np.asarray([row["imu_wrench_world"] for row in records])
    pressure = np.asarray([row["pressure_wrench_world"] for row in records])
    fused = np.asarray([row["fused_wrench_world"] for row in records])
    labels = ["τx", "τy", "τz", "fx", "fy", "fz"]
    units = ["N·m", "N·m", "N·m", "N", "N", "N"]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        ax.plot(time, truth[:, index], color="#222222", linewidth=1.4, label="MuJoCo真实值")
        ax.plot(time, imu[:, index], color="#d95f02", linewidth=0.9, label="IMU动力学")
        ax.plot(time, pressure[:, index], color="#1b9e77", linewidth=0.9, label="足底阵列")
        ax.plot(time, fused[:, index], color="#7570b3", linewidth=1.0, label="IMU+阵列融合")
        ax.set_title(f"{labels[index]} ({units[index]})")
        ax.set_xlabel("时间 (s)")
        ax.grid(True, alpha=0.25)
        if index == 0:
            ax.legend(fontsize=8)

    figure_path = OUTPUT_DIR / "sensor_wrench_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    error_fig, error_axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    error_axes[0].plot(
        time,
        np.linalg.norm(imu[:, :3] - truth[:, :3], axis=1),
        label="IMU合力矩误差",
    )
    error_axes[0].plot(
        time,
        np.linalg.norm(pressure[:, :3] - truth[:, :3], axis=1),
        label="阵列合力矩误差",
    )
    error_axes[0].plot(
        time,
        np.linalg.norm(fused[:, :3] - truth[:, :3], axis=1),
        label="融合合力矩误差",
    )
    error_axes[0].set_ylabel("合力矩误差 (N·m)")
    error_axes[0].set_title("传感器估计误差趋势（合力与合力矩分开度量）")
    error_axes[0].grid(True, alpha=0.25)
    error_axes[0].legend()
    error_axes[1].plot(
        time,
        np.linalg.norm(imu[:, 3:] - truth[:, 3:], axis=1),
        label="IMU合力误差",
    )
    error_axes[1].plot(
        time,
        np.linalg.norm(pressure[:, 3:] - truth[:, 3:], axis=1),
        label="阵列合力误差",
    )
    error_axes[1].plot(
        time,
        np.linalg.norm(fused[:, 3:] - truth[:, 3:], axis=1),
        label="融合合力误差",
    )
    error_axes[1].set_xlabel("时间 (s)")
    error_axes[1].set_ylabel("合力误差 (N)")
    error_axes[1].grid(True, alpha=0.25)
    error_axes[1].legend()
    error_path = OUTPUT_DIR / "sensor_wrench_error.png"
    error_fig.savefig(error_path, dpi=180)
    plt.close(error_fig)
    print(f"已保存图像：{figure_path.relative_to(ROOT)}")
    print(f"已保存图像：{error_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
