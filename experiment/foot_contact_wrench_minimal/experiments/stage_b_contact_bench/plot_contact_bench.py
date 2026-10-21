"""绘制阶段 B 数字实验台的接触扳手与离散法向载荷。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "stage_b_bench"
INPUT_PATH = OUTPUT_DIR / "contact_truth.jsonl"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def configure_chinese_font() -> None:
    """优先使用 Windows 中文字体，避免图中中文显示为方框。"""
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
        raise RuntimeError("没有找到阶段 B 接触真值记录，请先运行 run_contact_bench.py")

    time = np.asarray([row["time_s"] for row in records])
    wrench = np.asarray([row["contact_wrench_world"] for row in records])
    point_names = records[0].get(
        "normal_force_names",
        [f"contact_{index}" for index in range(len(records[0]["normal_forces_n"]))],
    )
    normal_points = np.asarray([row["normal_forces_n"] for row in records])
    normal_sum = np.sum(normal_points, axis=1)

    fig, (ax_force, ax_torque, ax_normal) = plt.subplots(
        3, 1, figsize=(10, 9), constrained_layout=True
    )
    for index, label in ((3, "f_x"), (4, "f_y"), (5, "f_z")):
        ax_force.plot(time, wrench[:, index], label=label, linewidth=1.2)
    ax_force.set_title("阶段 B：地面对足的合接触扳手")
    ax_force.set_ylabel("合力 (N)")
    ax_force.grid(True, alpha=0.3)
    ax_force.legend(ncol=3, fontsize=8)

    for index, label in ((0, "τ_x"), (1, "τ_y"), (2, "τ_z")):
        ax_torque.plot(time, wrench[:, index], label=label, linewidth=1.2)
    ax_torque.set_ylabel("合力矩 (N·m)")
    ax_torque.set_xlim(time[0], time[-1])
    if float(np.max(np.abs(wrench[:, :3]))) < 1e-9:
        ax_torque.set_ylim(-1e-9, 1e-9)
        ax_torque.text(
            0.01,
            0.9,
            "合力矩仅含数值舍入误差（< 1e-9 N·m）",
            transform=ax_torque.transAxes,
            va="top",
        )
    ax_torque.grid(True, alpha=0.3)
    ax_torque.legend(ncol=3, fontsize=8)

    final_normal = normal_points[-1].reshape(5, 10)
    image = ax_normal.imshow(
        final_normal,
        origin="lower",
        extent=[-100, 100, -50, 50],
        aspect="equal",
        cmap="viridis",
        vmin=0.0,
        vmax=max(float(np.max(final_normal)), 1e-12),
    )
    fig.colorbar(image, ax=ax_normal, label="法向力 (N/单元)")
    ax_normal.set_title("末时刻 10×5 离散阵列法向载荷")
    ax_normal.set_xlabel("足底纵向 x (mm)")
    ax_normal.set_ylabel("足底横向 y (mm)")
    ax_normal.set_xticks(np.arange(-100, 101, 20))
    ax_normal.set_yticks(np.arange(-50, 51, 20))
    for y_index in range(6):
        ax_normal.axhline(-50 + 20 * y_index, color="white", linewidth=0.5)
    for x_index in range(11):
        ax_normal.axvline(-100 + 20 * x_index, color="white", linewidth=0.5)
    ax_torque.set_xlabel("时间 (s)")
    ax_force.set_xlim(time[0], time[-1])

    output_path = OUTPUT_DIR / "bench_contact_wrench.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"已保存图像：{output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
