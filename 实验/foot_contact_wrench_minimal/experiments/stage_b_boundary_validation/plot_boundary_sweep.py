"""Plot theory-versus-simulation boundary validation in Chinese."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "stage_b_validation"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def configure_font() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(FONT_PATH)
        ).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    configure_font()
    summaries = json.loads(
        (OUTPUT_DIR / "validation_summary.json").read_text(encoding="utf-8")
    )
    labels = [row["label_zh"] for row in summaries]
    ratios = np.asarray(
        [
            row["simulation_to_theory_ratio"]
            if row["simulation_to_theory_ratio"] is not None
            else np.nan
            for row in summaries
        ],
        dtype=float,
    )
    positions = np.arange(len(labels))
    fig, (axis_ratio, axis_error) = plt.subplots(1, 2, figsize=(13, 6))
    axis_ratio.axvline(1.0, color="#2e7d4f", linewidth=1.5, label="理论边界")
    axis_ratio.barh(positions, ratios, color="#2676b8", height=0.62, label="仿真/理论")
    axis_ratio.set_yticks(positions, labels)
    axis_ratio.invert_yaxis()
    axis_ratio.set_xlabel("仿真失效载荷 / 理论边界")
    axis_ratio.set_title("边界比值（理想值为 1）")
    axis_ratio.grid(axis="x", alpha=0.25)
    axis_ratio.legend()
    for position, ratio in zip(positions, ratios):
        if np.isfinite(ratio):
            axis_ratio.text(ratio, position, f" {ratio:.3f}", va="center")
        else:
            axis_ratio.text(0.02, position, "未检测", va="center")

    errors = 100.0 * (ratios - 1.0)
    colors = np.where(errors >= 0.0, "#d9843b", "#c33c2f")
    axis_error.barh(positions, errors, color=colors, height=0.62)
    axis_error.axvline(0.0, color="#2e7d4f", linewidth=1.2)
    axis_error.set_yticks(positions, labels)
    axis_error.invert_yaxis()
    axis_error.set_xlabel("有符号相对误差 (%)")
    axis_error.set_title("正值：仿真边界更大；负值：仿真更早失效")
    axis_error.grid(axis="x", alpha=0.25)
    for position, error in zip(positions, errors):
        if np.isfinite(error):
            axis_error.text(error, position, f" {error:+.1f}%", va="center")
    fig.suptitle("局部接触扳手理论边界与 MuJoCo 准静态失效边界验证", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = OUTPUT_DIR / "boundary_validation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(12, 13), sharex=False, sharey=True)
    for axis, summary in zip(axes.flat, summaries):
        records = [
            json.loads(line)
            for line in (OUTPUT_DIR / f"{summary['name']}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        time = [row["time_s"] for row in records]
        utilization = [row["utilization"] for row in records]
        axis.plot(time, utilization, color="#2676b8", linewidth=1.1)
        axis.axhline(1.0, color="#2e7d4f", linewidth=1.0, label="理论边界")
        if summary["failure_time_s"] is not None:
            axis.axvline(summary["failure_time_s"], color="#c33c2f", linestyle="--", linewidth=1.0, label="持续失效")
        axis.set_title(summary["label_zh"], fontsize=10)
        axis.set_xlabel("加载时间 (s)")
        axis.set_ylabel("载荷 / 理论边界")
        axis.grid(True, alpha=0.22)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2)
    fig.suptitle("各方向准静态加载过程", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    trace_path = OUTPUT_DIR / "sweep_traces.png"
    fig.savefig(trace_path, dpi=180)
    plt.close(fig)
    print(f"已保存验证图：{path.relative_to(ROOT)}")
    print(f"已保存扫描轨迹：{trace_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
