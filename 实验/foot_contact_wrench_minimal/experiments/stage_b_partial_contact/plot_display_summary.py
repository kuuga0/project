"""生成用于汇报的连续部分支撑汇总图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "stage_b_partial_contact"
RECORDS_PATH = OUTPUT_DIR / "partial_contact_records.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "partial_contact_summary.json"
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
        for line in RECORDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summaries = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    names = list(dict.fromkeys(row["scenario"] for row in records))
    reference_rows = {
        name: [
            row
            for row in records
            if row["scenario"] == name and row["phase"] == "reference"
        ][-1]
        for name in names
    }
    summary_by_name = {row["scenario"]: row for row in summaries}
    common_pressure_max = max(
        float(np.max(row["normal_forces_n"])) for row in reference_rows.values()
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.2, 8.8),
        gridspec_kw={"height_ratios": [1.0, 1.1]},
        constrained_layout=True,
    )
    fig.suptitle(
        "MuJoCo 连续部分支撑：接触分布与局部支撑边界对比",
        fontsize=18,
        fontweight="bold",
    )

    image = None
    for index, name in enumerate(names):
        row = reference_rows[name]
        axis = axes[0, index]
        pressure = np.asarray(row["normal_forces_n"], dtype=float).reshape(5, 10)
        image = axis.imshow(
            pressure,
            origin="lower",
            extent=[-100, 100, -50, 50],
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=max(common_pressure_max, 1e-12),
        )
        summary = summary_by_name[name]
        axis.set_title(
            f"{row['scenario_label_zh']}\n"
            f"承载点 {row['observed_active_count']}/50 | "
            f"理论边界 {summary['theory_path_progress']:.3f}",
            fontsize=11,
        )
        axis.set_xlabel("足底纵向 x (mm)")
        if index == 0:
            axis.set_ylabel("足底横向 y (mm)")
        else:
            axis.set_ylabel("")

    if image is not None:
        fig.colorbar(
            image,
            ax=axes[0, :].tolist(),
            fraction=0.025,
            pad=0.02,
            label="单元法向载荷 (N)",
        )

    boundary_labels = ["理论边界", "最后稳定", "首次失效", "确认失效"]
    boundary_colors = ["#2563a6", "#f08a24", "#c43d32", "#3a9650"]
    for index, name in enumerate(names):
        axis = axes[1, index]
        summary = summary_by_name[name]
        values = [
            summary["theory_path_progress"],
            summary["last_stable_path_progress"],
            summary["first_failure_path_progress"],
            summary["confirmed_failure_path_progress"],
        ]
        values = [np.nan if value is None else float(value) for value in values]
        bars = axis.bar(
            np.arange(4),
            values,
            color=boundary_colors,
            width=0.68,
        )
        axis.set_ylim(0.0, 0.85)
        axis.set_xticks(np.arange(4), ["理论", "稳定", "首次失效", "确认失效"], rotation=28)
        axis.set_title(reference_rows[name]["scenario_label_zh"], fontsize=12)
        axis.set_ylabel("载荷作用点归一化路径进度" if index == 0 else "")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    output_path = OUTPUT_DIR / "display_partial_contact_summary.png"
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"已保存展示汇总图：{output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
