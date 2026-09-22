"""绘制阶段 A 局部接触扳手集合的二维投影。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import ConvexHull

from foot_contact_wrench.utils.config import load_yaml, local_set_from_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "local_set" / "stage_a_baseline.yml"
OUTPUT_DIR = ROOT / "outputs" / "stage_a_baseline"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def configure_chinese_font() -> None:
    """优先使用 Windows 中文字体，避免图中中文显示为方框。"""
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(FONT_PATH)
        ).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def projected_support_polygon(local_set, axes: tuple[int, int], samples: int) -> np.ndarray:
    points: list[np.ndarray] = []
    for angle in np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False):
        direction_2d = np.array([np.cos(angle), np.sin(angle)])
        direction = np.zeros(6)
        direction[list(axes)] = direction_2d
        result = local_set.support(direction)
        if not result.feasible:
            raise RuntimeError(f"support query failed: {result.status}")
        points.append(result.wrench[list(axes)])
    projected = np.asarray(points)
    hull = ConvexHull(projected)
    return projected[hull.vertices]


def main() -> None:
    configure_chinese_font()
    config = load_yaml(CONFIG_PATH)
    local_set = local_set_from_config(config)
    samples = int(config.get("plot", {}).get("samples", 181))
    projections = config.get("plot", {}).get("projections", [])
    if len(projections) != 2:
        raise ValueError("baseline config must contain two plot projections")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, projection in zip(axes, projections):
        index_pair = tuple(int(index) for index in projection["axes"])
        polygon = projected_support_polygon(local_set, index_pair, samples)
        axis.fill(polygon[:, 0], polygon[:, 1], color="#4c78a8", alpha=0.18)
        axis.plot(
            np.r_[polygon[:, 0], polygon[0, 0]],
            np.r_[polygon[:, 1], polygon[0, 1]],
            color="#1f4e79",
            linewidth=1.5,
            label="局部接触扳手集合投影",
        )
        axis.set_xlabel(f"{projection['labels'][0]} ({projection['units'][0]})")
        axis.set_ylabel(f"{projection['labels'][1]} ({projection['units'][1]})")
        axis.set_title(projection.get("title", projection["name"]))
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")
    working_normal = np.full(
        local_set.model.point_count,
        local_set.normal_load / local_set.model.point_count,
    )
    working_wrench = local_set.model.wrench(
        working_normal, np.zeros((local_set.model.point_count, 2))
    )
    for axis, projection in zip(axes, projections):
        index_pair = tuple(int(index) for index in projection["axes"])
        point = working_wrench[list(index_pair)]
        axis.scatter(
            point[0], point[1], color="#d95f02", s=42, zorder=4,
            label="当前工作扳手",
        )
        axis.legend(fontsize=9, loc="lower left")
    fig.suptitle(
        "阶段 A：局部接触扳手集合二维投影\n"
        f"总法向载荷 N={local_set.normal_load:g} N；候选摩擦系数 μ={local_set.model.friction[0]:g}"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "baseline_projections.png", dpi=180)
    plt.close(fig)

    metadata = {
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "normal_load_n": local_set.normal_load,
        "friction_coefficients": local_set.model.friction.tolist(),
        "projection_axes": [list(map(int, item["axes"])) for item in projections],
    }
    (OUTPUT_DIR / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"已保存图像：{(OUTPUT_DIR / 'baseline_projections.png').relative_to(ROOT)}")

    grid = config.get("contact_grid")
    if grid is not None:
        size = np.asarray(grid["size_m"], dtype=float)
        resolution = np.asarray(grid["resolution_m"], dtype=float)
        fig_grid, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        for index, point in enumerate(local_set.model.points):
            axis.add_patch(
                Rectangle(
                    1000.0 * (point[:2] - 0.5 * resolution),
                    1000.0 * resolution[0], 1000.0 * resolution[1],
                    facecolor="#dceaf7", edgecolor="#356a93", linewidth=0.8,
                )
            )
            axis.text(
                1000.0 * point[0], 1000.0 * point[1], str(index),
                ha="center", va="center", fontsize=7,
            )
        axis.set_xlim(-500.0 * size[0], 500.0 * size[0])
        axis.set_ylim(-500.0 * size[1], 500.0 * size[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("足底纵向 x (mm)")
        axis.set_ylabel("足底横向 y (mm)")
        axis.set_title(
            f"矩形足底离散阵列：{local_set.model.point_count} 个有限面积单元\n"
            "足底 200 mm × 100 mm；单元 20 mm × 20 mm"
        )
        grid_path = OUTPUT_DIR / "contact_grid_layout.png"
        fig_grid.savefig(grid_path, dpi=180)
        plt.close(fig_grid)
        print(f"已保存图像：{grid_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
