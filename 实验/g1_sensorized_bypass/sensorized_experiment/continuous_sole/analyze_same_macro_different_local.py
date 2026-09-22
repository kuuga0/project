"""Find repeated macro states whose plantar/IMU observations differ.

Pairs are restricted to the same terrain and different episodes.  This avoids
using adjacent timestamps as evidence and asks a stronger question: when the
ankle wrench/CoP looks similar, do the distributed sensors still distinguish
the state?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def _matrix(rows: list[dict[str, str]], names: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)


def _zscore(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.std(x, axis=0)
    scale[scale < 1e-9] = 1.0
    return (x - np.mean(x, axis=0)) / scale, scale


def _chinese_font() -> FontProperties:
    for candidate in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if candidate.exists():
            return FontProperties(fname=str(candidate))
    return FontProperties(family="DejaVu Sans")


def plot_macro_vs_enhanced(
    macro_distance: np.ndarray,
    enhanced_distance: np.ndarray,
    figure_path: Path,
    macro_threshold: float = 0.25,
    enhanced_threshold: float = 1.0,
) -> None:
    """Plot macro distance against the joint plantar-plus-IMU distance."""
    font = _chinese_font()
    near_macro = macro_distance < macro_threshold
    large_enhanced = enhanced_distance > enhanced_threshold
    focus = near_macro & large_enhanced
    near_but_not_large = near_macro & ~large_enhanced
    far_macro = ~near_macro
    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=180)
    ax.scatter(
        macro_distance[far_macro], enhanced_distance[far_macro],
        s=12, alpha=0.22, color="#9aa0a6", edgecolors="none",
        label="宏观量不够接近",
    )
    ax.scatter(
        macro_distance[near_but_not_large], enhanced_distance[near_but_not_large],
        s=18, alpha=0.55, color="#f2a900", edgecolors="none",
        label="宏观量接近，联合差异不大",
    )
    ax.scatter(
        macro_distance[focus], enhanced_distance[focus],
        s=24, alpha=0.85, color="#d94841", edgecolors="white", linewidths=0.35,
        label="宏观量接近，但足底＋IMU差异大",
    )
    ax.axvline(
        macro_threshold, color="#b36b00", linestyle="--", linewidth=1.2,
        label=f"宏观近邻阈值 {macro_threshold:g}",
    )
    ax.axhline(
        enhanced_threshold, color="#a61b1b", linestyle="--", linewidth=1.2,
        label=f"联合差异阈值 {enhanced_threshold:g}",
    )
    ax.set_xlabel("宏观量差异（踝部六维力/力矩＋CoP，标准化距离）", fontproperties=font)
    ax.set_ylabel("足底阵列＋踝部 IMU 联合差异（标准化距离）", fontproperties=font)
    ax.set_title("宏观量相近时，足底＋IMU 是否仍能区分状态？", fontproperties=font)
    ax.grid(True, color="#d9dde2", linewidth=0.55, alpha=0.75)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(prop=font, loc="upper left", frameon=True, fontsize=9)
    summary = (
        f"样本对 {len(macro_distance)}；宏观近邻 {int(np.sum(near_macro))}；"
        f"近邻且联合差异大 {int(np.sum(focus))}"
    )
    ax.text(
        0.99, 0.03, summary, transform=ax.transAxes, ha="right", va="bottom",
        fontproperties=font, fontsize=9, color="#333333",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#c7cbd1"},
    )
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze(
    paths: list[Path],
    macro_tol: float = 0.75,
    figure_path: Path | None = None,
) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))
    if not rows:
        raise ValueError("no rows")
    macro_names = [
        f"{foot}_ankle_{component}"
        for foot in ("left", "right")
        for component in ("fx", "fy", "fz", "tx", "ty", "tz")
    ] + [
        f"{foot}_cop_{component}_m"
        for foot in ("left", "right")
        for component in ("x", "y")
    ]
    local_names = [
        name for name in rows[0]
        if (name.startswith("left_cell_") or name.startswith("right_cell_"))
        and (name.endswith("_fx") or name.endswith("_fy") or name.endswith("_fz"))
    ]
    imu_names = [
        name for name in rows[0]
        if (name.startswith("left_imu_") or name.startswith("right_imu_"))
        and (name.endswith("rad_s") or name.endswith("m_s2"))
    ]
    macro, _ = _zscore(_matrix(rows, macro_names))
    local, _ = _zscore(_matrix(rows, local_names))
    imu, _ = _zscore(_matrix(rows, imu_names))
    active_names = [
        name for name in rows[0]
        if (name.startswith("left_cell_") or name.startswith("right_cell_")) and name.endswith("_fz")
    ]
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["scene"], []).append(index)

    pairs: list[dict[str, float | str | int]] = []
    for scene, indices in groups.items():
        for i in indices:
            candidates = [j for j in indices if rows[j]["episode"] != rows[i]["episode"]]
            if not candidates:
                continue
            distances = np.linalg.norm(macro[candidates] - macro[i], axis=1) / np.sqrt(macro.shape[1])
            best_at = int(np.argmin(distances))
            j = candidates[best_at]
            if float(distances[best_at]) > macro_tol:
                continue
            local_distance = float(np.linalg.norm(local[i] - local[j]) / np.sqrt(local.shape[1]))
            imu_distance = float(np.linalg.norm(imu[i] - imu[j]) / np.sqrt(imu.shape[1]))
            enhanced_distance = float(np.sqrt((local_distance ** 2 + imu_distance ** 2) / 2.0))
            pairs.append({
                "scene": scene,
                "i": i,
                "j": j,
                "macro_distance": float(distances[best_at]),
                "plantar_distance": local_distance,
                "imu_distance": imu_distance,
                "plantar_imu_joint_distance": enhanced_distance,
                "time_i_s": float(rows[i]["time_s"]),
                "time_j_s": float(rows[j]["time_s"]),
                "episode_i": int(rows[i]["episode"]),
                "episode_j": int(rows[j]["episode"]),
                "active_cells_i": int(sum(float(rows[i][name]) > 1e-6 for name in active_names)),
                "active_cells_j": int(sum(float(rows[j][name]) > 1e-6 for name in active_names)),
                "imu_norm_i": float(np.linalg.norm(imu[i])),
                "imu_norm_j": float(np.linalg.norm(imu[j])),
            })
    if not pairs:
        raise ValueError("no cross-episode pairs within macro tolerance")
    macro_d = np.asarray([p["macro_distance"] for p in pairs])
    plantar_d = np.asarray([p["plantar_distance"] for p in pairs])
    imu_d = np.asarray([p["imu_distance"] for p in pairs])
    enhanced_d = np.asarray([p["plantar_imu_joint_distance"] for p in pairs])
    macro_near = macro_d < 0.25
    enhanced_large = enhanced_d > 1.0
    if figure_path is not None:
        plot_macro_vs_enhanced(macro_d, enhanced_d, figure_path)
    result = {
        "inputs": [str(path) for path in paths],
        "macro_definition": macro_names,
        "pair_rule": "same scene, different episode, nearest standardized macro state",
        "macro_tolerance": macro_tol,
        "pairs": len(pairs),
        "macro_distance_median": float(np.median(macro_d)),
        "plantar_distance_median": float(np.median(plantar_d)),
        "imu_distance_median": float(np.median(imu_d)),
        "plantar_imu_joint_distance_definition": "sqrt((plantar_distance^2 + imu_distance^2) / 2), where both component distances are RMS distances in separately standardized feature spaces",
        "plantar_imu_joint_distance_median": float(np.median(enhanced_d)),
        "plantar_imu_joint_distance_gt_1_fraction": float(np.mean(enhanced_large)),
        "plantar_imu_joint_distance_gt_1_when_macro_lt_0.25": float(np.mean(enhanced_large[macro_near])) if np.any(macro_near) else 0.0,
        "macro_near_lt_0.25_pairs": int(np.sum(macro_near)),
        "macro_near_and_joint_gt_1_pairs": int(np.sum(macro_near & enhanced_large)),
        "plantar_to_macro_distance_ratio": float(np.median(plantar_d) / max(np.median(macro_d), 1e-9)),
        "imu_to_macro_distance_ratio": float(np.median(imu_d) / max(np.median(macro_d), 1e-9)),
        "fraction_plantar_distance_gt_1": float(np.mean(plantar_d > 1.0)),
        "fraction_imu_distance_gt_1": float(np.mean(imu_d > 1.0)),
        "fraction_plantar_gt_1_when_macro_gt_0.5": float(np.mean(plantar_d[macro_d < 0.5] > 1.0)) if np.any(macro_d < 0.5) else 0.0,
        "fraction_plantar_gt_1_when_macro_gt_0.25": float(np.mean(plantar_d[macro_d < 0.25] > 1.0)) if np.any(macro_d < 0.25) else 0.0,
        "per_scene": {
            scene: {
                "pairs": sum(1 for p in pairs if p["scene"] == scene),
                "macro_distance_median": float(np.median([p["macro_distance"] for p in pairs if p["scene"] == scene])),
                "plantar_distance_median": float(np.median([p["plantar_distance"] for p in pairs if p["scene"] == scene])),
                "imu_distance_median": float(np.median([p["imu_distance"] for p in pairs if p["scene"] == scene])),
                "plantar_gt_1_fraction": float(np.mean([p["plantar_distance"] > 1.0 for p in pairs if p["scene"] == scene])),
            }
            for scene in sorted(groups)
        },
        "examples": sorted(pairs, key=lambda p: float(p["plantar_distance"]), reverse=True)[:10],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--macro-tolerance", type=float, default=0.75)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    result = analyze(args.csv_files, args.macro_tolerance, args.figure)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
