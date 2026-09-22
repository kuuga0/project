"""Quantify information that is lost by reducing a foot to ankle F/T + CoP.

This is intentionally a small, dependency-light analysis (CSV + NumPy).  It
does not claim that a virtual contact array is already a hardware truth.  It
answers the narrower first experiment question: on identical walking runs,
do detailed cell forces and ankle IMU channels vary in ways that cannot be
recovered from the six-axis ankle wrench and CoP alone?

Example::

    uv run --no-project --with numpy python \
      biped_walking/sensorized_experiment/analyze_sensor_information.py \
      biped_walking/sensorized_experiment/outputs/*_sensorized.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


FOOTS = ("left", "right")
WRENCH = ("fx", "fy", "fz", "tx", "ty", "tz")
IMU = (
    "gx_rad_s",
    "gy_rad_s",
    "gz_rad_s",
    "ax_m_s2",
    "ay_m_s2",
    "az_m_s2",
)


def _read_csv(paths: Iterable[Path], stride: int) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    field_names: list[str] | None = None
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"missing CSV header: {path}")
            if field_names is None:
                field_names = list(reader.fieldnames)
            elif list(reader.fieldnames) != field_names:
                raise ValueError(f"incompatible header in {path}")
            file_rows = list(reader)
        # Keep the temporal structure but avoid counting 50 Hz neighboring
        # samples as independent evidence in the summary statistics.
        rows.extend(file_rows[:: max(1, stride)])
    if not rows or field_names is None:
        raise ValueError("no rows found")
    return rows, field_names


def _matrix(rows: list[dict[str, str]], names: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)


def _zscore(values: np.ndarray, mean: np.ndarray | None = None, scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = np.mean(values, axis=0)
    if scale is None:
        scale = np.std(values, axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    return (values - mean) / scale, mean, scale


def _feature_names(field_names: list[str]) -> tuple[list[str], list[str], list[str]]:
    macro: list[str] = []
    array: list[str] = []
    imu: list[str] = []
    for foot in FOOTS:
        macro.extend([f"{foot}_ankle_{component}" for component in WRENCH])
        macro.extend([f"{foot}_cop_x_m", f"{foot}_cop_y_m"])
        array.extend(
            name
            for name in field_names
            if name.startswith(f"{foot}_cell_") and name.rsplit("_", 1)[-1] in {"fx", "fy", "fz"}
        )
        imu.extend([f"{foot}_imu_{component}" for component in IMU])
    return macro, array, imu


def _array_shape(array_names: list[str]) -> tuple[int, int]:
    """Infer the rectangular array dimensions from names in a CSV header."""

    indices = []
    for name in array_names:
        # Names are e.g. left_cell_7_3_fx.  A strict parse avoids accidentally
        # interpreting the force component suffix as a cell coordinate.
        parts = name.split("_")
        if len(parts) == 5 and parts[1] == "cell":
            indices.append((int(parts[2]), int(parts[3])))
    if not indices:
        return 0, 0
    return max(ix for ix, _ in indices) + 1, max(iy for _, iy in indices) + 1


def _nearest_cross_scene(
    macro_z: np.ndarray,
    rich_z: np.ndarray,
    array_z: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int]:
    """Compare detailed distances for pairs that look alike macroscopically."""

    nearest_macro: list[float] = []
    nearest_rich: list[float] = []
    nearest_array: list[float] = []
    n = len(labels)
    for i in range(n):
        mask = labels != labels[i]
        if not np.any(mask):
            continue
        candidates = np.flatnonzero(mask)
        macro_dist = np.linalg.norm(macro_z[candidates] - macro_z[i], axis=1)
        j = int(candidates[int(np.argmin(macro_dist))])
        nearest_macro.append(float(np.min(macro_dist) / np.sqrt(macro_z.shape[1])))
        nearest_rich.append(float(np.linalg.norm(rich_z[j] - rich_z[i]) / np.sqrt(rich_z.shape[1])))
        nearest_array.append(float(np.linalg.norm(array_z[j] - array_z[i]) / np.sqrt(array_z.shape[1])))

    if not nearest_macro:
        return {"pairs": 0}
    return {
        "pairs": len(nearest_macro),
        "macro_distance_median": float(np.median(nearest_macro)),
        "rich_distance_median": float(np.median(nearest_rich)),
        "array_distance_median": float(np.median(nearest_array)),
        "rich_to_macro_neighbor_ratio": float(np.median(nearest_rich) / max(np.median(nearest_macro), 1e-9)),
    }


def _nearest_centroid_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, object]:
    """Leave-one-episode-group-out nearest-centroid terrain classification."""

    predictions: list[str] = []
    truth: list[str] = []
    for group in np.unique(groups):
        train = groups != group
        test = ~train
        train_labels = np.unique(labels[train])
        if len(train_labels) < 2 or not np.any(test):
            continue
        train_z, mean, scale = _zscore(features[train])
        test_z, _, _ = _zscore(features[test], mean, scale)
        centroids = np.stack([np.mean(train_z[labels[train] == label], axis=0) for label in train_labels])
        distances = np.linalg.norm(test_z[:, None, :] - centroids[None, :, :], axis=2)
        pred = train_labels[np.argmin(distances, axis=1)]
        predictions.extend([str(value) for value in pred])
        truth.extend([str(value) for value in labels[test]])

    if not truth:
        return {"samples": 0}
    prediction_array = np.asarray(predictions)
    truth_array = np.asarray(truth)
    per_scene = {
        scene: float(np.mean(prediction_array[truth_array == scene] == scene))
        for scene in np.unique(truth_array)
    }
    return {
        "samples": len(truth),
        "correct": int(np.sum(prediction_array == truth_array)),
        "accuracy": float(np.mean(prediction_array == truth_array)),
        "per_scene_accuracy": per_scene,
    }


def _grouped_macro_to_array_rmse(macro: np.ndarray, array: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    """Cross-validated normalized error when macro channels predict the array."""

    errors: list[np.ndarray] = []
    for group in np.unique(groups):
        train = groups != group
        test = ~train
        if np.sum(train) < macro.shape[1] + 2 or not np.any(test):
            continue
        x_train, x_mean, x_scale = _zscore(macro[train])
        x_test, _, _ = _zscore(macro[test], x_mean, x_scale)
        y_train, y_mean, y_scale = _zscore(array[train])
        x_aug = np.column_stack([np.ones(len(x_train)), x_train])
        beta, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
        pred = np.column_stack([np.ones(len(x_test)), x_test]) @ beta
        y_test, _, _ = _zscore(array[test], y_mean, y_scale)
        errors.append(pred - y_test)
    if not errors:
        return {"samples": 0}
    residual = np.concatenate(errors, axis=0)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    return {
        "samples": int(residual.shape[0]),
        "normalized_rmse": rmse,
        "unexplained_fraction": float(min(1.0, rmse * rmse)),
        "explained_fraction": float(max(0.0, 1.0 - rmse * rmse)),
    }


def _contact_distribution_stats(rows: list[dict[str, str]], array_names: list[str]) -> dict[str, float | int]:
    """Report how often force is split across multiple plantar cells."""

    active_counts: list[int] = []
    nonzero_counts: list[int] = []
    concentration: list[float] = []
    foot_names = {
        foot: [name for name in array_names if name.startswith(f"{foot}_cell_")] for foot in FOOTS
    }
    for row in rows:
        for names in foot_names.values():
            values = np.asarray([float(row[name]) for name in names], dtype=np.float64)
            # Every third component is the foot-frame normal/vertical component.
            fz = np.maximum(values[2::3], 0.0)
            total = float(np.sum(fz))
            if total <= 1e-8:
                continue
            p = fz / total
            nonzero_counts.append(int(np.sum(fz > 1e-8)))
            active_counts.append(int(np.sum(fz > 0.01 * total)))
            concentration.append(float(np.sum(p * p)))
    if not active_counts:
        return {"contact_frames": 0}
    return {
        "contact_foot_frames": len(active_counts),
        "mean_nonzero_cells": float(np.mean(nonzero_counts)),
        "median_nonzero_cells": float(np.median(nonzero_counts)),
        "mean_active_cells_above_1pct": float(np.mean(active_counts)),
        "median_active_cells_above_1pct": float(np.median(active_counts)),
        "mean_cell_concentration": float(np.mean(concentration)),
        "median_cell_concentration": float(np.median(concentration)),
    }


def analyze(paths: list[Path], stride: int) -> dict[str, object]:
    rows, field_names = _read_csv(paths, stride)
    macro_names, array_names, imu_names = _feature_names(field_names)
    macro = _matrix(rows, macro_names)
    array = _matrix(rows, array_names)
    imu = _matrix(rows, imu_names)
    added = np.column_stack([array, imu])
    rich = np.column_stack([macro, added])
    labels = np.asarray([row["scene"] for row in rows])
    groups = np.asarray([f"{row['scene']}:episode_{row['episode']}" for row in rows])

    macro_z, _, _ = _zscore(macro)
    added_z, _, _ = _zscore(added)
    array_z, _, _ = _zscore(array)
    distribution = _contact_distribution_stats(rows, array_names)
    array_nx, array_ny = _array_shape(array_names)
    result: dict[str, object] = {
        "inputs": [str(path) for path in paths],
        "rows_after_stride": len(rows),
        "stride": stride,
        "scenes": {str(scene): int(np.sum(labels == scene)) for scene in np.unique(labels)},
        "feature_dimensions": {
            "macro_ankle_wrench_plus_cop": macro.shape[1],
            "plantar_array_triaxial": array.shape[1],
            "ankle_imu": imu.shape[1],
            "combined": rich.shape[1],
        },
        "array_shape": [array_nx, array_ny, 3],
        "array_occupancy": {
            "cells_per_foot": array_nx * array_ny,
            "mean_nonzero_cells_fraction_per_foot": float(
                distribution.get("mean_nonzero_cells", 0.0) / max(array_nx * array_ny, 1)
            ),
        },
        "contact_distribution": distribution,
        "macro_neighbor_test": _nearest_cross_scene(macro_z, added_z, array_z, labels),
        "terrain_classification": {
            "macro_only": _nearest_centroid_accuracy(macro, labels, groups),
            "array_plus_imu": _nearest_centroid_accuracy(np.column_stack([array, imu]), labels, groups),
            "macro_plus_array": _nearest_centroid_accuracy(np.column_stack([macro, array]), labels, groups),
            "macro_plus_array_plus_imu": _nearest_centroid_accuracy(rich, labels, groups),
        },
        "macro_to_array_prediction": _grouped_macro_to_array_rmse(macro, array, groups),
    }
    return result


def _markdown_report(result: dict[str, object]) -> str:
    neighbor = result["macro_neighbor_test"]
    classification = result["terrain_classification"]
    prediction = result["macro_to_array_prediction"]
    distribution = result["contact_distribution"]
    return f"""# 传感信息对比报告

本报告由 `analyze_sensor_information.py` 根据同一批行走轨迹生成。宏观基线是双脚踝部六维接触扳手与 CoP；扩展输入依次加入足底 `{result['array_shape'][0]}×{result['array_shape'][1]}` 单元的三维力和踝部 IMU。

## 数据规模

- 采样行数（降采样后）：`{result['rows_after_stride']}`
- 特征维数：宏观 `{result['feature_dimensions']['macro_ankle_wrench_plus_cop']}`，足底阵列 `{result['feature_dimensions']['plantar_array_triaxial']}`，IMU `{result['feature_dimensions']['ankle_imu']}`
- 各地形行数：`{result['scenes']}`
- 有接触的足-帧数：`{distribution.get('contact_foot_frames', 0)}`；每只脚平均非零单元数：`{distribution.get('mean_nonzero_cells', float('nan')):.2f}`；平均超过 1% 载荷阈值的单元数：`{distribution.get('mean_active_cells_above_1pct', float('nan')):.2f}`

## 结果一：宏观量相近时，阵列/IMU 仍然不同

对每一帧，寻找另一种地形中宏观特征最接近的帧，再比较完整传感特征。宏观近邻的归一化距离中位数为 `{neighbor.get('macro_distance_median', float('nan')):.4f}`；同一对样本的阵列+IMU 距离中位数为 `{neighbor.get('rich_distance_median', float('nan')):.4f}`，比值为 `{neighbor.get('rich_to_macro_neighbor_ratio', float('nan')):.2f}`。这表示六维扳手和 CoP 的相似并不意味着足底受力分布和踝部运动状态相似。

## 结果二：宏观量不能重建各单元三维力

用留一组（地形+episode）交叉验证，以宏观量线性预测所有阵列单元三维力，归一化 RMSE 为 `{prediction.get('normalized_rmse', float('nan')):.4f}`，未解释比例约为 `{prediction.get('unexplained_fraction', float('nan')):.2%}`。这是“合力/一阶力矩”到“多个单元三维力”的欠定性在数据上的直接表现。

## 结果三：地形区分能力

用留一 episode 的最近质心分类作为简单可复现实验：

|输入 |准确率|
|---|---:|
|踝部六维扳手 + CoP|{classification['macro_only'].get('accuracy', float('nan')):.2%}|
|足底阵列 + 踝部 IMU（不含宏观量）|{classification['array_plus_imu'].get('accuracy', float('nan')):.2%}|
|宏观量 + 足底阵列|{classification['macro_plus_array'].get('accuracy', float('nan')):.2%}|
|宏观量 + 足底阵列 + IMU|{classification['macro_plus_array_plus_imu'].get('accuracy', float('nan')):.2%}|

这个分类器只是验证“是否存在可区分信息”，不是最终的地形识别器。真正的边界更新还需要下一步定义接触约束/不确定性集合，并用外力扰动或多种落足姿态做独立验证。

## 解释与限制

当前阵列是基于 MuJoCo 解算接触点位置和三维接触力的虚拟旁路传感器；它没有把几何单元直接宣称为真实压力阵列，也没有改变原策略的动力学。其价值是先验证信息结构：宏观扳手与 CoP 只保留总力和一阶力矩，阵列保留了多单元载荷分配、切向力分配、接触单元数量等高阶信息，IMU 还提供了足部运动状态。后续若要做严格结论，应把阵列换成带有限感知区域的物理足底模型，并加入可控外部扰动和重复试验。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="sensorized CSV files")
    parser.add_argument("--stride", type=int, default=5, help="keep every Nth row (default: 5)")
    parser.add_argument("--json", type=Path, default=None, help="JSON report path")
    parser.add_argument("--markdown", type=Path, default=None, help="Markdown report path")
    args = parser.parse_args()
    if args.stride <= 0:
        parser.error("stride must be positive")
    result = analyze(args.inputs, args.stride)
    json_path = args.json or args.inputs[0].with_name("sensor_information_report.json")
    markdown_path = args.markdown or args.inputs[0].with_name("sensor_information_report.md")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_markdown_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
