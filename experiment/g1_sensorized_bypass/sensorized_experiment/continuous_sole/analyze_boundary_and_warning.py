"""Build three offline evidence items from timestamped continuous-sole data.

1. Similar ankle wrench/CoP states can have different plantar/IMU states.
2. Plantar/IMU features can warn before an observed fall.
3. Active plantar support updates directional loading margins.

No controller is changed and no walking policy is trained. Reported margins
are sensor-conditioned contact-feasibility estimates, not measured fall loads.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOLE_X = (-0.08, 0.15)
SOLE_Y = (-0.06, 0.06)
NX, NY = 8, 4
DX = (SOLE_X[1] - SOLE_X[0]) / NX
DY = (SOLE_Y[1] - SOLE_Y[0]) / NY
CELL_POS = np.asarray(
    [
        (SOLE_X[0] + (ix + 0.5) * DX, SOLE_Y[0] + (iy + 0.5) * DY)
        for ix in range(NX)
        for iy in range(NY)
    ],
    dtype=float,
)
MU = 1.0
LOAD_HEIGHT_M = 0.5
MIN_STANCE_LOAD_N = 20.0
PERSISTENCE_SAMPLES = 3

DIRECTIONS = {
    "forward": np.asarray([1.0, 0.0]),
    "backward": np.asarray([-1.0, 0.0]),
    "left": np.asarray([0.0, 1.0]),
    "right": np.asarray([0.0, -1.0]),
}


def _v(row: dict[str, str], name: str) -> float:
    return float(row[name])


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        summary_path = path.with_suffix(".json")
        survival_times: dict[int, float] = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            survival_times = {int(item["episode"]): float(item["survival_time_s"]) for item in summary.get("episodes", [])}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(path)
                row["_survival_time_s"] = str(survival_times.get(int(row["episode"]), ""))
                rows.append(row)
    return rows


def cell_forces(row: dict[str, str], foot: str) -> np.ndarray:
    return np.asarray(
        [
            [
                _v(row, f"{foot}_cell_{ix}_{iy}_fx"),
                _v(row, f"{foot}_cell_{ix}_{iy}_fy"),
                _v(row, f"{foot}_cell_{ix}_{iy}_fz"),
            ]
            for ix in range(NX)
            for iy in range(NY)
        ],
        dtype=float,
    )


def ankle_imu(row: dict[str, str], foot: str) -> np.ndarray:
    return np.asarray(
        [
            _v(row, f"{foot}_imu_gx_rad_s"),
            _v(row, f"{foot}_imu_gy_rad_s"),
            _v(row, f"{foot}_imu_gz_rad_s"),
            _v(row, f"{foot}_imu_ax_m_s2"),
            _v(row, f"{foot}_imu_ay_m_s2"),
            _v(row, f"{foot}_imu_az_m_s2"),
        ],
        dtype=float,
    )


def foot_macro(row: dict[str, str], foot: str) -> np.ndarray:
    return np.asarray(
        [_v(row, f"{foot}_ankle_{name}") for name in ("fx", "fy", "fz", "tx", "ty", "tz")]
        + [_v(row, f"{foot}_cop_x_m"), _v(row, f"{foot}_cop_y_m")],
        dtype=float,
    )


def friction_reserve(force: np.ndarray, direction: np.ndarray) -> float:
    """Additional resultant force before the aggregate Coulomb disk is hit."""
    normal = max(float(force[2]), 0.0)
    tangent = force[:2]
    radius = MU * normal
    projection = float(tangent @ direction)
    discriminant = projection * projection + radius * radius - float(tangent @ tangent)
    if discriminant < 0.0:
        return 0.0
    return max(0.0, -projection + float(np.sqrt(discriminant)))


def support_bounds(forces: np.ndarray, use_cell_extent: bool) -> tuple[float, float, float, float] | None:
    normal = np.maximum(forces[:, 2], 0.0)
    total = float(np.sum(normal))
    threshold = max(1.0, 0.01 * total)
    active = normal >= threshold
    if not np.any(active):
        return None
    x, y = CELL_POS[active, 0], CELL_POS[active, 1]
    pad_x = 0.5 * DX if use_cell_extent else 0.0
    pad_y = 0.5 * DY if use_cell_extent else 0.0
    return (
        max(SOLE_X[0], float(np.min(x)) - pad_x),
        min(SOLE_X[1], float(np.max(x)) + pad_x),
        max(SOLE_Y[0], float(np.min(y)) - pad_y),
        min(SOLE_Y[1], float(np.max(y)) + pad_y),
    )


def directional_margins(row: dict[str, str], foot: str) -> dict[str, dict[str, float]] | None:
    forces = cell_forces(row, foot)
    total_normal = float(np.sum(np.maximum(forces[:, 2], 0.0)))
    if total_normal < MIN_STANCE_LOAD_N or not int(float(row[f"{foot}_cop_valid"])):
        return None
    macro_force = np.asarray([_v(row, f"{foot}_ankle_{axis}") for axis in ("fx", "fy", "fz")])
    cop = np.asarray([_v(row, f"{foot}_cop_x_m"), _v(row, f"{foot}_cop_y_m")])
    bounds = {
        "macro": (SOLE_X[0], SOLE_X[1], SOLE_Y[0], SOLE_Y[1]),
        "array_lower": support_bounds(forces, False),
        "array_upper": support_bounds(forces, True),
    }
    output: dict[str, dict[str, float]] = {}
    for model, limits in bounds.items():
        if limits is None:
            continue
        xmin, xmax, ymin, ymax = limits
        edge_distance = {
            "forward": max(xmax - cop[0], 0.0),
            "backward": max(cop[0] - xmin, 0.0),
            "left": max(ymax - cop[1], 0.0),
            "right": max(cop[1] - ymin, 0.0),
        }
        output[model] = {}
        for name, direction in DIRECTIONS.items():
            tipping_limit = total_normal * edge_distance[name] / LOAD_HEIGHT_M
            output[model][name] = min(friction_reserve(macro_force, direction), tipping_limit)
    return output


def plantar_summary(row: dict[str, str]) -> np.ndarray:
    features: list[float] = []
    for foot in ("left", "right"):
        forces = cell_forces(row, foot)
        normal = np.maximum(forces[:, 2], 0.0)
        total = float(np.sum(normal))
        threshold = max(1.0, 0.01 * total)
        active = normal >= threshold
        fractions = normal / max(total, 1e-9)
        utilization = np.linalg.norm(forces[:, :2], axis=1) / np.maximum(MU * normal, 1e-9)
        positive = fractions > 0.0
        features.extend(
            [
                float(np.count_nonzero(active)),
                float(np.max(fractions)) if total > 0.0 else 0.0,
                float(-np.sum(fractions[positive] * np.log(fractions[positive]))) if total > 0.0 else 0.0,
                float(np.sum(fractions[CELL_POS[:, 0] > 0.035])),
                float(np.sum(fractions[CELL_POS[:, 1] > 0.0])),
                float(np.max(utilization[active])) if np.any(active) else 0.0,
            ]
        )
        features.extend(fractions.tolist())
    return np.asarray(features, dtype=float)


def both_ankle_imu(row: dict[str, str]) -> np.ndarray:
    return np.concatenate([ankle_imu(row, "left"), ankle_imu(row, "right")])


def both_feet_macro(row: dict[str, str]) -> np.ndarray:
    return np.concatenate([foot_macro(row, "left"), foot_macro(row, "right")])


def robust_score(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    center = np.median(train, axis=0)
    mad_scale = 1.4826 * np.median(np.abs(train - center), axis=0)
    standard = np.std(train, axis=0)
    scale = np.maximum(mad_scale, 0.1 * standard)
    scale[scale < 1e-6] = 1.0
    z = np.clip(np.abs((values - center) / scale), 0.0, 25.0)
    return np.sqrt(np.mean(z * z, axis=1))


def persistent_starts(times: np.ndarray, scores: np.ndarray, threshold: float) -> list[float]:
    over = scores > threshold
    starts: list[float] = []
    for start in range(0, len(over) - PERSISTENCE_SAMPLES + 1):
        if bool(np.all(over[start : start + PERSISTENCE_SAMPLES])):
            if start == 0 or not over[start - 1]:
                starts.append(float(times[start]))
    return starts


def warning_result(times: np.ndarray, scores: np.ndarray, threshold: float, event_time: float) -> dict[str, object]:
    starts = persistent_starts(times, scores, threshold)
    windows: dict[str, object] = {}
    for window_s in (0.2, 0.5, 1.0):
        candidates = [time for time in starts if event_time - window_s <= time < event_time]
        warning_time = min(candidates) if candidates else None
        windows[f"{window_s:.1f}s"] = {
            "detected": warning_time is not None,
            "warning_time_s": warning_time,
            "lead_time_s": event_time - warning_time if warning_time is not None else 0.0,
        }
    return {
        "threshold": threshold,
        "first_persistent_crossing_s": starts[0] if starts else None,
        "persistent_crossings": len(starts),
        "detections_by_prefall_window": windows,
    }


def anomaly_analysis(rows: list[dict[str, str]], output_dir: Path) -> dict[str, object]:
    results: list[dict[str, object]] = []
    calibration_false_alarms: dict[str, object] = {}
    falls = {(row["scene"], int(row["episode"])) for row in rows if row["outcome"] == "fall"}
    for scene, episode in sorted(falls):
        train_rows = [row for row in rows if row["scene"] == scene and row["outcome"] != "fall"]
        test_rows = [row for row in rows if row["scene"] == scene and int(row["episode"]) == episode]
        macro_train = np.asarray([both_feet_macro(row) for row in train_rows])
        macro_test = np.asarray([both_feet_macro(row) for row in test_rows])
        plantar_train = np.asarray([np.concatenate([both_feet_macro(row), plantar_summary(row)]) for row in train_rows])
        plantar_test = np.asarray([np.concatenate([both_feet_macro(row), plantar_summary(row)]) for row in test_rows])
        enhanced_train = np.asarray([np.concatenate([both_feet_macro(row), plantar_summary(row), both_ankle_imu(row)]) for row in train_rows])
        enhanced_test = np.asarray([np.concatenate([both_feet_macro(row), plantar_summary(row), both_ankle_imu(row)]) for row in test_rows])
        methods = {
            "macro_only": (macro_train, macro_test),
            "macro_plus_plantar": (plantar_train, plantar_test),
            "macro_plus_plantar_plus_imu": (enhanced_train, enhanced_test),
        }
        scored = {
            name: (robust_score(train, train), robust_score(train, test))
            for name, (train, test) in methods.items()
        }
        thresholds = {name: float(np.percentile(scores[0], 99.0)) for name, scores in scored.items()}
        if scene not in calibration_false_alarms:
            normal_groups = sorted({int(row["episode"]) for row in train_rows})
            per_method: dict[str, object] = {}
            for name, (train, _) in methods.items():
                false_episodes = 0
                crossings = 0
                duration_s = 0.0
                for normal_episode in normal_groups:
                    episode_rows = [row for row in train_rows if int(row["episode"]) == normal_episode]
                    if name == "macro_only":
                        episode_values = np.asarray([both_feet_macro(row) for row in episode_rows])
                    elif name == "macro_plus_plantar":
                        episode_values = np.asarray([np.concatenate([both_feet_macro(row), plantar_summary(row)]) for row in episode_rows])
                    else:
                        episode_values = np.asarray([np.concatenate([both_feet_macro(row), plantar_summary(row), both_ankle_imu(row)]) for row in episode_rows])
                    episode_scores = robust_score(train, episode_values)
                    episode_times = np.asarray([_v(row, "time_s") for row in episode_rows])
                    starts = persistent_starts(episode_times, episode_scores, thresholds[name])
                    false_episodes += int(bool(starts))
                    crossings += len(starts)
                    duration_s += float(episode_times[-1] - episode_times[0] + 0.02)
                per_method[name] = {
                    "normal_episodes": len(normal_groups),
                    "episodes_with_false_alarm": false_episodes,
                    "episode_false_alarm_rate": false_episodes / max(len(normal_groups), 1),
                    "persistent_false_alarms_per_hour": crossings / max(duration_s / 3600.0, 1e-9),
                }
            calibration_false_alarms[scene] = per_method
        times = np.asarray([_v(row, "time_s") for row in test_rows])
        event_time = float(test_rows[0]["_survival_time_s"]) if test_rows[0]["_survival_time_s"] else float(times[-1] + np.median(np.diff(times)))
        results.append(
            {
                "scene": scene,
                "episode": episode,
                "event_time_s": event_time,
                "calibration_rows": len(train_rows),
                "threshold_percentile": 99.0,
                "persistence_samples": PERSISTENCE_SAMPLES,
                **{
                    name: warning_result(times, test_scores, thresholds[name], event_time)
                    for name, (_, test_scores) in scored.items()
                },
            }
        )
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        labels = {"macro_only": "ankle wrench + CoP", "macro_plus_plantar": "+ plantar", "macro_plus_plantar_plus_imu": "+ plantar + ankle IMU"}
        for color_index, (name, (_, test_scores)) in enumerate(scored.items()):
            ax.plot(times, test_scores, label=labels[name], linewidth=1.25)
            ax.axhline(thresholds[name], color=f"C{color_index}", linestyle="--", alpha=0.55)
        ax.axvline(event_time, color="black", linestyle=":", label="fall")
        ax.set(xlabel="Time (s)", ylabel="Robust anomaly score", title=f"{scene} episode {episode}: offline warning")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"warning_{scene}_episode_{episode}.png", dpi=180)
        plt.close(fig)
    method_names = ("macro_only", "macro_plus_plantar", "macro_plus_plantar_plus_imu")
    aggregate: dict[str, object] = {}
    for method in method_names:
        aggregate[method] = {}
        for window in ("0.2s", "0.5s", "1.0s"):
            detections = [bool(event[method]["detections_by_prefall_window"][window]["detected"]) for event in results]
            leads = [float(event[method]["detections_by_prefall_window"][window]["lead_time_s"]) for event in results if event[method]["detections_by_prefall_window"][window]["detected"]]
            aggregate[method][window] = {
                "detected_events": int(sum(detections)),
                "detection_rate": float(np.mean(detections)) if detections else 0.0,
                "median_lead_time_s_when_detected": float(np.median(leads)) if leads else 0.0,
            }
    return {"fall_events": len(results), "aggregate": aggregate, "calibration_false_alarms": calibration_false_alarms, "events": results, "limitation": "Thresholds and false-alarm rates use the same non-fall calibration episodes, so they are preliminary. This is an offline warning study, not a failure-rate estimate or a closed-loop result."}


def _method_values(rows: list[dict[str, str]], method: str) -> np.ndarray:
    if method == "macro_only":
        return np.asarray([both_feet_macro(row) for row in rows])
    if method == "macro_plus_plantar":
        return np.asarray([np.concatenate([both_feet_macro(row), plantar_summary(row)]) for row in rows])
    return np.asarray([np.concatenate([both_feet_macro(row), plantar_summary(row), both_ankle_imu(row)]) for row in rows])


def threshold_sweep(rows: list[dict[str, str]], output_dir: Path) -> dict[str, object]:
    percentiles = (95.0, 97.5, 99.0, 99.5, 99.9)
    methods = ("macro_only", "macro_plus_plantar", "macro_plus_plantar_plus_imu")
    output: dict[str, object] = {}
    fall_scenes = sorted({row["scene"] for row in rows if row["outcome"] == "fall"})
    for scene in fall_scenes:
        normal_groups = sorted({int(row["episode"]) for row in rows if row["scene"] == scene and row["outcome"] != "fall"})
        fall_groups = sorted({int(row["episode"]) for row in rows if row["scene"] == scene and row["outcome"] == "fall"})
        train_rows = [row for row in rows if row["scene"] == scene and row["outcome"] != "fall"]
        scene_result: dict[str, object] = {}
        for method in methods:
            train_values = _method_values(train_rows, method)
            train_scores = robust_score(train_values, train_values)
            method_result: dict[str, object] = {}
            for percentile in percentiles:
                threshold = float(np.percentile(train_scores, percentile))
                false_episodes = 0
                for episode in normal_groups:
                    episode_rows = [row for row in train_rows if int(row["episode"]) == episode]
                    scores = robust_score(train_values, _method_values(episode_rows, method))
                    times = np.asarray([_v(row, "time_s") for row in episode_rows])
                    false_episodes += int(bool(persistent_starts(times, scores, threshold)))
                detections = {"0.2s": [], "0.5s": [], "1.0s": []}
                leads = {"0.2s": [], "0.5s": [], "1.0s": []}
                for episode in fall_groups:
                    episode_rows = [row for row in rows if row["scene"] == scene and int(row["episode"]) == episode]
                    scores = robust_score(train_values, _method_values(episode_rows, method))
                    times = np.asarray([_v(row, "time_s") for row in episode_rows])
                    event_time = float(episode_rows[0]["_survival_time_s"]) if episode_rows[0]["_survival_time_s"] else float(times[-1] + np.median(np.diff(times)))
                    starts = persistent_starts(times, scores, threshold)
                    for window_s in (0.2, 0.5, 1.0):
                        key = f"{window_s:.1f}s"
                        candidates = [time for time in starts if event_time - window_s <= time < event_time]
                        detections[key].append(bool(candidates))
                        if candidates:
                            leads[key].append(event_time - min(candidates))
                method_result[f"p{percentile:g}"] = {
                    "threshold": threshold,
                    "normal_episode_false_alarm_rate": false_episodes / max(len(normal_groups), 1),
                    **{
                        f"detection_rate_{key}": float(np.mean(values)) if values else 0.0
                        for key, values in detections.items()
                    },
                    "median_lead_time_0.5s": float(np.median(leads["0.5s"])) if leads["0.5s"] else 0.0,
                }
            scene_result[method] = method_result
        output[scene] = {"normal_episodes": len(normal_groups), "fall_episodes": len(fall_groups), "methods": scene_result}
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        plot_labels = {"macro_only": "ankle wrench + CoP", "macro_plus_plantar": "+ plantar", "macro_plus_plantar_plus_imu": "+ plantar + ankle IMU"}
        for method in methods:
            points = list(scene_result[method].items())
            xs = [float(result["normal_episode_false_alarm_rate"]) for _, result in points]
            ys = [float(result["detection_rate_0.5s"]) for _, result in points]
            ax.plot(xs, ys, marker="o", label=plot_labels[method])
            for (percentile, _), x_value, y_value in zip(points, xs, ys):
                ax.annotate(percentile, (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel="Observed normal-episode false-alarm rate", ylabel="Fall detection rate in final 0.5 s", title=f"{scene}: warning trade-off")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / f"warning_tradeoff_{scene}.png", dpi=180)
        plt.close(fig)
    return output


def boundary_analysis(rows: list[dict[str, str]], output_dir: Path, stride: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["scene"], int(row["episode"])), []).append(row)
    for group_rows in grouped.values():
        for row in group_rows[::stride]:
            for foot in ("left", "right"):
                margins = directional_margins(row, foot)
                if margins is None:
                    continue
                record: dict[str, object] = {"scene": row["scene"], "episode": int(row["episode"]), "time_s": _v(row, "time_s"), "foot": foot}
                for name, component in zip(("fx", "fy", "fz", "tx", "ty", "tz", "cop_x", "cop_y"), foot_macro(row, foot)):
                    record[f"macro_input_{name}"] = float(component)
                imu = ankle_imu(row, foot)
                record["imu_gyro_norm_rad_s"] = float(np.linalg.norm(imu[:3]))
                record["imu_specific_force_norm_m_s2"] = float(np.linalg.norm(imu[3:]))
                for model, directional in margins.items():
                    for name, margin in directional.items():
                        record[f"{model}_{name}_n"] = margin
                records.append(record)
    directions: dict[str, object] = {}
    for direction in DIRECTIONS:
        macro = np.asarray([float(record[f"macro_{direction}_n"]) for record in records])
        lower = np.asarray([float(record[f"array_lower_{direction}_n"]) for record in records])
        upper = np.asarray([float(record[f"array_upper_{direction}_n"]) for record in records])
        directions[direction] = {
            "macro_median_n": float(np.median(macro)),
            "array_lower_median_n": float(np.median(lower)),
            "array_upper_median_n": float(np.median(upper)),
            "array_upper_to_macro_ratio_median": float(np.median(upper / np.maximum(macro, 1e-9))),
            "paired_reduction_median_n": float(np.median(macro - upper)),
            "paired_reduction_90pct_n": float(np.percentile(macro - upper, 90.0)),
            "fraction_array_upper_at_least_20pct_smaller": float(np.mean(upper < 0.8 * macro)),
        }
    labels = list(DIRECTIONS)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].bar(x, [100.0 * directions[k]["fraction_array_upper_at_least_20pct_smaller"] for k in labels])
    axes[0].set_xticks(x, labels)
    axes[0].set(ylabel="Stance samples (%)", title="Array margin at least 20% smaller")
    axes[1].bar(x, [directions[k]["paired_reduction_90pct_n"] for k in labels], color="C1")
    axes[1].set_xticks(x, labels)
    axes[1].set(ylabel="Macro minus array margin (N)", title="90th-percentile correction")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "loading_margin_summary.png", dpi=180)
    plt.close(fig)
    report = {
        "stance_foot_samples": len(records),
        "sample_stride": stride,
        "assumptions": {"friction_coefficient": MU, "external_load_height_m": LOAD_HEIGHT_M, "normal_load_frozen": True, "minimum_stance_load_n": MIN_STANCE_LOAD_N},
        "directions": directions,
        "array_interval_definition": "lower uses active cell centers; upper uses full extents of active integration cells; true within-cell contact support is unresolved by the present CSV.",
        "scope": "single-foot contact loading margin, not whole-robot fall stability",
    }
    return report, records


def same_macro_boundary_pairs(records: list[dict[str, object]], output_dir: Path, tolerance: float = 0.25) -> dict[str, object]:
    macro_keys = [f"macro_input_{name}" for name in ("fx", "fy", "fz", "tx", "ty", "tz", "cop_x", "cop_y")]
    margin_keys = [f"array_upper_{name}_n" for name in DIRECTIONS]
    macro_margin_keys = [f"macro_{name}_n" for name in DIRECTIONS]
    macro = np.asarray([[float(record[key]) for key in macro_keys] for record in records])
    margins = np.asarray([[float(record[key]) for key in margin_keys] for record in records])
    macro_margins = np.asarray([[float(record[key]) for key in macro_margin_keys] for record in records])
    macro_scale = np.std(macro, axis=0)
    macro_scale[macro_scale < 1e-6] = 1.0
    margin_scale = np.std(margins, axis=0)
    margin_scale[margin_scale < 1.0] = 1.0
    normalized_macro = (macro - np.mean(macro, axis=0)) / macro_scale
    pairs: list[dict[str, object]] = []
    for i, record in enumerate(records):
        candidates = [
            j for j, other in enumerate(records)
            if other["scene"] == record["scene"]
            and other["foot"] == record["foot"]
            and other["episode"] != record["episode"]
        ]
        if not candidates:
            continue
        distances = np.sqrt(np.mean((normalized_macro[candidates] - normalized_macro[i]) ** 2, axis=1))
        j = candidates[int(np.argmin(distances))]
        macro_distance = float(np.min(distances))
        if macro_distance > tolerance:
            continue
        array_distance = float(np.sqrt(np.mean(((margins[i] - margins[j]) / margin_scale) ** 2)))
        macro_margin_distance = float(np.sqrt(np.mean(((macro_margins[i] - macro_margins[j]) / margin_scale) ** 2)))
        array_abs_difference = np.abs(margins[i] - margins[j])
        macro_abs_difference = np.abs(macro_margins[i] - macro_margins[j])
        pairs.append(
            {
                "scene": record["scene"],
                "foot": record["foot"],
                "episode_a": record["episode"],
                "time_a_s": record["time_s"],
                "episode_b": records[j]["episode"],
                "time_b_s": records[j]["time_s"],
                "macro_state_distance": macro_distance,
                "macro_margin_distance": macro_margin_distance,
                "array_margin_distance": array_distance,
                "array_margin_max_abs_difference_n": float(np.max(array_abs_difference)),
                "macro_margin_max_abs_difference_n": float(np.max(macro_abs_difference)),
                "array_margins_a_n": {name: float(margins[i, k]) for k, name in enumerate(DIRECTIONS)},
                "array_margins_b_n": {name: float(margins[j, k]) for k, name in enumerate(DIRECTIONS)},
                "imu_gyro_norm_a_rad_s": record["imu_gyro_norm_rad_s"],
                "imu_gyro_norm_b_rad_s": records[j]["imu_gyro_norm_rad_s"],
            }
        )
    if not pairs:
        return {"pairs": 0, "macro_tolerance": tolerance}
    array_distances = np.asarray([float(pair["array_margin_distance"]) for pair in pairs])
    macro_margin_distances = np.asarray([float(pair["macro_margin_distance"]) for pair in pairs])
    top_pairs = sorted(pairs, key=lambda pair: float(pair["array_margin_max_abs_difference_n"]), reverse=True)[:10]
    top = top_pairs[0]
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    labels = list(DIRECTIONS)
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, [top["array_margins_a_n"][name] for name in labels], width, label=f"episode {top['episode_a']}, t={top['time_a_s']:.2f}s")
    ax.bar(x + width / 2, [top["array_margins_b_n"][name] for name in labels], width, label=f"episode {top['episode_b']}, t={top['time_b_s']:.2f}s")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Array-conditioned loading margin (N)")
    ax.set_title(f"Similar macro state (distance {top['macro_state_distance']:.3f}), different boundary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "same_macro_different_boundary.png", dpi=180)
    plt.close(fig)
    return {
        "pairs": len(pairs),
        "macro_tolerance": tolerance,
        "macro_margin_distance_median": float(np.median(macro_margin_distances)),
        "array_margin_distance_median": float(np.median(array_distances)),
        "fraction_array_margin_distance_gt_0.5": float(np.mean(array_distances > 0.5)),
        "median_array_margin_max_abs_difference_n": float(np.median([pair["array_margin_max_abs_difference_n"] for pair in pairs])),
        "fraction_array_margin_max_abs_difference_gt_20n": float(np.mean([pair["array_margin_max_abs_difference_n"] > 20.0 for pair in pairs])),
        "examples": top_pairs,
    }


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.csv_files)
    warning = anomaly_analysis(rows, figure_dir)
    warning["threshold_sweep"] = threshold_sweep(rows, figure_dir)
    boundary, records = boundary_analysis(rows, figure_dir, args.stride)
    boundary["same_macro_different_boundary"] = same_macro_boundary_pairs(records, figure_dir)
    report = {
        "inputs": [str(path) for path in args.csv_files],
        "evidence_1_extra_information": "See same_macro_different_local.json; actual data contain macro-near pairs with distinct plantar/IMU observations.",
        "evidence_2_early_warning": warning,
        "evidence_3_directional_loading_margin": boundary,
    }
    report_path = args.output_dir / "three_claims_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_records(args.output_dir / "directional_loading_margins.csv", records)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
