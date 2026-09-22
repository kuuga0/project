"""Generate presentation figures for the continuous-sole project overview.

This module is display-only. It loads the generated MuJoCo model and the
existing result JSON files, then writes PNG figures under outputs. It does
not modify the walking policy, model XML, CSV logs, or controller.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


HERE = Path(__file__).resolve().parent
SCENE_DIR = HERE / "scenes"
OUTPUT_DIR = HERE / "outputs" / "project_overview_figures"


def chinese_font() -> FontProperties | None:
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ):
        if path.exists():
            return FontProperties(fname=str(path))
    return None


CN = chinese_font()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_mujoco_scenes() -> list[Path]:
    import mujoco

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for scene_name in ("flat", "rough", "stairs"):
        scene_path = SCENE_DIR / f"{scene_name}.xml"
        old_cwd = Path.cwd()
        try:
            os.chdir(scene_path.parent)
            model = mujoco.MjModel.from_xml_path(scene_path.name)
        finally:
            os.chdir(old_cwd)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
        camera.distance = 2.35
        camera.azimuth = 135.0
        camera.elevation = -18.0
        camera.lookat[:] = np.array([0.35, 0.0, 0.55])
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        output = OUTPUT_DIR / f"mujoco_{scene_name}.png"
        plt.imsave(output, pixels)
        renderer.close()
        rendered.append(output)
    return rendered


def render_chinese_summary() -> Path:
    report = load_json(HERE / "outputs" / "three_claims_rough20" / "three_claims_report.json")
    integration_payload = load_json(HERE / "outputs" / "uniform_speed_integrated" / "integration_report.json")
    integration = integration_payload.get("files", integration_payload)

    warning = report["evidence_2_early_warning"]
    p975_macro = warning["threshold_sweep"]["rough"]["methods"]["macro_only"]["p97.5"]
    p975_enhanced = warning["threshold_sweep"]["rough"]["methods"]["macro_plus_plantar_plus_imu"]["p97.5"]
    p999_macro = warning["threshold_sweep"]["rough"]["methods"]["macro_only"]["p99.9"]
    p999_enhanced = warning["threshold_sweep"]["rough"]["methods"]["macro_plus_plantar_plus_imu"]["p99.9"]

    directions = report["evidence_3_directional_loading_margin"]["directions"]
    direction_names = ["前向", "后向", "左向", "右向"]
    direction_keys = ["forward", "backward", "left", "right"]
    reduction = [100.0 * directions[key]["fraction_array_upper_at_least_20pct_smaller"] for key in direction_keys]

    scene_names = ["粗糙地形", "斜坡", "楼梯"]
    active_left: list[float] = []
    active_right: list[float] = []
    for item in integration:
        active_left.append(item["feet"]["left"]["mean_nonzero_normal_cells"])
        active_right.append(item["feet"]["right"]["mean_nonzero_normal_cells"])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), dpi=150)
    title_kwargs = {"fontproperties": CN} if CN else {}
    label_kwargs = {"fontproperties": CN} if CN else {}
    width = 0.34

    ax = axes[0, 0]
    x = np.arange(2)
    ax.bar(x - width / 2, [100.0 * p975_macro["detection_rate_0.5s"], 100.0 * p999_macro["detection_rate_0.5s"]], width, label="宏观量：踝部扳手+CoP")
    ax.bar(x + width / 2, [100.0 * p975_enhanced["detection_rate_0.5s"], 100.0 * p999_enhanced["detection_rate_0.5s"]], width, label="足底阵列+IMU")
    ax.set_xticks(x, ["正常阈值 p97.5", "正常阈值 p99.9"], **label_kwargs)
    ax.set_ylim(0, 100)
    ax.set_ylabel("跌倒前0.5 s检出率 (%)", **label_kwargs)
    ax.set_title("预警能力对比（rough，16个跌倒回合）", **title_kwargs)
    ax.legend(prop=CN, fontsize=8) if CN else ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.bar(np.arange(len(direction_names)), reduction, color="#d97706")
    ax.set_xticks(np.arange(len(direction_names)), direction_names, **label_kwargs)
    ax.set_ylim(0, max(reduction) * 1.25)
    ax.set_ylabel("阵列余量至少低20%的样本比例 (%)", **label_kwargs)
    ax.set_title("足底分布对方向性加载余量的修正", **title_kwargs)
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(reduction):
        ax.text(index, value + 1.0, f"{value:.1f}%", ha="center", **label_kwargs)

    ax = axes[1, 0]
    x = np.arange(len(scene_names))
    ax.bar(x - width / 2, active_left, width, label="左脚")
    ax.bar(x + width / 2, active_right, width, label="右脚")
    ax.set_xticks(x, scene_names, **label_kwargs)
    ax.set_ylabel("每帧平均非零法向单元数", **label_kwargs)
    ax.set_title("连续足底积分后的有效单元数量", **title_kwargs)
    ax.legend(prop=CN, fontsize=8) if CN else ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    labels = ["原姿态\n宏观判断", "原姿态\n阵列判断", "姿态修正后\n阵列判断"]
    values = [6.93, 3.25, 6.93]
    bars = ax.bar(np.arange(3), values, color=["#2563eb", "#dc2626", "#16a34a"])
    ax.axhline(5.0, color="black", linestyle="--", linewidth=1.0, label="稳定阈值 5 N")
    ax.set_xticks(np.arange(3), labels, **label_kwargs)
    ax.set_ylabel("最小方向性加载余量 (N)", **label_kwargs)
    ax.set_title("姿态修正反事实：rough episode 12右脚", **title_kwargs)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.2f}", ha="center")

    fig.suptitle("足底分布感知项目结果摘要", **title_kwargs, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output = OUTPUT_DIR / "results_summary_cn.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    outputs = render_mujoco_scenes()
    outputs.append(render_chinese_summary())
    print("generated:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
