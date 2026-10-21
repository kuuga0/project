"""Render the four MuJoCo continuous-support scenes as one presentation figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import mujoco
import numpy as np

from foot_contact_wrench.mujoco_io.bench import initialize_free_foot, load_bench_model
from foot_contact_wrench.mujoco_io.partial_contact import (
    build_partial_contact_xml,
    set_support_scenario,
)
from foot_contact_wrench.utils.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "bench" / "stage_b_partial_contact.yml"
OUTPUT_DIR = ROOT / "outputs" / "stage_b_partial_contact"
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
    config = load_yaml(CONFIG_PATH)
    base_config = load_yaml(ROOT / config["base_config"])
    model_xml = OUTPUT_DIR / "generated_partial_contact_bench.xml"
    if not model_xml.exists():
        build_partial_contact_xml(
            ROOT / base_config["model_xml"],
            model_xml,
            config["scenarios"],
            support_geom_prefix=config["support_geom_prefix"],
            support_height_m=float(config["support_height_m"]),
            support_friction=config["support_friction"],
        )
    model, data = load_bench_model(str(model_xml))
    model.opt.gravity[:] = np.asarray(base_config["gravity_mps2"], dtype=float)
    body_id = model.body("foot").id
    # The default MuJoCo offscreen framebuffer is 640 pixels wide.
    renderer = mujoco.Renderer(model, height=420, width=620)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.02]
    camera.distance = 0.42
    camera.azimuth = 135.0
    camera.elevation = -25.0

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)
    axes_flat = axes.ravel()
    for axis, scenario in zip(axes_flat, config["scenarios"]):
        initialize_free_foot(
            model,
            data,
            base_config["initial_position_m"],
            base_config["initial_linear_velocity_mps"],
            base_config["initial_angular_velocity_radps"],
        )
        set_support_scenario(
            model,
            data,
            scenario["name"],
            config["support_geom_prefix"],
        )
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        image = renderer.render()
        axis.imshow(image)
        axis.set_title(scenario["label_zh"], fontsize=13, fontweight="bold")
        axis.set_axis_off()

    fig.suptitle("MuJoCo 单足连续部分支撑模型：四种足地接触条件", fontsize=18, fontweight="bold")
    output_path = OUTPUT_DIR / "display_mujoco_contact_scenes.png"
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    renderer.close()
    print(f"已保存 MuJoCo 场景总图：{output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
