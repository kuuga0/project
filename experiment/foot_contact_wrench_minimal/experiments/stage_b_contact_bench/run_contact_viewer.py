"""打开 MuJoCo 查看器并叠加真值、法向阵列重建值和局部能力范围。"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from foot_contact_wrench.analysis.comparison import support_wrench_samples
from foot_contact_wrench.mujoco_io.bench import (
    contact_wrench_about_body,
    initialize_free_foot,
    load_bench_model,
)
from foot_contact_wrench.utils.config import load_yaml, local_set_from_config
from foot_contact_wrench.visualization.wrench_scene import (
    ACTUAL_FORCE,
    ACTUAL_MOMENT,
    MODEL_FORCE,
    MODEL_MOMENT,
    WrenchSceneOptions,
    add_wrench_cone,
    add_wrench,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "bench" / "stage_b_bench.yml"


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    model, data = load_bench_model(str(ROOT / config["model_xml"]))
    model.opt.timestep = float(config["timestep_s"])
    model.opt.gravity[:] = np.asarray(config["gravity_mps2"], dtype=float)
    local_set = local_set_from_config(config)
    direction_count = int(config.get("viewer", {}).get("cone_direction_count", 32))
    wrench_samples_foot = support_wrench_samples(local_set, direction_count)
    viewer_config = config.get("viewer", {})
    options = WrenchSceneOptions(
        force_scale_m_per_n=float(viewer_config.get("force_scale_m_per_n", 0.003)),
        moment_scale_m_per_nm=float(
            viewer_config.get("moment_scale_m_per_nm", 0.04)
        ),
    )
    realtime_factor = float(viewer_config.get("realtime_factor", 1.0))
    state = {
        "actual": True,
        "model": True,
        "range": True,
        "paused": False,
        "reset": False,
    }

    def print_state() -> None:
        print(
            "可见性："
            f"[1]实际={'开' if state['actual'] else '关'}，"
            f"[2]法向阵列重建={'开' if state['model'] else '关'}，"
            f"[3]局部扳手包络={'开' if state['range'] else '关'}，"
            f"[Space]暂停={'是' if state['paused'] else '否'}"
        )

    def key_callback(keycode: int) -> None:
        if keycode == ord("1"):
            state["actual"] = not state["actual"]
        elif keycode == ord("2"):
            state["model"] = not state["model"]
        elif keycode == ord("3"):
            state["range"] = not state["range"]
        elif keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (ord("R"), ord("r")):
            state["reset"] = True
        else:
            return
        print_state()

    initialize_free_foot(
        model,
        data,
        config["initial_position_m"],
        config["initial_linear_velocity_mps"],
        config["initial_angular_velocity_radps"],
    )
    print("颜色：红/橙=MuJoCo实际合力/合力矩，蓝/青=模型合力/合力矩，绿色箭头束=局部扳手包络")
    print("参考点：所有扳手与范围箭头均相对于足部质心，并从质心出发。")
    print("按键：1/2/3 切换三组显示，Space 暂停，R 重置")
    print(f"局部范围：{len(wrench_samples_foot)} 个完整可行扳手见证形成锥状包络，每个见证显示合力和合力矩方向。")
    print("说明：固定总法向载荷后是 CWC 的有界截面，锥状箭头是可视化表达，不代表集合仍是无界数学锥。")
    print_state()

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=True, show_right_ui=True
    ) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.02]
        viewer.cam.distance = 0.65
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        while viewer.is_running():
            loop_start = time.perf_counter()
            if state["reset"]:
                initialize_free_foot(
                    model,
                    data,
                    config["initial_position_m"],
                    config["initial_linear_velocity_mps"],
                    config["initial_angular_velocity_radps"],
                )
                state["reset"] = False
            if not state["paused"]:
                mujoco.mj_step(model, data)

            sample = contact_wrench_about_body(model, data, "foot")
            body_id = model.body("foot").id
            rotation_world_from_foot = np.asarray(
                data.xmat[body_id], dtype=float
            ).reshape(3, 3)
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                if state["actual"]:
                    add_wrench(
                        viewer.user_scn,
                        sample.com_position_world,
                        sample.wrench_world,
                        options,
                        ACTUAL_FORCE,
                        ACTUAL_MOMENT,
                        "actual",
                    )
                if state["model"]:
                    add_wrench(
                        viewer.user_scn,
                        sample.com_position_world,
                        sample.pressure_model_wrench_world,
                        options,
                        MODEL_FORCE,
                        MODEL_MOMENT,
                        "normal-array model",
                    )
                if state["range"]:
                    add_wrench_cone(
                        viewer.user_scn,
                        sample.com_position_world,
                        rotation_world_from_foot,
                        wrench_samples_foot,
                        options,
                    )
            viewer.sync()
            target = model.opt.timestep / max(realtime_factor, 1e-6)
            delay = target - (time.perf_counter() - loop_start)
            if delay > 0.0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
