"""Build a read-only copy of the G1 scenes with one continuous sole per foot.

The downloaded candidate remains untouched.  The generated model disables the
original ankle-roll collision geoms and adds a thin rectangular sole plate
whose footprint is the complete sensing area.  The later logger integrates
the contact traction quadrature from this sole into a grid; it does not treat
each grid cell as a MuJoCo geom.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
CANDIDATE_ROOT = PROJECT_ROOT / "external" / "candidates" / "humanoid-sim-eval-learning"
BASE_MODEL = CANDIDATE_ROOT / "assets" / "g1" / "g1_12dof.xml"
BASE_SCENES = CANDIDATE_ROOT / "scenes"
MODEL_DIR = HERE / "model"
SCENE_DIR = HERE / "scenes"
MODEL_PATH = MODEL_DIR / "g1_12dof_continuous.xml"

FOOT_CONFIG = {
    "left": {
        "body": "left_ankle_roll_link",
        "collision_bodies": ("left_ankle_pitch_link", "left_ankle_roll_link"),
        "sole_name": "left_continuous_sole",
    },
    "right": {
        "body": "right_ankle_roll_link",
        "collision_bodies": ("right_ankle_pitch_link", "right_ankle_roll_link"),
        "sole_name": "right_continuous_sole",
    },
}


def _format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def build_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(BASE_MODEL)
    root = tree.getroot()

    mesh_root = root.find("asset")
    if mesh_root is None:
        raise RuntimeError("G1 model has no asset section")
    mesh_dir = BASE_MODEL.parent / "meshes"
    for mesh in mesh_root.findall("mesh"):
        file_name = mesh.get("file")
        if file_name:
            mesh.set("file", str(Path("..") / ".." / ".." / "external" / "candidates" / "humanoid-sim-eval-learning" / "assets" / "g1" / "meshes" / Path(file_name).name).replace("\\", "/"))

    for foot, config in FOOT_CONFIG.items():
        for body_name in config["collision_bodies"]:
            body = root.find(f".//body[@name='{body_name}']")
            if body is None:
                raise RuntimeError(f"missing body {body_name}")
            # Disable all original ankle/foot collision geoms.  The appended
            # sole plate is the only collision surface for this foot.
            for geom in body.findall("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

        body = root.find(f".//body[@name='{config['body']}']")
        if body is None:
            raise RuntimeError(f"missing body {config['body']}")

        sole = ET.Element(
            "geom",
            {
                "name": config["sole_name"],
                "type": "box",
                # x: [-0.08, 0.15], y: [-0.06, 0.06], bottom z: -0.035.
                "pos": _format_vec((0.035, 0.0, -0.030)),
                "size": _format_vec((0.115, 0.060, 0.005)),
                "contype": "1",
                "conaffinity": "1",
                "density": "0",
                "rgba": "0.12 0.12 0.12 1",
                # Keep the candidate's default contact friction explicit in
                # the generated experiment model.
                "friction": "1 0.005 0.0001",
            },
        )
        body.append(sole)

    ET.indent(tree, space="  ")
    tree.write(MODEL_PATH, encoding="utf-8", xml_declaration=False)
    return MODEL_PATH


def build_scenes() -> list[Path]:
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for source in sorted(BASE_SCENES.glob("*.xml")):
        text = source.read_text(encoding="utf-8")
        old_include = '../assets/g1/g1_12dof.xml'
        if old_include not in text:
            raise RuntimeError(f"cannot find G1 include in {source}")
        text = text.replace(old_include, '../model/g1_12dof_continuous.xml')
        destination = SCENE_DIR / source.name
        destination.write_text(text, encoding="utf-8")
        outputs.append(destination)

    height_field = BASE_SCENES / "height_field.png"
    if height_field.exists():
        shutil.copy2(height_field, SCENE_DIR / height_field.name)
    return outputs


def main() -> None:
    model = build_model()
    scenes = build_scenes()
    print(f"wrote {model}")
    print("wrote scenes:")
    for scene in scenes:
        print(f"  {scene}")


if __name__ == "__main__":
    main()
