#!/usr/bin/env python3
"""Display four static terrain variants and one H2 Plus in the local MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import time
import xml.etree.ElementTree as ET


LAB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LAB_DIR.parents[1]
DEFAULT_URDF = (
    PROJECT_DIR
    / "experiment"
    / "robot_asset"
    / "h2_plus"
    / "H2_Plus_mujoco_with_footrix_no_dex_hands.urdf"
)
DEFAULT_XML = LAB_DIR / "outputs" / "h2_plus_four_terrains.xml"


TERRAINS = (
    {
        "name": "ordinary_hard",
        "label": "ordinary hard",
        "color": (0.38, 0.38, 0.38, 1),
        "surfaces": ((4.0, 4.0, 0.10, 0.0, 0.0, -0.05),),
    },
    {
        "name": "low_friction_and_difference",
        "label": "friction difference",
        "color": (0.25, 0.43, 0.62, 1),
        "surfaces": (
            (4.0, 2.0, 0.10, 0.0, 1.0, -0.05),
            (4.0, 2.0, 0.10, 0.0, -1.0, -0.05),
        ),
    },
    {
        "name": "mild_deformable_covering_proxy",
        "label": "covering proxy",
        "color": (0.55, 0.35, 0.22, 1),
        "surfaces": ((3.0, 3.0, 0.06, 0.0, 0.0, 0.03),),
    },
    {
        "name": "local_support_edge",
        "label": "support edge",
        "color": (0.42, 0.42, 0.42, 1),
        "surfaces": (
            (2.0, 4.0, 0.10, -1.0, 0.0, -0.05),
            (2.0, 4.0, 0.10, 1.0, 0.0, -0.025),
        ),
    },
)


def _fmt(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.8g}" for value in values)


def _rpy_to_quat(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _vec(node: ET.Element | None, attribute: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if node is None:
        return default
    return tuple(float(value) for value in node.get(attribute, _fmt(default)).split())


def _origin(node: ET.Element | None) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    xyz = _vec(node, "xyz", (0.0, 0.0, 0.0))
    rpy = _vec(node, "rpy", (0.0, 0.0, 0.0))
    return xyz, _rpy_to_quat(rpy)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character == "_" else "_" for character in value)


def _parse_urdf(path: Path) -> tuple[ET.Element, dict[str, ET.Element], dict[str, ET.Element]]:
    root = ET.parse(path).getroot()
    links = {node.attrib["name"]: node for node in root.findall("link")}
    joints = {node.attrib["name"]: node for node in root.findall("joint")}
    children = {joint.find("child").attrib["link"] for joint in joints.values()}
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root link, found {roots}")
    return root, links, joints


def _collect_tree(
    joints: dict[str, ET.Element],
) -> tuple[str, dict[str, list[ET.Element]]]:
    children: dict[str, list[ET.Element]] = {}
    child_names = set()
    for joint in joints.values():
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        children.setdefault(parent, []).append(joint)
        child_names.add(child)
    all_links = {joint.find("child").attrib["link"] for joint in joints.values()}
    all_links.update(joint.find("parent").attrib["link"] for joint in joints.values())
    roots = sorted(all_links - child_names)
    if len(roots) != 1:
        raise ValueError(f"Expected one joint-tree root, found {roots}")
    return roots[0], children


def _add_visuals(
    body: ET.Element,
    link: ET.Element,
    robot_index: int,
    mesh_files: dict[str, str],
) -> None:
    for visual_index, visual in enumerate(link.findall("visual")):
        geometry = visual.find("geometry")
        mesh = geometry.find("mesh") if geometry is not None else None
        if mesh is None or "filename" not in mesh.attrib:
            continue
        filename = mesh.attrib["filename"]
        mesh_name = mesh_files[filename]
        xyz, quat = _origin(visual.find("origin"))
        geom = ET.SubElement(
            body,
            "geom",
            {
                "name": f"visual_{robot_index}_{visual_index}_{_safe_name(filename)}",
                "type": "mesh",
                "mesh": mesh_name,
                "pos": _fmt(xyz),
                "quat": _fmt(quat),
                "contype": "0",
                "conaffinity": "0",
                "rgba": "0.68 0.70 0.74 1",
            },
        )
        material = visual.find("material/color")
        if material is not None:
            geom.set("rgba", material.attrib.get("rgba", "0.68 0.70 0.74 1"))


def _add_link(
    parent_body: ET.Element,
    link_name: str,
    links: dict[str, ET.Element],
    children: dict[str, list[ET.Element]],
    robot_index: int,
    mesh_files: dict[str, str],
) -> None:
    body = ET.SubElement(parent_body, "body", {"name": f"r{robot_index}_{_safe_name(link_name)}"})
    _add_visuals(body, links[link_name], robot_index, mesh_files)
    for joint in children.get(link_name, []):
        child_name = joint.find("child").attrib["link"]
        xyz, quat = _origin(joint.find("origin"))
        child_body_parent = ET.SubElement(
            body,
            "body",
            {
                "name": f"r{robot_index}_{_safe_name(child_name)}",
                "pos": _fmt(xyz),
                "quat": _fmt(quat),
            },
        )
        joint_type = joint.attrib.get("type", "fixed")
        if joint_type in {"revolute", "continuous", "prismatic"}:
            axis = _vec(joint.find("axis"), "xyz", (0.0, 0.0, 1.0))
            joint_attributes = {
                "name": f"r{robot_index}_{_safe_name(joint.attrib['name'])}",
                "type": "hinge" if joint_type != "prismatic" else "slide",
                "axis": _fmt(axis),
            }
            limit = joint.find("limit")
            if limit is not None and joint_type != "continuous":
                joint_attributes["range"] = _fmt(
                    (
                        float(limit.attrib.get("lower", "-3.14159265")),
                        float(limit.attrib.get("upper", "3.14159265")),
                    )
                )
            ET.SubElement(child_body_parent, "joint", joint_attributes)
        _add_visuals(child_body_parent, links[child_name], robot_index, mesh_files)
        for descendant in children.get(child_name, []):
            _add_link(child_body_parent, descendant.find("child").attrib["link"], links, children, robot_index, mesh_files)


def build_mjcf(urdf_path: Path, spacing: float, root_height: float) -> str:
    _, links, joints = _parse_urdf(urdf_path)
    root_name, children = _collect_tree(joints)
    mesh_files: dict[str, str] = {}
    for link in links.values():
        for mesh in link.findall(".//visual/geometry/mesh"):
            filename = mesh.attrib["filename"]
            mesh_files.setdefault(filename, f"mesh_{len(mesh_files):03d}")

    mjcf = ET.Element("mujoco", {"model": "h2_plus_four_static_terrains"})
    compiler = ET.SubElement(mjcf, "compiler", {"meshdir": str(urdf_path.parent), "angle": "radian"})
    compiler.set("autolimits", "true")
    asset = ET.SubElement(mjcf, "asset")
    for filename, mesh_name in mesh_files.items():
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": filename})
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 8", "dir": "0 0 -1", "diffuse": "1 1 1"})
    center_x = (len(TERRAINS) - 1) * spacing / 2
    for index, terrain in enumerate(TERRAINS):
        offset_x = index * spacing
        for surface_index, (sx, sy, sz, px, py, pz) in enumerate(terrain["surfaces"]):
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": f"terrain_{index}_{surface_index}",
                    "type": "box",
                    "size": _fmt((sx / 2, sy / 2, sz / 2)),
                    "pos": _fmt((offset_x + px, py, pz)),
                    "rgba": _fmt(terrain["color"]),
                    "contype": "1",
                    "conaffinity": "1",
                },
            )
        ET.SubElement(
            worldbody,
            "site",
            {
                "name": f"label_{index}_{terrain['name']}",
                "pos": _fmt((offset_x, -2.35, 0.05)),
                "size": "0.01",
                "rgba": "1 1 1 1",
            },
        )
    root_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": f"robot_{TERRAINS[0]['name']}",
            "pos": _fmt((0.0, 0.0, root_height)),
        },
    )
    _add_visuals(root_body, links[root_name], 0, mesh_files)
    for joint in children.get(root_name, []):
        _add_link(root_body, joint.find("child").attrib["link"], links, children, 0, mesh_files)
    ET.SubElement(mjcf, "statistic", {"center": _fmt((center_x, 0.0, 0.8)), "extent": "8"})
    ET.indent(mjcf, space="  ")
    return ET.tostring(mjcf, encoding="unicode")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--spacing", type=float, default=6.0)
    parser.add_argument("--root-height", type=float, default=1.01)
    parser.add_argument("--xml-output", type=Path, default=DEFAULT_XML)
    args = parser.parse_args()
    if not math.isfinite(args.spacing) or args.spacing < 4.5:
        parser.error("--spacing must be finite and at least 4.5 m")
    if not math.isfinite(args.root_height) or args.root_height <= 0:
        parser.error("--root-height must be finite and positive")

    import mujoco
    import mujoco.viewer

    mjcf_text = build_mjcf(args.urdf.resolve(), args.spacing, args.root_height)
    xml_path = args.xml_output.expanduser().resolve()
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(mjcf_text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    print(f"Scene XML: {xml_path}")
    print("Loaded four static terrain variants and one H2 Plus in MuJoCo.")
    print("No mj_step call is made; close the viewer to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = ((len(TERRAINS) - 1) * args.spacing / 2, 0.0, 0.8)
        viewer.cam.distance = 11.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -18
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
