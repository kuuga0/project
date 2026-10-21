#!/usr/bin/env python3
"""Load H2 Plus in Isaac Lab and collect a compact joint/root/contact trajectory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import traceback
import xml.etree.ElementTree as ET

LAB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LAB_DIR.parents[1]
DEFAULT_URDF = PROJECT_DIR / "experiment" / "robot_asset" / "h2_plus" / "H2_Plus_with_footrix_no_dex_hands.urdf"
DEFAULT_OUTPUT_ROOT = LAB_DIR / "outputs" / "h2_plus_runs"
EXPECTED_REVOLUTE_JOINTS = 31
DEX_HAND_NAME = re.compile(r"finger|thumb|index|middle|ring|pinky|sharpa|_mcp|_pip|_dip|elastomer", re.I)


def inspect_urdf(path: Path) -> dict:
    """Validate the selected URDF and resolve every referenced local mesh."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF not found: {path}")
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected a URDF <robot> root, got <{root.tag}>")

    links = [item.get("name", "") for item in root.findall("link")]
    if len(links) != len(set(links)):
        raise ValueError("URDF contains duplicate link names")
    link_set = set(links)

    joints = []
    child_links = []
    for node in root.findall("joint"):
        name, kind = node.get("name", ""), node.get("type", "")
        parent = node.find("parent")
        child = node.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {name!r} is missing parent or child")
        parent_name, child_name = parent.get("link", ""), child.get("link", "")
        if parent_name not in link_set or child_name not in link_set:
            raise ValueError(f"Joint {name!r} references an unknown link")
        child_links.append(child_name)
        if kind == "revolute":
            limit = node.find("limit")
            if limit is None or limit.get("effort") is None:
                raise ValueError(f"Revolute joint {name!r} has no effort limit")
            joints.append({"name": name, "effort_limit": float(limit.get("effort"))})

    if len(child_links) != len(set(child_links)):
        raise ValueError("URDF gives a link more than one parent joint")
    roots = sorted(link_set - set(child_links))
    if len(roots) != 1:
        raise ValueError(f"Expected one root link, found {roots}")
    revolute_names = [item["name"] for item in joints]
    dex_names = [name for name in links + [item.get("name", "") for item in root.findall("joint")]
                 if DEX_HAND_NAME.search(name)]
    if dex_names:
        raise ValueError("URDF contains dexterous-hand links or joints: " + ", ".join(dex_names[:8]))
    if len(joints) != EXPECTED_REVOLUTE_JOINTS:
        raise ValueError(f"Expected {EXPECTED_REVOLUTE_JOINTS} revolute joints, found {len(joints)}")

    required_feet = {"left_ankle_pitch_link", "right_ankle_pitch_link"}
    if not required_feet.issubset(link_set):
        raise ValueError(f"Missing foot links: {', '.join(sorted(required_feet - link_set))}")

    meshes = []
    for node in root.findall(".//mesh"):
        filename = node.get("filename")
        if not filename:
            continue
        if filename.startswith(("package://", "http://", "https://")):
            raise ValueError(f"Mesh URI is not a local relative path: {filename}")
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = path.parent / mesh_path
        mesh_path = mesh_path.resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing mesh: {filename} -> {mesh_path}")
        meshes.append((filename, mesh_path))

    foot_mesh_names = [name.lower() for name, _ in meshes if "pitch_foot_visual" in name.lower()]
    for side in ("left", "right"):
        if not any(f"{side}_pitch_foot_visual" in name for name in foot_mesh_names):
            raise ValueError(f"Missing {side} FOOTRIX foot mesh reference")

    return {
        "path": path,
        "link_names": links,
        "root_link": roots[0],
        "joint_names": revolute_names,
        "joint_effort_limits": {item["name"]: item["effort_limit"] for item in joints},
        "link_count": len(links),
        "mesh_count": len(meshes),
        "meshes": meshes,
        "imu_links": [name for name in links if "imu" in name.lower()],
        "has_explicit_floating_joint": any(node.get("type") == "floating" for node in root.findall("joint")),
    }


def make_parser(AppLauncher=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "smoke", "collect"), default="check")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--usd-dir", type=Path, default=LAB_DIR / "outputs" / "h2_plus_usd")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration for collect mode")
    parser.add_argument("--physics-hz", type=float, default=240.0)
    parser.add_argument("--record-hz", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=240, help="Physics steps for smoke mode")
    parser.add_argument("--root-height", type=float, default=1.01, help="Initial pelvis height for the zero-joint pose [m]")
    parser.add_argument("--position-stiffness", type=float, default=80.0, help="Provisional position-hold gain")
    parser.add_argument("--position-damping", type=float, default=8.0, help="Provisional position-hold damping")
    parser.add_argument("--perturb-joint", default="", help="Optional joint name for a sine position command")
    parser.add_argument("--perturb-amplitude", type=float, default=0.0, help="Position-command amplitude [rad]")
    parser.add_argument("--perturb-frequency", type=float, default=0.5, help="Position-command frequency [Hz]")
    if AppLauncher is not None:
        AppLauncher.add_app_launcher_args(parser)
        parser.set_defaults(device="cpu")
    return parser


def _to_json(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def _single_vector(value):
    """Extract one 3D vector from a one-environment, one-body, one-filter sensor tensor."""
    value = _to_json(value)
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or len(value) != 3:
        return None
    if any(item is None or not isinstance(item, (int, float)) for item in value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_to_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = DEFAULT_OUTPUT_ROOT / stamp
    suffix = 0
    candidate = root
    while candidate.exists():
        suffix += 1
        candidate = DEFAULT_OUTPUT_ROOT / f"{stamp}_{suffix:02d}"
    return candidate


def _contact_sensor(ContactSensor, ContactSensorCfg, robot_prim: str, side: str):
    return ContactSensor(ContactSensorCfg(
        prim_path=f"{robot_prim}/.*{side}_ankle_pitch_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["/World/Ground"],
        track_contact_points=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
    ))


def main(argv=None) -> int:
    check_args, _ = make_parser().parse_known_args(argv)
    if check_args.mode == "check":
        audit = inspect_urdf(check_args.urdf)
        print(f"URDF check passed: root={audit['root_link']}, links={audit['link_count']}, "
              f"revolute_joints={len(audit['joint_names'])}, meshes={audit['mesh_count']}")
        print("URDF:", audit["path"])
        if audit["imu_links"]:
            print("IMU-like links:", ", ".join(audit["imu_links"]))
        else:
            print("IMU: no link/frame in URDF; no IMU reading will be synthesized")
        return 0

    from isaaclab.app import AppLauncher

    args = make_parser(AppLauncher).parse_args(argv)
    if args.duration <= 0 or args.physics_hz <= 0 or args.record_hz <= 0:
        raise ValueError("duration, physics-hz, and record-hz must be positive")
    if args.steps < 1 or args.root_height <= 0:
        raise ValueError("steps and root-height must be positive")
    if args.position_stiffness < 0 or args.position_damping < 0 or args.perturb_frequency <= 0:
        raise ValueError("position gains must be nonnegative and perturb-frequency positive")
    if args.perturb_amplitude and not args.perturb_joint:
        raise ValueError("Set --perturb-joint when --perturb-amplitude is nonzero")
    record_stride = round(args.physics_hz / args.record_hz)
    if record_stride < 1 or not math.isclose(args.physics_hz / record_stride, args.record_hz, rel_tol=1e-8):
        raise ValueError("record-hz must divide physics-hz evenly")
    actual_record_hz = args.physics_hz / record_stride
    audit = inspect_urdf(args.urdf)

    simulation_app = AppLauncher(args).app
    output = None
    metadata = None
    started = time.monotonic()
    try:
        import isaaclab
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from isaaclab.utils.version import get_isaac_sim_version

        usd_dir = args.usd_dir.expanduser().resolve()
        usd_dir.mkdir(parents=True, exist_ok=True)
        converter = UrdfConverter(UrdfConverterCfg(
            asset_path=str(audit["path"]),
            usd_dir=str(usd_dir),
            usd_file_name="H2_Plus_sensorized.usd",
            force_usd_conversion=True,
            fix_base=False,
            merge_fixed_joints=False,
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="none"),
        ))
        usd_path = Path(converter.usd_path)

        sim_dt = 1.0 / args.physics_hz
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=sim_dt, device=args.device))
        ground = sim_utils.CuboidCfg(
            size=(8.0, 8.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
        )
        ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.05))
        light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)

        robot_prim = "/World/H2Plus"
        robot = Articulation(ArticulationCfg(
            prim_path=robot_prim,
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path), activate_contact_sensors=True),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, args.root_height)),
            actuators={"body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=audit["joint_effort_limits"],
                stiffness=args.position_stiffness,
                damping=args.position_damping,
            )},
        ))
        left_contact = _contact_sensor(ContactSensor, ContactSensorCfg, robot_prim, "left")
        right_contact = _contact_sensor(ContactSensor, ContactSensorCfg, robot_prim, "right")
        sim.set_camera_view(eye=(2.5, 2.5, 1.8), target=(0.0, 0.0, 0.8))
        sim.reset()
        robot.update(sim_dt)
        left_contact.update(sim_dt)
        right_contact.update(sim_dt)

        imported_names = list(robot.joint_names)
        expected_names = set(audit["joint_names"])
        if set(imported_names) != expected_names:
            missing = sorted(expected_names - set(imported_names))
            unexpected = sorted(set(imported_names) - expected_names)
            raise RuntimeError(f"USD joint-name mismatch; missing={missing}, unexpected={unexpected}")
        if robot.is_fixed_base:
            raise RuntimeError("URDF import produced a fixed-base articulation; expected floating base")
        for side, sensor in (("left", left_contact), ("right", right_contact)):
            if sensor.num_bodies != 1 or len(sensor.body_names) != 1:
                raise RuntimeError(f"{side} ContactSensor must resolve to one foot body; found {sensor.body_names}")
            for field in ("net_forces_w", "force_matrix_w", "friction_forces_w", "contact_pos_w"):
                if getattr(sensor.data, field, None) is None:
                    raise RuntimeError(f"{side} ContactSensor does not expose {field}; refusing an incomplete contact record")
            for field in ("force_matrix_w", "friction_forces_w", "contact_pos_w"):
                if getattr(sensor.data, field).shape[2] != 1:
                    raise RuntimeError(f"{side} ContactSensor expected one ground filter shape in {field}")
        if args.perturb_joint and args.perturb_joint not in imported_names:
            raise ValueError(f"Unknown perturb joint: {args.perturb_joint}")

        output = (args.output_dir.expanduser().resolve() if args.output_dir else _default_output_dir())
        output.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema_version": 2,
            "record_schema_version": 1,
            "run_id": datetime.now(timezone.utc).isoformat(),
            "run_status": "running",
            "mode": args.mode,
            "source": {
                "urdf": audit["path"].as_posix(),
                "urdf_sha256": _sha256(audit["path"]),
                "usd": usd_path.as_posix(),
                "root_link": audit["root_link"],
                "mesh_count": audit["mesh_count"],
                "meshes": [
                    {"reference": ref, "path": mesh_path.as_posix(), "sha256": _sha256(mesh_path)}
                    for ref, mesh_path in audit["meshes"]
                ],
                "collector_script_sha256": _sha256(Path(__file__).resolve()),
            },
            "runtime": {
                "isaaclab": getattr(isaaclab, "__version__", "unknown"),
                "isaacsim": str(get_isaac_sim_version()),
                "torch": torch.__version__,
                "device": args.device,
                "headless": bool(args.headless),
                "physics_backend": "PhysX",
            },
            "floating_base": {
                "requested": True,
                "urdf_root_link": audit["root_link"],
                "urdf_explicit_floating_joint": audit["has_explicit_floating_joint"],
                "converter_fix_base": False,
                "runtime_verified": bool(not robot.is_fixed_base),
            },
            "joints": {
                "names_in_articulation_order": imported_names,
                "position_command_unit": "rad",
                "velocity_state_unit": "rad/s",
                "urdf_effort_limits": audit["joint_effort_limits"],
                "position_hold": {
                    "type": "implicit_position_drive",
                    "stiffness": args.position_stiffness,
                    "damping": args.position_damping,
                    "status": "provisional_simulation_parameters_not_hardware_calibration",
                },
                "effort_telemetry_recorded": False,
            },
            "initial_state": {
                "root_position_world_m": [0.0, 0.0, args.root_height],
                "joint_positions": "USD/URDF default joint pose",
            },
            "ground": {"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0, "status": "baseline_simulation_values_not_measured_surface"},
            "sampling": {
                "physics_hz": args.physics_hz,
                "record_hz": actual_record_hz,
                "record_stride_physics_steps": record_stride,
                "requested_duration_s": args.duration if args.mode == "collect" else None,
            },
            "contacts": {
                "sensor": "Isaac Lab ContactSensor",
                "filter": "/World/Ground",
                "world_frame": True,
                "net_and_filtered_force_identity": "normal_only",
                "friction_force_identity": "filtered_friction_force_for_ground_partner",
                "contact_position_identity": "average_world_contact_position_per_foot_ground_pair",
                "max_contact_data_count_per_prim": 64,
                "resolved_filter_shape_count": 1,
                "footrix_status": "not_simulated_as_per-cell_sensor",
            },
            "imu": {
                "links_found_by_name": audit["imu_links"],
                "status": "not_configured_no_verified_mount_transform",
            },
            "excitation": {
                "joint": args.perturb_joint or None,
                "amplitude_rad": args.perturb_amplitude,
                "frequency_hz": args.perturb_frequency if args.perturb_amplitude else None,
            },
            "data": {
                "trajectory": (output / "trajectory.jsonl").as_posix() if args.mode == "collect" else None,
                "fields": [
                    "simulation_time_s", "command_joint_pos_rad", "joint_pos_rad", "joint_vel_rad_s",
                    "root_pos_w_m", "root_quat_wxyz", "root_lin_vel_w_m_s", "root_ang_vel_w_rad_s",
                    "left_foot_net_normal_force_w_n", "right_foot_net_normal_force_w_n",
                    "left_foot_ground_normal_force_w_n", "right_foot_ground_normal_force_w_n",
                    "left_foot_ground_friction_force_w_n", "right_foot_ground_friction_force_w_n",
                    "left_foot_average_ground_contact_pos_w_m", "right_foot_average_ground_contact_pos_w_m",
                ] if args.mode == "collect" else [],
            },
        }
        _write_json(output / "metadata.json", metadata)

        target = robot.data.joint_pos.clone()
        perturb_index = imported_names.index(args.perturb_joint) if args.perturb_joint else None
        total_steps = args.steps if args.mode == "smoke" else round(args.duration * args.physics_hz)
        completed_steps = 0
        record_count = 0
        trajectory = (output / "trajectory.jsonl").open("x", encoding="utf-8") if args.mode == "collect" else None
        try:
            for step in range(total_steps):
                if not simulation_app.is_running():
                    metadata["run_status"] = "interrupted"
                    break
                sim_time = step * sim_dt
                command = target.clone()
                if perturb_index is not None and args.perturb_amplitude:
                    phase = 2.0 * math.pi * args.perturb_frequency * sim_time
                    command[:, perturb_index] += args.perturb_amplitude * math.sin(phase)
                robot.set_joint_position_target(command)
                robot.write_data_to_sim()
                sim.step(render=not args.headless)
                robot.update(sim_dt)
                left_contact.update(sim_dt)
                right_contact.update(sim_dt)
                completed_steps = step + 1

                if args.mode == "collect" and step % record_stride == 0:
                    row = {
                        "sample_index": record_count,
                        "simulation_time_s": completed_steps * sim_dt,
                        "physics_step": completed_steps,
                        "command_joint_pos_rad": _to_json(command[0]),
                        "joint_pos_rad": _to_json(robot.data.joint_pos[0]),
                        "joint_vel_rad_s": _to_json(robot.data.joint_vel[0]),
                        "root_pos_w_m": _to_json(robot.data.root_pos_w[0]),
                        "root_quat_wxyz": _to_json(robot.data.root_quat_w[0]),
                        "root_lin_vel_w_m_s": _to_json(robot.data.root_lin_vel_w[0]),
                        "root_ang_vel_w_rad_s": _to_json(robot.data.root_ang_vel_w[0]),
                        "left_foot_net_normal_force_w_n": _single_vector(left_contact.data.net_forces_w),
                        "right_foot_net_normal_force_w_n": _single_vector(right_contact.data.net_forces_w),
                        "left_foot_ground_normal_force_w_n": _single_vector(left_contact.data.force_matrix_w),
                        "right_foot_ground_normal_force_w_n": _single_vector(right_contact.data.force_matrix_w),
                        "left_foot_ground_friction_force_w_n": _single_vector(left_contact.data.friction_forces_w),
                        "right_foot_ground_friction_force_w_n": _single_vector(right_contact.data.friction_forces_w),
                        "left_foot_average_ground_contact_pos_w_m": _single_vector(left_contact.data.contact_pos_w),
                        "right_foot_average_ground_contact_pos_w_m": _single_vector(right_contact.data.contact_pos_w),
                    }
                    trajectory.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    trajectory.flush()
                    record_count += 1
        finally:
            if trajectory is not None:
                trajectory.close()

        if metadata["run_status"] == "running":
            metadata["run_status"] = "complete"
        metadata["steps_completed"] = completed_steps
        metadata["records_written"] = record_count
        metadata["simulated_duration_s"] = completed_steps * sim_dt
        metadata["wall_time_s"] = time.monotonic() - started
        _write_json(output / "metadata.json", metadata)
        print(f"USD generated: {usd_path}")
        print(f"Isaac Lab loaded: bodies={robot.num_bodies}, joints={robot.num_joints}")
        print(f"Run status: {metadata['run_status']}; steps={completed_steps}; records={record_count}; output={output}")

        if not args.headless:
            while simulation_app.is_running():
                simulation_app.update()
    except BaseException as error:
        if output is not None and metadata is not None:
            metadata["run_status"] = "failed"
            metadata["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            metadata["wall_time_s"] = time.monotonic() - started
            try:
                _write_json(output / "metadata.json", metadata)
            except Exception:
                pass
        raise
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
