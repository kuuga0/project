#!/usr/bin/env python3
"""Read an MJCF with CPU MuJoCo and write a model inventory, without stepping it.

Only the Python standard library is imported until the model has been fingerprinted.
MuJoCo 3.3.7 is the target API. No viewer, Isaac, torch, or third-party file writes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import traceback


COUNTS = (
    "nbody", "njnt", "nu", "nq", "nv", "ngeom", "nsite", "nmesh", "nhfield",
    "ntendon", "nsensor", "nsensordata", "npair", "nexclude", "neq", "nkey",
)
# Raw field names intentionally match mjModel, including compiled defaults/units.
TABLES = {
    "bodies": ("BODY", "nbody", "body_parentid body_rootid body_weldid body_mocapid "
               "body_pos body_quat body_ipos body_iquat body_mass body_inertia"),
    "joints": ("JOINT", "njnt", "jnt_bodyid jnt_type jnt_qposadr jnt_dofadr "
               "jnt_pos jnt_axis jnt_limited jnt_range jnt_actfrclimited "
               "jnt_actfrcrange jnt_stiffness jnt_margin"),
    "actuators": ("ACTUATOR", "nu", "actuator_trntype actuator_trnid actuator_gear "
                  "actuator_ctrllimited actuator_ctrlrange actuator_forcelimited "
                  "actuator_forcerange actuator_actlimited actuator_actrange "
                  "actuator_dyntype actuator_gaintype actuator_biastype "
                  "actuator_dynprm actuator_gainprm actuator_biasprm "
                  "actuator_actadr actuator_actnum actuator_lengthrange actuator_cranklength"),
    "geoms": ("GEOM", "ngeom", "geom_bodyid geom_type geom_dataid geom_group "
              "geom_pos geom_quat geom_size geom_aabb geom_rgba geom_contype "
              "geom_conaffinity geom_condim geom_friction geom_margin geom_gap "
              "geom_solref geom_solimp geom_solmix geom_priority"),
    "sites": ("SITE", "nsite", "site_bodyid site_type site_group site_pos site_quat site_size"),
    "meshes": ("MESH", "nmesh", "mesh_vertadr mesh_vertnum mesh_faceadr mesh_facenum "
               "mesh_scale mesh_pos mesh_quat"),
    "tendons": ("TENDON", "ntendon", "tendon_adr tendon_num tendon_limited tendon_range "
                "tendon_stiffness tendon_damping tendon_length0"),
    "sensors": ("SENSOR", "nsensor", "sensor_type sensor_objtype sensor_objid "
                "sensor_reftype sensor_refid sensor_dim sensor_adr"),
    "contact_pairs": ("PAIR", "npair", "pair_geom1 pair_geom2 pair_dim pair_friction "
                      "pair_margin pair_gap pair_solref pair_solreffriction pair_solimp"),
    "contact_exclusions": ("EXCLUDE", "nexclude", "exclude_signature"),
    "equalities": ("EQUALITY", "neq", "eq_type eq_objtype eq_obj1id eq_obj2id "
                   "eq_active0 eq_data"),
}
FOOT_TERMS = ("foot", "ankle", "toe", "sole", "heel")
END_TERMS = ("hand", "wrist", "palm", "finger", "gripper", "end_effector", "tool")


def plain(value):
    """Convert MuJoCo's NumPy arrays/scalars without importing NumPy ourselves."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def enum_value(mj, enum_class, value):
    number = int(value)
    try:
        name = getattr(mj, enum_class)(number).name
    except ValueError:
        name = None  # Preserve unknown numeric values, never invent enum labels.
    return {"id": number, "name": name}


def reference(mj, model, kind, index):
    index = int(index)
    if index < 0:
        return None
    return {
        "id": index,
        "name": mj.mj_id2name(model, getattr(mj.mjtObj, "mjOBJ_" + kind), index),
    }


def transmission_targets(mj, model, kind, ids):
    """Decode trnid by transmission type; it is not always a joint index."""
    mapping = {
        "mjTRN_JOINT": (("joint", "JOINT", 0),),
        "mjTRN_JOINTINPARENT": (("joint", "JOINT", 0),),
        "mjTRN_TENDON": (("tendon", "TENDON", 0),),
        "mjTRN_SITE": (("site", "SITE", 0), ("reference_site", "SITE", 1)),
        "mjTRN_SLIDERCRANK": (("crank_site", "SITE", 0), ("slider_site", "SITE", 1)),
        "mjTRN_BODY": (("body", "BODY", 0),),
    }
    return {
        label: reference(mj, model, obj, ids[position])
        for label, obj, position in mapping.get(kind, ())
    }


def body_chain(bodies, body_id):
    """Root-to-leaf ancestry; world has no parent in the derived tree."""
    chain = []
    seen = set()
    while body_id is not None:
        if body_id in seen:
            raise ValueError("Cycle in compiled body hierarchy")
        seen.add(body_id)
        body = bodies[body_id]
        chain.append(body_id)
        body_id = None if body["parent"] is None else body["parent"]["id"]
    return chain[::-1]


def structure_hints(inventory):
    matches = []
    bodies = inventory["bodies"]
    for table, body_field in (("bodies", None), ("joints", "jnt_bodyid"),
                              ("geoms", "geom_bodyid"), ("sites", "site_bodyid")):
        for item in inventory[table]:
            name = (item["name"] or "").lower()
            foot = [term for term in FOOT_TERMS if term in name]
            end = [term for term in END_TERMS if term in name]
            if foot or end:
                body_id = item["id"] if body_field is None else item["fields"][body_field]
                matches.append({
                    "table": table, "id": item["id"], "name": item["name"],
                    "matched_foot_terms": foot, "matched_end_effector_terms": end,
                    "body_id": body_id, "ancestor_body_ids": body_chain(bodies, body_id),
                })
    return {
        "method": "case-insensitive name substrings; candidates for manual inspection only",
        "foot_terms": list(FOOT_TERMS), "end_effector_terms": list(END_TERMS),
        "name_matches": matches,
        "leaf_body_ids": [b["id"] for b in bodies if b["id"] != 0 and not b["child_body_ids"]],
        "h2_plus_correspondence": "not_verified",
    }


def extract_inventory(mj, model):
    result = {}
    for table, (kind, count, fields) in TABLES.items():
        result[table] = [
            {**reference(mj, model, kind, i),
             "fields": {key: plain(getattr(model, key)[i]) for key in fields.split()}}
            for i in range(int(getattr(model, count)))
        ]

    for body in result["bodies"]:
        i, fields = body["id"], body["fields"]
        body["parent"] = None if i == 0 else reference(mj, model, "BODY", fields["body_parentid"])
        body["child_body_ids"] = [b["id"] for b in result["bodies"]
                                   if b["id"] != 0 and b["fields"]["body_parentid"] == i]
        for table, field in (("joints", "jnt_bodyid"), ("geoms", "geom_bodyid"),
                             ("sites", "site_bodyid")):
            body[table[:-1] + "_ids"] = [x["id"] for x in result[table] if x["fields"][field] == i]

    for joint in result["joints"]:
        i, fields = joint["id"], joint["fields"]
        joint["body"] = reference(mj, model, "BODY", fields["jnt_bodyid"])
        joint["type"] = enum_value(mj, "mjtJoint", fields["jnt_type"])
        # Addresses avoid assuming 1 joint == 1 qpos == 1 velocity DoF.
        q0, d0 = fields["jnt_qposadr"], fields["jnt_dofadr"]
        q1 = int(model.jnt_qposadr[i + 1]) if i + 1 < model.njnt else int(model.nq)
        d1 = int(model.jnt_dofadr[i + 1]) if i + 1 < model.njnt else int(model.nv)
        joint["qpos_indices"] = list(range(q0, q1))
        joint["dof_indices"] = list(range(d0, d1))
        joint["qpos0"] = plain(model.qpos0[q0:q1])
        joint["dof_parameters"] = {
            key: plain(getattr(model, key)[d0:d1])
            for key in ("dof_damping", "dof_armature", "dof_frictionloss")
        }
        joint["direct_actuator_ids"] = []

    for actuator in result["actuators"]:
        fields = actuator["fields"]
        for label, field, enum in (
            ("transmission", "actuator_trntype", "mjtTrn"),
            ("dynamics", "actuator_dyntype", "mjtDyn"),
            ("gain", "actuator_gaintype", "mjtGain"),
            ("bias", "actuator_biastype", "mjtBias"),
        ):
            actuator[label] = enum_value(mj, enum, fields[field])
        actuator["targets"] = transmission_targets(
            mj, model, actuator["transmission"]["name"], fields["actuator_trnid"])
        target_joint = actuator["targets"].get("joint")
        if target_joint is not None:
            result["joints"][target_joint["id"]]["direct_actuator_ids"].append(actuator["id"])

    for table, prefix in (("geoms", "geom"), ("sites", "site")):
        for item in result[table]:
            fields = item["fields"]
            item["body"] = reference(mj, model, "BODY", fields[prefix + "_bodyid"])
            item["type"] = enum_value(mj, "mjtGeom", fields[prefix + "_type"])
            if table == "geoms":
                item["mesh"] = (reference(mj, model, "MESH", fields["geom_dataid"])
                                if item["type"]["name"] == "mjGEOM_MESH" else None)

    for tendon in result["tendons"]:
        fields = tendon["fields"]
        start, count = fields["tendon_adr"], fields["tendon_num"]
        tendon["path"] = []
        for i in range(start, start + count):
            wrap = enum_value(mj, "mjtWrap", model.wrap_type[i])
            kind = {"mjWRAP_JOINT": "JOINT", "mjWRAP_SITE": "SITE",
                    "mjWRAP_SPHERE": "GEOM", "mjWRAP_CYLINDER": "GEOM"}.get(wrap["name"])
            tendon["path"].append({
                "wrap_type": wrap, "wrap_objid": int(model.wrap_objid[i]),
                "wrap_prm": plain(model.wrap_prm[i]),
                "object": reference(mj, model, kind, model.wrap_objid[i]) if kind else None,
            })

    for pair in result["contact_pairs"]:
        pair["geoms"] = [reference(mj, model, "GEOM", pair["fields"][key])
                         for key in ("pair_geom1", "pair_geom2")]
    for exclusion in result["contact_exclusions"]:
        signature = int(exclusion["fields"]["exclude_signature"])
        exclusion["bodies"] = [reference(mj, model, "BODY", i)
                               for i in ((signature >> 16) & 0xFFFF, signature & 0xFFFF)]
    for sensor in result["sensors"]:
        fields = sensor["fields"]
        sensor["type"] = enum_value(mj, "mjtSensor", fields["sensor_type"])
        for label, type_key, id_key in (("object", "sensor_objtype", "sensor_objid"),
                                       ("reference", "sensor_reftype", "sensor_refid")):
            obj_type = enum_value(mj, "mjtObj", fields[type_key])
            index = int(fields[id_key])
            sensor[label] = {"type": obj_type, "id": index,
                             "name": mj.mj_id2name(model, fields[type_key], index) if index >= 0 else None}

    result["options"] = {key: plain(getattr(model.opt, key))
                         for key in ("timestep", "gravity", "disableflags", "enableflags")}
    result["qpos0"] = plain(model.qpos0)
    result["structure_hints"] = structure_hints(result)
    return result


def inspect_model(model_path):
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "success": False, "load_success": False, "stage": "resolve_model_path",
        "source": {"requested_path": str(model_path), "resolved_path": None,
                   "sha256": None, "size_bytes": None,
                   "digest_scope": "entry file only; excludes referenced XML/meshes/textures"},
        "runtime": {"python": platform.python_version(), "executable": sys.executable,
                    "platform": platform.platform(), "mujoco_version": None},
        "counts": None, "inventory": None, "error": None,
        "scope": {"backend": "CPU MuJoCo", "mj_forward_called": False,
                  "simulation_steps": 0, "h2_plus_correspondence": "not_verified"},
    }
    try:
        path = Path(model_path).expanduser().resolve()
        report["source"]["resolved_path"] = path.as_posix()
        report["stage"] = "fingerprint_model_file"
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        report["source"].update(sha256=digest.hexdigest(), size_bytes=size)
        report["stage"] = "import_mujoco"
        mj = importlib.import_module("mujoco")
        report["runtime"]["mujoco_version"] = mj.__version__
        report["stage"] = "load_model"
        model = mj.MjModel.from_xml_path(str(path))
        report["load_success"] = True
        report["stage"] = "extract_inventory"
        report["counts"] = {key: int(getattr(model, key)) for key in COUNTS}
        report["inventory"] = extract_inventory(mj, model)
        report["stage"] = "complete"
        report["success"] = True
    except Exception as exc:
        report["error"] = {"type": f"{type(exc).__module__}.{type(exc).__name__}",
                           "message": str(exc), "traceback": traceback.format_exc()}
    return report


def write_report(report, output):
    # Serialize before creating a file; no NaN/Infinity extensions in the JSON.
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects source assets and previous evidence.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to an existing MJCF/XML entry file")
    parser.add_argument("--output", required=True, help="New JSON output path (existing files are refused)")
    args = parser.parse_args(argv)
    report = inspect_model(args.model)
    if report["error"] is not None:
        print(report["error"]["traceback"], file=sys.stderr, end="")
    try:
        output = write_report(report, args.output)
    except Exception:
        print("Failed to write JSON report:", file=sys.stderr)
        traceback.print_exc()
        return 2
    print(json.dumps({"success": report["success"], "load_success": report["load_success"],
                      "stage": report["stage"], "counts": report["counts"],
                      "output": output.as_posix()}, ensure_ascii=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
