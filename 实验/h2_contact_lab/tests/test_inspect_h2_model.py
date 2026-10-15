"""Offline error/relationship tests plus optional real-MuJoCo MJCF tests.

Run with: python -m unittest discover -s tests -v
Real-MuJoCo tests skip only when the package is absent, not when its import fails.
"""

from contextlib import redirect_stderr, redirect_stdout
from enum import IntEnum
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_h2_model.py"
SPEC = importlib.util.spec_from_file_location("inspect_h2_model", SCRIPT)
inspect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspect)
HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="h2-inspect-", dir=SCRIPT.parents[1])
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "模型 with spaces.xml"
        self.model.write_text('<mujoco model="offline_fixture"/>', encoding="utf-8")

    def test_missing_file_reports_full_exception_without_import(self):
        with patch.object(inspect.importlib, "import_module") as importer:
            report = inspect.inspect_model(self.directory / "missing.xml")
        importer.assert_not_called()
        self.assertFalse(report["success"])
        self.assertFalse(report["load_success"])
        self.assertEqual(report["stage"], "fingerprint_model_file")
        self.assertIn("FileNotFoundError", report["error"]["traceback"])

    def test_missing_dependency_retains_path_and_fingerprint(self):
        with patch.object(inspect.importlib, "import_module", side_effect=ModuleNotFoundError("mujoco")):
            report = inspect.inspect_model(self.model)
        self.assertEqual(report["stage"], "import_mujoco")
        self.assertEqual(report["source"]["resolved_path"], self.model.resolve().as_posix())
        self.assertEqual(report["source"]["sha256"], hashlib.sha256(self.model.read_bytes()).hexdigest())
        self.assertEqual(report["source"]["size_bytes"], self.model.stat().st_size)
        self.assertIsNone(report["runtime"]["mujoco_version"])

    def test_load_exception_is_distinct_from_extraction_exception(self):
        fake_mj = SimpleNamespace(__version__="test-double", MjModel=SimpleNamespace())
        with patch.object(inspect.importlib, "import_module", return_value=fake_mj):
            with patch.object(fake_mj.MjModel, "from_xml_path", create=True,
                              side_effect=ValueError("missing included asset")):
                report = inspect.inspect_model(self.model)
            self.assertFalse(report["load_success"])
            self.assertEqual(report["stage"], "load_model")
            self.assertIn("missing included asset", report["error"]["traceback"])
            with patch.object(fake_mj.MjModel, "from_xml_path", create=True,
                              return_value=SimpleNamespace()):
                report = inspect.inspect_model(self.model)
            self.assertTrue(report["load_success"])
            self.assertFalse(report["success"])
            self.assertEqual(report["stage"], "extract_inventory")

    def test_cli_failure_writes_json_stdout_and_stderr(self):
        output = self.directory / "结果 with spaces" / "inspection.json"
        run = subprocess.run(
            [sys.executable, str(SCRIPT), "--model", str(self.directory / "missing.xml"),
             "--output", str(output)], capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False,
        )
        self.assertEqual(run.returncode, 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(report["success"])
        self.assertIn("Traceback", run.stderr)
        self.assertEqual(json.loads(run.stdout)["output"], output.resolve().as_posix())

    def test_existing_output_and_source_are_not_overwritten(self):
        output = self.directory / "existing.json"
        output.write_bytes(b"previous evidence\n")
        source_bytes = self.model.read_bytes()
        for destination in (output, self.model):
            with self.subTest(destination=destination):
                with self.assertRaises(FileExistsError):
                    inspect.write_report({"success": True}, destination)
        self.assertEqual(output.read_bytes(), b"previous evidence\n")
        self.assertEqual(self.model.read_bytes(), source_bytes)

    def test_output_failure_has_nonzero_exit_and_traceback(self):
        stderr, stdout = io.StringIO(), io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout):
            code = inspect.main(["--model", str(self.directory / "missing.xml"),
                                 "--output", str(self.model / "impossible.json")])
        self.assertEqual(code, 2)
        self.assertIn("Failed to write JSON", stderr.getvalue())
        self.assertIn("FileNotFoundError", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")

    def test_json_rejects_nonfinite_before_creating_file(self):
        output = self.directory / "nonfinite.json"
        with self.assertRaises(ValueError):
            inspect.write_report({"value": float("nan")}, output)
        self.assertFalse(output.exists())

    def test_transmissions_do_not_assume_joint_indices(self):
        kinds = SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_TENDON=2, mjOBJ_SITE=3, mjOBJ_BODY=4)
        fake_mj = SimpleNamespace(mjtObj=kinds, mj_id2name=lambda model, kind, i: f"{kind}:{i}")
        cases = {
            "mjTRN_JOINT": {"joint": {"id": 2, "name": "1:2"}},
            "mjTRN_JOINTINPARENT": {"joint": {"id": 2, "name": "1:2"}},
            "mjTRN_TENDON": {"tendon": {"id": 2, "name": "2:2"}},
            "mjTRN_SITE": {"site": {"id": 2, "name": "3:2"}, "reference_site": None},
            "mjTRN_BODY": {"body": {"id": 2, "name": "4:2"}},
            "future_type": {},
        }
        for kind, expected in cases.items():
            with self.subTest(kind=kind):
                self.assertEqual(inspect.transmission_targets(fake_mj, None, kind, [2, -1]), expected)
        slider = inspect.transmission_targets(fake_mj, None, "mjTRN_SLIDERCRANK", [2, 5])
        self.assertEqual(slider["crank_site"]["id"], 2)
        self.assertEqual(slider["slider_site"]["id"], 5)

    def test_unknown_enum_keeps_numeric_id(self):
        class TestEnum(IntEnum):
            KNOWN = 0
        self.assertEqual(inspect.enum_value(SimpleNamespace(TestEnum=TestEnum), "TestEnum", 99),
                         {"id": 99, "name": None})

    def test_hints_keep_unnamed_leaves_and_do_not_certify_h2_plus(self):
        inventory = {
            "bodies": [
                {"id": 0, "name": "world", "parent": None, "child_body_ids": [1]},
                {"id": 1, "name": None, "parent": {"id": 0}, "child_body_ids": []},
            ],
            "joints": [], "geoms": [],
            "sites": [{"id": 0, "name": "left_sole", "fields": {"site_bodyid": 1}}],
        }
        hints = inspect.structure_hints(inventory)
        self.assertEqual(hints["leaf_body_ids"], [1])
        self.assertEqual(hints["name_matches"][0]["ancestor_body_ids"], [0, 1])
        self.assertEqual(hints["h2_plus_correspondence"], "not_verified")


# Synthetic fixture, not an H2 model, not a standing/control experiment.
MJCF = """<mujoco model="inspection_fixture">
  <compiler angle="degree"/>
  <asset><mesh name="tetra" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="floating"/>
      <geom name="base_mesh" type="mesh" mesh="tetra"/>
      <body name="left_foot" pos="0 0 -0.5">
        <joint name="ankle" type="hinge" axis="0 1 0" limited="true" range="-30 45"/>
        <geom name="foot_collision" type="box" size="0.1 0.05 0.02" friction="0.7 0.01 0.002"/>
        <geom type="sphere" size="0.01" contype="0" conaffinity="0"/>
        <site name="left_sole" pos="0 0 -0.02"/>
      </body>
      <body name="right_wrist" pos="0 0.4 0">
        <joint name="wrist_ball" type="ball"/>
        <geom type="sphere" size="0.04"/>
        <site name="right_palm"/>
      </body>
      <body name="carriage" pos="0 -0.4 0">
        <joint name="slide" type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <tendon><fixed name="coupled"><joint joint="ankle" coef="1"/><joint joint="slide" coef="2"/></fixed></tendon>
  <actuator>
    <motor name="ankle_motor" joint="ankle" gear="2" ctrllimited="true" ctrlrange="-2 3"
           forcelimited="true" forcerange="-10 10"/>
    <general name="wrist_parent" jointinparent="wrist_ball" gear="1 0 0 0 0 0"/>
    <general name="palm_force" site="right_palm" refsite="left_sole" gear="1 0 0 0 0 0"/>
    <motor name="tendon_motor" tendon="coupled"/>
    <adhesion name="foot_adhesion" body="left_foot" ctrlrange="0 1"/>
  </actuator>
  <sensor><force name="sole_force" site="left_sole"/><jointpos name="ankle_pos" joint="ankle"/></sensor>
  <contact><pair name="foot_floor" geom1="floor" geom2="foot_collision"/>
    <exclude name="base_wrist" body1="base" body2="right_wrist"/></contact>
</mujoco>"""


@unittest.skipUnless(HAS_MUJOCO, "MuJoCo is absent; no package installation attempted")
class MuJoCoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="h2-inspect-mj-", dir=SCRIPT.parents[1])
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "synthetic.xml"
        self.model.write_text(MJCF, encoding="utf-8")

    def test_compiled_inventory_and_actual_fields(self):
        before = self.model.read_bytes()
        report = inspect.inspect_model(self.model)
        self.assertTrue(report["success"], report["error"])
        self.assertEqual(report["counts"]["njnt"], 4)
        self.assertEqual(report["counts"]["nq"], 13)
        self.assertEqual(report["counts"]["nv"], 11)
        inventory = report["inventory"]
        joints = {j["name"]: j for j in inventory["joints"]}
        self.assertEqual(len(joints["floating"]["qpos_indices"]), 7)
        self.assertEqual(len(joints["wrist_ball"]["dof_indices"]), 3)
        self.assertAlmostEqual(joints["ankle"]["fields"]["jnt_range"][0], -math.pi / 6)
        self.assertEqual(joints["ankle"]["direct_actuator_ids"], [0])
        self.assertEqual(inventory["actuators"][0]["fields"]["actuator_ctrlrange"], [-2, 3])
        self.assertEqual(inventory["actuators"][2]["targets"]["reference_site"]["name"], "left_sole")
        self.assertEqual(inventory["actuators"][3]["targets"]["tendon"]["name"], "coupled")
        self.assertEqual(inventory["actuators"][4]["targets"]["body"]["name"], "left_foot")
        self.assertEqual(len(inventory["tendons"][0]["path"]), 2)
        geoms = {g["name"]: g for g in inventory["geoms"] if g["name"]}
        self.assertEqual(geoms["base_mesh"]["mesh"]["name"], "tetra")
        self.assertEqual(geoms["foot_collision"]["fields"]["geom_friction"], [0.7, 0.01, 0.002])
        self.assertTrue(any(g["name"] is None and g["fields"]["geom_contype"] == 0
                            for g in inventory["geoms"]))
        self.assertEqual(inventory["contact_pairs"][0]["geoms"][1]["name"], "foot_collision")
        self.assertEqual([b["name"] for b in inventory["contact_exclusions"][0]["bodies"]],
                         ["base", "right_wrist"])
        self.assertEqual(inventory["sensors"][0]["object"]["name"], "left_sole")
        self.assertEqual(self.model.read_bytes(), before)
        output = inspect.write_report(report, self.directory / "inventory.json")
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

    def test_bad_xml_is_a_load_failure(self):
        self.model.write_text("<mujoco><broken", encoding="utf-8")
        report = inspect.inspect_model(self.model)
        self.assertFalse(report["load_success"])
        self.assertEqual(report["stage"], "load_model")
        self.assertIn("Traceback", report["error"]["traceback"])

    def test_includes_resolve_relative_to_model_not_current_directory(self):
        child = self.directory / "child.xml"
        child.write_text('<mujoco><worldbody><body name="included_foot"/></worldbody></mujoco>', encoding="utf-8")
        self.model.write_text('<mujoco><include file="child.xml"/></mujoco>', encoding="utf-8")
        report = inspect.inspect_model(self.model)
        self.assertTrue(report["success"], report["error"])
        self.assertIn("included_foot", [b["name"] for b in report["inventory"]["bodies"]])
        self.assertEqual(report["source"]["sha256"], hashlib.sha256(self.model.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
