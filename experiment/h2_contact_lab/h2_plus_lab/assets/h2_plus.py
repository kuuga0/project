"""H2 Plus asset configuration for the project-local Isaac Lab package.

The URDF under ``experiment/robot_asset/h2_plus`` remains the asset source of
truth.  This module only describes how Isaac Lab should import and control it;
it does not modify the Isaac Lab source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_PROJECT_DIR = Path(__file__).resolve().parents[4]
H2_PLUS_URDF = _PROJECT_DIR / "experiment" / "robot_asset" / "h2_plus" / "H2_Plus_with_footrix_no_dex_hands.urdf"
H2_PLUS_USD_DIR = _PROJECT_DIR / "experiment" / "h2_contact_lab" / "output" / "h2_plus_usd"


@dataclass(frozen=True)
class H2PlusImuMountInterface:
    """Asset-level interface for the calibrated torso IMU mount.

    The IMU is attached to ``torso_link`` at its origin, whose axes are
    defined as x-forward, y-left and z-up.  Foot IMU mounts are configured
    separately in the locomotion sensor module.
    """

    enabled: bool = True
    body_name: str | None = "torso_link"
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


H2_PLUS_IMU_MOUNT = H2PlusImuMountInterface()


def _implicit_position_actuator(joint_names_expr: list[str]) -> ImplicitActuatorCfg:
    """Create the provisional simulation actuator used by all H2 groups.

    The 80/8 gains are inherited from the existing project collector as
    simulation-only starting values.  Effort and velocity limits are not
    overridden here; the URDF limits remain the source of those limits.
    """

    return ImplicitActuatorCfg(
        joint_names_expr=joint_names_expr,
        stiffness=80.0,
        damping=8.0,
    )


H2_PLUS_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(H2_PLUS_URDF),
        usd_dir=str(H2_PLUS_USD_DIR),
        usd_file_name="H2_Plus_sensorized.usd",
        force_usd_conversion=True,
        fix_base=False,
        root_link_name="pelvis",
        merge_fixed_joints=False,
        self_collision=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The root is the URDF's pelvis frame.  The height is the initial
        # pelvis-origin placement in the environment frame, not a COM height.
        pos=(0.0, 0.0, 1.02),
    ),
    actuators={
        "legs": _implicit_position_actuator(
            [
                ".*_hip_(pitch|roll|yaw)_joint",
                ".*_knee_joint",
                ".*_ankle_(roll|pitch)_joint",
            ]
        ),
        "waist": _implicit_position_actuator(["waist_(yaw|roll|pitch)_joint"]),
        "head": _implicit_position_actuator(["head_(pitch|yaw)_joint"]),
        "arms": _implicit_position_actuator(
            [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
        ),
    },
)


__all__ = [
    "H2_PLUS_CFG",
    "H2_PLUS_IMU_MOUNT",
    "H2_PLUS_URDF",
    "H2_PLUS_USD_DIR",
    "H2PlusImuMountInterface",
]
