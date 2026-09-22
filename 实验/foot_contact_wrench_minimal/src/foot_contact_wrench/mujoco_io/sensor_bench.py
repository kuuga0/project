"""Shared excitation profile for the ideal IMU/pressure-array benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np


def sinusoidal_excitation_wrench(
    sensor_config: dict[str, Any], time_s: float
) -> np.ndarray:
    """Return the configured world-frame COM wrench in ``[moment; force]`` order."""
    excitation = sensor_config["excitation"]
    two_pi = 2.0 * np.pi
    return np.array(
        [
            float(excitation["moment_x_nm"])
            * np.sin(two_pi * float(excitation["moment_x_hz"]) * time_s),
            float(excitation["moment_y_nm"])
            * np.sin(two_pi * float(excitation["moment_y_hz"]) * time_s),
            float(excitation["moment_z_nm"])
            * np.sin(two_pi * float(excitation["moment_z_hz"]) * time_s),
            float(excitation["force_x_n"])
            * np.sin(two_pi * float(excitation["force_x_hz"]) * time_s),
            float(excitation["force_y_n"])
            * np.sin(two_pi * float(excitation["force_y_hz"]) * time_s),
            float(excitation["force_z_n"]),
        ],
        dtype=float,
    )
