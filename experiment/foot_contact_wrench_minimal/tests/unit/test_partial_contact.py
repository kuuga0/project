from pathlib import Path

import mujoco
import numpy as np

from foot_contact_wrench.mujoco_io.partial_contact import (
    build_partial_contact_xml,
    loading_state,
    normal_path_wrench_direction,
    observed_normal_wrench_set,
    pressure_active_mask,
    set_support_scenario,
)


ROOT = Path(__file__).resolve().parents[2]


def test_generated_model_uses_continuous_patches_not_one_patch_per_cell(tmp_path):
    scenarios = [
        {
            "name": "half",
            "support_boxes": [
                {"center_xy_m": [-0.05, 0.0], "size_xy_m": [0.1, 0.1]}
            ],
        },
        {
            "name": "notched",
            "support_boxes": [
                {"center_xy_m": [-0.04, 0.0], "size_xy_m": [0.12, 0.1]},
                {"center_xy_m": [0.06, 0.025], "size_xy_m": [0.08, 0.05]},
            ],
        },
    ]
    output = tmp_path / "partial.xml"
    build_partial_contact_xml(
        ROOT / "models" / "bench" / "single_foot_bench.xml", output, scenarios
    )
    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    support_names = [
        model.geom(index).name
        for index in range(model.ngeom)
        if model.geom(index).name.startswith("support_surface_")
    ]
    assert support_names == [
        "support_surface_half_00",
        "support_surface_notched_00",
        "support_surface_notched_01",
    ]
    assert set_support_scenario(model, data, "notched") == 2
    enabled = [
        name
        for name in support_names
        if model.geom_contype[model.geom(name).id] != 0
    ]
    assert enabled == ["support_surface_notched_00", "support_surface_notched_01"]
    assert sum(name.startswith("contact_") for name in [model.geom(i).name for i in range(model.ngeom)]) == 50


def test_pressure_mask_and_set_do_not_accept_environment_mask():
    forces = np.array([40.0, 0.0, 20.0])
    assert pressure_active_mask(forces, 0.01).tolist() == [True, False, True]
    base_config = {
        "tangent_basis": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        "normal_force_max_n": 100.0,
    }
    local_set = observed_normal_wrench_set(
        base_config,
        points_foot=[[-0.05, 0.0, -0.02], [0.0, 0.0, -0.02], [0.05, 0.0, -0.02]],
        normal_forces=forces,
        normal_foot=[0.0, 0.0, 1.0],
        active_force_threshold_n=0.01,
    )
    assert local_set is not None
    assert local_set.model.active.tolist() == [True, False, True]
    assert np.all(local_set.model.friction == 0.0)
    assert np.isclose(local_set.normal_load, 60.0)
    filtered_wrench = local_set.model.wrench(
        np.where(local_set.model.active, forces, 0.0), np.zeros((3, 2))
    )
    assert local_set.contains(filtered_wrench).feasible


def test_loading_profile_is_continuous_and_starts_without_external_load():
    config = {
        "settle_duration_s": 0.2,
        "preload_ramp_duration_s": 0.8,
        "reference_hold_duration_s": 0.5,
        "sweep_duration_s": 5.0,
        "applied_normal_force_n": 60.0,
        "load_application_height_m": 0.025,
    }
    scenario = {
        "load_path": {"start_xy_m": [-0.04, 0.0], "end_xy_m": [0.08, 0.0]}
    }
    assert loading_state(0.0, config, scenario).normal_force_n == 0.0
    end_preload = loading_state(1.0 - 1e-9, config, scenario)
    start_hold = loading_state(1.0, config, scenario)
    assert np.isclose(end_preload.normal_force_n, start_hold.normal_force_n, atol=1e-6)
    sweep_start = loading_state(1.5, config, scenario)
    sweep_end = loading_state(6.5, config, scenario)
    assert sweep_start.phase == "sweep"
    assert np.allclose(sweep_start.application_point_foot_m[:2], [-0.04, 0.0])
    assert sweep_end.complete
    assert np.allclose(sweep_end.application_point_foot_m[:2], [0.08, 0.0])


def test_normal_path_direction_matches_equilibrium_contact_moment():
    direction, magnitude = normal_path_wrench_direction(
        [-0.04, 0.0, 0.025], [0.08, 0.0, 0.025], 60.0
    )
    assert np.allclose(direction, [0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    assert np.isclose(magnitude, 7.2)
