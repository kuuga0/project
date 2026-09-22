from pathlib import Path

import numpy as np

from foot_contact_wrench.mujoco_io.bench import load_bench_model, run_free_body_bench


ROOT = Path(__file__).resolve().parents[2]


def test_single_foot_bench_generates_ground_contact():
    model, data = load_bench_model(str(ROOT / "models" / "bench" / "single_foot_bench.xml"))
    records = run_free_body_bench(
        model,
        data,
        duration_s=0.4,
        initial_position=np.array([0.0, 0.0, 0.025]),
    )
    assert any(record["contact_count"] > 0 for record in records)
    final = records[-1]
    assert final["contact_count"] > 0
    assert final["contact_count"] <= 50
    assert final["contact_constraint_count"] >= final["contact_count"]
    assert final["normal_force_names"] == [
        f"contact_{index:02d}" for index in range(50)
    ]
    assert len(final["normal_forces_n"]) == 50
    assert final["contact_wrench_world"][5] > 0.0
    assert np.allclose(
        final["pressure_model_wrench_foot"],
        final["contact_wrench_foot"],
        atol=1e-8,
    )
    assert "wrench_reference_world_m" not in final
    assert np.isclose(final["com_position_world_m"][2], 0.026, atol=1e-4)
    assert np.isclose(
        sum(final["normal_forces_n"]), 4.0 * 9.81, atol=0.2
    )
