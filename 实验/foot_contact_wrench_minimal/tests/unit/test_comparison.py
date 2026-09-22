import mujoco
import numpy as np

from foot_contact_wrench.analysis.comparison import (
    axial_support_bounds,
    support_wrench_samples,
)
from foot_contact_wrench.contact.discrete import DiscreteContactModel
from foot_contact_wrench.optimization.local_set import LocalWrenchSet
from foot_contact_wrench.visualization.wrench_scene import (
    WrenchSceneOptions,
    add_wrench,
    add_wrench_cone,
)


def test_axial_support_bounds_contain_symmetric_working_wrench():
    model = DiscreteContactModel.create(
        points=[[0.1, 0.05, 0.0], [0.1, -0.05, 0.0], [-0.1, 0.05, 0.0], [-0.1, -0.05, 0.0]],
        normal=[0.0, 0.0, 1.0],
        tangent_basis=np.eye(3)[:, :2],
        friction=0.5,
        normal_force_max=200.0,
    )
    local_set = LocalWrenchSet(model, 400.0)
    lower, upper = axial_support_bounds(local_set)
    working = model.wrench(np.full(4, 100.0), np.zeros((4, 2)))
    assert np.all(lower <= working + 1e-6)
    assert np.all(working <= upper + 1e-6)


def test_wrench_scene_adds_expected_custom_geometries():
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    scene = mujoco.MjvScene(model, maxgeom=32)
    options = WrenchSceneOptions()
    add_wrench(
        scene,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 10.0],
        options,
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.5, 0.0, 1.0],
        "test",
        [0.0, 0.0, 0.0],
    )
    assert scene.ngeom == 2


def test_all_scene_arrows_start_at_the_foot_com():
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    scene = mujoco.MjvScene(model, maxgeom=32)
    reference = np.array([0.1, -0.2, 0.3])
    options = WrenchSceneOptions()
    add_wrench(
        scene,
        reference,
        np.ones(6),
        options,
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.5, 0.0, 1.0],
        "test",
    )
    add_wrench_cone(
        scene, reference, np.eye(3), np.tile(np.ones(6), (3, 1)), options
    )
    for geom in scene.geoms[: scene.ngeom]:
        assert np.allclose(geom.pos, reference)


def test_support_wrench_samples_keep_force_and_moment_together():
    model = DiscreteContactModel.create(
        points=[[0.1, 0.05, -0.026], [0.1, -0.05, -0.026], [-0.1, 0.05, -0.026], [-0.1, -0.05, -0.026]],
        normal=[0.0, 0.0, 1.0],
        tangent_basis=np.eye(3)[:, :2],
        friction=0.5,
        normal_force_max=200.0,
    )
    samples = support_wrench_samples(LocalWrenchSet(model, 40.0), 8)
    assert samples.shape == (8, 6)
    assert np.allclose(samples[:, 5], 40.0, atol=1e-6)


def test_wrench_cone_uses_two_arrows_per_complete_sample():
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    scene = mujoco.MjvScene(model, maxgeom=32)
    add_wrench_cone(
        scene,
        [0.0, 0.0, 0.0],
        np.eye(3),
        np.tile([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], (3, 1)),
        WrenchSceneOptions(),
    )
    assert scene.ngeom == 6
    assert all(np.allclose(geom.pos, [0.0, 0.0, 0.0]) for geom in scene.geoms[:6])
