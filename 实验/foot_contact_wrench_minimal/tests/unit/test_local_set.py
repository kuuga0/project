import numpy as np

from foot_contact_wrench.contact.discrete import DiscreteContactModel
from foot_contact_wrench.optimization.local_set import LocalWrenchSet


def square_set() -> LocalWrenchSet:
    model = DiscreteContactModel.create(
        points=[[0.1, 0.05, 0.0], [0.1, -0.05, 0.0], [-0.1, 0.05, 0.0], [-0.1, -0.05, 0.0]],
        normal=[0.0, 0.0, 1.0],
        tangent_basis=np.eye(3)[:, :2],
        friction=[0.5] * 4,
        normal_force_max=[200.0] * 4,
    )
    return LocalWrenchSet(model, 400.0)


def test_support_tangential_force_equals_mu_times_total_normal_load():
    wrench_set = square_set()
    direction = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    result = wrench_set.support(direction)
    assert result.feasible
    assert np.isclose(result.objective, 200.0, atol=1e-4)
    assert np.isclose(result.objective, wrench_set.support_reduced(direction), atol=1e-4)
    assert np.isclose(np.sum(result.normal_forces), 400.0, atol=1e-5)


def test_reduced_support_matches_socp_on_mixed_direction():
    wrench_set = square_set()
    direction = np.array([0.7, -0.3, 0.2, 0.5, -0.4, 0.8])
    result = wrench_set.support(direction)
    assert result.feasible
    assert np.isclose(result.objective, wrench_set.support_reduced(direction), atol=1e-4)


def test_symmetric_normal_load_has_zero_roll_and_pitch_moment():
    wrench_set = square_set()
    normal = np.full(4, 100.0)
    wrench = wrench_set.model.wrench(normal, np.zeros((4, 2)))
    assert np.allclose(wrench, [0.0, 0.0, 0.0, 0.0, 0.0, 400.0], atol=1e-10)


def test_member_query_returns_contact_force_witness():
    wrench_set = square_set()
    target = wrench_set.model.wrench(np.full(4, 100.0), np.zeros((4, 2)))
    result = wrench_set.contains(target)
    assert result.feasible
    assert np.allclose(result.wrench, target, atol=1e-5)


def test_inactive_point_is_zeroed():
    model = DiscreteContactModel.create(
        points=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        normal=[0.0, 0.0, 1.0],
        tangent_basis=np.eye(3)[:, :2],
        friction=[0.5, 0.5],
        active=[True, False],
    )
    wrench_set = LocalWrenchSet(model, 100.0)
    result = wrench_set.support([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert result.feasible
    assert np.isclose(result.normal_forces[1], 0.0, atol=1e-8)
    assert np.isclose(result.tangential_forces[1, :], [0.0, 0.0], atol=1e-8).all()


def test_radial_capacity_from_symmetric_working_point():
    wrench_set = square_set()
    working = wrench_set.model.wrench(np.full(4, 100.0), np.zeros((4, 2)))
    result = wrench_set.radial_capacity(working, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert result.feasible
    assert np.isclose(result.objective, 200.0, atol=1e-4)
