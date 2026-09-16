"""Rotation algebra: Kabsch stays proper, quaternions stay unit, conversions round-trip, batches equal singles."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smpl18.skeleton import rotations as r

PLATE_LOCAL_M = np.array(
    [[0.0000, 0.0000, 0.0], [0.0531, 0.0000, 0.0], [0.0531, 0.0550, 0.0], [-0.0109, 0.0641, 0.0]]
)
NON_PLANAR_LOCAL_M = np.vstack([PLATE_LOCAL_M, [[0.02, 0.02, 0.03]]])


def _centred(points: np.ndarray) -> np.ndarray:
    return points - points.mean(axis=0)


def _random_quaternions(rng, n):
    q = rng.standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


class TestKabsch:
    def test_a_known_rotation_is_recovered_to_machine_precision(self):
        rotation = Rotation.from_euler("xyz", [0.7, -0.3, 1.9]).as_matrix()
        local = _centred(NON_PLANAR_LOCAL_M)
        world = local @ rotation.T
        recovered = r.kabsch_rotation(local, world)
        np.testing.assert_allclose(recovered, rotation, atol=1e-12)

    def test_a_planar_plate_gives_a_proper_rotation(self):
        rotation = Rotation.from_euler("xyz", [0.3, -1.1, 2.4]).as_matrix()
        local = _centred(PLATE_LOCAL_M)
        recovered = r.kabsch_rotation(local, local @ rotation.T)
        assert np.linalg.det(recovered) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(recovered @ recovered.T, np.eye(3), atol=1e-12)

    def test_batched_and_single_covariances_give_identical_rotations(self, rng):
        local = _centred(NON_PLANAR_LOCAL_M)
        rotations = Rotation.random(6, random_state=7).as_matrix()
        worlds = np.einsum("fij,mj->fmi", rotations, local) + rng.normal(0, 1e-4, (6, 5, 3))
        covariances = np.einsum("mi,fmj->fij", local, worlds - worlds.mean(axis=1, keepdims=True))

        batched = r.proper_rotation_from_covariance(covariances)
        singles = np.stack([r.proper_rotation_from_covariance(c) for c in covariances])

        np.testing.assert_allclose(batched, singles, atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(batched), 1.0, atol=1e-12)

    def test_a_mirrored_covariance_still_yields_a_proper_rotation(self):
        local = _centred(NON_PLANAR_LOCAL_M)
        mirrored = local.copy()
        mirrored[:, 2] *= -1.0
        rotation = r.proper_rotation_from_covariance(local.T @ mirrored)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)


class TestQuaternions:
    def test_multiply_by_conjugate_is_identity(self, rng):
        q = _random_quaternions(rng, 8)
        np.testing.assert_allclose(r.multiply(q, r.conjugate(q)), [[1, 0, 0, 0]] * 8, atol=1e-12)

    def test_normalise_refuses_zero(self):
        with pytest.raises(ValueError):
            r.normalise(np.zeros((1, 4)))

    def test_canonicalise_sign_keeps_neighbours_in_one_hemisphere(self, rng):
        q = _random_quaternions(rng, 10)
        q[3:6] *= -1.0
        out = r.canonicalise_sign(q)
        assert all(np.dot(out[i - 1], out[i]) >= 0.0 for i in range(1, 10))
        with pytest.raises(ValueError):
            r.canonicalise_sign(np.zeros((3, 3)))

    def test_log_and_exp_are_inverse(self, rng):
        q = _random_quaternions(rng, 8)
        q[q[:, 0] < 0] *= -1.0
        np.testing.assert_allclose(r.exp_pure(r.log_unit(q)), q, atol=1e-12)
        np.testing.assert_allclose(r.log_unit(np.array([[1.0, 0, 0, 0]])), 0.0, atol=1e-15)

    def test_slerp_hits_the_end_points_and_stays_unit(self, rng):
        q0, q1 = _random_quaternions(rng, 2)
        both0, both1 = np.repeat(q0[None], 3, 0), np.repeat(q1[None], 3, 0)
        out = r.slerp(both0, both1, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(out[0], q0, atol=1e-12)
        np.testing.assert_allclose(np.abs(np.dot(out[2], q1)), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)
        # the midpoint bisects the arc, whichever sign q1 arrived with
        flipped = r.slerp(both0, -both1, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(np.abs(np.sum(flipped * out, axis=1)), 1.0, atol=1e-12)

    def test_body_angular_velocity_of_a_constant_rate_is_exact(self):
        dt, rate = 0.01, np.array([0.3, -1.2, 2.0])
        frames = 20
        q = np.stack([
            r.exp_pure(np.concatenate([[0.0], rate * dt * k / 2.0]))[0] for k in range(frames)
        ])
        omega = r.body_angular_velocity(q, dt)
        assert np.isnan(omega[0]).all() and np.isnan(omega[-1]).all()
        np.testing.assert_allclose(omega[1:-1], np.tile(rate, (frames - 2, 1)), atol=1e-9)
        with pytest.raises(ValueError):
            r.body_angular_velocity(q, 0.0)


class TestConversions:
    def test_axis_angle_matrix_round_trip_keeps_batch_shape(self, rng):
        vectors = rng.normal(0, 0.8, (5, 24, 3))
        matrices = r.axis_angle_to_matrix(vectors)
        assert matrices.shape == (5, 24, 3, 3)
        np.testing.assert_allclose(r.matrix_to_axis_angle(matrices), vectors, atol=1e-12)
        single = r.axis_angle_to_matrix(vectors[2, 7])
        assert single.shape == (3, 3)
        np.testing.assert_array_equal(single, matrices[2, 7])

    def test_matrix_quaternion_round_trip(self, rng):
        matrices = Rotation.random(12, random_state=3).as_matrix()
        q = r.matrix_to_quaternion(matrices)
        assert q.shape == (12, 4)
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(r.quaternion_to_matrix(q), matrices, atol=1e-12)
        np.testing.assert_array_equal(r.matrix_to_quaternion(matrices[4]), q[4])

    def test_quaternion_to_matrix_agrees_with_scipy_and_normalises(self, rng):
        q = _random_quaternions(rng, 10) * 3.0
        mine = r.quaternion_to_matrix(q)
        theirs = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(mine, theirs, atol=1e-14)
        np.testing.assert_array_equal(r.quaternion_to_matrix(q[3]), mine[3])

    def test_identity_maps_to_identity_everywhere(self):
        np.testing.assert_array_equal(r.axis_angle_to_matrix(np.zeros(3)), np.eye(3))
        np.testing.assert_array_equal(r.matrix_to_axis_angle(np.eye(3)), np.zeros(3))
        np.testing.assert_allclose(np.abs(r.matrix_to_quaternion(np.eye(3))), [1, 0, 0, 0])
        np.testing.assert_array_equal(r.quaternion_to_matrix([1.0, 0, 0, 0]), np.eye(3))

    def test_mean_rotation_of_a_constant_returns_that_rotation(self):
        rotation = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
        np.testing.assert_allclose(r.mean_rotation(np.repeat(rotation[None], 7, axis=0)), rotation, atol=1e-9)
