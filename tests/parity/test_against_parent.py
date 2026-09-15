"""The moved code computes what the code it came from computes.

Runs only inside the SOMA Synthetic IMU checkout this package was carved out of (the parent's
``src/soma_synthetic_imu`` and ``scripts/poc/anthro_smpl.py`` must exist); anywhere else it
skips. Once the parent deletes its copies these tests are retired with them.

Equality is exact (``assert_array_equal``) where the arithmetic is the same sequence of
operations. Where the new code batches what the parent did frame by frame (``einsum`` against a
per-frame ``@``), the floating-point order differs and the check is ``allclose`` at 1e-12; each
such case says so.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from smpl24.model.load import Model
from smpl24.skeleton import definition, frames, kinematics, rotations

PARENT = Path(__file__).resolve().parents[4]
PARENT_SRC = PARENT / "src"
ANTHRO = PARENT / "scripts" / "poc" / "anthro_smpl.py"

pytestmark = pytest.mark.skipif(
    not ((PARENT_SRC / "soma_synthetic_imu").is_dir() and ANTHRO.is_file()),
    reason="parent SOMA checkout not present",
)


@pytest.fixture(scope="module")
def parent():
    if str(PARENT_SRC) not in sys.path:
        sys.path.append(str(PARENT_SRC))
    spec = importlib.util.spec_from_file_location("_parent_anthro_smpl", ANTHRO)
    anthro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anthro)
    from soma_synthetic_imu.addbio_retarget import shape_fit, smpl_correspondence, world_frame
    from soma_synthetic_imu.kinematics import quaternion, rigid_body

    return {
        "anthro": anthro, "shape_fit": shape_fit, "correspondence": smpl_correspondence,
        "world_frame": world_frame, "quaternion": quaternion, "rigid_body": rigid_body,
    }


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260915)


@pytest.fixture
def model_arrays(rng) -> dict:
    vertices = 60
    regressor = rng.random((24, vertices))
    regressor /= regressor.sum(axis=1, keepdims=True)
    return {
        "v_template": rng.normal(0, 0.3, (vertices, 3)),
        "shapedirs": rng.normal(0, 0.01, (vertices, 3, 10)),
        "J_regressor": regressor,
        "kintree_parents": np.array(definition.PARENTS, dtype=np.int64),
    }


def test_the_skeleton_definition_is_the_parents(parent):
    anthro, corr = parent["anthro"], parent["correspondence"]
    assert tuple(definition.PARENTS[1:]) == tuple(int(p) for p in anthro.SMPL24_PARENTS[1:])
    assert tuple(definition.PARENTS[1:]) == tuple(int(p) for p in corr.SMPL24_PARENTS[1:])
    assert tuple(definition.JOINT_NAMES) == tuple(anthro.JOINT24_NAMES)
    assert tuple(definition.JOINT_NAMES) == tuple(corr.SMPL24_NAMES)
    assert tuple(definition.SEGMENTS) == tuple(tuple(s) for s in anthro.SEGMENTS)


def test_rest_joints_match_both_parent_copies(parent, model_arrays, rng):
    betas = rng.normal(0, 1, 10)
    new = kinematics.rest_joints(Model(**model_arrays), betas)
    np.testing.assert_array_equal(new, parent["anthro"].rest_joints(model_arrays, betas))
    np.testing.assert_array_equal(new, parent["shape_fit"].smpl_rest_joints(model_arrays, betas))


def test_betas_wider_than_the_model_are_truncated_like_the_parent(parent, model_arrays, rng):
    betas = rng.normal(0, 1, 16)
    np.testing.assert_array_equal(
        kinematics.rest_joints(Model(**model_arrays), betas),
        parent["anthro"].rest_joints(model_arrays, betas),
    )


def test_segment_lengths_match(parent, rng):
    rest = rng.normal(0, 0.3, (24, 3))
    np.testing.assert_array_equal(kinematics.segment_lengths(rest), parent["anthro"].segment_lengths(rest))


def test_single_frame_fk_matches_with_the_pelvis_at_its_rest_position(parent, rng):
    # parent: pelvis at j_rest[0], no translation argument -> new with trans = 0
    rest = rng.normal(0, 0.3, (24, 3))
    local = rotations.axis_angle_to_matrix(rng.normal(0, 0.6, (24, 3)))
    parent_positions, parent_global = parent["anthro"].smpl_fk_positions(rest, local)
    positions, world = kinematics.fk(rest, local, np.zeros(3))
    np.testing.assert_array_equal(world, parent_global)
    # per-frame `@` in the parent against `einsum` here: allclose, not bit-equal
    np.testing.assert_allclose(positions, parent_positions, rtol=0, atol=1e-12)


def test_batched_fk_matches_with_the_pelvis_at_the_origin(parent, rng):
    # parent batch: pelvis at the origin -> new with trans = -j_rest[0]
    rest = rng.normal(0, 0.3, (24, 3))
    local = rotations.axis_angle_to_matrix(rng.normal(0, 0.6, (7, 24, 3)))
    expected = parent["anthro"].fk_positions_batch(rest, local)
    positions, _ = kinematics.fk_batch(rest, local, np.tile(-rest[0], (7, 1)))
    # the same einsum sequence, but rest[0] + (-rest[0]) is added first here
    np.testing.assert_allclose(positions, expected, rtol=0, atol=1e-12)


def test_gravity_frame_change_matches_the_parent_to_z_up(parent):
    for gravity in ((0.0, -9.80665, 0.0), (0.0, 0.0, -9.81), (0.3, -9.7, 0.2), (0.0, 9.81, 0.0)):
        new = frames.transform_for_gravity(gravity, target_up="z")
        old = parent["world_frame"].transform_for_gravity(gravity)
        np.testing.assert_allclose(new.rotation, old.rotation, rtol=0, atol=1e-15)
        vectors = np.random.default_rng(1).normal(0, 1, (5, 3))
        np.testing.assert_allclose(new.apply(vectors), old.apply(vectors), rtol=0, atol=1e-15)


def test_quaternion_algebra_matches(parent, rng):
    q = parent["quaternion"]
    a = rotations.normalise(rng.normal(0, 1, (9, 4)))
    b = rotations.normalise(rng.normal(0, 1, (9, 4)))
    np.testing.assert_array_equal(rotations.normalise(a), q.normalise(a))
    np.testing.assert_array_equal(rotations.conjugate(a), q.conjugate(a))
    np.testing.assert_array_equal(rotations.multiply(a, b), q.multiply(a, b))
    np.testing.assert_array_equal(rotations.canonicalise_sign(a), q.canonicalise_sign(a))
    np.testing.assert_array_equal(rotations.log_unit(a), q.log_unit(a))
    v = rng.normal(0, 0.5, (9, 3))
    np.testing.assert_array_equal(rotations.exp_pure(v), q.exp_pure(v))
    t = rng.random(9)
    np.testing.assert_array_equal(rotations.slerp(a, b, t), q.slerp(a, b, t))
    series = rotations.normalise(np.cumsum(rng.normal(0, 0.05, (30, 4)), axis=0) + [1, 0, 0, 0])
    np.testing.assert_array_equal(
        rotations.body_angular_velocity(series, 0.01), q.body_angular_velocity(series, 0.01)
    )


def test_kabsch_with_reflection_correction_matches(parent, rng):
    for _ in range(20):
        covariance = rng.normal(0, 1, (3, 3))
        np.testing.assert_array_equal(
            rotations.proper_rotation_from_covariance(covariance),
            parent["rigid_body"].proper_rotation_from_covariance(covariance),
        )
