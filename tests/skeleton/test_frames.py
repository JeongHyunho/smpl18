"""Frame changes: any pair of up axes, the pinned y->z matrix, and root-only application to a pose."""

from __future__ import annotations

import json

import numpy as np
import pytest

from smpl24.skeleton import frames as f
from smpl24.skeleton import kinematics as k
from smpl24.skeleton.rotations import axis_angle_to_matrix

GRAVITY_Y_UP = (0.0, -9.80665, 0.0)
GRAVITY_Z_UP = (0.0, 0.0, -9.80665)
Y_TO_Z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


@pytest.fixture(scope="module")
def y_to_z():
    return f.transform_for_gravity(GRAVITY_Y_UP, target_up="z")


def test_the_rotation_is_a_rotation(y_to_z):
    np.testing.assert_allclose(y_to_z.rotation @ y_to_z.rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(y_to_z.rotation) == pytest.approx(1.0)


def test_gravity_lands_exactly_on_the_target_constant(y_to_z):
    rotated = y_to_z.rotation @ np.asarray(GRAVITY_Y_UP)
    np.testing.assert_array_equal(rotated, GRAVITY_Z_UP)


def test_the_y_to_z_matrix_and_quaternion_are_pinned(y_to_z):
    """A change here silently reorients every artifact, so it has to fail a test first."""
    np.testing.assert_array_equal(y_to_z.rotation, Y_TO_Z)
    np.testing.assert_array_equal(f.up_axis_rotation("y", "z"), Y_TO_Z)
    np.testing.assert_array_equal(f.frame_change("y", "z").rotation, Y_TO_Z)
    assert y_to_z.quaternion_wxyz == pytest.approx((np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0))


def test_up_and_forward_land_where_expected(y_to_z):
    np.testing.assert_allclose(y_to_z.apply([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(y_to_z.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(y_to_z.apply([0.0, 0.0, 1.0]), [0.0, -1.0, 0.0], atol=1e-12)


def test_it_applies_to_stacks_of_points_and_inverts(y_to_z):
    points = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    out = y_to_z.apply(points)
    assert out.shape == points.shape
    for i in range(2):
        for j in range(4):
            np.testing.assert_allclose(out[i, j], y_to_z.rotation @ points[i, j])
    np.testing.assert_allclose(y_to_z.invert(out), points, atol=1e-12)
    with pytest.raises(ValueError):
        y_to_z.apply(np.zeros((3, 4)))


def test_an_already_target_up_source_is_left_alone():
    identity = f.transform_for_gravity(GRAVITY_Z_UP, target_up="z")
    np.testing.assert_array_equal(identity.rotation, np.eye(3))
    np.testing.assert_array_equal(f.up_axis_rotation("z", "z"), np.eye(3))


def test_a_source_hanging_the_other_way_up_is_still_handled():
    flipped = f.transform_for_gravity((0.0, 9.80665, 0.0), target_up="z")
    np.testing.assert_allclose(flipped.rotation @ [0.0, 9.80665, 0.0], [0.0, 0.0, -9.80665], atol=1e-12)
    assert np.linalg.det(flipped.rotation) == pytest.approx(1.0)
    antiparallel = f.transform_for_gravity((0.0, 0.0, 9.80665), target_up="z")
    np.testing.assert_array_equal(antiparallel.rotation, np.diag([1.0, -1.0, -1.0]))


def test_zero_length_directions_are_refused():
    with pytest.raises(ValueError):
        f.rotation_from_gravity((0.0, 0.0, 0.0), target_up="z")
    with pytest.raises(ValueError):
        f.rotation_between([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_unknown_axes_are_refused():
    with pytest.raises(ValueError, match="up axis"):
        f.up_axis_rotation("y", "w")
    with pytest.raises(ValueError, match="up axis"):
        f.transform_for_gravity(GRAVITY_Y_UP, target_up="up")


@pytest.mark.parametrize("source", f.UP_AXES)
@pytest.mark.parametrize("target", f.UP_AXES)
def test_every_axis_pair_is_a_proper_rotation_carrying_up_onto_up_exactly(source, target):
    rotation = f.up_axis_rotation(source, target)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-15)
    unit = {"x": [1.0, 0, 0], "y": [0, 1.0, 0], "z": [0, 0, 1.0]}
    np.testing.assert_array_equal(rotation @ unit[source], unit[target])
    np.testing.assert_array_equal(rotation.T, f.up_axis_rotation(target, source))
    assert set(np.abs(rotation).ravel()) <= {0.0, 1.0}


def test_gravity_and_axis_forms_agree():
    for source, gravity in (("y", GRAVITY_Y_UP), ("z", GRAVITY_Z_UP), ("x", (-9.80665, 0.0, 0.0))):
        for target in f.UP_AXES:
            np.testing.assert_array_equal(
                f.transform_for_gravity(gravity, target_up=target).rotation,
                f.up_axis_rotation(source, target),
            )


class TestApplyToPose:
    PELVIS = np.array([0.003, -0.22, 0.028])

    def test_only_the_root_changes_and_the_kind_is_kept(self, rng, y_to_z):
        poses = rng.normal(0, 0.5, (4, 24, 3))
        trans = rng.normal(0, 1, (4, 3))
        out_vectors, out_trans = y_to_z.apply_to_pose(poses, trans, pelvis_rest=self.PELVIS)
        assert out_vectors.shape == poses.shape
        np.testing.assert_array_equal(out_vectors[:, 1:], poses[:, 1:])
        assert np.all(out_vectors[:, 0] != poses[:, 0])
        np.testing.assert_allclose(
            out_trans, y_to_z.apply(trans) + y_to_z.apply(self.PELVIS) - self.PELVIS, atol=1e-15
        )

        matrices = axis_angle_to_matrix(poses)
        out_matrices, _ = y_to_z.apply_to_pose(matrices, trans, pelvis_rest=self.PELVIS)
        assert out_matrices.shape == matrices.shape
        np.testing.assert_array_equal(out_matrices[:, 1:], matrices[:, 1:])
        np.testing.assert_array_equal(out_matrices[:, 0], y_to_z.rotation @ matrices[:, 0])
        np.testing.assert_allclose(axis_angle_to_matrix(out_vectors), out_matrices, atol=1e-12)

    def test_a_pelvis_at_the_origin_reduces_to_rotating_the_translation(self, rng, y_to_z):
        poses = rng.normal(0, 0.5, (4, 24, 3))
        trans = rng.normal(0, 1, (4, 3))
        _, out_trans = y_to_z.apply_to_pose(poses, trans, pelvis_rest=np.zeros(3))
        np.testing.assert_array_equal(out_trans, y_to_z.apply(trans))

    def test_forward_kinematics_of_the_changed_pose_is_the_changed_forward_kinematics(self, model, rng, y_to_z):
        # Shift the rest skeleton so its pelvis is off the model origin, as a real SMPL body's is;
        # with the pelvis at the origin the (C - I) j_0 term vanishes and would go untested.
        rest = k.rest_joints(model, np.array([0.4, 0.0])) + self.PELVIS
        assert np.linalg.norm(rest[0]) > 0.1
        poses = rng.normal(0, 0.5, (4, 24, 3))
        trans = rng.normal(0, 1, (4, 3))
        positions, world = k.fk_batch(rest, poses, trans)
        new_poses, new_trans = y_to_z.apply_to_pose(poses, trans, pelvis_rest=rest[0])
        new_positions, new_world = k.fk_batch(rest, new_poses, new_trans)
        # the whole body turns rigidly: every world joint and every segment rotation gets C
        np.testing.assert_allclose(new_positions, y_to_z.apply(positions), atol=1e-12)
        np.testing.assert_allclose(new_world, y_to_z.rotation @ world, atol=1e-12)

    def test_single_frame_and_batch_agree(self, rng, y_to_z):
        poses = rng.normal(0, 0.5, (3, 24, 3))
        trans = rng.normal(0, 1, (3, 3))
        batch_poses, batch_trans = y_to_z.apply_to_pose(poses, trans, pelvis_rest=self.PELVIS)
        single_poses, single_trans = y_to_z.apply_to_pose(poses[1], trans[1], pelvis_rest=self.PELVIS)
        np.testing.assert_array_equal(single_poses, batch_poses[1])
        np.testing.assert_array_equal(single_trans, batch_trans[1])

    def test_wrong_shapes_are_refused(self, y_to_z):
        with pytest.raises(ValueError):
            y_to_z.apply_to_pose(np.zeros((4, 23, 3)), np.zeros((4, 3)), pelvis_rest=self.PELVIS)
        with pytest.raises(ValueError):
            y_to_z.apply_to_pose(np.zeros((4, 24, 3)), np.zeros((4, 2)), pelvis_rest=self.PELVIS)
        with pytest.raises(ValueError):
            y_to_z.apply_to_pose(np.zeros((4, 24, 3)), np.zeros((4, 3)), pelvis_rest=np.zeros(2))


def test_the_record_is_plain_data(y_to_z):
    record = y_to_z.record()
    assert set(record) == {"rotation", "quaternion_wxyz", "source_up", "target_up", "source_gravity"}
    assert record["target_up"] == "z" and record["source_up"] is None
    assert record["source_gravity"] == list(GRAVITY_Y_UP)
    assert np.asarray(json.loads(json.dumps(record))["rotation"]).tolist() == Y_TO_Z.tolist()
    named = f.frame_change("y", "z").record()
    assert named["source_up"] == "y" and named["source_gravity"] is None
