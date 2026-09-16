"""Rest joints from betas, forward kinematics, and the translation convention p_0 = j_0 + t."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smpl18.skeleton import kinematics as k
from smpl18.skeleton.definition import PARENTS, SEGMENTS
from smpl18.skeleton.rotations import axis_angle_to_matrix


def _identity_pose(frames: int | None = None) -> np.ndarray:
    single = np.repeat(np.eye(3)[None], 24, axis=0)
    return single if frames is None else np.repeat(single[None], frames, axis=0)


class TestRestJoints:
    def test_betas_zero_gives_the_regressed_template(self, model):
        rest = k.rest_joints(model, np.zeros(2))
        assert rest.shape == (24, 3)
        np.testing.assert_allclose(rest, model.v_template, atol=1e-12)

    def test_betas_apply_the_shape_directions(self, model):
        rest = k.rest_joints(model, np.array([1.0, 0.0]))
        np.testing.assert_allclose(rest, model.v_template + model.shapedirs[:, :, 0], atol=1e-12)

    def test_betas_are_padded_or_truncated_to_the_model_width(self, model):
        wide = k.rest_joints(model, np.array([0.5, -0.2, 7.0, 7.0]))
        exact = k.rest_joints(model, np.array([0.5, -0.2]))
        short = k.rest_joints(model, np.array([0.5]))
        np.testing.assert_array_equal(wide, exact)
        np.testing.assert_array_equal(short, k.rest_joints(model, np.array([0.5, 0.0])))

    def test_shaped_vertices_feed_the_regressor(self, model):
        betas = np.array([0.3, 0.1])
        np.testing.assert_array_equal(k.rest_joints(model, betas), model.J_regressor @ k.shaped_vertices(model, betas))


class TestSegmentLengths:
    def test_lengths_are_the_bone_norms_and_positive(self, model):
        rest = k.rest_joints(model, np.zeros(2))
        lengths = k.segment_lengths(rest)
        assert lengths.shape == (len(SEGMENTS),) and np.all(lengths > 0)
        for i, (_name, proximal, distal) in enumerate(SEGMENTS):
            assert lengths[i] == pytest.approx(float(np.linalg.norm(rest[distal] - rest[proximal])), abs=1e-12)

    def test_single_link_lengths_survive_any_pose(self, model, rng):
        # A segment whose ends are parent and child is one rigid bone. `trunk` runs pelvis ->
        # spine3 across spine1 and spine2, so its length legitimately changes with the lumbar pose.
        rest = k.rest_joints(model, np.zeros(2))
        positions, _ = k.fk_batch(rest, rng.normal(0, 0.5, (6, 24, 3)), rng.normal(0, 1, (6, 3)))
        rigid = [i for i, (_n, proximal, distal) in enumerate(SEGMENTS) if PARENTS[distal] == proximal]
        spanning = [name for name, proximal, distal in SEGMENTS if PARENTS[distal] != proximal]
        assert spanning == ["trunk"]
        for frame in positions:
            np.testing.assert_allclose(
                k.segment_lengths(frame)[rigid], k.segment_lengths(rest)[rigid], atol=1e-12
            )
        for joint in range(1, 24):
            bone = np.linalg.norm(positions[:, joint] - positions[:, PARENTS[joint]], axis=1)
            assert bone.std() < 1e-12


class TestForwardKinematics:
    def test_identity_pose_returns_the_rest_skeleton_shifted_by_trans(self, model):
        rest = k.rest_joints(model, np.zeros(2))
        trans = np.array([0.3, -1.2, 4.0])
        positions, world = k.fk(rest, _identity_pose(), trans)
        np.testing.assert_allclose(positions, rest + trans, atol=1e-12)
        np.testing.assert_array_equal(world, _identity_pose())

    def test_the_pelvis_lands_at_rest_zero_plus_trans(self, model, rng):
        """``trans`` moves the model origin; the pelvis is at ``j_0`` in the rest skeleton."""
        rest = k.rest_joints(model, np.array([0.7, 0.0]))
        offset = np.array([0.05, 0.9, -0.02])
        shifted = rest + offset                       # a rest skeleton whose pelvis is off the origin
        poses = rng.normal(0, 0.6, (5, 24, 3))
        trans = rng.normal(0, 2, (5, 3))
        positions, _ = k.fk_batch(shifted, poses, trans)
        np.testing.assert_array_equal(positions[:, 0], shifted[0] + trans)
        at_origin, _ = k.fk_batch(shifted, poses, -np.broadcast_to(shifted[0], (5, 3)))
        np.testing.assert_array_equal(at_origin[:, 0], 0.0)

    def test_a_shoulder_rotation_swings_the_arm_down(self, model):
        rest = k.rest_joints(model, np.zeros(2))
        pose = _identity_pose()
        pose[16] = Rotation.from_rotvec([0, 0, -np.pi / 2]).as_matrix()
        positions, _ = k.fk(rest, pose, np.zeros(3))
        assert positions[20][1] < rest[20][1] - 0.1
        for joint in (0, 1, 2, 7, 9, 12, 15, 17, 21):
            np.testing.assert_allclose(positions[joint], rest[joint], atol=1e-12)

    def test_axis_angle_and_matrix_poses_agree(self, model, rng):
        rest = k.rest_joints(model, np.zeros(2))
        vectors = rng.normal(0, 0.5, (4, 24, 3))
        trans = rng.normal(0, 1, (4, 3))
        from_vectors = k.fk_batch(rest, vectors, trans)
        from_matrices = k.fk_batch(rest, axis_angle_to_matrix(vectors), trans)
        np.testing.assert_array_equal(from_vectors[0], from_matrices[0])
        np.testing.assert_array_equal(from_vectors[1], from_matrices[1])

    def test_single_frame_equals_the_batch_entry(self, model, rng):
        rest = k.rest_joints(model, np.zeros(2))
        vectors = rng.normal(0, 0.5, (4, 24, 3))
        trans = rng.normal(0, 1, (4, 3))
        positions, world = k.fk_batch(rest, vectors, trans)
        single_p, single_w = k.fk(rest, vectors[2], trans[2])
        np.testing.assert_array_equal(single_p, positions[2])
        np.testing.assert_array_equal(single_w, world[2])

    def test_world_rotations_are_the_accumulated_locals(self, model, rng):
        rest = k.rest_joints(model, np.zeros(2))
        local = axis_angle_to_matrix(rng.normal(0, 0.5, (3, 24, 3)))
        _, world = k.fk_batch(rest, local, np.zeros((3, 3)))
        np.testing.assert_array_equal(world, k.global_rotations(local))
        for joint in range(1, 24):
            np.testing.assert_allclose(world[:, joint], world[:, PARENTS[joint]] @ local[:, joint], atol=1e-14)

    def test_local_and_global_rotations_are_inverse(self, rng):
        local = axis_angle_to_matrix(rng.normal(0, 0.7, (5, 24, 3)))
        world = k.global_rotations(local)
        np.testing.assert_allclose(k.local_rotations(world), local, atol=1e-12)
        np.testing.assert_allclose(k.global_rotations(k.local_rotations(world)), world, atol=1e-12)
        np.testing.assert_array_equal(k.local_rotations(world)[:, 0], world[:, 0])

    @pytest.mark.parametrize(
        "rest_shape,pose_shape,trans_shape",
        [((23, 3), (2, 24, 3), (2, 3)), ((24, 3), (2, 23, 3), (2, 3)), ((24, 3), (2, 24, 3), (3, 3)),
         ((24, 3), (2, 24, 4), (2, 3)), ((24, 3), (24, 3, 3), (2, 3))],
    )
    def test_wrong_shapes_are_refused(self, rest_shape, pose_shape, trans_shape):
        with pytest.raises(ValueError):
            k.fk_batch(np.zeros(rest_shape), np.zeros(pose_shape), np.zeros(trans_shape))

    def test_single_frame_refuses_batches(self, model):
        rest = k.rest_joints(model, np.zeros(2))
        with pytest.raises(ValueError):
            k.fk(rest, np.zeros((2, 24, 3)), np.zeros(3))
        with pytest.raises(ValueError):
            k.fk(rest, np.zeros((24, 3)), np.zeros((1, 3)))
