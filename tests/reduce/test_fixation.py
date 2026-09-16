"""The exact-absorption property, the 18-joint selection, and the rest pose.

The first test is the one the whole reduction rests on: whatever the constants, no world
orientation below a frozen joint changes. If it ever fails, the constants stop being free and the
fit in test_fit.py is solving the wrong problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from smpl18 import reduce
from smpl18.skeleton.definition import (
    AFFECTED_BY_FREEZE,
    FROZEN_JOINTS,
    JOINT18_NAMES,
    JOINT_NAMES,
    KEEP18,
    NUM_JOINTS,
)
from smpl18.skeleton.kinematics import global_rotations
from smpl18.skeleton.rotations import axis_angle_to_matrix

from .conftest import identity_series, wandering_series


def arbitrary_constants(rng: np.random.Generator, scale: float = 0.8) -> np.ndarray:
    return axis_angle_to_matrix(rng.normal(0.0, scale, (len(FROZEN_JOINTS), 3)))


class TestOrientationSurvives:
    def test_every_world_orientation_but_the_frozen_joints_is_unchanged(self, rng):
        """The absorption is exact, so this holds for constants nobody chose sensibly."""
        local = wandering_series(rng, 12)
        for _trial in range(5):
            reduced = reduce.apply(local, arbitrary_constants(rng, scale=1.5))
            before = global_rotations(local)
            after = global_rotations(reduced)
            for joint in range(NUM_JOINTS):
                if joint in FROZEN_JOINTS:
                    continue
                np.testing.assert_allclose(
                    after[:, joint], before[:, joint], atol=1e-12,
                    err_msg=f"{JOINT_NAMES[joint]} turned",
                )

    def test_the_frozen_joints_hold_their_constant_on_every_frame(self, rng):
        local = wandering_series(rng, 7)
        constants = arbitrary_constants(rng)
        reduced = reduce.apply(local, constants)
        for column, joint in enumerate(FROZEN_JOINTS):
            for frame in range(reduced.shape[0]):
                np.testing.assert_array_equal(reduced[frame, joint], constants[column])

    def test_joints_that_neither_freeze_nor_absorb_come_through_untouched(self, rng):
        local = wandering_series(rng, 7)
        reduced = reduce.apply(local, arbitrary_constants(rng))
        rewritten = set(FROZEN_JOINTS) | {9, 16, 17}
        for joint in range(NUM_JOINTS):
            if joint not in rewritten:
                np.testing.assert_array_equal(reduced[:, joint], local[:, joint])

    def test_the_absorbers_are_the_only_rewritten_joints_and_they_do_move(self, rng):
        local = wandering_series(rng, 7)
        reduced = reduce.apply(local, arbitrary_constants(rng))
        for joint in (9, 16, 17):
            assert not np.allclose(reduced[:, joint], local[:, joint])

    def test_wrong_shapes_are_refused(self, rng):
        local = wandering_series(rng, 3)
        with pytest.raises(ValueError, match="local rotations"):
            reduce.apply(local[0], np.repeat(np.eye(3)[None], 4, axis=0))
        with pytest.raises(ValueError, match="constants"):
            reduce.apply(local, np.repeat(np.eye(3)[None], 3, axis=0))


class TestRestPose:
    def test_identity_rotations_reduce_to_identity(self, rest):
        """The rest pose is the sanity check: nothing to freeze, so nothing may move."""
        local = identity_series(5)
        constants = np.repeat(np.eye(3)[None], len(FROZEN_JOINTS), axis=0)
        reduced = reduce.apply(local, constants)
        np.testing.assert_array_equal(reduced, local)
        error = reduce.residual(local, rest, constants)
        assert error.rms_m == 0.0 and error.max_m == 0.0


class TestSelection:
    def test_to_18_takes_the_kept_joints_in_ascending_order(self, rng):
        local = wandering_series(rng, 6)
        reduced = reduce.apply(local, arbitrary_constants(rng))
        selected = reduce.to_18(reduced)
        assert selected.shape == (6, 18, 3, 3)
        for column, joint in enumerate(KEEP18):
            np.testing.assert_array_equal(selected[:, column], reduced[:, joint])

    def test_to_18_works_on_any_trailing_shape(self, rng):
        positions = rng.normal(0.0, 1.0, (4, NUM_JOINTS, 3))
        np.testing.assert_array_equal(reduce.to_18(positions), positions[:, list(KEEP18)])
        labels = np.array(JOINT_NAMES, dtype="<U16")[None].repeat(4, axis=0)
        assert list(reduce.to_18(labels)[0]) == list(JOINT18_NAMES)

    def test_names_18_labels_the_axis_to_18_returns(self):
        assert reduce.names_18() == JOINT18_NAMES
        assert reduce.names_18() == tuple(JOINT_NAMES[j] for j in KEEP18)
        assert len(reduce.names_18()) == 18

    def test_the_affected_joints_are_a_subset_of_the_kept_ones(self):
        assert set(AFFECTED_BY_FREEZE) <= set(KEEP18)

    def test_an_array_without_24_joints_is_refused(self, rng):
        with pytest.raises(ValueError, match="joints on axis 1"):
            reduce.to_18(rng.normal(0.0, 1.0, (4, 18, 3)))
