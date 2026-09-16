"""Fitting the four constants: what it recovers, what it improves, and what it refuses.

The fit exists because the mean rotation is only the right answer when the frozen joints are
still. Both cases are here: with still joints the fit must reproduce the constant exactly, and
with moving ones it must beat the mean it started from -- and report both numbers, since a corpus
that only carried the fitted residual could not tell a reader whether the fit was worth running.
"""

from __future__ import annotations

import numpy as np
import pytest

from smpl18 import reduce
from smpl18.skeleton.definition import (
    AFFECTED_BY_FREEZE,
    FROZEN_JOINT_NAMES,
    FROZEN_JOINTS,
    JOINT_NAMES,
)
from smpl18.skeleton.kinematics import global_rotations
from smpl18.skeleton.rotations import axis_angle_to_matrix

from .conftest import identity_series, swing, wandering_series


def still_series(rng: np.random.Generator, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """A take whose frozen joints never move, with the constant they are held at."""
    constants = axis_angle_to_matrix(rng.normal(0.0, 0.4, (len(FROZEN_JOINTS), 3)))
    local = wandering_series(rng, frames)
    for column, joint in enumerate(FROZEN_JOINTS):
        local[:, joint] = constants[column]
    return local, constants


def moving_series(rng: np.random.Generator, frames: int) -> np.ndarray:
    """A take whose frozen joints swing: lumbar flexion and bend, and both shoulder girdles.

    The amplitudes are about 20 degrees, a trunk and a pair of shoulders that move enough for the
    reduction to cost something. What the fit can win back grows with them: the mean rotation is
    first-order right, so the gain is the curvature the mean ignores.
    """
    local = wandering_series(rng, frames)
    local[:, 3] = swing(frames, (1.0, 0.0, 0.0), 0.35, 0.0)
    local[:, 6] = swing(frames, (0.0, 0.0, 1.0), 0.30, 1.0)
    local[:, 13] = swing(frames, (0.0, 0.0, 1.0), 0.40, 2.0)
    local[:, 14] = swing(frames, (0.0, 1.0, 0.0), 0.40, 4.0)
    return local


class TestStillJoints:
    def test_the_constant_is_recovered_and_nothing_moves(self, rest, rng, settings):
        local, constants = still_series(rng, 30)
        result = reduce.fit_constants(local, rest, settings=settings)
        np.testing.assert_allclose(result.constants, constants, atol=1e-9)
        assert result.initial.rms_m < 1e-12 and result.fitted.rms_m < 1e-12
        assert result.fitted.max_m < 1e-12
        assert result.success

    def test_the_rest_pose_is_reduced_to_itself(self, rest, settings):
        local = identity_series(20)
        result = reduce.fit_constants(local, rest, settings=settings)
        identities = np.broadcast_to(np.eye(3), (len(FROZEN_JOINTS), 3, 3))
        np.testing.assert_allclose(result.constants, identities, atol=1e-12)
        assert result.fitted.rms_m == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_array_equal(reduce.apply(local, result.constants), local)


class TestMovingJoints:
    def test_the_fit_beats_the_mean_it_started_from(self, rest, rng, settings):
        local = moving_series(rng, 60)
        result = reduce.fit_constants(local, rest, settings=settings)
        assert result.initial.rms_m > 1e-3, "the fixture must actually cost something"
        assert result.fitted.rms_m < result.initial.rms_m
        # Not knife-edge: a fit that only tied with its own starting guess would pass a bare `<`
        # on rounding alone, and a corpus recording that pair would be recording nothing.
        assert result.fitted.rms_m < 0.99 * result.initial.rms_m
        assert result.fitted.max_m > 0.0
        assert result.success

    def test_both_residuals_describe_the_constants_returned(self, rest, rng, settings):
        """With fewer frames than the sample size every frame is measured, so this is exact."""
        local = moving_series(rng, 30)
        result = reduce.fit_constants(local, rest, settings=settings)
        assert result.frames == 30
        fitted = reduce.residual(local, rest, result.constants)
        initial = reduce.residual(local, rest, reduce.initial_constants(local))
        assert fitted == result.fitted and initial == result.initial

    def test_the_orientations_still_survive_the_fitted_constants(self, rest, rng, settings):
        local = moving_series(rng, 24)
        result = reduce.fit_constants(local, rest, settings=settings)
        before = global_rotations(local)
        after = global_rotations(reduce.apply(local, result.constants))
        for joint in reduce.names_18():
            index = JOINT_NAMES.index(joint)
            np.testing.assert_allclose(after[:, index], before[:, index], atol=1e-12)


class TestReporting:
    def test_the_result_carries_plain_types_for_the_corpus(self, rest, rng, settings):
        result = reduce.fit_constants(moving_series(rng, 50), rest, settings=settings)
        assert result.joint_names == FROZEN_JOINT_NAMES
        assert result.constants.shape == (4, 3, 3)
        assert result.rotation_vectors.shape == (4, 3)
        assert type(result.frames) is int and type(result.success) is bool
        assert isinstance(result.message, str) and result.message
        for error in (result.initial, result.fitted):
            assert type(error.rms_m) is float and type(error.max_m) is float

    def test_per_joint_rms_splits_the_pooled_number(self, rest, rng, settings):
        local = moving_series(rng, 30)
        result = reduce.fit_constants(local, rest, settings=settings)
        per_joint = reduce.per_joint_rms(local, rest, result.constants)
        assert list(per_joint) == [JOINT_NAMES[j] for j in AFFECTED_BY_FREEZE]
        pooled = np.sqrt(np.mean(np.square(list(per_joint.values()))))
        assert pooled == pytest.approx(result.fitted.rms_m, rel=1e-12)
        assert max(per_joint.values()) <= result.fitted.max_m

    def test_only_the_sampled_frames_are_measured(self, rest, rng, settings):
        settings["reduce"]["sample_frames"] = 11
        result = reduce.fit_constants(moving_series(rng, 200), rest, settings=settings)
        assert result.frames == 11

    def test_frames_are_sampled_evenly_across_the_take(self):
        np.testing.assert_array_equal(reduce.sample_indices(4, 10), np.arange(4))
        np.testing.assert_array_equal(reduce.sample_indices(9, 5), [0, 2, 4, 6, 8])
        assert len(set(reduce.sample_indices(1000, 40))) == 40


class TestDeterminism:
    def test_the_same_input_and_settings_give_the_same_constants(self, rest, rng, settings):
        local = moving_series(rng, 45)
        first = reduce.fit_constants(local, rest, settings=settings)
        second = reduce.fit_constants(local, rest, settings=settings)
        np.testing.assert_array_equal(first.constants, second.constants)
        assert first.fitted == second.fitted and first.initial == second.initial
        assert (first.frames, first.success, first.message) == (
            second.frames, second.success, second.message
        )


class TestSettings:
    @pytest.mark.parametrize("key", ["sample_frames", "optimiser", "max_evaluations"])
    def test_a_missing_key_is_named_rather_than_guessed(self, rest, rng, settings, key):
        del settings["reduce"][key]
        with pytest.raises(ValueError, match=f"reduce.{key}"):
            reduce.fit_constants(wandering_series(rng, 5), rest, settings=settings)

    def test_a_missing_section_says_what_it_needs(self, rest, rng):
        with pytest.raises(ValueError, match="reduce.sample_frames"):
            reduce.fit_constants(wandering_series(rng, 5), rest, settings={})

    def test_counts_must_be_positive(self, settings):
        settings["reduce"]["sample_frames"] = 0
        with pytest.raises(ValueError, match="at least 1"):
            reduce.FitSettings.from_settings(settings)

    def test_the_section_is_read_into_the_three_numbers(self, settings):
        options = reduce.FitSettings.from_settings(settings)
        assert (options.sample_frames, options.optimiser, options.max_evaluations) == (
            40, "trf", 200
        )
