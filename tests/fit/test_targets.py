import numpy as np
import pytest

from smpl18.fit.targets import (
    OrientationTargets,
    PositionTargets,
    Targets,
    concatenate,
    provenance_for,
)
from smpl18.skeleton.definition import JOINT_NAMES


def _positions(frames=3, joints=(1, 4)):
    return PositionTargets(joints, np.zeros((frames, len(joints), 3)),
                           np.ones((frames, len(joints)), bool), np.ones(len(joints)),
                           [f"c{j}" for j in joints])


def by_name(codes):
    return dict(zip(JOINT_NAMES, codes))


def test_non_finite_samples_are_invalid_and_blanked() -> None:
    values = np.zeros((2, 1, 3))
    values[1, 0, 0] = np.nan
    targets = PositionTargets([1], values, np.ones((2, 1), bool), [1.0], ["hip"])
    assert targets.valid.tolist() == [[True], [False]]
    assert np.isnan(targets.positions[1]).all()


def test_joint_names_and_indices_both_name_a_joint() -> None:
    targets = PositionTargets(["left_hip", 4], np.zeros((1, 2, 3)), np.ones((1, 2), bool),
                              [1, 1], ["a", "b"])
    assert targets.joints == (1, 4)
    with pytest.raises(ValueError):
        PositionTargets(["nose"], np.zeros((1, 1, 3)), np.ones((1, 1), bool), [1], ["a"])


def test_weights_and_labels_must_cover_every_column() -> None:
    with pytest.raises(ValueError, match="weights"):
        PositionTargets([1, 4], np.zeros((1, 2, 3)), np.ones((1, 2), bool), [1.0], ["a", "b"])
    with pytest.raises(ValueError, match="weights"):
        PositionTargets([1], np.zeros((1, 1, 3)), np.ones((1, 1), bool), [-1.0], ["a"])
    with pytest.raises(ValueError, match="labels"):
        PositionTargets([1], np.zeros((1, 1, 3)), np.ones((1, 1), bool), [1.0], [])


def test_invalid_orientations_become_identity() -> None:
    rotations = np.full((2, 1, 3, 3), np.nan)
    rotations[0, 0] = np.eye(3)
    targets = OrientationTargets([0], rotations, np.ones((2, 1), bool), [1.0], ["pelvis"])
    assert targets.valid.tolist() == [[True], [False]]
    np.testing.assert_array_equal(targets.rotations[1, 0], np.eye(3))


def test_a_bone_seen_at_both_ends_is_measured_and_a_leaf_is_not() -> None:
    codes = by_name(provenance_for(["left_hip", "left_knee", "left_ankle"]))
    assert codes["pelvis"] == "measured"
    assert codes["left_hip"] == "measured"          # hip and knee both observed
    assert codes["left_knee"] == "measured"         # knee and ankle both observed
    assert codes["left_ankle"] == "absent"          # nothing below the ankle is observed
    assert codes["spine1"] == "absent"
    assert codes["right_hip"] == "absent"


def test_a_joint_above_an_observed_one_is_derived() -> None:
    codes = by_name(provenance_for(["spine3", "neck"]))
    assert codes["spine1"] == "derived"
    assert codes["spine2"] == "derived"
    assert codes["spine3"] == "measured"            # spine3 and its child neck observed
    assert codes["neck"] == "absent"


def test_an_orientation_target_measures_its_joint() -> None:
    codes = by_name(provenance_for(["left_wrist"], ["left_wrist", "head"]))
    assert codes["left_wrist"] == "measured"
    assert codes["head"] == "measured"
    assert codes["neck"] == "derived"


def test_fill_rules_override_what_is_inferred() -> None:
    codes = by_name(provenance_for(["spine3", "neck"], fill={"spine1": "weld",
                                                             "left_foot": "distribute"}))
    assert codes["spine1"] == "absent"
    assert codes["left_foot"] == "derived"
    with pytest.raises(ValueError, match="cannot be welded"):
        provenance_for(["spine3", "neck"], fill={"spine3": "weld"})
    with pytest.raises(ValueError, match="fill rule"):
        provenance_for(["spine3"], fill={"spine1": "glue"})
    with pytest.raises(ValueError, match="not an SMPL-24 joint"):
        provenance_for(["spine3"], fill={"tail": "weld"})


def test_free_joints_leave_out_the_welded_ones() -> None:
    targets = Targets(_positions(), None, 100.0, provenance_for([1, 4]))
    assert targets.free_joints == (0, 1)            # the knee has nothing below it


def test_concatenation_needs_the_same_columns() -> None:
    first = Targets(_positions(2), None, 100.0, provenance_for([1, 4]))
    both = concatenate([first, first])
    assert both.frames == 4
    other = Targets(_positions(2, joints=(2, 5)), None, 100.0, provenance_for([2, 5]))
    with pytest.raises(ValueError, match="disagree"):
        concatenate([first, other])


def test_targets_refuse_a_bad_rate_or_mismatched_frames() -> None:
    with pytest.raises(ValueError, match="fps"):
        Targets(_positions(), None, 0.0, provenance_for([1, 4]))
    orientations = OrientationTargets([0], np.tile(np.eye(3), (5, 1, 1, 1)),
                                      np.ones((5, 1), bool), [1.0], ["pelvis"])
    with pytest.raises(ValueError, match="frame count"):
        Targets(_positions(3), orientations, 100.0, provenance_for([1, 4]))
