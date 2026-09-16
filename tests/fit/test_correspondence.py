from pathlib import Path

import numpy as np
import pytest

from smpl18.fit.correspondence import Correspondence, CorrespondenceError
from smpl18.skeleton.definition import JOINT_NAMES
from smpl18.skeleton.frames import frame_change
from smpl18.sources.base import SegmentPlacement, SkeletonModel

PACKAGE = Path(__file__).resolve().parents[2]
TABLES = sorted((PACKAGE / "configs" / "correspondence").glob("*.yaml"))


def table(**joints):
    return {"schema": "smpl18_correspondence_v1", "id": "t", "names": "centres",
            "joints": joints}


@pytest.mark.parametrize("path", TABLES, ids=[p.stem for p in TABLES])
def test_every_shipped_table_loads(path) -> None:
    loaded = Correspondence.load(path)
    assert loaded.sha256 and loaded.entries
    assert set(loaded.entries) <= set(JOINT_NAMES)


def test_unknown_keys_and_bad_entries_are_refused() -> None:
    with pytest.raises(CorrespondenceError, match="unknown keys"):
        Correspondence.from_mapping({**table(), "extra": 1})
    with pytest.raises(CorrespondenceError, match="not an SMPL-24 joint"):
        Correspondence.from_mapping(table(tail={"centre": "X"}))
    with pytest.raises(CorrespondenceError, match="source or a fill rule"):
        Correspondence.from_mapping(table(left_hip={}))
    with pytest.raises(CorrespondenceError, match="takes no fill rule"):
        Correspondence.from_mapping(table(left_hip={"centre": "LHJC", "fill": "weld"}))
    with pytest.raises(CorrespondenceError, match="unknown keys"):
        Correspondence.from_mapping(table(left_hip={"source": ["hip_l"]}))
    with pytest.raises(CorrespondenceError, match="names must be"):
        Correspondence.from_mapping({**table(), "names": "markers"})
    with pytest.raises(CorrespondenceError, match="true or false"):
        Correspondence.from_mapping(table(left_hip={"centre": "LHJC", "position": "no"}))


def test_centres_map_by_label_and_report_what_is_missing() -> None:
    loaded = Correspondence.from_mapping({
        **table(left_hip={"centre": "LHJC"}, left_knee={"centre": "LKJC", "weight": 0.5},
                left_ankle={"centre": "LAJC"}, right_hip={"centre": "RHJC", "segment": "RTH"},
                spine1={"fill": "weld"}),
        "prefixes": ["sub1:"],
        "aliases": {"L_KNEE": "LKJC"},
    })
    names = ("sub1:LHJC", "L_KNEE", "LAJC", "RHJC")
    positions = np.random.default_rng(0).normal(size=(5, 4, 3))
    valid = np.ones((5, 4), bool)
    valid[2, 0] = False
    rotations = {"RTH": np.tile(np.eye(3), (5, 1, 1))}
    targets, use = loaded.centre_targets(names, positions, valid, fps=50.0, rotations=rotations)
    assert targets.positions.joints == (1, 2, 4, 7)
    np.testing.assert_array_equal(targets.positions.positions[:, 2], positions[:, 1])
    assert targets.positions.weights.tolist() == [1.0, 1.0, 0.5, 1.0]
    assert not targets.positions.valid[2, 0]
    assert targets.orientations.joints == (2,)
    assert use.positions["left_knee"] == "L_KNEE"
    assert use.fill == {"spine1": "weld"}
    assert dict(zip(JOINT_NAMES, targets.provenance))["spine1"] == "absent"

    partial = ("LHJC", "LAJC")
    _, use = loaded.centre_targets(partial, positions[:, :2], valid[:, :2], fps=50.0)
    assert use.missing == ["LKJC", "RHJC", "RTH"]


def test_a_centres_table_is_not_a_skeleton_table() -> None:
    loaded = Correspondence.from_mapping(table(left_hip={"centre": "LHJC"}))
    with pytest.raises(CorrespondenceError, match="names centres"):
        loaded.skeleton_targets(None, None, frame=frame_change("y", "y"), fps=1.0)


class Chain(SkeletonModel):
    """pelvis -> thigh -> shank, joints named after what they move."""

    bodies = ("pelvis", "thigh", "shank")
    joints = ("root", "hip", "knee")
    parents = {"pelvis": None, "thigh": "pelvis", "shank": "thigh"}
    joint_child_bodies = {"root": "pelvis", "hip": "thigh", "knee": "shank"}
    coordinate_names = ()
    rotational_coordinates = ()

    def rest_transform(self, body):
        return np.eye(3), np.zeros(3)

    def motion(self, coordinates):
        frames = 4
        turn = np.tile(np.eye(3), (frames, 1, 1))
        centre = np.arange(frames * 3, dtype=float).reshape(frames, 3)
        return SegmentPlacement(
            rotations={"pelvis": turn, "thigh": turn * 1.0, "shank": turn},
            positions={body: centre for body in self.bodies},
            joint_centres={"root": centre, "hip": centre + 1, "knee": centre + 2},
        )


def test_skeleton_targets_take_centres_and_child_bodies_in_the_output_frame() -> None:
    loaded = Correspondence.from_mapping({
        "schema": "smpl18_correspondence_v1", "id": "chain", "names": "opensim_joints",
        "joints": {
            "pelvis": {"source": ["root"], "position": False},
            "left_hip": {"source": ["hip"]},
            "left_knee": {"source": ["knee", "ankle"]},       # the ankle is not in this model
        },
    })
    skeleton = Chain()
    change = frame_change("z", "y")
    targets, use = loaded.skeleton_targets(skeleton, skeleton.motion({}), frame=change, fps=30.0)
    assert targets.positions.joints == (1, 4)
    np.testing.assert_allclose(targets.positions.positions[:, 0],
                               change.apply(np.arange(12.0).reshape(4, 3) + 1))
    assert targets.orientations.joints == (0, 1)
    np.testing.assert_allclose(targets.orientations.rotations[:, 0],
                               np.broadcast_to(change.rotation, (4, 3, 3)))
    assert use.missing == ["ankle"]
    assert use.orientations == {"pelvis": "pelvis", "left_hip": "thigh"}


def test_the_lumbar_block_places_and_turns_the_spine() -> None:
    loaded = Correspondence.from_mapping({
        "schema": "smpl18_correspondence_v1", "id": "lumbar", "names": "opensim_joints",
        "joints": {"left_hip": {"source": ["hip"]}, "spine1": {"fill": "distribute"},
                   "spine2": {"fill": "distribute"}, "spine3": {"fill": "distribute"}},
        "lumbar": {"source_joint": "knee", "smpl": ["spine1", "spine2", "spine3"],
                   "split": "equal_thirds", "centre_on": "spine1"},
        "trunk_body": "shank",
    })
    skeleton = Chain()
    targets, use = loaded.skeleton_targets(skeleton, skeleton.motion({}),
                                           frame=frame_change("y", "y"), fps=30.0)
    assert 3 in targets.positions.joints and use.positions["spine1"] == "knee"
    assert 9 in targets.orientations.joints and use.orientations["spine3"] == "shank"
    codes = dict(zip(JOINT_NAMES, targets.provenance))
    assert codes["spine3"] == "measured" and codes["spine2"] == "derived"
    with pytest.raises(CorrespondenceError, match="centre_on"):
        Correspondence.from_mapping({**table(), "lumbar": {
            "source_joint": "back", "smpl": ["spine1"], "centre_on": "spine3"}})


def test_two_targets_on_one_joint_keep_apart() -> None:
    loaded = Correspondence.from_mapping({
        "schema": "smpl18_correspondence_v1", "id": "twice", "names": "opensim_joints",
        "joints": {"spine1": {"source": ["knee"]}, "left_hip": {"source": ["hip"]}},
        "lumbar": {"source_joint": "knee", "smpl": ["spine1", "spine2", "spine3"],
                   "centre_on": "spine1"},
    })
    skeleton = Chain()
    targets, _ = loaded.skeleton_targets(skeleton, skeleton.motion({}),
                                         frame=frame_change("y", "y"), fps=30.0)
    labels = targets.positions.labels
    assert len(set(labels)) == len(labels)
    assert "spine1 <- knee" in labels and "spine1 <- knee #2" in labels


def test_the_opensim_table_reads_gait2392_knee_names() -> None:
    loaded = Correspondence.load(PACKAGE / "configs" / "correspondence" / "opensim_rajagopal.yaml")
    assert loaded._canonical("knee_l") == "walker_knee_l"
    assert loaded._canonical("walker_knee_r") == "walker_knee_r"
