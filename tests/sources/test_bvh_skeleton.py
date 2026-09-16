"""BVH forward kinematics, checked against arithmetic done by hand.

The places a BVH reading goes wrong: composing the rotation channels in some order other than
the one the joint lists, carrying a child's offset by the child's own rotation instead of the
parent's, and scaling angles along with lengths. Each has a test that fails if it happens.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smpl18.formats import bvh
from smpl18.sources import SegmentPlacement, SkeletonModel
from smpl18.sources.skeleton import BvhSkeleton, coordinates_from_frames
from smpl18.sources.skeleton import bvh as bvh_skeleton

ATOL = 1e-12

# A root with position channels and a non-zero offset, a spine in ZXY, and a head with an End
# Site. Offsets are centimetres.
CLIP = """HIERARCHY
ROOT Hips
{
    OFFSET 1.0 90.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.0 10.0 2.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Head
        {
            OFFSET 0.0 30.0 0.0
            CHANNELS 3 Xrotation Yrotation Zrotation
            End Site
            {
                OFFSET 0.0 15.0 0.0
            }
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.01
10 0 -5 30 -20 45 10 20 -30 5 -15 60
0 0 0 0 0 0 0 0 0 0 0 0
"""
SCALE = 0.01


@pytest.fixture(scope="module")
def clip() -> bvh.BvhFile:
    return bvh.parse(CLIP)


@pytest.fixture(scope="module")
def skeleton(clip) -> BvhSkeleton:
    return BvhSkeleton(clip, length_scale=SCALE)


def about(axis: str, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def first_frame(skeleton, clip) -> SegmentPlacement:
    values = coordinates_from_frames(skeleton, clip)
    return skeleton.motion({name: column[:1] for name, column in values.items()})


def test_it_is_a_skeleton_model_whose_joints_are_its_bodies(skeleton):
    assert isinstance(skeleton, SkeletonModel)
    assert skeleton.bodies == skeleton.joints == ("Hips", "Spine", "Head")
    assert skeleton.parents == {"Hips": None, "Spine": "Hips", "Head": "Spine"}
    assert skeleton.joint_child_bodies == {"Hips": "Hips", "Spine": "Spine", "Head": "Head"}
    assert skeleton.length_scale == SCALE


def test_one_coordinate_per_channel_in_column_order(skeleton):
    assert skeleton.coordinate_names == (
        "Hips.Xposition", "Hips.Yposition", "Hips.Zposition",
        "Hips.Zrotation", "Hips.Xrotation", "Hips.Yrotation",
        "Spine.Zrotation", "Spine.Xrotation", "Spine.Yrotation",
        "Head.Xrotation", "Head.Yrotation", "Head.Zrotation",
    )
    assert skeleton.rotational_coordinates == skeleton.coordinate_names[3:]


def test_frames_become_radians_and_metres(skeleton, clip):
    values = coordinates_from_frames(skeleton, clip)
    assert list(values) == list(skeleton.coordinate_names)
    np.testing.assert_allclose(values["Hips.Xposition"], [0.1, 0.0])
    np.testing.assert_allclose(values["Hips.Zposition"], [-0.05, 0.0])
    np.testing.assert_allclose(values["Hips.Zrotation"], [np.pi / 6, 0.0])
    np.testing.assert_allclose(values["Head.Zrotation"], [np.pi / 3, 0.0])


def test_forward_kinematics_matches_hand_arithmetic(skeleton, clip):
    placed = first_frame(skeleton, clip)
    hips_rotation = about("z", 30) @ about("x", -20) @ about("y", 45)
    hips_position = np.array([1.0 + 10.0, 90.0 + 0.0, 0.0 - 5.0]) * SCALE
    spine_rotation = hips_rotation @ about("z", 10) @ about("x", 20) @ about("y", -30)
    spine_position = hips_position + hips_rotation @ (np.array([0.0, 10.0, 2.0]) * SCALE)
    head_rotation = spine_rotation @ about("x", 5) @ about("y", -15) @ about("z", 60)
    head_position = spine_position + spine_rotation @ (np.array([0.0, 30.0, 0.0]) * SCALE)
    tip = head_position + head_rotation @ (np.array([0.0, 15.0, 0.0]) * SCALE)

    np.testing.assert_allclose(placed.rotations["Hips"][0], hips_rotation, atol=ATOL)
    np.testing.assert_allclose(placed.positions["Hips"][0], hips_position, atol=ATOL)
    np.testing.assert_allclose(placed.rotations["Spine"][0], spine_rotation, atol=ATOL)
    np.testing.assert_allclose(placed.positions["Spine"][0], spine_position, atol=ATOL)
    np.testing.assert_allclose(placed.rotations["Head"][0], head_rotation, atol=ATOL)
    np.testing.assert_allclose(placed.positions["Head"][0], head_position, atol=ATOL)
    assert list(placed.joint_centres) == ["Hips", "Spine", "Head", "Head.end"]
    np.testing.assert_allclose(placed.joint_centres["Head"][0], head_position, atol=ATOL)
    np.testing.assert_allclose(placed.joint_centres["Head.end"][0], tip, atol=ATOL)


def test_channel_order_is_the_order_the_joint_lists(skeleton, clip):
    placed = first_frame(skeleton, clip)
    listed = about("z", 30) @ about("x", -20) @ about("y", 45)
    reversed_order = about("y", 45) @ about("x", -20) @ about("z", 30)
    np.testing.assert_allclose(placed.rotations["Hips"][0], listed, atol=ATOL)
    assert not np.allclose(placed.rotations["Hips"][0], reversed_order, atol=1e-3)
    # the same numbers under a different declared order are a different pose
    reordered = bvh.parse(CLIP.replace(
        "CHANNELS 3 Zrotation Xrotation Yrotation", "CHANNELS 3 Yrotation Xrotation Zrotation"
    ))
    other = first_frame(BvhSkeleton(reordered, length_scale=SCALE), reordered)
    assert not np.allclose(other.rotations["Spine"][0], placed.rotations["Spine"][0], atol=1e-3)


def test_the_rest_pose_is_the_scaled_offsets(skeleton, clip):
    rest = skeleton.motion(
        {name: column[1:] for name, column in coordinates_from_frames(skeleton, clip).items()}
    )
    np.testing.assert_allclose(rest.positions["Hips"][0], [0.01, 0.9, 0.0], atol=ATOL)
    np.testing.assert_allclose(rest.positions["Head"][0], [0.01, 1.3, 0.02], atol=ATOL)
    np.testing.assert_allclose(rest.joint_centres["Head.end"][0], [0.01, 1.45, 0.02], atol=ATOL)
    for joint in skeleton.joints:
        np.testing.assert_allclose(rest.rotations[joint][0], np.eye(3), atol=ATOL)
    rotation, translation = skeleton.rest_transform("Spine")
    np.testing.assert_array_equal(rotation, np.eye(3))
    np.testing.assert_allclose(translation, [0.0, 0.1, 0.02])
    with pytest.raises(KeyError, match="Neck"):
        skeleton.rest_transform("Neck")


def test_the_length_scale_applies_to_offsets_and_positions_but_not_angles(clip):
    placed = {}
    for scale in (0.01, 0.02):
        model = BvhSkeleton(clip, length_scale=scale)
        placed[scale] = first_frame(model, clip)
    for joint in ("Hips", "Spine", "Head"):
        np.testing.assert_allclose(
            placed[0.02].positions[joint], 2.0 * placed[0.01].positions[joint], atol=ATOL
        )
        np.testing.assert_allclose(
            placed[0.02].rotations[joint], placed[0.01].rotations[joint], atol=ATOL
        )


def test_a_batch_gives_what_single_frames_give(skeleton):
    rng = np.random.default_rng(5)
    block = rng.normal(0.0, 1.0, (9, len(skeleton.coordinate_names)))
    batched = skeleton.motion(dict(zip(skeleton.coordinate_names, block.T)))
    for frame in range(block.shape[0]):
        single = skeleton.motion(
            {n: block[frame:frame + 1, i] for i, n in enumerate(skeleton.coordinate_names)}
        )
        for name, centre in single.joint_centres.items():
            np.testing.assert_allclose(batched.joint_centres[name][frame], centre[0], atol=ATOL)
        for joint, rotation in single.rotations.items():
            np.testing.assert_allclose(batched.rotations[joint][frame], rotation[0], atol=ATOL)


def test_several_roots_are_each_placed_from_the_origin():
    two = bvh.parse(
        "HIERARCHY\nROOT A\n{\n OFFSET 0 0 0\n CHANNELS 1 Zrotation\n}\n"
        "ROOT B\n{\n OFFSET 100 0 0\n CHANNELS 1 Xposition\n}\n"
        "MOTION\nFrames: 1\nFrame Time: 0.1\n90 50\n"
    )
    model = BvhSkeleton(two, length_scale=0.01)
    placed = model.motion(coordinates_from_frames(model, two))
    np.testing.assert_allclose(placed.rotations["A"][0], about("z", 90), atol=ATOL)
    np.testing.assert_allclose(placed.positions["B"][0], [1.5, 0.0, 0.0], atol=ATOL)
    assert model.parents == {"A": None, "B": None}


def test_coordinates_must_be_complete_known_and_aligned(skeleton):
    full = {name: np.zeros(3) for name in skeleton.coordinate_names}
    with pytest.raises(ValueError, match=r"no values for channels \['Head.Zrotation'\]"):
        skeleton.motion({k: v for k, v in full.items() if k != "Head.Zrotation"})
    with pytest.raises(ValueError, match="does not declare.*Tail"):
        skeleton.motion({**full, "Tail.Xrotation": np.zeros(3)})
    with pytest.raises(ValueError, match="disagree on the number of frames"):
        skeleton.motion({**full, "Head.Zrotation": np.zeros(2)})
    with pytest.raises(ValueError, match=r"must be a \(T,\) array"):
        skeleton.motion({**full, "Head.Zrotation": np.zeros((3, 1))})


def test_the_frames_must_belong_to_the_skeleton(skeleton):
    other = bvh.parse(CLIP.replace("CHANNELS 3 Xrotation Yrotation Zrotation",
                                   "CHANNELS 3 Zrotation Yrotation Xrotation"))
    with pytest.raises(ValueError, match="differ from the skeleton's"):
        coordinates_from_frames(skeleton, other)


@pytest.mark.parametrize("scale", [0.0, -0.01, float("nan"), float("inf")])
def test_the_length_scale_must_be_a_positive_number(clip, scale):
    with pytest.raises(ValueError, match="length_scale"):
        BvhSkeleton(clip, length_scale=scale)


def test_the_length_scale_has_no_default(clip):
    with pytest.raises(TypeError):
        BvhSkeleton(clip)  # type: ignore[call-arg]


def test_names_that_cannot_key_the_placement_are_refused():
    twice = bvh.parse(CLIP.replace("JOINT Head", "JOINT Spine"))
    with pytest.raises(ValueError, match=r"repeated: \['Spine'\]"):
        BvhSkeleton(twice, length_scale=SCALE)
    clash = bvh.parse(
        "HIERARCHY\nROOT Head\n{\n OFFSET 0 0 0\n CHANNELS 1 Zrotation\n"
        " JOINT Head.end\n {\n  OFFSET 0 1 0\n  CHANNELS 1 Zrotation\n }\n"
        " End Site\n {\n  OFFSET 0 1 0\n }\n}\n"
        "MOTION\nFrames: 1\nFrame Time: 0.1\n0 0\n"
    )
    with pytest.raises(ValueError, match="collide with End Site"):
        BvhSkeleton(clash, length_scale=SCALE)


def test_read_builds_the_skeleton_from_a_file(tmp_path: Path):
    path = tmp_path / "clip.bvh"
    path.write_text(CLIP, encoding="utf-8")
    model = bvh_skeleton.read(path, length_scale=SCALE)
    assert model.joints == ("Hips", "Spine", "Head")
    np.testing.assert_array_equal(model.file.frames, bvh.parse(CLIP).frames)
