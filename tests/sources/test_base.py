import numpy as np
import pytest

from smpl18.sources import (
    Format,
    JointCentres,
    MarkerTrajectories,
    Provenance,
    SegmentPlacement,
    SkeletonModel,
    SkeletonMotion,
    SmplParameters,
    SourceKind,
    Subject,
    Trial,
    UpAxis,
)

SUBJECT = Subject(id="s", gender="neutral")
TRIAL = Trial(id="t", subject="s")


def test_enumerations_spell_the_schema_words() -> None:
    assert [k.value for k in SourceKind] == [
        "smpl_parameters", "skeleton_motion", "joint_centres", "marker_trajectories"]
    assert Format("osim_mot") is Format.OSIM_MOT
    assert [p.value for p in Provenance] == ["measured", "derived", "absent"]
    assert str(UpAxis.Z) == "z"


def test_parameters_validate_shapes_and_default_the_mask() -> None:
    params = SmplParameters(SUBJECT, TRIAL, poses=np.zeros((3, 24, 3)), betas=np.zeros((1, 10)),
                            trans=np.zeros((3, 3)), fps=100, up_axis="y")
    assert params.frame_valid.tolist() == [True, True, True]
    assert params.betas.shape == (10,)
    assert params.up_axis is UpAxis.Y
    assert params.frames == 3 and params.joints == 24
    with pytest.raises(ValueError, match="trans"):
        SmplParameters(SUBJECT, TRIAL, np.zeros((3, 24, 3)), np.zeros(10), np.zeros((2, 3)),
                       fps=100, up_axis="y")
    with pytest.raises(ValueError, match="poses"):
        SmplParameters(SUBJECT, TRIAL, np.zeros((3, 72)), np.zeros(10), np.zeros((3, 3)),
                       fps=100, up_axis="y")
    with pytest.raises(ValueError, match="fps"):
        SmplParameters(SUBJECT, TRIAL, np.zeros((3, 24, 3)), np.zeros(10), np.zeros((3, 3)),
                       fps=0, up_axis="y")
    with pytest.raises(ValueError):
        SmplParameters(SUBJECT, TRIAL, np.zeros((3, 24, 3)), np.zeros(10), np.zeros((3, 3)),
                       fps=100, up_axis="x")


class TwoBodyModel(SkeletonModel):
    """A root and one child hinged about z, enough to exercise the protocol."""

    bodies = ("root", "child")
    joints = ("ground_root", "hinge")
    parents = {"root": None, "child": "root"}
    coordinate_names = ("root_tx", "hinge_angle")

    def rest_transform(self, body):
        return np.eye(3), (np.zeros(3) if body == "root" else np.array([1.0, 0.0, 0.0]))

    def motion(self, coordinates):
        angle = np.asarray(coordinates["hinge_angle"])
        frames = angle.shape[0]
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.zeros((frames, 3, 3))
        rotation[:, 0, 0], rotation[:, 0, 1] = c, -s
        rotation[:, 1, 0], rotation[:, 1, 1] = s, c
        rotation[:, 2, 2] = 1.0
        root = np.stack([coordinates["root_tx"], np.zeros(frames), np.zeros(frames)], axis=1)
        child = root + np.array([1.0, 0.0, 0.0])
        return SegmentPlacement(
            rotations={"root": np.tile(np.eye(3), (frames, 1, 1)), "child": rotation},
            positions={"root": root, "child": child},
            joint_centres={"ground_root": root, "hinge": child},
        )


def test_skeleton_model_is_abstract() -> None:
    with pytest.raises(TypeError):
        SkeletonModel()


def test_skeleton_motion_runs_the_model_over_its_coordinates() -> None:
    motion = SkeletonMotion(SUBJECT, TRIAL, TwoBodyModel(), ("root_tx", "hinge_angle"),
                            np.array([[0.0, 0.0], [1.0, np.pi / 2]]), fps=50, up_axis="y",
                            driven_bodies=("root",))
    placed = motion.placed()
    np.testing.assert_allclose(placed.joint_centres["hinge"][1], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(placed.rotations["child"][1, 1, 0], 1.0)
    assert motion.frames == 2 and motion.frame_valid.all()
    with pytest.raises(ValueError, match="coordinates"):
        SkeletonMotion(SUBJECT, TRIAL, TwoBodyModel(), ("root_tx", "hinge_angle"),
                       np.zeros((2, 3)), fps=50, up_axis="y")


def test_joint_centres_derive_validity_from_finiteness() -> None:
    positions = np.zeros((2, 2, 3))
    positions[1, 0, 2] = np.nan
    centres = JointCentres(SUBJECT, TRIAL, ("a", "b"), positions, fps=100, up_axis="z",
                           segment_rotations={"body": np.tile(np.eye(3), (2, 1, 1))})
    assert centres.valid.tolist() == [[True, True], [False, True]]
    assert centres.centre("b").shape == (2, 3)
    with pytest.raises(ValueError, match="segment rotation"):
        JointCentres(SUBJECT, TRIAL, ("a", "b"), positions, fps=100, up_axis="z",
                     segment_rotations={"body": np.eye(3)})
    with pytest.raises(ValueError, match="positions"):
        JointCentres(SUBJECT, TRIAL, ("a",), positions, fps=100, up_axis="z")


def test_marker_trajectories_validate_labels_and_mask() -> None:
    markers = MarkerTrajectories(SUBJECT, TRIAL, ("L", "R"), np.zeros((4, 2, 3)), fps=200,
                                 up_axis="y", valid=np.ones((4, 2), dtype=bool))
    assert markers.frames == 4 and markers.labels == ("L", "R")
    with pytest.raises(ValueError, match="valid"):
        MarkerTrajectories(SUBJECT, TRIAL, ("L", "R"), np.zeros((4, 2, 3)), fps=200,
                           up_axis="y", valid=np.ones((4, 3), dtype=bool))
