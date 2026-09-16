from pathlib import Path

import numpy as np
import pytest

from smpl18 import synthetic
from smpl18.model.demo import demo_model
from smpl18.skeleton.definition import JOINT_NAMES
from smpl18.skeleton.kinematics import fk_batch, rest_joints
from smpl18.sources.markers import MarkerSet, MarkerSetError, fill_gaps, length_scale

PACKAGE = Path(__file__).resolve().parents[2]
SETS = sorted((PACKAGE / "configs" / "markersets").glob("*.yaml"))
CONVENTIONAL = PACKAGE / "configs" / "markersets" / "conventional_full_body.yaml"


@pytest.fixture(scope="module")
def capture():
    rest = rest_joints(demo_model(), np.array([0.4, -0.2, 0.3, 0.1, 0.2, -0.3, 0.1, 0.0, 0.2, 0.1]))
    local, trans = synthetic.walking_motion(30, 100.0, seed=7)
    labels, markers = synthetic.conventional_markers(rest, local, trans)
    positions, world = fk_batch(rest, local, trans)
    return labels, markers, positions, world


def minimal(**extra):
    base = {"schema": "smpl18_markerset_v1", "id": "m",
            "centres": {"left_wrist": {"point": ["A", "B"]}}}
    base.update(extra)
    return base


@pytest.mark.parametrize("path", SETS, ids=[p.stem for p in SETS])
def test_every_shipped_marker_set_loads(path) -> None:
    loaded = MarkerSet.load(path)
    assert loaded.sha256 and loaded.labels and loaded.centres


def test_the_rules_give_back_the_centres_the_markers_came_from(capture) -> None:
    labels, markers, positions, _ = capture
    markerset = MarkerSet.load(CONVENTIONAL)
    result = markerset.evaluate(labels, markers, np.ones(markers.shape[:2], bool),
                                synthetic.MEASUREMENTS)
    for joint, (rule, _, _) in markerset.centres.items():
        expected = positions[:, JOINT_NAMES.index(joint)]
        # The hip regression's coefficients are printed to seven digits, and the knees and
        # ankles hang off the hips.
        tolerance = 1e-6 if joint.endswith(("hip", "knee", "ankle")) else 1e-9
        np.testing.assert_allclose(result.centres[joint], expected, atol=tolerance,
                                   err_msg=f"{joint} ({rule})")
    assert result.lengths["asis_width"] > 0.2
    for name, rotation in result.frames.items():
        np.testing.assert_allclose(rotation @ np.swapaxes(rotation, -1, -2),
                                   np.broadcast_to(np.eye(3), rotation.shape), atol=1e-12,
                                   err_msg=name)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_frames_follow_their_segments(capture) -> None:
    labels, markers, _, world = capture
    markerset = MarkerSet.load(CONVENTIONAL)
    result = markerset.evaluate(labels, markers, np.ones(markers.shape[:2], bool),
                                synthetic.MEASUREMENTS)
    for joint, (frame, _) in markerset.segments.items():
        offset = np.swapaxes(world[:, JOINT_NAMES.index(joint)], -1, -2) @ result.frames[frame]
        spread = np.abs(offset - offset[:1]).max()
        # The upper-arm frame leans on the elbow marker, which the synthetic placement puts on
        # the normal of the arm's plane; the stand-in's arm bones are not quite in one plane with
        # the flexion axis, so that frame wobbles by a few degrees. The others are rigid.
        assert spread < 0.1, f"{joint}: marker frame {frame} drifts by {spread:.3f}"


def test_targets_carry_positions_orientations_and_provenance(capture) -> None:
    labels, markers, _, _ = capture
    markerset = MarkerSet.load(CONVENTIONAL)
    targets, _ = markerset.targets(labels, markers, np.ones(markers.shape[:2], bool),
                                   synthetic.MEASUREMENTS, fps=100.0)
    assert len(targets.positions.joints) == len(markerset.centres)
    assert len(targets.orientations.joints) == len(markerset.segments)
    codes = dict(zip(JOINT_NAMES, targets.provenance))
    assert codes["left_knee"] == "measured" and codes["spine1"] == "derived"
    assert codes["left_foot"] == "absent" and codes["left_hand"] == "absent"


def test_a_lost_marker_invalidates_what_depends_on_it(capture) -> None:
    labels, markers, _, _ = capture
    valid = np.ones(markers.shape[:2], bool)
    valid[5, labels.index("LASI")] = False
    result = MarkerSet.load(CONVENTIONAL).evaluate(labels, markers, valid, synthetic.MEASUREMENTS)
    for joint in ("left_hip", "right_hip", "left_knee", "left_ankle"):
        assert not result.centre_valid[joint][5], joint
        assert np.isnan(result.centres[joint][5]).all()
    assert result.centre_valid["left_wrist"][5]
    assert not result.frame_valid["pelvis"][5]


def test_missing_measurements_and_markers_are_named(capture) -> None:
    labels, markers, _, _ = capture
    markerset = MarkerSet.load(CONVENTIONAL)
    measurements = dict(synthetic.MEASUREMENTS)
    del measurements["knee_width_left"]
    with pytest.raises(MarkerSetError, match="knee_width_left"):
        markerset.evaluate(labels, markers, np.ones(markers.shape[:2], bool), measurements)
    keep = [i for i, label in enumerate(labels) if label != "RTOE"]
    with pytest.raises(MarkerSetError, match="RTOE"):
        markerset.evaluate([labels[i] for i in keep], markers[:, keep],
                           np.ones((markers.shape[0], len(keep)), bool), synthetic.MEASUREMENTS)


def test_the_chord_is_square_and_at_the_distance() -> None:
    rng = np.random.default_rng(3)
    proximal, lateral, wand = (rng.normal(size=(20, 3)) for _ in range(3))
    data = minimal(measurements=["d"], centres={
        "left_knee": {"chord": {"proximal": "P", "lateral": "L", "plane": "W", "distance": {"d": 1.0},
                                "side": "away_from_plane"}},
        "right_knee": {"chord": {"proximal": "P", "lateral": "L", "plane": "W", "distance": {"d": 1.0},
                                 "side": "toward_plane"}},
    })
    markers = np.stack([proximal, lateral, wand], axis=1)
    result = MarkerSet.from_mapping(data).evaluate(("P", "L", "W"), markers,
                                                   np.ones((20, 3), bool), {"d": 0.1})
    away, toward = result.centres["left_knee"], result.centres["right_knee"]
    for centre in (away, toward):
        np.testing.assert_allclose(np.linalg.norm(centre - lateral, axis=1), 0.1)
        np.testing.assert_allclose(np.sum((centre - lateral) * (proximal - centre), axis=1), 0,
                                   atol=1e-12)
        normal = np.cross(proximal - lateral, wand - lateral)
        np.testing.assert_allclose(np.sum((centre - lateral) * normal, axis=1), 0, atol=1e-12)
    # On opposite sides of the line from the lateral marker to the proximal centre.
    side = np.cross(proximal - lateral, wand - lateral)
    assert np.all(np.sum(np.cross(proximal - lateral, toward - lateral) * side, axis=1) > 0)
    assert np.all(np.sum(np.cross(proximal - lateral, away - lateral) * side, axis=1) < 0)


def test_the_hinge_centre_lies_on_the_normal_through_the_marker() -> None:
    # One arm, flexing, carried about by a random rigid motion per frame.
    from scipy.spatial.transform import Rotation

    frames = 10
    turns = Rotation.random(frames, random_state=4).as_matrix()
    shifts = np.random.default_rng(4).normal(size=(frames, 3))
    angle = np.radians(np.linspace(20, 110, frames))
    local = {
        "P": np.tile([0.0, 0.3, 0.0], (frames, 1)),
        "D": np.stack([np.zeros(frames), -0.25 * np.cos(angle), 0.25 * np.sin(angle)], axis=1),
        "L": np.tile([0.04, 0.0, 0.0], (frames, 1)),
        "R": np.tile([0.2, 0.15, 0.0], (frames, 1)),
        "C": np.zeros((frames, 3)),
    }
    world = {name: np.einsum("tij,tj->ti", turns, points) + shifts
             for name, points in local.items()}
    data = minimal(centres={"left_elbow": {"hinge": {
        "proximal": "P", "distal": "D", "lateral": "L", "lateral_reference": "R",
        "distance": 0.04, "min_flexion_deg": 0, "drift_tolerance": 0.02}}})
    markers = np.stack([world[name] for name in "PDLR"], axis=1)
    result = MarkerSet.from_mapping(data).evaluate(("P", "D", "L", "R"), markers,
                                                   np.ones((frames, 4), bool), {})
    np.testing.assert_allclose(result.centres["left_elbow"], world["C"], atol=1e-9)
    assert result.centre_valid["left_elbow"].all() and not result.warnings


def test_descriptions_are_checked() -> None:
    with pytest.raises(MarkerSetError, match="schema"):
        MarkerSet.from_mapping(minimal(schema="other"))
    with pytest.raises(MarkerSetError, match="unknown keys"):
        MarkerSet.from_mapping(minimal(colour="red"))
    with pytest.raises(MarkerSetError, match="exactly one"):
        MarkerSet.from_mapping(minimal(centres={"neck": {"point": "A", "chord": {}}}))
    with pytest.raises(MarkerSetError, match="neither lengths nor"):
        MarkerSet.from_mapping(minimal(frames={"f": {"origin": "A",
                                                     "primary": {"axis": "x", "from": "A", "to": "B"},
                                                     "secondary": {"axis": "y", "from": "A", "to": "C"}}},
                                       centres={"neck": {"offset": {"frame": "f", "x": {"height": 1}}}}))
    with pytest.raises(MarkerSetError, match="same axis"):
        MarkerSet.from_mapping(minimal(frames={"f": {
            "primary": {"axis": "x", "from": "A", "to": "B"},
            "secondary": {"axis": "x", "from": "A", "to": "C"}}}))
    with pytest.raises(MarkerSetError, match="not an SMPL-24 joint"):
        MarkerSet.from_mapping(minimal(centres={"nose": {"point": "A"}}))


def test_a_rule_that_needs_itself_is_refused() -> None:
    data = minimal(frames={"f": {"origin": {"centre": "neck"},
                                 "primary": {"axis": "x", "from": "A", "to": "B"},
                                 "secondary": {"axis": "y", "from": "A", "to": "C"}}},
                   centres={"neck": {"offset": {"frame": "f", "x": 0.1}}})
    markers = np.random.default_rng(0).normal(size=(3, 3, 3))
    with pytest.raises(MarkerSetError, match="depends on itself"):
        MarkerSet.from_mapping(data).evaluate(("A", "B", "C"), markers, np.ones((3, 3), bool), {})


def test_short_interior_gaps_are_bridged_and_the_rest_left() -> None:
    frames = np.arange(20, dtype=float)
    positions = np.stack([frames, 2 * frames, -frames], axis=1)[:, None, :].repeat(2, axis=1)
    valid = np.ones((20, 2), bool)
    valid[4:7, 0] = False          # interior, 3 frames
    valid[10:15, 0] = False        # interior, 5 frames
    valid[0:2, 1] = False          # at the start
    filled, now_valid = fill_gaps(positions, valid, 3)
    assert now_valid[4:7, 0].all()
    np.testing.assert_allclose(filled[4:7, 0], positions[4:7, 0])
    assert not now_valid[10:15, 0].any()
    assert not now_valid[0:2, 1].any()
    _, same = fill_gaps(positions, valid, 0)
    np.testing.assert_array_equal(same, valid)


def test_length_units_are_named() -> None:
    assert length_scale("mm") == 0.001 and length_scale(" M ") == 1.0
    with pytest.raises(MarkerSetError, match="unknown length unit"):
        length_scale("inches")


def elbow(flexion_deg, *, reference_offset, frames=12):
    """Shoulder above the elbow, the forearm flexed forward, the flexion axis along +x."""
    angle = np.radians(np.full(frames, flexion_deg))
    centre = np.zeros((frames, 3))
    proximal = np.tile([0.0, 0.28, 0.0], (frames, 1))
    distal = np.stack([np.zeros(frames), -0.25 * np.cos(angle), 0.25 * np.sin(angle)], axis=1)
    axis = np.array([1.0, 0.0, 0.0])
    lateral = centre + 0.04 * axis
    # The upper-arm marker sits on the lateral side of the humerus, half-way up.
    reference = 0.5 * proximal + reference_offset * axis
    return np.stack([proximal, distal, lateral, reference], axis=1), centre


def hinge_set(min_flexion, tolerance=0.02):
    return MarkerSet.from_mapping(minimal(centres={"left_elbow": {"hinge": {
        "proximal": "P", "distal": "D", "lateral": "L", "lateral_reference": "R",
        "distance": 0.04, "min_flexion_deg": min_flexion, "drift_tolerance": tolerance}}}))


@pytest.mark.parametrize("flexion", [15.0, 40.0, 90.0, 120.0])
def test_the_hinge_stays_on_the_reference_side(flexion) -> None:
    # The reference is nearer the arm's plane than the epicondyle marker (3.5 cm against 4), which
    # used to let the iteration settle on the mirror solution, yet still beyond the line from the
    # shoulder centre to that marker (2 cm out at its level).
    markers, centre = elbow(flexion, reference_offset=0.035)
    result = hinge_set(10).evaluate(("P", "D", "L", "R"), markers, np.ones((12, 4), bool), {})
    np.testing.assert_allclose(result.centres["left_elbow"], centre, atol=1e-9)
    assert result.centre_valid["left_elbow"].all()


def test_a_reference_inside_the_line_is_not_trusted_when_barely_flexed() -> None:
    markers, _ = elbow(12.0, reference_offset=0.01)
    result = hinge_set(10).evaluate(("P", "D", "L", "R"), markers, np.ones((12, 4), bool), {})
    assert not result.centre_valid["left_elbow"].any()


def test_a_straight_limb_leaves_the_hinge_centre_invalid() -> None:
    markers, _ = elbow(3.0, reference_offset=0.035)
    result = hinge_set(10).evaluate(("P", "D", "L", "R"), markers, np.ones((12, 4), bool), {})
    assert not result.centre_valid["left_elbow"].any()
    with pytest.raises(MarkerSetError, match="min_flexion_deg"):
        hinge_set(180)
    with pytest.raises(MarkerSetError, match="drift_tolerance"):
        MarkerSet.from_mapping(minimal(centres={"left_elbow": {"hinge": {
            "proximal": "P", "distal": "D", "lateral": "L", "lateral_reference": "R",
            "distance": 0.04, "min_flexion_deg": 10}}}))
    with pytest.raises(MarkerSetError, match="drift_tolerance"):
        hinge_set(10, tolerance=0)


def test_the_hip_regression_uses_the_mean_leg_length(capture) -> None:
    labels, markers, positions, _ = capture
    markerset = MarkerSet.load(CONVENTIONAL)
    valid = np.ones(markers.shape[:2], bool)
    base = markerset.evaluate(labels, markers, valid, synthetic.MEASUREMENTS)
    longer = dict(synthetic.MEASUREMENTS, leg_length_right=synthetic.MEASUREMENTS["leg_length_right"] + 0.1)
    moved = markerset.evaluate(labels, markers, valid, longer)
    # The right leg's length moves the left hip too, through the mean, but less than the right.
    left = np.abs(moved.centres["left_hip"] - base.centres["left_hip"]).max()
    right = np.abs(moved.centres["right_hip"] - base.centres["right_hip"]).max()
    assert 0 < left < right


def sweep(lateral_offset, back_offset, flexions):
    """One trial flexing through ``flexions``: shoulder above, forearm flexing towards +z, the
    upper-arm marker ``lateral_offset`` out and ``back_offset`` towards -z (the side the forearm
    does not flex to) at 0.6 of the upper arm."""
    angle = np.radians(np.asarray(flexions, dtype=float))
    frames = angle.size
    centre = np.zeros((frames, 3))
    proximal = np.tile([0.0, 0.30, 0.0], (frames, 1))
    distal = np.stack([np.zeros(frames), -0.25 * np.cos(angle), 0.25 * np.sin(angle)], axis=1)
    lateral = np.tile([0.04, 0.0, 0.0], (frames, 1))
    reference = np.tile([lateral_offset, 0.30 * 0.4, -back_offset], (frames, 1))
    return np.stack([proximal, distal, lateral, reference], axis=1), centre


FLEXIONS = np.arange(14, 121, 15)


@pytest.mark.parametrize("lateral_offset, back_offset", [(0.06, 0.04), (0.04, 0.02),
                                                         (0.04, 0.04), (0.03, 0.02)])
def test_a_marker_towards_the_back_still_gives_the_right_centre(lateral_offset,
                                                                back_offset) -> None:
    markers, centre = sweep(lateral_offset, back_offset, FLEXIONS)
    result = hinge_set(10).evaluate(("P", "D", "L", "R"), markers,
                                    np.ones((FLEXIONS.size, 4), bool), {})
    valid = result.centre_valid["left_elbow"]
    assert valid.any() and not result.warnings
    np.testing.assert_allclose(result.centres["left_elbow"][valid], centre[valid], atol=1e-9)


def test_a_marker_inside_the_line_is_flagged_and_not_trusted() -> None:
    markers, centre = sweep(0.01, 0.0, FLEXIONS)
    result = hinge_set(10).evaluate(("P", "D", "L", "R"), markers,
                                    np.ones((FLEXIONS.size, 4), bool), {})
    assert result.warnings and "left_elbow" in result.warnings[0]
    assert not result.centre_valid["left_elbow"].any()
