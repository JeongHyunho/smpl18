import numpy as np
import pytest

import smpl18.fit.pose as pose_module
from smpl18.fit.pose import PoseSettings, calibrate_orientations, initial_pose, solve_pose
from smpl18.fit.targets import OrientationTargets, PositionTargets, Targets, provenance_for
from smpl18.skeleton.definition import BODY_JOINTS
from smpl18.skeleton.kinematics import fk_batch
from smpl18.skeleton.rotations import axis_angle_to_matrix, matrix_to_axis_angle

#: Every body joint whose position SMPL itself defines, except the pelvis (sources rarely have it).
OBSERVED = (1, 2, 4, 5, 7, 8, 10, 11, 9, 12, 15, 16, 17, 18, 19, 20, 21)
#: Segments a rich source orients.
ORIENTED = (0, 9, 15, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21)
#: Oriented segments with no observed joint below them.
LEAVES = (15, 20, 21)


def full_targets(positions, world, *, offsets=None, orient=True):
    frames = positions.shape[0]
    columns = PositionTargets(OBSERVED, positions[:, OBSERVED],
                              np.ones((frames, len(OBSERVED)), bool),
                              np.ones(len(OBSERVED)), [str(j) for j in OBSERVED])
    orientations = None
    if orient:
        rotations = world[:, ORIENTED]
        if offsets is not None:
            rotations = rotations @ np.swapaxes(offsets, -1, -2)[None]
        orientations = OrientationTargets(ORIENTED, rotations,
                                          np.ones((frames, len(ORIENTED)), bool),
                                          np.ones(len(ORIENTED)), [f"o{j}" for j in ORIENTED])
    return Targets(columns, orientations, 100.0,
                   provenance_for(OBSERVED, ORIENTED if orient else ()))


def angle_between(a, b):
    return np.linalg.norm(matrix_to_axis_angle(np.swapaxes(a, -1, -2) @ b), axis=-1)


def test_positions_and_orientations_give_the_motion_back(rest, walk, settings) -> None:
    local, trans, positions, world = walk
    offsets = axis_angle_to_matrix(np.random.default_rng(1).normal(0, 1, (len(ORIENTED), 3)))
    fit = solve_pose(rest, full_targets(positions, world, offsets=offsets), settings)
    assert fit.passes == ("positions", "orientations")
    assert fit.frame_valid.all()
    solved, solved_world = fk_batch(rest, fit.local, fit.trans)
    observed = list(OBSERVED)
    assert np.abs(solved[:, observed] - positions[:, observed]).max() < 3e-3
    # The joints no target observes (pelvis, spine1, spine2, collars) are placed by the chain
    # and the prior; they still land close.
    body = list(BODY_JOINTS)
    assert np.abs(solved[:, body] - positions[:, body]).max() < 1e-2
    # Segments whose twist something below them shows come back to within a few degrees.
    anchored = [j for j in ORIENTED if j not in LEAVES]
    assert angle_between(solved_world[:, anchored], world[:, anchored]).max() < np.radians(3)
    # A leaf's twist (head, hands) is seen only through its own frame, whose constant is
    # calibrated, so it comes back up to a constant: the error must not move.
    error = np.swapaxes(world[:, list(LEAVES)], -1, -2) @ solved_world[:, list(LEAVES)]
    assert angle_between(error, error[:1]).max() < np.radians(1)
    assert fit.position_error.rms < 1e-3
    assert fit.orientation_error.rms < np.radians(1)


def test_the_calibration_finds_the_constant_between_frames(rest, walk) -> None:
    local, trans, positions, world = walk
    offsets = axis_angle_to_matrix(np.random.default_rng(2).normal(0, 1, (len(ORIENTED), 3)))
    targets = full_targets(positions, world, offsets=offsets)
    found = calibrate_orientations(rest, local, trans, targets.orientations)
    np.testing.assert_allclose(found, offsets, atol=1e-9)


def test_positions_alone_place_every_observed_joint(rest, walk, settings) -> None:
    local, trans, positions, world = walk
    fit = solve_pose(rest, full_targets(positions, world, orient=False), settings)
    assert fit.passes == ("positions",) and fit.orientation_error is None
    solved, _ = fk_batch(rest, fit.local, fit.trans)
    assert np.abs(solved[:, list(OBSERVED)] - positions[:, list(OBSERVED)]).max() < 3e-3


def test_the_initial_guess_is_already_close(rest, walk) -> None:
    _local, _trans, positions, world = walk
    guess_local, guess_trans = initial_pose(rest, full_targets(positions, world, orient=False))
    guessed, _ = fk_batch(rest, guess_local, guess_trans)
    hips = [1, 2]
    assert np.abs(guessed[:, hips] - positions[:, hips]).max() < 0.05


def test_the_analytic_gradient_is_the_true_one(rest, walk, settings) -> None:
    local, trans, positions, world = walk
    frames = slice(0, 3)
    offsets = axis_angle_to_matrix(np.random.default_rng(3).normal(0, 1, (len(ORIENTED), 3)))
    targets = full_targets(positions[frames], world[frames])
    options = PoseSettings.from_settings({"pose": {**settings["pose"], "smoothing_weight": 0.3}})
    rng = np.random.default_rng(4)
    start = local[frames] @ axis_angle_to_matrix(rng.normal(0, 0.2, (3, 24, 3)))
    references = (local[frames][:, 0], matrix_to_axis_angle(local[frames]), trans[frames] + 0.01)
    problem = pose_module._problem(rest, targets, options, offsets, references)
    state = (start[:, 0].copy(), matrix_to_axis_angle(start), trans[frames] + 0.02)
    _, _, gradient = pose_module._evaluate(problem, *state, normal_equations=True)
    numeric = np.zeros_like(gradient)
    for column in range(problem.size):
        step = np.zeros_like(gradient)
        step[:, column] = 1e-6
        up, _, _ = pose_module._evaluate(problem, *pose_module._apply(problem, *state, step),
                                         normal_equations=False)
        down, _, _ = pose_module._evaluate(problem, *pose_module._apply(problem, *state, -step),
                                           normal_equations=False)
        numeric[:, column] = (up - down) / 2e-6
    np.testing.assert_allclose(numeric, 2.0 * gradient, atol=1e-7)


def test_a_frame_with_too_few_targets_holds_its_neighbour(rest, walk, settings) -> None:
    local, trans, positions, world = walk
    targets = full_targets(positions, world, orient=False)
    valid = targets.positions.valid.copy()
    valid[10, 3:] = False                              # three targets left on frame 10
    sparse = Targets(PositionTargets(OBSERVED, targets.positions.positions, valid,
                                     targets.positions.weights, targets.positions.labels),
                     None, 100.0, targets.provenance)
    fit = solve_pose(rest, sparse, settings)
    assert not fit.frame_valid[10] and fit.frame_valid.sum() == 39
    np.testing.assert_array_equal(fit.local[10], fit.local[9])
    np.testing.assert_array_equal(fit.trans[10], fit.trans[9])


def test_no_solvable_frame_is_an_error(rest, walk, settings) -> None:
    _local, _trans, positions, world = walk
    targets = full_targets(positions, world, orient=False)
    few = Targets(PositionTargets(OBSERVED[:2], positions[:, list(OBSERVED[:2])],
                                  np.ones((40, 2), bool), [1, 1], ["a", "b"]),
                  None, 100.0, provenance_for(OBSERVED[:2]))
    with pytest.raises(ValueError, match="valid position targets"):
        solve_pose(rest, few, settings)
    assert targets.frames == 40


def test_joints_nothing_observes_stay_at_rest(rest, walk, settings) -> None:
    local, _trans, positions, _world = walk
    legs = (1, 2, 4, 5, 7, 8)
    targets = Targets(PositionTargets(legs, positions[:, list(legs)], np.ones((40, 6), bool),
                                      np.ones(6), [str(j) for j in legs]),
                      None, 100.0, provenance_for(legs))
    fit = solve_pose(rest, targets, settings)
    for joint in (3, 6, 9, 12, 15, 16, 18, 20):          # spine, neck, head and an arm
        np.testing.assert_array_equal(fit.local[:, joint], np.tile(np.eye(3), (40, 1, 1)))


def test_the_smoothing_pass_runs_when_weighted(rest, walk, settings) -> None:
    local, _trans, positions, world = walk
    noisy = positions + np.random.default_rng(5).normal(0, 0.01, positions.shape)
    settings["pose"]["smoothing_weight"] = 0.2
    fit = solve_pose(rest, full_targets(noisy, world, orient=False), settings)
    assert fit.passes == ("positions", "smoothing")
    settings["pose"]["smoothing_weight"] = 0.0
    rough = solve_pose(rest, full_targets(noisy, world, orient=False), settings)
    body = list(BODY_JOINTS)
    jitter = [np.abs(np.diff(matrix_to_axis_angle(f.local[:, body]), 2, axis=0)).sum(axis=2).mean()
              for f in (fit, rough)]
    assert jitter[0] < 0.8 * jitter[1]


def test_every_setting_is_required_and_checked(settings) -> None:
    with pytest.raises(ValueError, match="no 'pose' section"):
        PoseSettings.from_settings({})
    for key in settings["pose"]:
        partial = {"pose": {k: v for k, v in settings["pose"].items() if k != key}}
        with pytest.raises(ValueError, match=f"pose.{key}"):
            PoseSettings.from_settings(partial)
    with pytest.raises(ValueError, match="at least 3"):
        PoseSettings.from_settings({"pose": {**settings["pose"], "min_position_targets": 2}})
    with pytest.raises(ValueError, match="positive"):
        PoseSettings.from_settings({"pose": {**settings["pose"], "tolerance": 0}})
