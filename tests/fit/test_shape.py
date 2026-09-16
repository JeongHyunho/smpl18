import numpy as np
import pytest

from smpl18.fit.shape import ShapeSettings, fit_shape, measured_lengths, rigid_pairs, shape_basis
from smpl18.fit.targets import PositionTargets, Targets, provenance_for
from smpl18.skeleton.definition import JOINT_NAMES, SEGMENTS
from smpl18.skeleton.kinematics import rest_joints, segment_lengths

OBSERVED = (1, 2, 4, 5, 7, 8, 10, 11, 9, 12, 15, 16, 17, 18, 19, 20, 21)


def targets_of(positions, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    values = positions[:, list(OBSERVED)] + rng.normal(0, noise, (positions.shape[0], len(OBSERVED), 3))
    return Targets(PositionTargets(OBSERVED, values, np.ones(values.shape[:2], bool),
                                   np.ones(len(OBSERVED)), [JOINT_NAMES[j] for j in OBSERVED]),
                   None, 100.0, provenance_for(OBSERVED))


def test_rigid_pairs_are_bones_and_siblings_only() -> None:
    pairs = set(rigid_pairs([1, 2, 4, 9, 12, 16, 17]))
    assert (1, 4) in pairs                       # hip to knee: a bone
    assert (1, 2) in pairs                       # two children of the pelvis
    assert (9, 12) in pairs                      # spine3 to neck
    assert (16, 17) not in pairs                 # the shoulders hang off different collars
    assert (1, 9) not in pairs                   # the spine bends between them


def test_measured_lengths_are_medians_over_seen_frames(rest, walk) -> None:
    *_, positions, _ = walk
    targets = targets_of(positions)
    lengths = measured_lengths(targets, [(1, 4)])
    assert lengths[(1, 4)] == pytest.approx(np.linalg.norm(rest[4] - rest[1]), abs=1e-9)


def test_the_basis_is_linear_in_the_betas(model) -> None:
    base, directions = shape_basis(model, 10)
    betas = np.linspace(-1, 1, 10)
    np.testing.assert_allclose(base + directions @ betas, rest_joints(model, betas), atol=1e-12)


def test_refinement_brings_the_skeleton_back(model, rest, walk, settings) -> None:
    *_, positions, _ = walk
    settings["shape"]["prior_weight"] = 1e-9
    fit = fit_shape(model, targets_of(positions), settings)
    fitted = rest_joints(model, fit.betas)
    wanted = segment_lengths(rest)
    found = segment_lengths(fitted)
    limbs = [i for i, (name, *_) in enumerate(SEGMENTS) if name not in ("trunk",)]
    assert np.abs(found[limbs] - wanted[limbs]).max() < 3e-3
    assert fit.bone_rms_m < 2e-3
    assert fit.position_rms_m < 3e-3
    assert fit.frames_used == 40 and fit.refinements == 2 and fit.fitted_betas == 10


def test_the_bone_length_step_alone_fits_the_lengths(model, rest, walk, settings) -> None:
    *_, positions, _ = walk
    settings["shape"].update(prior_weight=0.0, refinements=0)
    fit = fit_shape(model, targets_of(positions), settings)
    assert fit.bone_rms_m < 1e-5
    assert np.isnan(fit.position_rms_m) and fit.frames_used == 0
    assert set(fit.bones) >= {"left_hip-left_knee", "left_elbow-left_wrist", "left_hip-right_hip"}


def test_noise_leaves_the_shape_close(model, rest, walk, settings) -> None:
    *_, positions, _ = walk
    fit = fit_shape(model, targets_of(positions, noise=0.005), settings)
    found = segment_lengths(rest_joints(model, fit.betas))
    assert np.abs(found - segment_lengths(rest)).max() < 0.02


def test_the_prior_holds_the_betas_near_zero(model, walk, settings) -> None:
    *_, positions, _ = walk
    settings["shape"]["prior_weight"] = 1.0
    fit = fit_shape(model, targets_of(positions), settings)
    assert np.abs(fit.betas).max() < 0.05


def test_fewer_betas_leave_the_rest_at_zero(model, walk, settings) -> None:
    *_, positions, _ = walk
    settings["shape"].update(betas=3, refinements=1)
    fit = fit_shape(model, targets_of(positions), settings)
    assert fit.betas.shape == (10,) and np.all(fit.betas[3:] == 0) and fit.fitted_betas == 3


def test_every_setting_is_required(settings) -> None:
    for key in settings["shape"]:
        partial = {"shape": {k: v for k, v in settings["shape"].items() if k != key}}
        with pytest.raises(ValueError, match=f"shape.{key}"):
            ShapeSettings.from_settings(partial)
    with pytest.raises(ValueError, match="negative"):
        ShapeSettings.from_settings({"shape": {**settings["shape"], "prior_weight": -1}})
