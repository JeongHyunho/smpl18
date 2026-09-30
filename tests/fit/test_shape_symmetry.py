"""The shape fit on the three skeleton symmetry modes, and its left/right equality condition.

The body the targets come from is the symmetric stand-in; the model fitted to them is a lopsided
copy of it (:func:`lopsided_model`), whose template and shape directions are not mirror images,
as SMPL's are not.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from smpl18 import synthetic
from smpl18.fit import shape as shape_module
from smpl18.fit.shape import ShapeSettings, fit_shape, rigid_pairs, shape_basis
from smpl18.fit.targets import PositionTargets, Targets, provenance_for
from smpl18.model.demo import demo_model
from smpl18.skeleton.definition import JOINT_NAMES, NUM_JOINTS, PARENTS
from smpl18.skeleton.kinematics import fk_batch, rest_joints
from smpl18.skeleton.symmetry import mirror_pairs

from .conftest import BETAS, lopsided_model

OBSERVED = (1, 2, 4, 5, 7, 8, 10, 11, 9, 12, 15, 16, 17, 18, 19, 20, 21)
#: Every bone of the tree, as (parent, joint).
BONES = [(PARENTS[joint], joint) for joint in range(1, NUM_JOINTS)]
#: The fit of the lopsided model below by the code before the symmetry modes existed (the same
#: settings, ``shape.betas`` 10), as float hex: after refinement, and after the bone-length step
#: alone. Where they were taken they match the ``none`` mode with the condition off to the bit.
GOLDEN_REFINED = (
    "0x1.e67553f237774p-1", "-0x1.f002c9ce60466p-1", "-0x1.5c4baaeb07101p-4",
    "-0x1.500f28938b71ap-1", "-0x1.cb733acc36c91p-3", "0x1.12b147b921112p-1",
    "-0x1.a39d134aba868p-2", "-0x1.b08a083c943fap-6", "-0x1.a7ecc24340361p-6",
    "-0x1.1e1dc33534ef3p-5",
)
GOLDEN_BONE_STEP = (
    "0x1.7d10cfc05e1fep-1", "-0x1.cab66e6e5c9afp-1", "0x1.19c8dd42cf4d0p-4",
    "-0x1.074257592d36cp+0", "-0x1.4d7af14be07c9p-3", "0x1.3d8b328454fc9p+0",
    "-0x1.1b3a11ba4629ap-2", "0x1.1285734acc202p-2", "-0x1.82f652f160872p-2",
    "-0x1.cacc51da0433ap-2",
)


@pytest.fixture(scope="module")
def targets() -> Targets:
    """40 walking frames of the symmetric stand-in at ``BETAS``, 17 joints observed."""
    truth = rest_joints(demo_model(), BETAS)
    local, trans = synthetic.walking_motion(40, 100.0, seed=3)
    positions, _ = fk_batch(truth, local, trans)
    values = positions[:, list(OBSERVED)]
    return Targets(PositionTargets(OBSERVED, values, np.ones(values.shape[:2], bool),
                                   np.ones(len(OBSERVED)), [JOINT_NAMES[j] for j in OBSERVED]),
                   None, 100.0, provenance_for(OBSERVED))


def with_shape(settings: dict, **values) -> dict:
    return {**settings, "shape": {**settings["shape"], **values}}


def side_differences(rest: np.ndarray, pairs=BONES) -> np.ndarray:
    """``l_L - l_R`` for every left/right couple among ``pairs``."""
    return np.array([
        np.linalg.norm(rest[a[1]] - rest[a[0]]) - np.linalg.norm(rest[b[1]] - rest[b[0]])
        for a, b in mirror_pairs(pairs)
    ])


def test_the_lopsided_model_is_lopsided(targets, settings) -> None:
    """Without the modes and without the condition the fit leaves the sides apart, so the tests
    below can tell the modes from doing nothing."""
    fit = fit_shape(lopsided_model("none"), targets, with_shape(settings, lr_equality_weight=0.0))
    assert fit.lr_max_difference_m > 2e-3


def test_the_skeleton_mode_makes_the_equality_condition_vanish(targets, settings) -> None:
    model = lopsided_model("skeleton")
    base, directions = shape_basis(model, 10)
    for betas in np.random.default_rng(4).normal(0.0, 1.5, (10, 10)):
        assert np.abs(side_differences(base + directions @ betas)).max() < 1e-12
    held = fit_shape(model, targets, settings)
    free = fit_shape(model, targets, with_shape(settings, lr_equality_weight=0.0))
    assert held.lr_equality_weight == 100.0 and free.lr_equality_weight == 0.0
    np.testing.assert_array_equal(held.betas, free.betas)
    assert held.model_symmetry == "skeleton" and held.lr_max_difference_m < 1e-9
    assert np.abs(side_differences(rest_joints(model, held.betas))).max() < 1e-9


def test_the_template_mode_needs_and_gets_the_equality_condition(targets, settings) -> None:
    model = lopsided_model("template")
    free = fit_shape(model, targets, with_shape(settings, lr_equality_weight=0.0))
    held = fit_shape(model, targets, settings)
    assert free.lr_max_difference_m > 0.5e-3        # the shape directions alone part the sides
    assert held.model_symmetry == "template" and held.lr_max_difference_m < 0.5e-3
    # Every observed couple of bones, not only the ones the record summarises.
    observed = [bone for bone in BONES if set(bone) <= set(OBSERVED)]
    assert np.abs(side_differences(rest_joints(model, held.betas), observed)).max() < 0.5e-3


def test_the_bone_length_step_alone_holds_the_sides_equal(targets, settings) -> None:
    """Its residual ``w (l_L - l_R)`` closes the gap the unmirrored shape directions open."""
    model = lopsided_model("template")
    bone_only = with_shape(settings, refinements=0)
    loose = fit_shape(model, targets, with_shape(bone_only, lr_equality_weight=0.0))
    tight = fit_shape(model, targets, bone_only)
    assert loose.lr_max_difference_m > 0.5e-3 and tight.lr_max_difference_m < 0.5e-3


def test_smpls_own_skeleton_refuses_the_condition(targets, settings) -> None:
    """No beta removes the template's own asymmetry, so on ``none`` the condition is refused
    rather than left to outweigh the targets."""
    with pytest.raises(ValueError, match="lr_equality_weight must be 0 with model.symmetry none"):
        fit_shape(lopsided_model("none"), targets, settings)
    ShapeSettings.from_settings(settings).check_symmetry("template")
    ShapeSettings.from_settings(with_shape(settings, lr_equality_weight=0.0)).check_symmetry("none")


def test_the_fit_records_its_mode_and_condition(targets, settings) -> None:
    model = lopsided_model("template")
    fit = fit_shape(model, targets, with_shape(settings, lr_equality_weight=7.5))
    assert fit.model_symmetry == "template" and fit.lr_equality_weight == 7.5
    largest = np.abs(side_differences(rest_joints(model, fit.betas), rigid_pairs(OBSERVED))).max()
    assert fit.lr_max_difference_m == pytest.approx(largest, abs=1e-12)


def test_the_condition_weight_is_not_negative(settings) -> None:
    with pytest.raises(ValueError, match="lr_equality_weight"):
        ShapeSettings.from_settings(with_shape(settings, lr_equality_weight=-1.0))


# --- the fit as it was: the package's own formulas from before the modes ---------------------


def _old_shape_basis(model, count):
    count = min(count, model.num_betas)
    base = model.J_regressor @ (model.v_template + np.einsum("vij,j->vi", model.shapedirs,
                                                             np.zeros(model.num_betas)))
    directions = np.einsum("jv,vib->jib", model.J_regressor, model.shapedirs[:, :, :count])
    return base, directions


def _old_bone_step(base, directions, lengths, options):
    count = directions.shape[2]
    if not lengths or count == 0:
        return np.zeros(count)
    pairs = list(lengths)
    first = np.array([a for a, _ in pairs])
    second = np.array([b for _, b in pairs])
    wanted = np.array([lengths[p] for p in pairs])
    scale = 1.0 / np.sqrt(len(pairs))
    prior = np.sqrt(options.prior_weight)

    def residual(beta):
        joints = base + directions @ beta
        model_lengths = np.linalg.norm(joints[second] - joints[first], axis=1)
        return np.concatenate([scale * (model_lengths - wanted), prior * beta])

    return least_squares(residual, np.zeros(count), method=options.optimiser).x


def _old_linear_step(base, directions, local, targets, prior_weight, *_new_arguments):
    frames = local.shape[0]
    count = directions.shape[2]
    zero = np.zeros((frames, 3))
    placed, world = fk_batch(base, local, zero)
    moved = np.empty((frames, NUM_JOINTS, 3, count))
    moved[:, 0] = directions[0]
    for joint in range(1, NUM_JOINTS):
        parent = PARENTS[joint]
        moved[:, joint] = moved[:, parent] + np.einsum(
            "tik,kb->tib", world[:, parent], directions[joint] - directions[parent]
        )
    columns = targets.positions
    weight = (columns.weights[None] * columns.valid) ** 2
    total = weight.sum(axis=1)
    usable = total > 0
    weight, total = weight[usable], total[usable]
    design = moved[usable][:, columns.joints]
    offset = placed[usable][:, columns.joints] - np.where(
        columns.valid[usable][..., None], columns.positions[usable], 0.0
    )
    design = design - np.einsum("tk,tkib->tib", weight, design)[:, None] / total[:, None, None, None]
    offset = offset - np.einsum("tk,tki->ti", weight, offset)[:, None] / total[:, None, None]
    mass = weight.sum()
    lhs = np.einsum("tk,tkib,tkic->bc", weight, design, design) / mass
    rhs = -np.einsum("tk,tkib,tki->b", weight, design, offset) / mass
    beta = np.linalg.solve(lhs + prior_weight * np.eye(count), rhs)
    error = np.einsum("tkib,b->tki", design, beta) + offset
    rms = float(np.sqrt(np.einsum("tk,tki,tki->", weight, error, error) / mass))
    return beta, rms


def test_none_with_the_condition_off_is_the_fit_as_it_was(targets, settings, monkeypatch) -> None:
    """Bit for bit: the fit before the modes existed, run here from its own formulas, and the
    ``none`` mode with ``lr_equality_weight`` 0."""
    model = lopsided_model("none")
    options = with_shape(settings, lr_equality_weight=0.0)
    betas = np.random.default_rng(2).normal(0.0, 1.0, 10)
    np.testing.assert_array_equal(
        rest_joints(model, betas),
        model.J_regressor @ (model.v_template + np.einsum("vij,j->vi", model.shapedirs, betas)),
    )
    for new, old in zip(shape_basis(model, 10), _old_shape_basis(model, 10)):
        np.testing.assert_array_equal(new, old)

    fitted = fit_shape(model, targets, options)
    fitted_bone_step = fit_shape(model, targets, with_shape(options, refinements=0))
    with monkeypatch.context() as patch:
        patch.setattr(shape_module, "shape_basis", _old_shape_basis)
        patch.setattr(shape_module, "_bone_step", _old_bone_step)
        patch.setattr(shape_module, "_linear_step", _old_linear_step)
        before = fit_shape(model, targets, options)
        before_bone_step = fit_shape(model, targets, with_shape(options, refinements=0))
    np.testing.assert_array_equal(fitted.betas, before.betas)
    np.testing.assert_array_equal(fitted_bone_step.betas, before_bone_step.betas)
    assert fitted.bone_rms_m == before.bone_rms_m
    assert fitted.position_rms_m == before.position_rms_m

    # And those formulas are the package's own as it was: its numbers for this fit.
    golden = np.array([float.fromhex(value) for value in GOLDEN_REFINED])
    golden_bone_step = np.array([float.fromhex(value) for value in GOLDEN_BONE_STEP])
    np.testing.assert_allclose(fitted.betas, golden, rtol=0, atol=1e-9)
    np.testing.assert_allclose(fitted_bone_step.betas, golden_bone_step, rtol=0, atol=1e-9)
