"""The subject's shape: betas that make the SMPL rest skeleton match what the targets measured.

Two steps, both on the subject's pooled frames:

1. **Bone lengths.** Every pair of observed joints whose distance no rotation can change -- a
   parent and its child, or two children of one parent -- gives a length, the median over the
   frames where both are seen. Betas are fitted to those lengths by nonlinear least squares.
   This needs no pose and is a sound start, but it sees nothing of the trunk, where no observed
   pair is rigid.
2. **Refinement.** With the betas held, poses are solved on a sample of frames; with those
   rotations held, every observed joint centre is linear in the betas (``J(beta) = J0 + D beta``
   and forward kinematics adds rotated rest offsets), so the betas are re-solved in closed form
   against all the targets at once, each frame's translation eliminated. Repeating this uses the
   trunk and every other chain the targets span.

Both objectives are means of squared distances (m^2) plus ``prior_weight * |beta|^2``, so the one
weight means the same thing in both. Ten betas cannot reach every body; the residuals are
returned so the corpus can say how far the fit got.

The basis follows the model's left/right symmetry mode (:mod:`smpl18.skeleton.symmetry`). Both
steps also hold the left and right sides equal, with ``w = lr_equality_weight``: the bone-length
step adds ``w (l_L - l_R)`` for every left/right couple of observed rigid pairs, and the
refinement adds ``w^2 |A(J0 + D beta)|^2`` over the observed joints whose mirror partner is
observed too, ``A`` being the antisymmetric part -- still quadratic in the betas, so still closed
form. On the ``skeleton`` basis both terms are zero whatever the betas (and are left out, so the
weight cannot change the fit); on ``template`` they hold down what the unmirrored shape directions
would put between the sides; on ``none`` the condition is refused
(:meth:`ShapeSettings.check_symmetry`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from smpl18.model.load import Model
from smpl18.reduce.fit import sample_indices
from smpl18.skeleton.definition import CHILDREN, JOINT_NAMES, NUM_JOINTS, PARENTS
from smpl18.skeleton.kinematics import fk_batch, rest_joints
from smpl18.skeleton.symmetry import (
    MIRROR_PARTNER,
    antisymmetric_part,
    mirror_pairs,
    symmetric_part,
)

from .pose import PoseSettings, solve_pose
from .targets import Targets

__all__ = [
    "SETTINGS_SECTION",
    "ShapeFit",
    "ShapeSettings",
    "fit_shape",
    "measured_lengths",
    "rigid_pairs",
    "shape_basis",
]

SETTINGS_SECTION = "shape"
_SETTING_NAMES = ("betas", "prior_weight", "optimiser", "sample_frames", "refinements",
                  "lr_equality_weight")


@dataclass(frozen=True)
class ShapeSettings:
    #: How many betas are fitted (at most the model's width); the rest stay zero.
    betas: int
    #: Weight of ``|beta|^2`` against the mean squared distance in m^2.
    prior_weight: float
    #: ``scipy.optimize.least_squares`` method for the bone-length step.
    optimiser: str
    #: Frames the refinement solves poses on, spread over the subject's trials.
    sample_frames: int
    #: Pose-then-betas rounds after the bone-length step; zero keeps the bone-length betas.
    refinements: int
    #: ``w`` of the left/right equality condition (per metre of length or position difference);
    #: zero leaves it out.
    lr_equality_weight: float

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> ShapeSettings:
        section = settings.get(SETTINGS_SECTION) if isinstance(settings, Mapping) else None
        if not isinstance(section, Mapping):
            raise ValueError(
                f"settings has no {SETTINGS_SECTION!r} section; the shape fit needs "
                + ", ".join(f"{SETTINGS_SECTION}.{name}" for name in _SETTING_NAMES)
            )
        missing = [name for name in _SETTING_NAMES if name not in section]
        if missing:
            raise ValueError(
                "settings is missing " + ", ".join(f"{SETTINGS_SECTION}.{n}" for n in missing)
            )
        values = cls(
            betas=int(section["betas"]),
            prior_weight=float(section["prior_weight"]),
            optimiser=str(section["optimiser"]),
            sample_frames=int(section["sample_frames"]),
            refinements=int(section["refinements"]),
            lr_equality_weight=float(section["lr_equality_weight"]),
        )
        if values.betas < 0 or values.refinements < 0 or values.sample_frames < 1:
            raise ValueError(
                f"{SETTINGS_SECTION}.betas and .refinements must not be negative and "
                ".sample_frames must be at least 1"
            )
        if values.prior_weight < 0 or values.lr_equality_weight < 0:
            raise ValueError(
                f"{SETTINGS_SECTION}.prior_weight and .lr_equality_weight must not be negative"
            )
        return values

    def check_symmetry(self, symmetry: str) -> None:
        """Refuse the equality condition on SMPL's own skeleton (``none``).

        There the template's antisymmetric part ``A(J0)`` is an offset no beta removes, and the
        refinement's ``w^2 |A(J0 + D beta)|^2`` at a useful weight outweighs the targets: on a
        lopsided test body it left the sides further apart than no condition at all and more
        than tripled the bone error. ``none`` is meant as SMPL's fit as it was, which is
        ``lr_equality_weight`` 0; the condition belongs with a mirrored template.
        """
        if symmetry == "none" and self.lr_equality_weight:
            raise ValueError(
                f"{SETTINGS_SECTION}.lr_equality_weight must be 0 with model.symmetry none: on "
                "SMPL's own template the condition cannot be met and would outweigh the targets "
                "(use model.symmetry template or skeleton to have the sides equal)"
            )


@dataclass(frozen=True)
class ShapeFit:
    """The betas and what they leave unexplained."""

    #: Model width; entries past ``settings.betas`` are zero.
    betas: np.ndarray
    #: Per rigid pair ``"a-b"``: measured length, model length, and their difference, metres.
    bones: dict[str, dict[str, float]]
    #: RMS of the bone-length differences, metres (NaN without rigid pairs).
    bone_rms_m: float
    #: RMS joint-centre distance on the refinement sample with its poses held (NaN without
    #: refinement), metres.
    position_rms_m: float
    #: Frames the refinement used.
    frames_used: int
    refinements: int
    fitted_betas: int
    #: The model's left/right symmetry mode the betas were fitted on.
    model_symmetry: str
    #: ``w`` of the left/right equality condition.
    lr_equality_weight: float
    #: Largest ``|l_L - l_R|`` over the left/right couples of fitted rigid pairs, metres (NaN
    #: when no couple was observed on both sides).
    lr_max_difference_m: float


def shape_basis(model: Model, count: int) -> tuple[np.ndarray, np.ndarray]:
    """``J0 (24, 3)`` and ``D (24, 3, count)`` with ``J(beta) = J0 + D @ beta``, on the model's
    symmetry mode: ``J0`` is mirrored in ``template`` and ``skeleton``, ``D`` in ``skeleton``."""
    count = min(count, model.num_betas)
    base = rest_joints(model, np.zeros(0))
    directions = np.einsum("jv,vib->jib", model.J_regressor, model.shapedirs[:, :, :count])
    if model.symmetry == "skeleton":
        directions = symmetric_part(directions)
    return base, directions


def rigid_pairs(observed) -> list[tuple[int, int]]:
    """Observed joint pairs whose distance is fixed by the rest skeleton alone: a joint and its
    child, and two children of one joint."""
    observed = set(int(j) for j in observed)
    pairs = []
    for joint in range(NUM_JOINTS):
        children = [c for c in CHILDREN[joint] if c in observed]
        if joint in observed:
            pairs.extend((joint, child) for child in children)
        for first, a in enumerate(children):
            pairs.extend((a, b) for b in children[first + 1:])
    return pairs


def _joint_positions(targets: Targets) -> tuple[np.ndarray, np.ndarray]:
    columns = targets.positions
    total = np.zeros((targets.frames, NUM_JOINTS, 3))
    count = np.zeros((targets.frames, NUM_JOINTS))
    for column, joint in enumerate(columns.joints):
        valid = columns.valid[:, column]
        total[valid, joint] += columns.positions[valid, column]
        count[valid, joint] += 1.0
    has = count > 0
    return total / np.where(has, count, 1.0)[..., None], has


def measured_lengths(targets: Targets, pairs) -> dict[tuple[int, int], float]:
    """Median distance per pair over the frames where both ends are observed."""
    points, has = _joint_positions(targets)
    out = {}
    for a, b in pairs:
        both = has[:, a] & has[:, b]
        if both.any():
            out[(a, b)] = float(np.median(np.linalg.norm(points[both, b] - points[both, a],
                                                         axis=1)))
    return out


def _symmetric(base, directions) -> bool:
    """Whether ``J0 + D beta`` is its own mirror image for every beta (the ``skeleton`` basis).

    The equality condition is then zero whatever the betas and is left out: carried along as
    rows of exact zeros it would still nudge the optimiser's path through rounding alone.
    """
    return not (np.any(antisymmetric_part(base)) or np.any(antisymmetric_part(directions)))


def _bone_step(base, directions, lengths, options: ShapeSettings) -> np.ndarray:
    count = directions.shape[2]
    if not lengths or count == 0:
        return np.zeros(count)
    pairs = list(lengths)
    first = np.array([a for a, _ in pairs])
    second = np.array([b for _, b in pairs])
    wanted = np.array([lengths[p] for p in pairs])
    scale = 1.0 / np.sqrt(len(pairs))
    prior = np.sqrt(options.prior_weight)
    row = {pair: index for index, pair in enumerate(pairs)}
    couples = [(row[left], row[right]) for left, right in mirror_pairs(pairs)]
    equality = options.lr_equality_weight if couples else 0.0
    if equality and _symmetric(base, directions):
        equality = 0.0
    left = np.array([a for a, _ in couples], dtype=np.int64)
    right = np.array([b for _, b in couples], dtype=np.int64)

    def residual(beta):
        joints = base + directions @ beta
        model_lengths = np.linalg.norm(joints[second] - joints[first], axis=1)
        parts = [scale * (model_lengths - wanted), prior * beta]
        if equality:
            parts.append(equality * (model_lengths[left] - model_lengths[right]))
        return np.concatenate(parts)

    return least_squares(residual, np.zeros(count), method=options.optimiser).x


def _linear_step(base, directions, local, targets: Targets, prior_weight,
                 lr_equality_weight=0.0) -> tuple[np.ndarray, float]:
    """Betas that best place every observed centre with the rotations ``local`` held.

    Each frame's translation is free and eliminated by centring that frame on its weighted mean.
    The equality condition adds ``w^2 |A(J0 + D beta)|^2`` over the observed joints whose mirror
    partner is observed too; the returned RMS is the positions' alone.
    """
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
    if lr_equality_weight and not _symmetric(base, directions):
        seen = {int(j) for j, any_valid in zip(columns.joints, columns.valid.any(axis=0))
                if any_valid}
        paired = sorted(j for j in seen if MIRROR_PARTNER[j] in seen)
        if paired:
            apart = antisymmetric_part(base)[paired]
            apart_by = antisymmetric_part(directions)[paired]
            square = lr_equality_weight**2
            lhs = lhs + square * np.einsum("jib,jic->bc", apart_by, apart_by)
            rhs = rhs - square * np.einsum("jib,ji->b", apart_by, apart)
    beta = np.linalg.solve(lhs + prior_weight * np.eye(count), rhs)
    error = np.einsum("tkib,b->tki", design, beta) + offset
    rms = float(np.sqrt(np.einsum("tk,tki,tki->", weight, error, error) / mass))
    return beta, rms


def fit_shape(model: Model, targets: Targets, settings: Mapping[str, Any]) -> ShapeFit:
    """Fit the subject's betas to ``targets``, which may pool several trials end to end.

    The refinement's pose solves use the ``pose`` settings, positions only. The skeleton is the
    one the model's own symmetry mode builds (a conversion sets that mode from ``model.symmetry``
    before it gets here).
    """
    options = ShapeSettings.from_settings(settings)
    options.check_symmetry(model.symmetry)
    pose_options = PoseSettings.from_settings(settings)
    base, directions = shape_basis(model, options.betas)
    width = model.num_betas
    pairs = rigid_pairs(targets.positions.joints)
    lengths = measured_lengths(targets, pairs)
    beta = _bone_step(base, directions, lengths, options)

    frames_used = 0
    position_rms = float("nan")
    if options.refinements and directions.shape[2]:
        enough = np.flatnonzero(
            targets.positions.valid.sum(axis=1) >= pose_options.min_position_targets
        )
        if enough.size:
            sample = targets.take(enough[sample_indices(enough.size, options.sample_frames)])
            frames_used = sample.frames
            for _ in range(options.refinements):
                rest = base + directions @ beta
                pose = solve_pose(rest, sample, settings, positions_only=True)
                beta, position_rms = _linear_step(base, directions, pose.local, sample,
                                                  options.prior_weight,
                                                  options.lr_equality_weight)

    joints = base + directions @ beta
    bones = {}
    fitted = {}
    for (a, b), wanted in lengths.items():
        length = float(np.linalg.norm(joints[b] - joints[a]))
        fitted[(a, b)] = length
        bones[f"{JOINT_NAMES[a]}-{JOINT_NAMES[b]}"] = {
            "measured_m": wanted, "model_m": length, "difference_m": length - wanted,
        }
    differences = np.array([entry["difference_m"] for entry in bones.values()])
    sides = [abs(fitted[left] - fitted[right]) for left, right in mirror_pairs(list(lengths))]
    full = np.zeros(width)
    full[: beta.size] = beta
    return ShapeFit(
        betas=full,
        bones=bones,
        bone_rms_m=float(np.sqrt(np.mean(differences**2))) if differences.size else float("nan"),
        position_rms_m=position_rms,
        frames_used=frames_used,
        refinements=options.refinements if frames_used else 0,
        fitted_betas=int(beta.size),
        model_symmetry=model.symmetry,
        lr_equality_weight=options.lr_equality_weight,
        lr_max_difference_m=float(max(sides)) if sides else float("nan"),
    )
