"""Per-frame SMPL-24 pose from targets: local rotations and a translation that reach them.

The unknowns of one frame are the model-origin translation, the root's world rotation and the
local rotation of every joint the targets can reach (``Targets.free_joints``); a joint nothing
observes is held at identity. The residual has four blocks, each weighted from the settings:

* **positions**: FK joint centre minus target, per valid target column;
* **orientations**: ``log(Q^T G_j)`` between the SMPL segment rotation ``G_j`` and the target
  ``Q = S A``, the source segment rotation ``S`` times a per-target constant ``A``;
* **pose prior**: the rotation vector of every free non-root joint, which keeps unobserved twist
  at rest and shares a turn among joints the targets cannot tell apart (a spine whose only
  observation is its top) instead of piling it onto one;
* **smoothing** (optional pass): the difference from the average of the two neighbouring frames of
  the previous pass.

The source frame of an orientation target is not SMPL's, so ``A`` is unknown. It is calibrated
from the data: a first pass solves positions only, and ``A`` is the mean rotation of ``S^T G``
over the frames given -- one trial here, or all of a subject's trials when the caller pools them
(``smpl18.convert.fit_subject`` does). A second pass then uses the orientations. The calibration
is therefore relative to the typical posture, which is what makes twist observable where
positions alone leave it free (a straight knee, a forearm, a head); the mean twist of a segment
seen only through its own frame is the one the positions-only pass settled on.

Frames are independent within a pass, so they are solved together: Levenberg-Marquardt with an
analytic Jacobian, the step damping kept per frame. The root's rotation is updated on the group,
from the left in world axes, which keeps its update exact however far the body turns; every other
joint is updated additively on its rotation vector, which keeps the pose prior and the smoothing
linear, so their Gauss-Newton model is exact. Initial rotations come from aligning, joint by joint
down the tree, the rest offsets to the nearest observed descendants (Kabsch where there are two
or more, the smallest rotation where there is one).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from smpl18.skeleton.definition import (
    CHILDREN,
    FROZEN_JOINTS,
    HINGE_AXES,
    NUM_JOINTS,
    PARENTS,
    ROOT,
)
from smpl18.skeleton.kinematics import fk_batch, local_rotations
from smpl18.skeleton.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    mean_rotation,
    proper_rotation_from_covariance,
)

from .targets import OrientationTargets, Targets

__all__ = [
    "SETTINGS_SECTION",
    "PoseFit",
    "PoseSettings",
    "TargetError",
    "calibrate_orientations",
    "hold_nearest",
    "initial_pose",
    "solve_pose",
]

SETTINGS_SECTION = "pose"
_SETTING_NAMES = (
    "iterations",
    "tolerance",
    "damping",
    "position_weight",
    "orientation_weight",
    "prior_weight",
    "frozen_prior_weight",
    "hinge_prior_weight",
    "smoothing_weight",
    "chunk_frames",
    "min_position_targets",
)

#: Below this a direction has no length to align; a numerical guard, not a tuning value.
_TINY = 1e-12


@dataclass(frozen=True)
class PoseSettings:
    """The numbers the solve needs, read from the ``pose`` section of a settings mapping."""

    #: Most Levenberg-Marquardt iterations per pass.
    iterations: int
    #: A frame stops when its largest update (radians or metres) falls below this.
    tolerance: float
    #: Initial damping; shrinks after an accepted step, grows after a rejected one.
    damping: float
    #: Residual weight of a position target (per metre).
    position_weight: float
    #: Residual weight of an orientation target (metres of position error per radian).
    orientation_weight: float
    #: Weight of the pose prior on free non-root joints (metres per radian).
    prior_weight: float
    #: The same for the four joints the corpus freezes (``FROZEN_JOINTS``). Heavier than
    #: ``prior_weight``, it moves a turn the targets cannot place into the joints that keep it
    #: after the reduction, which lowers what the freeze costs.
    frozen_prior_weight: float
    #: The prior on the knees' and elbows' rotation off their flexion axis (``HINGE_AXES``).
    #: Heavier than ``prior_weight``: with ball joints where the body has hinges, positions
    #: cannot tell a twisted thigh or upper arm from a knee or elbow turned sideways, and this
    #: settles it the way the body does.
    hinge_prior_weight: float
    #: Weight of the smoothing pass (per radian and per metre); zero skips the pass.
    smoothing_weight: float
    #: Frames solved per batch; bounds memory, not the result.
    chunk_frames: int
    #: A frame with fewer valid position targets is not solved and is marked invalid.
    min_position_targets: int

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> PoseSettings:
        section = settings.get(SETTINGS_SECTION) if isinstance(settings, Mapping) else None
        if not isinstance(section, Mapping):
            raise ValueError(
                f"settings has no {SETTINGS_SECTION!r} section; the pose solve needs "
                + ", ".join(f"{SETTINGS_SECTION}.{name}" for name in _SETTING_NAMES)
            )
        missing = [name for name in _SETTING_NAMES if name not in section]
        if missing:
            raise ValueError(
                "settings is missing " + ", ".join(f"{SETTINGS_SECTION}.{n}" for n in missing)
            )
        values = cls(
            iterations=int(section["iterations"]),
            tolerance=float(section["tolerance"]),
            damping=float(section["damping"]),
            position_weight=float(section["position_weight"]),
            orientation_weight=float(section["orientation_weight"]),
            prior_weight=float(section["prior_weight"]),
            frozen_prior_weight=float(section["frozen_prior_weight"]),
            hinge_prior_weight=float(section["hinge_prior_weight"]),
            smoothing_weight=float(section["smoothing_weight"]),
            chunk_frames=int(section["chunk_frames"]),
            min_position_targets=int(section["min_position_targets"]),
        )
        for name in ("iterations", "chunk_frames"):
            if getattr(values, name) < 1:
                raise ValueError(f"{SETTINGS_SECTION}.{name} must be at least 1")
        if values.min_position_targets < 3:
            raise ValueError(
                f"{SETTINGS_SECTION}.min_position_targets must be at least 3: fewer points "
                "cannot fix a rotation and a translation"
            )
        for name in ("tolerance", "damping", "position_weight"):
            if getattr(values, name) <= 0:
                raise ValueError(f"{SETTINGS_SECTION}.{name} must be positive")
        for name in ("orientation_weight", "prior_weight", "frozen_prior_weight",
                     "hinge_prior_weight", "smoothing_weight"):
            if getattr(values, name) < 0:
                raise ValueError(f"{SETTINGS_SECTION}.{name} must not be negative")
        return values


@dataclass(frozen=True)
class TargetError:
    """How far the solved pose is from its targets over the frames that were solved."""

    rms: float
    max: float
    #: The same RMS per target column, keyed by the column's label.
    per_target: dict[str, float]


@dataclass(frozen=True)
class PoseFit:
    """A solved trial: ``(T, 24, 3, 3)`` local rotations and ``(T, 3)`` model-origin translation."""

    local: np.ndarray
    trans: np.ndarray
    #: False where the frame had too few targets; such a frame holds its nearest solved pose.
    frame_valid: np.ndarray
    #: ``(M, 3, 3)`` constants ``A`` of the orientation targets, or None without them.
    calibration: np.ndarray | None
    #: In metres.
    position_error: TargetError
    #: In radians; None without orientation targets.
    orientation_error: TargetError | None
    #: Passes run: positions, then orientations, then smoothing, as applicable.
    passes: tuple[str, ...]


# --- small batched algebra ----------------------------------------------------------------------


def _skew(vectors: np.ndarray) -> np.ndarray:
    out = np.zeros(vectors.shape + (3,), dtype=np.float64)
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    out[..., 0, 1], out[..., 0, 2] = -z, y
    out[..., 1, 0], out[..., 1, 2] = z, -x
    out[..., 2, 0], out[..., 2, 1] = -y, x
    return out


def _right_jacobian_inverse(vectors: np.ndarray) -> np.ndarray:
    """``J_r^{-1}(r)``: how ``log(R exp(d))`` moves with a small ``d``, for rotation vectors ``r``."""
    angle = np.linalg.norm(vectors, axis=-1)
    skew = _skew(vectors)
    small = angle < 1e-6
    safe = np.where(small, 1.0, angle)
    # 1/a^2 - (1 + cos a) / (2 a sin a), whose limit at zero is 1/12. Near a half turn the
    # sine vanishes; a rotation vector never exceeds pi, and one within 1e-6 of it is clipped.
    sine = np.sin(np.minimum(safe, np.pi - 1e-6))
    coefficient = np.where(
        small, 1.0 / 12.0, 1.0 / safe**2 - (1.0 + np.cos(safe)) / (2.0 * safe * sine)
    )
    return (np.eye(3) + 0.5 * skew + coefficient[..., None, None] * (skew @ skew))


def _smallest_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Batched smallest rotation carrying direction ``source`` onto ``target``; ``(..., 3, 3)``."""
    a_norm = np.linalg.norm(source, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    usable = (a_norm[..., 0] > _TINY) & (b_norm[..., 0] > _TINY)
    a = source / np.where(a_norm > _TINY, a_norm, 1.0)
    b = target / np.where(b_norm > _TINY, b_norm, 1.0)
    axis = np.cross(a, b)
    sine = np.linalg.norm(axis, axis=-1)
    cosine = np.sum(a * b, axis=-1)
    skew = _skew(axis)
    factor = np.where(sine > _TINY, (1.0 - cosine) / np.where(sine > _TINY, sine, 1.0) ** 2, 0.0)
    out = np.eye(3) + skew + factor[..., None, None] * (skew @ skew)
    # Opposite directions: a half turn about any axis perpendicular to them.
    opposite = (sine <= _TINY) & (cosine < 0.0) & usable
    if opposite.any():
        helper = np.where(np.abs(a[..., :1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        perpendicular = np.cross(a, helper)
        perpendicular /= np.linalg.norm(perpendicular, axis=-1, keepdims=True)
        half_turn = 2.0 * perpendicular[..., :, None] * perpendicular[..., None, :] - np.eye(3)
        out = np.where(opposite[..., None, None], half_turn, out)
    return np.where(usable[..., None, None], out, np.eye(3))


def _subtree_mask() -> np.ndarray:
    """``mask[k, j]``: joint ``j`` is ``k`` or one of its ancestors, so turning ``j`` moves ``k``."""
    mask = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=bool)
    for joint in range(NUM_JOINTS):
        node = joint
        while node >= 0:
            mask[joint, node] = True
            node = PARENTS[node]
    return mask


_ANCESTOR_OR_SELF = _subtree_mask()


# --- initial guess ------------------------------------------------------------------------------


def _per_joint(targets: Targets) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 24, 3)`` mean of the valid position columns per joint and ``(T, 24)`` whether any."""
    frames = targets.frames
    total = np.zeros((frames, NUM_JOINTS, 3))
    count = np.zeros((frames, NUM_JOINTS))
    columns = targets.positions
    for column, joint in enumerate(columns.joints):
        valid = columns.valid[:, column]
        total[valid, joint] += columns.positions[valid, column]
        count[valid, joint] += 1.0
    has = count > 0
    return total / np.where(has, count, 1.0)[..., None], has


def _frontier(joint: int, observed: set[int]) -> list[int]:
    """The nearest observed joints below ``joint``, one or more per child branch."""
    out: list[int] = []
    for child in CHILDREN[joint]:
        if child in observed:
            out.append(child)
        else:
            out.extend(_frontier(child, observed))
    return out


def _kabsch_about_origin(rest_vectors, target_vectors, weights) -> np.ndarray:
    covariance = np.einsum("tn,tni,tnj->tij", weights, rest_vectors, target_vectors)
    return proper_rotation_from_covariance(covariance)


def initial_pose(rest: np.ndarray, targets: Targets) -> tuple[np.ndarray, np.ndarray]:
    """A starting pose from the position targets alone: ``(T, 24, 3, 3)`` local, ``(T, 3)`` trans."""
    rest = np.asarray(rest, dtype=np.float64)
    frames = targets.frames
    point, has = _per_joint(targets)
    observed = set(targets.positions.joints)
    free = set(targets.free_joints)

    root_points = _frontier(ROOT, observed) + ([ROOT] if ROOT in observed else [])
    if len(root_points) < 3:
        root_points = sorted(observed)
    weights = has[:, root_points].astype(np.float64)
    total = weights.sum(axis=1)
    safe = np.where(total > 0, total, 1.0)
    rest_centre = weights @ rest[root_points] / safe[:, None]
    target_points = np.where(has[:, root_points, None], point[:, root_points], 0.0)
    target_centre = np.einsum("tn,tni->ti", weights, target_points) / safe[:, None]
    rotation = _kabsch_about_origin(
        rest[root_points][None] - rest_centre[:, None], target_points - target_centre[:, None],
        weights,
    )
    rotation = np.where((total >= 3)[:, None, None], rotation, np.eye(3))
    world = np.empty((frames, NUM_JOINTS, 3, 3))
    positions = np.empty((frames, NUM_JOINTS, 3))
    world[:, ROOT] = rotation
    trans = target_centre - np.einsum("tij,tj->ti", rotation, rest_centre - rest[ROOT]) - rest[ROOT]
    trans = np.where((total > 0)[:, None], trans, 0.0)
    positions[:, ROOT] = rest[ROOT] + trans

    for joint in range(1, NUM_JOINTS):
        parent = PARENTS[joint]
        positions[:, joint] = positions[:, parent] + np.einsum(
            "tij,j->ti", world[:, parent], rest[joint] - rest[parent]
        )
        branch = _frontier(joint, observed) if joint in free else []
        if not branch:
            world[:, joint] = world[:, parent]
            continue
        rest_vectors = np.einsum("tij,nj->tni", world[:, parent], rest[branch] - rest[joint])
        target_vectors = point[:, branch] - positions[:, joint][:, None]
        valid = has[:, branch]
        count = valid.sum(axis=1)
        turn = np.tile(np.eye(3), (frames, 1, 1))
        single = count == 1
        if single.any():
            which = np.argmax(valid[single], axis=1)
            rows = np.arange(which.size)
            turn[single] = _smallest_rotation(
                rest_vectors[single][rows, which], target_vectors[single][rows, which]
            )
        several = count >= 2
        if several.any():
            turn[several] = _kabsch_about_origin(
                rest_vectors[several],
                np.where(valid[several][..., None], target_vectors[several], 0.0),
                valid[several].astype(np.float64),
            )
        world[:, joint] = turn @ world[:, parent]
    local = local_rotations(world)
    for joint in range(NUM_JOINTS):
        if joint not in free:
            local[:, joint] = np.eye(3)
    return local, trans


# --- the least-squares problem ------------------------------------------------------------------


@dataclass(frozen=True)
class _Problem:
    rest: np.ndarray
    free: np.ndarray
    pos_joints: np.ndarray
    pos_target: np.ndarray
    pos_weight: np.ndarray
    #: ``(target column, free-joint slot)`` pairs where turning the joint moves the target.
    pos_pairs: tuple[np.ndarray, np.ndarray]
    ori_joints: np.ndarray
    ori_target: np.ndarray
    ori_weight: np.ndarray
    ori_pairs: tuple[np.ndarray, np.ndarray]
    #: Prior weight per free non-root joint and rotation-vector component, ``(F - 1, 3)``.
    prior_weights: np.ndarray
    smoothing_weight: float
    #: Smoothing references: the root's world rotation ``(T, 3, 3)``, every joint's rotation
    #: vector ``(T, 24, 3)`` and the translation ``(T, 3)``.
    ref_root: np.ndarray | None
    ref_vectors: np.ndarray | None
    ref_trans: np.ndarray | None

    @property
    def size(self) -> int:
        return 3 + 3 * self.free.size

    def take(self, frames: np.ndarray) -> _Problem:
        def pick(array):
            return None if array is None else array[frames]

        return _Problem(
            self.rest, self.free, self.pos_joints, self.pos_target[frames],
            self.pos_weight[frames], self.pos_pairs, self.ori_joints,
            self.ori_target[frames], self.ori_weight[frames], self.ori_pairs,
            self.prior_weights, self.smoothing_weight,
            pick(self.ref_root), pick(self.ref_vectors), pick(self.ref_trans),
        )


def _prior_weights(joints: np.ndarray, options: PoseSettings) -> np.ndarray:
    """``(J, 3)`` prior weights per joint and rotation-vector component."""
    weights = np.full((joints.size, 3), options.prior_weight)
    for row, joint in enumerate(joints):
        if joint in FROZEN_JOINTS:
            weights[row] = options.frozen_prior_weight
        elif joint in HINGE_AXES:
            weights[row] = options.hinge_prior_weight
            weights[row, HINGE_AXES[joint]] = options.prior_weight
    return weights


def _problem(rest, targets: Targets, options: PoseSettings, calibration, references) -> _Problem:
    free = np.array(targets.free_joints, dtype=np.int64)
    columns = targets.positions
    pos_joints = np.array(columns.joints, dtype=np.int64)
    pos_weight = options.position_weight * columns.weights[None] * columns.valid
    pos_target = np.where(columns.valid[..., None], columns.positions, 0.0)
    orientations: OrientationTargets | None = targets.orientations
    if orientations is None or calibration is None or options.orientation_weight == 0.0:
        ori_joints = np.zeros(0, dtype=np.int64)
        ori_target = np.zeros((targets.frames, 0, 3, 3))
        ori_weight = np.zeros((targets.frames, 0))
    else:
        ori_joints = np.array(orientations.joints, dtype=np.int64)
        ori_target = orientations.rotations @ calibration[None]
        ori_weight = (options.orientation_weight * orientations.weights[None]
                      * orientations.valid)
    ref_root, ref_vectors, ref_trans = references if references is not None else (None,) * 3
    pos_pairs = np.nonzero(_ANCESTOR_OR_SELF[pos_joints][:, free])
    ori_pairs = np.nonzero(_ANCESTOR_OR_SELF[ori_joints][:, free])
    return _Problem(
        rest=np.asarray(rest, dtype=np.float64),
        free=free,
        pos_joints=pos_joints,
        pos_target=pos_target,
        pos_weight=pos_weight,
        pos_pairs=pos_pairs,
        ori_joints=ori_joints,
        ori_target=ori_target,
        ori_weight=ori_weight,
        ori_pairs=ori_pairs,
        prior_weights=_prior_weights(free[1:], options),
        smoothing_weight=options.smoothing_weight if references is not None else 0.0,
        ref_root=ref_root,
        ref_vectors=ref_vectors,
        ref_trans=ref_trans,
    )


def _left_jacobian(vectors: np.ndarray) -> np.ndarray:
    """``J_l(v)``: ``exp(v + d) = exp(J_l(v) d) exp(v)`` to first order in ``d``."""
    angle = np.linalg.norm(vectors, axis=-1)
    skew = _skew(vectors)
    small = angle < 1e-6
    safe = np.where(small, 1.0, angle)
    first = np.where(small, 0.5, (1.0 - np.cos(safe)) / safe**2)
    second = np.where(small, 1.0 / 6.0, (safe - np.sin(safe)) / safe**3)
    return np.eye(3) + first[..., None, None] * skew + second[..., None, None] * (skew @ skew)


def _rows(frames: int, count: int, size: int, pairs, blocks: np.ndarray) -> np.ndarray:
    """Jacobian rows ``(T, count, 3, size)`` from ``(T, P, 3, 3)`` blocks, one per pair
    ``(row, free-joint slot)``; every other entry is zero."""
    rows, slots = pairs
    out = np.zeros((frames, count, 3, size))
    columns = 3 + 3 * slots[:, None] + np.arange(3)
    # Advanced indices split by a slice put their broadcast shape (P, 3) first.
    out[:, rows[:, None], :, columns] = blocks.transpose(1, 3, 0, 2)
    return out


def _add_block(normal, gradient, slot: int, jacobian: np.ndarray, residual: np.ndarray) -> None:
    """Add a residual that depends on one parameter triple only: ``(T, 3, 3)``, ``(T, 3)``.
    Slot -1 is the translation."""
    start = 3 + 3 * slot if slot >= 0 else 0
    transposed = np.swapaxes(jacobian, -1, -2)
    normal[:, start:start + 3, start:start + 3] += transposed @ jacobian
    gradient[:, start:start + 3] += (transposed @ residual[..., None])[..., 0]


def _local(root: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    local = axis_angle_to_matrix(vectors)
    local[:, ROOT] = root
    return local


def _evaluate(problem: _Problem, root, vectors, trans, *, normal_equations: bool):
    """Per-frame cost, and when asked the Gauss-Newton normal equations ``J^T J``, ``J^T r``.

    The state is the root's world rotation, a rotation vector per joint and the translation. The
    root is updated on the group (``exp(d) R``); every other joint additively on its rotation
    vector, which keeps the pose prior and the smoothing of those joints linear, so the
    Gauss-Newton model of them is exact and the solve converges in a few steps even for joints
    turned far from rest. The prior and smoothing residuals touch one parameter triple each, so
    their share of the normal equations is added block by block.
    """
    frames = root.shape[0]
    size = problem.size
    local = _local(root, vectors)
    positions, world = fk_batch(problem.rest, local, trans)
    free = problem.free
    cost = np.zeros(frames)
    normal = np.zeros((frames, size, size)) if normal_equations else None
    gradient = np.zeros((frames, size)) if normal_equations else None

    axes = None
    if normal_equations:
        # World axis of a unit step in each free joint's parameters.
        parents = np.array([PARENTS[joint] for joint in free[1:]])
        axes = np.empty((frames, free.size, 3, 3))
        axes[:, 0] = np.eye(3)
        axes[:, 1:] = world[:, parents] @ _left_jacobian(vectors[:, free[1:]])
    dense_rows, dense_values = [], []

    if problem.pos_joints.size:
        reached = positions[:, problem.pos_joints]
        weight = problem.pos_weight[..., None]
        value = weight * (reached - problem.pos_target)
        cost += np.einsum("tki,tki->t", value, value)
        if normal_equations:
            rows, slots = problem.pos_pairs
            lever = reached[:, rows] - positions[:, free[slots]]
            jacobian = _rows(frames, problem.pos_joints.size, size, problem.pos_pairs,
                             -_skew(lever) @ axes[:, slots])
            jacobian[..., :3] = np.eye(3)
            dense_rows.append((weight[..., None] * jacobian).reshape(frames, -1, size))
            dense_values.append(value.reshape(frames, -1))

    if problem.ori_joints.size:
        segment = world[:, problem.ori_joints]
        error = matrix_to_axis_angle(np.swapaxes(problem.ori_target, -1, -2) @ segment)
        weight = problem.ori_weight[..., None]
        value = weight * error
        cost += np.einsum("tki,tki->t", value, value)
        if normal_equations:
            rows, slots = problem.ori_pairs
            lead = _right_jacobian_inverse(error) @ np.swapaxes(segment, -1, -2)
            jacobian = _rows(frames, problem.ori_joints.size, size, problem.ori_pairs,
                             lead[:, rows] @ axes[:, slots])
            dense_rows.append((weight[..., None] * jacobian).reshape(frames, -1, size))
            dense_values.append(value.reshape(frames, -1))

    if normal_equations and dense_rows:
        # One dense product: at these sizes BLAS beats summing the sparse 3x3 blocks.
        rows = np.concatenate(dense_rows, axis=1)
        values = np.concatenate(dense_values, axis=1)
        transposed = np.swapaxes(rows, 1, 2)
        normal += transposed @ rows
        gradient += (transposed @ values[..., None])[..., 0]

    joints = free[1:]
    weight = problem.prior_weights
    if joints.size and weight.any():
        value = weight[None] * vectors[:, joints]
        cost += np.einsum("tki,tki->t", value, value)
        if normal_equations:
            index = 3 + 3 * np.arange(1, free.size)
            for axis in range(3):
                normal[:, index + axis, index + axis] += weight[:, axis] ** 2
            # after the translation and the root
            gradient[:, 6:] += (weight[None] * value).reshape(frames, -1)

    weight = problem.smoothing_weight
    if weight > 0.0:
        turn = matrix_to_axis_angle(np.swapaxes(problem.ref_root, -1, -2) @ root)
        drift = weight * np.concatenate([
            (trans - problem.ref_trans)[:, None],
            turn[:, None],
            vectors[:, joints] - problem.ref_vectors[:, joints],
        ], axis=1)
        cost += np.einsum("tki,tki->t", drift, drift)
        if normal_equations:
            _add_block(normal, gradient, -1, np.broadcast_to(weight * np.eye(3), (frames, 3, 3)),
                       drift[:, 0])
            _add_block(normal, gradient, 0,
                       weight * _right_jacobian_inverse(turn) @ np.swapaxes(root, -1, -2),
                       drift[:, 1])
            index = 3 + 3 * np.arange(1, free.size)
            for axis in range(3):
                normal[:, index + axis, index + axis] += weight**2
            gradient[:, 6:] += weight * drift[:, 2:].reshape(frames, -1)

    return cost, normal, gradient


def _apply(problem: _Problem, root, vectors, trans, step):
    frames = root.shape[0]
    moves = step[:, 3:].reshape(frames, problem.free.size, 3)
    new_vectors = vectors.copy()
    new_vectors[:, problem.free[1:]] += moves[:, 1:]
    return axis_angle_to_matrix(moves[:, 0]) @ root, new_vectors, trans + step[:, :3]


def _levenberg_marquardt(problem: _Problem, local, trans, options: PoseSettings):
    root = local[:, ROOT].copy()
    vectors = matrix_to_axis_angle(local)
    trans = trans.copy()
    cost, normal, gradient = _evaluate(problem, root, vectors, trans, normal_equations=True)
    damping = np.full(root.shape[0], options.damping)
    active = np.ones(root.shape[0], dtype=bool)
    identity = np.eye(problem.size)
    for _ in range(options.iterations):
        frames = np.flatnonzero(active)
        if frames.size == 0:
            break
        damped = normal[frames] + damping[frames, None, None] * identity
        step = -np.linalg.solve(damped, gradient[frames][..., None])[..., 0]
        subset = problem.take(frames)
        candidate = _apply(subset, root[frames], vectors[frames], trans[frames], step)
        new_cost, _, _ = _evaluate(subset, *candidate, normal_equations=False)
        better = new_cost < cost[frames]
        kept = frames[better]
        if kept.size:
            root[kept], vectors[kept], trans[kept] = (part[better] for part in candidate)
            cost[kept], normal[kept], gradient[kept] = _evaluate(
                problem.take(kept), root[kept], vectors[kept], trans[kept], normal_equations=True
            )
            damping[kept] = np.maximum(damping[kept] / 3.0, _TINY)
        damping[frames[~better]] *= 4.0
        done = np.abs(step).max(axis=1) < options.tolerance
        active[frames[done]] = False
    return _local(root, vectors), trans


def _solve(rest, targets: Targets, options, start, calibration=None, references=None):
    local, trans = start
    out_local, out_trans = local.copy(), trans.copy()
    for first in range(0, targets.frames, options.chunk_frames):
        frames = slice(first, first + options.chunk_frames)
        chunk_refs = None
        if references is not None:
            chunk_refs = tuple(part[frames] for part in references)
        problem = _problem(rest, targets.take(frames), options, calibration, chunk_refs)
        out_local[frames], out_trans[frames] = _levenberg_marquardt(
            problem, local[frames], trans[frames], options
        )
    return out_local, out_trans


# --- calibration, smoothing, reporting ----------------------------------------------------------


def calibrate_orientations(rest, local, trans, orientations: OrientationTargets) -> np.ndarray:
    """``(M, 3, 3)``: per target, the mean over valid frames of ``S^T G`` for the solved pose.

    A target valid on no frame gets the identity; its weight is zero wherever it is used.
    """
    _, world = fk_batch(rest, local, trans)
    relative = np.swapaxes(orientations.rotations, -1, -2) @ world[:, orientations.joints]
    out = np.tile(np.eye(3), (len(orientations.joints), 1, 1))
    for column in range(len(orientations.joints)):
        valid = orientations.valid[:, column]
        if valid.any():
            out[column] = mean_rotation(relative[valid, column])
    return out


def _neighbour_average(local, trans, contiguous):
    """The smoothing references: per frame, the midpoint of its two neighbours (slerp at one
    half) as a root rotation and as rotation vectors, and the mean neighbour translation. A frame
    without two contiguous neighbours references itself, and so does a joint whose midpoint's
    rotation vector lies on the far side of a half turn from its own."""
    vectors = matrix_to_axis_angle(local)
    ref_root, ref_vectors, ref_trans = local[:, ROOT].copy(), vectors.copy(), trans.copy()
    inner = np.flatnonzero(contiguous)
    if inner.size:
        half = 0.5 * matrix_to_axis_angle(
            np.swapaxes(local[inner - 1], -1, -2) @ local[inner + 1]
        )
        middle = local[inner - 1] @ axis_angle_to_matrix(half)
        ref_root[inner] = middle[:, ROOT]
        middle_vectors = matrix_to_axis_angle(middle)
        near = np.linalg.norm(middle_vectors - vectors[inner], axis=-1) < np.pi / 2
        ref_vectors[inner] = np.where(near[..., None], middle_vectors, vectors[inner])
        ref_trans[inner] = 0.5 * (trans[inner - 1] + trans[inner + 1])
    return ref_root, ref_vectors, ref_trans


def hold_nearest(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid frames with the nearest valid frame's value (earlier wins a tie)."""
    frames = np.arange(valid.size)
    good = np.flatnonzero(valid)
    position = np.searchsorted(good, frames)
    left = good[np.clip(position - 1, 0, good.size - 1)]
    right = good[np.clip(position, 0, good.size - 1)]
    nearest = np.where(np.abs(frames - left) <= np.abs(right - frames), left, right)
    return values[nearest]


def _position_error(rest, local, trans, targets: Targets) -> TargetError:
    positions, _ = fk_batch(rest, local, trans)
    columns = targets.positions
    distance = np.linalg.norm(positions[:, columns.joints] - columns.positions, axis=2)
    valid = columns.valid
    per_target = {}
    for column, label in enumerate(columns.labels):
        values = distance[valid[:, column], column]
        per_target[label] = float(np.sqrt(np.mean(values**2))) if values.size else float("nan")
    values = distance[valid]
    return TargetError(
        rms=float(np.sqrt(np.mean(values**2))) if values.size else float("nan"),
        max=float(values.max()) if values.size else float("nan"),
        per_target=per_target,
    )


def _orientation_error(rest, local, trans, targets: Targets, calibration) -> TargetError:
    _, world = fk_batch(rest, local, trans)
    orientations = targets.orientations
    target = orientations.rotations @ calibration[None]
    angle = np.linalg.norm(
        matrix_to_axis_angle(np.swapaxes(target, -1, -2) @ world[:, orientations.joints]), axis=2
    )
    per_target = {}
    for column, label in enumerate(orientations.labels):
        values = angle[orientations.valid[:, column], column]
        per_target[label] = float(np.sqrt(np.mean(values**2))) if values.size else float("nan")
    values = angle[orientations.valid]
    return TargetError(
        rms=float(np.sqrt(np.mean(values**2))) if values.size else float("nan"),
        max=float(values.max()) if values.size else float("nan"),
        per_target=per_target,
    )


def solve_pose(rest, targets: Targets, settings: Mapping[str, Any], *,
               positions_only: bool = False, calibration: np.ndarray | None = None,
               start: PoseFit | None = None) -> PoseFit:
    """Solve every frame of ``targets`` against the rest skeleton ``rest`` ``(24, 3)``.

    ``positions_only`` skips the orientation and smoothing passes, which is what a shape fit and
    a calibration want. ``calibration`` gives the orientation constants ``(M, 3, 3)`` instead of
    calibrating them on this trial alone, so that one subject's trials share them. ``start`` is a
    positions-only solve of the same targets on the same skeleton, which then is not repeated.
    """
    options = PoseSettings.from_settings(settings)
    rest = np.asarray(rest, dtype=np.float64)
    solved = targets.positions.valid.sum(axis=1) >= options.min_position_targets
    if not solved.any():
        raise ValueError(
            f"no frame has {options.min_position_targets} valid position targets; nothing to solve"
        )
    frames = np.flatnonzero(solved)
    subset = targets.take(frames)
    if start is not None:
        if start.passes != ("positions",) or not np.array_equal(start.frame_valid, solved):
            raise ValueError("start must be a positions-only solve of the same targets")
        local, trans = start.local[frames], start.trans[frames]
    else:
        local, trans = _solve(rest, subset, options, initial_pose(rest, subset))
    passes = ["positions"]

    use_orientations = (not positions_only and subset.orientations is not None
                        and options.orientation_weight > 0)
    if use_orientations:
        if calibration is None:
            calibration = calibrate_orientations(rest, local, trans, subset.orientations)
        elif calibration.shape != (len(subset.orientations.joints), 3, 3):
            raise ValueError("calibration must hold one constant per orientation target")
        local, trans = _solve(rest, subset, options, (local, trans), calibration)
        passes.append("orientations")
    else:
        calibration = None

    if not positions_only and options.smoothing_weight > 0 and frames.size >= 3:
        contiguous = np.zeros(frames.size, dtype=bool)
        contiguous[1:-1] = (frames[1:-1] - frames[:-2] == 1) & (frames[2:] - frames[1:-1] == 1)
        references = _neighbour_average(local, trans, contiguous)
        local, trans = _solve(rest, subset, options, (local, trans), calibration, references)
        passes.append("smoothing")

    position_error = _position_error(rest, local, trans, subset)
    orientation_error = (
        None if calibration is None else _orientation_error(rest, local, trans, subset, calibration)
    )
    full_local = np.empty((targets.frames, NUM_JOINTS, 3, 3))
    full_trans = np.empty((targets.frames, 3))
    full_local[frames], full_trans[frames] = local, trans
    if not solved.all():
        full_local = hold_nearest(np.where(solved[:, None, None, None], full_local, 0.0), solved)
        full_trans = hold_nearest(np.where(solved[:, None], full_trans, 0.0), solved)
    return PoseFit(
        local=full_local,
        trans=full_trans,
        frame_valid=solved,
        calibration=calibration,
        position_error=position_error,
        orientation_error=orientation_error,
        passes=tuple(passes),
    )
