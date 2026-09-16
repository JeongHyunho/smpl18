"""What a fit aims at: observations already named as SMPL-24 joints, in metres, in the output frame.

Every source kind except ``smpl_parameters`` becomes a :class:`Targets` before anything is fitted.
A marker set turns labelled markers into joint centres and segment frames; a skeleton's forward
kinematics gives joint centres and segment rotations; a joint-centre file is read as it stands.
From here on nothing knows where the numbers came from, which is what lets one shape fit and one
pose solve serve every kind.

Two kinds of target exist:

* a **position** target says where an SMPL joint centre is, ``(T, K, 3)``;
* an **orientation** target says how the segment an SMPL joint moves is turned, ``(T, M, 3, 3)``.
  The source's segment frame is not the SMPL segment frame, so an orientation target is compared
  after one constant per target, which the pose solve calibrates (``smpl18.fit.pose``).

Provenance is decided here, once, from which joints the targets reach (:func:`provenance_for`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from smpl18.skeleton.definition import CHILDREN, JOINT_NAMES, NUM_JOINTS, PARENTS, ROOT
from smpl18.sources.base import Provenance

__all__ = [
    "OrientationTargets",
    "PositionTargets",
    "Targets",
    "concatenate",
    "provenance_for",
]

#: Rules a correspondence or marker set may give a joint that no target reaches directly.
FILL_WELD = "weld"
FILL_DISTRIBUTE = "distribute"
FILL_RULES = (FILL_WELD, FILL_DISTRIBUTE)


def _joint_indices(joints: Sequence[int | str]) -> tuple[int, ...]:
    out = []
    for joint in joints:
        index = JOINT_NAMES.index(joint) if isinstance(joint, str) else int(joint)
        if not 0 <= index < NUM_JOINTS:
            raise ValueError(f"no SMPL-24 joint {joint!r}")
        out.append(index)
    return tuple(out)


@dataclass(frozen=True)
class PositionTargets:
    """Where SMPL joint centres are: one column per observation, several may name one joint."""

    joints: tuple[int, ...]
    #: ``(T, K, 3)`` metres; NaN where ``valid`` is false.
    positions: np.ndarray
    #: ``(T, K)``.
    valid: np.ndarray
    #: ``(K,)`` relative weights; the solve multiplies them by its position weight.
    weights: np.ndarray
    #: What each column was made from, for reports (a centre name, a rule, a source joint).
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        joints = _joint_indices(self.joints)
        positions = np.asarray(self.positions, dtype=np.float64)
        if positions.ndim != 3 or positions.shape[1:] != (len(joints), 3):
            raise ValueError(f"positions must be (T, {len(joints)}, 3), got {positions.shape}")
        valid = np.asarray(self.valid, dtype=bool) & np.isfinite(positions).all(axis=2)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if weights.shape != (len(joints),) or (weights < 0).any():
            raise ValueError(f"weights must be {len(joints)} non-negative numbers")
        if len(self.labels) != len(joints):
            raise ValueError(f"labels must name {len(joints)} columns")
        positions = np.where(valid[..., None], positions, np.nan)
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "labels", tuple(self.labels))

    @property
    def frames(self) -> int:
        return int(self.positions.shape[0])

    def take(self, frames) -> PositionTargets:
        """The same targets on a subset of frames (an index array or a slice)."""
        return PositionTargets(self.joints, self.positions[frames], self.valid[frames],
                               self.weights, self.labels)


@dataclass(frozen=True)
class OrientationTargets:
    """How the segments some SMPL joints move are turned, in the source's own segment frames."""

    joints: tuple[int, ...]
    #: ``(T, M, 3, 3)`` world rotation of the source segment; identity where ``valid`` is false.
    rotations: np.ndarray
    #: ``(T, M)``.
    valid: np.ndarray
    #: ``(M,)`` relative weights; the solve multiplies them by its orientation weight.
    weights: np.ndarray
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        joints = _joint_indices(self.joints)
        rotations = np.asarray(self.rotations, dtype=np.float64)
        if rotations.ndim != 4 or rotations.shape[1:] != (len(joints), 3, 3):
            raise ValueError(f"rotations must be (T, {len(joints)}, 3, 3), got {rotations.shape}")
        valid = np.asarray(self.valid, dtype=bool) & np.isfinite(rotations).all(axis=(2, 3))
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if weights.shape != (len(joints),) or (weights < 0).any():
            raise ValueError(f"weights must be {len(joints)} non-negative numbers")
        if len(self.labels) != len(joints):
            raise ValueError(f"labels must name {len(joints)} columns")
        rotations = np.where(valid[..., None, None], rotations, np.eye(3))
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "labels", tuple(self.labels))

    @property
    def frames(self) -> int:
        return int(self.rotations.shape[0])

    def take(self, frames) -> OrientationTargets:
        return OrientationTargets(self.joints, self.rotations[frames], self.valid[frames],
                                  self.weights, self.labels)


@dataclass(frozen=True)
class Targets:
    """Everything one trial offers the fit, and the provenance that follows from it."""

    positions: PositionTargets
    orientations: OrientationTargets | None
    fps: float
    #: One of ``measured`` / ``derived`` / ``absent`` per SMPL-24 joint (:func:`provenance_for`).
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.orientations is not None and self.orientations.frames != self.positions.frames:
            raise ValueError("position and orientation targets disagree on the frame count")
        provenance = tuple(Provenance(value).value for value in self.provenance)
        if len(provenance) != NUM_JOINTS:
            raise ValueError(f"provenance must name {NUM_JOINTS} joints")
        object.__setattr__(self, "provenance", provenance)

    @property
    def frames(self) -> int:
        return self.positions.frames

    @property
    def free_joints(self) -> tuple[int, ...]:
        """Joints the pose solve may turn: the root, and every joint not ``absent``."""
        return tuple(
            joint for joint in range(NUM_JOINTS)
            if joint == ROOT or self.provenance[joint] != Provenance.ABSENT.value
        )

    def take(self, frames) -> Targets:
        return Targets(
            self.positions.take(frames),
            None if self.orientations is None else self.orientations.take(frames),
            self.fps,
            self.provenance,
        )


def _subtree(joint: int) -> tuple[int, ...]:
    out = [joint]
    for child in CHILDREN[joint]:
        out.extend(_subtree(child))
    return tuple(out)


def provenance_for(
    position_joints: Sequence[int | str],
    orientation_joints: Sequence[int | str] = (),
    fill: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Decide, from which joints the targets reach, what each SMPL joint's rotation rests on.

    * ``measured``: an orientation target observes the joint's segment, or the joint and one of
      its children both carry a position target, so the bone between them is seen at both ends.
      The root is measured whenever anything is observed.
    * ``derived``: not measured, but a target lies somewhere below it, so the solve turns it --
      the rotation is shared out among the joints above that target by the pose prior.
    * ``absent``: nothing below it is observed; the solve holds it at identity (welded).

    ``fill`` overrides the last two for named joints: ``weld`` makes a joint ``absent`` (held at
    identity even though something below it is observed), ``distribute`` makes it ``derived``.
    A measured joint cannot be welded; that would discard an observation.
    """
    positioned = set(_joint_indices(position_joints))
    oriented = set(_joint_indices(orientation_joints))
    observed = positioned | oriented
    out = []
    for joint in range(NUM_JOINTS):
        if joint == ROOT:
            measured = bool(observed)
        else:
            measured = joint in oriented or (
                joint in positioned and any(child in positioned for child in CHILDREN[joint])
            )
        below = any(node in observed for node in _subtree(joint) if node != joint) or (
            joint in oriented
        )
        if measured:
            out.append(Provenance.MEASURED.value)
        elif below:
            out.append(Provenance.DERIVED.value)
        else:
            out.append(Provenance.ABSENT.value)
    for name, rule in (fill or {}).items():
        joint = JOINT_NAMES.index(name) if name in JOINT_NAMES else None
        if joint is None:
            raise ValueError(f"fill names {name!r}, which is not an SMPL-24 joint")
        if rule not in FILL_RULES:
            raise ValueError(f"fill rule for {name} must be one of {FILL_RULES}, got {rule!r}")
        if out[joint] == Provenance.MEASURED.value:
            if rule == FILL_WELD:
                raise ValueError(f"{name} is observed directly and cannot be welded")
            continue
        if joint == ROOT:
            continue
        out[joint] = Provenance.ABSENT.value if rule == FILL_WELD else Provenance.DERIVED.value
    return tuple(out)


def concatenate(parts: Sequence[Targets]) -> Targets:
    """Several trials' targets end to end, for a subject-level fit. The columns must agree."""
    if not parts:
        raise ValueError("nothing to concatenate")
    first = parts[0]
    for other in parts[1:]:
        if other.positions.joints != first.positions.joints:
            raise ValueError("trials disagree on their position targets")
        if (other.orientations is None) != (first.orientations is None) or (
            other.orientations is not None and other.orientations.joints != first.orientations.joints
        ):
            raise ValueError("trials disagree on their orientation targets")
    positions = PositionTargets(
        first.positions.joints,
        np.concatenate([p.positions.positions for p in parts]),
        np.concatenate([p.positions.valid for p in parts]),
        first.positions.weights,
        first.positions.labels,
    )
    orientations = None
    if first.orientations is not None:
        orientations = OrientationTargets(
            first.orientations.joints,
            np.concatenate([p.orientations.rotations for p in parts]),
            np.concatenate([p.orientations.valid for p in parts]),
            first.orientations.weights,
            first.orientations.labels,
        )
    return Targets(positions, orientations, first.fps, first.provenance)


assert all(PARENTS[j] < j for j in range(1, NUM_JOINTS))
