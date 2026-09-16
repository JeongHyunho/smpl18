"""What the fixation costs, in metres: the joint centres it moves.

Orientation survives any constants (see :mod:`smpl18.reduce.fixation`), so position is the only
thing a constant can get wrong, and it is measured one way here so that the fit, the record in
the corpus and any later validation report the same number. The comparison runs both poses
through the same forward kinematics with the same rest skeleton and a zero translation: the
pelvis is common to both, so a translation would cancel anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smpl18.skeleton.definition import AFFECTED_BY_FREEZE, JOINT_NAMES
from smpl18.skeleton.kinematics import fk_batch

from .fixation import apply

__all__ = ["PositionError", "displacement", "per_joint_rms", "residual"]

#: ``AFFECTED_BY_FREEZE`` as an index array, for the same reason ``fixation._KEEP`` is one.
_AFFECTED = np.array(AFFECTED_BY_FREEZE, dtype=np.int64)


@dataclass(frozen=True)
class PositionError:
    """How far the fixation moved the affected joint centres; plain floats, for the corpus."""

    rms_m: float
    max_m: float


def displacement(local_rotations, rest_joints, constants) -> np.ndarray:
    """Per-frame displacement ``(T, 9, 3)`` of the affected joints, reduced minus original.

    Kept as vectors rather than distances because the fit needs the signed components and the
    summaries below need only a norm away from them.
    """
    reduced_rotations = apply(local_rotations, constants)      # validates both shapes
    frames = reduced_rotations.shape[0]
    if frames == 0:
        raise ValueError("cannot measure the fixation over zero frames")
    trans = np.zeros((frames, 3), dtype=np.float64)
    original, _ = fk_batch(rest_joints, local_rotations, trans)
    reduced, _ = fk_batch(rest_joints, reduced_rotations, trans)
    return reduced[:, _AFFECTED] - original[:, _AFFECTED]


def residual(local_rotations, rest_joints, constants) -> PositionError:
    """RMS and worst joint-centre displacement over every frame given, in metres."""
    distance = np.linalg.norm(displacement(local_rotations, rest_joints, constants), axis=2)
    return PositionError(
        rms_m=float(np.sqrt(np.mean(np.square(distance)))),
        max_m=float(distance.max()),
    )


def per_joint_rms(local_rotations, rest_joints, constants) -> dict[str, float]:
    """The same error split per affected joint, so a reader sees where the cost landed.

    A fit that leaves a large residual usually leaves it in one place -- the shoulder girdle of a
    subject who shrugs, say -- and a single pooled number hides that.
    """
    distance = np.linalg.norm(displacement(local_rotations, rest_joints, constants), axis=2)
    return {
        JOINT_NAMES[joint]: float(np.sqrt(np.mean(np.square(distance[:, column]))))
        for column, joint in enumerate(AFFECTED_BY_FREEZE)
    }
