"""Applying the fixation: four constants in, a reduced 24-joint pose out, then the 18 kept.

Three lines do the whole reduction (``docs/primer.md`` section 5.1). Each frozen joint is set to
its constant and the kept joint just below it is pre-multiplied by the inverse of what was
removed, so the product along the chain is unchanged: **every world orientation below a frozen
joint survives exactly, for any constants at all**. Nothing here chooses the constants; that is
:mod:`smpl18.reduce.fit`, and what a choice costs is :mod:`smpl18.reduce.residual`.
"""

from __future__ import annotations

import numpy as np

from smpl18.skeleton.definition import (
    ABSORBERS,
    FROZEN_JOINT_NAMES,
    FROZEN_JOINTS,
    JOINT18_NAMES,
    KEEP18,
    NUM_JOINTS,
)

__all__ = ["apply", "as_constants", "as_local_rotations", "names_18", "to_18"]

#: ``KEEP18`` as an index array: a tuple in ``array[:, KEEP18]`` would be read as further axes.
_KEEP = np.array(KEEP18, dtype=np.int64)

_SPINE1, _SPINE2, _LEFT_COLLAR, _RIGHT_COLLAR = FROZEN_JOINTS
_SPINE3 = ABSORBERS[_SPINE1]
_LEFT_SHOULDER = ABSORBERS[_LEFT_COLLAR]
_RIGHT_SHOULDER = ABSORBERS[_RIGHT_COLLAR]


def as_local_rotations(local_rotations) -> np.ndarray:
    """``(T, 24, 3, 3)`` float64 local rotations, or a ``ValueError`` naming what arrived."""
    local = np.asarray(local_rotations, dtype=np.float64)
    if local.ndim != 4 or local.shape[1:] != (NUM_JOINTS, 3, 3):
        raise ValueError(f"expected (T, 24, 3, 3) local rotations, got {local.shape}")
    return local


def as_constants(constants) -> np.ndarray:
    """``(4, 3, 3)`` float64 constants, one per frozen joint in the order of ``FROZEN_JOINTS``."""
    values = np.asarray(constants, dtype=np.float64)
    if values.shape != (len(FROZEN_JOINTS), 3, 3):
        raise ValueError(
            f"expected ({len(FROZEN_JOINTS)}, 3, 3) constants, one per frozen joint "
            f"{FROZEN_JOINT_NAMES}, got {values.shape}"
        )
    return values


def apply(local_rotations, constants) -> np.ndarray:
    """Fix the four joints and hand their motion to the joint below, keeping orientations exact.

    ``local_rotations`` is ``(T, 24, 3, 3)`` and ``constants`` is ``(4, 3, 3)`` in the order of
    ``FROZEN_JOINTS``. The result is ``(T, 24, 3, 3)`` again -- still 24 joints, because the
    absorbed rotation has to be written somewhere before the kept 18 are taken by :func:`to_18`.
    Joints that are neither frozen nor absorbers come through untouched, bit for bit.
    """
    local = as_local_rotations(local_rotations)
    constant = dict(zip(FROZEN_JOINTS, as_constants(constants)))

    reduced = local.copy()
    reduced[:, _SPINE3] = constant[_SPINE2].T @ constant[_SPINE1].T @ (
        local[:, _SPINE1] @ local[:, _SPINE2] @ local[:, _SPINE3]
    )
    reduced[:, _LEFT_SHOULDER] = constant[_LEFT_COLLAR].T @ (
        local[:, _LEFT_COLLAR] @ local[:, _LEFT_SHOULDER]
    )
    reduced[:, _RIGHT_SHOULDER] = constant[_RIGHT_COLLAR].T @ (
        local[:, _RIGHT_COLLAR] @ local[:, _RIGHT_SHOULDER]
    )
    for joint, matrix in constant.items():
        reduced[:, joint] = matrix
    return reduced


def to_18(reduced_24: np.ndarray) -> np.ndarray:
    """Take the 18 stored joints along axis 1 of any ``(T, 24, ...)`` array.

    Rotations, positions, provenance codes: only the joint axis matters, so the trailing shape
    and the dtype are the caller's. Call it on the output of :func:`apply`, never on a pose whose
    frozen joints still move -- the rotations it drops are redundant only after the fixation.
    """
    values = np.asarray(reduced_24)
    if values.ndim < 2 or values.shape[1] != NUM_JOINTS:
        raise ValueError(f"expected (T, 24, ...) with the joints on axis 1, got {values.shape}")
    return values[:, _KEEP]


def names_18() -> tuple[str, ...]:
    """The joint names in the order :func:`to_18` returns them, for labelling that axis."""
    return JOINT18_NAMES
