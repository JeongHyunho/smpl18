"""SMPL rest skeleton and forward kinematics, as in ``docs/primer.md`` sections 2.2 and 2.3.

Translation convention, stated once: ``trans`` moves the model origin, not the pelvis. The
pelvis sits at ``j_0`` in the rest skeleton, so its world position is ``p_0 = j_0 + trans``.
A caller holding a pelvis-rooted translation ``p_0`` passes ``p_0 - j_0``.

All batched operations use one formulation (``einsum`` for the rotated offsets, stacked
``matmul`` for the rotation chain) so that a single frame and a batch of frames agree to the bit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .definition import NUM_JOINTS, PARENTS, SEGMENTS
from .rotations import axis_angle_to_matrix

if TYPE_CHECKING:
    from smpl24.model.load import Model

__all__ = [
    "fk",
    "fk_batch",
    "global_rotations",
    "local_rotations",
    "rest_joints",
    "segment_lengths",
    "shaped_vertices",
]


def shaped_vertices(model: Model, betas) -> np.ndarray:
    """``v_template + shapedirs . betas`` with ``betas`` zero-padded or truncated to the model's width."""
    shapedirs = model.shapedirs
    width = shapedirs.shape[2]
    padded = np.zeros(width, dtype=np.float64)
    given = np.asarray(betas, dtype=np.float64).ravel()
    take = min(width, given.size)
    padded[:take] = given[:take]
    return model.v_template + np.einsum("vij,j->vi", shapedirs, padded)


def rest_joints(model: Model, betas) -> np.ndarray:
    """Rest-pose joint centres ``J(betas) = J_regressor @ shaped_vertices``, shape ``(24, 3)``."""
    return model.J_regressor @ shaped_vertices(model, betas)


def segment_lengths(rest: np.ndarray, segments=SEGMENTS) -> np.ndarray:
    """Straight-line length of each ``(name, proximal, distal)`` segment in the rest skeleton."""
    rest = np.asarray(rest, dtype=np.float64)
    return np.array(
        [np.linalg.norm(rest[distal] - rest[proximal]) for _name, proximal, distal in segments],
        dtype=np.float64,
    )


def _as_local_matrices(poses, *, batched: bool) -> np.ndarray:
    """``(..., 24, 3)`` axis-angle or ``(..., 24, 3, 3)`` matrices to ``(T, 24, 3, 3)`` float64."""
    poses = np.asarray(poses, dtype=np.float64)
    leading = 1 if batched else 0
    if poses.ndim == leading + 2 and poses.shape[-2:] == (NUM_JOINTS, 3):
        matrices = axis_angle_to_matrix(poses)
    elif poses.ndim == leading + 3 and poses.shape[-3:] == (NUM_JOINTS, 3, 3):
        matrices = poses
    else:
        raise ValueError(
            f"expected {'(T, ' if batched else '('}24, 3) axis-angle or "
            f"{'(T, ' if batched else '('}24, 3, 3) matrices, got {poses.shape}"
        )
    return matrices if batched else matrices[None]


def global_rotations(local: np.ndarray) -> np.ndarray:
    """Accumulate local rotations ``(T, 24, 3, 3)`` along the tree: ``G_j = G_parent(j) @ R_j``."""
    local = np.asarray(local, dtype=np.float64)
    if local.shape[-3:] != (NUM_JOINTS, 3, 3):
        raise ValueError(f"expected (T, 24, 3, 3) local rotations, got {local.shape}")
    accumulated = np.empty_like(local)
    accumulated[:, 0] = local[:, 0]
    for joint in range(1, NUM_JOINTS):
        accumulated[:, joint] = accumulated[:, PARENTS[joint]] @ local[:, joint]
    return accumulated


def local_rotations(accumulated: np.ndarray) -> np.ndarray:
    """Invert :func:`global_rotations`: ``R_j = G_parent(j)^T @ G_j``, the root unchanged."""
    accumulated = np.asarray(accumulated, dtype=np.float64)
    if accumulated.shape[-3:] != (NUM_JOINTS, 3, 3):
        raise ValueError(f"expected (T, 24, 3, 3) global rotations, got {accumulated.shape}")
    local = np.empty_like(accumulated)
    local[:, 0] = accumulated[:, 0]
    for joint in range(1, NUM_JOINTS):
        local[:, joint] = np.einsum(
            "tji,tjk->tik", accumulated[:, PARENTS[joint]], accumulated[:, joint]
        )
    return local


def fk_batch(rest: np.ndarray, poses, trans) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics over ``T`` frames.

    ``rest`` is ``(24, 3)`` from :func:`rest_joints`; ``poses`` is ``(T, 24, 3)`` axis-angle
    or ``(T, 24, 3, 3)`` local rotation matrices; ``trans`` is ``(T, 3)``. Returns joint
    positions ``(T, 24, 3)`` and the world rotation of the segment each joint moves,
    ``(T, 24, 3, 3)``. The pelvis lands at ``rest[0] + trans``.
    """
    rest = np.asarray(rest, dtype=np.float64)
    if rest.shape != (NUM_JOINTS, 3):
        raise ValueError(f"expected (24, 3) rest joints, got {rest.shape}")
    local = _as_local_matrices(poses, batched=True)
    trans = np.asarray(trans, dtype=np.float64)
    if trans.shape != (local.shape[0], 3):
        raise ValueError(f"expected ({local.shape[0]}, 3) translations, got {trans.shape}")

    world_rotation = global_rotations(local)
    positions = np.empty((local.shape[0], NUM_JOINTS, 3), dtype=np.float64)
    positions[:, 0] = rest[0] + trans
    for joint in range(1, NUM_JOINTS):
        parent = PARENTS[joint]
        positions[:, joint] = positions[:, parent] + np.einsum(
            "tij,j->ti", world_rotation[:, parent], rest[joint] - rest[parent]
        )
    return positions, world_rotation


def fk(rest: np.ndarray, pose, trans) -> tuple[np.ndarray, np.ndarray]:
    """One frame of :func:`fk_batch`: ``pose`` is ``(24, 3)`` or ``(24, 3, 3)``, ``trans`` is ``(3,)``."""
    trans = np.asarray(trans, dtype=np.float64)
    if trans.shape != (3,):
        raise ValueError(f"expected a (3,) translation, got {trans.shape}")
    positions, world_rotation = fk_batch(rest, _as_local_matrices(pose, batched=False), trans[None])
    return positions[0], world_rotation[0]
