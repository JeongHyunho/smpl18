"""Rotation algebra: quaternions in ``(w, x, y, z)`` order, conversions, slerp, Kabsch.

Quaternions are written against the ``(w, x, y, z)`` order directly rather than converted at
every call site, which is where sign and order mistakes hide. ``q`` and ``-q`` are the same
rotation, so :func:`canonicalise_sign` goes before any interpolation.

Conversions between axis-angle, matrices and quaternions go through ``scipy.spatial.transform``
in the exact form the converters used before this package existed, so their results are the
same to the bit; :func:`quaternion_to_matrix` keeps its explicit formula for the same reason.
Every operation accepts a batch over leading axes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "axis_angle_to_matrix",
    "body_angular_velocity",
    "canonicalise_sign",
    "conjugate",
    "exp_pure",
    "kabsch_rotation",
    "log_unit",
    "matrix_to_axis_angle",
    "matrix_to_quaternion",
    "mean_rotation",
    "multiply",
    "normalise",
    "proper_rotation_from_covariance",
    "quaternion_to_matrix",
    "slerp",
]


# --- quaternion algebra ------------------------------------------------------------------------


def normalise(q: np.ndarray) -> np.ndarray:
    """Scale quaternions to unit norm along the last axis."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("cannot normalise a zero quaternion")
    return q / norm


def conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate, which is the inverse for unit quaternions."""
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, broadcasting over leading axes."""
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def canonicalise_sign(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions of a ``(frames, 4)`` series share a hemisphere.

    Without this a source that legitimately emits ``-q`` on one frame makes any interpolation
    swing through the far side of the sphere, which downstream looks like an enormous angular
    velocity.
    """
    out = np.asarray(q, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 4:
        raise ValueError(f"expected a (frames, 4) quaternion series, got {out.shape}")
    for index in range(1, out.shape[0]):
        if np.dot(out[index - 1], out[index]) < 0.0:
            out[index] = -out[index]
    return out


def log_unit(q: np.ndarray) -> np.ndarray:
    """Logarithm of a unit quaternion as a pure quaternion ``(0, v)``.

    ``v`` is the rotation axis scaled by half the rotation angle.
    """
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    vector = q[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1)
    scalar = np.clip(q[..., 0], -1.0, 1.0)
    angle = np.arctan2(vector_norm, scalar)

    # As the rotation vanishes the axis becomes undefined; the limit of angle/sin(angle) is
    # 1, so scaling by the angle directly gives the correct zero vector without a 0/0.
    scale = np.where(vector_norm > 1e-12, angle / np.where(vector_norm > 1e-12, vector_norm, 1.0), 0.0)
    out = np.zeros_like(q)
    out[..., 1:] = vector * scale[..., None]
    return out


def exp_pure(v: np.ndarray) -> np.ndarray:
    """Exponential of a pure quaternion ``(0, v)``, giving a unit quaternion."""
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    vector = v[..., 1:]
    angle = np.linalg.norm(vector, axis=-1)
    out = np.zeros_like(v)
    out[..., 0] = np.cos(angle)
    scale = np.where(angle > 1e-12, np.sin(angle) / np.where(angle > 1e-12, angle, 1.0), 1.0)
    out[..., 1:] = vector * scale[..., None]
    return out


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between two quaternion series at fractions ``t``."""
    q0 = np.atleast_2d(np.asarray(q0, dtype=np.float64))
    q1 = np.atleast_2d(np.asarray(q1, dtype=np.float64))
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))[:, None]

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)

    # As theta goes to zero, sin((1-t)theta)/sin(theta) tends to (1-t) and sin(t.theta)/
    # sin(theta) tends to t. Those limits are evaluated directly for near-parallel inputs,
    # where the ratio would otherwise be 0/0.
    close = dot > 1.0 - 1e-9
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    safe_sin = np.where(close, 1.0, sin_theta)
    weight0 = np.where(close, 1.0 - t, np.sin((1.0 - t) * theta) / safe_sin)
    weight1 = np.where(close, t, np.sin(t * theta) / safe_sin)
    return normalise(weight0 * q0 + weight1 * q1)


def body_angular_velocity(orientation_wxyz: np.ndarray, dt_s: float) -> np.ndarray:
    """Body-frame angular velocity of a ``(frames, 4)`` series by the central log map.

    ``log(q(t-dt)^-1 q(t+dt)) / (2 dt)`` is exact for a constant rate, where differencing
    rotation matrices carries an ``O(dt^2)`` error growing with the cube of the rate. The
    first and last frame have no two-sided neighbour and are returned as NaN.

    The relative quaternion is taken in the positive-scalar hemisphere, which selects the
    shorter arc and makes the result independent of the caller's sign convention.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    frames = orientation_wxyz.shape[0]
    rates = np.full((frames, 3), np.nan)
    if frames < 3:
        return rates
    relative = multiply(conjugate(orientation_wxyz[:-2]), orientation_wxyz[2:])
    relative = np.where(relative[:, :1] < 0.0, -relative, relative)
    # log_unit returns the half-angle vector, so twice it is the rotation vector.
    rates[1:-1] = 2.0 * log_unit(relative)[:, 1:] / (2.0 * dt_s)
    return rates


# --- conversions ---------------------------------------------------------------------------------


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """``(..., 3)`` rotation vectors to ``(..., 3, 3)`` matrices."""
    vectors = np.asarray(axis_angle, dtype=np.float64)
    return Rotation.from_rotvec(vectors.reshape(-1, 3)).as_matrix().reshape(vectors.shape[:-1] + (3, 3))


def matrix_to_axis_angle(matrices: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` matrices to ``(..., 3)`` rotation vectors."""
    matrices = np.asarray(matrices, dtype=np.float64)
    return Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(matrices.shape[:-2] + (3,))


def matrix_to_quaternion(matrices: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` matrices to ``(..., 4)`` unit quaternions in ``(w, x, y, z)`` order.

    The sign is whatever scipy returns; apply :func:`canonicalise_sign` to a series.
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    xyzw = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(matrices.shape[:-2] + (4,))
    return np.concatenate((xyzw[..., 3:4], xyzw[..., :3]), axis=-1)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """``(..., 4)`` quaternions in ``(w, x, y, z)`` order to ``(..., 3, 3)`` matrices; normalises first."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def mean_rotation(matrices: np.ndarray) -> np.ndarray:
    """Chordal-L2 mean of a stack ``(n, 3, 3)`` of rotation matrices, as one ``(3, 3)``."""
    return Rotation.from_matrix(np.asarray(matrices, dtype=np.float64)).mean().as_matrix()


# --- Kabsch --------------------------------------------------------------------------------------


def proper_rotation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Nearest proper rotation to a cross-covariance ``H = local.T @ world``.

    Returns ``V diag(1, 1, d) U^T`` for ``H = U S V^T``, with ``d`` the sign that keeps the
    determinant positive. In the column-vector convention that is the ``R`` satisfying
    ``world = R @ local``; a caller working in row-vector form takes its transpose.

    Accepts one ``(3, 3)`` matrix or a batch ``(n, 3, 3)`` and returns the same shape. The
    determinant correction keeps a noisy or near-degenerate constellation from producing a
    reflection, which would silently mirror everything downstream.
    """
    u, _, vt = np.linalg.svd(covariance)
    v = np.swapaxes(vt, -1, -2)
    ut = np.swapaxes(u, -1, -2)
    determinant = np.linalg.det(v @ ut)
    sign = np.where(determinant < 0.0, -1.0, 1.0)
    corrected_v = v.copy()
    corrected_v[..., :, 2] *= sign[..., None]
    return corrected_v @ ut


def kabsch_rotation(local: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Rotation taking centred ``local`` points ``(n, 3)`` onto centred ``world``, guaranteed proper.

    The caller centres both sets; nothing here subtracts a centroid.
    """
    return proper_rotation_from_covariance(local.T @ world)
