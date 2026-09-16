"""Change of world frame: one fixed rotation, applied to the root rotation and translation only.

Data recorded Z-up and a model defined Y-up differ by a constant rotation ``C``. Per
``docs/primer.md`` section 3.4, ``C`` acts on the root rotation and the translation
(``R_0' = C R_0``, ``t' = C (j_0 + t) - j_0``) and touches no other joint; every other
frame-bound vector, gravity included, turns the same way.

``C`` is the smallest rotation carrying the source's up direction onto the target's, which
introduces no spin about the vertical and so is reproducible from the axes alone. It can be
built from a pair of up-axis names or from the gravity vector a source declares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .definition import NUM_JOINTS
from .rotations import axis_angle_to_matrix, matrix_to_axis_angle

__all__ = [
    "UP_AXES",
    "FrameChange",
    "frame_change",
    "rotation_between",
    "rotation_from_gravity",
    "transform_for_gravity",
    "up_axis_rotation",
]

UP_AXES: tuple[str, ...] = ("x", "y", "z")
_UNIT = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _unit_up(axis: str) -> np.ndarray:
    if axis not in _UNIT:
        raise ValueError(f"up axis must be one of {UP_AXES}, got {axis!r}")
    return np.array(_UNIT[axis], dtype=np.float64)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The smallest rotation carrying direction ``source`` onto direction ``target``.

    Smallest, rather than any: it adds no spin about the target. Opposite directions turn a
    half circle about the basis axis least aligned with them (x for a vertical pair).
    """
    source = np.asarray(source, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    for name, vector in (("source", source), ("target", target)):
        if float(np.linalg.norm(vector)) == 0.0:
            raise ValueError(f"{name} direction has zero length; cannot orient the frame")
    unit = source / float(np.linalg.norm(source))
    aim = target / float(np.linalg.norm(target))

    axis = np.cross(unit, aim)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(unit, aim))
    if sine == 0.0:
        if cosine > 0.0:
            return np.eye(3)
        flip = np.eye(3)[int(np.argmin(np.abs(unit)))]
        perpendicular = flip - unit * float(np.dot(flip, unit))
        cross = _skew(perpendicular / float(np.linalg.norm(perpendicular)))
        return np.eye(3) + 2.0 * cross @ cross

    cross = _skew(axis)
    return np.eye(3) + cross + cross @ cross * ((1.0 - cosine) / sine**2)


def rotation_from_gravity(gravity, *, target_up: str) -> np.ndarray:
    """The smallest rotation carrying the declared ``gravity`` onto minus the target up axis."""
    return rotation_between(gravity, -_unit_up(target_up))


def up_axis_rotation(source_up: str, target_up: str) -> np.ndarray:
    """The rotation from a ``source_up``-up frame into a ``target_up``-up frame; identity when equal."""
    return rotation_between(_unit_up(source_up), _unit_up(target_up))


def _quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2.0
        components = [0.0, 0.0, 0.0]
        components[i] = 0.25 * scale
        components[j] = (rotation[j, i] + rotation[i, j]) / scale
        components[k] = (rotation[k, i] + rotation[i, k]) / scale
        w = (rotation[k, j] - rotation[j, k]) / scale
        x, y, z = components
    if w < 0.0:                       # one hemisphere, so the recorded constant is unique
        w, x, y, z = -w, -x, -y, -z
    return (float(w), float(x), float(y), float(z))


@dataclass(frozen=True)
class FrameChange:
    """A fixed rotation from a source world frame into a target frame, with where it came from."""

    rotation: np.ndarray
    target_up: str
    source_up: str | None = None
    source_gravity: tuple[float, float, float] | None = None

    @property
    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        return _quaternion_wxyz(self.rotation)

    def apply(self, vectors) -> np.ndarray:
        """Rotate any array of 3-vectors, whatever its leading shape."""
        array = np.asarray(vectors, dtype=np.float64)
        if array.shape[-1] != 3:
            raise ValueError(f"expected trailing axis of 3, got {array.shape}")
        return array @ self.rotation.T

    def invert(self, vectors) -> np.ndarray:
        """Undo :meth:`apply`."""
        array = np.asarray(vectors, dtype=np.float64)
        if array.shape[-1] != 3:
            raise ValueError(f"expected trailing axis of 3, got {array.shape}")
        return array @ self.rotation

    def apply_to_pose(self, poses, trans, *, pelvis_rest) -> tuple[np.ndarray, np.ndarray]:
        """Express a pose in the target frame: ``R_0' = C R_0``, joints 1..23 unchanged, and
        ``t' = C t + (C - I) j_0`` with ``j_0 = pelvis_rest``, the pelvis of the rest skeleton.

        The root rotation turns the body about the pelvis, which sits at ``j_0``, not at the
        model origin that ``trans`` moves; the second term is what keeps the world pelvis at
        ``C (j_0 + t)``. It vanishes only when ``j_0`` is the origin (``docs/primer.md``
        section 3.4). Forward kinematics of the result equals ``C`` applied to forward
        kinematics of the input, positions and rotations alike.

        ``poses`` is ``(..., 24, 3)`` axis-angle or ``(..., 24, 3, 3)`` matrices, one frame or
        a batch; ``trans`` is ``(..., 3)``. The returned pose has the kind it was given.
        """
        poses = np.asarray(poses, dtype=np.float64)
        pelvis = np.asarray(pelvis_rest, dtype=np.float64)
        if pelvis.shape != (3,):
            raise ValueError(f"expected a (3,) rest pelvis, got {pelvis.shape}")
        out = poses.copy()
        if poses.shape[-2:] == (NUM_JOINTS, 3):
            out[..., 0, :] = matrix_to_axis_angle(self.rotation @ axis_angle_to_matrix(poses[..., 0, :]))
        elif poses.shape[-3:] == (NUM_JOINTS, 3, 3):
            out[..., 0, :, :] = self.rotation @ poses[..., 0, :, :]
        else:
            raise ValueError(f"expected (..., 24, 3) axis-angle or (..., 24, 3, 3) matrices, got {poses.shape}")
        trans = np.asarray(trans, dtype=np.float64)
        if trans.shape[-1] != 3:
            raise ValueError(f"expected trailing axis of 3, got {trans.shape}")
        return out, self.apply(trans + pelvis) - pelvis

    def record(self) -> dict:
        """The constant and its provenance, as plain values for a manifest."""
        return {
            "rotation": self.rotation.tolist(),
            "quaternion_wxyz": list(self.quaternion_wxyz),
            "source_up": self.source_up,
            "target_up": self.target_up,
            "source_gravity": None if self.source_gravity is None else list(self.source_gravity),
        }


def frame_change(source_up: str, target_up: str) -> FrameChange:
    """The frame change between two named up axes."""
    return FrameChange(
        rotation=up_axis_rotation(source_up, target_up), target_up=target_up, source_up=source_up
    )


def transform_for_gravity(gravity, *, target_up: str) -> FrameChange:
    """The frame change that carries a source's declared gravity onto minus the target up axis."""
    source = np.asarray(gravity, dtype=np.float64).reshape(3)
    _unit_up(target_up)
    return FrameChange(
        rotation=rotation_from_gravity(source, target_up=target_up),
        target_up=target_up,
        source_gravity=(float(source[0]), float(source[1]), float(source[2])),
    )
