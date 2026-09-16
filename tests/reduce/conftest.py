"""A rest skeleton and pose builders, all in code: the reduction needs no file to be exercised.

The rest skeleton is crude -- spine and neck up +Y, legs down -Y, collars out to the sides, arms
along +/-X -- but every offset the reduction depends on is non-zero (spine1 -> spine2 -> spine3
and collar -> shoulder), which is what makes a wrong constant actually move the joints below it.
A skeleton whose frozen joints sat on top of their children would pass these tests for the wrong
reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from smpl18.skeleton.definition import FROZEN_JOINTS, NUM_JOINTS, PARENTS
from smpl18.skeleton.rotations import axis_angle_to_matrix

#: Offset of each joint from its parent, in metres. The pelvis is the origin.
OFFSETS: dict[int, tuple[float, float, float]] = {
    1: (0.09, -0.06, 0.0), 2: (-0.09, -0.06, 0.0), 3: (0.0, 0.10, 0.0),
    4: (0.0, -0.40, 0.0), 5: (0.0, -0.40, 0.0), 6: (0.0, 0.11, 0.0),
    7: (0.0, -0.40, 0.0), 8: (0.0, -0.40, 0.0), 9: (0.0, 0.12, 0.0),
    10: (0.0, -0.06, 0.15), 11: (0.0, -0.06, 0.15), 12: (0.0, 0.21, 0.0),
    13: (0.07, 0.16, 0.0), 14: (-0.07, 0.16, 0.0), 15: (0.0, 0.10, 0.0),
    16: (0.11, 0.05, 0.0), 17: (-0.11, 0.05, 0.0), 18: (0.26, 0.0, 0.0),
    19: (-0.26, 0.0, 0.0), 20: (0.25, 0.0, 0.0), 21: (-0.25, 0.0, 0.0),
    22: (0.08, 0.0, 0.0), 23: (-0.08, 0.0, 0.0),
}

#: A settings mapping with the section :func:`smpl18.reduce.fit_constants` reads.
SETTINGS: dict[str, dict[str, object]] = {
    "reduce": {"sample_frames": 40, "optimiser": "trf", "max_evaluations": 200},
}


def rest_skeleton() -> np.ndarray:
    """``(24, 3)`` rest joint centres built from :data:`OFFSETS` down the tree."""
    rest = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
    for joint in range(1, NUM_JOINTS):
        rest[joint] = rest[PARENTS[joint]] + np.array(OFFSETS[joint], dtype=np.float64)
    return rest


def identity_series(frames: int) -> np.ndarray:
    """``(T, 24, 3, 3)`` of identities: the rest pose, held."""
    return np.broadcast_to(np.eye(3), (frames, NUM_JOINTS, 3, 3)).copy()


def wandering_series(rng: np.random.Generator, frames: int) -> np.ndarray:
    """``(T, 24, 3, 3)`` where every joint moves a little, frozen joints included."""
    return axis_angle_to_matrix(rng.normal(0.0, 0.25, (frames, NUM_JOINTS, 3)))


def hold_frozen(local: np.ndarray, constants: np.ndarray) -> np.ndarray:
    """The same series with the four frozen joints held at ``constants`` on every frame."""
    held = local.copy()
    for column, joint in enumerate(FROZEN_JOINTS):
        held[:, joint] = constants[column]
    return held


def swing(frames: int, axis, amplitude_rad: float, phase: float) -> np.ndarray:
    """``(T, 3, 3)``: one joint swinging about ``axis``, the way a trunk or a shoulder girdle does.

    Smooth rather than random, so the mean rotation is the sensible guess it would be on real
    capture and the fit has to earn its improvement.
    """
    angle = amplitude_rad * np.sin(np.linspace(0.0, 2.0 * np.pi, frames) + phase)
    return axis_angle_to_matrix(angle[:, None] * np.asarray(axis, dtype=np.float64))


@pytest.fixture
def rest() -> np.ndarray:
    return rest_skeleton()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def settings() -> dict[str, dict[str, object]]:
    return {section: dict(values) for section, values in SETTINGS.items()}
