"""Choosing the four constants: the least-squares fit that decides what the reduction costs.

Because orientation survives any constants, the fit can spend all twelve of its parameters on
position: it minimises the joint-centre displacement of the joints a frozen joint can move
(``AFFECTED_BY_FREEZE``), over frames spread evenly across what the caller passes. The starting
guess is each frozen joint's mean rotation, which is the exact answer when the joint does not
move and an honest baseline when it does -- so both residuals are reported, and the corpus can
say what the optimisation bought instead of asserting that it helped (``docs/primer.md`` 5.2).

The fit is per subject, not per trial: the caller pools the subject's frames, since the constants
describe that body's posture in the frozen joints and must agree across its trials.

Numbers come from the settings mapping, never from a default in a signature.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from smpl18.skeleton.definition import AFFECTED_BY_FREEZE, FROZEN_JOINT_NAMES, FROZEN_JOINTS
from smpl18.skeleton.kinematics import fk_batch
from smpl18.skeleton.rotations import axis_angle_to_matrix, matrix_to_axis_angle, mean_rotation

from .fixation import apply, as_local_rotations
from .residual import PositionError, residual

__all__ = [
    "SETTINGS_SECTION",
    "FitSettings",
    "FittedConstants",
    "fit_constants",
    "initial_constants",
    "sample_indices",
]

#: Where in a settings mapping the three numbers below live.
SETTINGS_SECTION = "reduce"
_SETTING_NAMES = ("sample_frames", "optimiser", "max_evaluations")

_AFFECTED = np.array(AFFECTED_BY_FREEZE, dtype=np.int64)
_PARAMETERS = (len(FROZEN_JOINTS), 3)


@dataclass(frozen=True)
class FitSettings:
    """The three numbers :func:`fit_constants` needs, read from one settings mapping."""

    #: Most frames the fit measures; a longer input is sampled evenly down to this.
    sample_frames: int
    #: The ``scipy.optimize.least_squares`` method, e.g. ``trf``.
    optimiser: str
    #: Cap on residual evaluations, passed as ``least_squares(max_nfev=...)``.
    max_evaluations: int

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> FitSettings:
        """Read the ``reduce`` section, naming a missing key instead of choosing a number for it."""
        section = settings.get(SETTINGS_SECTION) if isinstance(settings, Mapping) else None
        if not isinstance(section, Mapping):
            raise ValueError(
                f"settings has no {SETTINGS_SECTION!r} section; the fit needs "
                + ", ".join(f"{SETTINGS_SECTION}.{name}" for name in _SETTING_NAMES)
            )
        for name in _SETTING_NAMES:
            if name not in section:
                raise ValueError(f"settings is missing {SETTINGS_SECTION}.{name}")
        sample_frames = int(section["sample_frames"])
        max_evaluations = int(section["max_evaluations"])
        for name, value in (("sample_frames", sample_frames), ("max_evaluations", max_evaluations)):
            if value < 1:
                raise ValueError(f"{SETTINGS_SECTION}.{name} must be at least 1, got {value}")
        return cls(
            sample_frames=sample_frames,
            optimiser=str(section["optimiser"]),
            max_evaluations=max_evaluations,
        )


@dataclass(frozen=True)
class FittedConstants:
    """The four constants and what they cost, in the plain types the corpus records."""

    #: ``(4, 3, 3)`` rotation matrices, in the order of ``FROZEN_JOINTS``.
    constants: np.ndarray
    joint_names: tuple[str, ...]
    #: The mean-rotation guess and the fitted answer, both measured on the frames below.
    initial: PositionError
    fitted: PositionError
    #: How many frames were measured, which is the sample, not the input length.
    frames: int
    success: bool
    message: str

    @property
    def rotation_vectors(self) -> np.ndarray:
        """The constants as four rotation vectors: the twelve numbers actually solved for."""
        return matrix_to_axis_angle(self.constants)


def initial_constants(local_rotations) -> np.ndarray:
    """Each frozen joint's mean rotation over the frames, ``(4, 3, 3)``.

    Exact when the joint is already still, and what the fitted residual is reported against.
    """
    local = as_local_rotations(local_rotations)
    return np.stack([mean_rotation(local[:, joint]) for joint in FROZEN_JOINTS])


def sample_indices(frames: int, most: int) -> np.ndarray:
    """Frame indices spread evenly over the input, so a long take costs no more than a short one."""
    if frames < 1:
        raise ValueError("cannot fit the constants without frames")
    if frames <= most:
        return np.arange(frames, dtype=np.int64)
    return np.linspace(0, frames - 1, most).round().astype(np.int64)


def fit_constants(local_rotations, rest_joints, *, settings: Mapping[str, Any]) -> FittedConstants:
    """Fit the four constants that displace the affected joint centres least.

    ``local_rotations`` is ``(T, 24, 3, 3)`` and ``rest_joints`` the subject's ``(24, 3)`` rest
    skeleton. Twelve parameters -- four rotation vectors -- against the flattened position
    difference of the affected joints on the sampled frames, solved by
    ``scipy.optimize.least_squares``. Sum of squares over the flattened difference is the sum of
    squares of the distances, so minimising it minimises the RMS the result reports.
    """
    options = FitSettings.from_settings(settings)
    local = as_local_rotations(local_rotations)
    rest = np.asarray(rest_joints, dtype=np.float64)
    sampled = local[sample_indices(local.shape[0], options.sample_frames)]
    trans = np.zeros((sampled.shape[0], 3), dtype=np.float64)

    original, _ = fk_batch(rest, sampled, trans)
    target = original[:, _AFFECTED]

    def difference(parameters: np.ndarray) -> np.ndarray:
        constants = axis_angle_to_matrix(parameters.reshape(_PARAMETERS))
        positions, _ = fk_batch(rest, apply(sampled, constants), trans)
        return (positions[:, _AFFECTED] - target).ravel()

    start = initial_constants(sampled)
    solution = least_squares(
        difference,
        matrix_to_axis_angle(start).ravel(),
        method=options.optimiser,
        max_nfev=options.max_evaluations,
    )
    fitted = axis_angle_to_matrix(solution.x.reshape(_PARAMETERS))
    return FittedConstants(
        constants=fitted,
        joint_names=FROZEN_JOINT_NAMES,
        initial=residual(sampled, rest, start),
        fitted=residual(sampled, rest, fitted),
        frames=int(sampled.shape[0]),
        success=bool(solution.success),
        message=str(solution.message),
    )
