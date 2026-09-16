"""The 18-joint reduction: four joints frozen to per-subject constants, two hands dropped.

A corpus stores 18 rotations per frame, not 24 (``docs/primer.md`` section 5). ``spine1``,
``spine2``, ``left_collar`` and ``right_collar`` become one constant each per subject, the two
hand joints go, and the kept joint below each frozen one absorbs what was removed. That absorption
is exact, so the reduction costs nothing in world orientation, whatever constants are chosen --
which is why the constants are free to be chosen for something else.

What they are chosen for is position: a frozen joint sits between two segments, so a constant that
differs from the joint's real rotation carries the whole upper body with it. :func:`fit_constants`
minimises that displacement and reports it, fitted and initial, in metres, for the corpus to carry.
"""

from .fit import (
    SETTINGS_SECTION,
    FitSettings,
    FittedConstants,
    fit_constants,
    initial_constants,
    sample_indices,
)
from .fixation import apply, names_18, to_18
from .residual import PositionError, displacement, per_joint_rms, residual

__all__ = [
    "SETTINGS_SECTION",
    "FitSettings",
    "FittedConstants",
    "PositionError",
    "apply",
    "displacement",
    "fit_constants",
    "initial_constants",
    "names_18",
    "per_joint_rms",
    "residual",
    "sample_indices",
    "to_18",
]
