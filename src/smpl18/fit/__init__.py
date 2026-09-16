"""Fitting SMPL-24 to what a source observed: targets, the subject's shape, the per-frame pose.

``targets`` is the common form every source kind but SMPL parameters is brought to; ``shape``
fits the betas; ``pose`` solves rotations and translation per frame; ``correspondence`` reads
the tables that say which source joint or centre feeds which SMPL joint.
"""

from .pose import (
    PoseFit,
    PoseSettings,
    TargetError,
    calibrate_orientations,
    initial_pose,
    solve_pose,
)
from .shape import ShapeFit, ShapeSettings, fit_shape, measured_lengths, rigid_pairs, shape_basis
from .targets import OrientationTargets, PositionTargets, Targets, concatenate, provenance_for

__all__ = [
    "OrientationTargets",
    "PoseFit",
    "PoseSettings",
    "PositionTargets",
    "ShapeFit",
    "ShapeSettings",
    "TargetError",
    "Targets",
    "calibrate_orientations",
    "concatenate",
    "fit_shape",
    "initial_pose",
    "measured_lengths",
    "provenance_for",
    "rigid_pairs",
    "shape_basis",
    "solve_pose",
]
