"""Articulated-skeleton models: bodies in a tree, joints between them, forward kinematics.

A skeleton-motion source is a model plus coordinate values per frame. The model is evaluated
from the file's own description, never from a table of known skeletons:

    opensim  an OpenSim model (custom joints, spline-driven and coupled coordinates), and the
             coordinates of a ``.mot`` / ``.sto`` table in radians and metres
    bvh      a Biovision hierarchy (offsets and Euler channels) at a caller-stated length scale,
             and the coordinates of its motion rows

Each module also has ``read``, which builds the model straight from a file.
"""

from . import bvh, opensim
from .bvh import BvhSkeleton, coordinates_from_frames
from .opensim import OpenSimSkeleton, UnsupportedModelError, coordinates_from_mot

__all__ = [
    "BvhSkeleton",
    "OpenSimSkeleton",
    "UnsupportedModelError",
    "bvh",
    "coordinates_from_frames",
    "coordinates_from_mot",
    "opensim",
]
