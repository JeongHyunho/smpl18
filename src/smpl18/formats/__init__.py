"""File-format readers. Each returns a table in the file's own names and interprets nothing.

The rule: a reader changes no unit, no frame and no name, maps no value, and defaults nothing.
It reports what the file declares (``inDegrees``, ``Units``, ``up_axis``, a gravity vector, a
biological-sex string) and leaves the meaning to a source kind bound by a profile. A format has
no idea which dataset it serves.

    npz          arrays by key, ``allow_pickle=False``
    pickle_safe  a pickle container through a whitelisting unpickler, no code execution
    jsonfile     the decoded object (named so as not to shadow the standard library)
    mat          MATLAB variables by name; ``struct_to_dict`` for plain containers
    osim         an OpenSim model: bodies, joints, coordinates, markers, gravity, derived orders
    mot          an OpenSim ``.mot`` / ``.sto`` table: columns, values, header flags
    trc          marker trajectories (``TrcTable``) from an OpenSim-format ``.trc``
    c3d          marker trajectories (``TrcTable``) from a ``.c3d``, through ``ezc3d``
    b3d          a ``.b3d`` container: embedded models, passes, subject fields, frames
    bvh          a Biovision Hierarchy file: joint tree, channels, motion rows

Every reader raises ``FormatError`` (or a subclass) when a file contradicts its own header.
"""

from . import b3d, bvh, c3d, jsonfile, mat, mot, npz, osim, pickle_safe, trc
from .errors import FormatError
from .tables import TrcTable

__all__ = [
    "FormatError",
    "TrcTable",
    "b3d",
    "bvh",
    "c3d",
    "jsonfile",
    "mat",
    "mot",
    "npz",
    "osim",
    "pickle_safe",
    "trc",
]
