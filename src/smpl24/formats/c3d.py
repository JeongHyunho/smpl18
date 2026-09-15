"""Read a ``.c3d`` marker file into a ``TrcTable`` through ``ezc3d``, in the file's own units.

``ezc3d`` is imported inside ``read`` so that the rest of the package imports without it.
"""

from __future__ import annotations

import os

import numpy as np

from .errors import FormatError
from .tables import TrcTable

__all__ = ["read"]


def _parameter(block: dict, name: str) -> list:
    entry = block.get(name)
    return list(entry["value"]) if entry is not None else []


def read(path: str | os.PathLike[str]) -> TrcTable:
    """Labels and units from the ``POINT`` parameters, points ``(frames, markers, 3)``.

    A sample whose residual is negative (the C3D convention for an invalid point) or whose
    coordinates are not finite is invalid and NaN. ``times_s`` counts from the first stored
    frame at the point rate; ``frame_numbers`` are as ``ezc3d`` reports the header's first frame.
    """
    try:
        import ezc3d
    except ImportError as exc:
        raise ImportError(
            "reading .c3d files needs the ezc3d package (pip install ezc3d)"
        ) from exc

    path = os.fspath(path)
    container = ezc3d.c3d(path)
    point = container["parameters"]["POINT"]
    labels = tuple(str(label) for label in _parameter(point, "LABELS"))
    rate_values = _parameter(point, "RATE")
    if not rate_values:
        raise FormatError(f"{os.path.basename(path)}: POINT:RATE is not declared")
    rate = float(rate_values[0])
    units_values = _parameter(point, "UNITS")
    units = str(units_values[0]) if units_values else ""

    raw = np.asarray(container["data"]["points"], dtype=np.float64)  # (4, markers, frames)
    if raw.ndim != 3 or raw.shape[0] < 3:
        raise FormatError(f"{os.path.basename(path)}: points block is shaped {raw.shape}")
    frames, markers = raw.shape[2], raw.shape[1]
    if markers != len(labels):
        raise FormatError(
            f"{os.path.basename(path)}: {markers} point columns against {len(labels)} labels"
        )
    positions = np.ascontiguousarray(np.transpose(raw[:3], (2, 1, 0)))  # (frames, markers, 3)

    valid = np.isfinite(positions).all(axis=2)
    residuals = np.asarray(container["data"]["meta_points"]["residuals"], dtype=np.float64)
    if residuals.size:
        valid &= residuals[0].T >= 0.0
    positions[~valid] = np.nan

    header = container["header"]["points"]
    first_frame = int(header["first_frame"])
    frame_numbers = first_frame + np.arange(frames, dtype=np.int64)
    times = np.arange(frames, dtype=np.float64) / rate
    return TrcTable(
        labels=labels,
        data_rate_hz=rate,
        units=units,
        positions=positions,
        valid=valid,
        frame_numbers=frame_numbers,
        times_s=times,
        metadata={
            "first_frame": first_frame,
            "last_frame": int(header["last_frame"]),
            "descriptions": tuple(str(d) for d in _parameter(point, "DESCRIPTIONS")),
        },
    )
