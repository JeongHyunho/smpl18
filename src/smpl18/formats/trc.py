"""Read an OpenSim-format ``.trc`` marker file into a ``TrcTable``, in the file's own units, and
write one back.

The layout is five header lines -- file type, the metadata names, the metadata values, the
marker names, the axis labels -- and some writers add a blank line after the axis row. The
first data row is found rather than assumed: an earlier reader started at a fixed line and so
dropped frame 0 of every trial, silently, because the remaining frames were all valid. The
header's own ``NumFrames`` is the check that would have caught it, so it is asserted.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np

from .errors import FormatError
from .tables import TrcTable

__all__ = ["read", "write"]

_HEADER_SEARCH_LINES = 8
_FIRST_MARKER_COLUMN = 2  # after Frame# and Time


def _first_data_line(lines: list[str], name: str) -> int:
    """Index of the first sample row.

    The axis row is the landmark because it is the only header line whose first two fields are
    empty while later fields are filled; the marker-name row above it names ``Frame#`` and
    ``Time``.
    """
    for index in range(min(len(lines), _HEADER_SEARCH_LINES)):
        fields = lines[index].split("\t")
        if len(fields) >= 3 and not fields[0].strip() and not fields[1].strip():
            start = index + 1
            while start < len(lines) and not lines[start].strip():
                start += 1
            return start
    raise FormatError(
        f"{name}: no axis row found in the first {_HEADER_SEARCH_LINES} lines; "
        "not an OpenSim .trc"
    )


def _metadata(lines: list[str], name: str) -> dict[str, str]:
    """The metadata names row zipped with the values row, as text."""
    if len(lines) < 4:
        raise FormatError(f"{name}: fewer than four header lines; not an OpenSim .trc")
    keys = [value.strip() for value in lines[1].split("\t")]
    values = [value.strip() for value in lines[2].split("\t")]
    return {key: values[slot] if slot < len(values) else "" for slot, key in enumerate(keys) if key}


def _declared_frame_count(metadata: dict[str, str]) -> int | None:
    """``NumFrames`` from the file's own metadata row, or ``None`` if it does not say."""
    value = metadata.get("NumFrames")
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _required_float(metadata: dict[str, str], key: str, name: str) -> float:
    value = metadata.get(key, "")
    try:
        return float(value)
    except ValueError as exc:
        raise FormatError(f"{name} does not declare a numeric {key}") from exc


def _optional_float(metadata: dict[str, str], key: str) -> float | None:
    """A numeric header field, or None when the file leaves it blank or non-numeric."""
    try:
        return float(metadata.get(key, ""))
    except ValueError:
        return None


def _sample(fields: list[str], start: int) -> tuple[float, float, float] | None:
    """Three numbers from ``fields[start:start + 3]``, or None when any is blank or not a number."""
    out = []
    for slot in range(start, start + 3):
        text = fields[slot].strip() if slot < len(fields) else ""
        if not text:
            return None
        try:
            out.append(float(text))
        except ValueError:
            return None
    return (out[0], out[1], out[2])


def read(path: str | os.PathLike[str]) -> TrcTable:
    """Labels, rates, units and every marker sample of one ``.trc`` file, as written."""
    path = pathlib.Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata = _metadata(lines, path.name)
    name_row = lines[3].split("\t")
    starts = [
        (value.strip(), index)
        for index, value in enumerate(name_row)
        if index >= _FIRST_MARKER_COLUMN and value.strip()
    ]
    labels = tuple(label for label, _ in starts)
    first = _first_data_line(lines, path.name)
    rows = [line.split("\t") for line in lines[first:] if line.strip()]
    declared = _declared_frame_count(metadata)
    if declared is not None and declared != len(rows):
        raise FormatError(
            f"{path.name} declares NumFrames {declared} and carries {len(rows)} data rows"
        )

    positions = np.full((len(rows), len(labels), 3), np.nan, dtype=np.float64)
    valid = np.zeros((len(rows), len(labels)), dtype=bool)
    frame_numbers = np.empty(len(rows), dtype=np.int64)
    times = np.empty(len(rows), dtype=np.float64)
    for frame, row in enumerate(rows):
        try:
            frame_numbers[frame] = int(float(row[0]))
            times[frame] = float(row[1])
        except (IndexError, ValueError) as exc:
            raise FormatError(
                f"{path.name}: data row {frame} has no frame number and time"
            ) from exc
        for slot, (_, start) in enumerate(starts):
            sample = _sample(row, start)
            if sample is not None:
                positions[frame, slot] = sample
                valid[frame, slot] = True
        if not np.isfinite(positions[frame]).all():
            valid[frame] &= np.isfinite(positions[frame]).all(axis=1)
            positions[frame][~valid[frame]] = np.nan

    units = metadata.get("Units", "")
    if not units:
        raise FormatError(f"{path.name} does not declare Units")
    return TrcTable(
        labels=labels,
        data_rate_hz=_required_float(metadata, "DataRate", path.name),
        units=units,
        positions=positions,
        valid=valid,
        frame_numbers=frame_numbers,
        times_s=times,
        camera_rate_hz=_optional_float(metadata, "CameraRate"),
        metadata=dict(metadata),
    )


def write(
    path: str | os.PathLike[str],
    labels,
    positions: np.ndarray,
    *,
    rate_hz: float,
    units: str,
    valid: np.ndarray | None = None,
) -> pathlib.Path:
    """Write ``(frames, markers, 3)`` positions, already in ``units``, as an OpenSim ``.trc``.

    An invalid or non-finite sample is written as empty fields, which :func:`read` reads back as
    invalid. Frames are numbered from 1 and timed from 0 at ``rate_hz``.
    """
    path = pathlib.Path(path)
    labels = [str(label) for label in labels]
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (len(labels), 3):
        raise ValueError(f"positions must be (frames, {len(labels)}, 3), got {positions.shape}")
    frames = positions.shape[0]
    mask = np.isfinite(positions).all(axis=2)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    tab = "\t"
    rate = f"{float(rate_hz):g}"
    lines = [
        tab.join(["PathFileType", "4", "(X/Y/Z)", path.name]),
        tab.join(["DataRate", "CameraRate", "NumFrames", "NumMarkers", "Units", "OrigDataRate",
                  "OrigDataStartFrame", "OrigNumFrames"]),
        tab.join([rate, rate, str(frames), str(len(labels)), units, rate, "1", str(frames)]),
        tab.join(["Frame#", "Time"] + [tab.join([label, "", ""]) for label in labels]),
        tab.join(["", ""] + [tab.join([f"X{i}", f"Y{i}", f"Z{i}"])
                             for i in range(1, len(labels) + 1)]),
        "",
    ]
    for frame in range(frames):
        fields = [str(frame + 1), f"{frame / float(rate_hz):.6f}"]
        for marker in range(len(labels)):
            if mask[frame, marker]:
                fields.extend(f"{value:.6f}" for value in positions[frame, marker])
            else:
                fields.extend(["", "", ""])
        lines.append(tab.join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
