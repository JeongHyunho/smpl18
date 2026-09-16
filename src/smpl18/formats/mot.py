"""Read an OpenSim ``.mot`` / ``.sto`` table: header flags, column names, the numeric block.

Values come back exactly as written. Whether a column is degrees or metres is what the header's
``inDegrees`` flag and the column's name say, and reading that is the caller's business; this
module only reports the flag.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import numpy as np

from .errors import FormatError

__all__ = ["MotTable", "read"]


@dataclass(frozen=True)
class MotTable:
    """Column names and the numeric block, plus every header line the file wrote."""

    columns: tuple[str, ...]
    values: np.ndarray
    #: The ``inDegrees`` header flag: True for ``yes``, False for anything else, None when the
    #: file does not declare it (``.sto`` files often do not).
    in_degrees: bool | None
    #: Every ``key=value`` header line before ``endheader``, keyed as written.
    header: dict[str, str]

    @property
    def frame_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def time_s(self) -> np.ndarray:
        """The ``time`` column; the first column when no column is named ``time``."""
        if "time" in self.columns:
            return self.column("time")
        return self.values[:, 0]

    def column(self, name: str) -> np.ndarray:
        """One column by name; a KeyError names the column when the file has none."""
        try:
            return self.values[:, self.columns.index(name)]
        except ValueError as exc:
            raise KeyError(f"no column named {name!r}") from exc


def read(path: str | os.PathLike[str]) -> MotTable:
    """Column names, the numeric block, and the header flags of one ``.mot`` / ``.sto`` file."""
    path = pathlib.Path(path)
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_degrees: bool | None = None
    header: dict[str, str] = {}
    header_end: int | None = None
    for index, line in enumerate(text):
        stripped = line.strip()
        if stripped == "endheader":
            header_end = index
            break
        if stripped.lower().startswith("indegrees"):
            in_degrees = stripped.split("=", 1)[1].strip().lower() == "yes"
        if "=" in stripped:
            key, value = stripped.split("=", 1)
            header[key.strip()] = value.strip()
    if header_end is None:
        raise FormatError(f"no endheader in {path.name}")
    if header_end + 1 >= len(text):
        raise FormatError(f"{path.name}: no column row after endheader")
    columns = tuple(text[header_end + 1].split())
    rows = [line.split() for line in text[header_end + 2:] if line.strip()]
    if rows:
        try:
            values = np.asarray(rows, dtype=np.float64)
        except ValueError as exc:
            raise FormatError(f"{path.name}: a data row is not numeric or not rectangular") from exc
    else:
        values = np.empty((0, len(columns)), dtype=np.float64)
    if values.shape[1] != len(columns):
        raise FormatError(
            f"{path.name}: {values.shape[1]} numeric columns against {len(columns)} names"
        )
    return MotTable(columns=columns, values=values, in_degrees=in_degrees, header=header)
