"""A subject file: who the trials belong to, which body model to use, and what was measured.

Schema ``smpl18_subject_v1``, one YAML file per subject::

    schema: smpl18_subject_v1
    id: S01
    gender: female            # male | female | neutral: selects the body model
    measurements:             # metres; whatever the marker set declares it needs
      marker_radius: 0.007
      leg_length_left: 0.86

A marker set's rules read ``measurements``; the other source kinds need only ``id`` and
``gender``, which the command line can give instead of a file.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from smpl18.model.select import GENDERS

__all__ = ["SCHEMA_ID", "SubjectError", "SubjectInfo"]

SCHEMA_ID = "smpl18_subject_v1"
_KEYS = {"schema", "id", "gender", "measurements", "description"}


class SubjectError(ValueError):
    pass


@dataclass(frozen=True)
class SubjectInfo:
    id: str
    gender: str
    measurements: Mapping[str, float] = field(default_factory=dict)
    #: ``file`` when read from a subject file, ``argument`` when given on the command line.
    source: str = "argument"
    #: The same for the gender alone, which selects the body model.
    gender_from: str = "argument"
    path: Path | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.id or any(c in self.id for c in '/\\:*?"<>|'):
            raise SubjectError(f"subject id {self.id!r} cannot name a directory")
        if self.gender not in GENDERS:
            raise SubjectError(f"gender must be one of {GENDERS}, got {self.gender!r}")
        values = {}
        for name, value in self.measurements.items():
            try:
                values[str(name)] = float(value)
            except (TypeError, ValueError) as error:
                raise SubjectError(f"measurement {name} is not a number: {value!r}") from error
        object.__setattr__(self, "measurements", values)

    @classmethod
    def load(cls, path: str | Path) -> SubjectInfo:
        path = Path(path)
        raw = path.read_bytes()
        data = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(data, Mapping):
            raise SubjectError(f"{path}: expected a mapping")
        unknown = set(data) - _KEYS
        if unknown:
            raise SubjectError(f"{path}: unknown keys {sorted(unknown)}")
        if data.get("schema") != SCHEMA_ID:
            raise SubjectError(f"{path}: schema must be {SCHEMA_ID}")
        for key in ("id", "gender"):
            if key not in data:
                raise SubjectError(f"{path}: missing key {key!r}")
        return cls(
            id=str(data["id"]),
            gender=str(data["gender"]),
            measurements=data.get("measurements") or {},
            source="file",
            gender_from="file",
            path=path,
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    def with_overrides(self, *, id: str | None = None, gender: str | None = None,
                       measurements: Mapping[str, float] | None = None) -> SubjectInfo:
        """A copy with command-line values taking precedence over the file's."""
        merged = dict(self.measurements)
        merged.update(measurements or {})
        return SubjectInfo(
            id=id or self.id,
            gender=gender or self.gender,
            measurements=merged,
            source=self.source if not (id or gender or measurements) else f"{self.source}+argument",
            gender_from="argument" if gender else self.gender_from,
            path=self.path,
            sha256=self.sha256,
        )

    def record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "gender_from": self.gender_from,
            "file": None if self.path is None else self.path.name,
            "sha256": self.sha256,
            "measurements": dict(self.measurements),
        }
