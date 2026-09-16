"""Turn a profile's bindings plus a format reader's tables into a source-kind dataclass.

Only the ``smpl_parameters`` kind is bound here for now: its tables are arrays by key (npz,
pickle, json) and need no forward kinematics. The other kinds' bindings arrive with their
skeleton models and readers.

A *table* is whatever the reader returned: a mapping of arrays, possibly nested. A field
reference in a profile is either a top-level key or a list of keys walking into the nesting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from smpl18.sources.base import SmplParameters, Subject, Trial

__all__ = [
    "BindingError",
    "MissingField",
    "UnresolvedGender",
    "bind_parameters",
    "read_field",
    "resolve_fps",
    "resolve_gender",
]


class BindingError(ValueError):
    pass


class MissingField(BindingError, KeyError):
    def __str__(self) -> str:  # KeyError would quote the message
        return self.args[0]


class UnresolvedGender(BindingError):
    pass


def _path(field: str | Sequence[str]) -> tuple[str, ...]:
    return (field,) if isinstance(field, str) else tuple(field)


def has_field(tables: Any, field: str | Sequence[str]) -> bool:
    current = tables
    for key in _path(field):
        try:
            if key not in current:
                return False
            current = current[key]
        except TypeError:
            return False
    return True


def read_field(tables: Any, field: str | Sequence[str], *, what: str = "field") -> Any:
    """Walk a field path into nested tables; a missing key names the path in the error."""
    current = tables
    walked: list[str] = []
    for key in _path(field):
        walked.append(key)
        try:
            present = key in current
        except TypeError:
            present = False
        if not present:
            raise MissingField(f"{what}: no {'/'.join(walked)!r} in the source tables")
        current = current[key]
    return current


def _as_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.ravel()[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return str(value).strip().lower()


def resolve_gender(binding: Mapping[str, Any], tables: Any) -> tuple[str, str]:
    """``(gender, provenance)`` per the profile's gender binding.

    ``constant`` declares it; ``field`` reads it and passes it through ``map`` (source value,
    stripped and lower-cased, to female/male/neutral); ``default`` catches values the map does
    not name and ``when_absent`` a field that is not there. Anything else is an
    ``UnresolvedGender``.
    """
    if "constant" in binding:
        return str(binding["constant"]), "constant"
    if not has_field(tables, binding["field"]):
        if "when_absent" in binding:
            return str(binding["when_absent"]), "when_absent"
        raise UnresolvedGender(
            f"gender field {binding['field']!r} is absent and the binding says nothing about "
            "a missing field (when_absent)"
        )
    raw = _as_text(read_field(tables, binding["field"], what="gender"))
    mapping = {str(k).strip().lower(): v for k, v in binding.get("map", {}).items()}
    if raw in mapping:
        gender = mapping[raw]
    elif "default" in binding:
        return str(binding["default"]), "default"
    else:
        raise UnresolvedGender(
            f"gender value {raw!r} is not in the profile's map {sorted(mapping)} and the "
            "binding has no default"
        )
    if "cross_check" in binding and has_field(tables, binding["cross_check"]):
        other = _as_text(read_field(tables, binding["cross_check"], what="gender cross_check"))
        if other[:1] != raw[:1]:
            raise UnresolvedGender(
                f"gender {raw!r} disagrees with {binding['cross_check']!r} = {other!r}"
            )
    return str(gender), "field"


def resolve_fps(
    binding: Mapping[str, Any], tables: Any, groups: Mapping[str, str] | None = None
) -> tuple[float, str]:
    """``(fps, provenance)``: the field, one of its aliases, or the fallback keyed by a group."""
    if "constant" in binding:
        return float(binding["constant"]), "constant"
    for candidate in (binding["field"], *binding.get("aliases", ())):
        if has_field(tables, candidate):
            value = float(np.asarray(read_field(tables, candidate, what="fps")).ravel()[0])
            if value <= 0:
                raise BindingError(f"fps field {candidate!r} holds {value}, not a rate")
            return value, "field"
    if "fallback" in binding:
        key_name = binding["fallback_key"]
        groups = groups or {}
        if key_name not in groups:
            raise BindingError(
                f"fps fallback is keyed by {{{key_name}}}, which the layout did not match"
            )
        group = groups[key_name]
        if group in binding["fallback"]:
            return float(binding["fallback"][group]), "fallback"
        raise BindingError(
            f"no fps in the source and no fallback for {key_name} {group!r}; the profile "
            f"lists {sorted(binding['fallback'])}"
        )
    raise MissingField(f"fps: no {binding['field']!r} in the source tables and no fallback")


def _poses(binding: Mapping[str, Any], tables: Any) -> np.ndarray:
    poses = np.asarray(read_field(tables, binding["field"], what="poses"), dtype=np.float64)
    layout = binding.get("layout", "T,J,3")
    if layout == "T,J*3":
        if poses.ndim != 2 or poses.shape[1] % 3:
            raise BindingError(f"poses laid out as T,J*3 must be [T, 3J], got {poses.shape}")
        return poses.reshape(poses.shape[0], -1, 3)
    if poses.ndim != 3 or poses.shape[2] != 3:
        raise BindingError(f"poses laid out as T,J,3 must be [T, J, 3], got {poses.shape}")
    return poses


def _betas(binding: Mapping[str, Any], tables: Any) -> np.ndarray:
    betas = np.asarray(read_field(tables, binding["field"], what="betas"), dtype=np.float64)
    if binding.get("frame", "all") == "first":
        if betas.ndim != 2:
            raise BindingError(f"betas with frame=first must be [T, B], got {betas.shape}")
        return betas[0]
    return betas.ravel()


def _optional_scalar(bindings: Mapping[str, Any], name: str, tables: Any) -> float | None:
    if name not in bindings or not has_field(tables, bindings[name]["field"]):
        return None
    value = float(np.asarray(read_field(tables, bindings[name]["field"], what=name)).ravel()[0])
    return value if np.isfinite(value) and value > 0 else None


def bind_parameters(
    bindings: Mapping[str, Any],
    tables: Any,
    *,
    up_axis: str,
    trial: Trial,
    subject_tables: Any = None,
) -> SmplParameters:
    """Build ``SmplParameters`` for one trial from a profile's ``bindings`` and the tables.

    ``subject_tables`` is where subject-level fields (gender, stature, mass) are read when they
    live in another file than the trial's; it defaults to the trial's own tables. Frames with a
    non-finite pose or translation are marked invalid, not repaired.
    """
    subject_tables = tables if subject_tables is None else subject_tables
    poses = _poses(bindings["poses"], tables)
    betas = _betas(bindings["betas"], tables)
    trans = np.asarray(read_field(tables, bindings["trans"]["field"], what="trans"),
                       dtype=np.float64)
    fps, fps_provenance = resolve_fps(bindings["fps"], tables, trial.groups)
    gender, gender_provenance = resolve_gender(bindings["gender"], subject_tables)
    frame_valid = (
        np.isfinite(poses).all(axis=(1, 2)) & np.isfinite(trans).all(axis=1)
        if trans.ndim == 2 and trans.shape[0] == poses.shape[0]
        else None
    )
    subject = Subject(
        id=trial.subject,
        gender=gender,
        stature_m=_optional_scalar(bindings, "stature_m", subject_tables),
        mass_kg=_optional_scalar(bindings, "mass_kg", subject_tables),
        gender_provenance=gender_provenance,
    )
    return SmplParameters(
        subject=subject, trial=trial, poses=poses, betas=betas, trans=trans, fps=fps,
        up_axis=up_axis, frame_valid=frame_valid, fps_provenance=fps_provenance,
    )
