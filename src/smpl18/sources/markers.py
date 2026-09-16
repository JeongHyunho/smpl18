"""Labelled surface markers to SMPL joint centres and segment frames, by a marker-set description.

Markers sit on the skin, not at the joints, so a marker set says how each joint centre follows
from them. The description is a YAML file (schema ``smpl18_markerset_v1``) and nothing about a
particular protocol is written in code; the rules are a small fixed vocabulary:

* ``point``: a marker, or the mean of several (a wrist between its two wrist-bar markers);
* ``offset``: a point expressed in a segment frame built from markers, each coordinate a linear
  combination of measured lengths and subject measurements plus a constant (regression hip
  centres are this);
* ``chord``: the centre ``C`` in the plane of a proximal centre ``P``, a lateral marker ``L`` and a
  plane marker ``W``, with the angle ``P-C-L`` square and ``|C - L|`` a given distance (the knee
  and ankle of the conventional gait model, with the thigh and shank wands);
* ``hinge``: the centre on a hinge axis through a lateral marker, the axis normal to the plane of
  the proximal centre, the centre and a distal centre, found by fixed-point iteration (an elbow,
  whose lateral epicondyle marker lies on the flexion axis). A lateral reference marker on the
  proximal segment picks between the two centres the geometry allows, and the pick must hold
  still on that segment within ``drift_tolerance``; a limb straighter than ``min_flexion_deg``
  leaves the plane undefined. Frames that fail either are left without the centre.

Frames (``frames``) are built from two directions: the primary axis exactly, the secondary made
orthogonal to it, the third completing a right-handed set. A ``segments`` entry turns a frame into
an orientation target for an SMPL joint; the constant between the marker frame and the SMPL
segment frame is calibrated by the pose solve, so a frame only has to move with its segment.

Numbers that belong to the subject (leg length, knee width, marker radius) are *measurements*,
supplied per subject; numbers the markers can show (the distance between the two anterior iliac
spines) are *lengths*, measured as the median over the trial. Nothing here has a default.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smpl18.fit.targets import OrientationTargets, PositionTargets, Targets, provenance_for
from smpl18.formats.tables import TrcTable
from smpl18.skeleton.definition import JOINT_NAMES

__all__ = [
    "SCHEMA_ID",
    "MarkerSet",
    "MarkerSetError",
    "MarkerSetResult",
    "fill_gaps",
    "from_table",
    "length_scale",
]

SCHEMA_ID = "smpl18_markerset_v1"
_AXES = ("x", "y", "z")
_SIDES = ("toward_plane", "away_from_plane")
_CONSTANT = "constant"
#: The hinge centre is a contraction with a factor near (axis offset / segment length), well
#: below one; this many rounds settles it far below a micrometre. An algorithm constant.
_HINGE_ROUNDS = 12
_TINY = 1e-12

#: Metres per unit, for the length units marker files declare.
_UNITS = {"m": 1.0, "meters": 1.0, "metres": 1.0, "cm": 0.01, "mm": 0.001,
          "millimeters": 0.001, "millimetres": 0.001}


class MarkerSetError(ValueError):
    """A marker set is malformed, or the data cannot satisfy it."""


def length_scale(units: str) -> float:
    """Metres per unit for a marker file's declared ``units``; unknown units are refused."""
    key = str(units).strip().lower()
    if key not in _UNITS:
        raise MarkerSetError(f"unknown length unit {units!r}; expected one of {sorted(_UNITS)}")
    return _UNITS[key]


# --- the description ----------------------------------------------------------------------------


def _point(value: Any, where: str):
    """Parse a point expression into ``("marker", label)``, ``("centre", joint)`` or
    ``("mean", [expressions])``."""
    if isinstance(value, str):
        return ("marker", value)
    if isinstance(value, list):
        if not value:
            raise MarkerSetError(f"{where}: an empty list names no point")
        return ("mean", [_point(item, f"{where}[{i}]") for i, item in enumerate(value)])
    if isinstance(value, Mapping) and len(value) == 1:
        (key, item), = value.items()
        if key == "centre":
            if item not in JOINT_NAMES:
                raise MarkerSetError(f"{where}: {item!r} is not an SMPL-24 joint")
            return ("centre", item)
        if key == "mean":
            return _point(list(item), where)
    raise MarkerSetError(
        f"{where}: a point is a marker label, a list of points, {{centre: <joint>}} or "
        f"{{mean: [...]}}; got {value!r}"
    )


def _linear(value: Any, where: str) -> dict[str, float]:
    if isinstance(value, (int, float)):
        return {_CONSTANT: float(value)}
    if not isinstance(value, Mapping) or not value:
        raise MarkerSetError(f"{where}: expected a number or a mapping of term: coefficient")
    try:
        return {str(k): float(v) for k, v in value.items()}
    except (TypeError, ValueError) as error:
        raise MarkerSetError(f"{where}: coefficients must be numbers") from error


def _keys(value: Any, where: str, allowed: set[str], required: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MarkerSetError(f"{where}: expected a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise MarkerSetError(f"{where}: unknown keys {sorted(unknown)}")
    missing = required - set(value)
    if missing:
        raise MarkerSetError(f"{where}: missing keys {sorted(missing)}")
    return value


@dataclass(frozen=True)
class _Axis:
    name: str
    start: Any
    end: Any


@dataclass(frozen=True)
class _Frame:
    origin: Any | None
    primary: _Axis
    secondary: _Axis


def _axis(value: Any, where: str) -> _Axis:
    value = _keys(value, where, {"axis", "from", "to"}, {"axis", "from", "to"})
    if value["axis"] not in _AXES:
        raise MarkerSetError(f"{where}.axis must be one of {_AXES}")
    return _Axis(value["axis"], _point(value["from"], f"{where}.from"),
                 _point(value["to"], f"{where}.to"))


def _frame(value: Any, where: str) -> _Frame:
    value = _keys(value, where, {"origin", "primary", "secondary"}, {"primary", "secondary"})
    primary = _axis(value["primary"], f"{where}.primary")
    secondary = _axis(value["secondary"], f"{where}.secondary")
    if primary.name == secondary.name:
        raise MarkerSetError(f"{where}: primary and secondary name the same axis")
    origin = _point(value["origin"], f"{where}.origin") if "origin" in value else None
    return _Frame(origin, primary, secondary)


_CENTRE_RULES = ("point", "offset", "chord", "hinge")


def _centre(value: Any, where: str, frames: Mapping[str, _Frame]) -> tuple[str, dict, float]:
    value = _keys(value, where, {*_CENTRE_RULES, "weight"}, set())
    rules = [rule for rule in _CENTRE_RULES if rule in value]
    if len(rules) != 1:
        raise MarkerSetError(f"{where}: give exactly one of {_CENTRE_RULES}")
    rule = rules[0]
    weight = float(value.get("weight", 1.0))
    if weight <= 0:
        raise MarkerSetError(f"{where}.weight must be positive")
    body = value[rule]
    here = f"{where}.{rule}"
    if rule == "point":
        return rule, {"point": _point(body, here)}, weight
    if rule == "offset":
        body = _keys(body, here, {"frame", "origin", *_AXES}, {"frame"})
        if body["frame"] not in frames:
            raise MarkerSetError(f"{here}.frame names no frame: {body['frame']!r}")
        origin = body.get("origin")
        if origin is None and frames[body["frame"]].origin is None:
            raise MarkerSetError(f"{here}: the frame has no origin, so the offset needs one")
        return rule, {
            "frame": body["frame"],
            "origin": None if origin is None else _point(origin, f"{here}.origin"),
            "components": [_linear(body.get(axis, 0.0), f"{here}.{axis}") for axis in _AXES],
        }, weight
    if rule == "chord":
        body = _keys(body, here, {"proximal", "lateral", "plane", "distance", "side"},
                     {"proximal", "lateral", "plane", "distance", "side"})
        if body["side"] not in _SIDES:
            raise MarkerSetError(f"{here}.side must be one of {_SIDES}")
        return rule, {
            "proximal": _point(body["proximal"], f"{here}.proximal"),
            "lateral": _point(body["lateral"], f"{here}.lateral"),
            "plane": _point(body["plane"], f"{here}.plane"),
            "distance": _linear(body["distance"], f"{here}.distance"),
            "toward": body["side"] == "toward_plane",
        }, weight
    keys = {"proximal", "distal", "lateral", "lateral_reference", "distance",
            "min_flexion_deg", "drift_tolerance"}
    body = _keys(body, here, keys, keys)
    try:
        min_flexion = float(body["min_flexion_deg"])
        tolerance = float(body["drift_tolerance"])
    except (TypeError, ValueError):
        raise MarkerSetError(f"{here}.min_flexion_deg and .drift_tolerance must be numbers") from None
    if not 0.0 <= min_flexion < 180.0:
        raise MarkerSetError(f"{here}.min_flexion_deg must lie in [0, 180)")
    if tolerance <= 0.0:
        raise MarkerSetError(f"{here}.drift_tolerance must be positive (metres)")
    return rule, {
        "min_flexion": np.radians(min_flexion),
        "drift_tolerance": tolerance,
        "reference_label": str(body["lateral_reference"]),
        "proximal": _point(body["proximal"], f"{here}.proximal"),
        "distal": _point(body["distal"], f"{here}.distal"),
        "lateral": _point(body["lateral"], f"{here}.lateral"),
        "reference": _point(body["lateral_reference"], f"{here}.lateral_reference"),
        "distance": _linear(body["distance"], f"{here}.distance"),
    }, weight


def _markers_in(expression, out: set[str]) -> None:
    kind, item = expression
    if kind == "marker":
        out.add(item)
    elif kind == "mean":
        for part in item:
            _markers_in(part, out)


@dataclass(frozen=True)
class MarkerSet:
    """A parsed marker-set description. Build one with :meth:`load` or :meth:`from_mapping`."""

    id: str
    description: str
    measurements: tuple[str, ...]
    lengths: Mapping[str, tuple[Any, Any]]
    frames: Mapping[str, _Frame]
    centres: Mapping[str, tuple[str, dict, float]]
    segments: Mapping[str, tuple[str, float]]
    fill: Mapping[str, str]
    path: Path | None = None
    sha256: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, path: Path | None = None,
                     sha256: str | None = None) -> MarkerSet:
        data = _keys(data, "<root>",
                     {"schema", "id", "description", "measurements", "lengths", "frames",
                      "centres", "segments", "fill"},
                     {"schema", "id", "centres"})
        if data["schema"] != SCHEMA_ID:
            raise MarkerSetError(f"schema must be {SCHEMA_ID}, got {data['schema']!r}")
        measurements = tuple(str(m) for m in data.get("measurements", ()) or ())
        lengths = {}
        for name, value in (data.get("lengths") or {}).items():
            value = _keys(value, f"lengths.{name}", {"between"}, {"between"})
            ends = value["between"]
            if not isinstance(ends, list) or len(ends) != 2:
                raise MarkerSetError(f"lengths.{name}.between must name two points")
            lengths[str(name)] = (_point(ends[0], f"lengths.{name}.between[0]"),
                                  _point(ends[1], f"lengths.{name}.between[1]"))
        names = set(lengths) | set(measurements)
        clash = (set(lengths) & set(measurements)) | ({_CONSTANT} & names)
        if clash:
            raise MarkerSetError(f"names used twice or reserved: {sorted(clash)}")
        frames = {str(name): _frame(value, f"frames.{name}")
                  for name, value in (data.get("frames") or {}).items()}
        centres = {}
        for joint, value in (data["centres"] or {}).items():
            if joint not in JOINT_NAMES:
                raise MarkerSetError(f"centres.{joint}: not an SMPL-24 joint")
            centres[joint] = _centre(value, f"centres.{joint}", frames)
        segments = {}
        for joint, value in (data.get("segments") or {}).items():
            if joint not in JOINT_NAMES:
                raise MarkerSetError(f"segments.{joint}: not an SMPL-24 joint")
            value = _keys(value, f"segments.{joint}", {"frame", "weight"}, {"frame"})
            if value["frame"] not in frames:
                raise MarkerSetError(f"segments.{joint}.frame names no frame: {value['frame']!r}")
            weight = float(value.get("weight", 1.0))
            if weight <= 0:
                raise MarkerSetError(f"segments.{joint}.weight must be positive")
            segments[joint] = (value["frame"], weight)
        known = set(lengths) | set(measurements)
        for joint, (rule, body, _) in centres.items():
            terms = []
            if rule == "offset":
                terms = [t for component in body["components"] for t in component]
            elif rule in ("chord", "hinge"):
                terms = list(body["distance"])
            unknown = {t for t in terms if t != _CONSTANT} - known
            if unknown:
                raise MarkerSetError(
                    f"centres.{joint} uses {sorted(unknown)}, which are neither lengths nor "
                    "declared measurements"
                )
        return cls(
            id=str(data["id"]),
            description=str(data.get("description", "")),
            measurements=measurements,
            lengths=lengths,
            frames=frames,
            centres=centres,
            segments=segments,
            fill=dict(data.get("fill") or {}),
            path=path,
            sha256=sha256,
        )

    @classmethod
    def load(cls, path: str | Path) -> MarkerSet:
        path = Path(path)
        raw = path.read_bytes()
        try:
            data = yaml.safe_load(raw.decode("utf-8"))
        except yaml.YAMLError as error:
            raise MarkerSetError(f"{path}: not valid YAML: {error}") from error
        try:
            return cls.from_mapping(data, path=path, sha256=hashlib.sha256(raw).hexdigest())
        except MarkerSetError as error:
            raise MarkerSetError(f"{path}: {error}") from None

    @property
    def labels(self) -> frozenset[str]:
        """Every marker label the rules read."""
        out: set[str] = set()
        for start, end in self.lengths.values():
            _markers_in(start, out)
            _markers_in(end, out)
        for frame in self.frames.values():
            for expression in (frame.origin, frame.primary.start, frame.primary.end,
                               frame.secondary.start, frame.secondary.end):
                if expression is not None:
                    _markers_in(expression, out)
        for rule, body, _ in self.centres.values():
            for key in ("point", "origin", "proximal", "lateral", "plane", "distal", "reference"):
                if body.get(key) is not None:
                    _markers_in(body[key], out)
        return frozenset(out)

    def evaluate(self, labels: Sequence[str], positions: np.ndarray, valid: np.ndarray,
                 measurements: Mapping[str, float]) -> MarkerSetResult:
        """Apply the rules to ``(T, M, 3)`` marker positions in metres."""
        missing = sorted(set(self.measurements) - set(measurements))
        if missing:
            raise MarkerSetError(
                f"marker set {self.id} needs the subject measurements {missing}"
            )
        absent = sorted(self.labels - set(labels))
        if absent:
            raise MarkerSetError(f"marker set {self.id} reads markers the data lacks: {absent}")
        return _Evaluation(self, labels, positions, valid, measurements).run()

    def targets(self, labels: Sequence[str], positions: np.ndarray, valid: np.ndarray,
                measurements: Mapping[str, float], *, fps: float) -> tuple[Targets, MarkerSetResult]:
        """The position and orientation targets the rules give, with their provenance."""
        result = self.evaluate(labels, positions, valid, measurements)
        joints = list(self.centres)
        position_targets = PositionTargets(
            joints,
            np.stack([result.centres[j] for j in joints], axis=1),
            np.stack([result.centre_valid[j] for j in joints], axis=1),
            np.array([self.centres[j][2] for j in joints]),
            [f"{j} ({self.centres[j][0]})" for j in joints],
        )
        orientation_targets = None
        if self.segments:
            oriented = list(self.segments)
            orientation_targets = OrientationTargets(
                oriented,
                np.stack([result.frames[self.segments[j][0]] for j in oriented], axis=1),
                np.stack([result.frame_valid[self.segments[j][0]] for j in oriented], axis=1),
                np.array([self.segments[j][1] for j in oriented]),
                [f"{j} (frame {self.segments[j][0]})" for j in oriented],
            )
        provenance = provenance_for(joints, list(self.segments), self.fill)
        return Targets(position_targets, orientation_targets, fps, provenance), result


@dataclass(frozen=True)
class MarkerSetResult:
    """What the rules produced: centres ``(T, 3)``, frames ``(T, 3, 3)``, their validity, and
    the lengths measured from the markers."""

    centres: dict[str, np.ndarray]
    centre_valid: dict[str, np.ndarray]
    frames: dict[str, np.ndarray]
    frame_valid: dict[str, np.ndarray]
    lengths: dict[str, float] = field(default_factory=dict)
    #: What the rules found doubtful in this trial, for the conversion record.
    warnings: list[str] = field(default_factory=list)


# --- evaluation ---------------------------------------------------------------------------------


def _unit(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    ok = norm[..., 0] > _TINY
    return vectors / np.where(ok[..., None], norm, 1.0), ok


class _Evaluation:
    def __init__(self, markerset: MarkerSet, labels, positions, valid, measurements):
        self.set = markerset
        self.index = {label: i for i, label in enumerate(labels)}
        self.positions = np.asarray(positions, dtype=np.float64)
        self.valid = np.asarray(valid, dtype=bool) & np.isfinite(self.positions).all(axis=2)
        self.measurements = {k: float(v) for k, v in measurements.items()}
        self.centres: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.lengths: dict[str, float] = {}
        self.busy: set[str] = set()
        self.warnings: list[str] = []

    def run(self) -> MarkerSetResult:
        for name in self.set.lengths:
            self.length(name)
        for joint in self.set.centres:
            self.centre(joint)
        for name in self.set.frames:
            self.frame(name)
        return MarkerSetResult(
            centres={k: v[0] for k, v in self.centres.items()},
            centre_valid={k: v[1] for k, v in self.centres.items()},
            frames={k: v[0] for k, v in self.frames.items()},
            frame_valid={k: v[1] for k, v in self.frames.items()},
            lengths=dict(self.lengths),
            warnings=list(self.warnings),
        )

    def _guard(self, key: str):
        if key in self.busy:
            raise MarkerSetError(f"marker set {self.set.id}: {key} depends on itself")
        self.busy.add(key)

    def point(self, expression) -> tuple[np.ndarray, np.ndarray]:
        kind, item = expression
        if kind == "marker":
            column = self.index[item]
            return self.positions[:, column], self.valid[:, column]
        if kind == "centre":
            return self.centre(item)
        parts = [self.point(part) for part in item]
        values = np.mean([p for p, _ in parts], axis=0)
        ok = np.logical_and.reduce([v for _, v in parts])
        return values, ok

    def quantity(self, terms: Mapping[str, float]) -> float:
        total = 0.0
        for name, coefficient in terms.items():
            if name == _CONSTANT:
                total += coefficient
            elif name in self.set.lengths:
                total += coefficient * self.length(name)
            else:
                total += coefficient * self.measurements[name]
        return total

    def length(self, name: str) -> float:
        if name not in self.lengths:
            self._guard(f"length {name}")
            start, end = (self.point(e) for e in self.set.lengths[name])
            both = start[1] & end[1]
            if not both.any():
                raise MarkerSetError(f"length {name}: its two points are never seen together")
            self.lengths[name] = float(np.median(
                np.linalg.norm(end[0][both] - start[0][both], axis=1)
            ))
            self.busy.discard(f"length {name}")
        return self.lengths[name]

    def frame(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in self.frames:
            self._guard(f"frame {name}")
            rule = self.set.frames[name]
            axes = {}
            ok = np.ones(self.positions.shape[0], dtype=bool)
            for which in (rule.primary, rule.secondary):
                start, start_ok = self.point(which.start)
                end, end_ok = self.point(which.end)
                axes[which.name] = end - start
                ok &= start_ok & end_ok
            first, first_ok = _unit(axes[rule.primary.name])
            second = axes[rule.secondary.name]
            second = second - np.sum(second * first, axis=-1, keepdims=True) * first
            second, second_ok = _unit(second)
            ok &= first_ok & second_ok
            basis = {rule.primary.name: first, rule.secondary.name: second}
            third = next(axis for axis in _AXES if axis not in basis)
            index = _AXES.index(third)
            basis[third] = np.cross(basis[_AXES[(index + 1) % 3]], basis[_AXES[(index + 2) % 3]])
            rotation = np.stack([basis[axis] for axis in _AXES], axis=-1)
            rotation = np.where(ok[:, None, None], rotation, np.nan)
            self.frames[name] = (rotation, ok)
            self.busy.discard(f"frame {name}")
        return self.frames[name]

    def centre(self, joint: str) -> tuple[np.ndarray, np.ndarray]:
        if joint in self.centres:
            return self.centres[joint]
        if joint not in self.set.centres:
            raise MarkerSetError(f"marker set {self.set.id} has no rule for the centre {joint}")
        self._guard(f"centre {joint}")
        rule, body, _ = self.set.centres[joint]
        if rule == "point":
            value, ok = self.point(body["point"])
        elif rule == "offset":
            value, ok = self._offset(body)
        elif rule == "chord":
            value, ok = self._chord(body)
        else:
            value, ok = self._hinge(joint, body)
        value = np.where(ok[:, None], value, np.nan)
        self.centres[joint] = (value, ok)
        self.busy.discard(f"centre {joint}")
        return self.centres[joint]

    def _offset(self, body):
        rotation, ok = self.frame(body["frame"])
        origin_expression = body["origin"] or self.set.frames[body["frame"]].origin
        origin, origin_ok = self.point(origin_expression)
        local = np.array([self.quantity(component) for component in body["components"]])
        return origin + np.einsum("tij,j->ti", np.nan_to_num(rotation), local), ok & origin_ok

    def _chord(self, body):
        proximal, p_ok = self.point(body["proximal"])
        lateral, l_ok = self.point(body["lateral"])
        plane, w_ok = self.point(body["plane"])
        distance = self.quantity(body["distance"])
        along = proximal - lateral
        span = np.linalg.norm(along, axis=-1)
        u, u_ok = _unit(along)
        across = plane - lateral
        across = across - np.sum(across * u, axis=-1, keepdims=True) * u
        v, v_ok = _unit(across)
        reachable = span > distance
        cosine = np.where(reachable, distance / np.where(span > 0, span, 1.0), 0.0)
        sine = np.sqrt(np.clip(1.0 - cosine**2, 0.0, 1.0))
        side = 1.0 if body["toward"] else -1.0
        centre = lateral + distance * (cosine[:, None] * u + side * sine[:, None] * v)
        return centre, p_ok & l_ok & w_ok & u_ok & v_ok & reachable

    def _hinge(self, joint: str, body):
        proximal, p_ok = self.point(body["proximal"])
        distal, d_ok = self.point(body["distal"])
        lateral, l_ok = self.point(body["lateral"])
        reference, r_ok = self.point(body["reference"])
        distance = self.quantity(body["distance"])
        ok = p_ok & d_ok & l_ok & r_ok

        def settle(side: float):
            """The fixed point reached from one side of the marker (the axis keeps its direction
            from round to round), and how far the reference sits beyond the line from the
            proximal centre to the lateral marker, towards the axis's side."""
            normal, n_ok = _unit(np.cross(proximal - lateral, distal - lateral))
            normal = side * normal
            centre = lateral - distance * normal
            for _ in range(_HINGE_ROUNDS):
                turned, n_ok = _unit(np.cross(proximal - centre, distal - centre))
                keep = np.sum(turned * normal, axis=-1) >= 0.0
                normal = np.where(keep[:, None], turned, -turned)
                centre = lateral - distance * normal
            along, a_ok = _unit(centre - proximal)
            reach = np.sum((lateral - proximal) * along, axis=-1)
            level = np.sum((reference - proximal) * along, axis=-1) / np.where(reach != 0, reach, 1)
            on_line = proximal + level[:, None] * (lateral - proximal)
            margin = np.sum((reference - on_line) * normal, axis=-1)
            lower, _ = _unit(distal - centre)
            flexion = np.pi - np.arccos(np.clip(np.sum(-along * lower, axis=-1), -1.0, 1.0))
            return centre, margin, flexion, n_ok & a_ok & (reach != 0)

        # Two centres fit the rule, one on each side of the lateral marker (the second is the
        # first turned about the line through the proximal and distal centres, so both keep the
        # segment lengths). A first pick takes, per frame, the one whose reference lies further
        # beyond the line from the proximal centre to the lateral marker: the right one when the
        # reference sits directly laterally and further out than that line. A reference shifted
        # towards the side the limb does not flex to can still favour the mirror on some frames,
        # so the pick is then checked against the segment itself: the proximal centre, the
        # lateral marker and the reference all ride on the proximal segment, and so does the
        # right centre, which therefore holds still in their frame while the mirror moves with
        # the flexion. Each frame takes the candidate nearer the trial's median there, within
        # the marker set's drift tolerance; a trial whose chosen centre still wanders beyond it
        # gets a warning, since then the median itself may sit on the mirror.
        first, first_margin, first_flexion, first_ok = settle(1.0)
        second, second_margin, second_flexion, second_ok = settle(-1.0)
        usable = ok & first_ok & second_ok
        y_axis, y_ok = _unit(lateral - proximal)
        x_axis = reference - proximal
        x_axis, x_ok = _unit(x_axis - np.sum(x_axis * y_axis, axis=-1, keepdims=True) * y_axis)
        frame = np.stack([x_axis, y_axis, np.cross(x_axis, y_axis)], axis=-1)
        usable &= y_ok & x_ok

        def on_segment(points):
            return np.einsum("tji,tj->ti", frame, points - proximal)

        pick_first = first_margin >= second_margin
        flexion = np.where(pick_first, first_flexion, second_flexion)
        margin = np.where(pick_first, first_margin, second_margin)
        seed = usable & (margin > 0.0) & (flexion >= body["min_flexion"])
        if not seed.any():
            return np.where(pick_first[:, None], first, second), np.zeros_like(usable)
        first_local, second_local = on_segment(first), on_segment(second)
        chosen = np.where(pick_first[:, None], first_local, second_local)
        middle = np.median(chosen[seed], axis=0)
        first_drift = np.linalg.norm(first_local - middle, axis=-1)
        second_drift = np.linalg.norm(second_local - middle, axis=-1)
        pick_first = first_drift <= second_drift
        drift = np.minimum(first_drift, second_drift)
        centre = np.where(pick_first[:, None], first, second)
        margin = np.where(pick_first, first_margin, second_margin)
        flexion = np.where(pick_first, first_flexion, second_flexion)
        tolerance = body["drift_tolerance"]
        checked = usable & (flexion >= body["min_flexion"])
        valid = checked & (margin > 0.0) & (drift <= tolerance)
        if checked.any():
            spread = float(np.percentile(drift[checked], 95))
            if spread > tolerance:
                # The median may itself sit on the mirror; no frame of this trial is trusted.
                valid = np.zeros_like(valid)
                self.warnings.append(
                    f"{joint}: the centre moves by {spread * 1000:.0f} mm (95th percentile) on the "
                    f"proximal segment, beyond the {tolerance * 1000:.0f} mm the marker set "
                    f"allows, so this trial has no {joint} centre; check that "
                    f"{body['reference_label']} sits directly laterally, further out than the "
                    "line to the lateral marker"
                )
        return centre, valid


# --- conditioning ---------------------------------------------------------------------------------


def fill_gaps(positions: np.ndarray, valid: np.ndarray, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly bridge each marker's interior gaps of at most ``max_frames`` frames.

    A gap at the start or end of a trial has only one side and is left invalid; so is a longer
    gap. ``max_frames`` zero fills nothing.
    """
    positions = np.array(positions, dtype=np.float64, copy=True)
    valid = np.array(valid, dtype=bool, copy=True)
    if max_frames <= 0:
        return positions, valid
    frames = np.arange(positions.shape[0])
    for column in range(positions.shape[1]):
        seen = valid[:, column]
        if seen.all() or seen.sum() < 2:
            continue
        missing = np.flatnonzero(~seen)
        runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
        known = np.flatnonzero(seen)
        for run in runs:
            if run[0] == 0 or run[-1] == frames[-1] or run.size > max_frames:
                continue
            for axis in range(3):
                positions[run, column, axis] = np.interp(
                    run, known, positions[known, column, axis]
                )
            valid[run, column] = True
    return positions, valid


def from_table(table: TrcTable) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, float]:
    """``(labels, positions in metres, valid, fps)`` from a ``.trc`` / ``.c3d`` table."""
    scale = length_scale(table.units)
    return table.labels, table.positions * scale, table.valid.copy(), float(table.data_rate_hz)
