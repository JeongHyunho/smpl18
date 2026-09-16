"""Correspondence tables: which source joint, body or centre feeds which SMPL-24 joint.

A table (schema ``smpl18_correspondence_v1``) is data. Its ``names`` say what its sources are:

* ``opensim_joints`` / ``bvh_joints``: an entry lists skeleton joints, ``source: [first, ...,
  last]``. The first joint's centre is the SMPL joint's **position** target; the body the last
  joint moves is its **orientation** target (the distal segment SMPL spans -- the calcaneus past
  the subtalar joint, the radius past the radioulnar one). ``position: false`` or
  ``orientation: false`` drops either, where the source's point or frame is not the SMPL one.
* ``centres``: an entry names a centre, ``centre: LKJC``, and optionally the segment whose
  rotation the source stores, ``segment: LSK``.

``fill`` gives a joint no source of its own a rule (``weld`` holds it at identity, ``distribute``
lets the pose prior share a turn out to it). A ``lumbar`` block says that one source joint spans
several SMPL spine joints: its centre is the position target of ``centre_on`` and the
``trunk_body`` rotation is the orientation target of the last of them; the pose prior shares the
turn among them, which is the ``equal_thirds`` split for a turn about one axis.

The table is intersected with the source: a joint the file lacks leaves its SMPL joint without
that target, and the report says so. ``aliases`` rename source names; ``prefixes`` are stripped
from them first (rigs that prefix every joint with a namespace).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smpl18.skeleton.definition import JOINT_NAMES
from smpl18.skeleton.frames import FrameChange
from smpl18.sources.base import SegmentPlacement, SkeletonModel

from .targets import FILL_RULES, OrientationTargets, PositionTargets, Targets, provenance_for

__all__ = [
    "SCHEMA_ID",
    "Correspondence",
    "CorrespondenceError",
    "CorrespondenceUse",
]

SCHEMA_ID = "smpl18_correspondence_v1"
NAME_KINDS = ("opensim_joints", "bvh_joints", "centres")
LUMBAR_SPLITS = ("equal_thirds",)

_TOP_KEYS = {"schema", "id", "description", "names", "joints", "lumbar", "trunk_body",
             "root_translation", "aliases", "prefixes"}
_SKELETON_ENTRY = {"source", "fill", "position", "orientation", "weight"}
_CENTRE_ENTRY = {"centre", "segment", "fill", "position", "orientation", "weight"}
_LUMBAR_KEYS = {"source_joint", "smpl", "split", "centre_on"}


class CorrespondenceError(ValueError):
    pass


@dataclass(frozen=True)
class _Entry:
    smpl: str
    sources: tuple[str, ...] = ()
    segment: str | None = None
    fill: str | None = None
    position: bool = True
    orientation: bool = True
    weight: float = 1.0


@dataclass(frozen=True)
class _Lumbar:
    source_joint: str
    smpl: tuple[str, ...]
    centre_on: str


@dataclass(frozen=True)
class CorrespondenceUse:
    """What a table found in one source: the targets it made and the names it missed."""

    positions: dict[str, str] = field(default_factory=dict)
    orientations: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    fill: dict[str, str] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "position_targets": dict(self.positions),
            "orientation_targets": dict(self.orientations),
            "missing_in_source": list(self.missing),
            "fill": dict(self.fill),
        }


def _check_keys(value: Any, where: str, allowed: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorrespondenceError(f"{where}: expected a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise CorrespondenceError(f"{where}: unknown keys {sorted(unknown)}")
    return value


def _flag(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise CorrespondenceError(f"{where} must be true or false")
    return value


def _labels(columns) -> list[str]:
    """``"<smpl joint> <- <source>"`` per column, numbered where a pair repeats."""
    labels, seen = [], {}
    for joint, _, _, _, source in columns:
        label = f"{joint} <- {source}"
        seen[label] = seen.get(label, 0) + 1
        labels.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
    return labels


@dataclass(frozen=True)
class Correspondence:
    id: str
    names: str
    entries: Mapping[str, _Entry]
    lumbar: _Lumbar | None
    trunk_body: str | None
    root_translation: tuple[str, ...]
    aliases: Mapping[str, str]
    prefixes: tuple[str, ...]
    description: str = ""
    path: Path | None = None
    sha256: str | None = None

    # --- loading ---------------------------------------------------------------------------

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, path: Path | None = None,
                     sha256: str | None = None) -> Correspondence:
        data = _check_keys(data, "<root>", _TOP_KEYS)
        for key in ("schema", "id", "names", "joints"):
            if key not in data:
                raise CorrespondenceError(f"missing key {key!r}")
        if data["schema"] != SCHEMA_ID:
            raise CorrespondenceError(f"schema must be {SCHEMA_ID}, got {data['schema']!r}")
        names = data["names"]
        if names not in NAME_KINDS:
            raise CorrespondenceError(f"names must be one of {NAME_KINDS}, got {names!r}")
        skeleton = names != "centres"
        entries = {}
        for smpl, value in (data["joints"] or {}).items():
            where = f"joints.{smpl}"
            if smpl not in JOINT_NAMES:
                raise CorrespondenceError(f"{where}: not an SMPL-24 joint")
            value = _check_keys(value, where, _SKELETON_ENTRY if skeleton else _CENTRE_ENTRY)
            fill = value.get("fill")
            if fill is not None and fill not in FILL_RULES:
                raise CorrespondenceError(f"{where}.fill must be one of {FILL_RULES}")
            if skeleton:
                sources = value.get("source", [])
                if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
                    raise CorrespondenceError(f"{where}.source must be a list of joint names")
                segment = None
            else:
                sources = [value["centre"]] if "centre" in value else []
                segment = value.get("segment")
            if not sources and segment is None and fill is None:
                raise CorrespondenceError(f"{where}: give a source or a fill rule")
            if (sources or segment) and fill is not None:
                raise CorrespondenceError(f"{where}: a joint with a source takes no fill rule")
            weight = float(value.get("weight", 1.0))
            if weight <= 0:
                raise CorrespondenceError(f"{where}.weight must be positive")
            entries[smpl] = _Entry(
                smpl=smpl,
                sources=tuple(sources),
                segment=segment,
                fill=fill,
                position=_flag(value.get("position", True), f"{where}.position"),
                orientation=_flag(value.get("orientation", True), f"{where}.orientation"),
                weight=weight,
            )
        lumbar = None
        if data.get("lumbar") is not None:
            block = _check_keys(data["lumbar"], "lumbar", _LUMBAR_KEYS)
            if block.get("split", "equal_thirds") not in LUMBAR_SPLITS:
                raise CorrespondenceError(f"lumbar.split must be one of {LUMBAR_SPLITS}")
            spine = tuple(block["smpl"])
            if not spine or any(j not in JOINT_NAMES for j in spine):
                raise CorrespondenceError("lumbar.smpl must list SMPL-24 joints")
            if block["centre_on"] not in spine:
                raise CorrespondenceError("lumbar.centre_on must be one of lumbar.smpl")
            lumbar = _Lumbar(str(block["source_joint"]), spine, str(block["centre_on"]))
        prefixes = tuple(str(p) for p in (data.get("prefixes") or ()))
        return cls(
            id=str(data["id"]),
            names=names,
            entries=entries,
            lumbar=lumbar,
            trunk_body=data.get("trunk_body"),
            root_translation=tuple(data.get("root_translation") or ()),
            aliases={str(k): str(v) for k, v in (data.get("aliases") or {}).items()},
            prefixes=prefixes,
            description=str(data.get("description", "")),
            path=path,
            sha256=sha256,
        )

    @classmethod
    def load(cls, path: str | Path) -> Correspondence:
        path = Path(path)
        raw = path.read_bytes()
        try:
            data = yaml.safe_load(raw.decode("utf-8"))
        except yaml.YAMLError as error:
            raise CorrespondenceError(f"{path}: not valid YAML: {error}") from error
        try:
            return cls.from_mapping(data, path=path, sha256=hashlib.sha256(raw).hexdigest())
        except CorrespondenceError as error:
            raise CorrespondenceError(f"{path}: {error}") from None

    def record(self) -> dict[str, Any]:
        return {"id": self.id, "sha256": self.sha256,
                "path": None if self.path is None else self.path.name}

    # --- matching names --------------------------------------------------------------------

    def _canonical(self, name: str) -> str:
        for prefix in self.prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return self.aliases.get(name, name)

    def _lookup(self, available: Sequence[str]) -> dict[str, str]:
        """Table name -> the name the source uses."""
        out: dict[str, str] = {}
        for name in available:
            out.setdefault(self._canonical(name), name)
        return out

    # --- building targets ------------------------------------------------------------------

    def _finish(self, positions, orientations, use: CorrespondenceUse, frames: int, fps: float,
                valid) -> Targets:
        if not positions:
            raise CorrespondenceError(f"correspondence {self.id} found no position target")
        joints = [column[0] for column in positions]
        position_targets = PositionTargets(
            joints,
            np.stack([column[1] for column in positions], axis=1),
            np.stack([np.broadcast_to(column[2], (frames,)) & valid for column in positions],
                     axis=1),
            np.array([column[3] for column in positions]),
            _labels(positions),
        )
        orientation_targets = None
        if orientations:
            orientation_targets = OrientationTargets(
                [column[0] for column in orientations],
                np.stack([column[1] for column in orientations], axis=1),
                np.stack([np.broadcast_to(column[2], (frames,)) & valid
                          for column in orientations], axis=1),
                np.array([column[3] for column in orientations]),
                _labels(orientations),
            )
        fill = {entry.smpl: entry.fill for entry in self.entries.values() if entry.fill}
        use.fill.update(fill)
        provenance = provenance_for(joints, [column[0] for column in orientations], fill)
        return Targets(position_targets, orientation_targets, fps, provenance)

    def skeleton_targets(self, skeleton: SkeletonModel, placement: SegmentPlacement, *,
                         frame: FrameChange, fps: float, frame_valid=None
                         ) -> tuple[Targets, CorrespondenceUse]:
        """Targets from a skeleton's forward kinematics, turned into the output frame."""
        if self.names == "centres":
            raise CorrespondenceError(f"correspondence {self.id} names centres, not joints")
        joints = self._lookup(skeleton.joints)
        bodies = set(skeleton.bodies)
        children = skeleton.joint_child_bodies
        frames = next(iter(placement.joint_centres.values())).shape[0]
        valid = np.ones(frames, bool) if frame_valid is None else np.asarray(frame_valid, bool)
        use = CorrespondenceUse()
        positions, orientations = [], []
        for smpl in JOINT_NAMES:
            entry = self.entries.get(smpl)
            if entry is None or not entry.sources:
                continue
            first, last = entry.sources[0], entry.sources[-1]
            if entry.position:
                if first in joints:
                    values = frame.apply(placement.joint_centres[joints[first]])
                    positions.append((smpl, values, np.isfinite(values).all(axis=1), entry.weight,
                                      joints[first]))
                    use.positions[smpl] = joints[first]
                else:
                    use.missing.append(first)
            if entry.orientation:
                if last in joints and children.get(joints[last]) in placement.rotations:
                    body = children[joints[last]]
                    values = frame.rotation @ placement.rotations[body]
                    orientations.append(
                        (smpl, values, np.isfinite(values).all(axis=(1, 2)), entry.weight, body)
                    )
                    use.orientations[smpl] = body
                elif last not in joints:
                    use.missing.append(last)
        if self.lumbar is not None:
            source = self.lumbar.source_joint
            if source in joints:
                values = frame.apply(placement.joint_centres[joints[source]])
                positions.append((self.lumbar.centre_on, values,
                                  np.isfinite(values).all(axis=1), 1.0, joints[source]))
                use.positions[self.lumbar.centre_on] = joints[source]
            else:
                use.missing.append(source)
            trunk = self.trunk_body
            if trunk is not None and trunk in bodies:
                values = frame.rotation @ placement.rotations[trunk]
                orientations.append((self.lumbar.smpl[-1], values,
                                     np.isfinite(values).all(axis=(1, 2)), 1.0, trunk))
                use.orientations[self.lumbar.smpl[-1]] = trunk
            elif trunk is not None:
                use.missing.append(trunk)
        use.missing[:] = sorted(set(use.missing))
        return self._finish(positions, orientations, use, frames, fps, valid), use

    def centre_targets(self, names: Sequence[str], positions: np.ndarray, valid: np.ndarray, *,
                       fps: float, rotations: Mapping[str, np.ndarray] | None = None,
                       rotations_valid: Mapping[str, np.ndarray] | None = None
                       ) -> tuple[Targets, CorrespondenceUse]:
        """Targets from joint-centre trajectories ``(T, K, 3)`` already in the output frame."""
        if self.names != "centres":
            raise CorrespondenceError(f"correspondence {self.id} names skeleton joints, not centres")
        lookup = self._lookup(names)
        index = {name: i for i, name in enumerate(names)}
        segments = self._lookup(list(rotations or {}))
        frames = positions.shape[0]
        use = CorrespondenceUse()
        position_list, orientation_list = [], []
        for smpl in JOINT_NAMES:
            entry = self.entries.get(smpl)
            if entry is None:
                continue
            if entry.sources and entry.position:
                name = entry.sources[0]
                if name in lookup:
                    column = index[lookup[name]]
                    position_list.append((smpl, positions[:, column], valid[:, column],
                                          entry.weight, lookup[name]))
                    use.positions[smpl] = lookup[name]
                else:
                    use.missing.append(name)
            if entry.segment is not None and entry.orientation:
                if entry.segment in segments:
                    actual = segments[entry.segment]
                    stack = rotations[actual]
                    ok = (np.ones(frames, bool) if rotations_valid is None
                          else rotations_valid[actual])
                    orientation_list.append((smpl, stack, ok, entry.weight, actual))
                    use.orientations[smpl] = actual
                else:
                    use.missing.append(entry.segment)
        use.missing[:] = sorted(set(use.missing))
        return self._finish(position_list, orientation_list, use, frames, fps,
                            np.ones(frames, bool)), use
