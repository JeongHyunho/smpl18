"""The on-disk corpus (``docs/corpus-format.md``): writing it, reading it, rebuilding 24 joints.

::

    <corpus>/SUMMARY.json
    <corpus>/<subject>/subject.json
    <corpus>/<subject>/<trial>.npz
    <corpus>/<subject>/<trial>.manifest.json

A trial stores the 18 kept joints. The subject record carries the four frozen constants, so
:meth:`CorpusTrial.local_rotations_24` gives the 24-joint pose back: the frozen joints at their
constants, the kept joints as stored (the absorbers already carry what the freeze removed), the
hands at identity. Every kept segment's world orientation is then that of the fitted 24-joint
pose exactly; joint positions are within the subject's recorded ``reduced_model.fit`` residual.

``SUMMARY.json`` is rebuilt from what the directory holds each time a subject is written, so
several runs can add subjects to one corpus.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from smpl18 import __version__
from smpl18.skeleton.definition import (
    FROZEN_JOINT_NAMES,
    FROZEN_JOINTS,
    JOINT18_NAMES,
    KEEP18,
    NUM_JOINTS,
)
from smpl18.skeleton.kinematics import fk_batch, rest_joints
from smpl18.skeleton.rotations import axis_angle_to_matrix, matrix_to_axis_angle

__all__ = [
    "FORMAT_ID",
    "FORMAT_VERSION",
    "TRIAL_KEYS",
    "Corpus",
    "CorpusError",
    "CorpusSubject",
    "CorpusTrial",
    "converter_record",
    "read_corpus",
    "write_subject",
    "write_summary",
    "write_trial",
]

FORMAT_ID = "smpl18_corpus"
FORMAT_VERSION = "1.0"
TRIAL_KEYS = ("poses", "joint_names", "trans", "fps", "up_axis", "joint_provenance",
              "frame_valid")


class CorpusError(ValueError):
    pass


def _commit() -> str | None:
    """The package checkout's commit, when it is one; ``None`` for an installed copy."""
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def converter_record() -> dict[str, Any]:
    return {"package": "smpl18", "version": __version__, "commit": _commit()}


def _plain(value: Any) -> Any:
    """JSON-ready copies of numpy values, recursively."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_plain(data), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_trial(subject_dir: Path, trial_id: str, *, poses: np.ndarray, trans: np.ndarray,
                fps: float, up_axis: str, joint_provenance, frame_valid: np.ndarray,
                manifest: dict[str, Any]) -> Path:
    """Write ``<trial>.npz`` (the 18-joint arrays, nothing else) and its manifest."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (len(KEEP18), 3):
        raise CorpusError(f"poses must be (T, 18, 3), got {poses.shape}")
    frames = poses.shape[0]
    trans = np.asarray(trans, dtype=np.float64)
    if trans.shape != (frames, 3):
        raise CorpusError(f"trans must be ({frames}, 3), got {trans.shape}")
    provenance = np.asarray(joint_provenance, dtype=str)
    if provenance.shape != (len(KEEP18),):
        raise CorpusError("joint_provenance must name the 18 kept joints")
    subject_dir.mkdir(parents=True, exist_ok=True)
    path = subject_dir / f"{trial_id}.npz"
    np.savez_compressed(
        path,
        poses=poses,
        joint_names=np.array(JOINT18_NAMES),
        trans=trans,
        fps=np.float64(fps),
        up_axis=np.array(up_axis),
        joint_provenance=provenance,
        frame_valid=np.asarray(frame_valid, dtype=bool),
    )
    _write_json(subject_dir / f"{trial_id}.manifest.json", {"trial_id": trial_id, **manifest})
    return path


def write_subject(subject_dir: Path, record: dict[str, Any]) -> Path:
    subject_dir.mkdir(parents=True, exist_ok=True)
    path = subject_dir / "subject.json"
    _write_json(path, record)
    return path


def write_summary(root: Path, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rebuild ``SUMMARY.json`` from the subjects and trials under ``root``."""
    root = Path(root)
    subjects, trials, frames = 0, 0, 0
    kinds, formats, settings, profiles = set(), set(), set(), {}
    provenance: dict[str, Counter] = {name: Counter() for name in JOINT18_NAMES}
    for subject_file in sorted(root.glob("*/subject.json")):
        subjects += 1
        for trial_file in sorted(subject_file.parent.glob("*.npz")):
            trials += 1
            with np.load(trial_file, allow_pickle=False) as data:
                frames += int(data["poses"].shape[0])
                for name, code in zip(data["joint_names"], data["joint_provenance"]):
                    provenance[str(name)][str(code)] += 1
            manifest_file = trial_file.with_name(trial_file.stem + ".manifest.json")
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                kinds.add(manifest["source"]["kind"])
                formats.add(manifest["source"]["format"])
                settings.add(manifest.get("settings_sha256"))
                profile = manifest.get("profile") or {}
                profiles[profile.get("id")] = profile
    summary = {
        "format_id": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "converter": converter_record(),
        "profile": next(iter(profiles.values())) if len(profiles) == 1 else sorted(
            p for p in profiles if p is not None
        ),
        "source_kind": next(iter(kinds)) if len(kinds) == 1 else sorted(kinds),
        "format": next(iter(formats)) if len(formats) == 1 else sorted(formats),
        "subjects": subjects,
        "trials": trials,
        "frames": frames,
        "settings_sha256": (next(iter(settings)) if len(settings) == 1
                            else sorted(s for s in settings if s)),
        "skipped": [],
        "provenance_counts": {name: dict(counts) for name, counts in provenance.items()},
    }
    previous = root / "SUMMARY.json"
    if previous.exists():
        summary["skipped"] = json.loads(previous.read_text(encoding="utf-8")).get("skipped", [])
    if extra:
        for key, value in extra.items():
            if key == "skipped":
                summary["skipped"] = [s for s in summary["skipped"]
                                      if s.get("subject") not in {v.get("subject") for v in value}]
                summary["skipped"].extend(value)
            else:
                summary[key] = value
    _write_json(previous, summary)
    return summary


# --- reading -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusTrial:
    id: str
    subject: CorpusSubject
    path: Path
    #: ``(T, 18, 3)`` axis-angle, in ``joint_names`` order.
    poses: np.ndarray
    joint_names: tuple[str, ...]
    trans: np.ndarray
    fps: float
    up_axis: str
    joint_provenance: tuple[str, ...]
    frame_valid: np.ndarray

    @cached_property
    def manifest(self) -> dict[str, Any]:
        path = self.path.with_name(self.path.stem + ".manifest.json")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @property
    def frames(self) -> int:
        return int(self.poses.shape[0])

    def local_rotations_24(self) -> np.ndarray:
        """``(T, 24, 3, 3)``: the frozen joints at the subject's constants, hands at identity."""
        local = np.tile(np.eye(3), (self.frames, NUM_JOINTS, 1, 1))
        local[:, list(KEEP18)] = axis_angle_to_matrix(self.poses)
        for joint, constant in zip(FROZEN_JOINTS, self.subject.constants):
            local[:, joint] = constant
        return local

    def poses_24(self) -> np.ndarray:
        """``(T, 24, 3)`` axis-angle of :meth:`local_rotations_24`."""
        return matrix_to_axis_angle(self.local_rotations_24())

    def joints_world(self, model) -> np.ndarray:
        """``(T, 24, 3)`` world joint centres with the subject's betas on ``model``."""
        rest = rest_joints(model, self.subject.betas)
        positions, _ = fk_batch(rest, self.local_rotations_24(), self.trans)
        return positions


@dataclass(frozen=True)
class CorpusSubject:
    id: str
    path: Path
    record: dict[str, Any]

    @property
    def gender(self) -> str:
        return self.record["gender"]

    @property
    def betas(self) -> np.ndarray:
        return np.asarray(self.record["betas"], dtype=np.float64)

    @property
    def constants(self) -> np.ndarray:
        """``(4, 3, 3)`` frozen-joint constants in the order of ``FROZEN_JOINTS``."""
        reduced = self.record["reduced_model"]
        if tuple(reduced["frozen_joints"]) != FROZEN_JOINT_NAMES:
            raise CorpusError(f"{self.path}: unexpected frozen joints {reduced['frozen_joints']}")
        return axis_angle_to_matrix(np.asarray(reduced["constants"], dtype=np.float64))

    def trial_ids(self) -> list[str]:
        return sorted(path.stem for path in self.path.glob("*.npz"))

    def trial(self, trial_id: str) -> CorpusTrial:
        path = self.path / f"{trial_id}.npz"
        if not path.exists():
            raise CorpusError(f"subject {self.id} has no trial {trial_id!r}")
        with np.load(path, allow_pickle=False) as data:
            unknown = set(data.files) - set(TRIAL_KEYS)
            if unknown:
                raise CorpusError(f"{path}: keys outside the corpus format: {sorted(unknown)}")
            listed = self.record.get("trials")
            if listed is not None and trial_id not in listed:
                raise CorpusError(
                    f"{path} was not converted with {self.id}'s record (its betas and frozen "
                    "constants belong to another run); convert the subject again"
                )
            return CorpusTrial(
                id=trial_id,
                subject=self,
                path=path,
                poses=data["poses"],
                joint_names=tuple(str(n) for n in data["joint_names"]),
                trans=data["trans"],
                fps=float(data["fps"]),
                up_axis=str(data["up_axis"]),
                joint_provenance=tuple(str(p) for p in data["joint_provenance"]),
                frame_valid=data["frame_valid"],
            )

    def trials(self) -> list[CorpusTrial]:
        return [self.trial(trial_id) for trial_id in self.trial_ids()]


@dataclass(frozen=True)
class Corpus:
    root: Path
    summary: dict[str, Any]

    def subject_ids(self) -> list[str]:
        return sorted(path.parent.name for path in self.root.glob("*/subject.json"))

    def subject(self, subject_id: str) -> CorpusSubject:
        path = self.root / subject_id / "subject.json"
        if not path.exists():
            raise CorpusError(f"no subject {subject_id!r} in {self.root}")
        return CorpusSubject(subject_id, path.parent,
                             json.loads(path.read_text(encoding="utf-8")))

    def subjects(self) -> list[CorpusSubject]:
        return [self.subject(subject_id) for subject_id in self.subject_ids()]


def read_corpus(root: str | Path) -> Corpus:
    root = Path(root)
    summary_path = root / "SUMMARY.json"
    if not summary_path.exists():
        raise CorpusError(f"{root} has no SUMMARY.json; not a corpus")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("format_id") != FORMAT_ID:
        raise CorpusError(f"{summary_path}: format_id is {summary.get('format_id')!r}")
    return Corpus(root, summary)
