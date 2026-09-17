"""A corpus trial back in the original SMPL structure, for everything that reads SMPL.

The corpus stores 18 joints because that is the model this package produces. Most other tools --
renderers, viewers, body-model code, the SMPL Blender add-on -- read the 24-joint structure instead,
so this module writes a trial out as ordinary SMPL parameters:

``poses`` ``(T, 72)`` axis-angle, ``betas``, ``trans`` ``(T, 3)``, a frame rate and a gender.

Nothing is approximated. The four frozen joints come back at the subject's constants and the hands
at identity, which is exactly the pose the corpus means (:meth:`smpl18.corpus.CorpusTrial.poses_24`);
every kept segment's world orientation is the fitted one, and joint positions are within the
residual the subject record states. Reading the file back with ``smpl18 convert smpl`` returns the
same rotations, which is what the round-trip test checks.

The npz also carries a few keys of its own -- ``joint_names``, ``joint_provenance``, ``frame_valid``,
``smpl18`` -- so that what was measured and what was inferred is not lost on the way out. A reader
that only knows SMPL ignores them.

``joint_provenance`` has one entry per SMPL joint, which is six more than the corpus stores: the
four frozen joints read ``constant`` (one rotation for the whole subject, not a per-frame
observation) and the two hands ``absent``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from smpl18 import __version__
from smpl18.skeleton.definition import FROZEN_JOINT_NAMES, JOINT_NAMES, NUM_JOINTS

if TYPE_CHECKING:
    from smpl18.corpus import CorpusSubject, CorpusTrial

__all__ = ["FPS_KEY", "SmplSequence", "joint_provenance_24", "sequence_from_trial",
           "write_sequence"]

#: The frame-rate key readers of this format look for first; the published parameter sets spell it
#: this way, and ``smpl18 convert smpl`` accepts it among others.
FPS_KEY = "mocap_framerate"
#: What a joint the corpus does not store per frame is called: a frozen joint carries one rotation
#: for the whole subject, and the hands carry none at all.
CONSTANT = "constant"
ABSENT = "absent"


def joint_provenance_24(stored_names: tuple[str, ...],
                        stored_provenance: tuple[str, ...]) -> tuple[str, ...]:
    """The corpus's 18 provenances spread over all 24 SMPL joints."""
    stored = dict(zip(stored_names, stored_provenance))
    return tuple(
        stored.get(name, CONSTANT if name in FROZEN_JOINT_NAMES else ABSENT)
        for name in JOINT_NAMES
    )


@dataclass(frozen=True)
class SmplSequence:
    """One trial as SMPL parameters: what every SMPL reader needs, and where it came from."""

    #: ``(T, 24, 3)`` axis-angle local rotations, hands at identity.
    poses: np.ndarray
    betas: np.ndarray
    trans: np.ndarray
    fps: float
    gender: str
    up_axis: str
    #: One per SMPL joint: the corpus's ``measured`` / ``derived`` / ``absent``, plus ``constant``
    #: for the frozen joints. And which frames the capture really had.
    joint_provenance: tuple[str, ...]
    frame_valid: np.ndarray
    about: dict[str, Any]

    @property
    def frames(self) -> int:
        return int(self.poses.shape[0])

    def arrays(self, *, flat: bool = True) -> dict[str, np.ndarray]:
        """The npz contents. ``flat`` writes ``poses`` as ``(T, 72)``, as most readers expect."""
        poses = self.poses.reshape(self.frames, NUM_JOINTS * 3) if flat else self.poses
        return {
            "poses": poses,
            "betas": np.asarray(self.betas, dtype=np.float64),
            "trans": np.asarray(self.trans, dtype=np.float64),
            FPS_KEY: np.array(float(self.fps)),
            "gender": np.array(self.gender),
            "up_axis": np.array(self.up_axis),
            "joint_names": np.array(JOINT_NAMES),
            "joint_provenance": np.array(self.joint_provenance),
            "frame_valid": np.asarray(self.frame_valid, dtype=bool),
            "smpl18": np.array(json.dumps(self.about, ensure_ascii=False, sort_keys=True)),
        }


def sequence_from_trial(trial: CorpusTrial) -> SmplSequence:
    """Rebuild one trial's 24-joint pose and gather what a SMPL file should say about it."""
    subject: CorpusSubject = trial.subject
    record = subject.record.get("reduced_model", {})
    about = {
        "written_by": f"smpl18 {__version__}",
        "subject": subject.id,
        "trial": trial.id,
        "corpus_joints": list(trial.joint_names),
        "frozen_joints": list(record.get("frozen_joints", ())),
        "frozen_joints_are": "the subject's fitted constants, not per-frame rotations",
        "hands": "identity: the corpus does not store them",
        "reduction_residual": record.get("fit"),
        "model_is_stand_in": subject.record.get("model_is_stand_in"),
    }
    return SmplSequence(
        poses=trial.poses_24(),
        betas=subject.betas,
        trans=trial.trans,
        fps=trial.fps,
        gender=subject.gender,
        up_axis=trial.up_axis,
        joint_provenance=joint_provenance_24(trial.joint_names, trial.joint_provenance),
        frame_valid=trial.frame_valid,
        about=about,
    )


def write_sequence(path: str | Path, sequence: SmplSequence, *, flat: bool = True) -> Path:
    """Write ``sequence`` as an npz at ``path``, creating its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **sequence.arrays(flat=flat))
    return path
