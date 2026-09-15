"""The four source kinds, as plain data.

A source kind is what a capture observes, not where it came from: SMPL-family parameters, an
articulated skeleton with joint angles, joint-centre trajectories, or labelled surface markers.
Each kind is one frozen dataclass of numpy arrays in SI units (metres, radians) with an explicit
up axis and a per-frame validity mask. Nothing here knows a dataset, a file format or a body
model; ``profile.bind`` fills these from a profile plus the tables a format reader returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

__all__ = [
    "Format",
    "JointCentres",
    "MarkerTrajectories",
    "Provenance",
    "SegmentPlacement",
    "SkeletonModel",
    "SkeletonMotion",
    "SmplParameters",
    "SourceKind",
    "Subject",
    "Trial",
    "UpAxis",
]


class SourceKind(StrEnum):
    SMPL_PARAMETERS = "smpl_parameters"
    SKELETON_MOTION = "skeleton_motion"
    JOINT_CENTRES = "joint_centres"
    MARKER_TRAJECTORIES = "marker_trajectories"


class Format(StrEnum):
    NPZ = "npz"
    PICKLE = "pickle"
    JSON = "json"
    OSIM_MOT = "osim_mot"
    B3D = "b3d"
    TRC = "trc"
    C3D = "c3d"
    BVH = "bvh"
    MAT = "mat"


class Provenance(StrEnum):
    """Where a joint's rotation came from; one code per SMPL joint in the corpus."""

    MEASURED = "measured"
    DERIVED = "derived"
    ABSENT = "absent"


class UpAxis(StrEnum):
    Y = "y"
    Z = "z"


@dataclass(frozen=True)
class Subject:
    """What a source says about the person. Every field but ``id`` may be unknown."""

    id: str
    gender: str | None = None
    stature_m: float | None = None
    mass_kg: float | None = None
    #: How the gender was decided: ``field`` (read from the source), ``constant`` (declared by
    #: the profile), ``default`` (the profile's fallback for an unmapped value).
    gender_provenance: str | None = None


@dataclass(frozen=True)
class Trial:
    id: str
    subject: str
    #: Placeholder values the layout matched (study, split, ...), for grouping and provenance.
    groups: Mapping[str, str] = field(default_factory=dict)
    #: Source files in the file's own roles (``poses``, ``osim``, ``mot`` ...), as strings.
    sources: Mapping[str, str] = field(default_factory=dict)


def _check_frames(name: str, array: np.ndarray, frames: int) -> None:
    if array.shape[0] != frames:
        raise ValueError(f"{name} has {array.shape[0]} frames, expected {frames}")


def _valid_mask(frame_valid: np.ndarray | None, frames: int) -> np.ndarray:
    if frame_valid is None:
        return np.ones(frames, dtype=bool)
    mask = np.asarray(frame_valid, dtype=bool)
    _check_frames("frame_valid", mask, frames)
    return mask


@dataclass(frozen=True)
class SmplParameters:
    """SMPL-family parameters for one trial: poses ``[T, J, 3]`` axis-angle, one ``betas``
    vector, ``trans`` ``[T, 3]``. ``J`` is whatever the source stored (24, 52, 55); trimming to
    24 and ``betas`` to the model width is the converter's job."""

    subject: Subject
    trial: Trial
    poses: np.ndarray
    betas: np.ndarray
    trans: np.ndarray
    fps: float
    up_axis: UpAxis
    frame_valid: np.ndarray = None
    #: ``field`` when the file carried the rate, ``fallback`` when the profile's table did.
    fps_provenance: str = "field"

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[2] != 3:
            raise ValueError(f"poses must be [T, J, 3], got {poses.shape}")
        trans = np.asarray(self.trans, dtype=np.float64)
        if trans.ndim != 2 or trans.shape[1] != 3:
            raise ValueError(f"trans must be [T, 3], got {trans.shape}")
        _check_frames("trans", trans, poses.shape[0])
        betas = np.asarray(self.betas, dtype=np.float64).ravel()
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "trans", trans)
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "up_axis", UpAxis(self.up_axis))
        object.__setattr__(self, "frame_valid", _valid_mask(self.frame_valid, poses.shape[0]))

    @property
    def frames(self) -> int:
        return int(self.poses.shape[0])

    @property
    def joints(self) -> int:
        return int(self.poses.shape[1])


@dataclass(frozen=True)
class SegmentPlacement:
    """Where a skeleton model put its bodies for ``T`` frames: world rotations ``[T, 3, 3]``
    and origins ``[T, 3]`` per body, and joint centres ``[T, 3]`` per joint."""

    rotations: Mapping[str, np.ndarray]
    positions: Mapping[str, np.ndarray]
    joint_centres: Mapping[str, np.ndarray]


class SkeletonModel(ABC):
    """An articulated skeleton: named bodies in a tree, joints between them, rest transforms,
    and forward kinematics from coordinate values. Concrete models (OpenSim-style custom joints
    with coupled coordinates, BVH offset hierarchies) implement this elsewhere."""

    @property
    @abstractmethod
    def bodies(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def joints(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def parents(self) -> Mapping[str, str | None]:
        """Parent body per body; ``None`` for the root."""

    @property
    @abstractmethod
    def coordinate_names(self) -> tuple[str, ...]:
        """Independent coordinates, in the order ``SkeletonMotion.coordinates`` uses."""

    @abstractmethod
    def rest_transform(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """``(rotation [3, 3], translation [3])`` of a body in its parent at zero coordinates."""

    @abstractmethod
    def motion(self, coordinates: Mapping[str, np.ndarray]) -> SegmentPlacement:
        """Forward kinematics for ``T`` frames of coordinate values, keyed by coordinate name."""


@dataclass(frozen=True)
class SkeletonMotion:
    """A skeleton model and ``T`` frames of its coordinates, in radians and metres."""

    subject: Subject
    trial: Trial
    model: SkeletonModel
    coordinate_names: tuple[str, ...]
    coordinates: np.ndarray
    fps: float
    up_axis: UpAxis
    frame_valid: np.ndarray = None
    #: Centres the source stored itself, if any; otherwise the model's FK supplies them.
    stored_joint_centres: Mapping[str, np.ndarray] | None = None
    #: Bodies something in the source actually observed; ``None`` means all of them.
    driven_bodies: tuple[str, ...] | None = None
    #: Coordinates a repair rewrote, in the order they sit in ``coordinate_names``.
    repaired_coordinates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.coordinates, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.coordinate_names):
            raise ValueError(
                f"coordinates must be [T, {len(self.coordinate_names)}], got {values.shape}"
            )
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        object.__setattr__(self, "coordinates", values)
        object.__setattr__(self, "coordinate_names", tuple(self.coordinate_names))
        object.__setattr__(self, "up_axis", UpAxis(self.up_axis))
        object.__setattr__(self, "frame_valid", _valid_mask(self.frame_valid, values.shape[0]))

    @property
    def frames(self) -> int:
        return int(self.coordinates.shape[0])

    def placed(self) -> SegmentPlacement:
        """Run the model's forward kinematics over every frame."""
        return self.model.motion(dict(zip(self.coordinate_names, self.coordinates.T)))


@dataclass(frozen=True)
class JointCentres:
    """World positions of ``K`` named joint centres per frame, ``[T, K, 3]`` in metres, with a
    ``[T, K]`` validity mask. A source that also places its segments may carry their world
    rotations ``[T, 3, 3]`` per body, which lets pose be transferred rather than solved."""

    subject: Subject
    trial: Trial
    names: tuple[str, ...]
    positions: np.ndarray
    fps: float
    up_axis: UpAxis
    valid: np.ndarray = None
    segment_rotations: Mapping[str, np.ndarray] | None = None
    frame_valid: np.ndarray = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        names = tuple(self.names)
        if positions.ndim != 3 or positions.shape[1:] != (len(names), 3):
            raise ValueError(
                f"positions must be [T, {len(names)}, 3], got {positions.shape}"
            )
        valid = (
            np.isfinite(positions).all(axis=2)
            if self.valid is None
            else np.asarray(self.valid, dtype=bool)
        )
        if valid.shape != positions.shape[:2]:
            raise ValueError(f"valid must be [T, {len(names)}], got {valid.shape}")
        if self.segment_rotations is not None:
            for body, stack in self.segment_rotations.items():
                stack = np.asarray(stack, dtype=np.float64)
                if stack.shape != (positions.shape[0], 3, 3):
                    raise ValueError(
                        f"segment rotation {body!r} must be [T, 3, 3], got {stack.shape}"
                    )
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "up_axis", UpAxis(self.up_axis))
        object.__setattr__(
            self, "frame_valid", _valid_mask(self.frame_valid, positions.shape[0])
        )

    @property
    def frames(self) -> int:
        return int(self.positions.shape[0])

    def centre(self, name: str) -> np.ndarray:
        return self.positions[:, self.names.index(name)]


@dataclass(frozen=True)
class MarkerTrajectories:
    """``M`` labelled surface markers per frame, ``[T, M, 3]`` in metres, with a ``[T, M]``
    validity mask (occluded samples are invalid, never zero)."""

    subject: Subject
    trial: Trial
    labels: tuple[str, ...]
    positions: np.ndarray
    fps: float
    up_axis: UpAxis
    valid: np.ndarray = None
    frame_valid: np.ndarray = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        labels = tuple(self.labels)
        if positions.ndim != 3 or positions.shape[1:] != (len(labels), 3):
            raise ValueError(
                f"positions must be [T, {len(labels)}, 3], got {positions.shape}"
            )
        valid = (
            np.isfinite(positions).all(axis=2)
            if self.valid is None
            else np.asarray(self.valid, dtype=bool)
        )
        if valid.shape != positions.shape[:2]:
            raise ValueError(f"valid must be [T, {len(labels)}], got {valid.shape}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "up_axis", UpAxis(self.up_axis))
        object.__setattr__(
            self, "frame_valid", _valid_mask(self.frame_valid, positions.shape[0])
        )

    @property
    def frames(self) -> int:
        return int(self.positions.shape[0])
